"""expert_prewarm.py — Hot-set expert pre-warm for DSv4-Flash MXFP4 engine.

Davide Zenati 2026-05-04.

Carica top-N expert per layer in GPU memory permanent (no LRU eviction)
per eliminare il bottleneck NVMe→GPU upload (~1.2s/expert miss) durante
decode. Selection strategy guidata dall'analisi C3_routing_analysis.json:

- Hash layers (0..2, num_hash_layers=3): top-N selezionato sui count empirici
  di tid2eid (mappa deterministica token→expert).
- Hyperconvex layers (3..42): top-N selezionato sul gate.bias come proxy
  noaux_tc balancer (expert con bias alto = preferiti dal balancer).

Memory budget (per_expert_mb = 12.75 MB MXFP4 packed + E8M0 scales):
- top_n=32 per-layer × 43 layer ≈ 17.1 GB GPU resident
- top_n=64 per-layer × 43 layer ≈ 34.3 GB GPU resident
- top_n=128 per-layer × 43 layer ≈ 68.5 GB GPU resident (~zero miss prob)

Hit probability (top-K=6 routed per token, uniform-routing approximation):
P(at_least_one_hit) = 1 - (1 - N/256)^6
- N=32 → 55.1%
- N=64 → 82.2%
- N=128 → 98.4%

API drop-in:
    from expert_prewarm import ExpertHotSet
    hs = ExpertHotSet(loader, n_layers=43, top_n=64, strategy='per_layer')
    hs.prewarm()
    packed, scale = hs.get_expert(layer_idx=12, eid=37)  # GPU resident or fallback mmap

Tutti i tensor in self.hot_experts vivono su CUDA permanent (mai liberati).
Cache miss → fallback lazy mmap via loader.load_mxfp4 (CPU mapped, ~1ms/probe).
"""
from __future__ import annotations


import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# Default analysis output (Phase 1 produces this)
DEFAULT_ANALYSIS = os.environ.get("DSV4_OUT", "/tmp") + "/C3_routing_analysis.json"

# DSv4-Flash topology constants (from config.json)
N_EXPERTS = 256
TOP_K_ROUTED = 6
NUM_HASH_LAYERS = 3
PER_EXPERT_MB_DEFAULT = 12.75


class ExpertHotSet:
    """Pre-load top-N expert per layer in GPU permanent memory.

    Parameters
    ----------
    loader : MXFP4Loader
        Already-opened safetensors loader (B1 module).
    n_layers : int
        Total MoE layers in the model (43 for DSv4-Flash).
    top_n : int
        Number of expert pre-warmed per layer (default 32).
    strategy : str
        - 'per_layer': top-N specifico per layer (raccomandato per DSv4)
        - 'global': top-N condivisi cross-layer (cheap, low coverage)
        - 'hybrid': N/2 globali + N/2 per-layer
    device : str | torch.device
        CUDA device dove parcheggiare gli expert (default "cuda").
    analysis_path : str | None
        Path al JSON prodotto da C3_analyze.py. Se None, ricalcola dal loader.
    """

    def __init__(
        self,
        loader,
        n_layers: int = 43,
        top_n: int = 32,
        strategy: str = "per_layer",
        device: str = "cuda",
        analysis_path: Optional[str] = DEFAULT_ANALYSIS,
        n_experts: int = N_EXPERTS,
    ):
        if strategy not in ("per_layer", "global", "hybrid"):
            raise ValueError(f"unknown strategy: {strategy!r}")
        self.loader = loader
        self.n_layers = n_layers
        self.top_n = top_n
        self.strategy = strategy
        self.device = torch.device(device)
        self.n_experts = n_experts

        # Selection: layer -> set of expert ids to pre-warm
        self.selection: Dict[int, set] = {}
        # Storage: (layer, eid) -> (packed_uint8 GPU, scale_uint8 GPU) per (w1, w2, w3)
        # We store one dict per weight to keep API symmetric with loader.load_mxfp4.
        # Keyed by (layer, eid, wkey) where wkey in {'w1','w2','w3'}.
        self.hot_experts: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

        # Stats
        self._stats = {
            "n_warm_calls": 0,
            "hits": 0,
            "misses": 0,
            "prewarm_seconds": 0.0,
            "warm_count": 0,
        }

        # Load analysis or compute on-the-fly
        self._analysis = self._load_or_build_analysis(analysis_path)
        self._build_selection()

    # ---------------------------------------------------------------- analysis
    def _load_or_build_analysis(self, analysis_path: Optional[str]) -> dict:
        if analysis_path is not None and os.path.isfile(analysis_path):
            with open(analysis_path) as f:
                a = json.load(f)
            return a
        # Fallback: build minimal analysis from loader (slow path)
        return self._build_analysis_from_loader()

    def _build_analysis_from_loader(self) -> dict:
        per_layer_top = {}
        for L in range(self.n_layers):
            if L < NUM_HASH_LAYERS:
                tname = f"layers.{L}.ffn.gate.tid2eid"
                if tname in self.loader:
                    arr = self.loader.load_raw(tname).cpu().numpy().astype(np.int64).flatten()
                    bc = np.bincount(arr, minlength=self.n_experts)[: self.n_experts]
                    order = np.argsort(bc)[::-1]
                    per_layer_top[L] = [int(e) for e in order[: self.top_n]]
                    continue
            bname = f"layers.{L}.ffn.gate.bias"
            if bname in self.loader:
                bias = self.loader.load_raw(bname).cpu().to(torch.float32).numpy().flatten()
                if bias.shape[0] >= self.n_experts:
                    order = np.argsort(bias[: self.n_experts])[::-1]
                    per_layer_top[L] = [int(e) for e in order[: self.top_n]]
                    continue
            # Last resort: identity 0..top_n-1
            per_layer_top[L] = list(range(self.top_n))
        return {"per_layer_top_n": per_layer_top, "global_top_n": list(range(self.top_n))}

    # --------------------------------------------------------------- selection
    def _build_selection(self):
        per_layer = self._analysis.get("per_layer_top_n", {})
        global_top = self._analysis.get("global_top_n", list(range(self.top_n)))

        # JSON keys are strings — normalize
        per_layer = {int(k): v for k, v in per_layer.items()}

        for L in range(self.n_layers):
            if self.strategy == "global":
                sel = set(global_top[: self.top_n])
            elif self.strategy == "per_layer":
                sel = set(per_layer.get(L, list(range(self.top_n)))[: self.top_n])
            elif self.strategy == "hybrid":
                half = self.top_n // 2
                sel = set(global_top[:half])
                # extend with per-layer specific (skip overlap)
                layer_specific = per_layer.get(L, list(range(self.top_n)))
                added = 0
                for e in layer_specific:
                    if e not in sel:
                        sel.add(e)
                        added += 1
                        if added >= self.top_n - half:
                            break
            else:
                raise RuntimeError(f"strategy {self.strategy!r} unreachable")
            self.selection[L] = sel

    # ---------------------------------------------------------------- prewarm
    def prewarm(self, verbose: bool = True) -> dict:
        """Move all selected expert (packed + scale) to GPU permanent memory.

        Returns a dict with {warm_count, gpu_mem_gb, elapsed_sec}.
        """
        t0 = time.time()
        warm = 0
        for L in range(self.n_layers):
            for eid in self.selection[L]:
                for wkey in ("w1", "w2", "w3"):
                    base = f"layers.{L}.ffn.experts.{eid}.{wkey}"
                    try:
                        packed, scale = self.loader.load_mxfp4(base)
                    except Exception as e:
                        if verbose:
                            print(f"[hotset] L{L} e{eid} {wkey} miss: {e}")
                        continue
                    # Move to GPU permanent. .contiguous() to drop mmap stride.
                    p_gpu = packed.contiguous().to(self.device, non_blocking=True)
                    s_gpu = scale.contiguous().to(self.device, non_blocking=True)
                    self.hot_experts[(L, eid, wkey)] = (p_gpu, s_gpu)
                    warm += 1
            if verbose and (L < 3 or L == self.n_layers - 1 or L % 10 == 9):
                gpu_mem = self._gpu_mem_bytes() / (1024 ** 3)
                print(f"[hotset] L{L:2d} warmed; cumulative {warm} tensors, "
                      f"GPU resident {gpu_mem:.2f} GB")
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        elapsed = time.time() - t0
        gpu_mem = self._gpu_mem_bytes() / (1024 ** 3)
        self._stats["prewarm_seconds"] = elapsed
        self._stats["warm_count"] = warm
        if verbose:
            print(f"[hotset] PREWARM done: {warm} tensors, {gpu_mem:.2f} GB GPU, {elapsed:.1f}s")
        return {"warm_count": warm, "gpu_mem_gb": round(gpu_mem, 2),
                "elapsed_sec": round(elapsed, 2)}

    def _gpu_mem_bytes(self) -> int:
        total = 0
        for (p, s) in self.hot_experts.values():
            total += p.numel() * p.element_size() + s.numel() * s.element_size()
        return total

    # ------------------------------------------------------------ runtime API
    def get_expert(
        self, layer_idx: int, eid: int, wkey: str = "w1"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (packed_uint8, scale_uint8) for an expert weight.

        Hot-path: GPU resident tensors (zero copy, just dict lookup).
        Cold-path: lazy mmap via loader.load_mxfp4 (CPU, ~1ms).

        wkey ∈ {'w1', 'w2', 'w3'} — DSv4-Flash MoE expert decomposition.
        """
        self._stats["n_warm_calls"] += 1
        key = (layer_idx, eid, wkey)
        cached = self.hot_experts.get(key)
        if cached is not None:
            self._stats["hits"] += 1
            return cached
        self._stats["misses"] += 1
        return self._lazy_load_mmap(layer_idx, eid, wkey)

    def has_expert(self, layer_idx: int, eid: int, wkey: str = "w1") -> bool:
        """Return True if (layer, eid, wkey) is in the GPU hot-set."""
        return (layer_idx, eid, wkey) in self.hot_experts

    def _lazy_load_mmap(
        self, layer_idx: int, eid: int, wkey: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base = f"layers.{layer_idx}.ffn.experts.{eid}.{wkey}"
        return self.loader.load_mxfp4(base)

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        s = dict(self._stats)
        n = max(s["n_warm_calls"], 1)
        s["hit_rate_pct"] = round(s["hits"] / n * 100, 2)
        s["gpu_mem_gb"] = round(self._gpu_mem_bytes() / (1024 ** 3), 3)
        s["n_layers"] = self.n_layers
        s["top_n"] = self.top_n
        s["strategy"] = self.strategy
        s["selection_size_avg"] = round(
            float(np.mean([len(v) for v in self.selection.values()])), 1
        )
        return s

    # ------------------------------------------------------------- evict / gc
    def evict_layer(self, layer_idx: int):
        """Drop all GPU tensors for one layer (free VRAM)."""
        keys = [k for k in self.hot_experts if k[0] == layer_idx]
        for k in keys:
            del self.hot_experts[k]
        torch.cuda.empty_cache() if self.device.type == "cuda" else None

    def evict_all(self):
        """Drop everything. Memory returns to allocator."""
        self.hot_experts.clear()
        torch.cuda.empty_cache() if self.device.type == "cuda" else None


__all__ = ["ExpertHotSet", "DEFAULT_ANALYSIS"]
