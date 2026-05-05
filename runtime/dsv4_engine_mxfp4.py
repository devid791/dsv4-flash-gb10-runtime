from __future__ import annotations
"""
dsv4_engine_mxfp4.py - DSv4-Flash native MXFP4 (routed) + FP8 E4M3 (attention/shared)
end-to-end engine (, sprint MXFP4 nativo).

Davide Zenati, 2026-05-04.

Refactor of `dsv4_engine_q3.py` (675 LOC, Q3_K_M backend) to consume the
ORIGINAL HF safetensors snapshot at `${DSV4_WEIGHTS}/` directly, with:

  - Routed experts:       MXFP4 packed nibbles + E8M0 32-elem block scales,
                          fed through `mxfp4_grouped_gemv` (kernel B3, top-K=6).
  - Attention + shared:   FP8 E4M3 + E8M0 128x128 2-D block scales,
                          fed through `fp8_e4m3_gemv` (kernel B2-bis).
  - Norms / embed / head /
    router gate / HC:     plain BF16 / F32 / I64 (loaded once, kept on GPU).

Architecture (preserved from Q3 engine, end-to-end identical wiring):

  hidden = embed(token_ids).repeat(n_hc=4)               [S, n_hc=4, hidden=4096]
  for L in 0..42:
    # --- Attention block w/ HyperConnection ---
    x_pre, _, post_a, comb_a = hc_pre(hidden, hc_attn_*[L])
    h_n = rmsnorm(x_pre, attn_norm[L])
    attn = MLA_attention(h_n, layer L)                   # FP8 wq_a/b, wkv, wo_a/b
    hidden = hc_post(attn, hidden, post_a, comb_a)

    # --- FFN/MoE block w/ HyperConnection ---
    x_pre, _, post_f, comb_f = hc_pre(hidden, hc_ffn_*[L])
    h_n2 = rmsnorm(x_pre, ffn_norm[L])
    gate_logits = h_n2 @ router_gate[L].T                # BF16 [S, 256]
    scores = sqrt(softplus(gate_logits))                 # sqrtsoftplus
    topk_w, topk_ids = topk(scores, K=6); topk_w = norm * 1.5
    routed_out = mxfp4_grouped_gemv(top-6 expert MXFP4)  # NATIVE kernel
    shared_out = SwiGLU_FP8(h_n2, sh_w1, sh_w2, sh_w3)   # NATIVE FP8 GEMV
    moe_out = routed_out + shared_out
    hidden = hc_post(moe_out, hidden, post_f, comb_f)

  final = hc_head(hidden, hc_head_*)                     # collapse n_hc=4 -> 1
  logits = head.weight @ rmsnorm(final[-1], final_norm)  # BF16 matmul

Memory plan (target <40 GB GPU on 121 GB unified):
  - BF16 plain (norms, embed=1.06GB, head=1.06GB, gate, router tid2eid):
                                                         ~ 2.5 GB resident
  - HC F32 weights (43 layer x 6 tensors + final 3):     ~ 35 MB resident
  - FP8 attention+shared per layer: ~128 MB packed x 43 = ~5.5 GB resident
  - MXFP4 routed (256 expert x 3 weights x ~4MB):        ~ 130 GB total -> CPU mmap
                                                         lazy GPU upload top-6/layer
  - GPU LRU cache for routed: bound at 32 expert/layer   ~ 16 GB cap
  - KV cache BF16:                                        small

Engine init target: <30 s (no full dequant, just open mmap + load BF16/F32/FP8).

Current status (matches README):
  - Forced-token R2-vs-C++ equivalence: 8/8 argmax match, logits cosine
    0.9994 - 0.9999.
  - Per-layer R2-vs-C++ cosine bisect: no first_bad_layer in 0..42.
  - MLA (full multi-head latent attention) and YARN compressed RoPE are
    wired in the C++ engine and validated against the reference.
  - Cold-load lifecycle: madvise(MADV_DONTNEED) + per-slot cudaEvent
    async path + batched (O(1) event/wait/madvise per pack call).

Out of scope for this release:
  - CUDA Graph capture
  - MTP / speculative decode
  - Request batching
  - OpenAI-compatible server
  - Lightning Indexer (sliding-window asymmetric attention)
"""


import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from mxfp4_loader import MXFP4Loader, MXFP4_BLOCK_SIZE
from rope_yarn import precompute_freqs_cis, apply_rotary_emb_2d
from fp8_e4m3_gemv_binding import fp8_e4m3_gemv, FP8_BLOCK_TILE
from mxfp4_grouped_gemv_binding import mxfp4_grouped_gemv
from hc_ffn import hc_pre, hc_post, hc_head


# --- Paths / config -----------------------------------------------------------

HF_SNAPSHOT = os.environ.get("DSV4_HF_PATH", os.environ.get("DSV4_WEIGHTS", ""))

# Architecture constants - verified empirically against HF config.json.
N_LAYERS         = 43
HIDDEN           = 4096
VOCAB            = 129280
NUM_HEADS        = 64
KV_LORA_RANK     = 512
Q_LORA_RANK      = 1024
HEAD_DIM         = 512        # full key dim per head (= 32768 / 64)
QK_ROPE_DIM      = 64         # last 64 dims of head get RoPE
N_GROUPS         = 8          # o_groups per HF config
O_LORA_RANK      = 1024       # o_lora_rank per HF config
GROUP_DIN        = NUM_HEADS * HEAD_DIM // N_GROUPS  # 64*512/8 = 4096
WINDOW_SIZE      = 128        # sliding window per HF config
MAX_SEQ_LEN_INIT = 4096       # precompute freqs_cis for this many positions
ROPE_THETA       = 10000.0
COMPRESS_ROPE_THETA = 160000.0  # compress_rope_theta per HF config (for compressed layers)
YARN_FACTOR      = 16.0
YARN_ORIGINAL    = 65536
YARN_BETA_FAST   = 32.0
YARN_BETA_SLOW   = 1.0
RMS_EPS          = 1e-6
N_EXPERTS        = 256
EXPERTS_PER_TOK  = 6
EXPERT_FF        = 2048       # moe_intermediate_size
ROUTED_SCALING   = 1.5
SWIGLU_LIMIT     = 10.0       # swiglu_limit per HF config (clamp on shared expert)
# Layer compress_ratios from HF config (43 entries, 0=pure SW no YARN, 4=ratio 4 YARN, 128=ratio 128 YARN)
COMPRESS_RATIOS  = [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                    4, 128, 0]

# HC hparams (config: hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6).
HC_N_HC          = 4
HC_SINKHORN_ITERS = 20
HC_EPS           = 1e-6


def _log(msg: str) -> None:
    sys.stdout.write(f"[mxfp4-engine] {msg}\n")
    sys.stdout.flush()


# --- RMSNorm helper ----------------------------------------------------------

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    """RMSNorm in fp32 then cast back. x [..., H], weight [H]."""
    in_dtype = x.dtype
    f = x.float()
    rms = torch.rsqrt((f * f).mean(-1, keepdim=True) + eps)
    return (f * rms * weight.float()).to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x)


# --- Per-layer weight container ---------------------------------------------

class LayerMXFP4:
    """Per-layer GPU-resident weights (FP8 + BF16 + F32). Routed experts are
    NOT held here - they live in CPU mmap and are uploaded lazily."""

    __slots__ = (
        "L",
        # norms (BF16 [H] / [q_lora] / [kv_lora])
        "attn_norm", "ffn_norm", "q_norm", "kv_norm", "attn_sink",
        # FP8 attention (uint8 raw + scale uint8 E8M0 [N/128, K/128])
        "wkv_w", "wkv_s",
        "wq_a_w", "wq_a_s",
        "wq_b_w", "wq_b_s",
        "wo_a_w", "wo_a_s",
        "wo_b_w", "wo_b_s",
        # router (BF16 [256, H], plus tid2eid I64 [vocab, K])
        "router_gate",
        # FP8 shared expert (single shared expert per spec)
        "sh_w1_w", "sh_w1_s",
        "sh_w2_w", "sh_w2_s",
        "sh_w3_w", "sh_w3_s",
        # HC F32 (loaded into GPU)
        "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
        "hc_ffn_fn",  "hc_ffn_base",  "hc_ffn_scale",
    )


# --- Engine ------------------------------------------------------------------

class DSv4EngineMXFP4:
    """End-to-end DSv4-Flash forward over HF safetensors with native MXFP4
    routed + FP8 E4M3 attention/shared kernels. Smoke-min for B4 sprint."""

    def __init__(self, hf_dir: str = HF_SNAPSHOT, *, max_layers: Optional[int] = None):
        t0 = time.time()
        _log(f"opening HF snapshot {hf_dir}")
        self.loader = MXFP4Loader(hf_dir)
        _log(f"loader: {len(self.loader)} tensors indexed (mmap, zero-copy)")

        self.n_layers = N_LAYERS if max_layers is None else min(N_LAYERS, max_layers)
        if max_layers is not None:
            _log(f"WARN truncated to {self.n_layers} layers for smoke")

        # -- BF16 plain top-level: embed, head, final norm
        _log("loading BF16 top-level (embed, head, norm) ...")
        t = time.time()
        self.embed = self.loader.load_bf16("embed.weight").contiguous().cuda()
        self.head = self.loader.load_bf16("head.weight").contiguous().cuda()
        self.final_norm = self.loader.load_bf16("norm.weight").contiguous().cuda()
        _log(f"  embed {tuple(self.embed.shape)} head {tuple(self.head.shape)} "
             f"norm {tuple(self.final_norm.shape)}  ({time.time()-t:.1f}s)")

        # -- HC head F32
        self.hc_head_fn    = self.loader.load_raw("hc_head_fn").float().contiguous().cuda()
        self.hc_head_base  = self.loader.load_raw("hc_head_base").float().contiguous().cuda()
        self.hc_head_scale = self.loader.load_raw("hc_head_scale").float().contiguous().cuda()
        _log(f"  hc_head: fn {tuple(self.hc_head_fn.shape)} "
             f"base {tuple(self.hc_head_base.shape)} scale {tuple(self.hc_head_scale.shape)}")

        # -- Per-layer load
        _log("loading per-layer FP8 attention + shared + BF16 router + F32 HC ...")
        self.layers: list[LayerMXFP4] = []
        t_lyr = time.time()
        for L in range(self.n_layers):
            tL = time.time()
            ly = LayerMXFP4()
            ly.L = L
            ly.attn_norm = self.loader.load_bf16(f"layers.{L}.attn_norm.weight").contiguous().cuda()
            ly.ffn_norm  = self.loader.load_bf16(f"layers.{L}.ffn_norm.weight").contiguous().cuda()
            ly.q_norm    = self.loader.load_bf16(f"layers.{L}.attn.q_norm.weight").contiguous().cuda()
            ly.kv_norm   = self.loader.load_bf16(f"layers.{L}.attn.kv_norm.weight").contiguous().cuda()
            ly.attn_sink = self.loader.load_raw(f"layers.{L}.attn.attn_sink").float().contiguous().cuda()

            ly.wkv_w,  ly.wkv_s  = self._load_fp8_to_gpu(f"layers.{L}.attn.wkv")
            ly.wq_a_w, ly.wq_a_s = self._load_fp8_to_gpu(f"layers.{L}.attn.wq_a")
            ly.wq_b_w, ly.wq_b_s = self._load_fp8_to_gpu(f"layers.{L}.attn.wq_b")
            ly.wo_a_w, ly.wo_a_s = self._load_fp8_to_gpu(f"layers.{L}.attn.wo_a")
            ly.wo_b_w, ly.wo_b_s = self._load_fp8_to_gpu(f"layers.{L}.attn.wo_b")

            ly.router_gate = self.loader.load_bf16(f"layers.{L}.ffn.gate.weight").contiguous().cuda()

            ly.sh_w1_w, ly.sh_w1_s = self._load_fp8_to_gpu(f"layers.{L}.ffn.shared_experts.w1")
            ly.sh_w2_w, ly.sh_w2_s = self._load_fp8_to_gpu(f"layers.{L}.ffn.shared_experts.w2")
            ly.sh_w3_w, ly.sh_w3_s = self._load_fp8_to_gpu(f"layers.{L}.ffn.shared_experts.w3")

            ly.hc_attn_fn    = self.loader.load_raw(f"layers.{L}.hc_attn_fn").float().contiguous().cuda()
            ly.hc_attn_base  = self.loader.load_raw(f"layers.{L}.hc_attn_base").float().contiguous().cuda()
            ly.hc_attn_scale = self.loader.load_raw(f"layers.{L}.hc_attn_scale").float().contiguous().cuda()
            ly.hc_ffn_fn     = self.loader.load_raw(f"layers.{L}.hc_ffn_fn").float().contiguous().cuda()
            ly.hc_ffn_base   = self.loader.load_raw(f"layers.{L}.hc_ffn_base").float().contiguous().cuda()
            ly.hc_ffn_scale  = self.loader.load_raw(f"layers.{L}.hc_ffn_scale").float().contiguous().cuda()

            self.layers.append(ly)
            if L == 0 or (L + 1) % 8 == 0 or L == self.n_layers - 1:
                _log(f"  layer {L:2d}/{self.n_layers}: {time.time()-tL:.2f}s "
                     f"(cum {time.time()-t_lyr:.1f}s)")
        _log(f"per-layer load done: {time.time()-t_lyr:.1f}s "
             f"(~{(time.time()-t_lyr)/self.n_layers:.2f}s/layer)")

        # -- Routed expert MXFP4: register mmap views, no GPU upload yet
        _log("registering routed-expert mmap views (no GPU upload yet) ...")
        t_exp = time.time()
        self._routed_meta: list[dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]] = []
        for L in range(self.n_layers):
            d_layer: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
            for E in range(N_EXPERTS):
                d_e: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                for wname in ("w1", "w2", "w3"):
                    w_packed, w_scale = self.loader.load_mxfp4(
                        f"layers.{L}.ffn.experts.{E}.{wname}"
                    )
                    d_e[wname] = (w_packed, w_scale)
                d_layer[E] = d_e
            self._routed_meta.append(d_layer)
        _log(f"routed-expert mmap registry built: {self.n_layers}*{N_EXPERTS}*3 = "
             f"{self.n_layers*N_EXPERTS*3} tensors  ({time.time()-t_exp:.1f}s)")

        # GPU LRU cache for routed expert weights
        self._routed_gpu_cache: list[dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]] = [
            {} for _ in range(self.n_layers)
        ]
        self._routed_lru: list[list[int]] = [[] for _ in range(self.n_layers)]
        self.ROUTED_CACHE_PER_LAYER = 32

        # KV cache: list of latent tensors [seq, kv_lora_rank=512] BF16 per layer.
        self._kv: list[Optional[torch.Tensor]] = [None] * self.n_layers
        self._pos = 0

        # Precompute RoPE/YARN freqs_cis - PER-LAYER selection.
        # HF spec: compress_ratio=0 layers (pure SW) -> NO YARN, base=rope_theta=10000
        #          compress_ratio>0 layers (compressed) -> YARN, base=compress_rope_theta=160000
        _log(f"precomputing RoPE freqs_cis: base (SW) + compressed (YARN) variants")
        self.freqs_cis_base = precompute_freqs_cis(
            dim=QK_ROPE_DIM, seqlen=MAX_SEQ_LEN_INIT,
            original_seq_len=0, base=ROPE_THETA,                 # NO YARN
            factor=1.0, beta_fast=YARN_BETA_FAST, beta_slow=YARN_BETA_SLOW,
            device="cuda",
        )
        self.freqs_cis_compress = precompute_freqs_cis(
            dim=QK_ROPE_DIM, seqlen=MAX_SEQ_LEN_INIT,
            original_seq_len=YARN_ORIGINAL, base=COMPRESS_ROPE_THETA,  # YARN
            factor=YARN_FACTOR, beta_fast=YARN_BETA_FAST, beta_slow=YARN_BETA_SLOW,
            device="cuda",
        )
        # Default for backward-compat
        self.freqs_cis = self.freqs_cis_base

        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / 1e9
        _log(f"GPU mem allocated (init, no routed cached): {mem:.2f} GB")
        _log(f"engine ready in {time.time()-t0:.1f}s")
        self._init_time_s = time.time() - t0
        self._init_mem_gb = mem

    # -- FP8 helpers ----------------------------------------------------------

    def _load_fp8_to_gpu(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load (weight, scale) FP8 pair -> GPU uint8 tensors."""
        w_u8, s_u8 = self.loader.load_fp8(name)
        return w_u8.contiguous().cuda(), s_u8.contiguous().cuda()

    def _fp8_linear(self, x: torch.Tensor, w_u8: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
        """y[n] = sum_k W[n,k] * x[k]. x BF16 [K] (or [S, K] -> per-row loop)."""
        if x.dim() == 1:
            return fp8_e4m3_gemv(w_u8, s_u8, x.contiguous())
        S, K = x.shape
        N = w_u8.shape[0]
        out = torch.empty(S, N, dtype=torch.bfloat16, device=x.device)
        for s in range(S):
            out[s] = fp8_e4m3_gemv(w_u8, s_u8, x[s].contiguous())
        return out

    # -- Routed expert lazy upload + LRU --------------------------------------

    def _ensure_expert_on_gpu(self, L: int, E: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return per-expert GPU dict. Uploads on first access, LRU-evicts.

        ROUND-2 wire: check engine._hotset BEFORE LRU. If all 3 wkeys present in
        hot-set, return zero-copy GPU resident dict. Else fallback to LRU upload.
        """
        # ROUND-2 hot-set fast path: zero-copy GPU resident
        hs = getattr(self, "_hotset", None)
        if hs is not None and hs.has_expert(L, E, "w1"):
            return {
                "w1": hs.get_expert(L, E, "w1"),
                "w2": hs.get_expert(L, E, "w2"),
                "w3": hs.get_expert(L, E, "w3"),
            }
        cache = self._routed_gpu_cache[L]
        lru = self._routed_lru[L]
        if E in cache:
            try:
                lru.remove(E)
            except ValueError:
                pass
            lru.append(E)
            return cache[E]

        while len(lru) >= self.ROUTED_CACHE_PER_LAYER:
            old_E = lru.pop(0)
            cache.pop(old_E, None)

        cpu_dict = self._routed_meta[L][E]
        gpu_dict: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for wname, (p_cpu, s_cpu) in cpu_dict.items():
            p_gpu = p_cpu.contiguous().cuda(non_blocking=False)
            s_gpu = s_cpu.contiguous().cuda(non_blocking=False)
            gpu_dict[wname] = (p_gpu, s_gpu)
        cache[E] = gpu_dict
        lru.append(E)
        return gpu_dict

    # -- MoE block ------------------------------------------------------------

    def _route(self, h_n: torch.Tensor, layer: LayerMXFP4) -> tuple[torch.Tensor, torch.Tensor]:
        """sqrtsoftplus router with norm + routed_scaling=1.5."""
        gate = torch.nn.functional.linear(h_n.float(), layer.router_gate.float())
        scores = torch.sqrt(torch.nn.functional.softplus(gate))
        topk_scores, topk_ids = torch.topk(scores, EXPERTS_PER_TOK, dim=-1)
        topk_w = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-20)
        topk_w = (topk_w * ROUTED_SCALING).to(torch.bfloat16)
        return topk_ids.to(torch.int32), topk_w

    def _routed_swiglu_grouped(
        self, h_n2: torch.Tensor, L: int, topk_ids_row: torch.Tensor, topk_w_row: torch.Tensor
    ) -> torch.Tensor:
        """Run grouped MXFP4 SwiGLU on top-K=6 experts for ONE token.
        h_n2: [HIDDEN] BF16. Returns [HIDDEN] BF16."""
        ids = [int(x) for x in topk_ids_row.tolist()]
        per_expert = [self._ensure_expert_on_gpu(L, e) for e in ids]

        W1_packed = torch.stack([d["w1"][0] for d in per_expert], dim=0).contiguous()
        W3_packed = torch.stack([d["w3"][0] for d in per_expert], dim=0).contiguous()
        W2_packed = torch.stack([d["w2"][0] for d in per_expert], dim=0).contiguous()
        W1_scale  = torch.stack([d["w1"][1] for d in per_expert], dim=0).contiguous()
        W3_scale  = torch.stack([d["w3"][1] for d in per_expert], dim=0).contiguous()
        W2_scale  = torch.stack([d["w2"][1] for d in per_expert], dim=0).contiguous()

        r = topk_w_row.float().contiguous()

        out = mxfp4_grouped_gemv(
            W1={"packed": W1_packed, "scale": W1_scale},
            W3={"packed": W3_packed, "scale": W3_scale},
            W2={"packed": W2_packed, "scale": W2_scale},
            scales=None,
            x=h_n2.contiguous(),
            r=r,
        )
        return out

    def _shared_swiglu_fp8(self, h_n2: torch.Tensor, layer: LayerMXFP4) -> torch.Tensor:
        """Native FP8 SwiGLU with swiglu_limit clamp (HF config swiglu_limit=10.0).
        out = w2(silu(clamp(w1 x, ±limit)) * clamp(w3 x, ±limit)).
        h_n2: [HIDDEN] BF16. Returns [HIDDEN] BF16."""
        gate = self._fp8_linear(h_n2, layer.sh_w1_w, layer.sh_w1_s)
        up   = self._fp8_linear(h_n2, layer.sh_w3_w, layer.sh_w3_s)
        if SWIGLU_LIMIT > 0:
            gate = torch.clamp(gate, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            up   = torch.clamp(up,   min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        mid  = silu(gate) * up
        out  = self._fp8_linear(mid,  layer.sh_w2_w, layer.sh_w2_s)
        return out

    def _moe_block(self, h_n2: torch.Tensor, L: int, layer: LayerMXFP4) -> torch.Tensor:
        """h_n2: [S, HIDDEN] BF16. Returns [S, HIDDEN] BF16."""
        S = h_n2.shape[0]
        topk_ids, topk_w = self._route(h_n2, layer)

        out = torch.empty_like(h_n2)
        for s in range(S):
            row = h_n2[s]
            routed = self._routed_swiglu_grouped(row, L, topk_ids[s], topk_w[s])
            shared = self._shared_swiglu_fp8(row, layer)
            out[s] = routed + shared
        return out

    # -- MLA attention --------------------------------------------------------

    def _mla_attention(self, h_n: torch.Tensor, layer: LayerMXFP4, layer_id: int) -> torch.Tensor:
        """Full MLA attention (HF inference/model.py spec).

        Architecture:
          - Q low-rank:    h -> wq_a (4096->1024) -> q_norm -> wq_b (1024->64*512=32768)
                          -> reshape [S, n_heads=64, head_dim=512] -> per-head RMSNorm -> RoPE on tail 64
          - KV shared:    h -> wkv (4096->512) -> kv_norm -> RoPE on tail 64
                          -> single MQA head broadcast to all 64 query heads
          - Attention:    sliding-window=128 causal + attn_sink fp32 per head
          - Output:       o[..., -rd:] inverse-RoPE -> view [S, n_groups=8, group_din=4096]
                          -> einsum wo_a [8, 1024, 4096] -> flatten [S, 8192] -> wo_b [4096, 8192] -> [S, 4096]
        """
        S = h_n.shape[0]
        rd = QK_ROPE_DIM
        device = h_n.device

        # --- Q path ---
        q_a_new  = self._fp8_linear(h_n, layer.wq_a_w, layer.wq_a_s)  # [S, 1024]
        q_a_new  = rmsnorm_torch(q_a_new, layer.q_norm, eps=RMS_EPS)
        q_full   = self._fp8_linear(q_a_new, layer.wq_b_w, layer.wq_b_s)  # [S, 32768]
        q = q_full.view(S, NUM_HEADS, HEAD_DIM)                       # [S, 64, 512]
        # Per-head RMSNorm (HF: q *= rsqrt(q.square().mean(-1)+eps))
        q_f = q.float()
        q_inv = torch.rsqrt(q_f.square().mean(-1, keepdim=True) + RMS_EPS)
        q = (q_f * q_inv).to(q.dtype)

        # --- KV path (shared latent, MQA) ---
        c_kv_new = self._fp8_linear(h_n, layer.wkv_w,  layer.wkv_s)   # [S, 512]
        kv = rmsnorm_torch(c_kv_new, layer.kv_norm, eps=RMS_EPS)      # [S, 512]

        # --- RoPE wire on tail rd=64 of Q (per head) and KV (shared) ---
        # Per-layer freqs_cis: pure SW layers use base, compressed use YARN
        compress_ratio = COMPRESS_RATIOS[layer_id] if layer_id < len(COMPRESS_RATIOS) else 0
        freqs_table = self.freqs_cis_compress if compress_ratio > 0 else self.freqs_cis_base
        pos_start = self._pos
        pos_end   = self._pos + S
        if pos_end > freqs_table.size(0):
            raise RuntimeError(f"position {pos_end} exceeds freqs_cis cache {freqs_table.size(0)}")
        f_slice = freqs_table[pos_start:pos_end]

        # Q rope tail in-place per HF semantics
        q_rope_view = q[..., -rd:].contiguous()
        apply_rotary_emb_2d(q_rope_view, f_slice)
        q = torch.cat([q[..., :-rd], q_rope_view], dim=-1)            # [S, 64, 512]

        # KV rope tail in-place (single shared head)
        kv_rope = kv[:, -rd:].unsqueeze(1).contiguous()
        apply_rotary_emb_2d(kv_rope, f_slice)
        kv = torch.cat([kv[:, :-rd], kv_rope.squeeze(1)], dim=-1)     # [S, 512]

        # --- KV cache append (post-norm + post-RoPE) ---
        if self._kv[layer_id] is None or self._kv[layer_id].shape[0] == 0:
            self._kv[layer_id] = kv
        else:
            self._kv[layer_id] = torch.cat([self._kv[layer_id], kv], dim=0)
        kv_cache = self._kv[layer_id]                                  # [P, 512]
        P = kv_cache.shape[0]

        # --- Attention scores ---
        # MQA: K=V=kv_cache (shared single head broadcast across all 64 heads)
        # einsum [S, 64, 512] @ [P, 512] -> [S, 64, P]
        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)
        scores = torch.einsum("shd,pd->shp", q.float(), kv_cache.float()) * softmax_scale

        # Causal mask + sliding window (window_size=128 from config)
        ar = torch.arange(P, device=device)
        q_pos = (self._pos + torch.arange(S, device=device)).unsqueeze(1)  # [S, 1]
        # mask out p > q_pos (future) OR p < q_pos - window + 1 (outside window)
        causal_mask = ar.unsqueeze(0) > q_pos
        win_mask = ar.unsqueeze(0) < (q_pos - WINDOW_SIZE + 1)
        mask = causal_mask | win_mask                                  # [S, P]
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        # Attention sink: per-head learned logit (concat as extra "key" position)
        sink = layer.attn_sink.float().view(1, -1, 1).expand(S, NUM_HEADS, 1)  # [S, 64, 1]
        scores_with_sink = torch.cat([scores, sink], dim=-1)            # [S, 64, P+1]
        attn_w = torch.softmax(scores_with_sink, dim=-1)
        attn_w_data = attn_w[..., :P]                                  # drop sink prob
        # V is shared single head -> [S, 64, 512]
        attn_out = torch.einsum("shp,pd->shd", attn_w_data, kv_cache.float())
        attn_out = attn_out.to(torch.bfloat16)                          # [S, 64, 512]

        # --- Output: inverse RoPE on tail rd=64 (HF: apply_rotary_emb(o[...,-rd:], freqs_cis, True)) ---
        o_rope_view = attn_out[..., -rd:].contiguous()
        apply_rotary_emb_2d(o_rope_view, f_slice, inverse=True)
        attn_out = torch.cat([attn_out[..., :-rd], o_rope_view], dim=-1)

        # --- Output projection (grouped) ---
        # HF: o.view(bsz, seqlen, n_groups=8, n_local_heads*head_dim/n_groups=4096)
        # group_din = NUM_HEADS * HEAD_DIM / N_GROUPS = 64*512/8 = 4096
        # wo_a weight in checkpoint: [8192, 4096] = [n_groups*o_lora=8192, group_din=4096]
        #   reshape -> [n_groups=8, o_lora=1024, group_din=4096]
        # einsum 'sgd,grd->sgr' (BF16 path per HF NOTE)
        # Then flatten -> [S, 8192] -> wo_b [4096, 8192] -> [S, 4096]
        o_grouped = attn_out.view(S, N_GROUPS, GROUP_DIN)              # [S, 8, 4096]
        # wo_a stored FP8 (8192, 4096) -> dequant once per call to BF16 for einsum
        # (mirrors HF NOTE: "could do FP8 einsum but BF16 for simplicity")
        wo_a_bf16 = self._fp8_dequant_to_bf16(layer.wo_a_w, layer.wo_a_s)   # [8192, 4096]
        wo_a_g = wo_a_bf16.view(N_GROUPS, O_LORA_RANK, GROUP_DIN)      # [8, 1024, 4096]
        o_proj = torch.einsum("sgd,grd->sgr", o_grouped.float(), wo_a_g.float())  # [S, 8, 1024]
        o_flat = o_proj.reshape(S, N_GROUPS * O_LORA_RANK).to(torch.bfloat16)     # [S, 8192]
        x = self._fp8_linear(o_flat, layer.wo_b_w, layer.wo_b_s)        # [S, 4096]
        return x

    def _fp8_dequant_to_bf16(self, w_u8: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
        """Dequantize FP8 E4M3 weight + UE8M0 block-128x128 scale to BF16. CACHED per (id(w),id(s))."""
        key = (w_u8.data_ptr(), s_u8.data_ptr())
        if not hasattr(self, "_wo_a_cache"):
            self._wo_a_cache = {}
        if key in self._wo_a_cache:
            return self._wo_a_cache[key]
        # FP8 E4M3 raw bytes -> float via torch view
        w_fp8 = w_u8.view(torch.float8_e4m3fn)
        w_f = w_fp8.float()                                             # [N, K]
        N, K = w_f.shape
        # Scale shape: [N//128, K//128] UE8M0 (FP8 E8M0)
        s_fp8 = s_u8.view(torch.float8_e8m0fnu)
        s_f = s_fp8.float()                                             # [N/128, K/128]
        # Broadcast scale to [N, K]
        s_full = s_f.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        s_full = s_full[:N, :K]
        w_bf16 = (w_f * s_full).to(torch.bfloat16).contiguous()
        # Cap cache size to avoid OOM
        if len(self._wo_a_cache) > 64:
            self._wo_a_cache.pop(next(iter(self._wo_a_cache)))
        self._wo_a_cache[key] = w_bf16
        return w_bf16

    # -- Forward --------------------------------------------------------------

    def forward_layer(self, hidden: torch.Tensor, L: int) -> torch.Tensor:
        """Forward through ONE layer with HC. hidden [S, n_hc=4, H] -> same."""
        layer = self.layers[L]

        x_pre_a, _pre, post_a, comb_a = hc_pre(
            hidden, layer.hc_attn_fn, layer.hc_attn_base, layer.hc_attn_scale,
            n_embd=HIDDEN, n_hc=HC_N_HC,
            sinkhorn_iters=HC_SINKHORN_ITERS, hc_eps=HC_EPS, norm_eps=RMS_EPS,
        )
        h_n = rmsnorm_torch(x_pre_a, layer.attn_norm, eps=RMS_EPS)
        attn = self._mla_attention(h_n, layer, L)
        hidden = hc_post(attn, hidden, post_a, comb_a)

        x_pre_f, _pre, post_f, comb_f = hc_pre(
            hidden, layer.hc_ffn_fn, layer.hc_ffn_base, layer.hc_ffn_scale,
            n_embd=HIDDEN, n_hc=HC_N_HC,
            sinkhorn_iters=HC_SINKHORN_ITERS, hc_eps=HC_EPS, norm_eps=RMS_EPS,
        )
        h_n2 = rmsnorm_torch(x_pre_f, layer.ffn_norm, eps=RMS_EPS)
        moe_out = self._moe_block(h_n2, L, layer)
        hidden = hc_post(moe_out, hidden, post_f, comb_f)
        return hidden

    def forward(self, input_ids: list[int]) -> torch.Tensor:
        """Forward pass: returns logits [VOCAB] for the LAST token (BF16, GPU)."""
        S = len(input_ids)
        ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")
        emb = torch.nn.functional.embedding(ids, self.embed).to(torch.bfloat16)
        hidden = emb.unsqueeze(1).repeat(1, HC_N_HC, 1).contiguous()

        for L in range(self.n_layers):
            hidden = self.forward_layer(hidden, L)

        self._pos += S

        final = hc_head(
            hidden, self.hc_head_fn, self.hc_head_base, self.hc_head_scale,
            n_embd=HIDDEN, n_hc=HC_N_HC, hc_eps=HC_EPS, norm_eps=RMS_EPS,
        )
        last = rmsnorm_torch(final[-1].contiguous(), self.final_norm, eps=RMS_EPS)
        logits = torch.nn.functional.linear(last, self.head)
        torch.cuda.synchronize()
        return logits

    def reset_kv(self) -> None:
        for L in range(self.n_layers):
            self._kv[L] = None
        self._pos = 0

    def generate(self, prompt_ids: list[int], max_tokens: int = 5) -> tuple[list[int], dict[str, float]]:
        """Greedy decode."""
        self.reset_kv()
        gen: list[int] = []
        t0 = time.time()
        logits = self.forward(prompt_ids)
        prefill_s = time.time() - t0
        next_id = int(logits.float().argmax().item())
        gen.append(next_id)
        t1 = time.time()
        decode_count = 0
        for _ in range(max_tokens - 1):
            logits = self.forward([next_id])
            next_id = int(logits.float().argmax().item())
            gen.append(next_id)
            decode_count += 1
        decode_s = max(time.time() - t1, 1e-9)
        return gen, {
            "prefill_s": prefill_s,
            "prefill_tps": len(prompt_ids) / max(prefill_s, 1e-9),
            "decode_s": decode_s,
            "decode_tps": decode_count / decode_s if decode_count > 0 else 0.0,
            "total_s": prefill_s + decode_s,
        }


# ----------------------------------------------------------------------------
# FINAL FACTORY - wire MLA + Hot-set + CUDA Graph + MTP 
# ----------------------------------------------------------------------------

def build_engine_full_optimized(
    hf_path: str = HF_SNAPSHOT,
    *,
    hotset_top_n: int = 64,
    enable_hotset: bool = True,
    enable_cuda_graph: bool = False,
    enable_mtp: bool = False,
    mtp_k: int = 1,
    max_layers: Optional[int] = None,
) -> "DSv4EngineMXFP4":
    """Build engine MLA full + optional perf patches (hot-set / CUDA-Graph / MTP).

    Each patch is wrapped in try/except: failure does NOT abort the build, only
    logs and falls back to the baseline. Order: hot-set first (touches loader),
    CUDA Graph next (captures decode after warm), MTP last (wraps generate).

    Returns the (possibly patched) engine. Inspect engine._patch_status for the
    actual list of patches that succeeded.
    """
    import time as _time
    _t_total = _time.time()
    engine = DSv4EngineMXFP4(hf_path, max_layers=max_layers)
    engine._patch_status = {
        "baseline_mla": True,
        "hotset": False,
        "cuda_graph": False,
        "mtp": False,
        "errors": [],
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _mem = torch.cuda.memory_allocated() / 1e9
        _log(f"[FINAL] base engine init done in {_time.time()-_t_total:.1f}s, GPU mem {_mem:.2f} GB")

    # 1. Hot-set (no LRU miss for top-N expert routing)
    if enable_hotset:
        try:
            from expert_prewarm_patch import patch_engine_with_hotset
            engine = patch_engine_with_hotset(
                engine,
                top_n=hotset_top_n,
                strategy="per_layer",
                n_layers=engine.n_layers,
                verbose=True,
            )
            engine._patch_status["hotset"] = True
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                _mem = torch.cuda.memory_allocated() / 1e9
                _log(f"[FINAL] +hotset top-{hotset_top_n} done, GPU mem {_mem:.2f} GB")
        except Exception as _e:
            engine._patch_status["errors"].append(f"hotset: {type(_e).__name__}: {_e}")
            _log(f"[FINAL] hot-set FAIL (continuing): {_e}")

    # 2. CUDA Graph (decode replay)
    if enable_cuda_graph:
        try:
            from cuda_graph_engine_patch import patch_engine_with_cuda_graph
            engine = patch_engine_with_cuda_graph(
                engine, n_layers=engine.n_layers, hidden=HIDDEN, n_hc=HC_N_HC,
            )
            engine._patch_status["cuda_graph"] = True
            _log("[FINAL] +cuda_graph done")
        except Exception as _e:
            engine._patch_status["errors"].append(f"cuda_graph: {type(_e).__name__}: {_e}")
            _log(f"[FINAL] cuda_graph FAIL (continuing): {_e}")

    # 3. MTP speculative decode (k=1, EAGLE-style verify)
    if enable_mtp:
        try:
            from mtp_engine_hook import patch_engine_with_mtp
            engine = patch_engine_with_mtp(engine, k=mtp_k)
            engine._patch_status["mtp"] = True
            _log(f"[FINAL] +MTP k={mtp_k} done")
        except Exception as _e:
            engine._patch_status["errors"].append(f"mtp: {type(_e).__name__}: {_e}")
            _log(f"[FINAL] MTP FAIL (continuing): {_e}")

    _log(f"[FINAL] build complete in {_time.time()-_t_total:.1f}s, status={engine._patch_status}")
    return engine




# --- CLI ---------------------------------------------------------------------

def main() -> int:
    hf = sys.argv[1] if len(sys.argv) > 1 else HF_SNAPSHOT
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_layers = int(os.environ.get("DSV4_MXFP4_MAX_LAYERS", "0")) or None
    eng = DSv4EngineMXFP4(hf, max_layers=max_layers)

    prompts = [
        ("BOS", [0]),
        ("Hi",  [0, 100, 105]),
    ]
    for name, ids in prompts:
        try:
            torch.cuda.synchronize()
            mem0 = torch.cuda.memory_allocated() / 1e9
            _log(f"\n=== prompt {name} ids={ids} max_new={n_tokens} ===")
            gen, t = eng.generate(ids, max_tokens=n_tokens)
            mem1 = torch.cuda.memory_allocated() / 1e9
            _log(f"  generated ids: {gen}")
            _log(f"  prefill: {t['prefill_s']:.2f}s ({t['prefill_tps']:.2f} tok/s)")
            _log(f"  decode:  {t['decode_s']:.2f}s ({t['decode_tps']:.2f} tok/s)")
            _log(f"  GPU mem before/after: {mem0:.2f} / {mem1:.2f} GB")
        except Exception as e:
            _log(f"  FAIL: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    _log("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
