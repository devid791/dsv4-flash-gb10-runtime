from __future__ import annotations
"""
dsv4_engine.py — DSv4-Flash native inference engine on NVIDIA GB10 (sm_121a).

Davide Zenati, 2026-05-03. NO vLLM. NO HF generate(). Only tokenizer
from transformers.

Architecture (top-down):

    DSv4Engine.generate(prompt, ...)        ── this file
        ├── tokenize via HF tokenizer      ── transformers.AutoTokenizer
        ├── loop:
        │     forward(input_ids) -> logits ── this file
        │     sample(logits, ...)          ── runtime.dsv4_sampler
        │     append next_token
        └── decode

    DSv4Engine.forward(input_ids):
        ├── embed lookup                    ── torch (BF16 unified)
        ├── for L in 0..num_layers-1:
        │     residual = hidden
        │     hidden_norm = rmsnorm(hidden) ── runtime.rmsnorm (libdsv4_rmsnorm.so)
        │     append KV (latent + k_rope)   ── libdsv4_kvcache.so
        │     indexer.append(hidden_norm)   ── libdsv4_indexer.so
        │     query_idx = q_proj_indexer    ── small linear (host PyTorch)
        │     top_k_blocks = indexer.score  ── libdsv4_indexer.so
        │     attn_out = mla_forward(...)   ── libdsv4_mla.so
        │     hidden = residual + attn_out
        │     residual2 = hidden
        │     hidden_norm2 = rmsnorm(hidden)
        │     # MoE block:
        │     gate_logits = router(hidden_norm2)            ── small linear (BF16)
        │     topk_ids, topk_w = top_k(gate_logits)         ── torch.topk
        │     moe_out = our_fp4_moe_swiglu_run(...)         ── libour_fp4moe.so
        │     # Shared expert via dense GEMM (3 calls SwiGLU) optional;
        │     # already folded inside fp4_moe_swiglu_run via shared_expert_*.bin
        │     hidden = residual2 + moe_out
        │
        ├── final RMSNorm
        └── lm_head: dense fp4 GEMM        ── libour_fp4gemm.so → logits

ABI summary (from headers in kernel/*):

  libdsv4_kvcache.so  ── kv_cache_fp8_init/free/append/gather/drop/
                          pool_acquire/release/stats
  libdsv4_indexer.so  ── lightning_indexer_init/append/score/
                          free_cache/free
  libdsv4_mla.so      ── mla_attention_init/forward/free
  libour_fp4moe.so    ── fp4_moe_weights_create_swiglu/destroy/
                          fp4_moe_swiglu_workspace_size/run
  libour_fp4gemm.so   ── fp4_weights_create/destroy/
                          fp4_gemm_workspace_size/run/run_packed
  libdsv4_rmsnorm.so  ── ds_rmsnorm_forward (this file ships rmsnorm.cu)

Memory budget verified vs DESIGN_SYSTEM.md §3:
  dense backbone        ~10 GiB  (mmap weights kept on NVMe; only active layer
                                  resident at any time via load_dense_only LRU)
  KV cache FP8 latent    ~9.6 GiB at 1 M ctx (slab pool)
  Lightning indexer cache STREAMED → 1.2 GiB @ 64K, ≤18 GiB @ 1 M
  MoE hot expert cache  ~24 GiB  (LRU 96 expert × 256 MiB, kernel-internal)
  activations workspace  ~2 GiB
                         ───────
  total resident         ≤30 GiB at 64K ctx, ≤55 GiB at 1 M ctx

Latency target (single sequence, warm cache):
  prefill: ~5 ms / token amortized
  decode:  ≤500 ms / token (24 tok/s) — kernels are the long pole
"""


import argparse
import ctypes
import json
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Local imports — runtime/ siblings.
_RUNTIME_DIR = Path(__file__).resolve().parent
if str(_RUNTIME_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DIR.parent))

from runtime import dsv4_sampler  # noqa: E402  (sibling)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

# ── Diagnostic helper (DSV4_DIAG=1) ─────────────────────────────────────────
def _diag(label, t):
    import os
    if os.environ.get("DSV4_DIAG") != "1":
        return
    try:
        x = t.detach()
        amax = float(x.abs().max())
        amean = float(x.abs().mean())
        n = float(x.float().norm())
        nz = int((x != 0).sum())
        print(f"[diag] {label}: shape={tuple(x.shape)} dtype={x.dtype} amax={amax:.4e} amean={amean:.4e} norm={n:.4e} nonzero={nz}", flush=True)
    except Exception as e:
        print(f"[diag] {label}: error {e}", flush=True)


def _log(msg: str) -> None:
    print(f"[dsv4_engine] {msg}", file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DSv4Config:
    """Subset of HF config.json we actually need.

    Defaults reflect DSv4-Flash (43L, head_dim=512, qk_nope=448, qk_rope=64,
    no kv_lora_rank — single-head MQA with head_dim=512 acts as the latent).
    For classical V3-MLA checkpoints (kv_lora_rank present, head_dim=128) the
    same dataclass works because from_json() prefers explicit JSON values and
    only synthesises V4 fallbacks when the JSON field is null/absent.
    """
    # Model topology
    num_hidden_layers: int = 43
    hidden_size: int = 4096
    intermediate_size: int = 18432           # dense MLP intermediate
    moe_intermediate_size: int = 2048        # per-expert FFN intermediate (N_inter)
    vocab_size: int = 129280

    # Attention — V4 style (head_dim>0 = single MQA head latent of size head_dim)
    num_attention_heads: int = 64
    num_key_value_heads: int = 1             # V4-Flash: 1 (MQA) ; V3 has 128
    head_dim: int = 512                      # V4 single-head MQA latent dim
    kv_lora_rank: int = 512                  # V4 fallback = head_dim ; V3 = 512
    q_lora_rank: int = 1024                  # V4=1024 ; V3=1536
    o_lora_rank: int = 1024                  # V4 only (LoRA on o_proj) ; V3 absent
    qk_nope_head_dim: int = 448              # V4=448 ; V3=128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 512                    # V4 fallback = head_dim ; V3=128
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 16.0        # V4-Flash YaRN factor=16 ; V3=40
    max_position_embeddings: int = 1_048_576

    # MoE
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6             # V4-Flash=6 ; V3=8

    # Lightning Indexer (DSA) — V4-Flash names: index_head_dim, index_n_heads, index_topk
    indexer_dim: int = 128                   # V4: index_head_dim
    indexer_heads: int = 64                  # V4: index_n_heads
    compression_factor_m: int = 4            # V4 has compress_ratios[] per-layer; 4 is the
                                             # finest non-zero ratio used by alternating layers.
                                             # The kernel currently uses one global m — we pick the
                                             # smallest >0 entry; layers using m=128 will fall back
                                             # to dense gather until per-layer m is plumbed through.
    indexer_top_k: int = 512                 # V4: index_topk

    # First N layers may be dense (no MoE) on some DSv4 variants. V4-Flash uses
    # hash-cluster routing on every layer (first_k_dense_replace=null) so we
    # default to 0 — every layer goes through the MoE path.
    first_k_dense_replace: int = 0

    # SwiGLU activation clamp limit (DSv4 config.json field, default None = no clamp).
    # When non-null, both shared-expert and routed-expert SwiGLU apply:
    #   gate = clamp(gate, max=limit)
    #   up   = clamp(up, min=-limit, max=limit)
    # This is REQUIRED for DSv4: without it the residual stream amplitude
    # explodes through later layers and the lm_head argmax produces gibberish
    # past the first lucky bigram (F1.13 root cause).
    swiglu_limit: float | None = 10.0

    @classmethod
    def from_json(cls, path: Path) -> "DSv4Config":
        with open(path) as f:
            j = json.load(f)
        # HF DSv4 keys can be nested under "moe_config" or top-level depending
        # on checkpoint vintage. Be permissive: look up flat first, fall back
        # to a sub-dict, fall back to dataclass default. We also accept the
        # alternate V4-Flash names (index_head_dim/index_n_heads/index_topk).
        _aliases = {
            "indexer_dim":          ("index_head_dim",),
            "indexer_heads":        ("index_n_heads",),
            "indexer_top_k":        ("index_topk",),
        }
        def g(name: str, default: Any) -> Any:
            for k in (name, *_aliases.get(name, ())):
                if k in j and j[k] is not None:
                    return j[k]
            for sub in ("moe_config", "indexer_config", "attn_config"):
                if sub in j and isinstance(j[sub], dict):
                    for k in (name, *_aliases.get(name, ())):
                        if k in j[sub] and j[sub][k] is not None:
                            return j[sub][k]
            # V4 specifics: synthesise sane fallbacks when JSON has null
            return default

        flds = {f.name: g(f.name, getattr(cls, f.name)) for f in cls.__dataclass_fields__.values()}
        cfg = cls(**flds)

        # ── V4-Flash null-field reconciliation ────────────────────────────────
        # head_dim drives kv_lora_rank / v_head_dim when the latter are null in
        # the JSON. DSv4-Flash sets kv_lora_rank=null, v_head_dim=null, head_dim=512;
        # the effective latent rank IS head_dim because V4 uses single-head MQA
        # (num_key_value_heads=1) and the wkv tensor projects hidden→head_dim
        # directly (no separate kv_lora_rank latent → kv_lora_rank up-projection).
        raw_kv_lora = j.get("kv_lora_rank")
        if raw_kv_lora is None:
            cfg.kv_lora_rank = cfg.head_dim
        raw_v_head = j.get("v_head_dim")
        if raw_v_head is None:
            cfg.v_head_dim = cfg.head_dim

        # compress_kv_dim is V4's name for the indexer-side latent KV dim; if
        # null we reuse kv_lora_rank (post-fallback) — keeps the indexer cache
        # struct happy.
        # (Field not stored on dataclass; left here as documentation.)

        # first_k_dense_replace=null in V4 → 0 (every layer is MoE).
        raw_dense = j.get("first_k_dense_replace")
        if raw_dense is None:
            cfg.first_k_dense_replace = 0

        return cfg

    @property
    def softmax_scale(self) -> float:
        # MLA softmax uses concat(qk_nope, qk_rope) head_dim
        return 1.0 / ((self.qk_nope_head_dim + self.qk_rope_head_dim) ** 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# ctypes wrappers — per-lib mini bindings
# ─────────────────────────────────────────────────────────────────────────────
class _Lib:
    """Convenience holder: load .so + register signatures."""

    def __init__(self, libs_dir: Path, fname: str):
        full = libs_dir / fname
        if not full.exists():
            raise FileNotFoundError(f"missing kernel lib: {full}")
        self.path = full
        self.lib = ctypes.CDLL(str(full), mode=ctypes.RTLD_GLOBAL)


def _bind_kvcache(lib: ctypes.CDLL) -> None:
    lib.kv_cache_fp8_init.restype = ctypes.c_int
    lib.kv_cache_fp8_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.kv_cache_fp8_free.restype = ctypes.c_int
    lib.kv_cache_fp8_free.argtypes = [ctypes.c_void_p]
    lib.kv_cache_fp8_append.restype = ctypes.c_int
    lib.kv_cache_fp8_append.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.kv_cache_fp8_gather.restype = ctypes.c_int
    lib.kv_cache_fp8_gather.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.kv_cache_fp8_drop.restype = ctypes.c_int
    lib.kv_cache_fp8_drop.argtypes = [ctypes.c_void_p, ctypes.c_int]
    # F1-A4: trim API for removing pad slots after prefill
    lib.kv_cache_fp8_trim.restype = ctypes.c_int
    lib.kv_cache_fp8_trim.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]


def _bind_indexer(lib: ctypes.CDLL) -> None:
    lib.lightning_indexer_init.restype = ctypes.c_int
    lib.lightning_indexer_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.lightning_indexer_append.restype = ctypes.c_int
    lib.lightning_indexer_append.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.lightning_indexer_score.restype = ctypes.c_int
    lib.lightning_indexer_score.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.lightning_indexer_free_cache.restype = ctypes.c_int
    lib.lightning_indexer_free_cache.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.lightning_indexer_free.restype = ctypes.c_int
    lib.lightning_indexer_free.argtypes = [ctypes.c_void_p]


def _bind_mla(lib: ctypes.CDLL) -> None:
    lib.mla_attention_init.restype = ctypes.c_int
    lib.mla_attention_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.mla_attention_forward.restype = ctypes.c_int
    lib.mla_attention_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.mla_attention_free.restype = ctypes.c_int
    lib.mla_attention_free.argtypes = [ctypes.c_void_p]


def _bind_mla_v4(lib: ctypes.CDLL) -> None:
    """ABI for libdsv4_mla_v4.so (V4-Flash MLA: per-layer LoRA pair on Q and O,
    single-head MQA latent on KV, attention sink, no Lightning Indexer).

    mla_v4_init(MLAv4Config*, MLAv4State**)
    mla_v4_set_layer_weights(state, layer_id,
        wkv_packed, wkv_scale, wkv_gscale,
        wq_a_packed, wq_a_scale, wq_a_gscale,
        q_norm_bf16,
        wq_b_packed, wq_b_scale, wq_b_gscale,
        wo_a_packed, wo_a_scale, wo_a_gscale,
        wo_b_packed, wo_b_scale, wo_b_gscale,
        attn_sink_bf16)
    mla_v4_forward(state, layer_id, hidden_bf16, seq_len, past_len,
        kv_cache_state, positions_i32, output_bf16, stream)
    mla_v4_free(state)
    """
    lib.mla_v4_init.restype = ctypes.c_int
    lib.mla_v4_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]

    lib.mla_v4_set_layer_weights.restype = ctypes.c_int
    lib.mla_v4_set_layer_weights.argtypes = [
        ctypes.c_void_p,    # state
        ctypes.c_int,       # layer_id
        # wkv triplet
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        # wq_a triplet
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        # q_norm bf16
        ctypes.c_void_p,
        # wq_b triplet
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        # wo_a triplet
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        # wo_b triplet
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        # attn_sink bf16
        ctypes.c_void_p,
    ]

    lib.mla_v4_forward.restype = ctypes.c_int
    lib.mla_v4_forward.argtypes = [
        ctypes.c_void_p,    # state
        ctypes.c_int,       # layer_id
        ctypes.c_void_p,    # hidden bf16 [seq_len, hidden]
        ctypes.c_int,       # seq_len (incl pad)
        ctypes.c_int,       # past_len
        ctypes.c_void_p,    # kv_cache_state (libdsv4_kvcache state*)
        ctypes.c_void_p,    # positions int32 [seq_len]
        ctypes.c_void_p,    # output bf16 [seq_len, hidden]
        ctypes.c_void_p,    # cudaStream_t
        ctypes.c_int,       # valid_seq_len (real prompt len; <=0 => use seq_len)
        ctypes.c_int,       # is_decode (F1-A4: 1=GEMV decode-path, 0=GEMM prefill)
    ]

    lib.mla_v4_free.restype = ctypes.c_int
    lib.mla_v4_free.argtypes = [ctypes.c_void_p]


def _bind_moe(lib: ctypes.CDLL) -> None:
    lib.fp4_moe_version.restype = ctypes.c_char_p
    lib.fp4_moe_weights_create_swiglu.restype = ctypes.c_void_p
    lib.fp4_moe_weights_create_swiglu.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.fp4_moe_weights_destroy.restype = None
    lib.fp4_moe_weights_destroy.argtypes = [ctypes.c_void_p]
    lib.fp4_moe_swiglu_workspace_size.restype = ctypes.c_size_t
    lib.fp4_moe_swiglu_workspace_size.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.fp4_moe_swiglu_run.restype = ctypes.c_int
    lib.fp4_moe_swiglu_run.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
    ]

    # F4 hot-set ABI (added 2026-05-04). Optional: bind only if symbol exists.
    if hasattr(lib, "fp4_moe_pin_expert_swiglu"):
        lib.fp4_moe_pin_expert_swiglu.restype = ctypes.c_int
        lib.fp4_moe_pin_expert_swiglu.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.fp4_moe_unpin_expert_swiglu.restype = ctypes.c_int
        lib.fp4_moe_unpin_expert_swiglu.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.fp4_moe_load_hot_set_from_json_swiglu.restype = ctypes.c_int
        lib.fp4_moe_load_hot_set_from_json_swiglu.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ]


def _bind_dense(lib: ctypes.CDLL) -> None:
    lib.fp4_gemm_version.restype = ctypes.c_char_p
    lib.fp4_weights_create.restype = ctypes.c_void_p
    lib.fp4_weights_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.fp4_weights_destroy.restype = None
    lib.fp4_weights_destroy.argtypes = [ctypes.c_void_p]
    lib.fp4_gemm_workspace_size.restype = ctypes.c_size_t
    lib.fp4_gemm_workspace_size.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.fp4_gemm_run.restype = ctypes.c_int
    lib.fp4_gemm_run.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
    ]


def _bind_rmsnorm(lib: ctypes.CDLL) -> None:
    # ds_rmsnorm_forward(out, in, weight, n_rows, n_cols, eps, stream)
    lib.ds_rmsnorm_forward.restype = ctypes.c_int
    lib.ds_rmsnorm_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Plain-C struct packers (must mirror .cuh layouts)
# ─────────────────────────────────────────────────────────────────────────────
def _pack_kvcache_cfg(cfg: DSv4Config) -> bytes:
    # struct KvCacheFp8Config { int num_layers, kv_lora_rank, qk_rope_head_dim,
    #                           slab_bytes, initial_slab_count, max_slab_count,
    #                           reserved_for_indexer; }  — 7 ints
    return struct.pack(
        "<7i",
        cfg.num_hidden_layers,
        cfg.kv_lora_rank,
        cfg.qk_rope_head_dim,
        64 * 1024 * 1024,    # slab_bytes
        64,                   # initial_slab_count = 4 GiB pool
        1024,                 # max_slab_count = 64 GiB hard cap
        16,                   # reserved_for_indexer
    )


def _pack_indexer_cfg(cfg: DSv4Config) -> bytes:
    # struct LightningIndexerConfig { int num_layers, hidden_size, indexer_dim,
    #                                 indexer_heads, compression_factor_m, top_k,
    #                                 max_position_embeddings, initial_slab_count,
    #                                 slab_bytes; } — 9 ints
    return struct.pack(
        "<9i",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.indexer_dim,
        cfg.indexer_heads,
        cfg.compression_factor_m,
        cfg.indexer_top_k,
        cfg.max_position_embeddings,
        16,
        64 * 1024 * 1024,
    )


def _pack_mla_cfg(cfg: DSv4Config) -> bytes:
    # Legacy V3 MLAConfig (kept for backward compat with libdsv4_mla.so)
    # struct MLAConfig { int num_layers, num_heads, head_dim, kv_lora_rank, q_lora_rank,
    #                    qk_rope_head_dim, qk_nope_head_dim, v_head_dim,
    #                    max_position_embeddings;
    #                    float rope_theta, rope_scaling_factor;
    #                    int hidden_size; float softmax_scale; }
    return struct.pack(
        "<9i 2f i f",
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
        cfg.kv_lora_rank,
        cfg.q_lora_rank,
        cfg.qk_rope_head_dim,
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
        cfg.max_position_embeddings,
        cfg.rope_theta,
        cfg.rope_scaling_factor,
        cfg.hidden_size,
        cfg.softmax_scale,
    )


def _pack_mla_v4_cfg(cfg: DSv4Config) -> bytes:
    """V4 MLA config: <9i 5f>
       num_layers, hidden_size, num_heads, qk_nope_head_dim, qk_rope_head_dim,
       head_dim, q_lora_rank, o_lora_rank, max_position_embeddings, _pad,
       rope_theta, compress_rope_theta, yarn_factor, beta_fast, beta_slow
    """
    return struct.pack(
        "<9i 5f",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        cfg.head_dim,
        cfg.q_lora_rank,
        cfg.o_lora_rank,
        cfg.max_position_embeddings,
        cfg.rope_theta,
        cfg.rope_theta,                     # compress_rope_theta — same as rope unless cfg adds it
        cfg.rope_scaling_factor,            # yarn_factor
        32.0,                               # beta_fast (DSv4 default)
        1.0,                                # beta_slow (DSv4 default)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dense backbone loader — reads dense_backbone.bin emitted by prequantize_dsv4.py
# ─────────────────────────────────────────────────────────────────────────────
DENS_MAGIC = 0x4E534544     # 'DENS' little-endian header tag

# Map raw dtype-string back to torch dtype.
_DTYPE_MAP = {
    "uint8": torch.uint8, "int8": torch.int8,
    "float32": torch.float32, "float16": torch.float16,
    "int32": torch.int32, "int64": torch.int64, "bool": torch.bool,
}


def load_dense_only(dense_path: Path, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Stream dense_backbone.bin into a dict of tensors, materialized on `device`.

    File format (must mirror prequantize_dsv4.py write loop):
        for each tensor:
            magic 'DENS' u32     (0x4E534544)
            klen   u32  + key bytes
            dtlen  u32  + dtype bytes
            ndim   u32  + ndim * i64 shape
            dlen   u64  + data bytes
    Tensors stored as fp32 (BF16/FP16 promoted on write); we cast to BF16 here
    for non-norm layers, keep FP32 for RMSNorm scales.
    """
    out: dict[str, torch.Tensor] = {}
    sz = dense_path.stat().st_size
    with open(dense_path, "rb") as f:
        while True:
            head4 = f.read(4)
            if not head4:
                break
            if len(head4) < 4 or struct.unpack("<I", head4)[0] != DENS_MAGIC:
                raise ValueError(f"dense_backbone.bin: bad magic at offset {f.tell()-4}")
            (klen,) = struct.unpack("<I", f.read(4))
            key = f.read(klen).decode()
            (dtlen,) = struct.unpack("<I", f.read(4))
            dt_str = f.read(dtlen).decode()
            (ndim,) = struct.unpack("<I", f.read(4))
            shape = list(struct.unpack(f"<{ndim}q", f.read(8 * ndim)))
            (dlen,) = struct.unpack("<Q", f.read(8))
            data = f.read(dlen)
            np_dt = dt_str
            src_dtype = _DTYPE_MAP.get(np_dt, torch.float32)
            t = torch.frombuffer(bytearray(data), dtype=src_dtype)
            t = t.view(*shape) if shape else t
            # Dtype rules:
            #   - uint8 source = NVFP4-packed nibble stream → keep uint8 (DO NOT cast)
            #   - *.weight_scale FP32 → keep fp32 (per-block scale, kernel reads as fp32)
            #   - *.weight_global_scale FP32 → keep fp32 (per-tensor global scale)
            #   - norm/sink/scalar params → keep FP32 for fidelity
            #   - everything else (BF16 logical, e.g. embed/lm_head/q_norm) → BF16
            if src_dtype == torch.uint8:
                # NVFP4-packed weight, stays uint8
                t = t.to(device=device)
            elif key.endswith("weight_scale") or key.endswith("weight_global_scale") or key.endswith("input_global_scale"):
                t = t.to(dtype=torch.float32, device=device)
            elif key.endswith("norm.weight") or "norm." in key or key.endswith("attn_sink"):
                t = t.to(dtype=torch.float32, device=device)
            else:
                t = t.to(dtype=torch.bfloat16, device=device)
            out[key] = t
        if f.tell() != sz:
            _log(f"WARN dense_backbone.bin trailing bytes ignored: {sz - f.tell()}")
    _log(f"loaded {len(out)} dense tensors from {dense_path.name} ({sz/1e9:.2f} GB)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer state container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LayerWeights:
    """All the dense (non-routed-expert) weights for ONE transformer layer.

    Field names keep the V3-MLA semantics (q_a_proj, kv_a_proj_with_mqa,
    kv_b_proj, o_proj) for kernel ABI continuity, but the slicer below maps
    BOTH naming conventions:

      V3 (HF DeepSeek-V3 / R1):
        layers.<L>.self_attn.q_a_proj.weight             → q_a_proj
        layers.<L>.self_attn.q_a_layernorm.weight        → q_a_layernorm
        layers.<L>.self_attn.q_b_proj.weight             → q_b_proj
        layers.<L>.self_attn.kv_a_proj_with_mqa.weight   → kv_a_proj_with_mqa
        layers.<L>.self_attn.kv_a_layernorm.weight       → kv_a_layernorm
        layers.<L>.self_attn.kv_b_proj.weight            → kv_b_proj
        layers.<L>.self_attn.o_proj.weight               → o_proj
        layers.<L>.input_layernorm.weight                → input_layernorm
        layers.<L>.post_attention_layernorm.weight       → post_attention_layernorm

      V4-Flash (DeepseekV4ForCausalLM, drop-in fields below):
        layers.<L>.attn.wq_a.weight                      → q_a_proj    (hidden→q_lora_rank=1024)
        layers.<L>.attn.q_norm.weight                    → q_a_layernorm
        layers.<L>.attn.wq_b.weight                      → q_b_proj    (q_lora→num_heads*head_dim=64*512)
        layers.<L>.attn.wkv.weight                       → kv_a_proj_with_mqa (hidden→head_dim=512)  *** V4 single MQA head, no kv_lora→kv_b decompose ***
        layers.<L>.attn.kv_norm.weight                   → kv_a_layernorm
        (V4 has NO kv_b_proj — the up-projection is folded into wq_b/wo_b LoRA pair)
        layers.<L>.attn.wo_a.weight + wo_b.weight        → o_proj      (LoRA pair, rank o_lora_rank=1024 — concat as a single tensor: see _v4_o_proj_fuse)
        layers.<L>.attn_norm.weight                      → input_layernorm
        layers.<L>.ffn_norm.weight                       → post_attention_layernorm

      Optional V4-only tensors stashed on this struct (forward path uses them
      only for the layers that have them):
        attn_sink:        layers.<L>.attn.attn_sink           (43/43 layers)
        compressor_*:     layers.<L>.attn.compressor.*        (41/43 layers, NOT layer 0/1)
        indexer_*:        layers.<L>.attn.indexer.*           (21/43 layers, even ≥2)
    """
    # MLA projections (BF16 device)
    q_a_proj: torch.Tensor          # [hidden, q_lora_rank]
    q_a_layernorm: torch.Tensor     # [q_lora_rank]
    q_b_proj: torch.Tensor          # [q_lora_rank, num_heads * (qk_nope+qk_rope)]
    kv_a_proj_with_mqa: torch.Tensor  # [hidden, kv_lora_rank + qk_rope_head_dim]  (V3) or [hidden, head_dim] (V4)
    kv_a_layernorm: torch.Tensor    # [kv_lora_rank] (V3) or [head_dim] (V4)
    kv_b_proj: torch.Tensor | None   # [kv_lora_rank, num_heads * (qk_nope + v_head_dim)] — V4: None (folded)
    o_proj: torch.Tensor            # [num_heads * v_head_dim, hidden] — V4: synthesised from wo_a/wo_b LoRA pair
    # Norms
    input_layernorm: torch.Tensor   # [hidden]
    post_attention_layernorm: torch.Tensor   # [hidden]
    # Router (BF16) — None for dense-only first_k layers
    router_weight: torch.Tensor | None = None
    # Indexer query projection (small linear, BF16) — V3 single tensor; V4 only on indexer layers
    indexer_q_proj: torch.Tensor | None = None
    # ── V4-Flash extras (None on V3 checkpoints) ───────────────────────────────
    attn_sink: torch.Tensor | None = None              # [num_heads] sink logits per head
    o_proj_a: torch.Tensor | None = None               # V4: wo_a [num_heads*v_head_dim, o_lora_rank]
    o_proj_b: torch.Tensor | None = None               # V4: wo_b [o_lora_rank, hidden]
    compressor_wkv: torch.Tensor | None = None         # V4 compressor.wkv
    compressor_wgate: torch.Tensor | None = None       # V4 compressor.wgate
    compressor_norm: torch.Tensor | None = None        # V4 compressor.norm
    compressor_ape: torch.Tensor | None = None         # V4 compressor.ape (axial positional)
    indexer_wq_b: torch.Tensor | None = None           # V4 indexer.wq_b
    indexer_weights_proj: torch.Tensor | None = None   # V4 indexer.weights_proj
    indexer_compressor_wkv: torch.Tensor | None = None
    indexer_compressor_wgate: torch.Tensor | None = None
    indexer_compressor_norm: torch.Tensor | None = None
    indexer_compressor_ape: torch.Tensor | None = None


# Sentinel returned by _slice_layer_weights().get() when a field is genuinely
# optional and absent from BOTH V3 and V4 naming.
_MISSING = object()


def _slice_layer_weights(dense: dict[str, torch.Tensor], layer_idx: int,
                         cfg: DSv4Config) -> LayerWeights:
    """Pluck per-layer tensors out of the flat dense dict by name pattern,
    transparently handling V3 (DeepSeek-V3 / R1) and V4-Flash naming.
    """
    p = f"layers.{layer_idx}"
    def get_first(*candidates: str, optional: bool = False) -> torch.Tensor | None:
        """Return the first tensor whose key matches any of the candidates,
        searching with and without the 'model.' prefix."""
        tried: list[str] = []
        for suffix in candidates:
            for prefix in (p, f"model.{p}"):
                k = f"{prefix}.{suffix}"
                tried.append(k)
                if k in dense:
                    return dense[k]
        if optional:
            return None
        raise KeyError(f"layer {layer_idx}: none of these keys found: {tried}")

    # ── Detect V4 by presence of attn.wq_a (V4) vs self_attn.q_a_proj (V3) ──
    v4 = any(k.startswith(f"{p}.attn.") or k.startswith(f"model.{p}.attn.") for k in dense)

    # ── Q LoRA pair ─────────────────────────────────────────────────────────
    q_a_proj      = get_first("attn.wq_a.weight",           "self_attn.q_a_proj.weight")
    q_a_layernorm = get_first("attn.q_norm.weight",         "self_attn.q_a_layernorm.weight")
    q_b_proj      = get_first("attn.wq_b.weight",           "self_attn.q_b_proj.weight")

    # ── KV path ─────────────────────────────────────────────────────────────
    # V3:  kv_a_proj_with_mqa = [hidden, kv_lora_rank + qk_rope_head_dim]
    #       kv_b_proj          = [kv_lora_rank, num_heads*(qk_nope+v_head_dim)]
    # V4:  wkv  = [hidden, head_dim]   — single MQA head, no separate b-side
    kv_a_proj_with_mqa = get_first("attn.wkv.weight",              "self_attn.kv_a_proj_with_mqa.weight")
    kv_a_layernorm     = get_first("attn.kv_norm.weight",          "self_attn.kv_a_layernorm.weight")
    kv_b_proj          = get_first("self_attn.kv_b_proj.weight",   optional=True)  # V3-only

    # ── O projection ─────────────────────────────────────────────────────────
    # V3: single o_proj.weight ; V4: LoRA pair wo_a + wo_b — we expose both
    # halves AND a synthesised dense product for kernel paths that still want
    # one tensor (init can register the synthesised one; forward can use the
    # rank-1024 pair to save FLOPs once the kernel is V4-aware).
    o_proj_a = get_first("attn.wo_a.weight", optional=True)
    o_proj_b = get_first("attn.wo_b.weight", optional=True)
    if o_proj_a is not None and o_proj_b is not None:
        # wo_a: [num_heads*v_head_dim, o_lora_rank]   (HF stores as [out, in])
        # wo_b: [o_lora_rank, hidden_size]
        # Effective o_proj = wo_b.T @ wo_a.T  →  [num_heads*v_head_dim, hidden_size]
        # We DO NOT materialise the dense product here (would cost ~134MB per layer
        # × 43 layers = 5.7GB). Instead we set o_proj=None for V4 and let the
        # MLA kernel apply wo_a + wo_b in two GEMMs. Calling code that relied on
        # `o_proj` being non-None must check for the LoRA pair first.
        o_proj_dense = None
    else:
        o_proj_dense = get_first("self_attn.o_proj.weight")

    # ── Norms ────────────────────────────────────────────────────────────────
    input_layernorm          = get_first("attn_norm.weight",            "input_layernorm.weight")
    post_attention_layernorm = get_first("ffn_norm.weight",             "post_attention_layernorm.weight")

    # ── Router ───────────────────────────────────────────────────────────────
    router_weight = get_first("ffn.gate.weight", optional=True)

    # ── V4 indexer extras (only on layers that ship them) ────────────────────
    indexer_wq_b              = get_first("attn.indexer.wq_b.weight",                  optional=True)
    indexer_weights_proj      = get_first("attn.indexer.weights_proj.weight",          optional=True)
    indexer_compressor_wkv    = get_first("attn.indexer.compressor.wkv.weight",        optional=True)
    indexer_compressor_wgate  = get_first("attn.indexer.compressor.wgate.weight",      optional=True)
    indexer_compressor_norm   = get_first("attn.indexer.compressor.norm.weight",       optional=True)
    indexer_compressor_ape    = get_first("attn.indexer.compressor.ape",               optional=True)

    # ── V4 main compressor (separate from indexer compressor) ────────────────
    compressor_wkv   = get_first("attn.compressor.wkv.weight",   optional=True)
    compressor_wgate = get_first("attn.compressor.wgate.weight", optional=True)
    compressor_norm  = get_first("attn.compressor.norm.weight",  optional=True)
    compressor_ape   = get_first("attn.compressor.ape",          optional=True)

    # ── Attention sink (V4 only) + V3 indexer.q_proj fallback ────────────────
    attn_sink       = get_first("attn.attn_sink",                optional=True)
    indexer_q_proj  = get_first("self_attn.indexer.q_proj.weight", optional=True)

    return LayerWeights(
        q_a_proj=q_a_proj,
        q_a_layernorm=q_a_layernorm,
        q_b_proj=q_b_proj,
        kv_a_proj_with_mqa=kv_a_proj_with_mqa,
        kv_a_layernorm=kv_a_layernorm,
        kv_b_proj=kv_b_proj,
        o_proj=o_proj_dense if o_proj_dense is not None else (o_proj_a if o_proj_a is not None else q_a_proj),  # placeholder for V4 path; never used directly when o_proj_a/b set
        input_layernorm=input_layernorm,
        post_attention_layernorm=post_attention_layernorm,
        router_weight=router_weight,
        indexer_q_proj=indexer_q_proj,
        attn_sink=attn_sink,
        o_proj_a=o_proj_a,
        o_proj_b=o_proj_b,
        compressor_wkv=compressor_wkv,
        compressor_wgate=compressor_wgate,
        compressor_norm=compressor_norm,
        compressor_ape=compressor_ape,
        indexer_wq_b=indexer_wq_b,
        indexer_weights_proj=indexer_weights_proj,
        indexer_compressor_wkv=indexer_compressor_wkv,
        indexer_compressor_wgate=indexer_compressor_wgate,
        indexer_compressor_norm=indexer_compressor_norm,
        indexer_compressor_ape=indexer_compressor_ape,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DSv4Engine — public class
# ─────────────────────────────────────────────────────────────────────────────
class DSv4Engine:
    def __init__(
        self,
        model_path: str,
        weights_dir: str,
        expert_root: str,
        libs_dir: str,
        max_context: int = 8192,
        device: str = "cuda",
        rmsnorm_eps: float = 1e-6,
    ):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.rmsnorm_eps = rmsnorm_eps
        self.max_context = max_context

        self.model_path = Path(model_path)
        self.weights_dir = Path(weights_dir)
        self.expert_root = Path(expert_root)
        self.libs_dir = Path(libs_dir)

        # 1. Config
        self.cfg = DSv4Config.from_json(self.model_path / "config.json")
        _log(f"config: {self.cfg.num_hidden_layers}L hidden={self.cfg.hidden_size} "
             f"heads={self.cfg.num_attention_heads} experts={self.cfg.n_routed_experts} "
             f"top_k={self.cfg.num_experts_per_tok} N_inter={self.cfg.moe_intermediate_size}")

        # 2. Tokenizer (only HF dep)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )
        _log(f"tokenizer: vocab={self.tokenizer.vocab_size} eos={self.tokenizer.eos_token_id}")

        # 3. Load .so libs
        self._kvcache = _Lib(self.libs_dir, "libdsv4_kvcache.so").lib
        self._indexer = _Lib(self.libs_dir, "libdsv4_indexer.so").lib
        # V4: libdsv4_mla_v4.so (new ABI, per-layer LoRA weights pushed from Python)
        self._mla = _Lib(self.libs_dir, "libdsv4_mla_v4.so").lib
        self._moe = _Lib(self.libs_dir, "libour_fp4moe.so").lib
        self._dense_lib = _Lib(self.libs_dir, "libour_fp4gemm.so").lib
        self._rmsnorm_lib = _Lib(self.libs_dir, "libdsv4_rmsnorm.so").lib
        for binder, lib in (
            (_bind_kvcache, self._kvcache),
            (_bind_indexer, self._indexer),
            (_bind_mla_v4, self._mla),
            (_bind_moe, self._moe),
            (_bind_dense, self._dense_lib),
            (_bind_rmsnorm, self._rmsnorm_lib),
        ):
            binder(lib)
        _log(f"moe lib version: {self._moe.fp4_moe_version().decode()}")
        _log(f"dense lib version: {self._dense_lib.fp4_gemm_version().decode()}")

        # 4. Init kernel state objects
        self._kv_state = ctypes.c_void_p()
        rc = self._kvcache.kv_cache_fp8_init(
            ctypes.c_char_p(_pack_kvcache_cfg(self.cfg)), ctypes.byref(self._kv_state)
        )
        if rc != 0:
            raise RuntimeError(f"kv_cache_fp8_init failed rc={rc}")
        _log("kv_cache_fp8 ready")

        self._idx_state = ctypes.c_void_p()
        rc = self._indexer.lightning_indexer_init(
            ctypes.c_char_p(_pack_indexer_cfg(self.cfg)), ctypes.byref(self._idx_state)
        )
        if rc != 0:
            raise RuntimeError(f"lightning_indexer_init failed rc={rc}")
        _log("lightning_indexer ready")

        self._mla_state = ctypes.c_void_p()
        rc = self._mla.mla_v4_init(
            ctypes.c_char_p(_pack_mla_v4_cfg(self.cfg)), ctypes.byref(self._mla_state)
        )
        if rc != 0:
            raise RuntimeError(f"mla_v4_init failed rc={rc}")
        _log("mla_v4 ready")

        # 5. MoE weights handle (per-layer; one handle covers all layers via expert_root subdirs).
        # The kernel's create_swiglu opens weights_root/layer_<L>/{shared,routed_*}_w{1,2,3}.bin
        # Resident hot cache cap = ~24 GiB / 256 MiB per slot = 96 slots.
        self._moe_handles: list[ctypes._Pointer] = []
        for L in range(self.cfg.num_hidden_layers):
            layer_dir = self.expert_root / f"layer_{L:02d}"
            if not layer_dir.exists():
                _log(f"WARN expert dir missing for layer {L}: {layer_dir} (will skip MoE on this layer)")
                self._moe_handles.append(None)
                continue
            h = self._moe.fp4_moe_weights_create_swiglu(
                str(layer_dir).encode(),
                self.cfg.n_routed_experts,
                self.cfg.moe_intermediate_size,
                self.cfg.hidden_size,
                int(os.environ.get("DSV4_MOE_CAP", "32")),  # cap reduced from 96 (env DSV4_MOE_CAP override; was hardcoded 8)
                ctypes.c_void_p(0),
            )
            if not h:
                raise RuntimeError(f"fp4_moe_weights_create_swiglu failed for layer {L}")
            self._moe_handles.append(ctypes.c_void_p(h))
        _log(f"moe handles ready: {sum(1 for h in self._moe_handles if h)} / {self.cfg.num_hidden_layers}")

        # F4 hot-set: pin top-K experts per layer to skip NVMe reload misses.
        # Synthetic Zipf distribution is the default (real profiler requires
        # working decode-path; Zipf is a reasonable seed for cold-start).
        # Env DSV4_HOT_SET_ENABLE=0 to disable; DSV4_HOT_SET_JSON to override.
        hot_set_enable = os.environ.get("DSV4_HOT_SET_ENABLE", "1") == "1"
        hot_set_path = os.environ.get(
            "DSV4_HOT_SET_JSON", os.path.join(os.environ.get("DSV4_OUT", "/tmp"), "hot_expert_set.json")
        )
        if hot_set_enable and hasattr(self._moe, "fp4_moe_load_hot_set_from_json_swiglu"):
            from pathlib import Path as _P
            if _P(hot_set_path).is_file():
                _log(f"[F4 hot-set] loading {hot_set_path}")
                t0_hs = time.perf_counter()
                total_pinned = 0
                fails = 0
                for L in range(self.cfg.num_hidden_layers):
                    h = self._moe_handles[L]
                    if h is None:
                        continue
                    rc = self._moe.fp4_moe_load_hot_set_from_json_swiglu(
                        h, hot_set_path.encode(), L,
                    )
                    if rc < 0:
                        fails += 1
                        _log(f"[F4 hot-set] WARN layer={L} rc={rc}")
                    else:
                        total_pinned += rc
                t_hs = time.perf_counter() - t0_hs
                _log(f"[F4 hot-set] pinned {total_pinned} experts across "
                     f"{self.cfg.num_hidden_layers} layers ({fails} fails) in {t_hs:.2f}s")
            else:
                _log(f"[F4 hot-set] disabled: file missing {hot_set_path}")
        else:
            _log("[F4 hot-set] disabled (env or symbol missing)")

        # ── F1.10 SHARED-EXPERT LOADER ─────────────────────────────────────
        # ROOT CAUSE: kernel/v2-moe/our_fp4_moe.cu hardcodes D_shared=nullptr
        # at line 2737 (Phase F TODO never implemented). DSv4 architecture
        # has shared_experts (always-on dense FFN) added at every MoE layer.
        # Skipping this for 43 layers cumulatively kills semantic content
        # → output is gibberish (' ranked colloqu —' instead of coherent text).
        #
        # FIX: dequantize shared_expert_w{1,2,3}.bin per-layer using the CPU
        # NVFP4 reference dequantizer (kernel/v2-moe/sf_interleaved_dequant.py),
        # store BF16 [N, K] device tensors, then in forward() compute
        #     shared_out = (silu(x @ w1.T) * (x @ w3.T)) @ w2.T
        # and add to moe_out before residual addition.
        #
        # Memory: 3 × 2048×4096 × 2 bytes BF16 = 50 MB per layer × 43 layers
        # = 2.16 GB unified GB10 RAM (acceptable; <2% of 128 GB).
        #
        # When verified, migrate this into our_fp4_moe.cu Phase F (kernel-side
        # GEMM3 fused with shared expert reduce — see line 1826 TODO).
        try:
            import sys
            from sf_interleaved_dequant import dequant_expert
            t0_se = time.perf_counter()
            self._shared_w1: list[torch.Tensor] = []  # [N_inter, H] BF16
            self._shared_w2: list[torch.Tensor] = []  # [H, N_inter] BF16
            self._shared_w3: list[torch.Tensor] = []  # [N_inter, H] BF16
            for L in range(self.cfg.num_hidden_layers):
                layer_dir = self.expert_root / f"layer_{L:02d}"
                w1_path = layer_dir / "shared_expert_w1.bin"
                w2_path = layer_dir / "shared_expert_w2.bin"
                w3_path = layer_dir / "shared_expert_w3.bin"
                if not (w1_path.exists() and w2_path.exists() and w3_path.exists()):
                    _log(f"WARN shared-expert files missing for layer {L}: skipping")
                    self._shared_w1.append(None)
                    self._shared_w2.append(None)
                    self._shared_w3.append(None)
                    continue
                w1_np, _ = dequant_expert(str(w1_path))   # [N_inter, H] fp32
                w2_np, _ = dequant_expert(str(w2_path))   # [H, N_inter] fp32
                w3_np, _ = dequant_expert(str(w3_path))   # [N_inter, H] fp32
                self._shared_w1.append(
                    torch.from_numpy(w1_np).to(self.dtype).to(self.device)
                )
                self._shared_w2.append(
                    torch.from_numpy(w2_np).to(self.dtype).to(self.device)
                )
                self._shared_w3.append(
                    torch.from_numpy(w3_np).to(self.dtype).to(self.device)
                )
            t_se = time.perf_counter() - t0_se
            n_loaded = sum(1 for w in self._shared_w1 if w is not None)
            _log(f"[F1.10 shared-expert] loaded {n_loaded}/{self.cfg.num_hidden_layers} "
                 f"layers in {t_se:.2f}s")
        except Exception as e:
            _log(f"[F1.10 shared-expert] FAILED to load: {type(e).__name__}: {e}")
            self._shared_w1 = [None] * self.cfg.num_hidden_layers
            self._shared_w2 = [None] * self.cfg.num_hidden_layers
            self._shared_w3 = [None] * self.cfg.num_hidden_layers


        # 6. Dense backbone (mmap → device tensor dict).
        # prequantize_dsv4.py historically writes dense_backbone.bin alongside the
        # per-layer expert dirs (i.e. inside expert_root), but older builds also
        # placed it next to the HF weights. Probe both.
        #
        # Prefer dense_backbone_nvfp4.bin (v3, 2026-05-03) when present: it
        # contains the 5 V4 MLA matmul (wkv, wq_a, wq_b, wo_a, wo_b) converted
        # FP8 E4M3 block-128x128 → NVFP4 packed (uint8 nibble + uint8 E4M3 SF
        # CUTLASS-swizzled + fp32 inverse global_scale per-tensor). Falls back
        # to the legacy v2 dense_backbone.bin (FP8 attn) if v3 is not present
        # (in which case the V4 MLA kernel push below WILL fail the dtype
        # sanity check at line ~932).
        backbone_path = None
        for cand in (self.weights_dir, self.expert_root, self.model_path):
            v3 = cand / "dense_backbone_nvfp4.bin"
            if v3.exists():
                backbone_path = v3
                _log(f"dense backbone: using v3 NVFP4-attn variant {v3}")
                break
        if backbone_path is None:
            for cand in (self.weights_dir, self.expert_root, self.model_path):
                v2 = cand / "dense_backbone.bin"
                if v2.exists():
                    backbone_path = v2
                    _log(f"dense backbone: falling back to v2 (FP8 attn) {v2}")
                    break
        if backbone_path is None:
            raise FileNotFoundError(
                f"dense_backbone[_nvfp4].bin not found in any of: "
                f"{self.weights_dir}, {self.expert_root}, {self.model_path}"
            )
        dense = load_dense_only(backbone_path, device=device)
        # Keep a reference so close() / re-runs don't free packed NVFP4 tensors
        # whose device pointers we hand to the V4 MLA kernel below.
        self._dense_tensors = dense
        # Embedding + lm_head + final norm — V3 uses embed_tokens/lm_head/norm,
        # V4-Flash uses embed/head/norm at the top level.
        for k in ("embed.weight", "embed_tokens.weight", "model.embed_tokens.weight"):
            t = dense.get(k)
            if t is not None:
                self.embed = t
                break
        else:
            self.embed = None
        for k in ("head.weight", "lm_head.weight"):
            t = dense.get(k)
            if t is not None:
                self.lm_head = t
                break
        else:
            self.lm_head = None
        self.final_norm = None
        for k in ("norm.weight", "model.norm.weight"):
            t = dense.get(k)
            if t is not None:
                self.final_norm = t
                break
        if any(t is None for t in (self.embed, self.lm_head, self.final_norm)):
            missing = [n for n, t in zip(("embed", "lm_head", "final_norm"), (self.embed, self.lm_head, self.final_norm)) if t is None]
            raise RuntimeError(
                f"dense_backbone missing {missing}. "
                f"Top-level keys present: {sorted(k for k in dense if 'layers.' not in k)[:20]}"
            )

        # Per-layer slices
        self.layers: list[LayerWeights] = [
            _slice_layer_weights(dense, L, self.cfg)
            for L in range(self.cfg.num_hidden_layers)
        ]

        # 6b. Push V4 attention weights (17 tensors per layer) to libdsv4_mla_v4.so.
        # Tensor name → ABI slot mapping. After the v3 backbone migration
        # (2026-05-03, --convert-attn-fp8-to-nvfp4), each of the 5 V4 MLA matmul
        # ships as a TRIPLET: NVFP4 packed weight (uint8 nibble pairs) +
        # CUTLASS-swizzled E4M3 SF (uint8) + RedHatAI-inverse global_scale fp32.
        # The kernel inverts the global_scale on use, matching the MoE expert
        # convention (our_fp4_moe.cu line 832).
        #   wkv  (hidden→head_dim)              : attn.wkv.{weight,weight_scale,weight_global_scale}
        #   wq_a (hidden→q_lora_rank)           : attn.wq_a.{weight,weight_scale,weight_global_scale}
        #   q_norm (bf16 [q_lora_rank])         : attn.q_norm.weight  (cast bf16)
        #   wq_b (q_lora_rank→num_heads*head_dim): attn.wq_b.{weight,weight_scale,weight_global_scale}
        #   wo_a (num_heads*v_head_dim→o_lora)  : attn.wo_a.{weight,weight_scale,weight_global_scale}
        #   wo_b (o_lora→hidden)                : attn.wo_b.{weight,weight_scale,weight_global_scale}
        #   attn_sink (bf16 [num_heads])        : attn.attn_sink (cast bf16)
        # We keep BF16-cast copies of q_norm and attn_sink alive in self._mla_v4_aux
        # so the device pointers stay valid for the lifetime of the engine.
        self._mla_v4_aux: list[dict] = []

        def _key(L: int, name: str) -> torch.Tensor | None:
            for prefix in (f"layers.{L}", f"model.layers.{L}"):
                k = f"{prefix}.{name}"
                if k in dense:
                    return dense[k]
            return None

        def _ptr_of(t: torch.Tensor | None) -> int:
            if t is None:
                return 0
            if not t.is_contiguous():
                t = t.contiguous()
            return t.data_ptr()

        for L in range(self.cfg.num_hidden_layers):
            wkv_p   = _key(L, "attn.wkv.weight")
            wkv_s   = _key(L, "attn.wkv.weight_scale")
            wkv_gs  = _key(L, "attn.wkv.weight_global_scale")
            wqa_p   = _key(L, "attn.wq_a.weight")
            wqa_s   = _key(L, "attn.wq_a.weight_scale")
            wqa_gs  = _key(L, "attn.wq_a.weight_global_scale")
            qnorm   = _key(L, "attn.q_norm.weight")
            wqb_p   = _key(L, "attn.wq_b.weight")
            wqb_s   = _key(L, "attn.wq_b.weight_scale")
            wqb_gs  = _key(L, "attn.wq_b.weight_global_scale")
            woa_p   = _key(L, "attn.wo_a.weight")
            woa_s   = _key(L, "attn.wo_a.weight_scale")
            woa_gs  = _key(L, "attn.wo_a.weight_global_scale")
            wob_p   = _key(L, "attn.wo_b.weight")
            wob_s   = _key(L, "attn.wo_b.weight_scale")
            wob_gs  = _key(L, "attn.wo_b.weight_global_scale")
            sink    = _key(L, "attn.attn_sink")

            missing = [n for n, t in (
                ("wkv.weight", wkv_p), ("wkv.weight_scale", wkv_s),
                ("wq_a.weight", wqa_p), ("wq_a.weight_scale", wqa_s),
                ("q_norm.weight", qnorm),
                ("wq_b.weight", wqb_p), ("wq_b.weight_scale", wqb_s),
                ("wo_a.weight", woa_p), ("wo_a.weight_scale", woa_s),
                ("wo_b.weight", wob_p), ("wo_b.weight_scale", wob_s),
                ("attn_sink", sink),
            ) if t is None]
            if missing:
                raise RuntimeError(
                    f"V4 MLA layer {L}: missing tensors in dense_backbone: {missing}"
                )

            # weight_global_scale is REQUIRED by the v3 NVFP4-attn schema. If the
            # backbone is the legacy v2 (FP8 attn) gscales will be None → caller
            # should run prequantize_dsv4.py with --convert-attn-fp8-to-nvfp4.
            missing_gs = [n for n, t in (
                ("wkv.weight_global_scale",  wkv_gs),
                ("wq_a.weight_global_scale", wqa_gs),
                ("wq_b.weight_global_scale", wqb_gs),
                ("wo_a.weight_global_scale", woa_gs),
                ("wo_b.weight_global_scale", wob_gs),
            ) if t is None]
            if missing_gs:
                raise RuntimeError(
                    f"V4 MLA layer {L}: missing weight_global_scale ({missing_gs}). "
                    f"This dense_backbone was produced WITHOUT --convert-attn-fp8-to-nvfp4. "
                    f"Re-run: python3 /tools/prequantize_dsv4.py "
                    f"--convert-attn-fp8-to-nvfp4 --only-attn-nvfp4 "
                    f"--src  --dst -expert"
                )

            # Sanity: packed tensors must be uint8 (NVFP4 nibble-packed); SF must
            # also be uint8 (E4M3 swizzled); gscale must be fp32 scalar.
            for n, t in (("wkv", wkv_p), ("wq_a", wqa_p), ("wq_b", wqb_p),
                          ("wo_a", woa_p), ("wo_b", wob_p)):
                if t.dtype != torch.uint8:
                    raise RuntimeError(
                        f"V4 MLA layer {L}: {n}.weight expected uint8 NVFP4 packed, "
                        f"got {t.dtype} (loader dtype-cast bug?)"
                    )
            for n, t in (("wkv", wkv_s), ("wq_a", wqa_s), ("wq_b", wqb_s),
                          ("wo_a", woa_s), ("wo_b", wob_s)):
                if t.dtype != torch.uint8:
                    raise RuntimeError(
                        f"V4 MLA layer {L}: {n}.weight_scale expected uint8 E4M3 SF, "
                        f"got {t.dtype} (this is the legacy v2 BF16 block-128 scale; "
                        f"re-run prequantize with --convert-attn-fp8-to-nvfp4)"
                    )
            for n, t in (("wkv", wkv_gs), ("wq_a", wqa_gs), ("wq_b", wqb_gs),
                          ("wo_a", woa_gs), ("wo_b", wob_gs)):
                if t.dtype != torch.float32:
                    raise RuntimeError(
                        f"V4 MLA layer {L}: {n}.weight_global_scale expected fp32, "
                        f"got {t.dtype}"
                    )

            # Cast q_norm + attn_sink to BF16 for the kernel; keep refs alive.
            qnorm_bf16 = qnorm.to(torch.bfloat16).contiguous()
            sink_bf16  = sink.to(torch.bfloat16).contiguous()
            self._mla_v4_aux.append({"q_norm": qnorm_bf16, "attn_sink": sink_bf16})

            rc = self._mla.mla_v4_set_layer_weights(
                self._mla_state, ctypes.c_int(L),
                # wkv triplet
                ctypes.c_void_p(_ptr_of(wkv_p)),
                ctypes.c_void_p(_ptr_of(wkv_s)),
                ctypes.c_void_p(_ptr_of(wkv_gs)),
                # wq_a triplet
                ctypes.c_void_p(_ptr_of(wqa_p)),
                ctypes.c_void_p(_ptr_of(wqa_s)),
                ctypes.c_void_p(_ptr_of(wqa_gs)),
                # q_norm bf16
                ctypes.c_void_p(_ptr_of(qnorm_bf16)),
                # wq_b triplet
                ctypes.c_void_p(_ptr_of(wqb_p)),
                ctypes.c_void_p(_ptr_of(wqb_s)),
                ctypes.c_void_p(_ptr_of(wqb_gs)),
                # wo_a triplet
                ctypes.c_void_p(_ptr_of(woa_p)),
                ctypes.c_void_p(_ptr_of(woa_s)),
                ctypes.c_void_p(_ptr_of(woa_gs)),
                # wo_b triplet
                ctypes.c_void_p(_ptr_of(wob_p)),
                ctypes.c_void_p(_ptr_of(wob_s)),
                ctypes.c_void_p(_ptr_of(wob_gs)),
                # attn_sink bf16
                ctypes.c_void_p(_ptr_of(sink_bf16)),
            )
            if rc != 0:
                raise RuntimeError(f"mla_v4_set_layer_weights L={L} rc={rc}")
        _log(f"mla_v4 weights pushed for {self.cfg.num_hidden_layers} layers")

        # 7. Reusable scratch (one big BF16 buffer for residual + post-attn paths)
        self._scratch_size = max_context * self.cfg.hidden_size * 2  # bytes
        self._scratch = torch.empty(
            (max_context, self.cfg.hidden_size), dtype=self.dtype, device=device
        )

        # 8. Workspace for MoE (sized once for max_batch=1, max_seq=128 decode chunk).
        ws_bytes = self._moe.fp4_moe_swiglu_workspace_size(
            128, self.cfg.moe_intermediate_size, self.cfg.hidden_size,
            self.cfg.n_routed_experts, self.cfg.num_experts_per_tok,
        )
        self._moe_ws = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
        _log(f"moe workspace: {ws_bytes/1e9:.2f} GB")

        # 9. Position pointer (grows across forward calls within a generate)
        self._pos = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _ptr(self, t: torch.Tensor) -> int:
        """Return raw device pointer of a contiguous tensor."""
        if not t.is_contiguous():
            t = t.contiguous()
        return t.data_ptr()

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Call libdsv4_rmsnorm.so kernel. x: [N, H] BF16; weight: [H] FP32. Out: BF16 [N,H]."""
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        rc = self._rmsnorm_lib.ds_rmsnorm_forward(
            ctypes.c_void_p(self._ptr(out)),
            ctypes.c_void_p(self._ptr(x)),
            ctypes.c_void_p(self._ptr(weight)),
            ctypes.c_int(n_rows), ctypes.c_int(n_cols),
            ctypes.c_float(self.rmsnorm_eps),
            ctypes.c_void_p(0),
        )
        if rc != 0:
            raise RuntimeError(f"ds_rmsnorm_forward rc={rc}")
        return out

    def _matmul_bf16(self, a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Plain BF16 matmul a @ w.T (HF stores w as [out, in])."""
        return torch.nn.functional.linear(a, w)

    def _route(self, hidden_norm: torch.Tensor, layer: LayerWeights) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute router gate logits → top-k expert ids and (softmax-normalized) weights.

        hidden_norm: [N, H] BF16. layer.router_weight: [n_routed_experts, H].
        Returns:
            topk_ids   [N, top_k] int32
            topk_w     [N, top_k] BF16  (softmax over selected logits — matches
                                         DSv4 router behaviour, Z-loss aux ignored
                                         at inference)
        """
        # F1.9 ROOT-CAUSE FIX (was: plain softmax, missing routed_scaling_factor):
        # DSv4 config requires scoring_func='sqrtsoftplus' + norm + routed_scaling_factor=1.5.
        # The previous softmax+no-scaling produced topk weights ~1/6 instead of ~0.25;
        # cumulative miss-scaling on 43 layers killed semantic content (gibberish output).
        gate_logits = self._matmul_bf16(hidden_norm.to(torch.float32), layer.router_weight.to(torch.float32))
        # scoring_func=sqrtsoftplus
        scores = torch.sqrt(torch.nn.functional.softplus(gate_logits))
        topk_scores, topk_ids = torch.topk(scores, self.cfg.num_experts_per_tok, dim=-1)
        # norm_topk_prob=True
        topk_w = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-20)
        # routed_scaling_factor=1.5 (read from cfg if available, fallback 1.5)
        scale = float(getattr(self.cfg, 'routed_scaling_factor', 1.5))
        topk_w = (topk_w * scale).to(self.dtype)
        return topk_ids.to(torch.int32).contiguous(), topk_w.contiguous()

    # ─────────────────────────────────────────────────────────────────────────
    # forward
    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, input_ids: list[int]) -> torch.Tensor:
        """Single-batch forward. Returns logits [vocab_size] for the LAST token only.

        For prefill, pass the full prompt list and we take the last logit (used for
        first-token sampling). For decode steps, pass [next_tok] (length 1).
        """
        seq_len = len(input_ids)
        if self._pos + seq_len > self.max_context:
            raise RuntimeError(f"context overflow: pos={self._pos} + len={seq_len} > max={self.max_context}")

        # 1. Embed lookup → BF16 hidden [seq_len, hidden_size] on device.
        ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        hidden = torch.nn.functional.embedding(ids, self.embed).to(self.dtype)
        _diag('L_init.embed', hidden)

        # 2. Layer loop.
        for L, lw in enumerate(self.layers):
            residual = hidden
            hidden_norm = self._rmsnorm(hidden, lw.input_layernorm)
            if L < 2: _diag(f'L{L}.hidden_norm', hidden_norm)

            # ── 2a. V4 path: wkv GEMM + KV append happen INSIDE mla_v4_forward.
            # Detect V4 by packed-uint8 weight (NVFP4 dense backbone). Skip the
            # legacy V3 host-side BF16 matmul + Python kv_cache_fp8_append in
            # that case — calling it would (a) crash on shape (uint8 nibble pack
            # is [N, K/2]) and (b) double-write the KV ring.
            if lw.kv_a_proj_with_mqa is not None and lw.kv_a_proj_with_mqa.dtype == torch.uint8:
                pass  # V4 kernel owns wkv + kv_cache_fp8_append
            else:
                # ── 2a. Compute compressed KV (latent + k_rope) for this step ──
                #
                # MLA "down-projection" of hidden into the latent KV space:
                #   kv_a = hidden @ kv_a_proj_with_mqa     # [N, kv_lora_rank + qk_rope_head_dim]
                #   c_kv = layernorm(kv_a[:, :kv_lora_rank])   ← latent (compressed)
                #   k_rope = kv_a[:, kv_lora_rank:]            ← decoupled positional
                kv_a = self._matmul_bf16(hidden_norm, lw.kv_a_proj_with_mqa)
                # V3-MLA: kv_a = [N, kv_lora_rank + qk_rope_head_dim]  (latent + decoupled rope)
                # V4-Flash: kv_a = [N, head_dim==kv_lora_rank]  (single MQA head; rope at attn-time)
                if kv_a.shape[1] == self.cfg.kv_lora_rank:
                    c_kv = kv_a
                    k_rope = torch.zeros(
                        (kv_a.shape[0], self.cfg.qk_rope_head_dim),
                        dtype=self.dtype, device=self.device,
                    )
                else:
                    c_kv = kv_a[:, :self.cfg.kv_lora_rank]
                    k_rope = kv_a[:, self.cfg.kv_lora_rank:]
                c_kv = self._rmsnorm(c_kv.contiguous(), lw.kv_a_layernorm)

                rc = self._kvcache.kv_cache_fp8_append(
                    self._kv_state, ctypes.c_int(L), ctypes.c_int(self._pos),
                    ctypes.c_int(1), ctypes.c_int(seq_len),
                    ctypes.c_void_p(self._ptr(c_kv)),
                    ctypes.c_void_p(self._ptr(k_rope.contiguous())),
                )
                if rc != 0:
                    raise RuntimeError(f"kv_cache_fp8_append L={L} rc={rc}")

            # ── 2b. Lightning Indexer: append + score ──
            # V3-DSv3 has Lightning Indexer (sparse top-k retrieval over compressed blocks).
            # V4-Flash has hc_attn_* (hierarchical cluster attention) instead — different
            # arch component, no indexer weights present in checkpoint. Detect V4 by
            # absence of indexer_q_proj AND absence of compressor: bypass indexer,
            # let MLA do dense attention over all blocks so far.
            # TODO V4 hc_attn: implement hc_attn_base/fn/scale routing for proper sparsity.
            # BLOCKER-7 FIX: bypass indexer whenever q_proj absent — slice fallback
            # would dim-mismatch (indexer_heads*dim > hidden_size). V4 ckpt has
            # neither q_proj nor compressor; if either is missing we cannot
            # actually score, so run dense path safely.
            v4_no_indexer = (lw.indexer_q_proj is None)
            total_blocks = (self._pos + seq_len + self.cfg.compression_factor_m - 1) // self.cfg.compression_factor_m
            if v4_no_indexer:
                # Dense path: all blocks visible.
                top_k_use = total_blocks
                topk_blocks = torch.arange(total_blocks, dtype=torch.int32, device=self.device).unsqueeze(0)
            else:
                rc = self._indexer.lightning_indexer_append(
                    self._idx_state, ctypes.c_void_p(self._ptr(hidden_norm)),
                    ctypes.c_int(L), ctypes.c_int(self._pos),
                    ctypes.c_int(seq_len), ctypes.c_int(1),
                )
                if rc != 0:
                    raise RuntimeError(f"lightning_indexer_append L={L} rc={rc}")

                # Project hidden_norm → indexer query (small linear)
                if lw.indexer_q_proj is not None:
                    idx_query = self._matmul_bf16(hidden_norm, lw.indexer_q_proj)
                    idx_query = idx_query.view(seq_len, self.cfg.indexer_heads, self.cfg.indexer_dim)
                else:
                    # Variant without separate indexer head — slice hidden_norm.
                    idx_query = hidden_norm[:, : self.cfg.indexer_heads * self.cfg.indexer_dim].view(
                        seq_len, self.cfg.indexer_heads, self.cfg.indexer_dim
                    )
                top_k_use = min(self.cfg.indexer_top_k, total_blocks)
                topk_blocks = torch.empty((1, top_k_use), dtype=torch.int32, device=self.device)
                rc = self._indexer.lightning_indexer_score(
                    self._idx_state, ctypes.c_void_p(self._ptr(idx_query.contiguous())),
                    ctypes.c_int(L), ctypes.c_int(1), ctypes.c_int(total_blocks),
                    ctypes.c_void_p(self._ptr(topk_blocks)),
                    ctypes.c_void_p(0),
                )
                if rc != 0:
                    raise RuntimeError(f"lightning_indexer_score L={L} rc={rc}")

            # ── 2c. MLA V4 attention forward ──
            # F1-A4: split decode (seq_len==1) vs prefill (seq_len>1).
            #   DECODE-PATH: no padding, single-token GEMV kernels — pad_seq=1.
            #     Calls mla_v4_forward with is_decode=1; orchestrator dispatches
            #     to v4_*_decode_forward GEMV variants (~16x compute reduction
            #     vs GEMM prefill on M=1).
            #   PREFILL-PATH: NVFP4 GEMM kernels require M %% 128 == 0. Pad hidden_norm
            #     with ZEROS, run kernel, slice attn_out back to real seq_len.
            #     valid_seq_len skips pad rows in KV append + softmax denom.
            #     After ALL layers complete, kv_cache_fp8_trim removes pad slots from KV.
            real_seq = seq_len
            is_decode = (real_seq == 1)
            if is_decode:
                pad_seq = 1
                hidden_padded = hidden_norm
            else:
                pad_seq = ((real_seq + 127) // 128) * 128
                if pad_seq != real_seq:
                    pad_extra = pad_seq - real_seq
                    hidden_padded = torch.cat([
                        hidden_norm,
                        torch.zeros((pad_extra, hidden_norm.shape[1]),
                                    dtype=hidden_norm.dtype, device=hidden_norm.device),
                    ], dim=0)
                else:
                    hidden_padded = hidden_norm
            attn_out_padded = torch.empty_like(hidden_padded)
            # positions = absolute token positions for RoPE; pad with last+i.
            positions = torch.arange(
                self._pos, self._pos + pad_seq, dtype=torch.int32, device=self.device
            )
            rc = self._mla.mla_v4_forward(
                self._mla_state,
                ctypes.c_int(L),
                ctypes.c_void_p(self._ptr(hidden_padded)),
                ctypes.c_int(pad_seq),
                ctypes.c_int(self._pos),                       # past_len
                ctypes.c_void_p(self._kv_state.value or 0),    # kv_cache_state*
                ctypes.c_void_p(self._ptr(positions)),
                ctypes.c_void_p(self._ptr(attn_out_padded)),
                ctypes.c_void_p(0),                            # default stream
                ctypes.c_int(real_seq),                        # valid_seq_len: skip pad in KV + softmax
                ctypes.c_int(1 if is_decode else 0),           # is_decode flag
            )
            if rc != 0:
                raise RuntimeError(f"mla_v4_forward L={L} rc={rc}")
            attn_out = attn_out_padded[:real_seq].contiguous()
            if L < 2: _diag(f'L{L}.attn_out', attn_out)

            # o_proj happens inside the MLA kernel (it has access via layer_idx
            # to the dense weight pointer table seeded at init); attn_out is
            # already projected back to hidden_size.
            hidden = residual + attn_out
            if L < 2: _diag(f'L{L}.post_attn_residual', hidden)

            # ── 2d. MoE block (or dense MLP for first_k_dense_replace layers) ──
            residual2 = hidden
            hidden_norm2 = self._rmsnorm(hidden, lw.post_attention_layernorm)
            if L < 2: _diag(f'L{L}.hidden_norm2', hidden_norm2)

            if L < self.cfg.first_k_dense_replace or lw.router_weight is None or self._moe_handles[L] is None:
                # Dense fallback: straight FFN via dense FP4 GEMM (gate/up/down).
                # For initial bring-up we use BF16 fallback; once dense_backbone
                # ships gate_proj/up_proj/down_proj for these layers we swap to
                # libour_fp4gemm.so.
                _df = self._dense_ffn(hidden_norm2, L)
                if L < 2: _diag(f'L{L}.dense_ffn', _df)
                hidden = residual2 + _df
                continue

            topk_ids, topk_w = self._route(hidden_norm2, lw)
            moe_out = torch.empty_like(hidden_norm2)
            rc = self._moe.fp4_moe_swiglu_run(
                ctypes.c_void_p(self._ptr(hidden_norm2)),
                self._moe_handles[L],
                ctypes.c_void_p(self._ptr(topk_ids)),
                ctypes.c_void_p(self._ptr(topk_w)),
                ctypes.c_void_p(self._ptr(moe_out)),
                ctypes.c_int(seq_len),
                ctypes.c_int(self.cfg.moe_intermediate_size),
                ctypes.c_int(self.cfg.hidden_size),
                ctypes.c_int(self.cfg.num_experts_per_tok),
                ctypes.c_float(1.0),
                ctypes.c_void_p(self._moe_ws.data_ptr()),
                ctypes.c_size_t(self._moe_ws.numel()),
                ctypes.c_void_p(0),
            )
            if rc != 0:
                raise RuntimeError(f"fp4_moe_swiglu_run L={L} rc={rc}")
            if L < 2: _diag(f'L{L}.moe_out', moe_out)

            # ── F1.10 SHARED-EXPERT FORWARD ────────────────────────────────
            # SwiGLU: out = (silu(x @ w1.T) * (x @ w3.T)) @ w2.T
            # Adds shared_expert contribution to MoE output (see init for context).
            sw1 = self._shared_w1[L]
            sw2 = self._shared_w2[L]
            sw3 = self._shared_w3[L]
            if sw1 is not None and sw2 is not None and sw3 is not None:
                x_se = hidden_norm2  # [seq_len, H]
                gate = torch.nn.functional.linear(x_se, sw1)         # [seq_len, N_inter]
                up   = torch.nn.functional.linear(x_se, sw3)         # [seq_len, N_inter]
                # F1.13: apply DSv4 swiglu_limit clamp (config.json: 10.0).
                # gate gets max-only clamp; up gets symmetric clamp.
                _lim = self.cfg.swiglu_limit
                if _lim is not None:
                    gate = torch.clamp(gate, max=_lim)
                    up   = torch.clamp(up, min=-_lim, max=_lim)
                mid  = torch.nn.functional.silu(gate) * up           # [seq_len, N_inter]
                shared_out = torch.nn.functional.linear(mid, sw2)    # [seq_len, H]
                if L < 2: _diag(f'L{L}.shared_out', shared_out)
                moe_out = moe_out + shared_out.to(moe_out.dtype)
                if L < 2: _diag(f'L{L}.moe_plus_shared', moe_out)

            hidden = residual2 + moe_out
            if L < 2: _diag(f'L{L}.post_moe_residual', hidden)

        # 3. Final norm + lm_head projection on the LAST token only.
        _diag('final.hidden_last', hidden[-1:].contiguous())
        last_hidden = self._rmsnorm(hidden[-1:].contiguous(), self.final_norm)
        _diag('final.last_hidden', last_hidden)
        # lm_head: [vocab, hidden] BF16 — plain BF16 matmul, vocab=129280 fits comfortably.
        logits = torch.nn.functional.linear(last_hidden, self.lm_head).squeeze(0)
        _diag('final.logits', logits)

        # F1-A4: KV cache trim API kept for safety, but mla_v4_forward already
        # appends only `valid_seq_len` real tokens (pad rows are skipped at append
        # time inside the orchestrator). So no trim is needed when valid_seq_len
        # is honoured by the V4 path. Trim remains available for callers that
        # blindly append pad_seq tokens. Left as no-op here — see kv_cache_fp8.cu
        # for the API spec.

        # Advance position pointer for next call.
        self._pos += seq_len
        return logits

    def _dense_ffn(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """BF16 fallback FFN for the first_k_dense_replace dense layers.

        Once we have FP4-quantized gate/up/down for these layers, replace with
        libour_fp4gemm.so call (3 GEMMs + SiLU * gate elementwise).
        """
        # NOTE: first_k_dense_replace dense weights are not exposed yet by the
        # dense_backbone slicer; this stub returns zero so the residual is just
        # passed through. A no-op IS correct in the absence of an MLP block but
        # will degrade quality on the first 3 layers — must be implemented
        # before claiming bit-exact. Tracked as a TODO in DESIGN_SYSTEM.md §8.
        return torch.zeros_like(x)

    # ─────────────────────────────────────────────────────────────────────────
    # generate
    # ─────────────────────────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        seed: int = -1,
        echo: bool = False,
    ) -> str:
        """Greedy/sampled autoregressive decode. Returns generated text only
        (or prompt+generated if echo=True)."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        _log(f"prompt: {len(prompt_ids)} tokens")

        # Reset KV cache for fresh request.
        self._kvcache.kv_cache_fp8_drop(self._kv_state, ctypes.c_int(-1))
        self._indexer.lightning_indexer_free_cache(self._idx_state, ctypes.c_int(-1))
        self._pos = 0

        # Prefill — feed the entire prompt in one forward (kernels handle seq_len > 1).
        t0 = time.time()
        logits = self.forward(prompt_ids)
        prefill_ms = (time.time() - t0) * 1000.0
        _log(f"prefill: {prefill_ms:.1f}ms ({len(prompt_ids)/prefill_ms*1000:.1f} tok/s)")

        generated: list[int] = []
        eos_id = self.tokenizer.eos_token_id

        # Decode loop
        t_decode_start = time.time()
        for step in range(max_new_tokens):
            next_id = dsv4_sampler.sample(logits, temperature, top_p, top_k, seed=seed if seed < 0 else seed + step)
            if next_id == eos_id:
                _log(f"EOS at step {step}")
                break
            generated.append(next_id)
            logits = self.forward([next_id])
        decode_s = max(time.time() - t_decode_start, 1e-9)
        _log(f"decode: {len(generated)} tok in {decode_s*1000:.1f}ms ({len(generated)/decode_s:.2f} tok/s)")

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if echo:
            text = prompt + text
        return text

    # ─────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────────────────
    def close(self) -> None:
        for h in self._moe_handles:
            if h is not None:
                self._moe.fp4_moe_weights_destroy(h)
        if self._mla_state:
            self._mla.mla_v4_free(self._mla_state)
        if self._idx_state:
            self._indexer.lightning_indexer_free(self._idx_state)
        if self._kv_state:
            self._kvcache.kv_cache_fp8_free(self._kv_state)
        _log("closed")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _cli() -> int:
    p = argparse.ArgumentParser(description="DSv4-Flash native inference")
    p.add_argument("--model-path", default=os.environ.get("DSV4_WEIGHTS", ""),
                   help="HF model dir (config.json + tokenizer files)")
    p.add_argument("--weights-dir", default=os.environ.get("DSV4_WEIGHTS", ""),
                   help="(unused for MXFP4 path) Output of prequantize_dsv4.py (dense_backbone.bin lives here)")
    p.add_argument("--expert-root", default=os.environ.get("DSV4_WEIGHTS", ""),
                   help="(unused for MXFP4 path) Per-layer expert dirs root (layer_NN/...)")
    p.add_argument("--libs-dir", default=os.path.join(os.environ.get("DSV4_KERNEL_DEPS", ""), "kernel"),
                   help="Where the .so files live")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--echo", action="store_true")
    args = p.parse_args()

    eng = DSv4Engine(
        model_path=args.model_path,
        weights_dir=args.weights_dir,
        expert_root=args.expert_root,
        libs_dir=args.libs_dir,
        max_context=args.max_context,
    )
    try:
        out = eng.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            echo=args.echo,
        )
        print(out)
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
