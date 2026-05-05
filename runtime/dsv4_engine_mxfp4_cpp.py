"""dsv4_engine_mxfp4_cpp.py — D2 SCAFFOLDING Python ctypes wrapper.

Davide Zenati, 2026-05-04 .

This is the THIN ctypes wrapper around `libdsv4_engine_mxfp4.so`. It wires the
D2 C ABI to a clean Python class `DSv4EngineMxfp4Cpp`. The class:

  * On __init__, dlopen()s the .so and calls `dsv4_mxfp4_init_engine` to
    create the engine (CUDA stream + cuBLAS + workspace + KV cache + hot-set
    bank + RoPE freqs precomputed + 4 kernel .so dlopen+dlsym).
  * Loads the R2 Python `DSv4EngineMXFP4` to get all GPU-resident weight
    pointers, then forwards them to C++ via `dsv4_mxfp4_set_top` /
    `dsv4_mxfp4_set_layer` / `dsv4_mxfp4_set_fp8_strides`.
  * Exposes `decode_step(token_id) -> int` calling the C ABI stub (D9 will
    replace with real forward).
  * On __del__, calls `dsv4_mxfp4_free_engine`.

CONSTRAINT: This wrapper does NOT touch `runtime/dsv4_engine_mxfp4.py` (the R2
reference is intoccabile). It only consumes its loaded weight tensors.

PRECONDITION: Kernel .so files must already be built:
  kernel/fp8-dense/libfp8_e4m3_gemv.so
  kernel/mxfp4-dense/libmxfp4_gemv.so
  kernel/mxfp4-routed/libmxfp4_grouped_gemv.so
  kernel/rmsnorm-fuse/librmsnorm_fuse.so
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# .so path resolution
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_KERNEL_DIR = _REPO_ROOT / "kernel"

LIB_PATH        = os.environ.get("DSV4_MXFP4_CPP_LIB",
                                 str(_KERNEL_DIR / "cpp-engine-mxfp4" / "libdsv4_engine_mxfp4.so"))
KERNEL_FP8      = os.environ.get("DSV4_KERNEL_FP8_DENSE",
                                 str(_KERNEL_DIR / "fp8-dense" / "libfp8_e4m3_gemv.so"))
KERNEL_MXFP4_D  = os.environ.get("DSV4_KERNEL_MXFP4_DENSE",
                                 str(_KERNEL_DIR / "mxfp4-dense" / "libmxfp4_gemv.so"))
KERNEL_MXFP4_R  = os.environ.get("DSV4_KERNEL_MXFP4_ROUTED",
                                 str(_KERNEL_DIR / "mxfp4-routed" / "libmxfp4_grouped_gemv.so"))
KERNEL_RMSNORM  = os.environ.get("DSV4_KERNEL_RMSNORM",
                                 str(_KERNEL_DIR / "rmsnorm-fuse" / "librmsnorm_fuse.so"))


def _log(msg: str) -> None:
    sys.stdout.write(f"[mxfp4-cpp-d2] {msg}\n")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# ctypes wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _load_lib() -> ctypes.CDLL:
    if not os.path.exists(LIB_PATH):
        raise FileNotFoundError(
            f"libdsv4_engine_mxfp4.so not found at {LIB_PATH}. "
            f"Run `make` in kernel/cpp-engine-mxfp4/ first.")
    lib = ctypes.CDLL(LIB_PATH, mode=ctypes.RTLD_GLOBAL)

    # init / free
    lib.dsv4_mxfp4_init_engine.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]
    lib.dsv4_mxfp4_init_engine.restype = ctypes.c_void_p

    lib.dsv4_mxfp4_free_engine.argtypes = [ctypes.c_void_p]
    lib.dsv4_mxfp4_free_engine.restype = None

    # KV cache + hotset
    lib.dsv4_mxfp4_alloc_kv_cache.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.dsv4_mxfp4_alloc_kv_cache.restype = ctypes.c_int
    lib.dsv4_mxfp4_reset_kv.argtypes = [ctypes.c_void_p]
    lib.dsv4_mxfp4_reset_kv.restype = None
    lib.dsv4_mxfp4_alloc_hotset_banks.argtypes = [ctypes.c_void_p]
    lib.dsv4_mxfp4_alloc_hotset_banks.restype = ctypes.c_int

    # Setters
    lib.dsv4_mxfp4_set_top.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.dsv4_mxfp4_set_top.restype = None

    # set_layer takes 28 void* + L (5 norms+sink + 10 fp8 attn + 1 router + 6 fp8 shared + 6 hc)
    # FIX D11: was 26 — last 2 (hc_ffn_base, hc_ffn_scale) got promoted to c_int and truncated to 32-bit
    set_layer_args = [ctypes.c_void_p, ctypes.c_int]
    set_layer_args.extend([ctypes.c_void_p] * 28)
    lib.dsv4_mxfp4_set_layer.argtypes = set_layer_args
    lib.dsv4_mxfp4_set_layer.restype = None

    lib.dsv4_mxfp4_set_fp8_strides.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
    ]
    lib.dsv4_mxfp4_set_fp8_strides.restype = None

    # Decode + introspection
    lib.dsv4_mxfp4_decode_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.dsv4_mxfp4_decode_step.restype = ctypes.c_int

    lib.dsv4_mxfp4_state_struct_size.argtypes = []
    lib.dsv4_mxfp4_state_struct_size.restype = ctypes.c_size_t
    lib.dsv4_mxfp4_layer_struct_size.argtypes = []
    lib.dsv4_mxfp4_layer_struct_size.restype = ctypes.c_size_t
    lib.dsv4_mxfp4_n_layers.argtypes = [ctypes.c_void_p]
    lib.dsv4_mxfp4_n_layers.restype = ctypes.c_int
    lib.dsv4_mxfp4_max_seq.argtypes = [ctypes.c_void_p]
    lib.dsv4_mxfp4_max_seq.restype = ctypes.c_int

    return lib


class DSv4EngineMxfp4Cpp:
    """C++ MXFP4 engine wrapper (D2 scaffolding).

    Usage:
        engine = DSv4EngineMxfp4Cpp(weights_path=os.environ.get("DSV4_WEIGHTS", ""),
                                    n_layers=43, max_seq=1024)
        engine.wire_weights(r2_engine)   # r2_engine is a DSv4EngineMXFP4 instance
        next_tok = engine.decode_step(token_id)
        del engine
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        n_layers: int = 43,
        max_seq: int = 1024,
    ):
        self.lib = _load_lib()
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.weights_path = weights_path or os.environ.get(
            "DSV4_HF_PATH", os.environ.get("DSV4_WEIGHTS", "")
        )

        _log(f"struct sizes: state={self.lib.dsv4_mxfp4_state_struct_size()} "
             f"layer={self.lib.dsv4_mxfp4_layer_struct_size()}")

        t0 = time.time()
        _log(f"calling init_engine(n_layers={n_layers}, max_seq={max_seq}) ...")
        _log(f"  fp8_dense    = {KERNEL_FP8}")
        _log(f"  mxfp4_dense  = {KERNEL_MXFP4_D}")
        _log(f"  mxfp4_routed = {KERNEL_MXFP4_R}")
        _log(f"  rmsnorm      = {KERNEL_RMSNORM}")

        self._state = self.lib.dsv4_mxfp4_init_engine(
            n_layers, max_seq,
            KERNEL_FP8.encode("utf-8"),
            KERNEL_MXFP4_D.encode("utf-8"),
            KERNEL_MXFP4_R.encode("utf-8"),
            KERNEL_RMSNORM.encode("utf-8"),
        )
        if not self._state:
            raise RuntimeError("dsv4_mxfp4_init_engine returned NULL")

        self._init_time = time.time() - t0
        _log(f"init_engine OK ({self._init_time:.2f}s)")

        # Query GPU memory after init
        try:
            import torch
            torch.cuda.synchronize()
            self._init_mem_gb = torch.cuda.memory_allocated() / 1e9
        except Exception:
            self._init_mem_gb = 0.0

        _log(f"GPU mem after init (workspace+KV+hotset+RoPE): {self._init_mem_gb:.2f} GB")

    # ----- Weight wiring (delegates to R2 engine for the actual safetensors load) -----

    def wire_weights(self, r2_engine) -> None:
        """Wire all weight pointers from a loaded DSv4EngineMXFP4 (R2)
        into the C++ State struct. Zero-copy: just pointer registration."""
        _log("wiring top-level weights from R2 engine ...")
        # Top-level (embed/head/final_norm + hc_head_*)
        self.lib.dsv4_mxfp4_set_top(
            self._state,
            r2_engine.embed.data_ptr(),
            r2_engine.head.data_ptr(),
            r2_engine.final_norm.data_ptr(),
            r2_engine.hc_head_fn.data_ptr(),
            r2_engine.hc_head_base.data_ptr(),
            r2_engine.hc_head_scale.data_ptr(),
        )

        _log(f"wiring {len(r2_engine.layers)} layer weights ...")
        for L, ly in enumerate(r2_engine.layers):
            self.lib.dsv4_mxfp4_set_layer(
                self._state, L,
                ly.attn_norm.data_ptr(), ly.ffn_norm.data_ptr(),
                ly.q_norm.data_ptr(),    ly.kv_norm.data_ptr(),
                ly.attn_sink.data_ptr(),
                ly.wkv_w.data_ptr(),  ly.wkv_s.data_ptr(),
                ly.wq_a_w.data_ptr(), ly.wq_a_s.data_ptr(),
                ly.wq_b_w.data_ptr(), ly.wq_b_s.data_ptr(),
                ly.wo_a_w.data_ptr(), ly.wo_a_s.data_ptr(),
                ly.wo_b_w.data_ptr(), ly.wo_b_s.data_ptr(),
                ly.router_gate.data_ptr(),
                ly.sh_w1_w.data_ptr(), ly.sh_w1_s.data_ptr(),
                ly.sh_w2_w.data_ptr(), ly.sh_w2_s.data_ptr(),
                ly.sh_w3_w.data_ptr(), ly.sh_w3_s.data_ptr(),
                ly.hc_attn_fn.data_ptr(), ly.hc_attn_base.data_ptr(), ly.hc_attn_scale.data_ptr(),
                ly.hc_ffn_fn.data_ptr(),  ly.hc_ffn_base.data_ptr(),  ly.hc_ffn_scale.data_ptr(),
            )

            # Strides + out_dims for FP8 GEMV dispatch (row-major contiguous).
            # FP8: 1 byte per element. Scale: E8M0 1 byte per 128x128 tile.
            def _wstride(t): return int(t.stride(0)) if t.dim() > 1 else int(t.shape[-1])
            def _odim(t): return int(t.shape[0]) if t.dim() > 1 else int(t.shape[0])

            self.lib.dsv4_mxfp4_set_fp8_strides(
                self._state, L,
                _odim(ly.wkv_w),  _wstride(ly.wkv_w),  _wstride(ly.wkv_s),
                _odim(ly.wq_a_w), _wstride(ly.wq_a_w), _wstride(ly.wq_a_s),
                _odim(ly.wq_b_w), _wstride(ly.wq_b_w), _wstride(ly.wq_b_s),
                _odim(ly.wo_a_w), _wstride(ly.wo_a_w), _wstride(ly.wo_a_s),
                _odim(ly.wo_b_w), _wstride(ly.wo_b_w), _wstride(ly.wo_b_s),
                _wstride(ly.sh_w1_w), _wstride(ly.sh_w1_s),
                _wstride(ly.sh_w2_w), _wstride(ly.sh_w2_s),
                _wstride(ly.sh_w3_w), _wstride(ly.sh_w3_s),
            )

        _log("all weights wired")

    # ----- Decode -----

    def decode_step(self, token_id: int) -> int:
        """Run ONE decode step. D2 returns 0 (stub); D9 will replace with real."""
        return self.lib.dsv4_mxfp4_decode_step(self._state, int(token_id))

    def reset_kv(self) -> None:
        self.lib.dsv4_mxfp4_reset_kv(self._state)

    # ----- Introspection -----

    @property
    def state(self) -> int:
        """Raw C void* (for debugging / advanced wiring)."""
        return self._state

    @property
    def init_time_seconds(self) -> float:
        return self._init_time

    @property
    def init_mem_gb(self) -> float:
        return self._init_mem_gb

    def __repr__(self) -> str:
        return (f"<DSv4EngineMxfp4Cpp state=0x{self._state:x} "
                f"n_layers={self.n_layers} max_seq={self.max_seq} "
                f"init_time={self._init_time:.2f}s mem={self._init_mem_gb:.2f}GB>")

    def __del__(self):
        try:
            if hasattr(self, "_state") and self._state:
                self.lib.dsv4_mxfp4_free_engine(self._state)
                self._state = None
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test entry (python -m runtime.dsv4_engine_mxfp4_cpp)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-layers", type=int, default=43)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument("--no-wire", action="store_true",
                   help="skip wiring weights (init-only smoke)")
    args = p.parse_args()

    eng = DSv4EngineMxfp4Cpp(n_layers=args.n_layers, max_seq=args.max_seq)
    print(repr(eng))

    if not args.no_wire:
        # Try to wire from R2 engine if available.
        try:
            from dsv4_engine_mxfp4 import DSv4EngineMXFP4
            r2 = DSv4EngineMXFP4(max_layers=args.n_layers)
            eng.wire_weights(r2)
        except Exception as e:
            print(f"[wire] skipped (R2 load failed: {type(e).__name__}: {e})")

    # Stub decode
    out = eng.decode_step(1)
    print(f"decode_step(1) -> {out}")
    del eng
    print("smoke OK")


# ─── D9: decode_step + generate convenience APIs ──────────────────────────
# Adds high-level decode wrappers around D2's ctypes ABI for D9 + D10 smoke.

def _ensure_d9_bindings(lib):
    """Bind D9 ABI extensions if not already bound."""
    import ctypes as _c
    # decode_step_real (the wire-up real impl, exported in addition to the stub)
    if not hasattr(lib, "_d9_bound"):
        if hasattr(lib, "dsv4_mxfp4_decode_step"):
            lib.dsv4_mxfp4_decode_step.restype = _c.c_int
            lib.dsv4_mxfp4_decode_step.argtypes = [_c.c_void_p, _c.c_int]
        lib._d9_bound = True
    return lib


def decode_step(engine, token_id):
    """Run one decode step on the C++ engine.

    Args:
        engine: DSv4EngineMxfp4Cpp instance (or raw ctypes lib + state pair)
        token_id: input token id (int)

    Returns:
        next_token (int) or negative on error.
    """
    if hasattr(engine, "lib") and hasattr(engine, "state"):
        lib, state = engine.lib, engine.state
    else:
        lib, state = engine
    _ensure_d9_bindings(lib)
    return int(lib.dsv4_mxfp4_decode_step(state, int(token_id)))


def generate(engine, prompt_ids, max_new_tokens=10):
    """Greedy generate `max_new_tokens` tokens from a prompt.

    Returns list[int] of generated token ids (does not include prompt).
    """
    out = []
    # Feed prompt (each token consumes one decode_step; final token's output
    # logits is returned as the next token).
    last = prompt_ids[0] if prompt_ids else 0
    for tok in prompt_ids:
        nxt = decode_step(engine, tok)
        if nxt < 0:
            return out  # abort
        last = nxt
    # Then continue generating
    out.append(last)
    for _ in range(max_new_tokens - 1):
        nxt = decode_step(engine, last)
        if nxt < 0:
            return out
        out.append(nxt)
        last = nxt
    return out


__all__ = list(globals().get("__all__", [])) + ["decode_step", "generate"]
