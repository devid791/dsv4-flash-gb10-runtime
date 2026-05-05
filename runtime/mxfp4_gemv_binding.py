"""mxfp4_gemv_binding.py — torch-friendly ctypes wrapper for libmxfp4_gemv.so. — DSv4-Flash MXFP4 native dense GEMV M=1.

Public API:
    mxfp4_gemv(W_packed, W_scale, x) -> y

All tensors must be CUDA. W_packed (N, K/2) uint8, W_scale (N, K/32) uint8,
x (K,) bfloat16. Returns y (N,) bfloat16.
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional

import torch


_LIB_PATH = os.environ.get(
    "MXFP4_GEMV_LIB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "kernel", "mxfp4-dense", "libmxfp4_gemv.so"),
)
_LIB_PATH = os.path.normpath(_LIB_PATH)


class _Lib:
    _instance: Optional["_Lib"] = None

    def __init__(self):
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(
                f"libmxfp4_gemv.so not found at {_LIB_PATH}; "
                f"run `make` in kernel/mxfp4-dense/ first."
            )
        self.lib = ctypes.CDLL(_LIB_PATH)
        # int mxfp4_gemv_m1(
        #     const __nv_bfloat16* x,
        #     const uint8_t* W_packed,
        #     const uint8_t* W_scale,
        #     int K,
        #     int N,
        #     int64_t packed_row_stride_bytes,
        #     int64_t scale_row_stride_bytes,
        #     __nv_bfloat16* y,
        #     cudaStream_t stream
        # )
        self.lib.mxfp4_gemv_m1.argtypes = [
            ctypes.c_void_p,          # x
            ctypes.c_void_p,          # W_packed
            ctypes.c_void_p,          # W_scale
            ctypes.c_int,             # K
            ctypes.c_int,             # N
            ctypes.c_int64,           # packed_row_stride_bytes
            ctypes.c_int64,           # scale_row_stride_bytes
            ctypes.c_void_p,          # y
            ctypes.c_void_p,          # stream
        ]
        self.lib.mxfp4_gemv_m1.restype = ctypes.c_int

    @classmethod
    def get(cls) -> "_Lib":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def mxfp4_gemv(
    W_packed: torch.Tensor,
    W_scale: torch.Tensor,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """MXFP4 GEMV M=1: y = x @ W.T  (i.e. y[n] = sum_k W[n,k] * x[k]).

    Args:
        W_packed: cuda uint8 [N, K/2] — packed FP4 nibbles, low=elem 2i, high=elem 2i+1.
        W_scale:  cuda uint8 [N, K/32] — E8M0 byte per 32-elem block.
        x:        cuda bfloat16 [K] — activation vector.
        out:      optional cuda bfloat16 [N] — destination buffer.
        stream:   optional torch.cuda.Stream — defaults to current stream.

    Returns:
        cuda bfloat16 [N] — y.

    Raises:
        RuntimeError on shape/dtype/device mismatch or kernel launch failure.
    """
    # ── Validation ────────────────────────────────────────────────────────
    if not (W_packed.is_cuda and W_scale.is_cuda and x.is_cuda):
        raise RuntimeError("all tensors must be on CUDA device")
    if W_packed.dtype != torch.uint8:
        raise RuntimeError(f"W_packed must be uint8, got {W_packed.dtype}")
    if W_scale.dtype != torch.uint8:
        raise RuntimeError(f"W_scale must be uint8, got {W_scale.dtype}")
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"x must be bfloat16, got {x.dtype}")
    if x.dim() != 1:
        raise RuntimeError(f"x must be 1-D [K], got shape {tuple(x.shape)}")
    if W_packed.dim() != 2:
        raise RuntimeError(f"W_packed must be 2-D [N, K/2], got shape {tuple(W_packed.shape)}")
    if W_scale.dim() != 2:
        raise RuntimeError(f"W_scale must be 2-D [N, K/32], got shape {tuple(W_scale.shape)}")

    K = x.shape[0]
    N = W_packed.shape[0]
    if W_packed.shape[1] != K // 2:
        raise RuntimeError(
            f"W_packed shape {tuple(W_packed.shape)} inconsistent with K={K} "
            f"(expected [_, {K//2}])"
        )
    if W_scale.shape != (N, K // 32):
        raise RuntimeError(
            f"W_scale shape {tuple(W_scale.shape)} != ({N}, {K//32})"
        )
    if K % 32 != 0:
        raise RuntimeError(f"K={K} must be multiple of 32")

    # ── Ensure contiguous ────────────────────────────────────────────────
    W_packed = W_packed.contiguous()
    W_scale = W_scale.contiguous()
    x = x.contiguous()

    # ── Output buffer ────────────────────────────────────────────────────
    if out is None:
        out = torch.empty(N, dtype=torch.bfloat16, device=x.device)
    else:
        if out.shape != (N,) or out.dtype != torch.bfloat16 or not out.is_cuda:
            raise RuntimeError(
                f"out must be cuda bfloat16 [{N}], got shape {tuple(out.shape)} "
                f"dtype {out.dtype} device {out.device}"
            )
        out = out.contiguous()

    # ── Launch ───────────────────────────────────────────────────────────
    lib = _Lib.get()
    stream_ptr = (stream.cuda_stream
                  if stream is not None
                  else torch.cuda.current_stream(x.device).cuda_stream)

    ret = lib.lib.mxfp4_gemv_m1(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(W_packed.data_ptr()),
        ctypes.c_void_p(W_scale.data_ptr()),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int64(W_packed.stride(0)),     # bytes per row (uint8 stride==bytes)
        ctypes.c_int64(W_scale.stride(0)),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(stream_ptr),
    )
    if ret != 0:
        raise RuntimeError(f"mxfp4_gemv_m1 returned {ret} (kernel launch failed)")
    return out


__all__ = ["mxfp4_gemv"]
