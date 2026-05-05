"""fp8_e4m3_gemv_binding.py — torch-friendly ctypes wrapper for libfp8_e4m3_gemv.so. — DSv4-Flash native FP8 E4M3 dense GEMV M=1.

Public API:
    fp8_e4m3_gemv(W_fp8, W_scale, x) -> y

All tensors must be CUDA. W_fp8 (N, K) uint8 (raw E4M3 storage),
W_scale (N/128, K/128) uint8 (E8M0), x (K,) bfloat16.
Returns y (N,) bfloat16.

The 128x128 2-D block-quant layout was verified empirically against the
published HF DeepSeek-V4-Flash safetensors snapshot (see header of
fp8_e4m3_gemv.cu and the test under tools/test_fp8_e4m3_gemv.py).
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional

import torch


# Hard-coded layout constant: DeepSeek-V4 attention + shared-expert tensors
# all use a 128x128 block-quant tile with one E8M0 byte per tile.
FP8_BLOCK_TILE = 128


_LIB_PATH = os.environ.get(
    "FP8_E4M3_GEMV_LIB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "kernel", "fp8-dense", "libfp8_e4m3_gemv.so"),
)
_LIB_PATH = os.path.normpath(_LIB_PATH)


class _Lib:
    _instance: Optional["_Lib"] = None

    def __init__(self):
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(
                f"libfp8_e4m3_gemv.so not found at {_LIB_PATH}; "
                f"run `make` in kernel/fp8-dense/ first."
            )
        self.lib = ctypes.CDLL(_LIB_PATH)
        # int fp8_e4m3_gemv_m1(
        #     const __nv_bfloat16* x,
        #     const uint8_t* W_fp8,
        #     const uint8_t* W_scale,
        #     int K,
        #     int N,
        #     int64_t weight_row_stride_bytes,
        #     int64_t scale_row_stride_bytes,
        #     __nv_bfloat16* y,
        #     cudaStream_t stream
        # )
        self.lib.fp8_e4m3_gemv_m1.argtypes = [
            ctypes.c_void_p,          # x
            ctypes.c_void_p,          # W_fp8
            ctypes.c_void_p,          # W_scale
            ctypes.c_int,             # K
            ctypes.c_int,             # N
            ctypes.c_int64,           # weight_row_stride_bytes
            ctypes.c_int64,           # scale_row_stride_bytes
            ctypes.c_void_p,          # y
            ctypes.c_void_p,          # stream
        ]
        self.lib.fp8_e4m3_gemv_m1.restype = ctypes.c_int

    @classmethod
    def get(cls) -> "_Lib":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def fp8_e4m3_gemv(
    W_fp8: torch.Tensor,
    W_scale: torch.Tensor,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """FP8 E4M3 GEMV M=1: y = x @ W.T (i.e. y[n] = sum_k W[n,k] * x[k]).

    Args:
        W_fp8:    cuda uint8 [N, K] — raw E4M3 storage (1 byte / elem).
        W_scale:  cuda uint8 [N/128, K/128] — one E8M0 byte per 128x128 tile.
        x:        cuda bfloat16 [K] — activation vector.
        out:      optional cuda bfloat16 [N] — destination buffer.
        stream:   optional torch.cuda.Stream — defaults to current stream.

    Returns:
        cuda bfloat16 [N] — y.

    Raises:
        RuntimeError on shape/dtype/device mismatch or kernel launch failure.
    """
    # ── Validation ────────────────────────────────────────────────────────
    if not (W_fp8.is_cuda and W_scale.is_cuda and x.is_cuda):
        raise RuntimeError("all tensors must be on CUDA device")
    if W_fp8.dtype != torch.uint8:
        raise RuntimeError(f"W_fp8 must be uint8 (raw E4M3), got {W_fp8.dtype}")
    if W_scale.dtype != torch.uint8:
        raise RuntimeError(f"W_scale must be uint8 (raw E8M0), got {W_scale.dtype}")
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"x must be bfloat16, got {x.dtype}")
    if x.dim() != 1:
        raise RuntimeError(f"x must be 1-D [K], got shape {tuple(x.shape)}")
    if W_fp8.dim() != 2:
        raise RuntimeError(f"W_fp8 must be 2-D [N, K], got shape {tuple(W_fp8.shape)}")
    if W_scale.dim() != 2:
        raise RuntimeError(f"W_scale must be 2-D [N/128, K/128], got shape {tuple(W_scale.shape)}")

    K = x.shape[0]
    N = W_fp8.shape[0]
    if W_fp8.shape[1] != K:
        raise RuntimeError(
            f"W_fp8 shape {tuple(W_fp8.shape)} inconsistent with K={K} "
            f"(expected [_, {K}])"
        )
    expected_scale = (N // FP8_BLOCK_TILE, K // FP8_BLOCK_TILE)
    if W_scale.shape != expected_scale:
        raise RuntimeError(
            f"W_scale shape {tuple(W_scale.shape)} != {expected_scale} "
            f"(expected one E8M0 byte per 128x128 tile)"
        )
    if K % FP8_BLOCK_TILE != 0:
        raise RuntimeError(f"K={K} must be multiple of {FP8_BLOCK_TILE}")
    if N % FP8_BLOCK_TILE != 0:
        raise RuntimeError(f"N={N} must be multiple of {FP8_BLOCK_TILE}")

    # ── Ensure contiguous ────────────────────────────────────────────────
    W_fp8 = W_fp8.contiguous()
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

    ret = lib.lib.fp8_e4m3_gemv_m1(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(W_fp8.data_ptr()),
        ctypes.c_void_p(W_scale.data_ptr()),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int64(W_fp8.stride(0)),         # bytes per row (uint8 stride==bytes)
        ctypes.c_int64(W_scale.stride(0)),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(stream_ptr),
    )
    if ret != 0:
        raise RuntimeError(f"fp8_e4m3_gemv_m1 returned {ret} (kernel launch failed)")
    return out


__all__ = ["fp8_e4m3_gemv", "FP8_BLOCK_TILE"]
