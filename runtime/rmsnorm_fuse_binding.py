"""ctypes binding for librmsnorm_fuse.so.

Exposes two ops over PyTorch BF16 CUDA tensors:

    rmsnorm_fuse_fwd(x, weight, eps=1e-6, out=None) -> Tensor
    rmsnorm_fuse_residual_add(x, residual, weight, eps=1e-6,
                              out_norm=None, out_new_residual=None)
        -> (out_norm, out_new_residual)

Both functions launch on the current CUDA stream. Inputs must be:
  - dtype  : torch.bfloat16
  - device : cuda
  - x, residual, out*: shape (N, H) — N rows of size H (contiguous last dim)
  - weight : shape (H,)  contiguous
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

_LIB: Optional[ctypes.CDLL] = None


def _resolve_lib_path() -> str:
    env = os.environ.get("RMSNORM_FUSE_LIB")
    if env:
        return env
    here = Path(__file__).resolve().parent
    cand = [
        here.parent / "kernel" / "rmsnorm-fuse" / "librmsnorm_fuse.so",
        ]
    for p in cand:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"librmsnorm_fuse.so not found in {cand}. Set $RMSNORM_FUSE_LIB.")


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(_resolve_lib_path())

    # int rmsnorm_fuse_fwd(const void* x, const void* weight, void* y,
    #                     int n_rows, int hidden, float eps,
    #                     cudaStream_t stream)
    lib.rmsnorm_fuse_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
    ]
    lib.rmsnorm_fuse_fwd.restype = ctypes.c_int

    lib.rmsnorm_fuse_residual_add.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
    ]
    lib.rmsnorm_fuse_residual_add.restype = ctypes.c_int

    _LIB = lib
    return lib


def _check_bf16_cuda(name: str, t: torch.Tensor) -> None:
    if t.dtype != torch.bfloat16:
        raise TypeError(f"{name}: expected bfloat16, got {t.dtype}")
    if not t.is_cuda:
        raise TypeError(f"{name}: expected CUDA tensor, got {t.device}")
    if not t.is_contiguous():
        raise ValueError(f"{name}: must be contiguous")


def rmsnorm_fuse_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """y[n, h] = weight[h] * x[n, h] / sqrt(mean(x[n, :]^2) + eps)."""
    _check_bf16_cuda("x", x)
    _check_bf16_cuda("weight", weight)
    if x.dim() < 2:
        x2 = x.unsqueeze(0)
        squeeze = True
    else:
        x2 = x.contiguous().view(-1, x.shape[-1])
        squeeze = False
    n_rows, hidden = x2.shape
    if weight.numel() != hidden:
        raise ValueError(
            f"weight numel {weight.numel()} != hidden {hidden}")
    if out is None:
        y = torch.empty_like(x2)
    else:
        _check_bf16_cuda("out", out)
        y = out.view(-1, hidden)
        if y.shape != x2.shape:
            raise ValueError(f"out shape {out.shape} != x shape {x.shape}")

    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = _lib().rmsnorm_fuse_fwd(
        ctypes.c_void_p(x2.data_ptr()),
        ctypes.c_void_p(weight.contiguous().data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(n_rows), ctypes.c_int(hidden),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"rmsnorm_fuse_fwd kernel rc={rc}")
    if squeeze:
        return y.view(x.shape)
    return y.view(x.shape)


def rmsnorm_fuse_residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out_norm: Optional[torch.Tensor] = None,
    out_new_residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """tmp = x + residual; y = weight * tmp / sqrt(mean(tmp^2) + eps).

    Returns (y_norm, new_residual=tmp).
    """
    _check_bf16_cuda("x", x)
    _check_bf16_cuda("residual", residual)
    _check_bf16_cuda("weight", weight)
    if x.shape != residual.shape:
        raise ValueError(f"x {x.shape} != residual {residual.shape}")
    orig_shape = x.shape
    x2  = x.view(-1, x.shape[-1])
    r2  = residual.contiguous().view(-1, x.shape[-1])
    n_rows, hidden = x2.shape
    if weight.numel() != hidden:
        raise ValueError(
            f"weight numel {weight.numel()} != hidden {hidden}")

    if out_norm is None:
        y = torch.empty_like(x2)
    else:
        _check_bf16_cuda("out_norm", out_norm)
        y = out_norm.view(-1, hidden)
    if out_new_residual is None:
        nr = torch.empty_like(x2)
    else:
        _check_bf16_cuda("out_new_residual", out_new_residual)
        nr = out_new_residual.view(-1, hidden)

    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = _lib().rmsnorm_fuse_residual_add(
        ctypes.c_void_p(x2.data_ptr()),
        ctypes.c_void_p(r2.data_ptr()),
        ctypes.c_void_p(weight.contiguous().data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_void_p(nr.data_ptr()),
        ctypes.c_int(n_rows), ctypes.c_int(hidden),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"rmsnorm_fuse_residual_add kernel rc={rc}")
    return y.view(orig_shape), nr.view(orig_shape)
