"""mxfp4_grouped_gemv_binding.py — torch-friendly ctypes wrapper for the
MXFP4 grouped GEMV top-K=6 kernel (, sm_121a Blackwell).

Public API:
    mxfp4_grouped_gemv(W1, W3, W2, scales, x, r) -> torch.Tensor

  W1, W3, W2: dict with keys
       'packed': torch.uint8 tensor of shape
                 - W1/W3: (n_exp, hidden, K_in/2)
                 - W2:    (n_exp, out_dim, hidden/2)
       'scale':  torch.uint8 tensor of shape
                 - W1/W3: (n_exp, hidden, K_in/MXFP4_BLOCK_ELEMS)
                 - W2:    (n_exp, out_dim, hidden/MXFP4_BLOCK_ELEMS)
  scales: alias unused — scales live inside W{1,3,2}['scale']. (Argument retained
          for API symmetry per spec.)
  x: torch.bfloat16 tensor (K_in,)
  r: torch.float32 tensor (n_exp,)

Returns torch.bfloat16 tensor (out_dim,).
"""
from __future__ import annotations

import ctypes
import os
from typing import Any, Dict

import torch

MXFP4_BLOCK_ELEMS = 32

_LIB_PATH = os.environ.get(
    "MXFP4_GROUPED_LIB",
    os.path.join(os.path.dirname(__file__),
                 "..", "kernel", "mxfp4-routed", "libmxfp4_grouped_gemv.so"),
)
_LIB_PATH = os.path.abspath(_LIB_PATH)

_lib: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(
            f"libmxfp4_grouped_gemv.so not found at {_LIB_PATH}. "
            f"Build with `make -C kernel/mxfp4-routed`."
        )
    lib = ctypes.CDLL(_LIB_PATH)

    # int mxfp4_grouped_gemv_topk(
    #   const __nv_bfloat16* x,
    #   const uint8_t* W1_packed, W3_packed, W2_packed,
    #   const uint8_t* W1_scale,  W3_scale,  W2_scale,
    #   const float* rweights,
    #   int K_in, int hidden, int out_dim, int n_exp,
    #   __nv_bfloat16* out,
    #   void* workspace,
    #   cudaStream_t stream
    # );
    lib.mxfp4_grouped_gemv_topk.restype = ctypes.c_int
    lib.mxfp4_grouped_gemv_topk.argtypes = [
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # W1_packed
        ctypes.c_void_p,  # W3_packed
        ctypes.c_void_p,  # W2_packed
        ctypes.c_void_p,  # W1_scale
        ctypes.c_void_p,  # W3_scale
        ctypes.c_void_p,  # W2_scale
        ctypes.c_void_p,  # rweights
        ctypes.c_int,     # K_in
        ctypes.c_int,     # hidden
        ctypes.c_int,     # out_dim
        ctypes.c_int,     # n_exp
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # workspace
        ctypes.c_void_p,  # cudaStream_t
    ]
    lib.mxfp4_grouped_workspace_bytes.restype = ctypes.c_size_t
    lib.mxfp4_grouped_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _lib = lib
    return lib


def _ck_dev_dtype(t: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if not t.is_cuda:
        raise ValueError(f"{name}: expected CUDA tensor, got device={t.device}")
    if t.dtype != dtype:
        raise ValueError(f"{name}: expected dtype {dtype}, got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{name}: tensor must be contiguous")


def mxfp4_grouped_gemv(
    W1: Dict[str, torch.Tensor],
    W3: Dict[str, torch.Tensor],
    W2: Dict[str, torch.Tensor],
    scales: Any,                      # unused — see module docstring
    x: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Run the MXFP4 grouped GEMV top-K kernel and return BF16 output.

    See module docstring for tensor shape contracts.
    """
    del scales  # API placeholder; scales live in W{1,3,2}['scale']

    lib = _load_lib()

    # Validate inputs
    _ck_dev_dtype(x, "x", torch.bfloat16)
    _ck_dev_dtype(r, "r", torch.float32)
    for name, w in (("W1", W1), ("W3", W3), ("W2", W2)):
        if "packed" not in w or "scale" not in w:
            raise ValueError(f"{name} must contain 'packed' and 'scale' tensors")
        _ck_dev_dtype(w["packed"], f"{name}.packed", torch.uint8)
        _ck_dev_dtype(w["scale"],  f"{name}.scale",  torch.uint8)

    K_in = x.numel()
    n_exp = W1["packed"].shape[0]
    hidden = W1["packed"].shape[1]
    out_dim = W2["packed"].shape[1]

    if W3["packed"].shape != W1["packed"].shape:
        raise ValueError(f"W3.packed shape {tuple(W3['packed'].shape)} != W1.packed shape {tuple(W1['packed'].shape)}")
    if W1["packed"].shape[2] != K_in // 2:
        raise ValueError(f"W1.packed last dim {W1['packed'].shape[2]} != K_in/2={K_in//2}")
    if W1["scale"].shape != (n_exp, hidden, K_in // MXFP4_BLOCK_ELEMS):
        raise ValueError(
            f"W1.scale shape {tuple(W1['scale'].shape)} != "
            f"({n_exp},{hidden},{K_in // MXFP4_BLOCK_ELEMS})"
        )
    if W3["scale"].shape != W1["scale"].shape:
        raise ValueError(f"W3.scale shape mismatch")
    if W2["packed"].shape[0] != n_exp or W2["packed"].shape[2] != hidden // 2:
        raise ValueError(
            f"W2.packed shape {tuple(W2['packed'].shape)} != "
            f"({n_exp},out_dim,{hidden // 2})"
        )
    if W2["scale"].shape != (n_exp, out_dim, hidden // MXFP4_BLOCK_ELEMS):
        raise ValueError(
            f"W2.scale shape {tuple(W2['scale'].shape)} != "
            f"({n_exp},{out_dim},{hidden // MXFP4_BLOCK_ELEMS})"
        )
    if r.shape != (n_exp,):
        raise ValueError(f"r shape {tuple(r.shape)} != ({n_exp},)")

    out = torch.empty(out_dim, dtype=torch.bfloat16, device=x.device)
    ws_bytes = lib.mxfp4_grouped_workspace_bytes(n_exp, hidden, out_dim)
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=x.device)

    stream = torch.cuda.current_stream(x.device).cuda_stream

    rc = lib.mxfp4_grouped_gemv_topk(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(W1["packed"].data_ptr()),
        ctypes.c_void_p(W3["packed"].data_ptr()),
        ctypes.c_void_p(W2["packed"].data_ptr()),
        ctypes.c_void_p(W1["scale"].data_ptr()),
        ctypes.c_void_p(W3["scale"].data_ptr()),
        ctypes.c_void_p(W2["scale"].data_ptr()),
        ctypes.c_void_p(r.data_ptr()),
        ctypes.c_int(K_in),
        ctypes.c_int(hidden),
        ctypes.c_int(out_dim),
        ctypes.c_int(n_exp),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"mxfp4_grouped_gemv_topk returned {rc}")
    return out


__all__ = ["mxfp4_grouped_gemv", "MXFP4_BLOCK_ELEMS"]
