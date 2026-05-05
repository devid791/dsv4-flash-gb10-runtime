"""
e8m0_decode.py — pure E8M0 (Microscaling MX) byte → FP32 scale decoder.

E8M0 spec (OCP Microscaling Formats v1.0, §5.4 "Shared scale data type"):
  - 8-bit unsigned exponent only, no sign bit, no mantissa
  - Bias = 127
  - byte ∈ [0, 254]  ⇒  real_scale = 2^(byte - 127)
        byte = 0     ⇒  2^-127  (smallest normal scale, ≈ 5.877e-39)
        byte = 127   ⇒  2^0     = 1.0
        byte = 255   ⇒  NaN     (sole encoded special; no Inf, no zero)

  - Range: 2^-127 .. 2^127  (≈ 5.88e-39 .. 1.70e38)

References:
  - OCP MX Formats Specification v1.0 (open-compute, Sep 2023), §5.4
  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Used by tecaprovn/deepseek-v4-flash-gguf which stores per-block E8M0 bytes in
companion F32 tensors (each scalar holds the raw byte cast to float). See
runtime/dsv4_engine_q3.py::dequant_q3_with_external_e8m0_scale.
"""
from __future__ import annotations

import numpy as np
import torch


_E8M0_NAN_BYTE = 0xFF   # 255
_E8M0_BIAS = 127


def e8m0_decode(byte_array: np.ndarray) -> np.ndarray:
    """Decode E8M0 bytes → FP32 scale values.

    Args:
        byte_array: numpy uint8 array of any shape. Values must be in [0, 255].

    Returns:
        float32 numpy array, same shape as input.
        - byte == 0xFF (255)  →  np.nan
        - else                 →  2^(byte - 127)

    Raises:
        TypeError: if dtype is not uint8 (use .astype(np.uint8) explicitly).
    """
    if not isinstance(byte_array, np.ndarray):
        byte_array = np.asarray(byte_array)
    if byte_array.dtype != np.uint8:
        raise TypeError(
            f"e8m0_decode expects np.uint8 input, got {byte_array.dtype}. "
            f"Cast explicitly with .astype(np.uint8) to avoid silent truncation."
        )

    # Cast to int32 BEFORE subtraction to allow negative exponents (byte<127).
    exp = byte_array.astype(np.int32) - _E8M0_BIAS
    # Mask NaN sentinel (0xFF) BEFORE pow to avoid fp64→fp32 overflow warning at 2^128.
    nan_mask = (byte_array == _E8M0_NAN_BYTE)
    safe_exp = np.where(nan_mask, np.int32(0), exp).astype(np.float64)
    # 2^exp, computed in float64 to retain precision down to 2^-127, then to f32.
    out = np.power(2.0, safe_exp).astype(np.float32)
    # Apply NaN sentinel
    if nan_mask.any():
        out = np.where(nan_mask, np.float32(np.nan), out)
    return out


def e8m0_decode_torch(byte_tensor: torch.Tensor) -> torch.Tensor:
    """GPU-friendly E8M0 decode → fp32 torch tensor (same shape, same device).

    Args:
        byte_tensor: torch.uint8 tensor (CPU or CUDA), any shape.

    Returns:
        float32 torch tensor on the same device, same shape.
        - byte == 0xFF (255)  →  NaN
        - else                 →  2^(byte - 127)

    Raises:
        TypeError: if dtype is not torch.uint8.
    """
    if not isinstance(byte_tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(byte_tensor)}")
    if byte_tensor.dtype != torch.uint8:
        raise TypeError(
            f"e8m0_decode_torch expects torch.uint8 input, got {byte_tensor.dtype}. "
            f"Cast explicitly with .to(torch.uint8)."
        )

    device = byte_tensor.device

    # Subtract bias in int32 to allow signed exponent.
    exp = byte_tensor.to(torch.int32) - _E8M0_BIAS
    nan_mask = (byte_tensor == _E8M0_NAN_BYTE)
    # Mask 0xFF to neutral exp=0 BEFORE pow to avoid fp64→fp32 overflow at 2^128.
    safe_exp = torch.where(nan_mask, torch.zeros_like(exp), exp).to(torch.float64)
    out = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=device),
                    safe_exp).to(torch.float32)
    if nan_mask.any():
        out = out.masked_fill(nan_mask, float("nan"))
    return out
