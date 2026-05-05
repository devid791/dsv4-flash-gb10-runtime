"""MXFP4 unpack — OCP Microscaling Formats v1.0 spec compliant.

MXFP4 = FP4 E2M1 element + per-block (32 elems) E8M0 scale (handled externally).

This module implements ONLY the nibble-decode step:
    packed uint8 bytes (2 nibbles/byte)  -->  bf16 tensor of FP4 codebook values.

External E8M0 scale application is the consumer's responsibility .

FP4 E2M1 layout (4 bits): [sign:1][exp:2][mantissa:1], no exponent bias adjustment
relative to the canonical OCP MX FP4 codebook.

Canonical OCP MX FP4 codebook (16 entries indexed by raw nibble value 0..15):
    [+0, +0.5, +1, +1.5, +2, +3, +4, +6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]

Reference: OCP Microscaling Formats Specification v1.0 (Sept 2023), Table 5.

Packing convention adopted here (matches majority of in-the-wild MXFP4 weight
checkpoints, e.g. RedHatAI / vLLM / mxfp4 OCP reference):
    nibble_low  (bits 0..3) = element 2*i
    nibble_high (bits 4..7) = element 2*i + 1

i.e. byte = (elem[2*i+1] << 4) | elem[2*i] & 0x0F.
"""
from __future__ import annotations


from typing import Tuple

import numpy as np
import torch


# Canonical OCP MX FP4 codebook (E2M1, no custom bias) — order MUST match nibble encoding.
# Index: raw 4-bit nibble value (0..15) -> real value.
FP4_E2M1_CODEBOOK: Tuple[float, ...] = (
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _build_lut(dtype: torch.dtype, device: torch.device | str = "cpu") -> torch.Tensor:
    """Build a 16-entry lookup table tensor in `dtype` on `device`."""
    return torch.tensor(FP4_E2M1_CODEBOOK, dtype=dtype, device=device)


def mxfp4_unpack(
    packed_bytes: np.ndarray | torch.Tensor,
    output_shape: tuple,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Unpack MXFP4-packed bytes to a real-valued tensor (no scale applied).

    Args:
        packed_bytes: uint8 array. Total nibbles MUST equal product(output_shape).
            Accepts numpy.ndarray or torch.Tensor of dtype uint8 / int8 / bool view.
            Shape is treated as flat nibble stream after row-major flatten.
        output_shape: target logical shape (in elements, NOT in bytes). The last
            dim is the one along which nibbles are unpacked, but since we work on
            the flat stream the only constraint is total count = N/2 bytes -> N elems.
        dtype: target floating dtype for the output (default torch.bfloat16).
        device: target device (default "cpu").

    Returns:
        torch.Tensor of `dtype` on `device`, shape == `output_shape`, with values
        drawn exclusively from FP4_E2M1_CODEBOOK.

    Raises:
        ValueError: if byte count does not match expected nibble count, or if
            packed input dtype is not 8-bit-wide.
    """
    # ---- normalize input to a uint8 numpy view (no copy when possible) ----
    if isinstance(packed_bytes, torch.Tensor):
        if packed_bytes.dtype not in (torch.uint8, torch.int8):
            raise ValueError(
                f"packed_bytes torch.Tensor must be uint8 or int8, got {packed_bytes.dtype}"
            )
        # contiguous + cpu numpy view as uint8
        bytes_np = packed_bytes.detach().contiguous().cpu().view(torch.uint8).numpy()
    else:
        if packed_bytes.dtype != np.uint8:
            # accept int8 by reinterpret
            if packed_bytes.dtype == np.int8:
                bytes_np = packed_bytes.view(np.uint8)
            else:
                raise ValueError(
                    f"packed_bytes ndarray must be uint8 or int8, got {packed_bytes.dtype}"
                )
        else:
            bytes_np = packed_bytes

    bytes_flat = np.ascontiguousarray(bytes_np).reshape(-1)

    expected_nibbles = 1
    for d in output_shape:
        expected_nibbles *= int(d)

    if bytes_flat.size * 2 != expected_nibbles:
        raise ValueError(
            f"byte count {bytes_flat.size} -> {bytes_flat.size * 2} nibbles "
            f"!= product(output_shape)={expected_nibbles}"
        )

    # ---- split nibbles: low = elem[2i], high = elem[2i+1] ----
    low = (bytes_flat & 0x0F).astype(np.int64)
    high = ((bytes_flat >> 4) & 0x0F).astype(np.int64)

    # interleave: [low0, high0, low1, high1, ...]
    nibbles = np.empty(bytes_flat.size * 2, dtype=np.int64)
    nibbles[0::2] = low
    nibbles[1::2] = high

    # ---- LUT lookup ----
    nibble_t = torch.from_numpy(nibbles).to(device)
    lut = _build_lut(dtype=dtype, device=device)
    out = lut[nibble_t]
    return out.reshape(output_shape).contiguous()


def mxfp4_pack(values: torch.Tensor | np.ndarray) -> np.ndarray:
    """Pack a float tensor whose values are exactly in FP4_E2M1_CODEBOOK into uint8 bytes.

    Inverse of `mxfp4_unpack` (assuming flat row-major order, low-nibble first).

    Args:
        values: float tensor/array, total element count must be even, every value
            must equal one of the 16 codebook entries (within bit-exact equality
            after cast to float64).

    Returns:
        np.ndarray dtype uint8 shape (N/2,).

    Raises:
        ValueError: on odd element count or out-of-codebook values.
    """
    if isinstance(values, torch.Tensor):
        arr = values.detach().contiguous().cpu().to(torch.float64).numpy().reshape(-1)
    else:
        arr = np.ascontiguousarray(values).astype(np.float64).reshape(-1)

    if arr.size % 2 != 0:
        raise ValueError(f"element count must be even for nibble packing, got {arr.size}")

    # Map each value -> nibble index. Use exact match against codebook (codebook
    # entries are exactly representable in float64).
    codebook = np.array(FP4_E2M1_CODEBOOK, dtype=np.float64)

    # For each value find its nibble index. We allow +0/-0 to map to their distinct
    # nibble (0 vs 8); use signbit to disambiguate.
    nibbles = np.empty(arr.size, dtype=np.uint8)
    for i, v in enumerate(arr):
        # find candidates by absolute-value match then pick by sign
        if v == 0.0:
            # +0.0 (signbit False) -> nibble 0; -0.0 (signbit True) -> nibble 8
            nibbles[i] = 8 if np.signbit(v) else 0
            continue
        matches = np.where(codebook == v)[0]
        if matches.size == 0:
            raise ValueError(
                f"value {v} at index {i} not in FP4 codebook {FP4_E2M1_CODEBOOK}"
            )
        nibbles[i] = matches[0]

    low = nibbles[0::2]
    high = nibbles[1::2]
    packed = ((high << 4) | low).astype(np.uint8)
    return packed


__all__ = ["FP4_E2M1_CODEBOOK", "mxfp4_unpack", "mxfp4_pack"]
