"""
RoPE/YARN positional encoding for DSv4-Flash.

Spec from:
- weights-bf16-hf/inference/model.py (precompute_freqs_cis + apply_rotary_emb)
- weights-bf16-hf/config.json (rope_theta, rope_scaling block)

DSv4 specifics:
- Uses torch.polar (complex64) for precomputed freqs_cis
- apply_rotary_emb: view_as_complex on last-dim, multiply by freqs_cis,
  view_as_real and copy back IN-PLACE on input tensor
- YARN: smooth ramp interpolation between low/high correction freqs
- NO mscale applied in DSv4 reference (different from raw YaRN paper)
- For DSv4 attn:  rope_head_dim=64, head_dim=512, factor=16, original=65536, base=10000
- For compress/indexer: same dims, base=160000 (compress_rope_theta), original=65536
- Pure SW layers (compress_ratio=0): NO YaRN (original_seq_len=0), base=10000 only
"""
from __future__ import annotations
import math
from functools import lru_cache
from typing import Optional

import torch


def _find_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _find_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int):
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_factor(low: float, high: float, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(linear, 0.0, 1.0)


@lru_cache(maxsize=4)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int = 0,
    base: float = 10000.0,
    factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Precompute rotary cis (complex64) freqs.

    Returns: [seqlen, dim/2] complex64 on device.

    When original_seq_len > 0: applies YARN smooth ramp to extend context.
    When original_seq_len = 0: pure RoPE (no YARN), base only.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    # Base inverse frequencies (positions in even dims of head)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if original_seq_len > 0:
        low, high = _find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1.0 - _linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1.0 - smooth) + freqs * smooth

    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)            # [seqlen, dim/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis.to(device)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply rotary embedding IN-PLACE on last dim of x.

    x:         [..., seq, n_heads, dim] OR [..., seq, dim]  (BF16/FP16/FP32)
    freqs_cis: [seq, dim/2] complex64
    inverse:   if True, applies conjugate (de-rotate; used for output dim)

    Returns: x mutated in place (same dtype as input).
    """
    y = x
    orig_dtype = x.dtype
    # view_as_complex on float32 view of (last dim split into pairs)
    x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_c.ndim == 3:
        # shape [batch, seq, dim/2]
        freqs_cis = freqs_cis.view(1, x_c.size(1), x_c.size(-1))
    elif x_c.ndim == 4:
        # shape [batch, seq, n_heads, dim/2]
        freqs_cis = freqs_cis.view(1, x_c.size(1), 1, x_c.size(-1))
    else:
        raise ValueError(f"unsupported ndim {x_c.ndim}")
    x_rot = torch.view_as_real(x_c * freqs_cis).flatten(-2)
    y.copy_(x_rot.to(orig_dtype))
    return y


def apply_rotary_emb_2d(
    x: torch.Tensor,        # [seq, n_heads, rope_dim]  (no batch)
    freqs_cis: torch.Tensor,  # [seq, rope_dim/2]
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE on tensor without batch dim (engine uses [S, H, D] layout).

    Mutates and returns x.
    """
    orig_dtype = x.dtype
    x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))  # [seq, n_heads, dim/2]
    if inverse:
        freqs_cis = freqs_cis.conj()
    # broadcast: [seq, 1, dim/2]
    freqs_cis = freqs_cis.view(x_c.size(0), 1, x_c.size(-1))
    x_rot = torch.view_as_real(x_c * freqs_cis).flatten(-2)
    x.copy_(x_rot.to(orig_dtype))
    return x


__all__ = ["precompute_freqs_cis", "apply_rotary_emb", "apply_rotary_emb_2d"]
