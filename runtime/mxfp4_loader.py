"""
mxfp4_loader.py - MXFP4 safetensors loader for DSv4-Flash HF checkpoint.

Davide Zenati 2026-05-04.

Reads the HuggingFace safetensors snapshot of `deepseek-ai/DeepSeek-V4-Flash`
on disk (159 GB, 46 shards, 69187 tensors) and exposes a simple lazy API
for the engine:

    loader = MXFP4Loader("${DSV4_WEIGHTS}/")

    # Routed-expert MXFP4 weight (I8 packed nibbles + F8_E8M0 block scales)
    w_packed_u8, scale_u8 = loader.load_mxfp4("layers.0.ffn.experts.0.w1")
        # w_packed_u8 : torch.uint8  shape [rows, cols/2]   (2 nibbles / byte)
        # scale_u8    : torch.uint8  shape [rows, cols/32]  (E8M0 byte / 32-elem block)

    # Attention / shared-expert FP8 weight (F8_E4M3 + F8_E8M0)
    w_fp8, scale_u8 = loader.load_fp8("layers.0.attn.wq_a")
        # w_fp8     : torch.uint8  raw E4M3 bit pattern, shape [rows, cols]
        # scale_u8  : torch.uint8  E8M0,  shape depends on tensor

    # BF16 plain tensor (norms, embeddings, router gate)
    t = loader.load_bf16("embed.weight")             # torch.bfloat16
    t = loader.load_bf16("layers.0.attn_norm.weight")

    # Raw escape hatch (any dtype)
    t = loader.load_raw("layers.0.attn.attn_sink")   # F32, etc.

    # Dequantize a routed-expert MXFP4 weight to BF16 (debug / cross-check path)
    w_bf16 = loader.dequant_mxfp4("layers.0.ffn.experts.0.w1")

Layout discovered empirically from the published HF checkpoint
(`model.safetensors.index.json` + `safe_open` probes):

    Routed experts  layers.{L}.ffn.experts.{E}.w{1|2|3}.weight   I8        [rows, cols/2]
                    layers.{L}.ffn.experts.{E}.w{1|2|3}.scale    F8_E8M0   [rows, cols/32]
    Shared experts  layers.{L}.ffn.shared_experts.w{1|2|3}.weight F8_E4M3  [rows, cols]
                    layers.{L}.ffn.shared_experts.w{1|2|3}.scale  F8_E8M0  varies
    Attention       layers.{L}.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight F8_E4M3
                    layers.{L}.attn.{...}.scale                       F8_E8M0
    Norms / gate    BF16
    Embed / lm_head BF16
    HC / sink       F32
    tid2eid         I64 (router pre-mapping)

Block size for E8M0 scales = 32 elements along the LAST dim of the weight
(not 128, not 256). For routed experts that means w_packed has shape
[K, N/2] and scale has shape [K, N/32].

Notes
-----
* `safetensors.safe_open` already mmaps the file lazily and returns
  zero-copy `torch.Tensor`s sharing the underlying mapping (we never
  copy the bytes into Python memory unless the caller `.to(device)` or
  `.contiguous()`). The framework owns the mmap lifetime.
* We open each shard at most ONCE and keep its `safe_open` context alive
  in `self._open_files` - closing happens in `MXFP4Loader.close()`.
* No transformers / no accelerate / no torch.load - purely the safetensors
  format parser.
"""
from __future__ import annotations


import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
from safetensors import safe_open

# Local A2 / A3 modules (sibling files, cherry-picked into B1 branch).
import e8m0_decode as _e8m0
import mxfp4_unpack as _mxfp4


# ---- safetensors dtype string constants (what `get_dtype()` returns) ----
_DTYPE_BF16     = "BF16"
_DTYPE_F32      = "F32"
_DTYPE_F16      = "F16"
_DTYPE_I8       = "I8"            # MXFP4 packed nibbles live here
_DTYPE_I64      = "I64"
_DTYPE_F8_E4M3  = "F8_E4M3"
_DTYPE_F8_E5M2  = "F8_E5M2"
_DTYPE_F8_E8M0  = "F8_E8M0"       # MX shared scale

# Default MXFP4 block size along the last dim (OCP MX v1.0 spec).
MXFP4_BLOCK_SIZE = 32


@dataclass
class TensorMeta:
    """Lightweight description of a tensor in the HF index."""
    name: str
    shard: str       # filename within the snapshot dir
    dtype: str       # safetensors dtype string ("I8", "BF16", ...)
    shape: Tuple[int, ...]


class MXFP4Loader:
    """Lazy reader over a HF safetensors snapshot.

    Parameters
    ----------
    snapshot_dir : str | Path
        Directory containing `model.safetensors.index.json` and the shard
        files (e.g. `model-00001-of-00046.safetensors`).
    """

    def __init__(self, snapshot_dir):
        self.dir = Path(snapshot_dir)
        idx_path = self.dir / "model.safetensors.index.json"
        if not idx_path.is_file():
            raise FileNotFoundError(f"index missing: {idx_path}")
        with open(idx_path, "r") as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]

        # Build the metadata cache lazily - we only know dtype/shape on demand.
        self._weight_map = weight_map
        self._meta_cache = {}
        self._open_files = {}

    # --- basic introspection -----------------------------------------------

    def __len__(self):
        return len(self._weight_map)

    def __contains__(self, name):
        return name in self._weight_map

    def names(self):
        return iter(self._weight_map.keys())

    def get_meta(self, name):
        """Return dtype + shape of `name` (cached after first probe)."""
        if name in self._meta_cache:
            return self._meta_cache[name]
        if name not in self._weight_map:
            raise KeyError(f"tensor not in HF index: {name!r}")
        shard = self._weight_map[name]
        f = self._open(shard)
        slc = f.get_slice(name)
        meta = TensorMeta(
            name=name,
            shard=shard,
            dtype=str(slc.get_dtype()),
            shape=tuple(slc.get_shape()),
        )
        self._meta_cache[name] = meta
        return meta

    # --- shard handle pool -------------------------------------------------

    def _open(self, shard):
        f = self._open_files.get(shard)
        if f is None:
            path = self.dir / shard
            # safe_open returns a context manager; we enter manually and
            # store the live handle. close() drops the dict, GC closes mmap.
            cm = safe_open(str(path), framework="pt")
            f = cm.__enter__()
            # Stash the context manager so __exit__ runs at close().
            self._open_files[shard] = (cm, f)
            return f
        return f[1]

    def close(self):
        for shard, (cm, _) in self._open_files.items():
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
        self._open_files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # --- core typed accessors ----------------------------------------------

    def _get_typed_tensor(self, name, expected_dtype, target_uint8=False):
        """Internal helper: open, validate dtype, return tensor.

        If `target_uint8` is True, reinterpret-cast the underlying bytes to
        torch.uint8 with shape preserved (used for I8 / F8_E* / F8_E8M0
        which we want to handle as raw bytes).
        """
        meta = self.get_meta(name)
        if isinstance(expected_dtype, str):
            ok = meta.dtype == expected_dtype
        else:
            ok = meta.dtype in expected_dtype
        if not ok:
            raise TypeError(
                f"{name!r}: expected dtype {expected_dtype}, got {meta.dtype}"
            )
        f = self._open(meta.shard)
        t = f.get_tensor(name)  # zero-copy torch.Tensor backed by mmap
        if target_uint8:
            # Reinterpret bytes as uint8 with same shape - safetensors gives
            # us a tensor whose element-stride matches the dtype, so just
            # `.view(torch.uint8)` works for 1-byte dtypes.
            if t.element_size() != 1:
                raise RuntimeError(
                    f"{name!r}: target_uint8 requested but element_size={t.element_size()}"
                )
            t = t.view(torch.uint8)
        return t

    # --- public API: MXFP4 routed experts ----------------------------------

    def load_mxfp4(self, name):
        """Load a routed-expert MXFP4 weight + its E8M0 scale.

        Parameters
        ----------
        name : str
            Tensor base name WITHOUT the `.weight` / `.scale` suffix
            (e.g. `"layers.0.ffn.experts.0.w1"`).

        Returns
        -------
        packed_uint8 : torch.uint8
            Shape [rows, cols/2]; nibble layout matches `mxfp4_unpack`
            (low nibble = even col).
        scale_uint8  : torch.uint8
            Shape [rows, cols/32]; raw E8M0 bytes
            (decode with `e8m0_decode_torch`).
        """
        w_name = name + ".weight"
        s_name = name + ".scale"
        w = self._get_typed_tensor(w_name, _DTYPE_I8, target_uint8=True)
        s = self._get_typed_tensor(s_name, _DTYPE_F8_E8M0, target_uint8=True)
        return w, s

    # --- public API: FP8 attention / shared experts ------------------------

    def load_fp8(self, name):
        """Load an FP8 (E4M3) weight + its E8M0 block scale.

        Used for attention projections and shared-expert weights. Returns the
        FP8 tensor as raw `torch.uint8` (caller is responsible for E4M3
        decode) plus the E8M0 scale as raw `torch.uint8`.
        """
        w_name = name + ".weight"
        s_name = name + ".scale"
        w = self._get_typed_tensor(
            w_name, (_DTYPE_F8_E4M3, _DTYPE_F8_E5M2), target_uint8=True
        )
        s = self._get_typed_tensor(s_name, _DTYPE_F8_E8M0, target_uint8=True)
        return w, s

    # --- public API: BF16 plain tensors ------------------------------------

    def load_bf16(self, name):
        """Load a plain BF16 tensor (norms, embed, lm_head, router gate)."""
        return self._get_typed_tensor(name, _DTYPE_BF16)

    def load_raw(self, name):
        """Escape hatch - returns the tensor as-is (any dtype, F32, I64...)."""
        meta = self.get_meta(name)
        f = self._open(meta.shard)
        return f.get_tensor(name)

    # --- debug / cross-check: MXFP4 -> BF16 dequantization -----------------

    def dequant_mxfp4(self, name, device="cpu"):
        """Dequantize a routed-expert MXFP4 weight to BF16.

        Slow path (CPU codebook lookup) used for ground-truth comparisons.
        For runtime, use the fused GPU GEMV (B2/B3 agents).
        """
        packed, scale_u8 = self.load_mxfp4(name)
        packed = packed.contiguous().cpu()
        scale_u8 = scale_u8.contiguous().cpu()

        rows, cols_packed = packed.shape
        cols = cols_packed * 2
        block = MXFP4_BLOCK_SIZE

        if scale_u8.shape != (rows, cols // block):
            raise ValueError(
                f"{name!r}: scale shape {tuple(scale_u8.shape)} "
                f"!= expected ({rows}, {cols // block})"
            )

        # Step 1: nibble unpack (CPU, returns FP32 codebook values).
        unpacked = _mxfp4.mxfp4_unpack(
            packed.numpy(),
            output_shape=(rows, cols),
            dtype=torch.float32,
            device="cpu",
        )

        # Step 2: apply E8M0 scale per 32-elem block along last dim.
        scale_f32 = torch.from_numpy(_e8m0.e8m0_decode(scale_u8.numpy()))
        # broadcast: [rows, cols/32, 1] * [rows, cols/32, 32] -> [rows, cols]
        unpacked = unpacked.view(rows, cols // block, block)
        scaled = unpacked * scale_f32.unsqueeze(-1)
        out = scaled.view(rows, cols).to(torch.bfloat16)

        if device != "cpu":
            out = out.to(device)
        return out

    # --- inventory helpers -------------------------------------------------

    def inventory(self, max_names=None):
        """Return a count of tensors per safetensors dtype (probes lazily)."""
        from collections import Counter
        c = Counter()
        names = list(self._weight_map.keys())
        if max_names is not None:
            names = names[:max_names]
        for n in names:
            c[self.get_meta(n).dtype] += 1
        return dict(c)


__all__ = ["MXFP4Loader", "TensorMeta", "MXFP4_BLOCK_SIZE"]
