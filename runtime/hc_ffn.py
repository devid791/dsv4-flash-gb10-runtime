"""
hc_ffn.py — DSv4-Flash HyperConnection (hc_pre / hc_post / hc_head).

Davide Zenati, 2026-05-04 (Q3-W2).

Implements the per-layer HyperConnection mechanism from the "Hyper-Connections"
paper (Zhu et al., 2024) as exposed in the DSv4-Flash GGUF. Every transformer
layer carries n_hc=4 parallel residual streams; before each block (attention,
FFN/MoE) we collapse them via a learned mixer (`hc_pre`) to produce the block
input, run the block, then redistribute the block output back into the n_hc
streams via a doubly-stochastic combiner (`hc_post`). After the last layer
`hc_head` collapses the four streams into one before the final RMSNorm + lm_head.

Shapes (per layer, for `hc_attn_*` / `hc_ffn_*`; `hc_head_*` analogous):
  hc_fn:     [hc_dim=16384, hc_mix=24]       GGUF (K, N)
             stored in torch as           [24, 16384]
             -> linear(flat, hc_fn) yields [T, 24]
  hc_base:   [24]
  hc_scale:  [3]   (pre_scale, post_scale, comb_scale)

For `hc_head_*`:
  hc_fn:     [16384, 4]  -> torch [4, 16384]
  hc_base:   [4]
  hc_scale:  [1]   (pre_scale only)

Pseudocode reference (from spec, paper Sec 3.2):
  flat   = rmsnorm(x.reshape(T, n_hc*n_embd))
  mixes  = flat @ hc_fn.T                              [T, hc_mix=24]
  pre    = sigmoid(mixes[:, 0:n_hc] * pre_scale + base[0:n_hc]) + hc_eps
  post   = 2 * sigmoid(mixes[:, n_hc:2*n_hc] * post_scale + base[n_hc:2*n_hc])
  comb   = softmax(mixes[:, 2*n_hc:].reshape(T, n_hc, n_hc) * comb_scale
                   + base[2*n_hc:].reshape(n_hc, n_hc), dim=-1) + hc_eps
  comb   = sinkhorn(comb, iters=sinkhorn_iters)        # asymmetric first iter (cols only)
  x_pre  = einsum("th,thd->td", pre, x)
  return x_pre, pre, post, comb

  out    = einsum("th,td->thd", post, block_out)
         + einsum("ths,tsd->thd", comb, residual)

Bit-exact caveat: Sinkhorn first iteration normalises COLUMNS only — match this
asymmetry exactly. Subsequent iterations alternate row/col normalisation.
"""
from __future__ import annotations


import torch


# ─── RMSNorm (no learned weight — pure normalisation) ───────────────────────

def _rms_norm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm without affine weight (the affine part is absorbed by hc_fn).
    x: [..., D]. Computed in fp32, returns same dtype as input."""
    in_dtype = x.dtype
    f = x.float()
    rms = torch.rsqrt((f * f).mean(-1, keepdim=True) + eps)
    return (f * rms).to(in_dtype)


# ─── Sinkhorn doubly-stochastic normalisation ───────────────────────────────

def _sinkhorn(comb: torch.Tensor, n_hc: int, iters: int, eps: float) -> torch.Tensor:
    """Doubly-stochastic projection of comb [T, n_hc, n_hc].

    First iteration: normalise COLUMNS only (asymmetric, matches reference).
    Then `iters - 1` rounds of (row-norm, col-norm).

    Indexing convention: comb[t, dst, src] is the weight that the dst stream
    receives from the src stream of the previous residual.
    """
    # First pass: column normalisation only (sum over dst axis = dim 1)
    col_sum = comb.sum(dim=1, keepdim=True) + eps          # [T, 1, n_hc]
    comb = comb / col_sum

    for _ in range(iters - 1):
        row_sum = comb.sum(dim=2, keepdim=True) + eps      # [T, n_hc, 1]
        comb = comb / row_sum
        col_sum = comb.sum(dim=1, keepdim=True) + eps      # [T, 1, n_hc]
        comb = comb / col_sum

    return comb


# ─── hc_pre ──────────────────────────────────────────────────────────────────

def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_base: torch.Tensor,
    hc_scale: torch.Tensor,
    *,
    n_embd: int,
    n_hc: int,
    sinkhorn_iters: int = 3,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-block HyperConnection mixer.

    Args:
      x:        [T, n_hc, n_embd]   multi-stream residual.
      hc_fn:    [hc_mix=24, hc_dim=16384]  torch row-major (linear weight).
      hc_base:  [24].
      hc_scale: [3]   (pre_scale, post_scale, comb_scale).

    Returns:
      x_pre:    [T, n_embd]   block input.
      pre:      [T, n_hc]     stream-collapse weights (re-used by None here).
      post:     [T, n_hc]     stream redistribution weights.
      comb:     [T, n_hc, n_hc] doubly-stochastic mixer.
    """
    T = x.shape[0]
    hc_mix = 2 * n_hc + n_hc * n_hc  # 4 + 4 + 16 = 24 for n_hc=4

    # 1) Flatten + RMSNorm (no affine weight).
    flat = x.reshape(T, n_hc * n_embd).contiguous()
    flat = _rms_norm_no_weight(flat, eps=norm_eps)

    # 2) Mixer linear: flat @ hc_fn.T -> [T, 24]. Use fp32 for the small
    # projection (24 outputs) so subsequent sigmoid/softmax are stable.
    mixes = torch.nn.functional.linear(flat.float(), hc_fn.float())  # [T, 24]

    pre_scale, post_scale, comb_scale = hc_scale[0].item(), hc_scale[1].item(), hc_scale[2].item()
    base32 = hc_base.float()

    # 3a) PRE in (hc_eps, 1+hc_eps).
    pre = torch.sigmoid(mixes[:, 0:n_hc] * pre_scale + base32[0:n_hc]) + hc_eps

    # 3b) POST in (0, 2).
    post = 2.0 * torch.sigmoid(mixes[:, n_hc:2 * n_hc] * post_scale + base32[n_hc:2 * n_hc])

    # 3c) COMB doubly-stochastic.
    comb_logits = (
        mixes[:, 2 * n_hc:].reshape(T, n_hc, n_hc) * comb_scale
        + base32[2 * n_hc:].reshape(n_hc, n_hc)
    )
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = _sinkhorn(comb, n_hc=n_hc, iters=sinkhorn_iters, eps=hc_eps)

    # 4) Weighted collapse: x_pre[t,d] = sum_h pre[t,h] * x[t,h,d].
    x_pre = torch.einsum("th,thd->td", pre.to(x.dtype), x)

    return x_pre, pre.to(x.dtype), post.to(x.dtype), comb.to(x.dtype)


# ─── hc_post ─────────────────────────────────────────────────────────────────

def hc_post(
    block_out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Post-block HyperConnection redistribution.

    Args:
      block_out: [T, n_embd]      output of the attention or FFN/MoE block.
      residual:  [T, n_hc, n_embd] previous multi-stream residual.
      post:      [T, n_hc]
      comb:      [T, n_hc, n_hc]

    Returns:
      [T, n_hc, n_embd]   updated multi-stream residual.
    """
    new_block = torch.einsum("th,td->thd", post, block_out)
    mixed_residual = torch.einsum("ths,tsd->thd", comb, residual)
    return new_block + mixed_residual


# ─── hc_head ─────────────────────────────────────────────────────────────────

def hc_head(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_base: torch.Tensor,
    hc_scale: torch.Tensor,
    *,
    n_embd: int,
    n_hc: int,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Final-layer HyperConnection collapse: PRE-only (no post, no comb,
    no Sinkhorn).

    Args:
      x:        [T, n_hc, n_embd].
      hc_fn:    [n_hc=4, hc_dim=16384]   torch row-major.
      hc_base:  [n_hc=4].
      hc_scale: [1]   pre_scale only.

    Returns:
      [T, n_embd].
    """
    T = x.shape[0]
    flat = x.reshape(T, n_hc * n_embd).contiguous()
    flat = _rms_norm_no_weight(flat, eps=norm_eps)
    mixes = torch.nn.functional.linear(flat.float(), hc_fn.float())  # [T, n_hc]
    pre_scale = hc_scale[0].item()
    pre = torch.sigmoid(mixes * pre_scale + hc_base.float()) + hc_eps
    return torch.einsum("th,thd->td", pre.to(x.dtype), x)
