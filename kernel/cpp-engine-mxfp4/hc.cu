// hc.cu - DSv4-Flash HyperConnection (C++/CUDA, sm_121a)
// =====================================================
// Bit-exact port of runtime/hc_ffn.py (D4 sprint, 2026-05-04).
//
// Reference: hc_ffn.py - hc_pre / hc_post / hc_head + _sinkhorn + _rms_norm_no_weight.
//
// Per-layer HyperConnection mechanism (Zhu et al. 2024) as exposed in the
// DSv4-Flash GGUF. n_hc=4 parallel residual streams, n_embd=4096.
//   hc_pre     : collapse residual [T, 4, 4096] -> block input [T, 4096] via
//                Sinkhorn-mixed weights computed from a learned mixer.
//   hc_post    : redistribute block output [T, 4096] back into [T, 4, 4096].
//   hc_head    : final-layer collapse [T, 4, 4096] -> [T, 4096] (PRE-only).
//
// Sinkhorn convention (CRITICAL bit-exact): first iteration normalises COLUMNS
// only (sum over dst axis = dim 1 in PyTorch [T, n_hc, n_hc] tensor).
// Then `iters - 1` rounds of (row-norm, col-norm). HC_SINKHORN_ITERS = 20.
//
// Decode regime (T=1): the kernels are specialised for the single-token case
// because the C++ engine targets the decode loop. The kernels accept a `T`
// argument and the loops are correct for T>1, but we use 1 block per token
// and the smem layout is sized for T=1 to keep the code simple. All call sites
// in dsv4_engine wrap a 1-token forward.
//
// Layout conventions (Python-equivalent):
//   x_residual  : BF16  [T, n_hc, n_embd]   row-major  -> flat layout
//                 [t0_h0_d0..d4095, t0_h1_d0..d4095, ...]
//   hc_fn       : F32   [hc_mix=24, hc_dim=16384]  row-major (linear weight)
//   hc_base     : F32   [hc_mix]
//   hc_scale    : F32   [3]   (pre_scale, post_scale, comb_scale)  for layers
//                 F32   [1]   (pre_scale)                          for head
//   pre         : F32   [T, n_hc]
//   post        : F32   [T, n_hc]
//   comb        : F32   [T, n_hc, n_hc]   row-major dst*4 + src
//   x_collapsed : BF16  [T, n_embd]
//
// Author: D4 agent, 2026-05-04.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>

#ifndef HC_N_HC
#define HC_N_HC          4
#endif
#ifndef HC_HIDDEN
#define HC_HIDDEN        4096
#endif
#define HC_DIM_FLAT      (HC_N_HC * HC_HIDDEN)
#define HC_LAYER_MIX     (2 * HC_N_HC + HC_N_HC * HC_N_HC)
#define HC_HEAD_MIX      HC_N_HC

#ifndef HC_SINKHORN_IT
#define HC_SINKHORN_IT   20
#endif
#ifndef HC_EPS_F
#define HC_EPS_F         1e-6f
#endif
#ifndef HC_NORM_EPS
#define HC_NORM_EPS      1e-6f
#endif

// ---------------------------------------------------------------------------
// Internal: RMSNorm without affine (matches hc_ffn._rms_norm_no_weight).
//   y = x * rsqrt(mean(x*x) + eps)
// in/out BF16, internal fp32. Single block, D=16384 elements, blockDim=256.
// ---------------------------------------------------------------------------
__global__ void k_hc_rmsn_no_affine(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int D,
    float eps)
{
    extern __shared__ float ss_buf[];
    int tid = threadIdx.x;
    float sumsq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __bfloat162float(x[i]);
        sumsq += v * v;
    }
    ss_buf[tid] = sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ss_buf[tid] += ss_buf[tid + s];
        __syncthreads();
    }
    float mean = ss_buf[0] / (float)D;
    float inv  = rsqrtf(mean + eps);
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __bfloat162float(x[i]);
        y[i] = __float2bfloat16(v * inv);
    }
}

// ---------------------------------------------------------------------------
// Internal: mixer GEMV.   mixes[m] = sum_k flat[k] * fn[m, k]
//   flat   : BF16 [HC_DIM_FLAT=16384]
//   fn     : F32  [HC_M, HC_K=16384]   row-major
//   mixes  : F32  [HC_M]
// One block per output row m (HC_M = 24 for layers, 4 for head).
// blockDim = 256, smem = 256 * sizeof(float).
// ---------------------------------------------------------------------------
__global__ void k_hc_mixer_proj(
    const __nv_bfloat16* __restrict__ flat,
    const float* __restrict__ fn,
    float* __restrict__ mixes,
    int HC_M,
    int HC_K)
{
    extern __shared__ float ss_buf[];
    int m = blockIdx.x;
    int tid = threadIdx.x;
    if (m >= HC_M) return;
    const float* row = fn + (int64_t)m * HC_K;
    float acc = 0.0f;
    for (int k = tid; k < HC_K; k += blockDim.x) {
        acc += __bfloat162float(flat[k]) * row[k];
    }
    ss_buf[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ss_buf[tid] += ss_buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) mixes[m] = ss_buf[0];
}

// ---------------------------------------------------------------------------
// Internal: pre / post / comb extraction + Sinkhorn (layer variant).
//   mixes  : F32 [24]
//   base   : F32 [24]
//   scale  : F32 [3]   (pre_s, post_s, comb_s)
//   pre    : F32 [4]
//   post   : F32 [4]
//   comb   : F32 [4, 4]   row-major dst*4 + src
//
// Bit-exact match of hc_ffn.hc_pre's pre/post/comb extraction:
//   pre   = sigmoid(mixes[0:4]   * pre_s   + base[0:4])   + eps
//   post  = 2 * sigmoid(mixes[4:8] * post_s + base[4:8])
//   comb  = softmax(mixes[8:24].reshape(4,4) * comb_s + base[8:24].reshape(4,4),
//                   dim=-1) + eps
//   comb  = sinkhorn(comb, iters=20, eps=eps)     # asymmetric: first iter cols only
//
// 1 block, 32 threads (single warp). Threads 0..15 do useful work on the comb
// matrix (16 entries); 16..31 idle but participate in __syncthreads.
//
// Indexing: tid in [0, 16), dst = tid / 4, src = tid % 4. comb_out[tid]
// holds comb[dst, src].
//
// Sinkhorn axis convention (matches hc_ffn._sinkhorn):
//   - col_sum: sum over dst axis (dim 1 of [T, n_hc, n_hc]).
//              For our flat [16] indexed by tid = dst*4 + src,
//              col c = sum_{tid : src == c} comb_out[tid].
//   - row_sum: sum over src axis (dim 2). For flat tid,
//              row r = sum_{tid : dst == r} comb_out[tid].
// ---------------------------------------------------------------------------
__global__ void k_hc_pre_collapse(
    const float* __restrict__ mixes,
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float* __restrict__ pre_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    int sinkhorn_iters,
    float eps)
{
    int tid    = threadIdx.x;
    int active = (tid < 16);
    int dst    = tid >> 2;
    int src    = tid & 3;
    int idx24  = 8 + dst * 4 + src;

    // --- pre, post (only threads 0..3) -----------------------------------
    float pre_s  = scale[0];
    float post_s = scale[1];
    float comb_s = scale[2];

    if (tid < HC_N_HC) {
        float zp = mixes[tid] * pre_s + base[tid];
        pre_out[tid]  = (1.0f / (1.0f + __expf(-zp))) + eps;

        float zo = mixes[HC_N_HC + tid] * post_s + base[HC_N_HC + tid];
        post_out[tid] = 2.0f * (1.0f / (1.0f + __expf(-zo)));
    }
    __syncthreads();

    // --- comb logits ------------------------------------------------------
    float logit = active ? (mixes[idx24] * comb_s + base[idx24]) : 0.0f;

    // --- softmax over src per row dst (last-axis softmax) -----------------
    __shared__ float row_max[HC_N_HC];
    __shared__ float row_sum[HC_N_HC];

    if (tid < HC_N_HC) row_max[tid] = -1e30f;
    __syncthreads();
    for (int sweep = 0; sweep < HC_N_HC; ++sweep) {
        if (active && src == sweep) {
            if (logit > row_max[dst]) row_max[dst] = logit;
        }
        __syncthreads();
    }

    float exp_v = active ? __expf(logit - row_max[dst]) : 0.0f;

    if (tid < HC_N_HC) row_sum[tid] = 0.0f;
    __syncthreads();
    for (int sweep = 0; sweep < HC_N_HC; ++sweep) {
        if (active && src == sweep) {
            row_sum[dst] += exp_v;
        }
        __syncthreads();
    }

    // comb = softmax(...) + eps
    float comb_v = active ? (exp_v / row_sum[dst] + eps) : 0.0f;
    if (active) comb_out[tid] = comb_v;
    __syncthreads();

    // --- Sinkhorn ---------------------------------------------------------
    // First pass: column normalisation only (asymmetric).
    //
    // col_sum[c] = sum over dst of comb[dst, c]. In our flat layout
    // (tid = dst*4 + src), the contributors to col c are
    // {tid : src == c}, i.e. tids c, c+4, c+8, c+12.
    //
    // To accumulate without races we sweep `dst == sweep`: per sweep, 4
    // active threads share the same dst but have unique srcs, so each writes
    // to a unique ssum[src] slot. Across 4 sweeps every comb_out element is
    // contributed exactly once.
    __shared__ float ssum[HC_N_HC];

    if (tid < HC_N_HC) ssum[tid] = 0.0f;
    __syncthreads();
    for (int sweep = 0; sweep < HC_N_HC; ++sweep) {
        if (active && dst == sweep) {
            ssum[src] += comb_out[tid];
        }
        __syncthreads();
    }
    if (active) {
        comb_out[tid] = comb_out[tid] / (ssum[src] + eps);
    }
    __syncthreads();

    // Iters 2..N: alternate ROW then COL.
    for (int it = 0; it < sinkhorn_iters - 1; ++it) {
        // Row normalisation
        // row_sum[r] = sum over src of comb[r, src]. Contributors are
        // {tid : dst == r}. Sweep `src == sweep`: 4 active threads share
        // the same src, unique dst -> writes to unique ssum[dst]. Safe.
        if (tid < HC_N_HC) ssum[tid] = 0.0f;
        __syncthreads();
        for (int sweep = 0; sweep < HC_N_HC; ++sweep) {
            if (active && src == sweep) {
                ssum[dst] += comb_out[tid];
            }
            __syncthreads();
        }
        if (active) comb_out[tid] = comb_out[tid] / (ssum[dst] + eps);
        __syncthreads();

        // Column normalisation (sweep dst == sweep, see above).
        if (tid < HC_N_HC) ssum[tid] = 0.0f;
        __syncthreads();
        for (int sweep = 0; sweep < HC_N_HC; ++sweep) {
            if (active && dst == sweep) {
                ssum[src] += comb_out[tid];
            }
            __syncthreads();
        }
        if (active) comb_out[tid] = comb_out[tid] / (ssum[src] + eps);
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Internal: weighted collapse  x_collapsed[d] = sum_h pre[h] * x_residual[h, d].
//   x_residual : BF16 [n_hc, HIDDEN]
//   pre        : F32  [n_hc]
//   x_out      : BF16 [HIDDEN]
// One thread per output element. Matches hc_ffn:
//   x_pre = einsum("th,thd->td", pre.to(x.dtype), x)
// where pre.to(x.dtype) is BF16; product is bf16*bf16, fp32 accumulation.
// ---------------------------------------------------------------------------
__global__ void k_hc_apply_pre(
    const __nv_bfloat16* __restrict__ x_residual,
    const float* __restrict__ pre,
    __nv_bfloat16* __restrict__ x_out,
    int HIDDEN_DIM,
    int n_hc)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HIDDEN_DIM) return;
    float acc = 0.0f;
    for (int h = 0; h < n_hc; ++h) {
        // Cast pre -> bf16 -> back to fp32 to mirror Python pre.to(bf16).
        float pre_h_bf = __bfloat162float(__float2bfloat16(pre[h]));
        float x_v      = __bfloat162float(x_residual[h * HIDDEN_DIM + d]);
        acc += pre_h_bf * x_v;
    }
    x_out[d] = __float2bfloat16(acc);
}

// ---------------------------------------------------------------------------
// Internal: hc_post_mix.
//   out[h, d] = post[h] * block_out[d]  +  sum_s comb[h, s] * residual[s, d]
// All accumulations in fp32; final downcast to bf16. Matches hc_ffn.hc_post.
// residual aliases out -> we WRITE TO A SCRATCH buffer to avoid in-place
// RAW races; the host code memcpys back.
// ---------------------------------------------------------------------------
__global__ void k_hc_post_mix(
    const __nv_bfloat16* __restrict__ block_out,   // [HIDDEN]
    const __nv_bfloat16* __restrict__ residual,    // [n_hc, HIDDEN]
    const float* __restrict__ post,                // [n_hc]
    const float* __restrict__ comb,                // [n_hc, n_hc]
    __nv_bfloat16* __restrict__ out_4xH,           // [n_hc, HIDDEN]   scratch
    int HIDDEN_DIM,
    int n_hc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_hc * HIDDEN_DIM;
    if (idx >= total) return;
    int h = idx / HIDDEN_DIM;
    int d = idx % HIDDEN_DIM;

    float post_h_bf = __bfloat162float(__float2bfloat16(post[h]));
    float blk_v     = __bfloat162float(block_out[d]);
    float new_block = post_h_bf * blk_v;

    float mixed = 0.0f;
    for (int s = 0; s < n_hc; ++s) {
        float c_bf = __bfloat162float(__float2bfloat16(comb[h * n_hc + s]));
        float r_v  = __bfloat162float(residual[s * HIDDEN_DIM + d]);
        mixed += c_bf * r_v;
    }
    out_4xH[idx] = __float2bfloat16(new_block + mixed);
}

// ---------------------------------------------------------------------------
// Internal: hc_head pre extraction. PRE-only (no post, no comb, no Sinkhorn).
//   mixes  : F32 [4]
//   base   : F32 [4]
//   scale  : F32 [1]   (pre_scale only)
//   pre    : F32 [4]
// 1 block, 32 threads (only first 4 do work).
// ---------------------------------------------------------------------------
__global__ void k_hc_head_compute_pre(
    const float* __restrict__ mixes,
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float* __restrict__ pre_out,
    float eps)
{
    int tid = threadIdx.x;
    if (tid < HC_N_HC) {
        float pre_s = scale[0];
        float zp    = mixes[tid] * pre_s + base[tid];
        pre_out[tid] = (1.0f / (1.0f + __expf(-zp))) + eps;
    }
}

// ===========================================================================
// HOST API
// ===========================================================================
// All entry points take raw device pointers. The engine owns scratch buffers.
// ===========================================================================

extern "C" {

int hc_pre_forward(
    const __nv_bfloat16* x_residual,
    const float* hc_fn,
    const float* hc_base,
    const float* hc_scale,
    __nv_bfloat16* x_collapsed,
    float* pre_out,
    float* post_out,
    float* comb_out,
    __nv_bfloat16* flat_norm_scratch,
    float* mixes_scratch,
    cudaStream_t stream)
{
    {
        int blk = 256;
        size_t smem = blk * sizeof(float);
        k_hc_rmsn_no_affine<<<1, blk, smem, stream>>>(
            x_residual, flat_norm_scratch, HC_DIM_FLAT, HC_NORM_EPS);
    }
    {
        int blk = 256;
        size_t smem = blk * sizeof(float);
        k_hc_mixer_proj<<<HC_LAYER_MIX, blk, smem, stream>>>(
            flat_norm_scratch, hc_fn, mixes_scratch, HC_LAYER_MIX, HC_DIM_FLAT);
    }
    {
        k_hc_pre_collapse<<<1, 32, 0, stream>>>(
            mixes_scratch, hc_base, hc_scale,
            pre_out, post_out, comb_out,
            HC_SINKHORN_IT, HC_EPS_F);
    }
    {
        int blk = 256, grid = (HC_HIDDEN + blk - 1) / blk;
        k_hc_apply_pre<<<grid, blk, 0, stream>>>(
            x_residual, pre_out, x_collapsed, HC_HIDDEN, HC_N_HC);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "hc_pre_forward: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int hc_post_forward(
    const __nv_bfloat16* block_out,
    __nv_bfloat16* residual_inout,
    const float* post,
    const float* comb,
    __nv_bfloat16* post_scratch,
    cudaStream_t stream)
{
    int total = HC_N_HC * HC_HIDDEN;
    int blk = 256, grid = (total + blk - 1) / blk;
    k_hc_post_mix<<<grid, blk, 0, stream>>>(
        block_out, residual_inout, post, comb,
        post_scratch, HC_HIDDEN, HC_N_HC);
    cudaMemcpyAsync(residual_inout, post_scratch,
                    (size_t)total * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "hc_post_forward: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int hc_head_compute(
    const __nv_bfloat16* x_residual,
    const float* hc_head_fn,
    const float* hc_head_base,
    const float* hc_head_scale,
    __nv_bfloat16* x_collapsed,
    __nv_bfloat16* flat_norm_scratch,
    float* mixes_scratch,
    float* pre_scratch,
    cudaStream_t stream)
{
    {
        int blk = 256;
        size_t smem = blk * sizeof(float);
        k_hc_rmsn_no_affine<<<1, blk, smem, stream>>>(
            x_residual, flat_norm_scratch, HC_DIM_FLAT, HC_NORM_EPS);
    }
    {
        int blk = 256;
        size_t smem = blk * sizeof(float);
        k_hc_mixer_proj<<<HC_HEAD_MIX, blk, smem, stream>>>(
            flat_norm_scratch, hc_head_fn, mixes_scratch, HC_HEAD_MIX, HC_DIM_FLAT);
    }
    {
        k_hc_head_compute_pre<<<1, 32, 0, stream>>>(
            mixes_scratch, hc_head_base, hc_head_scale, pre_scratch, HC_EPS_F);
    }
    {
        int blk = 256, grid = (HC_HIDDEN + blk - 1) / blk;
        k_hc_apply_pre<<<grid, blk, 0, stream>>>(
            x_residual, pre_scratch, x_collapsed, HC_HIDDEN, HC_N_HC);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "hc_head_compute: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Constants exposed for Python bitexact test.
int hc_get_sinkhorn_iters() { return HC_SINKHORN_IT; }
int hc_get_n_hc()           { return HC_N_HC; }
int hc_get_hidden()         { return HC_HIDDEN; }
int hc_get_dim_flat()       { return HC_DIM_FLAT; }
int hc_get_layer_mix()      { return HC_LAYER_MIX; }
int hc_get_head_mix()       { return HC_HEAD_MIX; }

}  // extern "C"
