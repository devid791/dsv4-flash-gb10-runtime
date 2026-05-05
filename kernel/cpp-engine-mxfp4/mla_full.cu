// mla_full.cu — MLA full attention for DSv4-Flash MXFP4 engine (C++ port).
// Project: // Authors: Davide Zenati
// License: MIT
// Target:  GB10 sm_121a (Blackwell, CUDA 13.2)
//
// Reference: runtime/dsv4_engine_mxfp4.py::_mla_attention()
//   - num_heads      = 64
//   - head_dim       = 512   (= q_lora pre-RMSN expanded)
//   - kv_lora_rank   = 512   (MQA: single shared K/V latent head)
//   - q_lora_rank    = 1024
//   - qk_rope_dim    = 64    (last 64 dims of Q-head and KV get RoPE)
//   - n_groups       = 8
//   - o_lora_rank    = 1024
//   - group_din      = 64*512/8 = 4096
//   - window_size    = 128   (sliding window + causal mask)
//   - rms_eps        = 1e-6
//
// All numerical paths follow Sprint 3 (commit 137ee5c) which had bit-exact
// MLA verified vs R2 Python on layer 0. Differences vs Sprint 3:
//   * Weights are FP8 E4M3 with E8M0 block-128 scale (same layout as
//     mxfp4 engine — wq_a, wq_b, wkv, wo_a, wo_b). The 5 GEMV calls go
//     through the existing libfp8_e4m3_gemv.so kernel, invoked from the
//     binding (ctypes from Python). This .cu owns ONLY the non-GEMV
//     kernels (RMSNorm, RoPE, scores, mask, softmax, attend) plus a
//     thin extern "C" entry point per kernel.
//   * MXFP4 routed/grouped GEMV (kernel B2) is NOT used here: MLA attention
//     adapters in mxfp4 engine are FP8, not MXFP4 (verified by reading
//     LayerMXFP4 attribute names wkv_w/wq_a_w/etc. all loaded via
//     _load_fp8_to_gpu).
//
// SCOPE TRUTH (D3, no fake claims):
//   * This file implements the per-step (M=1) decode forward of one MLA
//     layer. The orchestration (5 FP8 GEMV + the 7 kernels in this file
//     + sliced FP8 GEMV for grouped wo_a) is performed Python-side from
//     mla_full_binding.py. Bit-exact test compares orchestrated forward
//     to mxfp4 engine._mla_attention on layer 0, position 0, single token.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Compile-time constants (mirror runtime/dsv4_engine_mxfp4.py).
#define MLA_NUM_HEADS    64
#define MLA_HEAD_DIM     512
#define MLA_KV_LORA      512
#define MLA_QK_ROPE_DIM  64
#define MLA_N_GROUPS     8
#define MLA_O_LORA_RANK  1024
#define MLA_GROUP_DIN    4096   // NUM_HEADS * HEAD_DIM / N_GROUPS
#define MLA_WINDOW_SIZE  128
#define MLA_RMS_EPS      1e-6f

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

#define CK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[mla_full.cu] CUDA error %s at %s:%d : %s\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 : per-head Q RMSNorm
//   q[h, d] *= rsqrt(mean_d(q[h,:]**2) + eps)
//   Layout: q [NUM_HEADS, HEAD_DIM] BF16, in-place.
//   One CTA per head; block size = 256 (smem reduction over 512 dims).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_q_per_head_rmsn(
    __nv_bfloat16* __restrict__ q,
    int head_dim,
    float eps)
{
    extern __shared__ float ss_buf[];
    int h = blockIdx.x;
    int t = threadIdx.x;
    __nv_bfloat16* row = q + (int64_t)h * head_dim;

    float sumsq = 0.0f;
    for (int i = t; i < head_dim; i += blockDim.x) {
        float v = __bfloat162float(row[i]);
        sumsq += v * v;
    }
    ss_buf[t] = sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) ss_buf[t] += ss_buf[t + s];
        __syncthreads();
    }
    float inv = rsqrtf(ss_buf[0] / (float)head_dim + eps);
    for (int i = t; i < head_dim; i += blockDim.x) {
        float v = __bfloat162float(row[i]);
        row[i] = __float2bfloat16(v * inv);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2a : RoPE on tail rope_dim of each Q head
//   q[h, base + 2j]   = q[..2j]   * cos - q[..2j+1] * sin
//   q[h, base + 2j+1] = q[..2j]   * sin + q[..2j+1] * cos
//   freqs_real / freqs_imag : [max_seq, rope_dim/2] float32 (cos/sin).
//   inverse=1 ⇒ conjugate (sin → -sin); used for output dim derot.
//   One CTA per head, threads cover rope_dim/2 pairs.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_rope_apply_q(
    __nv_bfloat16* __restrict__ q,
    const float*   __restrict__ freqs_real,
    const float*   __restrict__ freqs_imag,
    int pos,
    int head_dim,
    int rope_dim,
    int rope_dim_half,
    int inverse)
{
    int h = blockIdx.x;
    int j = threadIdx.x;
    if (j >= rope_dim_half) return;
    int base = head_dim - rope_dim;
    __nv_bfloat16* row = q + (int64_t)h * head_dim;
    float c = freqs_real[(int64_t)pos * rope_dim_half + j];
    float s = freqs_imag[(int64_t)pos * rope_dim_half + j];
    if (inverse) s = -s;
    float a = __bfloat162float(row[base + 2*j]);
    float b = __bfloat162float(row[base + 2*j + 1]);
    float ar = a * c - b * s;
    float br = a * s + b * c;
    row[base + 2*j]     = __float2bfloat16(ar);
    row[base + 2*j + 1] = __float2bfloat16(br);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2b : RoPE on shared KV latent (single MQA head)
//   kv: [KV_LORA_RANK]  in-place. Same math as Q.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_rope_apply_kv(
    __nv_bfloat16* __restrict__ kv,
    const float*   __restrict__ freqs_real,
    const float*   __restrict__ freqs_imag,
    int pos,
    int kv_dim,
    int rope_dim,
    int rope_dim_half)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= rope_dim_half) return;
    int base = kv_dim - rope_dim;
    float c = freqs_real[(int64_t)pos * rope_dim_half + j];
    float s = freqs_imag[(int64_t)pos * rope_dim_half + j];
    float a = __bfloat162float(kv[base + 2*j]);
    float b = __bfloat162float(kv[base + 2*j + 1]);
    float ar = a * c - b * s;
    float br = a * s + b * c;
    kv[base + 2*j]     = __float2bfloat16(ar);
    kv[base + 2*j + 1] = __float2bfloat16(br);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2c : inverse RoPE on tail rope_dim of attention output (per head)
//   Equivalent to k_rope_apply_q(inverse=1).
//   Kept as separate symbol for readability / matching HF semantics.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_rope_inverse_o(
    __nv_bfloat16* __restrict__ o,
    const float*   __restrict__ freqs_real,
    const float*   __restrict__ freqs_imag,
    int pos,
    int head_dim,
    int rope_dim,
    int rope_dim_half)
{
    int h = blockIdx.x;
    int j = threadIdx.x;
    if (j >= rope_dim_half) return;
    int base = head_dim - rope_dim;
    __nv_bfloat16* row = o + (int64_t)h * head_dim;
    float c =  freqs_real[(int64_t)pos * rope_dim_half + j];
    float s = -freqs_imag[(int64_t)pos * rope_dim_half + j];   // conjugate
    float a = __bfloat162float(row[base + 2*j]);
    float b = __bfloat162float(row[base + 2*j + 1]);
    float ar = a * c - b * s;
    float br = a * s + b * c;
    row[base + 2*j]     = __float2bfloat16(ar);
    row[base + 2*j + 1] = __float2bfloat16(br);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3 : MLA attention scores per-head dot
//   scores[h, p] = (q[h,:] . kv_cache[p,:]) / sqrt(head_dim)
//   Layout: scores [NUM_HEADS, P+1]  float32 row-major (sink at scores[h, P]).
//   Grid = (P, NUM_HEADS), block = 256, two-stage warp reduction.
//   Sink slot is filled in the (p==0) thread per head with attn_sink[h].
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_mla_full_dot(
    const __nv_bfloat16* __restrict__ q,         // [NUM_HEADS, HEAD_DIM]
    const __nv_bfloat16* __restrict__ kv_cache,  // [P, KV_LORA_RANK]
    const float*         __restrict__ attn_sink, // [NUM_HEADS]
    int                              P,
    int                              head_dim,
    int                              kv_dim,
    float* __restrict__ scores)                  // [NUM_HEADS, P+1]
{
    int h = blockIdx.y;
    int p = blockIdx.x;
    int t = threadIdx.x;

    if (p < P) {
        float partial = 0.0f;
        const __nv_bfloat16* qh = q + (int64_t)h * head_dim;
        const __nv_bfloat16* kp = kv_cache + (int64_t)p * kv_dim;
        for (int d = t; d < head_dim; d += blockDim.x) {
            partial += __bfloat162float(qh[d]) * __bfloat162float(kp[d]);
        }
        // Block reduce (warp shuffle + warp_sum smem aggregation).
        __shared__ float warp_sum[8];
        unsigned mask = 0xFFFFFFFFu;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(mask, partial, off);
        int warp_id = t >> 5;
        int lane    = t & 31;
        if (lane == 0) warp_sum[warp_id] = partial;
        __syncthreads();
        if (warp_id == 0) {
            partial = (lane < (blockDim.x >> 5)) ? warp_sum[lane] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_down_sync(mask, partial, off);
            if (lane == 0) {
                scores[(int64_t)h * (P + 1) + p] = partial * rsqrtf((float)head_dim);
            }
        }
    }
    // One thread per head writes the sink logit (no scale applied — HF spec).
    if (p == 0 && t == 0) {
        scores[(int64_t)h * (P + 1) + P] = attn_sink[h];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4 : sliding window + causal mask
//   For decode (single token), q_pos = cur_pos = P-1.
//   Mask out p > q_pos (future) OR p < q_pos - WINDOW + 1 (outside window).
//   Sink slot at scores[h, P] is always kept.
//   One CTA per head, threads loop over p.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_mla_full_mask(
    float* __restrict__ scores,
    int P,
    int q_pos,
    int window)
{
    int h = blockIdx.x;
    int t = threadIdx.x;
    int low  = q_pos - window + 1;
    if (low < 0) low = 0;
    int high = q_pos;
    for (int p = t; p < P; p += blockDim.x) {
        if (p < low || p > high) {
            scores[(int64_t)h * (P + 1) + p] = -INFINITY;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5 : per-head softmax over [P+1] (sink at index P)
//   Standard subtract-max + exp + normalize. One CTA per head.
//   blockDim must be a power of 2; choose smallest >= P+1, clamp [32, 1024].
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_mla_full_softmax(
    float* __restrict__ scores,
    int P)
{
    extern __shared__ float smem[];
    int h = blockIdx.x;
    int t = threadIdx.x;
    int Pp1 = P + 1;
    float* row = scores + (int64_t)h * Pp1;

    // 1) max
    float m = -INFINITY;
    for (int i = t; i < Pp1; i += blockDim.x) {
        float v = row[i];
        if (v > m) m = v;
    }
    smem[t] = m;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s && smem[t + s] > smem[t]) smem[t] = smem[t + s];
        __syncthreads();
    }
    float maxv = smem[0];

    // 2) exp + sum
    float partial = 0.0f;
    for (int i = t; i < Pp1; i += blockDim.x) {
        float e = __expf(row[i] - maxv);
        row[i] = e;
        partial += e;
    }
    smem[t] = partial;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) smem[t] += smem[t + s];
        __syncthreads();
    }
    float sumv = smem[0];

    // 3) normalize
    for (int i = t; i < Pp1; i += blockDim.x) {
        row[i] /= sumv;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 6 : per-head attend (drop sink in V multiplication)
//   out[h, d] = sum_p attn_w[h, p] * kv_cache[p, d]   for p in [0, P).
//   Grid = (NUM_HEADS, ceil(head_dim/blk)), block = blk threads → covers d.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_mla_full_attend(
    const float*         __restrict__ attn_w,    // [NUM_HEADS, P+1]
    const __nv_bfloat16* __restrict__ kv_cache,  // [P, KV_LORA_RANK]
    int P,
    int head_dim,
    int kv_dim,
    __nv_bfloat16* __restrict__ out)
{
    int h = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (d >= head_dim) return;
    const float* aw = attn_w + (int64_t)h * (P + 1);
    float acc = 0.0f;
    for (int p = 0; p < P; ++p) {
        acc += aw[p] * __bfloat162float(kv_cache[(int64_t)p * kv_dim + d]);
    }
    out[(int64_t)h * head_dim + d] = __float2bfloat16(acc);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7 : KV cache append at position pos
//   Copies kv_buf [kv_dim] into kv_cache[pos, :].
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_kv_append(
    __nv_bfloat16* __restrict__ kv_cache,
    const __nv_bfloat16* __restrict__ kv_buf,
    int pos,
    int kv_dim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= kv_dim) return;
    kv_cache[(int64_t)pos * kv_dim + d] = kv_buf[d];
}

// ─────────────────────────────────────────────────────────────────────────────
// extern "C" entry points (one per kernel) for ctypes binding from Python.
// All take a CUstream as int64 (0 = current stream).
// Pointer arguments are raw device pointers (data_ptr() from torch tensors).
// ─────────────────────────────────────────────────────────────────────────────

extern "C" int mla_q_per_head_rmsn(
    __nv_bfloat16* q,     // [NUM_HEADS, HEAD_DIM] in/out
    int head_dim,
    float eps,
    cudaStream_t stream)
{
    int blk = 256;
    size_t smem = blk * sizeof(float);
    k_q_per_head_rmsn<<<MLA_NUM_HEADS, blk, smem, stream>>>(q, head_dim, eps);
    CK(cudaGetLastError());
    return 0;
}

extern "C" int mla_rope_apply_q(
    __nv_bfloat16* q,
    const float* freqs_real,
    const float* freqs_imag,
    int pos,
    int head_dim,
    int rope_dim,
    int inverse,
    cudaStream_t stream)
{
    int rope_dim_half = rope_dim / 2;
    k_rope_apply_q<<<MLA_NUM_HEADS, rope_dim_half, 0, stream>>>(
        q, freqs_real, freqs_imag, pos, head_dim, rope_dim, rope_dim_half, inverse);
    CK(cudaGetLastError());
    return 0;
}

extern "C" int mla_rope_apply_kv(
    __nv_bfloat16* kv,
    const float* freqs_real,
    const float* freqs_imag,
    int pos,
    int kv_dim,
    int rope_dim,
    cudaStream_t stream)
{
    int rope_dim_half = rope_dim / 2;
    k_rope_apply_kv<<<1, rope_dim_half, 0, stream>>>(
        kv, freqs_real, freqs_imag, pos, kv_dim, rope_dim, rope_dim_half);
    CK(cudaGetLastError());
    return 0;
}

extern "C" int mla_rope_inverse_o(
    __nv_bfloat16* o,
    const float* freqs_real,
    const float* freqs_imag,
    int pos,
    int head_dim,
    int rope_dim,
    cudaStream_t stream)
{
    int rope_dim_half = rope_dim / 2;
    k_rope_inverse_o<<<MLA_NUM_HEADS, rope_dim_half, 0, stream>>>(
        o, freqs_real, freqs_imag, pos, head_dim, rope_dim, rope_dim_half);
    CK(cudaGetLastError());
    return 0;
}

extern "C" int mla_kv_append(
    __nv_bfloat16* kv_cache,
    const __nv_bfloat16* kv_buf,
    int pos,
    int kv_dim,
    cudaStream_t stream)
{
    int blk = 256, grid = (kv_dim + blk - 1) / blk;
    k_kv_append<<<grid, blk, 0, stream>>>(kv_cache, kv_buf, pos, kv_dim);
    CK(cudaGetLastError());
    return 0;
}

extern "C" int mla_full_dot(
    const __nv_bfloat16* q,
    const __nv_bfloat16* kv_cache,
    const float* attn_sink,
    int P,
    int head_dim,
    int kv_dim,
    float* scores,
    cudaStream_t stream)
{
    dim3 grid((unsigned)P, (unsigned)MLA_NUM_HEADS);
    int blk = 256;
    k_mla_full_dot<<<grid, blk, 0, stream>>>(
        q, kv_cache, attn_sink, P, head_dim, kv_dim, scores);
    CK(cudaGetLastError());
    return 0;
}

extern "C" int mla_full_mask(
    float* scores,
    int P,
    int q_pos,
    int window,
    cudaStream_t stream)
{
    int blk = 256;
    k_mla_full_mask<<<MLA_NUM_HEADS, blk, 0, stream>>>(scores, P, q_pos, window);
    CK(cudaGetLastError());
    return 0;
}

extern "C" int mla_full_softmax(
    float* scores,
    int P,
    cudaStream_t stream)
{
    int Pp1 = P + 1;
    // Choose smallest power-of-2 blk >= Pp1, clamp to [32, 1024].
    int blk = 1;
    while (blk < Pp1) blk <<= 1;
    if (blk < 32)   blk = 32;
    if (blk > 1024) blk = 1024;
    size_t smem = (size_t)blk * sizeof(float);
    k_mla_full_softmax<<<MLA_NUM_HEADS, blk, smem, stream>>>(scores, P);
    CK(cudaGetLastError());
    return 0;
}

extern "C" int mla_full_attend(
    const float* attn_w,
    const __nv_bfloat16* kv_cache,
    int P,
    int head_dim,
    int kv_dim,
    __nv_bfloat16* out,
    cudaStream_t stream)
{
    int blk = 256;
    int grid_d = (head_dim + blk - 1) / blk;
    dim3 grid((unsigned)MLA_NUM_HEADS, (unsigned)grid_d);
    k_mla_full_attend<<<grid, blk, 0, stream>>>(
        attn_w, kv_cache, P, head_dim, kv_dim, out);
    CK(cudaGetLastError());
    return 0;
}

// Compile-time sanity check.
static_assert(MLA_GROUP_DIN == MLA_NUM_HEADS * MLA_HEAD_DIM / MLA_N_GROUPS,
              "GROUP_DIN must equal NUM_HEADS*HEAD_DIM/N_GROUPS");
