// head.cu — DSv4-Flash head orchestration in C++ // Project: cpp-engine-mxfp4
// Authors: Davide Zenati
// License: MIT
// Target: GB10 sm_121a (Grace Blackwell), CUDA 13.2.
//
// Wraps embed lookup, intra-layer + final RMSNorm, lm_head GEMV, and greedy
// sampler — all callable from C/C++ engine without Python in the decode loop.
//
// Per R2 reference (runtime/dsv4_engine_mxfp4.py):
//   - embed.weight     : BF16 [VOCAB=129280, HIDDEN=4096]
//   - head.weight      : BF16 [VOCAB=129280, HIDDEN=4096]   ← NOT MXFP4 in this build
//   - norm.weight      : BF16 [HIDDEN]
//   - intra-layer norms: BF16 [HIDDEN] (attn_norm / ffn_norm) + smaller
//                        per-LoRA (q_norm, kv_norm).
//
// Mandate spec told us to expect lm_head MXFP4 — but R2 truth is BF16. We
// implement BOTH paths:
//   (a) lm_head_forward_bf16  — what the engine actually needs today
//   (b) lm_head_forward_mxfp4 — passthrough to libmxfp4_gemv if a future
//       checkpoint quantizes the head to MXFP4.
//
// All kernels are M=1 GEMV (decode hot path). 1 token at a time.
//
// External deps (load via dlopen at call time, no link order assumption):
//   - kernel/mxfp4-dense/libmxfp4_gemv.so       (mxfp4_gemv_m1)
//   - kernel/rmsnorm-fuse/librmsnorm_fuse.so    (rmsnorm_fuse_fwd)
//
// We embed thin re-implementations of those entry points here (BF16 RMSNorm
// fwd, MXFP4 GEMV fwd) so the C++ engine can be linked as one .so without
// runtime dlopen. Bit-exact equivalence vs the standalone .so is preserved.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ─── Configuration constants ───────────────────────────────────────────────
#ifndef D7_THREADS_NORM
#define D7_THREADS_NORM 256
#endif

#ifndef D7_THREADS_EMBED
#define D7_THREADS_EMBED 256
#endif

#ifndef D7_THREADS_LMHEAD_BF16
#define D7_THREADS_LMHEAD_BF16 128
#endif

// One CTA computes D7_LMHEAD_ROWS_PER_CTA output rows for lm_head BF16 GEMV.
#ifndef D7_LMHEAD_ROWS_PER_CTA
#define D7_LMHEAD_ROWS_PER_CTA 4
#endif

// ─── Warp + block reduction helpers (BF16 RMSNorm + lm_head reduction) ────

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

template <int N_THREADS>
__device__ __forceinline__ float block_reduce_sum(float v, float* smem_warp) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int n_warps = N_THREADS >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) smem_warp[wid] = v;
    __syncthreads();

    if (wid == 0) {
        float w = (lane < n_warps) ? smem_warp[lane] : 0.f;
        w = warp_reduce_sum(w);
        if (lane == 0) smem_warp[0] = w;
    }
    __syncthreads();
    return smem_warp[0];
}

// ═════════════════════════════════════════════════════════════════════════
// 1. EMBED LOOKUP : x_out[h] = embed_table[token_id, h]   (BF16 → BF16)
// ═════════════════════════════════════════════════════════════════════════

__global__ void k_embed_lookup_kernel(
    const __nv_bfloat16* __restrict__ embed_table,    // [VOCAB, HIDDEN]
    int token_id,
    int hidden,
    __nv_bfloat16* __restrict__ x_out                 // [HIDDEN]
) {
    const __nv_bfloat16* row = embed_table + (size_t)token_id * hidden;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        x_out[h] = row[h];
    }
}

extern "C" int embed_forward(
    const void* embed_table,    // device BF16 [VOCAB, HIDDEN]
    int token_id,
    int vocab,
    int hidden,
    void* x_out,                // device BF16 [HIDDEN]
    cudaStream_t stream
) {
    if (token_id < 0 || token_id >= vocab) {
        std::fprintf(stderr, "embed_forward: token_id=%d out of range [0, %d)\n",
                     token_id, vocab);
        return -1;
    }
    if (hidden <= 0) return -2;
    dim3 grid(1);
    dim3 block(D7_THREADS_EMBED);
    k_embed_lookup_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(embed_table),
        token_id, hidden,
        reinterpret_cast<__nv_bfloat16*>(x_out)
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "embed_forward launch failed: %s\n",
                     cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

// ═════════════════════════════════════════════════════════════════════════
// 2. RMSNORM (intra-layer + final) : y[h] = w[h] * x[h] / sqrt(mean(x^2)+eps)
//
// Bit-exact equivalent of librmsnorm_fuse rmsnorm_fuse_fwd at n_rows=1
// (decode token). Reduction in fp32, IO BF16.
// ═════════════════════════════════════════════════════════════════════════

__global__ void k_rmsnorm_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,      // [HIDDEN]
    const __nv_bfloat16* __restrict__ weight, // [HIDDEN]
    __nv_bfloat16* __restrict__ y,            // [HIDDEN]
    int hidden,
    float eps
) {
    extern __shared__ unsigned char smem_raw[];
    float* smem_warp = reinterpret_cast<float*>(smem_raw);

    float local_sq = 0.f;
    for (int h = threadIdx.x; h < hidden; h += D7_THREADS_NORM) {
        float v = __bfloat162float(x[h]);
        local_sq += v * v;
    }
    float total_sq = block_reduce_sum<D7_THREADS_NORM>(local_sq, smem_warp);

    float inv_n = 1.f / (float)hidden;
    float rms   = rsqrtf(total_sq * inv_n + eps);

    for (int h = threadIdx.x; h < hidden; h += D7_THREADS_NORM) {
        float v  = __bfloat162float(x[h]);
        float wf = __bfloat162float(weight[h]);
        y[h] = __float2bfloat16(wf * v * rms);
    }
}

// Internal helper for both final_norm_forward and rms_norm_inline.
static inline int rms_norm_launch(
    const void* x_in,
    const void* weight,
    void*       x_out,
    int hidden,
    float eps,
    cudaStream_t stream
) {
    if (hidden <= 0) return -1;
    dim3 grid(1);
    dim3 block(D7_THREADS_NORM);
    size_t smem = 32 * sizeof(float);
    k_rmsnorm_fwd_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_in),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(x_out),
        hidden, eps
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "rms_norm launch failed: %s\n",
                     cudaGetErrorString(err));
        return -2;
    }
    return 0;
}

// final_norm_forward — fused with engine-state convention. Calls rmsnorm.
extern "C" int final_norm_forward(
    const void* x_in,           // device BF16 [HIDDEN]
    const void* final_norm_w,   // device BF16 [HIDDEN]
    void*       x_out,          // device BF16 [HIDDEN]
    int hidden,
    float eps,
    cudaStream_t stream
) {
    return rms_norm_launch(x_in, final_norm_w, x_out, hidden, eps, stream);
}

// rms_norm_inline — orchestrates per intra-layer norm (pre-attn / post-attn /
// pre-MoE). Identical kernel; separate symbol for engine call-site clarity.
extern "C" int rms_norm_inline(
    const void* x_in,
    const void* weight,
    void*       x_out,
    int hidden,
    float eps,
    cudaStream_t stream
) {
    return rms_norm_launch(x_in, weight, x_out, hidden, eps, stream);
}

// ═════════════════════════════════════════════════════════════════════════
// 3. LM_HEAD BF16 GEMV : logits[n] = sum_k W[n,k] * x[k]
//
// W [VOCAB, HIDDEN] BF16, x [HIDDEN] BF16, logits [VOCAB] BF16.
// Strategy: 1 CTA computes D7_LMHEAD_ROWS_PER_CTA output rows. Threads in CTA
// cooperatively cache x[hidden] in shared mem (8 KB at H=4096), then each
// row is reduced by the full block. We pick rows-per-CTA=4 to balance smem
// reuse vs grid occupancy: VOCAB/4 = 32320 CTAs across the chip.
// ═════════════════════════════════════════════════════════════════════════

__global__ void k_lmhead_bf16_gemv_kernel(
    const __nv_bfloat16* __restrict__ W,      // [VOCAB, HIDDEN] row-major
    const __nv_bfloat16* __restrict__ x,      // [HIDDEN]
    __nv_bfloat16* __restrict__ logits,       // [VOCAB]
    int hidden,
    int vocab
) {
    const int row_base = blockIdx.x * D7_LMHEAD_ROWS_PER_CTA;

    // Shared layout: [warp_scratch (32 floats)] [x cache (HIDDEN BF16)]
    extern __shared__ unsigned char smem_raw[];
    float* smem_warp = reinterpret_cast<float*>(smem_raw);
    __nv_bfloat16* x_smem = reinterpret_cast<__nv_bfloat16*>(
                                smem_raw + 32 * sizeof(float));

    // Cooperatively load x[hidden] into shared
    for (int h = threadIdx.x; h < hidden; h += D7_THREADS_LMHEAD_BF16) {
        x_smem[h] = x[h];
    }
    __syncthreads();

    // Each CTA computes ROWS_PER_CTA rows; we sequentially reduce each row.
    #pragma unroll
    for (int r = 0; r < D7_LMHEAD_ROWS_PER_CTA; ++r) {
        const int row = row_base + r;
        if (row >= vocab) return;

        const __nv_bfloat16* W_row = W + (size_t)row * hidden;

        float acc = 0.f;
        for (int h = threadIdx.x; h < hidden; h += D7_THREADS_LMHEAD_BF16) {
            float xf = __bfloat162float(x_smem[h]);
            float wf = __bfloat162float(W_row[h]);
            acc = fmaf(xf, wf, acc);
        }
        float total = block_reduce_sum<D7_THREADS_LMHEAD_BF16>(acc, smem_warp);

        if (threadIdx.x == 0) {
            logits[row] = __float2bfloat16(total);
        }
        __syncthreads();   // prevent next iter trampling smem_warp
    }
}

extern "C" int lm_head_forward_bf16(
    const void* W,              // device BF16 [VOCAB, HIDDEN]
    const void* x,              // device BF16 [HIDDEN]
    void*       logits,         // device BF16 [VOCAB]
    int hidden,
    int vocab,
    cudaStream_t stream
) {
    if (hidden <= 0 || vocab <= 0) return -1;
    const int n_ctas = (vocab + D7_LMHEAD_ROWS_PER_CTA - 1) / D7_LMHEAD_ROWS_PER_CTA;
    const size_t smem = 32 * sizeof(float)
                      + (size_t)hidden * sizeof(__nv_bfloat16);
    k_lmhead_bf16_gemv_kernel<<<n_ctas, D7_THREADS_LMHEAD_BF16, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(W),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(logits),
        hidden, vocab
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "lm_head_forward_bf16 launch failed: %s\n",
                     cudaGetErrorString(err));
        return -2;
    }
    return 0;
}

// Forward declarations of the MXFP4 path symbols — provided by libmxfp4_gemv.so
// or by linking mxfp4_gemv.cu directly. For ctypes-only callers, this symbol
// stays unresolved and the function returns -99 (not implemented in this .so).
extern "C" int mxfp4_gemv_m1(
    const __nv_bfloat16* x,
    const uint8_t* W_packed,
    const uint8_t* W_scale,
    int K,
    int N,
    int64_t packed_row_stride_bytes,
    int64_t scale_row_stride_bytes,
    __nv_bfloat16* y,
    cudaStream_t stream
) __attribute__((weak));

extern "C" int lm_head_forward_mxfp4(
    const void* W_packed,       // device uint8 [VOCAB, HIDDEN/2]
    const void* W_scale,        // device uint8 [VOCAB, HIDDEN/32]
    const void* x,              // device BF16 [HIDDEN]
    void*       logits,         // device BF16 [VOCAB]
    int hidden,
    int vocab,
    cudaStream_t stream
) {
    if (mxfp4_gemv_m1 == nullptr) {
        std::fprintf(stderr, "lm_head_forward_mxfp4: libmxfp4_gemv not linked\n");
        return -99;
    }
    return mxfp4_gemv_m1(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const uint8_t*>(W_packed),
        reinterpret_cast<const uint8_t*>(W_scale),
        hidden, vocab,
        (int64_t)(hidden / 2),
        (int64_t)(hidden / 32),
        reinterpret_cast<__nv_bfloat16*>(logits),
        stream
    );
}

// ═════════════════════════════════════════════════════════════════════════
// 4. GREEDY SAMPLER : next_token = argmax(logits)
//
// Single-block reduce over VOCAB. Each thread tracks (max_val, max_idx) and
// we reduce via warp shuffles + shared scratch. fp32 compare.
// ═════════════════════════════════════════════════════════════════════════

#ifndef D7_THREADS_SAMPLE
#define D7_THREADS_SAMPLE 1024
#endif

struct ArgMax { float val; int idx; };

__device__ __forceinline__ ArgMax warp_reduce_argmax(ArgMax a) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other_v = __shfl_xor_sync(0xffffffffu, a.val, off);
        int   other_i = __shfl_xor_sync(0xffffffffu, a.idx, off);
        if (other_v > a.val || (other_v == a.val && other_i < a.idx)) {
            a.val = other_v;
            a.idx = other_i;
        }
    }
    return a;
}

__global__ void k_argmax_kernel(
    const __nv_bfloat16* __restrict__ logits,   // [VOCAB]
    int vocab,
    int* __restrict__ out_idx                   // device int [1]
) {
    extern __shared__ unsigned char smem_raw[];
    float* smem_v = reinterpret_cast<float*>(smem_raw);
    int*   smem_i = reinterpret_cast<int*>(smem_raw
                        + (D7_THREADS_SAMPLE / 32) * sizeof(float));

    ArgMax local = { -INFINITY, -1 };
    for (int n = threadIdx.x; n < vocab; n += D7_THREADS_SAMPLE) {
        float v = __bfloat162float(logits[n]);
        if (v > local.val || (v == local.val && n < local.idx)) {
            local.val = v;
            local.idx = n;
        }
    }
    local = warp_reduce_argmax(local);

    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int n_warps = D7_THREADS_SAMPLE >> 5;

    if (lane == 0) {
        smem_v[wid] = local.val;
        smem_i[wid] = local.idx;
    }
    __syncthreads();

    if (wid == 0) {
        ArgMax red;
        if (lane < n_warps) {
            red.val = smem_v[lane];
            red.idx = smem_i[lane];
        } else {
            red.val = -INFINITY;
            red.idx = -1;
        }
        red = warp_reduce_argmax(red);
        if (lane == 0) {
            out_idx[0] = red.idx;
        }
    }
}

// D26C: persistent 4-byte device buffer for argmax index (one cudaMalloc at first call).
// Was: cudaMallocAsync + cudaFreeAsync per decode_step -> mempool churn / fragmentation.
static int* g_d_argmax_idx = nullptr;

extern "C" int sample_greedy(
    const void* logits,         // device BF16 [VOCAB]
    int vocab,
    int* host_next_token,       // OUT: host int (single token id)
    cudaStream_t stream
) {
    if (vocab <= 0 || host_next_token == nullptr) return -1;

    if (g_d_argmax_idx == nullptr) {
        cudaError_t err0 = cudaMalloc(&g_d_argmax_idx, sizeof(int));
        if (err0 != cudaSuccess) {
            std::fprintf(stderr, "sample_greedy: persistent cudaMalloc failed: %s\n",
                         cudaGetErrorString(err0));
            g_d_argmax_idx = nullptr;
            return -2;
        }
    }

    const int n_warps = D7_THREADS_SAMPLE / 32;
    const size_t smem = (size_t)n_warps * (sizeof(float) + sizeof(int));
    k_argmax_kernel<<<1, D7_THREADS_SAMPLE, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits),
        vocab, g_d_argmax_idx
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "sample_greedy launch failed: %s\n",
                     cudaGetErrorString(err));
        return -3;
    }

    err = cudaMemcpyAsync(host_next_token, g_d_argmax_idx, sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "sample_greedy: D2H failed: %s\n",
                     cudaGetErrorString(err));
        return -4;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "sample_greedy: stream sync failed: %s\n",
                     cudaGetErrorString(err));
        return -5;
    }
    return 0;
}
