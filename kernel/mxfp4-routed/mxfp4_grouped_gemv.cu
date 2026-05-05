// mxfp4_grouped_gemv.cu — Grouped MXFP4 GEMV for top-K=6 routed experts
//                          (DSv4-Flash, GB10 sm_121a Blackwell).
// Project: MXFP4 sprint /
// Authors: Davide Zenati
// License: MIT
//
// Native MXFP4 (OCP Microscaling Formats v1.0) routed-MoE forward for one decode token.
//
// Per-expert FFN (SwiGLU):
//     h_i  = silu(W1_i @ x)         // [hidden=2048]   gate proj
//     u_i  = (W3_i @ x)             // [hidden=2048]   up   proj
//     m_i  = h_i * u_i              // [hidden=2048]
//     y_i  = W2_i @ m_i             // [out=4096]      down proj
//     out += r_i * y_i              // FP32 acc, atomicAdd into out[4096]
//
// Two-stage launch (single C ABI entry):
//   Stage A: grid (hidden, n_exp) → SwiGLU intermediate M[n_exp, hidden] BF16
//   Stage B: grid (out,    n_exp) → down + atomicAdd weighted into FP32 accumulator
//   Cast:    FP32 → BF16 final output
//
// MXFP4 layout (OCP v1.0):
//   - Element: 4-bit FP4 E2M1 (sign:1 / exp:2 / mant:1) → codebook 16 entries
//     [+0,+0.5,+1,+1.5,+2,+3,+4,+6,-0,-0.5,-1,-1.5,-2,-3,-4,-6]
//   - Packing: 2 nibbles per byte. nibble_low = elem[2i], nibble_high = elem[2i+1]
//   - Scale: per-block E8M0 byte for every 32 contiguous elements
//     E8M0: byte ∈ [0,254] → scale = 2^(byte-127); byte=255 → NaN sentinel
//
// Per-row K=4096 reduction:
//   bytes_per_row = 4096/2 = 2048 packed bytes
//   scales_per_row = 4096/32 = 128 E8M0 bytes
//
// Per-row K=2048 reduction (down stage):
//   bytes_per_row = 1024 packed bytes
//   scales_per_row = 64 E8M0 bytes
//
// The activation is cached in shared memory once per CTA. The packed-weight
// stream is tiled across CTA threads in 32-element chunks (== one E8M0 block),
// so each thread handles ceil(N_blocks_per_row / blockDim) blocks.
//
// Reference (READ-ONLY architectural pattern, NOT cloned):
//   kernel/q3-routed/q3_k_grouped_gemv.cu  (Q3_K analog with 256-elem super-blocks)

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef MXFP4_GROUPED_K_EXPERTS
#define MXFP4_GROUPED_K_EXPERTS 6
#endif

#ifndef MXFP4_GROUPED_THREADS
#define MXFP4_GROUPED_THREADS 128
#endif

#ifndef MXFP4_GROUPED_MAX_RED
#define MXFP4_GROUPED_MAX_RED 4096   // smem activation cap (BF16 → 8 KB)
#endif

#define MXFP4_BLOCK_ELEMS 32          // OCP MX shared-scale granularity
#define MXFP4_BLOCK_BYTES 16          // 32 nibbles == 16 packed bytes

// ─── FP4 E2M1 codebook in constant memory ───────────────────────────────────
// Index by raw nibble value 0..15.
__constant__ float c_fp4_codebook[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ─── E8M0 byte → FP32 scale ────────────────────────────────────────────────
//   byte == 255 → NaN sentinel (treated as 0.0 here; should not appear in
//   well-formed weight tensors, but we degrade gracefully rather than NaN-poison).
__device__ __forceinline__ float e8m0_decode(uint8_t b) {
    if (b == 0xFF) return 0.0f;
    // 2^(b-127). exp2f handles negative exponents correctly.
    return exp2f((float)((int)b - 127));
}

// ─── Dequant + dot-product of one MXFP4 32-elem block against an x tile ─────
//
// packed: 16 bytes (32 nibbles)
// x_tile: 32 BF16 activations (already in smem)
// scale_byte: 1 E8M0 byte
//
// Returns the FP32 partial dot for this block.
__device__ __forceinline__ float mxfp4_block_dot(
    const uint8_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ x_tile,
    uint8_t scale_byte
) {
    const float s = e8m0_decode(scale_byte);
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < MXFP4_BLOCK_BYTES; ++i) {
        const uint8_t byte = packed[i];
        const int nlo = byte & 0x0F;
        const int nhi = (byte >> 4) & 0x0F;
        const float w0 = c_fp4_codebook[nlo];
        const float w1 = c_fp4_codebook[nhi];
        const float a0 = __bfloat162float(x_tile[2 * i + 0]);
        const float a1 = __bfloat162float(x_tile[2 * i + 1]);
        acc = fmaf(a0, w0, acc);
        acc = fmaf(a1, w1, acc);
    }
    return acc * s;
}

// ─── CTA-wide sum reduction (no warp shuffle dep — power-of-2 threads) ──────
__device__ __forceinline__ float cta_sum_f(float v, float* smem_red) {
    const int tid = threadIdx.x;
    smem_red[tid] = v;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) smem_red[tid] += smem_red[tid + off];
        __syncthreads();
    }
    return smem_red[0];
}

// ─── SiLU helper ────────────────────────────────────────────────────────────
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage A: SwiGLU(W1@x, W3@x) per (hidden_idx, expert_idx)
// ═══════════════════════════════════════════════════════════════════════════
//
// Grid:  (hidden, n_exp)        (e.g. 2048 × 6)
// Block: MXFP4_GROUPED_THREADS  (128)
//
// Each CTA computes M[exp, hidden_idx] = silu(W1_row · x) * (W3_row · x)
//
// Layouts (all device pointers):
//   W1_packed: [n_exp, hidden, K_in/2]              uint8
//   W3_packed: [n_exp, hidden, K_in/2]              uint8
//   W1_scale:  [n_exp, hidden, K_in/MXFP4_BLOCK_ELEMS]  uint8 (E8M0)
//   W3_scale:  [n_exp, hidden, K_in/MXFP4_BLOCK_ELEMS]  uint8 (E8M0)
//
//   Strides given in BYTES per (hidden) row and per expert.

__global__ void mxfp4_grouped_swiglu_kernel(
    const __nv_bfloat16* __restrict__ x,            // [K_in]
    const uint8_t* __restrict__ W1_packed,          // [n_exp, hidden, K_in/2]
    const uint8_t* __restrict__ W3_packed,
    const uint8_t* __restrict__ W1_scale,           // [n_exp, hidden, K_in/32]
    const uint8_t* __restrict__ W3_scale,
    int K_in,                                       // 4096
    int hidden,                                     // 2048
    int n_exp,                                      // 6
    int64_t pack_row_stride_bytes,                  // K_in/2
    int64_t pack_expert_stride_bytes,               // hidden * K_in/2
    int64_t scale_row_stride_bytes,                 // K_in/32
    int64_t scale_expert_stride_bytes,              // hidden * K_in/32
    __nv_bfloat16* __restrict__ M                   // [n_exp, hidden]
) {
    const int hidden_idx = blockIdx.x;
    const int exp_idx    = blockIdx.y;
    const int tid        = threadIdx.x;
    if (hidden_idx >= hidden || exp_idx >= n_exp) return;

    extern __shared__ float smem[];
    float* smem_red = smem;
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem + MXFP4_GROUPED_THREADS);

    // Cache activation in smem (shared across all CTAs computing same hidden_idx)
    for (int k = tid; k < K_in; k += MXFP4_GROUPED_THREADS) {
        smem_x[k] = x[k];
    }
    __syncthreads();

    const uint8_t* row_W1 = W1_packed
        + (int64_t)exp_idx * pack_expert_stride_bytes
        + (int64_t)hidden_idx * pack_row_stride_bytes;
    const uint8_t* row_W3 = W3_packed
        + (int64_t)exp_idx * pack_expert_stride_bytes
        + (int64_t)hidden_idx * pack_row_stride_bytes;
    const uint8_t* row_S1 = W1_scale
        + (int64_t)exp_idx * scale_expert_stride_bytes
        + (int64_t)hidden_idx * scale_row_stride_bytes;
    const uint8_t* row_S3 = W3_scale
        + (int64_t)exp_idx * scale_expert_stride_bytes
        + (int64_t)hidden_idx * scale_row_stride_bytes;

    const int n_blocks = K_in / MXFP4_BLOCK_ELEMS;   // 4096/32 = 128

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (int b = tid; b < n_blocks; b += MXFP4_GROUPED_THREADS) {
        const uint8_t* pk_g = row_W1 + b * MXFP4_BLOCK_BYTES;
        const uint8_t* pk_u = row_W3 + b * MXFP4_BLOCK_BYTES;
        const uint8_t* xt   = reinterpret_cast<const uint8_t*>(smem_x) + b * MXFP4_BLOCK_ELEMS * 2;
        const __nv_bfloat16* x_tile = reinterpret_cast<const __nv_bfloat16*>(xt);

        acc_g += mxfp4_block_dot(pk_g, x_tile, row_S1[b]);
        acc_u += mxfp4_block_dot(pk_u, x_tile, row_S3[b]);
    }

    // CTA reduction
    const float gate_v = cta_sum_f(acc_g, smem_red);
    __syncthreads();
    const float up_v   = cta_sum_f(acc_u, smem_red);

    if (tid == 0) {
        const float m = silu_f(gate_v) * up_v;
        M[(int64_t)exp_idx * hidden + hidden_idx] = __float2bfloat16(m);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage B: down_proj per expert + weighted reduction into FP32 accumulator
// ═══════════════════════════════════════════════════════════════════════════
//
// Grid:  (out_dim, n_exp)        (e.g. 4096 × 6)
// Block: MXFP4_GROUPED_THREADS   (128)
//
// Each CTA computes y_i[out_idx] = W2_row · m_i, scales by r_i,
// and atomicAdds into out_accum[out_idx] (FP32).
//
// Layouts:
//   W2_packed: [n_exp, out, hidden/2]              uint8
//   W2_scale:  [n_exp, out, hidden/32]             uint8 (E8M0)

__global__ void mxfp4_grouped_down_kernel(
    const __nv_bfloat16* __restrict__ M,            // [n_exp, hidden]
    const uint8_t* __restrict__ W2_packed,          // [n_exp, out, hidden/2]
    const uint8_t* __restrict__ W2_scale,           // [n_exp, out, hidden/32]
    const float*   __restrict__ rweights,           // [n_exp]
    int hidden,                                     // 2048
    int out_dim,                                    // 4096
    int n_exp,                                      // 6
    int64_t pack_row_stride_bytes,                  // hidden/2
    int64_t pack_expert_stride_bytes,               // out * hidden/2
    int64_t scale_row_stride_bytes,                 // hidden/32
    int64_t scale_expert_stride_bytes,              // out * hidden/32
    float* __restrict__ out_accum                   // [out_dim]
) {
    const int out_idx = blockIdx.x;
    const int exp_idx = blockIdx.y;
    const int tid     = threadIdx.x;
    if (out_idx >= out_dim || exp_idx >= n_exp) return;

    extern __shared__ float smem[];
    float* smem_red = smem;
    __nv_bfloat16* smem_m = reinterpret_cast<__nv_bfloat16*>(smem + MXFP4_GROUPED_THREADS);

    // Cache m_i in smem
    const __nv_bfloat16* m_ptr = M + (int64_t)exp_idx * hidden;
    for (int k = tid; k < hidden; k += MXFP4_GROUPED_THREADS) {
        smem_m[k] = m_ptr[k];
    }
    __syncthreads();

    const uint8_t* row_W = W2_packed
        + (int64_t)exp_idx * pack_expert_stride_bytes
        + (int64_t)out_idx * pack_row_stride_bytes;
    const uint8_t* row_S = W2_scale
        + (int64_t)exp_idx * scale_expert_stride_bytes
        + (int64_t)out_idx * scale_row_stride_bytes;

    const int n_blocks = hidden / MXFP4_BLOCK_ELEMS;   // 2048/32 = 64

    float acc = 0.0f;
    for (int b = tid; b < n_blocks; b += MXFP4_GROUPED_THREADS) {
        const uint8_t* pk = row_W + b * MXFP4_BLOCK_BYTES;
        const uint8_t* xt = reinterpret_cast<const uint8_t*>(smem_m) + b * MXFP4_BLOCK_ELEMS * 2;
        const __nv_bfloat16* m_tile = reinterpret_cast<const __nv_bfloat16*>(xt);
        acc += mxfp4_block_dot(pk, m_tile, row_S[b]);
    }

    const float y_v = cta_sum_f(acc, smem_red);
    if (tid == 0) {
        const float w_i = rweights[exp_idx];
        atomicAdd(&out_accum[out_idx], w_i * y_v);
    }
}

// ─── Cast FP32 accumulator → BF16 output ────────────────────────────────────
__global__ void cast_f32_to_bf16_kernel(
    const float* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = __float2bfloat16(in[idx]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Public C ABI
// ═══════════════════════════════════════════════════════════════════════════
extern "C" int mxfp4_grouped_gemv_topk(
    const __nv_bfloat16* x,                // device [K_in]
    const uint8_t* W1_packed,              // device [n_exp, hidden, K_in/2]
    const uint8_t* W3_packed,              // device [n_exp, hidden, K_in/2]
    const uint8_t* W2_packed,              // device [n_exp, out_dim, hidden/2]
    const uint8_t* W1_scale,               // device [n_exp, hidden, K_in/32]
    const uint8_t* W3_scale,               // device [n_exp, hidden, K_in/32]
    const uint8_t* W2_scale,               // device [n_exp, out_dim, hidden/32]
    const float*   rweights,               // device [n_exp]
    int K_in,                              // 4096
    int hidden,                            // 2048
    int out_dim,                           // 4096
    int n_exp,                             // 6
    __nv_bfloat16* out,                    // device [out_dim]
    void* workspace,                       // ≥ mxfp4_grouped_workspace_bytes()
    cudaStream_t stream
) {
    if (K_in % MXFP4_BLOCK_ELEMS != 0) {
        std::fprintf(stderr, "mxfp4 grouped: K_in=%d not multiple of %d\n",
                     K_in, MXFP4_BLOCK_ELEMS);
        return -1;
    }
    if (hidden % MXFP4_BLOCK_ELEMS != 0) {
        std::fprintf(stderr, "mxfp4 grouped: hidden=%d not multiple of %d\n",
                     hidden, MXFP4_BLOCK_ELEMS);
        return -2;
    }
    if (K_in > MXFP4_GROUPED_MAX_RED || hidden > MXFP4_GROUPED_MAX_RED) {
        std::fprintf(stderr, "mxfp4 grouped: K_in/hidden > %d (smem cap)\n",
                     MXFP4_GROUPED_MAX_RED);
        return -3;
    }
    if (n_exp != MXFP4_GROUPED_K_EXPERTS) {
        std::fprintf(stderr, "mxfp4 grouped: n_exp=%d != compiled-in %d\n",
                     n_exp, MXFP4_GROUPED_K_EXPERTS);
        return -4;
    }

    // Workspace partition: M[n_exp, hidden] BF16 + out_accum[out_dim] FP32
    __nv_bfloat16* M_buf = reinterpret_cast<__nv_bfloat16*>(workspace);
    float* out_accum = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(workspace)
        + (size_t)n_exp * hidden * sizeof(__nv_bfloat16)
    );

    cudaMemsetAsync(out_accum, 0, sizeof(float) * out_dim, stream);

    // Strides
    const int64_t pack_A_row    = (int64_t)(K_in / 2);
    const int64_t pack_A_expert = (int64_t)hidden * pack_A_row;
    const int64_t scale_A_row   = (int64_t)(K_in / MXFP4_BLOCK_ELEMS);
    const int64_t scale_A_expert= (int64_t)hidden * scale_A_row;

    const int64_t pack_B_row    = (int64_t)(hidden / 2);
    const int64_t pack_B_expert = (int64_t)out_dim * pack_B_row;
    const int64_t scale_B_row   = (int64_t)(hidden / MXFP4_BLOCK_ELEMS);
    const int64_t scale_B_expert= (int64_t)out_dim * scale_B_row;

    // Stage A
    {
        dim3 grid((unsigned)hidden, (unsigned)n_exp);
        dim3 block(MXFP4_GROUPED_THREADS);
        const size_t smem = sizeof(float) * MXFP4_GROUPED_THREADS
                          + sizeof(__nv_bfloat16) * (size_t)K_in;
        mxfp4_grouped_swiglu_kernel<<<grid, block, smem, stream>>>(
            x, W1_packed, W3_packed, W1_scale, W3_scale,
            K_in, hidden, n_exp,
            pack_A_row, pack_A_expert, scale_A_row, scale_A_expert,
            M_buf
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "stageA launch failed: %s\n", cudaGetErrorString(err));
            return -5;
        }
    }

    // Stage B
    {
        dim3 grid((unsigned)out_dim, (unsigned)n_exp);
        dim3 block(MXFP4_GROUPED_THREADS);
        const size_t smem = sizeof(float) * MXFP4_GROUPED_THREADS
                          + sizeof(__nv_bfloat16) * (size_t)hidden;
        mxfp4_grouped_down_kernel<<<grid, block, smem, stream>>>(
            M_buf, W2_packed, W2_scale, rweights,
            hidden, out_dim, n_exp,
            pack_B_row, pack_B_expert, scale_B_row, scale_B_expert,
            out_accum
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "stageB launch failed: %s\n", cudaGetErrorString(err));
            return -6;
        }
    }

    // Cast FP32 → BF16
    {
        const int threads = 256;
        const int blocks  = (out_dim + threads - 1) / threads;
        cast_f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(out_accum, out, out_dim);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cast launch failed: %s\n", cudaGetErrorString(err));
            return -7;
        }
    }

    return 0;
}

// Workspace size in bytes (caller-side preallocation).
extern "C" size_t mxfp4_grouped_workspace_bytes(int n_exp, int hidden, int out_dim) {
    return (size_t)n_exp * hidden * sizeof(__nv_bfloat16)
         + (size_t)out_dim * sizeof(float);
}
