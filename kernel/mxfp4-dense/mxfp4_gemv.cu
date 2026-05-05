// mxfp4_gemv.cu — MXFP4 GEMV M=1 kernel for DSv4-Flash dense path (GB10 sm_121a).
// Project: MXFP4 native sprint
// Authors: Davide Zenati
// License: MIT
//
// Decode-step GEMV for attention layers and shared experts whose weights live
// in MXFP4 native format (OCP MX FP4, per-block 32-element E8M0 scale).
//
// Layout (matches OCP MX v1.0 + RedHatAI / vLLM packing convention):
//
//   W_packed : uint8 [N, K/2] row-major
//              byte = (elem[2*i+1] << 4) | (elem[2*i] & 0x0F)
//              low nibble = elem index 2*i, high nibble = elem index 2*i+1.
//   W_scale  : uint8 [N, K/32] row-major
//              E8M0 byte for each 32-element block: real_scale = 2^(byte-127).
//              byte == 0xFF is NaN sentinel (we treat as scale=0 to be safe).
//   x        : __nv_bfloat16 [K]
//   y        : __nv_bfloat16 [N]
//
// Computation:
//   y[n] = sum_k W[n,k] * x[k]
//   W[n,k] = FP4_codebook[ unpack_nibble(W_packed[n, k/2]) ] * 2^(W_scale[n, k/32]-127)
//
// FP4 E2M1 codebook (16 entries, indexed by raw nibble 0..15):
//   [+0, +0.5, +1, +1.5, +2, +3, +4, +6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]
//
// CTA strategy:
//   1 CTA = MXFP4_GEMV_BLOCK_N output rows (default 64).
//   1 thread = 1 output row (Q3 reference pattern preserved).
//   Threads cooperatively cache x[K] in shared memory (BF16) once per CTA.
//   FP4 codebook in __constant__ memory (16 floats, broadcast read).
//
// Bit-exact reference: runtime/mxfp4_unpack.py + runtime/e8m0_decode.py + numpy.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef MXFP4_GEMV_BLOCK_N
#define MXFP4_GEMV_BLOCK_N 32        // output rows per CTA
#endif

#ifndef MXFP4_GEMV_THREADS
#define MXFP4_GEMV_THREADS 32        // 1 thread = 1 row
#endif

#ifndef MXFP4_GEMV_MAX_K
#define MXFP4_GEMV_MAX_K 8192        // smem cap for activation cache (16 KB BF16)
#endif

#define MXFP4_BLOCK_ELEMS  32        // OCP MX block size
#define MXFP4_BIAS         127

// ─── FP4 E2M1 codebook in constant memory (broadcast read across warps) ────
// Layout matches FP4_E2M1_CODEBOOK in runtime/mxfp4_unpack.py exactly.
__constant__ float c_fp4_codebook[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// ─── E8M0 byte → fp32 scale ────────────────────────────────────────────────
// 0xFF (NaN sentinel in spec) treated as 0.0f to keep accumulator finite.
// All other bytes: 2^(byte - 127). Uses ldexpf for exact power-of-two.
__device__ __forceinline__ float e8m0_byte_to_scale(uint8_t b) {
    if (b == 0xFF) return 0.0f;                    // NaN sentinel → safe
    int exp = (int)b - MXFP4_BIAS;                 // signed exponent
    return ldexpf(1.0f, exp);                      // 2^exp exact
}

// ─── GEMV kernel: 1 CTA = MXFP4_GEMV_BLOCK_N rows slab ─────────────────────
__global__ void mxfp4_gemv_m1_kernel(
    const __nv_bfloat16* __restrict__ x,           // [K]
    const uint8_t*       __restrict__ W_packed,    // [N, K/2]
    const uint8_t*       __restrict__ W_scale,     // [N, K/32]
    int K,
    int N,
    int64_t packed_row_stride_bytes,               // typically K/2
    int64_t scale_row_stride_bytes,                // typically K/32
    __nv_bfloat16* __restrict__ y                  // [N]
) {
    const int row_base = blockIdx.x * MXFP4_GEMV_BLOCK_N;
    const int tid      = threadIdx.x;
    const int my_row   = row_base + tid;

    // ── Stage 1: cache x[K] in shared memory (BF16) ────────────────────────
    extern __shared__ __nv_bfloat16 smem_x[];
    for (int k = tid; k < K; k += MXFP4_GEMV_THREADS) {
        smem_x[k] = x[k];
    }
    __syncthreads();

    if (my_row >= N) return;
    if (tid >= MXFP4_GEMV_BLOCK_N) return;

    // ── Stage 2: walk row, decode 32-elem blocks, accumulate ───────────────
    const uint8_t* row_packed = W_packed + (int64_t)my_row * packed_row_stride_bytes;
    const uint8_t* row_scale  = W_scale  + (int64_t)my_row * scale_row_stride_bytes;
    const int n_blocks        = K / MXFP4_BLOCK_ELEMS;          // K must be %32==0
    const int bytes_per_block = MXFP4_BLOCK_ELEMS / 2;          // 16 bytes/block

    float acc = 0.0f;

    #pragma unroll 1
    for (int b = 0; b < n_blocks; ++b) {
        const float scale = e8m0_byte_to_scale(__ldg(row_scale + b));
        const uint8_t* blk_bytes = row_packed + (int64_t)b * bytes_per_block;
        const int k_base = b * MXFP4_BLOCK_ELEMS;

        // 16 bytes → 32 elems (low nibble = even idx, high nibble = odd idx)
        #pragma unroll
        for (int by = 0; by < bytes_per_block; ++by) {
            const uint8_t bb = __ldg(blk_bytes + by);
            const int nib_lo = (int)(bb & 0x0F);
            const int nib_hi = (int)((bb >> 4) & 0x0F);
            const float w_lo = c_fp4_codebook[nib_lo] * scale;
            const float w_hi = c_fp4_codebook[nib_hi] * scale;
            const int k_lo = k_base + 2 * by;
            const int k_hi = k_base + 2 * by + 1;
            const float x_lo = __bfloat162float(smem_x[k_lo]);
            const float x_hi = __bfloat162float(smem_x[k_hi]);
            acc = fmaf(x_lo, w_lo, acc);
            acc = fmaf(x_hi, w_hi, acc);
        }
    }

    y[my_row] = __float2bfloat16(acc);
}

// ─── Public C ABI ─────────────────────────────────────────────────────────
extern "C" int mxfp4_gemv_m1(
    const __nv_bfloat16* x,                        // device [K]
    const uint8_t* W_packed,                       // device [N, K/2]
    const uint8_t* W_scale,                        // device [N, K/32]
    int K,
    int N,
    int64_t packed_row_stride_bytes,
    int64_t scale_row_stride_bytes,
    __nv_bfloat16* y,                              // device [N]
    cudaStream_t stream
) {
    if (K % MXFP4_BLOCK_ELEMS != 0) {
        std::fprintf(stderr, "mxfp4_gemv_m1: K=%d not multiple of %d\n",
                     K, MXFP4_BLOCK_ELEMS);
        return -1;
    }
    if (K > MXFP4_GEMV_MAX_K) {
        std::fprintf(stderr, "mxfp4_gemv_m1: K=%d > MXFP4_GEMV_MAX_K=%d\n",
                     K, MXFP4_GEMV_MAX_K);
        return -2;
    }
    const int n_ctas = (N + MXFP4_GEMV_BLOCK_N - 1) / MXFP4_GEMV_BLOCK_N;
    const size_t smem_bytes = (size_t)K * sizeof(__nv_bfloat16);

    mxfp4_gemv_m1_kernel<<<n_ctas, MXFP4_GEMV_THREADS, smem_bytes, stream>>>(
        x, W_packed, W_scale, K, N,
        packed_row_stride_bytes, scale_row_stride_bytes, y
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "mxfp4_gemv_m1 launch failed: %s\n",
                     cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

// ─── Optional helper for tile-size sweep / smaller CTA variant ────────────
// Same kernel body, parametrized at compile time via -DMXFP4_GEMV_BLOCK_N=...
// at build time. Default 64. Sweeps via separate Makefile targets.
