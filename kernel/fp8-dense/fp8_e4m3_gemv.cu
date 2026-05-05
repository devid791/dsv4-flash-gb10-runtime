// fp8_e4m3_gemv.cu — FP8 E4M3 GEMV M=1 kernel for DSv4-Flash dense path (GB10 sm_121a).
// Project: Native FP8 dense sprint
// Authors: Davide Zenati
// License: MIT
//
// Decode-step GEMV for attention layers (wq_a/wq_b/wkv/wo_a/wo_b) and
// shared-expert weights (sh_w1/sh_w2/sh_w3) whose weights live in
// FP8 E4M3 with a 2-D 128x128 E8M0 block-quant scale (DeepSeek-V4 layout).
//
// Layout (verified empirically on layers.0.attn.wq_a, .wkv, .wo_b,
// layers.0.ffn.shared_experts.w1/2/3):
//
//   W_fp8    : uint8 [N, K] row-major (raw E4M3 storage, 1 byte / elem).
//   W_scale  : uint8 [N/128, K/128] row-major (E8M0 byte / 128x128 tile).
//              real_scale = 2^(byte - 127). 0xFF is NaN sentinel
//              (treated as 0 here for safety).
//   x        : __nv_bfloat16 [K]
//   y        : __nv_bfloat16 [N]
//
// Computation:
//   y[n] = sum_k W_real[n,k] * x[k]
//   W_real[n,k] = fp8_e4m3_to_fp32(W_fp8[n,k]) * 2^(W_scale[n/128, k/128] - 127)
//
// CTA strategy (mirrors the B2 MXFP4 dense kernel pattern):
//   1 CTA = FP8_GEMV_BLOCK_N output rows (default 32). 1 thread = 1 row.
//   Threads cooperatively cache x[K] in shared memory (BF16) once per CTA.
//   FP8 dequant via the hardware __nv_cvt_fp8_to_halfraw intrinsic
//   (Blackwell sm_121a has native E4M3 conversion).
//
// Bit-exact reference: fp8_e4m3_gemv_binding.py uses torch.float8_e4m3fn
// view + scale broadcast + matmul as the ground-truth path.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#ifndef FP8_GEMV_BLOCK_N
#define FP8_GEMV_BLOCK_N 32          // output rows per CTA
#endif

#ifndef FP8_GEMV_THREADS
#define FP8_GEMV_THREADS 32          // 1 thread = 1 row in the slab
#endif

#ifndef FP8_GEMV_MAX_K
#define FP8_GEMV_MAX_K 16384         // smem cap for activation cache (32 KB BF16)
#endif

#define FP8_BLOCK_TILE   128         // DeepSeek-V4 2-D block-quant tile (rows & cols)
#define FP8_E8M0_BIAS    127

// ─── E8M0 byte → fp32 scale ───────────────────────────────────────────────
// 0xFF (NaN sentinel in OCP MX spec) → 0.0f (keeps accumulators finite if
// we ever hit a sentinel scale; checkpoint loading should never produce one).
__device__ __forceinline__ float e8m0_byte_to_scale(uint8_t b) {
    if (b == 0xFF) return 0.0f;
    int exp = (int)b - FP8_E8M0_BIAS;
    return ldexpf(1.0f, exp);
}

// ─── FP8 E4M3 byte → fp32 (hardware path) ─────────────────────────────────
// Blackwell sm_121a exposes __nv_cvt_fp8_to_halfraw which decodes one E4M3
// byte to a __half_raw. We then cast through __half → float.
__device__ __forceinline__ float fp8_e4m3_byte_to_fp32(uint8_t b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    __half h = hr;
    return __half2float(h);
}

// ─── GEMV kernel: 1 CTA = FP8_GEMV_BLOCK_N rows slab ──────────────────────
__global__ void fp8_e4m3_gemv_m1_kernel(
    const __nv_bfloat16* __restrict__ x,           // [K]
    const uint8_t*       __restrict__ W_fp8,       // [N, K]
    const uint8_t*       __restrict__ W_scale,     // [N/128, K/128]
    int K,
    int N,
    int64_t weight_row_stride_bytes,               // bytes per row in W_fp8 (typ. K)
    int64_t scale_row_stride_bytes,                // bytes per row in W_scale (typ. K/128)
    __nv_bfloat16* __restrict__ y                  // [N]
) {
    const int row_base = blockIdx.x * FP8_GEMV_BLOCK_N;
    const int tid      = threadIdx.x;
    const int my_row   = row_base + tid;

    // ── Stage 1: cache x[K] in shared memory (BF16) ────────────────────────
    extern __shared__ __nv_bfloat16 smem_x[];
    for (int k = tid; k < K; k += FP8_GEMV_THREADS) {
        smem_x[k] = x[k];
    }
    __syncthreads();

    if (my_row >= N) return;
    if (tid >= FP8_GEMV_BLOCK_N) return;

    // ── Stage 2: walk row, decode 128-col tiles, accumulate ───────────────
    // The scale row index = my_row / 128. Inside that scale row, the column
    // index advances every 128 K-elements.
    const int scale_row_idx = my_row / FP8_BLOCK_TILE;
    const uint8_t* row_w     = W_fp8   + (int64_t)my_row        * weight_row_stride_bytes;
    const uint8_t* row_scale = W_scale + (int64_t)scale_row_idx * scale_row_stride_bytes;

    const int n_col_tiles = K / FP8_BLOCK_TILE;        // K assumed % 128 == 0

    float acc = 0.0f;

    #pragma unroll 1
    for (int t = 0; t < n_col_tiles; ++t) {
        const float scale = e8m0_byte_to_scale(__ldg(row_scale + t));
        const int   k0    = t * FP8_BLOCK_TILE;

        // 128 elements per tile, unrolled in chunks of 8 for register reuse.
        #pragma unroll
        for (int j = 0; j < FP8_BLOCK_TILE; j += 8) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int   k_idx = k0 + j + u;
                const uint8_t bb  = __ldg(row_w + k_idx);
                const float w_val = fp8_e4m3_byte_to_fp32(bb) * scale;
                const float x_val = __bfloat162float(smem_x[k_idx]);
                acc = fmaf(x_val, w_val, acc);
            }
        }
    }

    y[my_row] = __float2bfloat16(acc);
}

// ─── Public C ABI ─────────────────────────────────────────────────────────
extern "C" int fp8_e4m3_gemv_m1(
    const __nv_bfloat16* x,                        // device [K]
    const uint8_t* W_fp8,                          // device [N, K]
    const uint8_t* W_scale,                        // device [N/128, K/128]
    int K,
    int N,
    int64_t weight_row_stride_bytes,
    int64_t scale_row_stride_bytes,
    __nv_bfloat16* y,                              // device [N]
    cudaStream_t stream
) {
    if (K % FP8_BLOCK_TILE != 0) {
        std::fprintf(stderr, "fp8_e4m3_gemv_m1: K=%d not multiple of %d\n",
                     K, FP8_BLOCK_TILE);
        return -1;
    }
    if (N % FP8_BLOCK_TILE != 0) {
        // N must be 128-aligned because scale row index = my_row / 128
        // and we expect every row to map cleanly into a scale row that exists.
        std::fprintf(stderr, "fp8_e4m3_gemv_m1: N=%d not multiple of %d\n",
                     N, FP8_BLOCK_TILE);
        return -2;
    }
    if (K > FP8_GEMV_MAX_K) {
        std::fprintf(stderr, "fp8_e4m3_gemv_m1: K=%d > FP8_GEMV_MAX_K=%d\n",
                     K, FP8_GEMV_MAX_K);
        return -3;
    }
    const int n_ctas = (N + FP8_GEMV_BLOCK_N - 1) / FP8_GEMV_BLOCK_N;
    const size_t smem_bytes = (size_t)K * sizeof(__nv_bfloat16);

    fp8_e4m3_gemv_m1_kernel<<<n_ctas, FP8_GEMV_THREADS, smem_bytes, stream>>>(
        x, W_fp8, W_scale, K, N,
        weight_row_stride_bytes, scale_row_stride_bytes, y
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "fp8_e4m3_gemv_m1 launch failed: %s\n",
                     cudaGetErrorString(err));
        return -4;
    }
    return 0;
}

// ─── Optional helper: fixed-tile compile-time variant ─────────────────────
// Compile with -DFP8_GEMV_BLOCK_N=16/32/64 from the Makefile sweep target
// to A/B test slab width (same kernel body, instantiated with a different N).
