// rmsnorm_fuse.cu — Fused RMSNorm + (optional) residual-add for DSv4-Flash
// Project: — RMSNorm + epilogue fuse
// Authors: Davide Zenati
// License: MIT
// Target: GB10 sm_121a (Grace Blackwell), CUDA 13.2.
//
// Operations exposed (extern "C"):
//
//   rmsnorm_fuse_fwd(x, weight, y, n_rows, hidden, eps)
//     y[r, h] = weight[h] * x[r, h] / sqrt( mean(x[r, :]**2) + eps )
//     dtype: x, weight, y all BF16 (__nv_bfloat16). Reduction in fp32.
//
//   rmsnorm_fuse_residual_add(x, residual, weight, y_norm, new_residual,
//                             n_rows, hidden, eps)
//     tmp[r, h]            = x[r, h] + residual[r, h]
//     new_residual[r, h]   = tmp[r, h]                (chain residual forward)
//     y_norm[r, h]         = weight[h] * tmp[r, h] / sqrt(mean(tmp**2) + eps)
//     dtype: all BF16. Reduction fp32. tmp lives in shared mem during compute.
//
// Launch: 1 CTA per row (blockIdx.x = r), 256 threads cooperatively reduce
// over `hidden` (each thread handles `hidden / 256` elements, default 16 for
// hidden=4096). Welford-style mean of squares using fp32 accumulators.
// Shared mem: hidden BF16 buffer (8 KB at H=4096) + small reduction scratch.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef RMSF_THREADS
#define RMSF_THREADS 256
#endif

#ifndef RMSF_MAX_HIDDEN
#define RMSF_MAX_HIDDEN 8192          // upper bound for shared-mem tmp buffer
#endif

// ─── Warp + block reduction helpers ────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

// Block reduction across RMSF_THREADS using warp shuffles + shared scratch.
// Returns the final sum in every thread (broadcast).
__device__ __forceinline__ float block_reduce_sum(float v, float* smem_warp) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int n_warps = RMSF_THREADS >> 5;

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

// ─── Kernel 1: pure RMSNorm fwd ────────────────────────────────────────────

extern "C" __global__ void rmsnorm_fuse_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    const int n_rows,
    const int hidden,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const __nv_bfloat16* x_row = x + (size_t)row * hidden;
    __nv_bfloat16* y_row       = y + (size_t)row * hidden;

    extern __shared__ unsigned char smem_raw[];
    float* smem_warp = reinterpret_cast<float*>(smem_raw);    // 8 floats (n_warps)

    // 1) compute sum of squares in fp32, streaming x without smem buffering.
    float local_sq = 0.f;
    for (int h = threadIdx.x; h < hidden; h += RMSF_THREADS) {
        float v = __bfloat162float(x_row[h]);
        local_sq += v * v;
    }
    float total_sq = block_reduce_sum(local_sq, smem_warp);

    // 2) compute scale = rsqrt(mean + eps), broadcast.
    float inv_n  = 1.f / (float)hidden;
    float rms    = rsqrtf(total_sq * inv_n + eps);

    // 3) write y[h] = weight[h] * x[h] * rms.
    for (int h = threadIdx.x; h < hidden; h += RMSF_THREADS) {
        float v  = __bfloat162float(x_row[h]);
        float wf = __bfloat162float(weight[h]);
        y_row[h] = __float2bfloat16(wf * v * rms);
    }
}

// ─── Kernel 2: fused residual-add + RMSNorm ────────────────────────────────
// tmp = x + residual  (fp32 reduction; written back as BF16 to new_residual)
// y   = weight * tmp / sqrt(mean(tmp^2) + eps)

extern "C" __global__ void rmsnorm_fuse_residual_add_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y_norm,
    __nv_bfloat16* __restrict__ new_residual,
    const int n_rows,
    const int hidden,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const size_t off = (size_t)row * hidden;
    const __nv_bfloat16* x_row   = x + off;
    const __nv_bfloat16* res_row = residual + off;
    __nv_bfloat16* y_row         = y_norm + off;
    __nv_bfloat16* new_res_row   = new_residual + off;

    // shared layout: [warp_scratch (8 floats)] [tmp BF16 buffer (hidden)]
    extern __shared__ unsigned char smem_raw[];
    float* smem_warp           = reinterpret_cast<float*>(smem_raw);
    __nv_bfloat16* tmp_smem    = reinterpret_cast<__nv_bfloat16*>(
                                     smem_raw + 32 * sizeof(float));

    // 1) tmp = x + residual, stash in shared (and emit new_residual now).
    float local_sq = 0.f;
    for (int h = threadIdx.x; h < hidden; h += RMSF_THREADS) {
        float vx = __bfloat162float(x_row[h]);
        float vr = __bfloat162float(res_row[h]);
        float t  = vx + vr;
        __nv_bfloat16 t_bf = __float2bfloat16(t);
        tmp_smem[h]      = t_bf;
        new_res_row[h]   = t_bf;
        local_sq        += t * t;
    }
    float total_sq = block_reduce_sum(local_sq, smem_warp);

    float inv_n = 1.f / (float)hidden;
    float rms   = rsqrtf(total_sq * inv_n + eps);

    // 2) y = weight * tmp * rms  (read tmp from shared, weight from gmem).
    for (int h = threadIdx.x; h < hidden; h += RMSF_THREADS) {
        float vt = __bfloat162float(tmp_smem[h]);
        float wf = __bfloat162float(weight[h]);
        y_row[h] = __float2bfloat16(wf * vt * rms);
    }
}

// ─── Host launchers (extern "C", ctypes-friendly) ──────────────────────────

extern "C" int rmsnorm_fuse_fwd(
    const void* x,
    const void* weight,
    void*       y,
    int n_rows,
    int hidden,
    float eps,
    cudaStream_t stream
) {
    if (n_rows <= 0 || hidden <= 0) return 0;
    if (hidden > RMSF_MAX_HIDDEN)    return 1;
    dim3 grid(n_rows);
    dim3 block(RMSF_THREADS);
    size_t smem = 32 * sizeof(float);   // warp scratch only
    rmsnorm_fuse_fwd_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(y),
        n_rows, hidden, eps);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : (int)err;
}

extern "C" int rmsnorm_fuse_residual_add(
    const void* x,
    const void* residual,
    const void* weight,
    void*       y_norm,
    void*       new_residual,
    int n_rows,
    int hidden,
    float eps,
    cudaStream_t stream
) {
    if (n_rows <= 0 || hidden <= 0) return 0;
    if (hidden > RMSF_MAX_HIDDEN)    return 1;
    dim3 grid(n_rows);
    dim3 block(RMSF_THREADS);
    size_t smem = 32 * sizeof(float) + (size_t)hidden * sizeof(__nv_bfloat16);
    rmsnorm_fuse_residual_add_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(y_norm),
        reinterpret_cast<__nv_bfloat16*>(new_residual),
        n_rows, hidden, eps);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : (int)err;
}
