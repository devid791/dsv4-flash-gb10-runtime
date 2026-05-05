// dsv4_engine_mxfp4.cu — D2 SCAFFOLDING for DSv4-Flash MXFP4 native C++ engine.
//
// Davide Zenati, 2026-05-04 .
//
// Goal of THIS file (D2 scope):
//   * Define `DSv4EngineMxfp4State` struct holding ALL per-layer pointers to
//     GPU-resident weights (FP8 attention/shared, BF16 norms/router/embed/head,
//     F32 HC) and to CPU-mmap routed expert weights (256 expert/layer * 3
//     tensors * 43 layers).
//   * Provide `init_engine(weights_path)` C ABI entry point that:
//       - opens HF safetensors snapshot via the pre-existing Python loader
//         (we do NOT re-implement it here; init_engine takes a callback),
//       - allocates GPU buffers for hidden state, residual stream, KV cache,
//         hot-set scratch, HC residual stream `[T, n_hc=4, n_embd=4096]`,
//       - precomputes RoPE freqs (base + compressed YARN variant),
//       - dlopen()s the 4 .so kernels and resolves entry points via dlsym.
//   * Provide `decode_step_mxfp4(state, token_id) -> next_token` STUB that
//     returns 0. Real implementation is D9's job (merge of D3-D8).
//   * Provide `free_engine(state)` cleanup C ABI entry point.
//
// HARD CONSTRAINT: NO forward implementation here. Only init/load/cleanup.
// Decode forward will be merged from D3 (RoPE+MLA), D4 (HC), D5 (MoE),
// D6 (FP8 GEMV calls), D7 (sampler), D8 (KV cache), D9 (orchestration).
//
// Build:
//   make  (in this directory) -> libdsv4_engine_mxfp4.so
//
// Reference: kernel/cpp-decode/dsv4_decode.cu 
//            runtime/dsv4_engine_mxfp4.py (R2 Python reference, 753 LOC)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


// ─────────────────────────────────────────────────────────────────────────────
// Architecture constants — match runtime/dsv4_engine_mxfp4.py L100-L130
// ─────────────────────────────────────────────────────────────────────────────

#define N_LAYERS              43
#define HIDDEN                4096
#define VOCAB                 129280
#define NUM_HEADS             64
#define KV_LORA_RANK          512
#define Q_LORA_RANK           1024
#define HEAD_DIM              512
#define QK_ROPE_DIM           64
#define N_GROUPS              8
#define O_LORA_RANK           1024
#define GROUP_DIN             4096   // NUM_HEADS * HEAD_DIM / N_GROUPS
#define WINDOW_SIZE           128
#define MAX_SEQ_LEN_INIT      4096
#define ROPE_THETA            10000.0f
#define COMPRESS_ROPE_THETA   160000.0f
#define YARN_FACTOR           16.0f
#define YARN_ORIGINAL         65536
#define YARN_BETA_FAST        32.0f
#define YARN_BETA_SLOW        1.0f
#define RMS_EPS               1e-6f
#define N_EXPERTS             256
#define EXPERTS_PER_TOK       6
#define EXPERT_FF             2048
#define ROUTED_SCALING        1.5f
#define SWIGLU_LIMIT          10.0f
#define HC_N_HC               4
#define HC_SINKHORN_ITERS     20
#define HC_EPS                1e-6f
#define HOTSET_PER_LAYER      64       // top-N expert kept GPU-resident per layer

// MXFP4 packing constants (block size = 32 elements, 1 E8M0 scale per block)
#define MXFP4_BLOCK_SIZE      32

// FP8 E4M3 packing constants (2-D block tile = 128x128)
#define FP8_BLOCK_TILE        128


// ─────────────────────────────────────────────────────────────────────────────
// Engine state structs LayerWeightsMxfp4 and DSv4EngineMxfp4State are defined in
// dsv4_engine_mxfp4_internal.h (D9 ODR fix — shared with wire-up TU).
// ─────────────────────────────────────────────────────────────────────────────

#include "dsv4_engine_mxfp4_internal.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helper macros
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "[D2] CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(_e));        \
        return -1;                                                  \
    }                                                               \
} while(0)

#define DLSYM_OR_FAIL(state_field, lib_handle, sym_name) do {       \
    *(void**)(&state_field) = dlsym(lib_handle, sym_name);          \
    if (state_field == NULL) {                                      \
        fprintf(stderr, "[D2] dlsym(%s) failed: %s\n",              \
                sym_name, dlerror());                                \
        return -1;                                                   \
    }                                                                \
} while(0)


// ─────────────────────────────────────────────────────────────────────────────
// dlopen the 4 kernel .so files and resolve symbols.
// .so paths are passed by Python wrapper (so we don't hardcode worktree path).
// ─────────────────────────────────────────────────────────────────────────────

extern "C" int dsv4_mxfp4_load_kernels(
    DSv4EngineMxfp4State* st,
    const char* path_fp8_dense,
    const char* path_mxfp4_dense,
    const char* path_mxfp4_routed,
    const char* path_rmsnorm)
{
    // Open all 4 .so files (RTLD_NOW = resolve all symbols immediately).
    st->lib_fp8_dense = dlopen(path_fp8_dense, RTLD_NOW | RTLD_LOCAL);
    if (!st->lib_fp8_dense) {
        fprintf(stderr, "[D2] dlopen(%s) failed: %s\n", path_fp8_dense, dlerror());
        return -1;
    }
    st->lib_mxfp4_dense = dlopen(path_mxfp4_dense, RTLD_NOW | RTLD_LOCAL);
    if (!st->lib_mxfp4_dense) {
        fprintf(stderr, "[D2] dlopen(%s) failed: %s\n", path_mxfp4_dense, dlerror());
        return -1;
    }
    st->lib_mxfp4_routed = dlopen(path_mxfp4_routed, RTLD_NOW | RTLD_LOCAL);
    if (!st->lib_mxfp4_routed) {
        fprintf(stderr, "[D2] dlopen(%s) failed: %s\n", path_mxfp4_routed, dlerror());
        return -1;
    }
    st->lib_rmsnorm = dlopen(path_rmsnorm, RTLD_NOW | RTLD_LOCAL);
    if (!st->lib_rmsnorm) {
        fprintf(stderr, "[D2] dlopen(%s) failed: %s\n", path_rmsnorm, dlerror());
        return -1;
    }

    // Resolve entry points.
    DLSYM_OR_FAIL(st->fn_fp8_e4m3_gemv_m1,        st->lib_fp8_dense,    "fp8_e4m3_gemv_m1");
    DLSYM_OR_FAIL(st->fn_mxfp4_grouped_gemv_topk, st->lib_mxfp4_routed, "mxfp4_grouped_gemv_topk");
    DLSYM_OR_FAIL(st->fn_mxfp4_grouped_workspace_bytes,
                  st->lib_mxfp4_routed, "mxfp4_grouped_workspace_bytes");
    DLSYM_OR_FAIL(st->fn_rmsnorm_fwd,             st->lib_rmsnorm,      "rmsnorm_fuse_fwd");
    DLSYM_OR_FAIL(st->fn_rmsnorm_residual_add,    st->lib_rmsnorm,      "rmsnorm_fuse_residual_add");

    fprintf(stderr, "[D2] kernels loaded (4 .so, 5 entry points resolved)\n");
    return 0;
}


// ─────────────────────────────────────────────────────────────────────────────
// Precompute RoPE freqs_cis (complex pairs cos/sin).
// We allocate two GPU buffers: base (no YARN) + compress (YARN factor=16).
// ─────────────────────────────────────────────────────────────────────────────

// --- D25: YARN piecewise ramp helpers (port R2 rope_yarn.precompute_freqs_cis) ---
// R2 rope_yarn.py reference functions:
//   _find_correction_dim(num_rotations, dim, base, max_seq_len)
//     -> dim * log(max_seq_len / (num_rotations * 2*pi)) / (2 * log(base))
//   _find_correction_range(low_rot=beta_fast, high_rot=beta_slow, dim, base, max_seq_len)
//     -> floor(corr_dim(low_rot)), ceil(corr_dim(high_rot)), clamped to [0, dim-1]
//   _linear_ramp_factor(low, high, dim) -> clamp((arange(dim) - low)/(high-low), 0, 1)
//
// IMPORTANT: R2 calls _find_correction_range with dim=full_rope_dim=64, then
// _linear_ramp_factor with dim=half=32 and the SAME (low, high) computed on full.
// We replicate this 1:1 (asymmetry is in R2 source).
static inline float d25_yarn_correction_dim(float num_rotations, int dim_full,
                                             float base, float max_pos) {
    const float TWO_PI = 6.28318530717958647692f;
    return (float)dim_full * logf(max_pos / (num_rotations * TWO_PI))
           / (2.0f * logf(base));
}
static inline void d25_yarn_correction_range(float beta_fast, float beta_slow,
                                              int dim_full, float base, float max_pos,
                                              int* out_low, int* out_high) {
    float low_f  = floorf(d25_yarn_correction_dim(beta_fast, dim_full, base, max_pos));
    float high_f = ceilf (d25_yarn_correction_dim(beta_slow, dim_full, base, max_pos));
    int low  = (int)low_f;
    int high = (int)high_f;
    if (low  < 0)            low  = 0;
    if (high > dim_full - 1) high = dim_full - 1;
    *out_low  = low;
    *out_high = high;
}
static inline float d25_yarn_linear_ramp(float low, float high, int idx) {
    if (low == high) high = high + 0.001f;
    float r = ((float)idx - low) / (high - low);
    if (r < 0.0f) r = 0.0f;
    if (r > 1.0f) r = 1.0f;
    return r;
}

static int dsv4_mxfp4_precompute_rope(DSv4EngineMxfp4State* st)
{
    const int dim = QK_ROPE_DIM;        // 64
    const int half = dim / 2;            // 32
    const int seqlen = MAX_SEQ_LEN_INIT; // 4096
    const size_t bytes = (size_t)seqlen * half * 2 * sizeof(float); // (cos, sin) pairs

    CUDA_CHECK(cudaMalloc(&st->freqs_cis_base,     bytes));
    CUDA_CHECK(cudaMalloc(&st->freqs_cis_compress, bytes));

    // Compute on host then memcpy.
    float* h_base = (float*)malloc(bytes);
    float* h_comp = (float*)malloc(bytes);
    if (!h_base || !h_comp) {
        fprintf(stderr, "[D2] OOM precomputing RoPE\n");
        return -1;
    }

    // Base (SW layers, no YARN): theta = ROPE_THETA = 10000
    for (int pos = 0; pos < seqlen; ++pos) {
        for (int i = 0; i < half; ++i) {
            float exponent = (float)(2 * i) / (float)dim;
            float freq = 1.0f / powf(ROPE_THETA, exponent);
            float angle = (float)pos * freq;
            h_base[(pos * half + i) * 2 + 0] = cosf(angle);
            h_base[(pos * half + i) * 2 + 1] = sinf(angle);
        }
    }

    // D25: Compressed YARN with piecewise NTK ramp (port R2 rope_yarn.precompute_freqs_cis).
    // theta=COMPRESS_ROPE_THETA=160000, factor=YARN_FACTOR=16, original=YARN_ORIGINAL=65536,
    // beta_fast=32, beta_slow=1.
    // Compute ramp bounds on FULL dim (=QK_ROPE_DIM=64). Use the same (low,high) for
    // half-index ramp — R2 does this asymmetrically too.
    int yarn_low_i, yarn_high_i;
    d25_yarn_correction_range(YARN_BETA_FAST, YARN_BETA_SLOW, dim,
                              COMPRESS_ROPE_THETA, (float)YARN_ORIGINAL,
                              &yarn_low_i, &yarn_high_i);
    fprintf(stderr, "[D25] YARN ramp bounds (dim_full=%d, theta=%.0f, factor=%.0f, original=%d): low=%d, high=%d\n",
            dim, COMPRESS_ROPE_THETA, YARN_FACTOR, YARN_ORIGINAL, yarn_low_i, yarn_high_i);
    const float yarn_low_f  = (float)yarn_low_i;
    const float yarn_high_f = (float)yarn_high_i;
    for (int pos = 0; pos < seqlen; ++pos) {
        for (int i = 0; i < half; ++i) {
            float exponent = (float)(2 * i) / (float)dim;
            float freq = 1.0f / powf(COMPRESS_ROPE_THETA, exponent);
            // R2 ramp: smooth = 1 - linear_ramp; freq = freq/factor * (1 - smooth) + freq * smooth
            float linear = d25_yarn_linear_ramp(yarn_low_f, yarn_high_f, i);
            float smooth = 1.0f - linear;
            float scaled = freq / YARN_FACTOR * (1.0f - smooth) + freq * smooth;
            float angle = (float)pos * scaled;
            h_comp[(pos * half + i) * 2 + 0] = cosf(angle);
            h_comp[(pos * half + i) * 2 + 1] = sinf(angle);
        }
    }

    CUDA_CHECK(cudaMemcpy(st->freqs_cis_base,     h_base, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st->freqs_cis_compress, h_comp, bytes, cudaMemcpyHostToDevice));
    free(h_base); free(h_comp);

    // Per-layer compress_ratios (HF config, 43 entries)
    static const int CR[N_LAYERS] = {
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 0
    };
    memcpy(st->compress_ratios, CR, sizeof(CR));

    fprintf(stderr, "[D2] RoPE freqs precomputed: base+compress (%zu bytes each)\n", bytes);
    return 0;
}


// ─────────────────────────────────────────────────────────────────────────────
// Allocate decode-step working buffers (BF16/F32) on GPU.
// ─────────────────────────────────────────────────────────────────────────────

static int dsv4_mxfp4_alloc_workspace(DSv4EngineMxfp4State* st)
{
    const size_t bf16 = sizeof(__nv_bfloat16);

    // HC residual stream: [HC_N_HC=4, HIDDEN=4096] BF16
    CUDA_CHECK(cudaMalloc(&st->hc_stream,    HC_N_HC * HIDDEN * bf16));

    // Per-step working buffers
    CUDA_CHECK(cudaMalloc(&st->hidden,       HIDDEN * bf16));
    CUDA_CHECK(cudaMalloc(&st->h_norm,       HIDDEN * bf16));
    CUDA_CHECK(cudaMalloc(&st->attn_out,     HIDDEN * bf16));
    CUDA_CHECK(cudaMalloc(&st->moe_out,      HIDDEN * bf16));
    CUDA_CHECK(cudaMalloc(&st->shared_out,   HIDDEN * bf16));
    CUDA_CHECK(cudaMalloc(&st->routed_out,   HIDDEN * bf16));

    // Attention intermediates
    CUDA_CHECK(cudaMalloc(&st->q_a_buf,      Q_LORA_RANK * bf16));
    CUDA_CHECK(cudaMalloc(&st->q_b_buf,      NUM_HEADS * HEAD_DIM * bf16));
    CUDA_CHECK(cudaMalloc(&st->kv_proj_buf, (2*KV_LORA_RANK + QK_ROPE_DIM) * bf16));
    CUDA_CHECK(cudaMalloc(&st->o_a_buf,      O_LORA_RANK * bf16));

    // Shared expert intermediates
    CUDA_CHECK(cudaMalloc(&st->sh_gate_buf,  EXPERT_FF * bf16));
    CUDA_CHECK(cudaMalloc(&st->sh_up_buf,    EXPERT_FF * bf16));
    CUDA_CHECK(cudaMalloc(&st->sh_act_buf,   EXPERT_FF * bf16));

    // Router
    CUDA_CHECK(cudaMalloc(&st->router_logits,  N_EXPERTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->router_scores,  N_EXPERTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->topk_ids,       EXPERTS_PER_TOK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&st->topk_weights,   EXPERTS_PER_TOK * sizeof(float)));

    // Output logits
    CUDA_CHECK(cudaMalloc(&st->logits,         VOCAB * sizeof(float)));

    fprintf(stderr, "[D2] decode workspace allocated (BF16 hidden + attn + MoE + logits)\n");
    return 0;
}


// ─────────────────────────────────────────────────────────────────────────────
// Allocate KV cache for max_seq tokens (BF16, latent KV_LORA_RANK per layer).
// ─────────────────────────────────────────────────────────────────────────────

extern "C" int dsv4_mxfp4_alloc_kv_cache(DSv4EngineMxfp4State* st, int max_seq)
{
    if (max_seq <= 0) max_seq = 1024;

    // D26C: idempotent. If already allocated and capacity ok, just reset cur_pos.
    if (st->kv_cache[0] != nullptr && st->max_seq >= max_seq) {
        st->cur_pos = 0;
        // (do not change st->max_seq; keep larger capacity allocated)
        return 0;
    }

    // D26C: free old buffers if any (was leaking 5.6 MB per call when called repeatedly).
    for (int L = 0; L < N_LAYERS; ++L) {
        if (st->kv_cache[L] != nullptr) {
            cudaFree(st->kv_cache[L]);
            st->kv_cache[L] = nullptr;
        }
    }

    st->max_seq = max_seq;
    st->cur_pos = 0;

    const size_t bytes_per_layer = (size_t)max_seq * KV_LORA_RANK * sizeof(__nv_bfloat16);
    for (int L = 0; L < N_LAYERS; ++L) {
        CUDA_CHECK(cudaMalloc(&st->kv_cache[L], bytes_per_layer));
    }
    double mb = (double)bytes_per_layer * N_LAYERS / 1e6;
    fprintf(stderr, "[D2] KV cache allocated: %d layers x %d seq x %d kv_lora = %.1f MB total\n",
            N_LAYERS, max_seq, KV_LORA_RANK, mb);
    return 0;
}


extern "C" void dsv4_mxfp4_reset_kv(DSv4EngineMxfp4State* st)
{
    st->cur_pos = 0;
    // We do NOT zero the buffers; cur_pos tracks valid prefix.
}


// ─────────────────────────────────────────────────────────────────────────────
// Hot-set bank allocation: per-layer GPU buffer sized for HOTSET_PER_LAYER
// experts (w1/w3 packed [EXPERT_FF, HIDDEN/2] uint8 + w2 [HIDDEN, EXPERT_FF/2]).
// Fill is deferred to D5 (MoE) once routed_meta CPU mmap layout is known.
// ─────────────────────────────────────────────────────────────────────────────

extern "C" int dsv4_mxfp4_alloc_hotset_banks(DSv4EngineMxfp4State* st)
{
    // MXFP4 packed: 2 nibbles per byte. HIDDEN/2 = 2048 bytes per row.
    const size_t w13_bytes_per_expert = (size_t)EXPERT_FF * (HIDDEN / 2);    // w1, w3
    const size_t w2_bytes_per_expert  = (size_t)HIDDEN    * (EXPERT_FF / 2); // w2

    // E8M0 scales: 1 byte per 32-element block (block axis = K direction)
    const size_t w13_scale_per_expert = (size_t)EXPERT_FF * (HIDDEN / MXFP4_BLOCK_SIZE);
    const size_t w2_scale_per_expert  = (size_t)HIDDEN    * (EXPERT_FF / MXFP4_BLOCK_SIZE);

    for (int L = 0; L < N_LAYERS; ++L) {
        CUDA_CHECK(cudaMalloc(&st->hotset_w1_packed[L], w13_bytes_per_expert * HOTSET_PER_LAYER));
        CUDA_CHECK(cudaMalloc(&st->hotset_w2_packed[L], w2_bytes_per_expert  * HOTSET_PER_LAYER));
        CUDA_CHECK(cudaMalloc(&st->hotset_w3_packed[L], w13_bytes_per_expert * HOTSET_PER_LAYER));
        CUDA_CHECK(cudaMalloc(&st->hotset_w1_scale[L],  w13_scale_per_expert * HOTSET_PER_LAYER));
        CUDA_CHECK(cudaMalloc(&st->hotset_w2_scale[L],  w2_scale_per_expert  * HOTSET_PER_LAYER));
        CUDA_CHECK(cudaMalloc(&st->hotset_w3_scale[L],  w13_scale_per_expert * HOTSET_PER_LAYER));
        st->layers[L].hotset_count = 0;  // empty until D5 fills
    }

    double per_layer_gb = (double)(2*w13_bytes_per_expert + w2_bytes_per_expert
                                 + 2*w13_scale_per_expert + w2_scale_per_expert)
                         * HOTSET_PER_LAYER / 1e9;
    fprintf(stderr, "[D2] hot-set banks allocated: %d layers x %d expert/layer = %.2f GB/layer\n",
            N_LAYERS, HOTSET_PER_LAYER, per_layer_gb);
    return 0;
}


// ─────────────────────────────────────────────────────────────────────────────
// init_engine: orchestrate stream/cublas + load kernels + allocate buffers.
// Weights themselves are wired in by Python after this returns (via
// dsv4_mxfp4_set_top + dsv4_mxfp4_set_layer setter functions).
// This mirrors the binding pattern from cpp-decode/dsv4_decode_binding.py.
// ─────────────────────────────────────────────────────────────────────────────

extern "C" DSv4EngineMxfp4State* dsv4_mxfp4_init_engine(
    int n_layers,
    int max_seq,
    const char* path_fp8_dense,
    const char* path_mxfp4_dense,
    const char* path_mxfp4_routed,
    const char* path_rmsnorm)
{
    DSv4EngineMxfp4State* st = (DSv4EngineMxfp4State*)calloc(1, sizeof(DSv4EngineMxfp4State));
    if (!st) {
        fprintf(stderr, "[D2] OOM allocating DSv4EngineMxfp4State\n");
        return NULL;
    }

    if (n_layers <= 0 || n_layers > N_LAYERS) n_layers = N_LAYERS;
    st->n_layers = n_layers;

    // CUDA stream + cuBLAS
    cudaError_t ce = cudaStreamCreate(&st->stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[D2] cudaStreamCreate failed: %s\n", cudaGetErrorString(ce));
        free(st);
        return NULL;
    }
    cublasStatus_t bs = cublasCreate(&st->cublas);
    if (bs != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[D2] cublasCreate failed: %d\n", (int)bs);
        cudaStreamDestroy(st->stream);
        free(st);
        return NULL;
    }
    cublasSetStream(st->cublas, st->stream);

    // Load kernel .so files
    if (dsv4_mxfp4_load_kernels(st,
            path_fp8_dense, path_mxfp4_dense, path_mxfp4_routed, path_rmsnorm) != 0) {
        fprintf(stderr, "[D2] load_kernels failed\n");
        free(st);
        return NULL;
    }

    // Workspace + KV cache + hot-set banks + RoPE
    if (dsv4_mxfp4_alloc_workspace(st) != 0)         { free(st); return NULL; }
    if (dsv4_mxfp4_alloc_kv_cache(st, max_seq) != 0) { free(st); return NULL; }
    if (dsv4_mxfp4_alloc_hotset_banks(st) != 0)      { free(st); return NULL; }
    if (dsv4_mxfp4_precompute_rope(st) != 0)         { free(st); return NULL; }

    fprintf(stderr, "[D2] init_engine OK: %d layers, max_seq=%d\n",
            st->n_layers, st->max_seq);
    return st;
}


// ─────────────────────────────────────────────────────────────────────────────
// Setter ABI for top-level + per-layer pointers.
// Python wrapper calls these once after the safetensors loader has uploaded
// each tensor to GPU. We only store device pointers + strides; no copy.
// ─────────────────────────────────────────────────────────────────────────────

extern "C" void dsv4_mxfp4_set_top(
    DSv4EngineMxfp4State* st,
    void* embed, void* head, void* final_norm,
    void* hc_head_fn, void* hc_head_base, void* hc_head_scale)
{
    st->embed         = (__nv_bfloat16*)embed;
    st->head          = (__nv_bfloat16*)head;
    st->final_norm    = (__nv_bfloat16*)final_norm;
    st->hc_head_fn    = (float*)hc_head_fn;
    st->hc_head_base  = (float*)hc_head_base;
    st->hc_head_scale = (float*)hc_head_scale;
}


extern "C" void dsv4_mxfp4_set_layer(
    DSv4EngineMxfp4State* st, int L,
    // Norms
    void* attn_norm, void* ffn_norm, void* q_norm, void* kv_norm, void* attn_sink,
    // FP8 attention (weight + scale uint8)
    void* wkv_w,  void* wkv_s,
    void* wq_a_w, void* wq_a_s,
    void* wq_b_w, void* wq_b_s,
    void* wo_a_w, void* wo_a_s,
    void* wo_b_w, void* wo_b_s,
    // Router gate (BF16)
    void* router_gate,
    // FP8 shared expert
    void* sh_w1_w, void* sh_w1_s,
    void* sh_w2_w, void* sh_w2_s,
    void* sh_w3_w, void* sh_w3_s,
    // HC F32
    void* hc_attn_fn, void* hc_attn_base, void* hc_attn_scale,
    void* hc_ffn_fn,  void* hc_ffn_base,  void* hc_ffn_scale)
{
    if (L < 0 || L >= N_LAYERS) {
        fprintf(stderr, "[D2] set_layer: invalid L=%d\n", L);
        return;
    }
    LayerWeightsMxfp4* ly = &st->layers[L];
    ly->L = L;
    ly->attn_norm    = (__nv_bfloat16*)attn_norm;
    ly->ffn_norm     = (__nv_bfloat16*)ffn_norm;
    ly->q_norm       = (__nv_bfloat16*)q_norm;
    ly->kv_norm      = (__nv_bfloat16*)kv_norm;
    ly->attn_sink    = (float*)attn_sink;

    ly->wkv_w  = (uint8_t*)wkv_w;   ly->wkv_s  = (uint8_t*)wkv_s;
    ly->wq_a_w = (uint8_t*)wq_a_w;  ly->wq_a_s = (uint8_t*)wq_a_s;
    ly->wq_b_w = (uint8_t*)wq_b_w;  ly->wq_b_s = (uint8_t*)wq_b_s;
    ly->wo_a_w = (uint8_t*)wo_a_w;  ly->wo_a_s = (uint8_t*)wo_a_s;
    ly->wo_b_w = (uint8_t*)wo_b_w;  ly->wo_b_s = (uint8_t*)wo_b_s;

    ly->router_gate = (__nv_bfloat16*)router_gate;
    ly->tid2eid = NULL;  // optional, set later if available

    ly->sh_w1_w = (uint8_t*)sh_w1_w;  ly->sh_w1_s = (uint8_t*)sh_w1_s;
    ly->sh_w2_w = (uint8_t*)sh_w2_w;  ly->sh_w2_s = (uint8_t*)sh_w2_s;
    ly->sh_w3_w = (uint8_t*)sh_w3_w;  ly->sh_w3_s = (uint8_t*)sh_w3_s;

    ly->hc_attn_fn    = (float*)hc_attn_fn;
    ly->hc_attn_base  = (float*)hc_attn_base;
    ly->hc_attn_scale = (float*)hc_attn_scale;
    ly->hc_ffn_fn     = (float*)hc_ffn_fn;
    ly->hc_ffn_base   = (float*)hc_ffn_base;
    ly->hc_ffn_scale  = (float*)hc_ffn_scale;
}


// Set FP8 row-strides for a layer. Must be called after set_layer because
// the Python loader knows the actual strides (depends on tensor layout).
extern "C" void dsv4_mxfp4_set_fp8_strides(
    DSv4EngineMxfp4State* st, int L,
    int wkv_out, int64_t wkv_w_rs, int64_t wkv_s_rs,
    int wq_a_out, int64_t wq_a_w_rs, int64_t wq_a_s_rs,
    int wq_b_out, int64_t wq_b_w_rs, int64_t wq_b_s_rs,
    int wo_a_out, int64_t wo_a_w_rs, int64_t wo_a_s_rs,
    int wo_b_out, int64_t wo_b_w_rs, int64_t wo_b_s_rs,
    int64_t sh_w1_w_rs, int64_t sh_w1_s_rs,
    int64_t sh_w2_w_rs, int64_t sh_w2_s_rs,
    int64_t sh_w3_w_rs, int64_t sh_w3_s_rs)
{
    if (L < 0 || L >= N_LAYERS) return;
    LayerWeightsMxfp4* ly = &st->layers[L];
    ly->wkv_out_dim  = wkv_out;   ly->wkv_w_row_stride  = wkv_w_rs;   ly->wkv_s_row_stride  = wkv_s_rs;
    ly->wq_a_out_dim = wq_a_out;  ly->wq_a_w_row_stride = wq_a_w_rs;  ly->wq_a_s_row_stride = wq_a_s_rs;
    ly->wq_b_out_dim = wq_b_out;  ly->wq_b_w_row_stride = wq_b_w_rs;  ly->wq_b_s_row_stride = wq_b_s_rs;
    ly->wo_a_out_dim = wo_a_out;  ly->wo_a_w_row_stride = wo_a_w_rs;  ly->wo_a_s_row_stride = wo_a_s_rs;
    ly->wo_b_out_dim = wo_b_out;  ly->wo_b_w_row_stride = wo_b_w_rs;  ly->wo_b_s_row_stride = wo_b_s_rs;
    ly->sh_w1_w_row_stride = sh_w1_w_rs;  ly->sh_w1_s_row_stride = sh_w1_s_rs;
    ly->sh_w2_w_row_stride = sh_w2_w_rs;  ly->sh_w2_s_row_stride = sh_w2_s_rs;
    ly->sh_w3_w_row_stride = sh_w3_w_rs;  ly->sh_w3_s_row_stride = sh_w3_s_rs;
}


// ─────────────────────────────────────────────────────────────────────────────
// decode_step_mxfp4 — STUB. Real implementation is D9 (merge of D3-D8).
// Returns 0 (BOS-like sentinel) so init smoke tests don't crash.
// ─────────────────────────────────────────────────────────────────────────────

// D9 wire-up: forward to real implementation in dsv4_engine_mxfp4_wire.cu
extern "C" int dsv4_mxfp4_decode_step_real(DSv4EngineMxfp4State* st, int token_id);

extern "C" int dsv4_mxfp4_decode_step(DSv4EngineMxfp4State* st, int token_id)
{
    return dsv4_mxfp4_decode_step_real(st, token_id);
}


// ─────────────────────────────────────────────────────────────────────────────
// free_engine: cleanup all GPU buffers + close .so handles + destroy stream.
// ─────────────────────────────────────────────────────────────────────────────

extern "C" void dsv4_mxfp4_wire_cleanup();

extern "C" void dsv4_mxfp4_free_engine(DSv4EngineMxfp4State* st)
{
    if (!st) return;
    dsv4_mxfp4_wire_cleanup();

    // Workspace
    if (st->hc_stream)     cudaFree(st->hc_stream);
    if (st->hidden)        cudaFree(st->hidden);
    if (st->h_norm)        cudaFree(st->h_norm);
    if (st->attn_out)      cudaFree(st->attn_out);
    if (st->moe_out)       cudaFree(st->moe_out);
    if (st->shared_out)    cudaFree(st->shared_out);
    if (st->routed_out)    cudaFree(st->routed_out);
    if (st->q_a_buf)       cudaFree(st->q_a_buf);
    if (st->q_b_buf)       cudaFree(st->q_b_buf);
    if (st->kv_proj_buf)   cudaFree(st->kv_proj_buf);
    if (st->o_a_buf)       cudaFree(st->o_a_buf);
    if (st->sh_gate_buf)   cudaFree(st->sh_gate_buf);
    if (st->sh_up_buf)     cudaFree(st->sh_up_buf);
    if (st->sh_act_buf)    cudaFree(st->sh_act_buf);
    if (st->router_logits) cudaFree(st->router_logits);
    if (st->router_scores) cudaFree(st->router_scores);
    if (st->topk_ids)      cudaFree(st->topk_ids);
    if (st->topk_weights)  cudaFree(st->topk_weights);
    if (st->logits)        cudaFree(st->logits);

    // KV cache + hot-set banks
    for (int L = 0; L < N_LAYERS; ++L) {
        if (st->kv_cache[L])         cudaFree(st->kv_cache[L]);
        if (st->hotset_w1_packed[L]) cudaFree(st->hotset_w1_packed[L]);
        if (st->hotset_w2_packed[L]) cudaFree(st->hotset_w2_packed[L]);
        if (st->hotset_w3_packed[L]) cudaFree(st->hotset_w3_packed[L]);
        if (st->hotset_w1_scale[L])  cudaFree(st->hotset_w1_scale[L]);
        if (st->hotset_w2_scale[L])  cudaFree(st->hotset_w2_scale[L]);
        if (st->hotset_w3_scale[L])  cudaFree(st->hotset_w3_scale[L]);
    }

    // RoPE
    if (st->freqs_cis_base)     cudaFree(st->freqs_cis_base);
    if (st->freqs_cis_compress) cudaFree(st->freqs_cis_compress);

    // Stream + cublas
    if (st->cublas) cublasDestroy(st->cublas);
    if (st->stream) cudaStreamDestroy(st->stream);

    // dlclose handles
    if (st->lib_fp8_dense)    dlclose(st->lib_fp8_dense);
    if (st->lib_mxfp4_dense)  dlclose(st->lib_mxfp4_dense);
    if (st->lib_mxfp4_routed) dlclose(st->lib_mxfp4_routed);
    if (st->lib_rmsnorm)      dlclose(st->lib_rmsnorm);

    free(st);
    fprintf(stderr, "[D2] engine freed\n");
}


// ─────────────────────────────────────────────────────────────────────────────
// Introspection ABI: struct sizes (used by Python ctypes wrapper to allocate)
// ─────────────────────────────────────────────────────────────────────────────

extern "C" size_t dsv4_mxfp4_state_struct_size(void) {
    return sizeof(DSv4EngineMxfp4State);
}

extern "C" size_t dsv4_mxfp4_layer_struct_size(void) {
    return sizeof(LayerWeightsMxfp4);
}

extern "C" int dsv4_mxfp4_n_layers(DSv4EngineMxfp4State* st) {
    return st ? st->n_layers : 0;
}

extern "C" int dsv4_mxfp4_max_seq(DSv4EngineMxfp4State* st) {
    return st ? st->max_seq : 0;
}

// EOF
