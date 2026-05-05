// dsv4_engine_mxfp4_wire.cu — D9 wire-up of the C++ decode loop.
//
// This file ties together the D2 scaffolding (DSv4EngineMxfp4State,
// dlopen kernel pointers) with the D3-D8 component .cu files and produces a
// real `dsv4_mxfp4_decode_step` end-to-end.
//
// Pipeline per token:
//   embed -> 43 layer × (HC_pre + RMS + MLA + HC_post + HC_pre + RMS +
//                        shared_ffn + routed + HC_post)
//        -> final_norm_via_HC_head -> lm_head BF16 -> argmax
//
// The wire-up keeps everything in the same translation unit so the engine
// state struct is fully visible. Component kernels live in sibling .cu files
// in this directory and are reached via extern "C" forward declarations.
//
// Author: D9 (2026-05-04)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>

// ─── Re-declare D2 engine state minimally (we only touch fields we use) ───
// NOTE: full struct lives in dsv4_engine_mxfp4.cu. We forward-declare the
// opaque shape with the SAME field layout; gcc is happy as long as the
// same definition is reached at link time. To avoid ODR risk we instead
// re-include via a local helper header generated below.

// Bring in the D2 type by symbol re-typing — we need the *layout*. The
// cleanest path is to re-embed minimal duplicates here. To keep ODR safe we
// place the full struct definition in an internal header that both files
// can share if needed in the future. For now, since the C++ engine is a
// single .so built from all sibling .cu files, we simply forward struct
// types and access fields via extern accessor helpers exported from D2.

// Forward declarations of D2 helpers we re-use (defined in dsv4_engine_mxfp4.cu)
struct DSv4EngineMxfp4State;
struct LayerWeightsMxfp4;

extern "C" int  dsv4_mxfp4_n_layers(DSv4EngineMxfp4State* st);
extern "C" int  dsv4_mxfp4_max_seq (DSv4EngineMxfp4State* st);
extern "C" size_t dsv4_mxfp4_state_struct_size(void);
extern "C" size_t dsv4_mxfp4_layer_struct_size(void);

// We need direct field access to the engine state. The cleanest way (given
// the .cu files compile to objects then link into one .so) is to declare an
// "engine accessor" function in D2 that exposes a const handle table to us.
// To keep this patch surgical, we instead promote the shared struct definition
// into this header `dsv4_engine_mxfp4_internal.h` (created beside this file).

#include "dsv4_engine_mxfp4_internal.h"

// ─── Component ABIs (D3 mla_full / D4 hc / D5 routed / D6 hotset / D7 head / D8 graph) ───

// D7 head.cu
extern "C" int embed_forward(const void* embed_table, int token_id, int vocab,
                             int hidden, void* x_out, cudaStream_t stream);
extern "C" int final_norm_forward(const void* x_in, const void* final_norm_w,
                                  void* x_out, int hidden, float eps, cudaStream_t stream);
extern "C" int rms_norm_inline(const void* x_in, const void* weight, void* x_out,
                               int hidden, float eps, cudaStream_t stream);
extern "C" int lm_head_forward_bf16(const void* W, const void* x, void* logits,
                                    int hidden, int vocab, cudaStream_t stream);
extern "C" int sample_greedy(const void* logits, int vocab, int* host_next_token,
                             cudaStream_t stream);

// D4 hc.cu
extern "C" int hc_pre_forward(
    const __nv_bfloat16* x_residual,
    const float* hc_fn, const float* hc_base, const float* hc_scale,
    __nv_bfloat16* x_collapsed,
    float* pre_out, float* post_out, float* comb_out,
    __nv_bfloat16* flat_norm_scratch, float* mixes_scratch,
    cudaStream_t stream);
extern "C" int hc_post_forward(
    const __nv_bfloat16* block_out, __nv_bfloat16* residual_inout,
    const float* post, const float* comb,
    __nv_bfloat16* post_scratch, cudaStream_t stream);
extern "C" int hc_head_compute(
    const __nv_bfloat16* x_residual,
    const float* hc_head_fn, const float* hc_head_base, const float* hc_head_scale,
    __nv_bfloat16* x_collapsed,
    __nv_bfloat16* flat_norm_scratch, float* mixes_scratch, float* pre_scratch,
    cudaStream_t stream);

// D3 mla_full.cu
extern "C" int mla_q_per_head_rmsn(__nv_bfloat16* q, int head_dim, float eps, cudaStream_t s);
extern "C" int mla_rope_apply_q(__nv_bfloat16* q, const float* fr, const float* fi,
                                int pos, int head_dim, int rope_dim, int inv, cudaStream_t s);
extern "C" int mla_rope_apply_kv(__nv_bfloat16* kv, const float* fr, const float* fi,
                                 int pos, int kv_dim, int rope_dim, cudaStream_t s);
extern "C" int mla_rope_inverse_o(__nv_bfloat16* o, const float* fr, const float* fi,
                                  int pos, int head_dim, int rope_dim, cudaStream_t s);
extern "C" int mla_kv_append(__nv_bfloat16* kv_cache, const __nv_bfloat16* kv_buf,
                             int pos, int kv_dim, cudaStream_t s);
extern "C" int mla_full_dot(const __nv_bfloat16* q, const __nv_bfloat16* kv_cache,
                            const float* attn_sink, int P, int head_dim, int kv_dim,
                            float* scores, cudaStream_t s);
extern "C" int mla_full_mask(float* scores, int P, int q_pos, int window, cudaStream_t s);
extern "C" int mla_full_softmax(float* scores, int P, cudaStream_t s);
extern "C" int mla_full_attend(const float* attn_w, const __nv_bfloat16* kv_cache,
                               int P, int head_dim, int kv_dim,
                               __nv_bfloat16* attn_out, cudaStream_t s);

// D5 routed.cu
struct DSv4RoutedState;
extern "C" DSv4RoutedState* dsv4_routed_state_alloc(int n_layers, cudaStream_t s);
extern "C" void dsv4_routed_state_free(DSv4RoutedState* st);
extern "C" int  dsv4_routed_dispatch(DSv4RoutedState* st, int layer_idx,
                                     const __nv_bfloat16* x_in, __nv_bfloat16* x_out);
extern "C" void dsv4_routed_set_router_gate(DSv4RoutedState* st, int layer_idx,
                                            const __nv_bfloat16* d_router_gate);
extern "C" void dsv4_routed_set_hotset(
    DSv4RoutedState* st,
    const uint8_t* (*get_w_packed)(void*, int, int, int),
    const uint8_t* (*get_w_scale )(void*, int, int, int),
    void* user);
extern "C" int  dsv4_routed_get_dispatch_count(const DSv4RoutedState* st);
extern "C" int  dsv4_routed_get_abort_flag(const DSv4RoutedState* st);

// D6 hotset.cu
struct ExpertGpuPtrs {
    void* packed[3];
    void* scale[3];
    size_t packed_bytes[3];
    size_t scale_bytes[3];
    int hit;
};
extern "C" int hotset_init(int n_layers, int n_experts, int hot_per_layer);
extern "C" int hotset_destroy();
extern "C" int hotset_register_cold(int layer, int expert_id, int wkey,
                                    void* packed_host, size_t packed_bytes,
                                    void* scale_host, size_t scale_bytes);
extern "C" int hotset_get_expert_gpu_ptr(int layer, int expert_id, ExpertGpuPtrs* out);
extern "C" int hotset_prewarm_expert(int layer, int expert_id);
extern "C" int hotset_set_consumer_stream(void* s);  // D28A
extern "C" int hotset_prefetch_batch(int layer, const int* ids, int n_ids);  // D28B1

// ─── Local constants (mirror python R2) ───
#define WIRE_HIDDEN              4096
#define WIRE_VOCAB               129280
#define WIRE_NUM_HEADS           64
#define WIRE_HEAD_DIM            512
#define WIRE_KV_LORA_RANK        512
#define WIRE_Q_LORA_RANK         1024
#define WIRE_QK_ROPE_DIM         64
#define WIRE_O_LORA_RANK         1024
#define WIRE_N_GROUPS            8
#define WIRE_GROUP_DIN           4096
#define WIRE_WINDOW_SIZE         128
#define WIRE_RMS_EPS             1e-6f
#define WIRE_HC_N_HC             4

// ─── Wire-up state (created on first decode_step) ───────────────────────
struct WireState {
    DSv4RoutedState* routed;             // D5 state
    int routed_init_for_n_layers;

    // HC scratch buffers (per-token)
    __nv_bfloat16* hc_collapsed;         // [HIDDEN] BF16 — block input
    float*         hc_pre;               // [N_HC]
    float*         hc_post;              // [N_HC]
    float*         hc_comb;              // [N_HC, N_HC]
    __nv_bfloat16* hc_norm_scratch;      // [N_HC*HIDDEN] BF16
    float*         hc_mixes_scratch;     // [LAYER_MIX, HIDDEN] FP32 (max LAYER_MIX=24)
    __nv_bfloat16* hc_post_scratch;      // [N_HC*HIDDEN] BF16
    float*         hc_head_pre_scratch;  // [N_HC]

    // HC residual stream [N_HC, HIDDEN]
    __nv_bfloat16* x_residual;           // owned

    // logits output (BF16 for D7 lm_head)
    __nv_bfloat16* logits_bf16;          // [VOCAB]

    // RoPE freqs split (real + imag) — D3 expects two separate buffers
    float* freqs_real_base;              // [MAX_SEQ, ROPE_DIM/2]
    float* freqs_imag_base;
    float* freqs_real_compress;
    float* freqs_imag_compress;
    int freqs_max_seq;

    // MLA scratch
    __nv_bfloat16* q_full_buf;           // [NUM_HEADS, HEAD_DIM]
    float*         scores_buf;           // [NUM_HEADS, P+1]
    int            scores_capacity;
    __nv_bfloat16* attn_per_head_buf;    // [NUM_HEADS, HEAD_DIM]
    __nv_bfloat16* o_lora_buf;           // [N_GROUPS * O_LORA_RANK]

    // Hot-set wired flag
    int hotset_initialized;
    int routed_hotset_wired;

    // Stream
    cudaStream_t stream;
};

static WireState g_wire = {0};

// Per-layer compress ratio (mirror D2)
static const int WIRE_CR[43] = {
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 0
};

// ─── Hot-set callback (D6 -> D5) ───────────────────────────────────────
static const uint8_t* wire_hotset_get_packed(void* /*user*/, int L, int E, int wkey) {
    ExpertGpuPtrs out = {0};
    int rc = hotset_get_expert_gpu_ptr(L, E, &out);
    if (rc != 0) {
        std::fprintf(stderr, "[D9-wire] hotset_get_expert_gpu_ptr(%d,%d) rc=%d\n", L, E, rc);
        return nullptr;
    }
    if (wkey < 0 || wkey >= 3) return nullptr;
    return reinterpret_cast<const uint8_t*>(out.packed[wkey]);
}
static const uint8_t* wire_hotset_get_scale(void* /*user*/, int L, int E, int wkey) {
    ExpertGpuPtrs out = {0};
    int rc = hotset_get_expert_gpu_ptr(L, E, &out);
    if (rc != 0) return nullptr;
    if (wkey < 0 || wkey >= 3) return nullptr;
    return reinterpret_cast<const uint8_t*>(out.scale[wkey]);
}

// ─── Lazy wire init ────────────────────────────────────────────────────
static int wire_init(DSv4EngineMxfp4State* st) {
    if (g_wire.routed) return 0;  // already initialized
    g_wire.stream = st->stream;

    int n_layers = st->n_layers;

    // Routed state
    g_wire.routed = dsv4_routed_state_alloc(n_layers, st->stream);
    if (!g_wire.routed) {
        std::fprintf(stderr, "[D9-wire] dsv4_routed_state_alloc failed\n");
        return -1;
    }
    g_wire.routed_init_for_n_layers = n_layers;

    // Bind router_gate per layer
    for (int L = 0; L < n_layers; ++L) {
        dsv4_routed_set_router_gate(g_wire.routed, L, st->layers[L].router_gate);
    }

    // Hot-set init (best-effort; engine continues even if missing — routed will ABORT)
    int hs_rc = hotset_init(n_layers, /*n_experts=*/256, /*hot_per_layer=*/64);
    if (hs_rc == 0) {
        g_wire.hotset_initialized = 1;
        // D28A: tell hotset which stream consumes the slot data so that miss
        // path can cudaStreamWaitEvent that stream against per-slot
        // copy_event (no CPU sync, GPU-side ordering only).
        hotset_set_consumer_stream((void*)g_wire.stream);
        // Wire callback to routed
        dsv4_routed_set_hotset(g_wire.routed,
                               wire_hotset_get_packed,
                               wire_hotset_get_scale,
                               /*user=*/nullptr);
        g_wire.routed_hotset_wired = 1;
    } else {
        std::fprintf(stderr, "[D9-wire] WARN: hotset_init rc=%d (routed will need register_cold)\n", hs_rc);
    }

    // Allocate HC scratch buffers
    cudaMalloc(&g_wire.hc_collapsed,        WIRE_HIDDEN * sizeof(__nv_bfloat16));
    cudaMalloc(&g_wire.hc_pre,              WIRE_HC_N_HC * sizeof(float));
    cudaMalloc(&g_wire.hc_post,             WIRE_HC_N_HC * sizeof(float));
    cudaMalloc(&g_wire.hc_comb,             WIRE_HC_N_HC * WIRE_HC_N_HC * sizeof(float));
    cudaMalloc(&g_wire.hc_norm_scratch,     WIRE_HC_N_HC * WIRE_HIDDEN * sizeof(__nv_bfloat16));
    cudaMalloc(&g_wire.hc_mixes_scratch,    24 * WIRE_HIDDEN * sizeof(float));  // max LAYER_MIX=24
    cudaMalloc(&g_wire.hc_post_scratch,     WIRE_HC_N_HC * WIRE_HIDDEN * sizeof(__nv_bfloat16));
    cudaMalloc(&g_wire.hc_head_pre_scratch, WIRE_HC_N_HC * sizeof(float));

    // HC residual stream [N_HC, HIDDEN]
    cudaMalloc(&g_wire.x_residual, WIRE_HC_N_HC * WIRE_HIDDEN * sizeof(__nv_bfloat16));

    // logits output (BF16 — D7 lm_head writes BF16)
    cudaMalloc(&g_wire.logits_bf16, WIRE_VOCAB * sizeof(__nv_bfloat16));

    // MLA scratch
    cudaMalloc(&g_wire.q_full_buf,        WIRE_NUM_HEADS * WIRE_HEAD_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&g_wire.attn_per_head_buf, WIRE_NUM_HEADS * WIRE_HEAD_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&g_wire.o_lora_buf,        WIRE_N_GROUPS * WIRE_O_LORA_RANK * sizeof(__nv_bfloat16));

    // Initial scores capacity (will grow on demand)
    g_wire.scores_capacity = 1024;
    cudaMalloc(&g_wire.scores_buf, WIRE_NUM_HEADS * (g_wire.scores_capacity + 1) * sizeof(float));

    // Split RoPE freqs (D2 stored as interleaved pairs; D3 expects real/imag separate)
    int half = WIRE_QK_ROPE_DIM / 2;
    int max_seq = 4096;  // matches D2 MAX_SEQ_LEN_INIT
    g_wire.freqs_max_seq = max_seq;
    size_t fbytes = max_seq * half * sizeof(float);
    cudaMalloc(&g_wire.freqs_real_base,     fbytes);
    cudaMalloc(&g_wire.freqs_imag_base,     fbytes);
    cudaMalloc(&g_wire.freqs_real_compress, fbytes);
    cudaMalloc(&g_wire.freqs_imag_compress, fbytes);

    // Read interleaved D2 buffers and split (host roundtrip — only at init)
    float* h_pair = (float*)malloc(2 * fbytes);
    if (!h_pair) { std::fprintf(stderr, "[D9-wire] OOM rope split\n"); return -2; }
    float* h_re = (float*)malloc(fbytes);
    float* h_im = (float*)malloc(fbytes);
    if (!h_re || !h_im) { free(h_pair); return -3; }

    auto split_pair = [&](float* d_pair, float* d_real, float* d_imag) {
        cudaMemcpy(h_pair, d_pair, 2 * fbytes, cudaMemcpyDeviceToHost);
        for (int i = 0; i < max_seq * half; ++i) {
            h_re[i] = h_pair[i*2 + 0];
            h_im[i] = h_pair[i*2 + 1];
        }
        cudaMemcpy(d_real, h_re, fbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_imag, h_im, fbytes, cudaMemcpyHostToDevice);
    };
    split_pair(st->freqs_cis_base,     g_wire.freqs_real_base,     g_wire.freqs_imag_base);
    split_pair(st->freqs_cis_compress, g_wire.freqs_real_compress, g_wire.freqs_imag_compress);
    free(h_pair); free(h_re); free(h_im);

    // Reset residual stream to zero (initial decode state)
    cudaMemset(g_wire.x_residual, 0, WIRE_HC_N_HC * WIRE_HIDDEN * sizeof(__nv_bfloat16));
    // Sync to ensure all init allocations + memsets + memcpys complete before
    // engine stream begins decode.
    cudaDeviceSynchronize();

    std::fprintf(stderr, "[D9-wire] init OK (n_layers=%d, hotset=%d)\n",
                 n_layers, g_wire.hotset_initialized);
    return 0;
}

static void wire_grow_scores(int P) {
    if (P + 1 <= g_wire.scores_capacity) return;
    int new_cap = g_wire.scores_capacity;
    while (new_cap < P + 1) new_cap *= 2;
    if (g_wire.scores_buf) cudaFree(g_wire.scores_buf);
    cudaMalloc(&g_wire.scores_buf, WIRE_NUM_HEADS * (new_cap + 1) * sizeof(float));
    g_wire.scores_capacity = new_cap;
}

// ─── FP8 GEMV thin shim around D2's dlopen'd fn ────────────────────────
static inline int wire_fp8_gemv(
    DSv4EngineMxfp4State* st,
    const __nv_bfloat16* x,
    const uint8_t* W_fp8, const uint8_t* W_scale,
    int K, int N,
    int64_t W_row_stride, int64_t S_row_stride,
    __nv_bfloat16* y)
{
    if (!st->fn_fp8_e4m3_gemv_m1) {
        std::fprintf(stderr, "[D9-wire] FP8 GEMV fn pointer NULL\n");
        return -1;
    }
    return st->fn_fp8_e4m3_gemv_m1(x, W_fp8, W_scale, K, N,
                                   W_row_stride, S_row_stride,
                                   y, st->stream);
}

// ─── MLA per-layer forward (single-token, decode) ──────────────────────
//   x_in [HIDDEN] BF16  -> x_out [HIDDEN] BF16
//   uses kv_cache_layer [max_seq, KV_LORA_RANK] BF16, cur_pos
// ─── D23: target-filtered MLA sub-block dumps (q pre/post-RoPE, attn_out) ───
struct D23State {
    int target_layer;
    __nv_bfloat16* buf_q_pre_rope;   // [NUM_HEADS * HEAD_DIM = 32768]
    __nv_bfloat16* buf_q_post_rope;
    __nv_bfloat16* buf_attn_out;
    int alloc_done;
};
static D23State g_d23 = {-2, nullptr, nullptr, nullptr, 0};

extern "C" int dsv4_mxfp4_d23_set_target(int L) { g_d23.target_layer = L; return 0; }
extern "C" int dsv4_mxfp4_d23_alloc() {
    if (g_d23.alloc_done) return 0;
    size_t sz = (size_t)WIRE_NUM_HEADS * WIRE_HEAD_DIM * sizeof(__nv_bfloat16);
    if (cudaMalloc(&g_d23.buf_q_pre_rope, sz)  != cudaSuccess) return -1;
    if (cudaMalloc(&g_d23.buf_q_post_rope, sz) != cudaSuccess) return -2;
    if (cudaMalloc(&g_d23.buf_attn_out, sz)    != cudaSuccess) return -3;
    g_d23.alloc_done = 1;
    return 0;
}
extern "C" const void* dsv4_mxfp4_d23_get_q_pre_rope(void)  { return (const void*)g_d23.buf_q_pre_rope; }
extern "C" const void* dsv4_mxfp4_d23_get_q_post_rope(void) { return (const void*)g_d23.buf_q_post_rope; }
extern "C" const void* dsv4_mxfp4_d23_get_attn_out(void)    { return (const void*)g_d23.buf_attn_out; }

static int wire_mla_layer(
    DSv4EngineMxfp4State* st, int L,
    const __nv_bfloat16* x_in, __nv_bfloat16* x_out, int cur_pos)
{
    LayerWeightsMxfp4* ly = &st->layers[L];
    cudaStream_t s = st->stream;

    // Q path: low-rank
    int rc;
    rc = wire_fp8_gemv(st, x_in, ly->wq_a_w, ly->wq_a_s,
                       WIRE_HIDDEN, WIRE_Q_LORA_RANK,
                       ly->wq_a_w_row_stride, ly->wq_a_s_row_stride,
                       st->q_a_buf);
    if (rc) return rc;
    // q_norm RMS on q_a [Q_LORA_RANK]
    rc = rms_norm_inline(st->q_a_buf, ly->q_norm, st->q_a_buf,
                         WIRE_Q_LORA_RANK, WIRE_RMS_EPS, s);
    if (rc) return rc;
    rc = wire_fp8_gemv(st, st->q_a_buf, ly->wq_b_w, ly->wq_b_s,
                       WIRE_Q_LORA_RANK, WIRE_NUM_HEADS * WIRE_HEAD_DIM,
                       ly->wq_b_w_row_stride, ly->wq_b_s_row_stride,
                       g_wire.q_full_buf);
    if (rc) return rc;
    // per-head Q rmsnorm
    rc = mla_q_per_head_rmsn(g_wire.q_full_buf, WIRE_HEAD_DIM, WIRE_RMS_EPS, s);
    if (rc) return rc;

    // KV path: shared latent
    rc = wire_fp8_gemv(st, x_in, ly->wkv_w, ly->wkv_s,
                       WIRE_HIDDEN, WIRE_KV_LORA_RANK,
                       ly->wkv_w_row_stride, ly->wkv_s_row_stride,
                       st->kv_proj_buf);
    if (rc) return rc;
    rc = rms_norm_inline(st->kv_proj_buf, ly->kv_norm, st->kv_proj_buf,
                         WIRE_KV_LORA_RANK, WIRE_RMS_EPS, s);
    if (rc) return rc;

    // Choose RoPE bank by per-layer compress ratio
    const float* fr = (st->compress_ratios[L] != 0)
        ? g_wire.freqs_real_compress : g_wire.freqs_real_base;
    const float* fi = (st->compress_ratios[L] != 0)
        ? g_wire.freqs_imag_compress : g_wire.freqs_imag_base;

    // D23 capture q_full PRE-RoPE
    if (g_d23.target_layer == L && g_d23.alloc_done) {
        cudaMemcpyAsync(g_d23.buf_q_pre_rope, g_wire.q_full_buf,
                        (size_t)WIRE_NUM_HEADS * WIRE_HEAD_DIM * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, s);
    }
    // RoPE on Q (per head) and KV
    rc = mla_rope_apply_q(g_wire.q_full_buf, fr, fi, cur_pos,
                          WIRE_HEAD_DIM, WIRE_QK_ROPE_DIM, /*inverse=*/0, s);
    if (rc) return rc;
    // D23 capture q_full POST-RoPE
    if (g_d23.target_layer == L && g_d23.alloc_done) {
        cudaMemcpyAsync(g_d23.buf_q_post_rope, g_wire.q_full_buf,
                        (size_t)WIRE_NUM_HEADS * WIRE_HEAD_DIM * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, s);
    }
    rc = mla_rope_apply_kv(st->kv_proj_buf, fr, fi, cur_pos,
                           WIRE_KV_LORA_RANK, WIRE_QK_ROPE_DIM, s);
    if (rc) return rc;

    // KV cache append
    rc = mla_kv_append(st->kv_cache[L], st->kv_proj_buf, cur_pos,
                       WIRE_KV_LORA_RANK, s);
    if (rc) return rc;

    int P = cur_pos + 1;
    wire_grow_scores(P);

    // attention scores [NUM_HEADS, P+1]
    rc = mla_full_dot(g_wire.q_full_buf, st->kv_cache[L], ly->attn_sink,
                      P, WIRE_HEAD_DIM, WIRE_KV_LORA_RANK,
                      g_wire.scores_buf, s);
    if (rc) return rc;
    rc = mla_full_mask(g_wire.scores_buf, P, cur_pos, WIRE_WINDOW_SIZE, s);
    if (rc) return rc;
    rc = mla_full_softmax(g_wire.scores_buf, P, s);
    if (rc) return rc;

    // attend
    rc = mla_full_attend(g_wire.scores_buf, st->kv_cache[L],
                         P, WIRE_HEAD_DIM, WIRE_KV_LORA_RANK,
                         g_wire.attn_per_head_buf, s);
    if (rc) return rc;
    // D23 capture attn_out (post-attend, before inverse RoPE)
    if (g_d23.target_layer == L && g_d23.alloc_done) {
        cudaMemcpyAsync(g_d23.buf_attn_out, g_wire.attn_per_head_buf,
                        (size_t)WIRE_NUM_HEADS * WIRE_HEAD_DIM * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, s);
    }

    // inverse RoPE on attn_out
    rc = mla_rope_inverse_o(g_wire.attn_per_head_buf, fr, fi, cur_pos,
                            WIRE_HEAD_DIM, WIRE_QK_ROPE_DIM, s);
    if (rc) return rc;

    // grouped output proj: 8 × FP8 GEMV into wo_a slices
    // wo_a [N=8192, K=4096] : per group g, slice wo_a[g*1024:(g+1)*1024, :]
    int rows_per_group = WIRE_O_LORA_RANK;            // 1024
    int scale_rows_per_g = rows_per_group / 128;       // 8
    for (int g = 0; g < WIRE_N_GROUPS; ++g) {
        const __nv_bfloat16* attn_g = g_wire.attn_per_head_buf + g * WIRE_GROUP_DIN;
        const uint8_t* W_g = ly->wo_a_w + (size_t)g * rows_per_group * ly->wo_a_w_row_stride;
        const uint8_t* S_g = ly->wo_a_s + (size_t)g * scale_rows_per_g * ly->wo_a_s_row_stride;
        __nv_bfloat16* y_g = g_wire.o_lora_buf + g * WIRE_O_LORA_RANK;
        rc = wire_fp8_gemv(st, attn_g, W_g, S_g,
                           WIRE_GROUP_DIN, WIRE_O_LORA_RANK,
                           ly->wo_a_w_row_stride, ly->wo_a_s_row_stride,
                           y_g);
        if (rc) return rc;
    }

    // wo_b: single FP8 GEMV [HIDDEN, N_GROUPS*O_LORA_RANK]
    rc = wire_fp8_gemv(st, g_wire.o_lora_buf, ly->wo_b_w, ly->wo_b_s,
                       WIRE_N_GROUPS * WIRE_O_LORA_RANK, WIRE_HIDDEN,
                       ly->wo_b_w_row_stride, ly->wo_b_s_row_stride,
                       x_out);
    return rc;
}

// ─── Shared expert FFN (FP8 GEMV gate/up/down + SwiGLU) ───────────────
//   x_in [HIDDEN] BF16 -> x_out [HIDDEN] BF16
__global__ void wire_swiglu_kernel(__nv_bfloat16* gate, const __nv_bfloat16* up,
                                   __nv_bfloat16* out, int n, float clamp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    // SwiGLU: silu(gate) * up. With a clamp on gate (DSv4 SWIGLU_LIMIT=10).
    if (g >  clamp) g =  clamp;
    if (g < -clamp) g = -clamp;
    float silu = g / (1.0f + expf(-g));
    out[i] = __float2bfloat16(silu * u);
}

static int wire_shared_ffn(DSv4EngineMxfp4State* st, int L,
                           const __nv_bfloat16* x_in, __nv_bfloat16* x_out)
{
    LayerWeightsMxfp4* ly = &st->layers[L];
    int rc;

    // gate: [EXPERT_FF, HIDDEN]
    rc = wire_fp8_gemv(st, x_in, ly->sh_w1_w, ly->sh_w1_s,
                       WIRE_HIDDEN, /*EXPERT_FF*/2048,
                       ly->sh_w1_w_row_stride, ly->sh_w1_s_row_stride,
                       st->sh_gate_buf);
    if (rc) return rc;
    // up: [EXPERT_FF, HIDDEN]
    rc = wire_fp8_gemv(st, x_in, ly->sh_w3_w, ly->sh_w3_s,
                       WIRE_HIDDEN, /*EXPERT_FF*/2048,
                       ly->sh_w3_w_row_stride, ly->sh_w3_s_row_stride,
                       st->sh_up_buf);
    if (rc) return rc;
    // SwiGLU activation
    int blk = 256, grid = (2048 + blk - 1) / blk;
    wire_swiglu_kernel<<<grid, blk, 0, st->stream>>>(
        st->sh_gate_buf, st->sh_up_buf, st->sh_act_buf, 2048, /*clamp*/10.0f);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[D9-wire] SwiGLU launch fail: %s\n", cudaGetErrorString(err));
        return -1;
    }
    // down: [HIDDEN, EXPERT_FF]
    rc = wire_fp8_gemv(st, st->sh_act_buf, ly->sh_w2_w, ly->sh_w2_s,
                       2048, WIRE_HIDDEN,
                       ly->sh_w2_w_row_stride, ly->sh_w2_s_row_stride,
                       x_out);
    return rc;
}

// ─── BF16 add-in-place (for residual MoE accumulation) ─────────────────
__global__ void wire_bf16_add_kernel(__nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}
static inline int wire_bf16_add(__nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t s) {
    int blk = 256, grid = (n + blk - 1) / blk;
    wire_bf16_add_kernel<<<grid, blk, 0, s>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ─── Real decode_step (overrides D2 stub via separate symbol; D2 stub
// caller can be patched to call us). To avoid linker conflict we expose a
// new symbol `dsv4_mxfp4_decode_step_real` and Python binding picks it up. ─

// ───────────────────── D18 layer-by-layer dump instrumentation ─────────────────────
// Captures intermediate forward-pass tensors at chosen layer L for R2 vs C++ bisect.
// target_layer == -1 means "capture pre-L0 only (post-embed+broadcast)".
// target_layer in [0..42] means "capture all 8 intra-layer block outputs at layer L".
// Buffers are pre-allocated once (alloc_buffers); capture is a cudaMemcpyAsync if matching.
// NO PERF impact when target_layer == -2 (off).
struct D18State {
    int target_layer;
    int dumped_pre_l0;
    __nv_bfloat16* buf_pre_l0;        // [4*4096]
    __nv_bfloat16* buf_hc_pre_attn;   // [4096]
    __nv_bfloat16* buf_rms_attn;      // [4096]
    __nv_bfloat16* buf_mla_out;       // [4096]
    __nv_bfloat16* buf_post_attn;     // [4*4096]
    __nv_bfloat16* buf_hc_pre_ffn;    // [4096]
    __nv_bfloat16* buf_rms_ffn;       // [4096]
    __nv_bfloat16* buf_moe_out;       // [4096] (post shared+routed merge)
    __nv_bfloat16* buf_post_ffn;      // [4*4096] (end-of-layer L)
    int buffers_alloc_done;
};
static D18State g_d18 = {-2, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0};

extern "C" int dsv4_mxfp4_d18_set_target_layer(int L) {
    g_d18.target_layer = L;
    g_d18.dumped_pre_l0 = 0;
    return 0;
}

extern "C" int dsv4_mxfp4_d18_alloc_buffers() {
    if (g_d18.buffers_alloc_done) return 0;
    cudaError_t e;
    e = cudaMalloc(&g_d18.buf_pre_l0,        4*4096*sizeof(__nv_bfloat16)); if (e) return -1;
    e = cudaMalloc(&g_d18.buf_hc_pre_attn,   4096*sizeof(__nv_bfloat16));   if (e) return -2;
    e = cudaMalloc(&g_d18.buf_rms_attn,      4096*sizeof(__nv_bfloat16));   if (e) return -3;
    e = cudaMalloc(&g_d18.buf_mla_out,       4096*sizeof(__nv_bfloat16));   if (e) return -4;
    e = cudaMalloc(&g_d18.buf_post_attn,     4*4096*sizeof(__nv_bfloat16)); if (e) return -5;
    e = cudaMalloc(&g_d18.buf_hc_pre_ffn,    4096*sizeof(__nv_bfloat16));   if (e) return -6;
    e = cudaMalloc(&g_d18.buf_rms_ffn,       4096*sizeof(__nv_bfloat16));   if (e) return -7;
    e = cudaMalloc(&g_d18.buf_moe_out,       4096*sizeof(__nv_bfloat16));   if (e) return -8;
    e = cudaMalloc(&g_d18.buf_post_ffn,      4*4096*sizeof(__nv_bfloat16)); if (e) return -9;
    g_d18.buffers_alloc_done = 1;
    return 0;
}

extern "C" const void* dsv4_mxfp4_d18_get_pre_l0(void)      { return (const void*)g_d18.buf_pre_l0; }
extern "C" const void* dsv4_mxfp4_d18_get_hc_pre_attn(void) { return (const void*)g_d18.buf_hc_pre_attn; }
extern "C" const void* dsv4_mxfp4_d18_get_rms_attn(void)    { return (const void*)g_d18.buf_rms_attn; }
extern "C" const void* dsv4_mxfp4_d18_get_mla_out(void)     { return (const void*)g_d18.buf_mla_out; }
extern "C" const void* dsv4_mxfp4_d18_get_post_attn(void)   { return (const void*)g_d18.buf_post_attn; }
extern "C" const void* dsv4_mxfp4_d18_get_hc_pre_ffn(void)  { return (const void*)g_d18.buf_hc_pre_ffn; }
extern "C" const void* dsv4_mxfp4_d18_get_rms_ffn(void)     { return (const void*)g_d18.buf_rms_ffn; }
extern "C" const void* dsv4_mxfp4_d18_get_moe_out(void)     { return (const void*)g_d18.buf_moe_out; }
extern "C" const void* dsv4_mxfp4_d18_get_post_ffn(void)    { return (const void*)g_d18.buf_post_ffn; }

static inline void d18_capture(int L_target, int L, void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    if (g_d18.target_layer != L) return;
    if (!g_d18.buffers_alloc_done) return;
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
}

// Embed-only short-circuit: runs embed_forward + 3x cudaMemcpyAsync HC broadcast,
// no layer forward. Used by D18a to verify cos hc0==hc1==hc2==hc3 == 1.0 exactly.
extern "C" int dsv4_mxfp4_d18_embed_only(DSv4EngineMxfp4State* st, int token_id) {
    if (wire_init(st) != 0) return -1;
    cudaMemsetAsync(g_wire.x_residual, 0,
                    (size_t)WIRE_HC_N_HC * WIRE_HIDDEN * sizeof(__nv_bfloat16),
                    st->stream);
    int rc = embed_forward(st->embed, token_id, WIRE_VOCAB, WIRE_HIDDEN,
                           g_wire.x_residual, st->stream);
    if (rc != 0) return rc;
    size_t row_bytes = (size_t)WIRE_HIDDEN * sizeof(__nv_bfloat16);
    for (int hc = 1; hc < WIRE_HC_N_HC; ++hc) {
        cudaError_t e = cudaMemcpyAsync(
            (char*)g_wire.x_residual + (size_t)hc * row_bytes,
            g_wire.x_residual, row_bytes,
            cudaMemcpyDeviceToDevice, st->stream);
        if (e != cudaSuccess) return -2;
    }
    cudaStreamSynchronize(st->stream);
    return 0;
}


// ───────────────────── D19 sub-bisect: MoE shared/routed split ─────────────────────
struct D19State {
    int target_layer;
    __nv_bfloat16* buf_shared_out;     // [4096] pre-add
    __nv_bfloat16* buf_routed_out;     // [4096] pre-add (post dispatch, before wire_bf16_add)
    int alloc_done;
};
static D19State g_d19 = {-2, nullptr, nullptr, 0};

extern "C" int dsv4_mxfp4_d19_set_target_layer(int L) {
    g_d19.target_layer = L;
    return 0;
}

extern "C" int dsv4_mxfp4_d19_alloc_buffers() {
    if (g_d19.alloc_done) return 0;
    if (cudaMalloc(&g_d19.buf_shared_out, 4096 * sizeof(__nv_bfloat16)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_d19.buf_routed_out, 4096 * sizeof(__nv_bfloat16)) != cudaSuccess) return -2;
    g_d19.alloc_done = 1;
    return 0;
}

extern "C" const void* dsv4_mxfp4_d19_get_shared_out(void) { return (const void*)g_d19.buf_shared_out; }
extern "C" const void* dsv4_mxfp4_d19_get_routed_out(void) { return (const void*)g_d19.buf_routed_out; }

// Helper to access routed state pointer (so the harness can call routed getters)
extern "C" void* dsv4_mxfp4_d19_get_routed_state(void) { return (void*)g_wire.routed; }

extern "C" int dsv4_mxfp4_decode_step_real(DSv4EngineMxfp4State* st, int token_id) {
    if (!st) return -1;

    // Lazy init
    if (wire_init(st) != 0) {
        std::fprintf(stderr, "[D9-wire] wire_init failed\n");
        return -2;
    }

    int n_layers = st->n_layers;
    int cur_pos = st->cur_pos;

    // 1. Embed token into x_residual stream 0 (tile across all 4 streams)
    //    R2 starts with [HIDDEN] embed; HC residual stream is [N_HC, HIDDEN].
    //    Embed → write to slot 0; other slots zero (already cleared at init).
    int rc = embed_forward(st->embed, token_id, WIRE_VOCAB, WIRE_HIDDEN,
                           g_wire.x_residual, st->stream);
    if (rc != 0) {
        std::fprintf(stderr, "[D9-wire] embed_forward rc=%d\n", rc);
        return -3;
    }
    // D17-B HC broadcast fix: tile slot 0 -> slots 1,2,3.
    // R2 reference (runtime/dsv4_engine_mxfp4.py:585):
    //     hidden = emb.unsqueeze(1).repeat(1, HC_N_HC, 1).contiguous()
    // Without broadcast hc_pre sees x_collapsed = pre[0]*emb at L0 instead
    // of (sum_h pre[h])*emb -> sticky [emb,0,0,0] drift accumulates per token.
    {
        // D17-B-DIAG: confirm fix block runs at first token
        static int d17b_diag_done = 0;
        if (!d17b_diag_done) {
            fprintf(stderr, "[D17B-fix-active] HC broadcast block ENTERED token=%d cur_pos=%d\n",
                    token_id, st->cur_pos);
            d17b_diag_done = 1;
        }
        size_t row_bytes = (size_t)WIRE_HIDDEN * sizeof(__nv_bfloat16);
        for (int hc = 1; hc < WIRE_HC_N_HC; ++hc) {
            cudaError_t e = cudaMemcpyAsync(
                (char*)g_wire.x_residual + (size_t)hc * row_bytes,
                g_wire.x_residual, row_bytes,
                cudaMemcpyDeviceToDevice, st->stream);
            if (e != cudaSuccess) {
                std::fprintf(stderr, "[D9-wire] hc broadcast hc=%d %s\n",
                             hc, cudaGetErrorString(e));
                return -3;
            }
        }
    }

    // D18 capture pre-L0 (post embed + HC broadcast). Special block_id = -1.
    if (g_d18.target_layer == -1 && !g_d18.dumped_pre_l0 && g_d18.buffers_alloc_done) {
        cudaMemcpyAsync(g_d18.buf_pre_l0, g_wire.x_residual,
                        (size_t)WIRE_HC_N_HC * WIRE_HIDDEN * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, st->stream);
        g_d18.dumped_pre_l0 = 1;
    }

    // 2. 43 layers
    for (int L = 0; L < n_layers; ++L) {
        LayerWeightsMxfp4* ly = &st->layers[L];

        // ── ATTENTION HALF ──
        // HC pre (collapse residual [N_HC, HIDDEN] -> hc_collapsed [HIDDEN])
        rc = hc_pre_forward(g_wire.x_residual,
                            ly->hc_attn_fn, ly->hc_attn_base, ly->hc_attn_scale,
                            g_wire.hc_collapsed,
                            g_wire.hc_pre, g_wire.hc_post, g_wire.hc_comb,
                            g_wire.hc_norm_scratch, g_wire.hc_mixes_scratch,
                            st->stream);
        if (rc) { std::fprintf(stderr, "[D9-wire] L%d hc_pre rc=%d\n", L, rc); return -10; }
        d18_capture(g_d18.target_layer, L, g_d18.buf_hc_pre_attn, g_wire.hc_collapsed,
                    (size_t)WIRE_HIDDEN * sizeof(__nv_bfloat16), st->stream);

        // RMS attn_norm
        rc = rms_norm_inline(g_wire.hc_collapsed, ly->attn_norm, st->h_norm,
                             WIRE_HIDDEN, WIRE_RMS_EPS, st->stream);
        if (rc) return -11;
        d18_capture(g_d18.target_layer, L, g_d18.buf_rms_attn, st->h_norm,
                    (size_t)WIRE_HIDDEN * sizeof(__nv_bfloat16), st->stream);

        // MLA full attention -> attn_out
        rc = wire_mla_layer(st, L, st->h_norm, st->attn_out, cur_pos);
        if (rc) { std::fprintf(stderr, "[D9-wire] L%d MLA rc=%d\n", L, rc); return -12; }
        d18_capture(g_d18.target_layer, L, g_d18.buf_mla_out, st->attn_out,
                    (size_t)WIRE_HIDDEN * sizeof(__nv_bfloat16), st->stream);

        // HC post (distribute attn_out into residual)
        rc = hc_post_forward(st->attn_out, g_wire.x_residual,
                             g_wire.hc_post, g_wire.hc_comb,
                             g_wire.hc_post_scratch, st->stream);
        if (rc) return -13;
        d18_capture(g_d18.target_layer, L, g_d18.buf_post_attn, g_wire.x_residual,
                    (size_t)WIRE_HC_N_HC * WIRE_HIDDEN * sizeof(__nv_bfloat16), st->stream);

        // ── FFN HALF ──
        rc = hc_pre_forward(g_wire.x_residual,
                            ly->hc_ffn_fn, ly->hc_ffn_base, ly->hc_ffn_scale,
                            g_wire.hc_collapsed,
                            g_wire.hc_pre, g_wire.hc_post, g_wire.hc_comb,
                            g_wire.hc_norm_scratch, g_wire.hc_mixes_scratch,
                            st->stream);
        if (rc) return -14;
        d18_capture(g_d18.target_layer, L, g_d18.buf_hc_pre_ffn, g_wire.hc_collapsed,
                    (size_t)WIRE_HIDDEN * sizeof(__nv_bfloat16), st->stream);

        // RMS ffn_norm
        rc = rms_norm_inline(g_wire.hc_collapsed, ly->ffn_norm, st->h_norm,
                             WIRE_HIDDEN, WIRE_RMS_EPS, st->stream);
        if (rc) return -15;
        d18_capture(g_d18.target_layer, L, g_d18.buf_rms_ffn, st->h_norm,
                    (size_t)WIRE_HIDDEN * sizeof(__nv_bfloat16), st->stream);

        // Shared expert
        rc = wire_shared_ffn(st, L, st->h_norm, st->shared_out);
        if (rc) { std::fprintf(stderr, "[D9-wire] L%d shared rc=%d\n", L, rc); return -16; }
        // D19 capture shared_out pre-add
        if (g_d19.target_layer == L && g_d19.alloc_done) {
            cudaMemcpyAsync(g_d19.buf_shared_out, st->shared_out,
                            (size_t)WIRE_HIDDEN * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice, st->stream);
        }

        // Routed expert dispatch (D5) -> routed_out
        if (g_wire.routed_hotset_wired) {
            rc = dsv4_routed_dispatch(g_wire.routed, L, st->h_norm, st->routed_out);
            if (rc != 0) {
                int af = dsv4_routed_get_abort_flag(g_wire.routed);
                std::fprintf(stderr, "[D9-wire] L%d routed_dispatch rc=%d abort_flag=%d\n",
                             L, rc, af);
                return -17;  // HARD ABORT — no silent skip
            }
            int dc = dsv4_routed_get_dispatch_count(g_wire.routed);
            if (dc != 6) {
                std::fprintf(stderr, "[D9-wire] L%d dispatch_count=%d (expected 6)\n", L, dc);
                return -18;
            }
            // D19 capture routed_out pre-add (post dispatch, BEFORE wire_bf16_add)
            if (g_d19.target_layer == L && g_d19.alloc_done) {
                cudaMemcpyAsync(g_d19.buf_routed_out, st->routed_out,
                                (size_t)WIRE_HIDDEN * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToDevice, st->stream);
            }
            // moe_out = shared_out + routed_out
            rc = wire_bf16_add(st->routed_out, st->shared_out, WIRE_HIDDEN, st->stream);
            if (rc) return -19;
            d18_capture(g_d18.target_layer, L, g_d18.buf_moe_out, st->routed_out,
                        (size_t)WIRE_HIDDEN * sizeof(__nv_bfloat16), st->stream);
            // HC post (distribute routed+shared into residual)
            rc = hc_post_forward(st->routed_out, g_wire.x_residual,
                                 g_wire.hc_post, g_wire.hc_comb,
                                 g_wire.hc_post_scratch, st->stream);
        } else {
            // No hot-set: fall back to shared-only (degraded, but keeps engine running for D10 bench harness)
            std::fprintf(stderr, "[D9-wire] L%d WARN: routed disabled (no hotset)\n", L);
            rc = hc_post_forward(st->shared_out, g_wire.x_residual,
                                 g_wire.hc_post, g_wire.hc_comb,
                                 g_wire.hc_post_scratch, st->stream);
        }
        if (rc) return -20;
        d18_capture(g_d18.target_layer, L, g_d18.buf_post_ffn, g_wire.x_residual,
                    (size_t)WIRE_HC_N_HC * WIRE_HIDDEN * sizeof(__nv_bfloat16), st->stream);
    }

    // 3. Final norm + HC head
    //    R2: hc_head_compute does (rmsnorm-no-affine -> mixer -> head_pre -> apply_pre)
    //    Then we apply final_norm (with weight) on the collapsed result.
    rc = hc_head_compute(g_wire.x_residual,
                         st->hc_head_fn, st->hc_head_base, st->hc_head_scale,
                         g_wire.hc_collapsed,
                         g_wire.hc_norm_scratch, g_wire.hc_mixes_scratch,
                         g_wire.hc_head_pre_scratch, st->stream);
    if (rc) { std::fprintf(stderr, "[D9-wire] hc_head rc=%d\n", rc); return -30; }

    rc = final_norm_forward(g_wire.hc_collapsed, st->final_norm, st->h_norm,
                            WIRE_HIDDEN, WIRE_RMS_EPS, st->stream);
    if (rc) return -31;

    // 4. lm_head BF16
    rc = lm_head_forward_bf16(st->head, st->h_norm, g_wire.logits_bf16,
                              WIRE_HIDDEN, WIRE_VOCAB, st->stream);
    if (rc) { std::fprintf(stderr, "[D9-wire] lm_head rc=%d\n", rc); return -32; }

    // 5. Greedy sample
    int next_token = 0;
    rc = sample_greedy(g_wire.logits_bf16, WIRE_VOCAB, &next_token, st->stream);
    if (rc) { std::fprintf(stderr, "[D9-wire] sample rc=%d\n", rc); return -33; }

    // Advance position
    st->cur_pos++;
    return next_token;
}

// ─── ABI bridge: override D2 stub via interceptor exposed under a new name ──
extern "C" int dsv4_mxfp4_decode_step_v2(DSv4EngineMxfp4State* st, int token_id) {
    return dsv4_mxfp4_decode_step_real(st, token_id);
}

// ─── Cleanup helper ────────────────────────────────────────────────────
extern "C" void dsv4_mxfp4_wire_cleanup() {
    if (g_wire.routed) { dsv4_routed_state_free(g_wire.routed); g_wire.routed = nullptr; }
    if (g_wire.hotset_initialized) { hotset_destroy(); g_wire.hotset_initialized = 0; }
    if (g_wire.hc_collapsed)        cudaFree(g_wire.hc_collapsed);
    if (g_wire.hc_pre)              cudaFree(g_wire.hc_pre);
    if (g_wire.hc_post)             cudaFree(g_wire.hc_post);
    if (g_wire.hc_comb)             cudaFree(g_wire.hc_comb);
    if (g_wire.hc_norm_scratch)     cudaFree(g_wire.hc_norm_scratch);
    if (g_wire.hc_mixes_scratch)    cudaFree(g_wire.hc_mixes_scratch);
    if (g_wire.hc_post_scratch)     cudaFree(g_wire.hc_post_scratch);
    if (g_wire.hc_head_pre_scratch) cudaFree(g_wire.hc_head_pre_scratch);
    if (g_wire.x_residual)          cudaFree(g_wire.x_residual);
    if (g_wire.logits_bf16)         cudaFree(g_wire.logits_bf16);
    if (g_wire.q_full_buf)          cudaFree(g_wire.q_full_buf);
    if (g_wire.scores_buf)          cudaFree(g_wire.scores_buf);
    if (g_wire.attn_per_head_buf)   cudaFree(g_wire.attn_per_head_buf);
    if (g_wire.o_lora_buf)          cudaFree(g_wire.o_lora_buf);
    if (g_wire.freqs_real_base)     cudaFree(g_wire.freqs_real_base);
    if (g_wire.freqs_imag_base)     cudaFree(g_wire.freqs_imag_base);
    if (g_wire.freqs_real_compress) cudaFree(g_wire.freqs_real_compress);
    if (g_wire.freqs_imag_compress) cudaFree(g_wire.freqs_imag_compress);
    memset(&g_wire, 0, sizeof(g_wire));
}


// ─── D17-B3 / D14 cherry-pick: dump getters for state inspection ──────
extern "C" const void* dsv4_mxfp4_d14_get_logits_ptr(void) {
    return (const void*)g_wire.logits_bf16;
}
extern "C" int dsv4_mxfp4_d14_get_vocab(void) {
    return WIRE_VOCAB;
}
extern "C" const void* dsv4_mxfp4_d14_get_x_residual_ptr(void) {
    return (const void*)g_wire.x_residual;
}
extern "C" const void* dsv4_mxfp4_d14_get_hnorm_ptr(DSv4EngineMxfp4State* st) {
    return st ? (const void*)st->h_norm : nullptr;
}
extern "C" int dsv4_mxfp4_d14_get_hidden(void) {
    return WIRE_HIDDEN;
}
extern "C" int dsv4_mxfp4_d14_get_n_hc(void) {
    return WIRE_HC_N_HC;
}
extern "C" int dsv4_mxfp4_d14_get_cur_pos(DSv4EngineMxfp4State* st) {
    return st ? st->cur_pos : -1;
}

// --- D24: YARN freqs verify (read-only getters, no buffer alloc) ---
extern "C" const void* dsv4_mxfp4_d24_get_freqs_real_base(void)     { return (const void*)g_wire.freqs_real_base; }
extern "C" const void* dsv4_mxfp4_d24_get_freqs_imag_base(void)     { return (const void*)g_wire.freqs_imag_base; }
extern "C" const void* dsv4_mxfp4_d24_get_freqs_real_compress(void) { return (const void*)g_wire.freqs_real_compress; }
extern "C" const void* dsv4_mxfp4_d24_get_freqs_imag_compress(void) { return (const void*)g_wire.freqs_imag_compress; }
extern "C" int         dsv4_mxfp4_d24_get_freqs_max_seq(void)       { return g_wire.freqs_max_seq; }
extern "C" int         dsv4_mxfp4_d24_get_freqs_half(void)          { return WIRE_QK_ROPE_DIM / 2; }

// EOF dsv4_engine_mxfp4_wire.cu
