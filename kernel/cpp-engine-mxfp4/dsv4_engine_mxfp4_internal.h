// dsv4_engine_mxfp4_internal.h — Shared definitions between D2 scaffold and
// D9 wire-up. Both dsv4_engine_mxfp4.cu and dsv4_engine_mxfp4_wire.cu include
// this header to see the SAME engine state struct.
//
// Author: D9 (extracted from D2 base for cross-TU sharing).

#ifndef DSV4_ENGINE_MXFP4_INTERNAL_H
#define DSV4_ENGINE_MXFP4_INTERNAL_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>

// ─── Architecture constants ─────────────────────────────────────────────
// (kept in sync with dsv4_engine_mxfp4.cu — defines stay there for D2's own
// sources; we redeclare with same names+values here so dsv4_engine_mxfp4.cu
// must NOT include this header to avoid duplicate macros. Wire-up file
// includes the header instead.)
//
// Constants used in struct definitions (must be #define so array bounds work).

#ifndef _DSV4_INTERNAL_H_CONSTS
#define _DSV4_INTERNAL_H_CONSTS
#define INTL_N_LAYERS              43
#define INTL_HIDDEN                4096
#define INTL_VOCAB                 129280
#define INTL_NUM_HEADS             64
#define INTL_KV_LORA_RANK          512
#define INTL_Q_LORA_RANK           1024
#define INTL_HEAD_DIM              512
#define INTL_QK_ROPE_DIM           64
#define INTL_N_GROUPS              8
#define INTL_O_LORA_RANK           1024
#define INTL_HOTSET_PER_LAYER      64
#endif

// ─── LayerWeightsMxfp4 (matches D2 layout exactly) ──────────────────────
struct LayerWeightsMxfp4 {
    int L;
    __nv_bfloat16* attn_norm;
    __nv_bfloat16* ffn_norm;
    __nv_bfloat16* q_norm;
    __nv_bfloat16* kv_norm;
    float*         attn_sink;

    uint8_t*  wkv_w;   uint8_t* wkv_s;
    uint8_t*  wq_a_w;  uint8_t* wq_a_s;
    uint8_t*  wq_b_w;  uint8_t* wq_b_s;
    uint8_t*  wo_a_w;  uint8_t* wo_a_s;
    uint8_t*  wo_b_w;  uint8_t* wo_b_s;

    int64_t wkv_w_row_stride;   int64_t wkv_s_row_stride;
    int64_t wq_a_w_row_stride;  int64_t wq_a_s_row_stride;
    int64_t wq_b_w_row_stride;  int64_t wq_b_s_row_stride;
    int64_t wo_a_w_row_stride;  int64_t wo_a_s_row_stride;
    int64_t wo_b_w_row_stride;  int64_t wo_b_s_row_stride;

    int wkv_out_dim;
    int wq_a_out_dim;
    int wq_b_out_dim;
    int wo_a_out_dim;
    int wo_b_out_dim;

    __nv_bfloat16* router_gate;
    int64_t*       tid2eid;

    uint8_t* sh_w1_w;  uint8_t* sh_w1_s;
    uint8_t* sh_w2_w;  uint8_t* sh_w2_s;
    uint8_t* sh_w3_w;  uint8_t* sh_w3_s;

    int64_t sh_w1_w_row_stride; int64_t sh_w1_s_row_stride;
    int64_t sh_w2_w_row_stride; int64_t sh_w2_s_row_stride;
    int64_t sh_w3_w_row_stride; int64_t sh_w3_s_row_stride;

    float* hc_attn_fn;
    float* hc_attn_base;
    float* hc_attn_scale;
    float* hc_ffn_fn;
    float* hc_ffn_base;
    float* hc_ffn_scale;

    int hotset_eids[INTL_HOTSET_PER_LAYER];
    int hotset_count;
};

// ─── DSv4EngineMxfp4State (matches D2 layout) ───────────────────────────
struct DSv4EngineMxfp4State {
    __nv_bfloat16* embed;
    __nv_bfloat16* head;
    __nv_bfloat16* final_norm;
    float*         hc_head_fn;
    float*         hc_head_base;
    float*         hc_head_scale;

    LayerWeightsMxfp4 layers[INTL_N_LAYERS];
    int n_layers;

    __nv_bfloat16* hc_stream;

    __nv_bfloat16* hidden;
    __nv_bfloat16* h_norm;
    __nv_bfloat16* attn_out;
    __nv_bfloat16* moe_out;
    __nv_bfloat16* shared_out;
    __nv_bfloat16* routed_out;

    __nv_bfloat16* q_a_buf;
    __nv_bfloat16* q_b_buf;
    __nv_bfloat16* kv_proj_buf;
    __nv_bfloat16* o_a_buf;

    __nv_bfloat16* sh_gate_buf;
    __nv_bfloat16* sh_up_buf;
    __nv_bfloat16* sh_act_buf;

    float* router_logits;
    float* router_scores;
    int*   topk_ids;
    float* topk_weights;

    float* logits;

    __nv_bfloat16* kv_cache[INTL_N_LAYERS];
    int max_seq;
    int cur_pos;

    uint8_t* hotset_w1_packed[INTL_N_LAYERS];
    uint8_t* hotset_w2_packed[INTL_N_LAYERS];
    uint8_t* hotset_w3_packed[INTL_N_LAYERS];
    uint8_t* hotset_w1_scale[INTL_N_LAYERS];
    uint8_t* hotset_w2_scale[INTL_N_LAYERS];
    uint8_t* hotset_w3_scale[INTL_N_LAYERS];

    float* freqs_cis_base;
    float* freqs_cis_compress;

    int compress_ratios[INTL_N_LAYERS];

    cudaStream_t stream;
    cublasHandle_t cublas;

    void* lib_fp8_dense;
    void* lib_mxfp4_dense;
    void* lib_mxfp4_routed;
    void* lib_rmsnorm;

    int (*fn_fp8_e4m3_gemv_m1)(
        const __nv_bfloat16* x,
        const uint8_t* W_fp8, const uint8_t* W_scale,
        int K, int N,
        int64_t weight_row_stride_bytes, int64_t scale_row_stride_bytes,
        __nv_bfloat16* y,
        cudaStream_t stream);

    int (*fn_mxfp4_grouped_gemv_topk)(
        const __nv_bfloat16* x,
        const uint8_t* W1_packed, const uint8_t* W3_packed, const uint8_t* W2_packed,
        const uint8_t* W1_scale,  const uint8_t* W3_scale,  const uint8_t* W2_scale,
        const float* rweights,
        int K_in, int hidden, int out_dim, int n_exp,
        __nv_bfloat16* out, void* workspace,
        cudaStream_t stream);
    size_t (*fn_mxfp4_grouped_workspace_bytes)(int n_exp, int hidden, int out_dim);

    int (*fn_rmsnorm_fwd)(const void* x, const void* weight, void* y,
                          int n_rows, int hidden, float eps, cudaStream_t stream);
    int (*fn_rmsnorm_residual_add)(const void* x, const void* residual, const void* weight,
                                    void* y, void* new_residual,
                                    int n_rows, int hidden, float eps, cudaStream_t stream);

    double init_time_seconds;
    double init_mem_gb;
};

#endif // DSV4_ENGINE_MXFP4_INTERNAL_H
