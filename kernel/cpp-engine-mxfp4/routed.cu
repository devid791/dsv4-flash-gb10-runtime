// routed.cu - DSv4-Flash MXFP4 routed expert dispatch (C++ pure)
// Project: MXFP4 /
// Authors: Davide Zenati
// License: MIT
//
// MANDATE: Wire routed expert dispatch in C++ pure using B3 grouped GEMV kernel.
// NO SILENT SKIP - root cause Sprint 6 fail. ABORT if expert_id out of range
// or hot-set probe fails for required expert.
//
// Reference: runtime/dsv4_engine_mxfp4.py linees 363-422 (R2 Python, 4/5 INTELLIGIBILE).
//
// Pipeline per token (per layer L):
//   1. gating logits   : g[256] = router_gate[256, HIDDEN] @ x_in[HIDDEN]      BF16
//   2. scores          : s[256] = sqrt(softplus(g))                            FP32
//   3. topK select     : (ids[6], scores[6]) = topk(s, K=6)
//   4. weights norm    : w[6] = scores / (sum(scores)+eps)
//   5. weights scale   : w[6] *= ROUTED_SCALING (1.5)
//   6. expert hot-set  : per id in ids[6], probe hot-set OR ABORT (no fallback here;
//                        D6 will add async upload). Verify id in [0, 256).
//   7. pack weights    : stack w1/w3/w2 packed+scale per expert into contiguous
//                        [n_exp=6, *, *] buffers
//   8. grouped GEMV    : call mxfp4_grouped_gemv_topk(...) -> out_routed[HIDDEN]
//   9. accumulate      : x_out += out_routed (caller will add shared expert)
//   10. sanity counter : state->routed_dispatch_count_per_token = 6 (ABORT if !=6)
//
// Constants (must mirror runtime/dsv4_engine_mxfp4.py):
//   HIDDEN=4096, N_EXPERTS=256, EXPERTS_PER_TOK=6, EXPERT_FF=2048, ROUTED_SCALING=1.5

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// === Constants (mirror Python R2) ===========================================
#define DSV4_HIDDEN          4096
#define DSV4_N_EXPERTS       256
#define DSV4_EXPERTS_PER_TOK 6
#define DSV4_EXPERT_FF       2048
#define DSV4_ROUTED_SCALING  1.5f
#define DSV4_TOPK_EPS        1e-20f

// === B3 kernel ABI (forward decl) ===========================================
extern "C" int mxfp4_grouped_gemv_topk(
    const __nv_bfloat16* x,
    const uint8_t* W1_packed, const uint8_t* W3_packed, const uint8_t* W2_packed,
    const uint8_t* W1_scale,  const uint8_t* W3_scale,  const uint8_t* W2_scale,
    const float*   rweights,
    int K_in, int hidden, int out_dim, int n_exp,
    __nv_bfloat16* out, void* workspace, cudaStream_t stream);

extern "C" size_t mxfp4_grouped_workspace_bytes(int n_exp, int hidden, int out_dim);

// === Hot-set probe ABI (provided by D6/C3 - here we declare a thin handle) ==
// Per ognuno dei 6 expert, l'oggetto hot-set deve fornire i puntatori device
// ai pesi contigui packed+scale. Se NON disponibile, returns NULL -> ABORT.
struct DSv4Hotset {
    // Function pointer table (iniettato da Python o C engine init)
    const uint8_t* (*get_w_packed)(void* user, int L, int E, int wname);
    const uint8_t* (*get_w_scale )(void* user, int L, int E, int wname);
    void* user;
};

// wname codes (matches R2 "w1"/"w3"/"w2" string keys)
enum { DSV4_W1 = 0, DSV4_W2 = 1, DSV4_W3 = 2 };  // D22 fix: align to Python "w1,w2,w3" register order

// === State container ========================================================
// Caller alloca questa struttura e passa il puntatore a routed_dispatch.
// Buffer interni preallocati per evitare malloc nel hot path.
struct DSv4RoutedState {
    // Device buffers (preallocated)
    float*         d_gate_logits;     // [N_EXPERTS] FP32
    float*         d_scores;          // [N_EXPERTS] FP32
    int*           d_topk_ids;        // [EXPERTS_PER_TOK] INT32
    float*         d_topk_w;          // [EXPERTS_PER_TOK] FP32
    int            h_topk_ids[DSV4_EXPERTS_PER_TOK];  // host mirror

    // Owned scratch (device) per stack temporaneo - contigui [6, *]
    uint8_t*       d_pack_scratch_w1_p;
    uint8_t*       d_pack_scratch_w1_s;
    uint8_t*       d_pack_scratch_w3_p;
    uint8_t*       d_pack_scratch_w3_s;
    uint8_t*       d_pack_scratch_w2_p;
    uint8_t*       d_pack_scratch_w2_s;

    // B3 workspace
    void*          d_grouped_workspace;
    size_t         grouped_workspace_bytes;

    // Hot-set handle
    DSv4Hotset     hotset;

    // BF16 router_gate per layer (pointers)
    const __nv_bfloat16** d_router_gate_per_layer;  // [N_LAYERS] each is [N_EXPERTS, HIDDEN]
    int            n_layers;

    // SANITY counters
    int            routed_dispatch_count_per_token;  // expected 6
    int            last_layer_idx;
    int            abort_flag;  // 0 = ok, !=0 = abort code

    // CUDA stream
    cudaStream_t   stream;
};

// === Sizes (per-expert) =====================================================
// W1: [EXPERT_FF=2048, HIDDEN=4096] mxfp4 -> packed bytes = 2048*4096/2 = 4194304 / expert
//     scale bytes = 2048 * 4096/32 = 262144 / expert
// W3: same as W1
// W2: [HIDDEN=4096, EXPERT_FF=2048] -> packed bytes = 4096*2048/2 = 4194304 / expert
//     scale bytes = 4096 * 2048/32 = 262144 / expert
#define W1_PACK_BYTES_PER_EXPERT  ((size_t)DSV4_EXPERT_FF * DSV4_HIDDEN / 2)
#define W1_SCALE_BYTES_PER_EXPERT ((size_t)DSV4_EXPERT_FF * DSV4_HIDDEN / 32)
#define W3_PACK_BYTES_PER_EXPERT  W1_PACK_BYTES_PER_EXPERT
#define W3_SCALE_BYTES_PER_EXPERT W1_SCALE_BYTES_PER_EXPERT
#define W2_PACK_BYTES_PER_EXPERT  ((size_t)DSV4_HIDDEN * DSV4_EXPERT_FF / 2)
#define W2_SCALE_BYTES_PER_EXPERT ((size_t)DSV4_HIDDEN * DSV4_EXPERT_FF / 32)

// === Kernel: BF16 router_gate GEMV ==========================================
// y[n] = sum_k W[n,k] * x[k]  (BF16 weights, BF16 input, FP32 output)
// W: [N=N_EXPERTS, K=HIDDEN] BF16 row-major, x: [K] BF16, y: [N] FP32
__global__ void router_gate_gemv_kernel(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ y,
    int N, int K
) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;

    const __nv_bfloat16* row = W + (int64_t)n * K;
    float acc = 0.0f;
    for (int k = tid; k < K; k += blk) {
        acc += __bfloat162float(row[k]) * __bfloat162float(x[k]);
    }
    // Block reduction
    __shared__ float sm[32];
    unsigned mask = 0xffffffff;
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(mask, acc, off);
    int lane = tid & 31;
    int wid  = tid >> 5;
    if (lane == 0) sm[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        float v = (tid < (blk + 31) / 32) ? sm[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
        if (tid == 0) y[n] = v;
    }
}

// === Kernel: scores = sqrt(softplus(gate)) ==================================
__global__ void sqrt_softplus_kernel(const float* g, float* s, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = g[i];
    // softplus(x) numerically stable: max(x,0) + log1p(exp(-|x|))
    float ax = fabsf(x);
    float sp = fmaxf(x, 0.0f) + log1pf(expf(-ax));
    s[i] = sqrtf(sp);
}

// === Host: TopK6 + norm + scaling ==========================================
//   Returns 0 on success. Sets ids[6] (sorted by score desc) and w[6] (FP32 normalized * 1.5).
//   ABORT (-1/-2) if any id < 0 || >= N_EXPERTS.
static int topk6_norm_scale_host(
    const float* d_scores, int* h_ids, float* h_w, cudaStream_t stream
) {
    static float h_scores[DSV4_N_EXPERTS];
    cudaError_t err = cudaMemcpyAsync(h_scores, d_scores, sizeof(float) * DSV4_N_EXPERTS,
                                      cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[D5][ABORT] D2H scores copy failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaStreamSynchronize(stream);

    // O(N*K) selection (N=256 K=6 -> 1536 ops, trivial)
    bool taken[DSV4_N_EXPERTS];
    for (int i = 0; i < DSV4_N_EXPERTS; ++i) taken[i] = false;
    float topk_scores[DSV4_EXPERTS_PER_TOK];
    for (int k = 0; k < DSV4_EXPERTS_PER_TOK; ++k) {
        int   best_i = -1;
        float best_v = -INFINITY;
        for (int i = 0; i < DSV4_N_EXPERTS; ++i) {
            if (taken[i]) continue;
            if (h_scores[i] > best_v) { best_v = h_scores[i]; best_i = i; }
        }
        if (best_i < 0 || best_i >= DSV4_N_EXPERTS) {
            std::fprintf(stderr, "[D5][ABORT] topk pick %d returned invalid id=%d\n", k, best_i);
            return -2;
        }
        h_ids[k]      = best_i;
        topk_scores[k] = best_v;
        taken[best_i] = true;
    }
    // Norm + scale
    double sum = 0.0;
    for (int k = 0; k < DSV4_EXPERTS_PER_TOK; ++k) sum += topk_scores[k];
    double denom = sum + (double)DSV4_TOPK_EPS;
    for (int k = 0; k < DSV4_EXPERTS_PER_TOK; ++k) {
        h_w[k] = (float)((topk_scores[k] / denom) * (double)DSV4_ROUTED_SCALING);
    }
    return 0;
}

// D28B1: batched cold-load extern from hotset.cu. We call it once per pack
// to amortise event/wait/host_func across the K topk experts.
extern "C" int hotset_prefetch_batch(int layer, const int* ids, int n_ids);

// === Pack-from-hotset device copies (NO SILENT SKIP) ========================
//   Per ogni expert id in ids[6], prendo i 6 puntatori device dal hot-set
//   e li copio in scratch contiguo [6, *]. Se hot-set ritorna NULL -> ABORT.
//
// D28B1: BEFORE the per-expert callback loop, kick off a single batched
// cold-load for all top-K experts. Inside hotset, this records ONE event on
// prefetch_stream + ONE cudaStreamWaitEvent on consumer + ONE deferred
// madvise host callback. After the batch, every get_w_* callback below is
// a hit (data in flight on prefetch_stream, consumer waits via the batch
// event before its D2Ds on st->stream).
static int pack_routed_topk_strict(
    DSv4RoutedState* st, int layer_idx, const int* ids
) {
    {
        int b_rc = hotset_prefetch_batch(layer_idx, ids, DSV4_EXPERTS_PER_TOK);
        if (b_rc < 0) {
            std::fprintf(stderr,
                "[D5][ABORT] D28B1 hotset_prefetch_batch L=%d rc=%d - NO SILENT SKIP\n",
                layer_idx, b_rc);
            return -12;
        }
    }
    for (int e = 0; e < DSV4_EXPERTS_PER_TOK; ++e) {
        int E = ids[e];
        if (E < 0 || E >= DSV4_N_EXPERTS) {
            std::fprintf(stderr,
                "[D5][ABORT] pack: expert id out of range L=%d slot=%d E=%d (must be [0,%d))\n",
                layer_idx, e, E, DSV4_N_EXPERTS);
            return -10;
        }

        const uint8_t* w1p = st->hotset.get_w_packed(st->hotset.user, layer_idx, E, DSV4_W1);
        const uint8_t* w1s = st->hotset.get_w_scale (st->hotset.user, layer_idx, E, DSV4_W1);
        const uint8_t* w3p = st->hotset.get_w_packed(st->hotset.user, layer_idx, E, DSV4_W3);
        const uint8_t* w3s = st->hotset.get_w_scale (st->hotset.user, layer_idx, E, DSV4_W3);
        const uint8_t* w2p = st->hotset.get_w_packed(st->hotset.user, layer_idx, E, DSV4_W2);
        const uint8_t* w2s = st->hotset.get_w_scale (st->hotset.user, layer_idx, E, DSV4_W2);

        if (!w1p || !w1s || !w3p || !w3s || !w2p || !w2s) {
            // Sprint 6 ROOT CAUSE: prima silently ritornavamo -1 e saltavamo l'expert.
            // Adesso ABORT. D6 implementera l'async upload on demand.
            std::fprintf(stderr,
                "[D5][ABORT] hot-set probe MISS L=%d slot=%d E=%d "
                "(w1p=%p w1s=%p w3p=%p w3s=%p w2p=%p w2s=%p) - NO SILENT SKIP\n",
                layer_idx, e, E,
                (const void*)w1p, (const void*)w1s,
                (const void*)w3p, (const void*)w3s,
                (const void*)w2p, (const void*)w2s);
            return -11;
        }

        // Copia D2D in scratch contiguo (offset = e * size_per_expert)
        cudaMemcpyAsync(
            st->d_pack_scratch_w1_p + (size_t)e * W1_PACK_BYTES_PER_EXPERT,
            w1p, W1_PACK_BYTES_PER_EXPERT, cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(
            st->d_pack_scratch_w1_s + (size_t)e * W1_SCALE_BYTES_PER_EXPERT,
            w1s, W1_SCALE_BYTES_PER_EXPERT, cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(
            st->d_pack_scratch_w3_p + (size_t)e * W3_PACK_BYTES_PER_EXPERT,
            w3p, W3_PACK_BYTES_PER_EXPERT, cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(
            st->d_pack_scratch_w3_s + (size_t)e * W3_SCALE_BYTES_PER_EXPERT,
            w3s, W3_SCALE_BYTES_PER_EXPERT, cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(
            st->d_pack_scratch_w2_p + (size_t)e * W2_PACK_BYTES_PER_EXPERT,
            w2p, W2_PACK_BYTES_PER_EXPERT, cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(
            st->d_pack_scratch_w2_s + (size_t)e * W2_SCALE_BYTES_PER_EXPERT,
            w2s, W2_SCALE_BYTES_PER_EXPERT, cudaMemcpyDeviceToDevice, st->stream);
    }
    return 0;
}

// ===========================================================================
// Public C ABI: dsv4_routed_dispatch
// ===========================================================================
//
// st        : preallocated state (router_gate ptrs, hot-set, scratch, workspace)
// layer_idx : MoE layer index
// x_in      : device [HIDDEN] BF16 (post-norm input)
// x_out     : device [HIDDEN] BF16 (output, OVERWRITTEN with routed contribution)
//
// Returns 0 on success. Negative on ABORT (any error sets st->abort_flag).
// ─── D21: target-filtered dump of pack_scratch (post pack_routed_topk_strict) ───
static uint8_t* g_d21_w1_p = nullptr;
static uint8_t* g_d21_w1_s = nullptr;
static uint8_t* g_d21_w2_p = nullptr;
static uint8_t* g_d21_w2_s = nullptr;
static uint8_t* g_d21_w3_p = nullptr;
static uint8_t* g_d21_w3_s = nullptr;
static int      g_d21_target_layer = -2;

extern "C" int dsv4_routed_d21_set_target(int L) { g_d21_target_layer = L; return 0; }
extern "C" int dsv4_routed_d21_alloc() {
    if (g_d21_w1_p) return 0;
    size_t total_w1p = DSV4_EXPERTS_PER_TOK * W1_PACK_BYTES_PER_EXPERT;
    size_t total_w1s = DSV4_EXPERTS_PER_TOK * W1_SCALE_BYTES_PER_EXPERT;
    size_t total_w2p = DSV4_EXPERTS_PER_TOK * W2_PACK_BYTES_PER_EXPERT;
    size_t total_w2s = DSV4_EXPERTS_PER_TOK * W2_SCALE_BYTES_PER_EXPERT;
    size_t total_w3p = DSV4_EXPERTS_PER_TOK * W3_PACK_BYTES_PER_EXPERT;
    size_t total_w3s = DSV4_EXPERTS_PER_TOK * W3_SCALE_BYTES_PER_EXPERT;
    if (cudaMalloc(&g_d21_w1_p, total_w1p) != cudaSuccess) return -1;
    if (cudaMalloc(&g_d21_w1_s, total_w1s) != cudaSuccess) return -2;
    if (cudaMalloc(&g_d21_w2_p, total_w2p) != cudaSuccess) return -3;
    if (cudaMalloc(&g_d21_w2_s, total_w2s) != cudaSuccess) return -4;
    if (cudaMalloc(&g_d21_w3_p, total_w3p) != cudaSuccess) return -5;
    if (cudaMalloc(&g_d21_w3_s, total_w3s) != cudaSuccess) return -6;
    return 0;
}
extern "C" const void* dsv4_routed_d21_get_w1_p() { return (const void*)g_d21_w1_p; }
extern "C" const void* dsv4_routed_d21_get_w1_s() { return (const void*)g_d21_w1_s; }
extern "C" const void* dsv4_routed_d21_get_w2_p() { return (const void*)g_d21_w2_p; }
extern "C" const void* dsv4_routed_d21_get_w2_s() { return (const void*)g_d21_w2_s; }
extern "C" const void* dsv4_routed_d21_get_w3_p() { return (const void*)g_d21_w3_p; }
extern "C" const void* dsv4_routed_d21_get_w3_s() { return (const void*)g_d21_w3_s; }

// ─── D20f: per-layer-filtered dump of gate_logits/scores/topk_ids/topk_w ───
static float* g_d20f_gate_logits_dump = nullptr;
static float* g_d20f_scores_dump = nullptr;
static int*   g_d20f_topk_ids_dump = nullptr;
static float* g_d20f_topk_w_dump = nullptr;
static int    g_d20f_target_layer = -2;

extern "C" int dsv4_routed_d20f_set_target(int L) { g_d20f_target_layer = L; return 0; }
extern "C" int dsv4_routed_d20f_alloc() {
    if (!g_d20f_gate_logits_dump) {
        if (cudaMalloc(&g_d20f_gate_logits_dump, DSV4_N_EXPERTS * sizeof(float)) != cudaSuccess) return -1;
    }
    if (!g_d20f_scores_dump) {
        if (cudaMalloc(&g_d20f_scores_dump, DSV4_N_EXPERTS * sizeof(float)) != cudaSuccess) return -2;
    }
    if (!g_d20f_topk_ids_dump) {
        if (cudaMalloc(&g_d20f_topk_ids_dump, DSV4_EXPERTS_PER_TOK * sizeof(int)) != cudaSuccess) return -3;
    }
    if (!g_d20f_topk_w_dump) {
        if (cudaMalloc(&g_d20f_topk_w_dump, DSV4_EXPERTS_PER_TOK * sizeof(float)) != cudaSuccess) return -4;
    }
    return 0;
}
extern "C" const void* dsv4_routed_d20f_get_gate_logits() { return (const void*)g_d20f_gate_logits_dump; }
extern "C" const void* dsv4_routed_d20f_get_scores()      { return (const void*)g_d20f_scores_dump; }
extern "C" const void* dsv4_routed_d20f_get_topk_ids()    { return (const void*)g_d20f_topk_ids_dump; }
extern "C" const void* dsv4_routed_d20f_get_topk_w()      { return (const void*)g_d20f_topk_w_dump; }

// ─── D20e: capture x_in passed to router_gate_gemv_kernel ───
static __nv_bfloat16* g_d20e_x_in_dump = nullptr;
static int g_d20e_target_layer = -2;

extern "C" int dsv4_routed_d20e_set_target(int L) { g_d20e_target_layer = L; return 0; }
extern "C" int dsv4_routed_d20e_alloc_buffer() {
    if (g_d20e_x_in_dump) return 0;
    cudaError_t e = cudaMalloc(&g_d20e_x_in_dump, DSV4_HIDDEN * sizeof(__nv_bfloat16));
    return e == cudaSuccess ? 0 : -1;
}
extern "C" const void* dsv4_routed_d20e_get_x_in_dump() {
    return (const void*)g_d20e_x_in_dump;
}

extern "C" int dsv4_routed_dispatch(
    DSv4RoutedState* st, int layer_idx,
    const __nv_bfloat16* x_in, __nv_bfloat16* x_out
) {
    if (!st) {
        std::fprintf(stderr, "[D5][ABORT] state==NULL\n");
        return -100;
    }
    if (layer_idx < 0 || layer_idx >= st->n_layers) {
        std::fprintf(stderr, "[D5][ABORT] layer_idx=%d out of [0,%d)\n",
                     layer_idx, st->n_layers);
        st->abort_flag = -101;
        return -101;
    }
    const __nv_bfloat16* router_gate = st->d_router_gate_per_layer[layer_idx];
    if (!router_gate) {
        std::fprintf(stderr, "[D5][ABORT] router_gate==NULL for layer %d\n", layer_idx);
        st->abort_flag = -102;
        return -102;
    }

    // D20e capture x_in immediately before kernel call
    if (g_d20e_target_layer == layer_idx && g_d20e_x_in_dump) {
        cudaMemcpyAsync(g_d20e_x_in_dump, x_in,
                        DSV4_HIDDEN * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, st->stream);
    }
    // 1. Gating logits  g[256] = router_gate[256, HIDDEN] @ x_in[HIDDEN]
    {
        dim3 grid((unsigned)DSV4_N_EXPERTS);
        dim3 block(256);
        router_gate_gemv_kernel<<<grid, block, 0, st->stream>>>(
            router_gate, x_in, st->d_gate_logits, DSV4_N_EXPERTS, DSV4_HIDDEN);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[D5][ABORT] router gemv launch: %s\n", cudaGetErrorString(err));
            st->abort_flag = -110; return -110;
        }
    }
    // D20f capture gate_logits at target layer
    if (g_d20f_target_layer == layer_idx && g_d20f_gate_logits_dump) {
        cudaMemcpyAsync(g_d20f_gate_logits_dump, st->d_gate_logits,
                        DSV4_N_EXPERTS * sizeof(float),
                        cudaMemcpyDeviceToDevice, st->stream);
    }

    // 2. scores = sqrt(softplus(g))
    {
        const int threads = 128;
        const int blocks = (DSV4_N_EXPERTS + threads - 1) / threads;
        sqrt_softplus_kernel<<<blocks, threads, 0, st->stream>>>(
            st->d_gate_logits, st->d_scores, DSV4_N_EXPERTS);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[D5][ABORT] sqrt_softplus launch: %s\n", cudaGetErrorString(err));
            st->abort_flag = -111; return -111;
        }
    }
    // D20f capture scores at target layer
    if (g_d20f_target_layer == layer_idx && g_d20f_scores_dump) {
        cudaMemcpyAsync(g_d20f_scores_dump, st->d_scores,
                        DSV4_N_EXPERTS * sizeof(float),
                        cudaMemcpyDeviceToDevice, st->stream);
    }

    // 3-5. TopK + norm + scale (host with single D2H, n=256 trivial)
    float h_w[DSV4_EXPERTS_PER_TOK];
    int rc = topk6_norm_scale_host(st->d_scores, st->h_topk_ids, h_w, st->stream);
    if (rc != 0) { st->abort_flag = rc; return rc; }

    cudaMemcpyAsync(st->d_topk_w, h_w, sizeof(float)*DSV4_EXPERTS_PER_TOK,
                    cudaMemcpyHostToDevice, st->stream);
    // D20f capture topk_ids and topk_w at target layer
    if (g_d20f_target_layer == layer_idx) {
        if (g_d20f_topk_ids_dump) {
            cudaMemcpyAsync(g_d20f_topk_ids_dump, st->h_topk_ids,
                            DSV4_EXPERTS_PER_TOK * sizeof(int),
                            cudaMemcpyHostToDevice, st->stream);
        }
        if (g_d20f_topk_w_dump) {
            cudaMemcpyAsync(g_d20f_topk_w_dump, h_w,
                            DSV4_EXPERTS_PER_TOK * sizeof(float),
                            cudaMemcpyHostToDevice, st->stream);
        }
    }

    // 6. STRICT pack from hot-set (no silent skip)
    rc = pack_routed_topk_strict(st, layer_idx, st->h_topk_ids);
    if (rc != 0) { st->abort_flag = rc; return rc; }
    // D21 capture pack_scratch at target layer
    if (g_d21_target_layer == layer_idx && g_d21_w1_p) {
        size_t w1p = DSV4_EXPERTS_PER_TOK * W1_PACK_BYTES_PER_EXPERT;
        size_t w1s = DSV4_EXPERTS_PER_TOK * W1_SCALE_BYTES_PER_EXPERT;
        size_t w2p = DSV4_EXPERTS_PER_TOK * W2_PACK_BYTES_PER_EXPERT;
        size_t w2s = DSV4_EXPERTS_PER_TOK * W2_SCALE_BYTES_PER_EXPERT;
        size_t w3p = DSV4_EXPERTS_PER_TOK * W3_PACK_BYTES_PER_EXPERT;
        size_t w3s = DSV4_EXPERTS_PER_TOK * W3_SCALE_BYTES_PER_EXPERT;
        cudaMemcpyAsync(g_d21_w1_p, st->d_pack_scratch_w1_p, w1p, cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(g_d21_w1_s, st->d_pack_scratch_w1_s, w1s, cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(g_d21_w2_p, st->d_pack_scratch_w2_p, w2p, cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(g_d21_w2_s, st->d_pack_scratch_w2_s, w2s, cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(g_d21_w3_p, st->d_pack_scratch_w3_p, w3p, cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(g_d21_w3_s, st->d_pack_scratch_w3_s, w3s, cudaMemcpyDeviceToDevice, st->stream);
    }

    // 7. Grouped GEMV (B3 kernel) -> x_out
    int b3 = mxfp4_grouped_gemv_topk(
        x_in,
        st->d_pack_scratch_w1_p, st->d_pack_scratch_w3_p, st->d_pack_scratch_w2_p,
        st->d_pack_scratch_w1_s, st->d_pack_scratch_w3_s, st->d_pack_scratch_w2_s,
        st->d_topk_w,
        DSV4_HIDDEN,        // K_in
        DSV4_EXPERT_FF,     // hidden (=2048)
        DSV4_HIDDEN,        // out_dim
        DSV4_EXPERTS_PER_TOK,
        x_out,
        st->d_grouped_workspace,
        st->stream
    );
    if (b3 != 0) {
        std::fprintf(stderr, "[D5][ABORT] B3 grouped gemv returned %d\n", b3);
        st->abort_flag = -120 + b3;
        return -120 + b3;
    }

    // 8. SANITY counter
    st->routed_dispatch_count_per_token = DSV4_EXPERTS_PER_TOK;
    st->last_layer_idx = layer_idx;
    st->abort_flag = 0;
    return 0;
}

// === State alloc/free helpers (Python-side ctypes friendly) =================
extern "C" DSv4RoutedState* dsv4_routed_state_alloc(int n_layers, cudaStream_t stream) {
    DSv4RoutedState* st = (DSv4RoutedState*)std::calloc(1, sizeof(DSv4RoutedState));
    if (!st) return nullptr;
    st->n_layers = n_layers;
    st->stream = stream;
    cudaMalloc(&st->d_gate_logits, sizeof(float) * DSV4_N_EXPERTS);
    cudaMalloc(&st->d_scores,      sizeof(float) * DSV4_N_EXPERTS);
    cudaMalloc(&st->d_topk_ids,    sizeof(int)   * DSV4_EXPERTS_PER_TOK);
    cudaMalloc(&st->d_topk_w,      sizeof(float) * DSV4_EXPERTS_PER_TOK);

    // Scratch packed buffers [6, *] contigui
    cudaMalloc(&st->d_pack_scratch_w1_p, DSV4_EXPERTS_PER_TOK * W1_PACK_BYTES_PER_EXPERT);
    cudaMalloc(&st->d_pack_scratch_w1_s, DSV4_EXPERTS_PER_TOK * W1_SCALE_BYTES_PER_EXPERT);
    cudaMalloc(&st->d_pack_scratch_w3_p, DSV4_EXPERTS_PER_TOK * W3_PACK_BYTES_PER_EXPERT);
    cudaMalloc(&st->d_pack_scratch_w3_s, DSV4_EXPERTS_PER_TOK * W3_SCALE_BYTES_PER_EXPERT);
    cudaMalloc(&st->d_pack_scratch_w2_p, DSV4_EXPERTS_PER_TOK * W2_PACK_BYTES_PER_EXPERT);
    cudaMalloc(&st->d_pack_scratch_w2_s, DSV4_EXPERTS_PER_TOK * W2_SCALE_BYTES_PER_EXPERT);

    // B3 workspace
    st->grouped_workspace_bytes = mxfp4_grouped_workspace_bytes(
        DSV4_EXPERTS_PER_TOK, DSV4_EXPERT_FF, DSV4_HIDDEN);
    cudaMalloc(&st->d_grouped_workspace, st->grouped_workspace_bytes);

    // Router gate per layer (caller fills via dsv4_routed_set_router_gate)
    st->d_router_gate_per_layer = (const __nv_bfloat16**)std::calloc(n_layers, sizeof(void*));

    st->routed_dispatch_count_per_token = 0;
    st->abort_flag = 0;
    return st;
}

extern "C" void dsv4_routed_state_free(DSv4RoutedState* st) {
    if (!st) return;
    cudaFree(st->d_gate_logits);
    cudaFree(st->d_scores);
    cudaFree(st->d_topk_ids);
    cudaFree(st->d_topk_w);
    cudaFree(st->d_pack_scratch_w1_p);
    cudaFree(st->d_pack_scratch_w1_s);
    cudaFree(st->d_pack_scratch_w3_p);
    cudaFree(st->d_pack_scratch_w3_s);
    cudaFree(st->d_pack_scratch_w2_p);
    cudaFree(st->d_pack_scratch_w2_s);
    cudaFree(st->d_grouped_workspace);
    std::free((void*)st->d_router_gate_per_layer);
    std::free(st);
}

extern "C" void dsv4_routed_set_router_gate(
    DSv4RoutedState* st, int layer_idx, const __nv_bfloat16* d_router_gate
) {
    if (!st || layer_idx < 0 || layer_idx >= st->n_layers) return;
    st->d_router_gate_per_layer[layer_idx] = d_router_gate;
}

extern "C" void dsv4_routed_set_hotset(
    DSv4RoutedState* st,
    const uint8_t* (*get_w_packed)(void*, int, int, int),
    const uint8_t* (*get_w_scale )(void*, int, int, int),
    void* user
) {
    if (!st) return;
    st->hotset.get_w_packed = get_w_packed;
    st->hotset.get_w_scale  = get_w_scale;
    st->hotset.user         = user;
}

extern "C" int dsv4_routed_get_dispatch_count(const DSv4RoutedState* st) {
    return st ? st->routed_dispatch_count_per_token : -1;
}

extern "C" int dsv4_routed_get_abort_flag(const DSv4RoutedState* st) {
    return st ? st->abort_flag : -999;
}

extern "C" int dsv4_routed_get_topk_id(const DSv4RoutedState* st, int slot) {
    if (!st || slot < 0 || slot >= DSV4_EXPERTS_PER_TOK) return -1;
    return st->h_topk_ids[slot];
}


// ─── D19 sub-bisect: getters for router_logits, scores, topk_w ───
extern "C" const void* dsv4_routed_get_gate_logits_ptr(const DSv4RoutedState* st) {
    return st ? (const void*)st->d_gate_logits : nullptr;
}
extern "C" const void* dsv4_routed_get_scores_ptr(const DSv4RoutedState* st) {
    return st ? (const void*)st->d_scores : nullptr;
}
extern "C" const void* dsv4_routed_get_topk_w_ptr(const DSv4RoutedState* st) {
    return st ? (const void*)st->d_topk_w : nullptr;
}


// ─── D20b: router_gate pointer getter per layer ───
extern "C" const void* dsv4_routed_get_router_gate_for_L(const DSv4RoutedState* st, int L) {
    if (!st || !st->d_router_gate_per_layer) return nullptr;
    if (L < 0) return nullptr;
    return (const void*)st->d_router_gate_per_layer[L];
}

// ─── D20c: standalone router GEMV invocation ───
// Re-uses the same kernel that compute_router_logits launches:
//   router_gate_gemv_kernel<<<grid, block, smem, stream>>>(W, x, y_fp32, N_EXP, HIDDEN);
// We need to call that kernel directly. Copy the existing kernel signature here:
extern __global__ void router_gate_gemv_kernel(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ y,
    int N, int K);

extern "C" int dsv4_routed_compute_router_logits_isolated(
    const void* router_gate,
    const void* x,
    void* y_out_fp32,
    cudaStream_t stream)
{
    if (!router_gate || !x || !y_out_fp32) return -1;
    const int N = DSV4_N_EXPERTS;
    const int K = DSV4_HIDDEN;
    dim3 grid((unsigned)N);
    dim3 block(256);
    size_t smem = 256 * sizeof(float);
    router_gate_gemv_kernel<<<grid, block, smem, stream>>>(
        (const __nv_bfloat16*)router_gate,
        (const __nv_bfloat16*)x,
        (float*)y_out_fp32, N, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[d20c] router GEMV launch failed: %s\n", cudaGetErrorString(e));
        return -2;
    }
    return 0;
}
