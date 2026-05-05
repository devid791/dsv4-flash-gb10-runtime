// D8 -- CUDA Graph capture for DSv4-Flash MXFP4 C++ decode loop
//
// Captures the per-token forward (43 layer + final + lm_head + sample) into
// a static cudaGraphExec_t and replays it for every subsequent decode token.
// Pre-allocates static KV cache of maximum size (256K context x 16 slot)
// and uses a device-resident kv_position counter updated via cudaMemcpyAsync
// (NO host-side branch, NO Python list mutation, NO mmap touch on hot path).
//
// Design contract (entry point provided by D9 / engine wiring):
//     void cpp_decode_step(EngineState* st, cudaStream_t stream);
// The engine is responsible for:
//   - reading the current token from st->d_token_in (int32, device)
//   - reading kv_position from st->d_kv_pos (int32, device)
//   - writing the next sampled token to st->d_token_out (int32, device)
//   - advancing st->d_kv_pos via a device-side increment kernel
//   - all per-layer attention/MoE compute MUST stay on device with no
//     CPU branch -- if a hot-set miss occurs, the engine signals via
//     st->h_hotset_miss_flag (host pinned u32) AFTER replay completes
//
// Hot-set miss handling:
//   replay_decode_graph() checks st->h_hotset_miss_flag after replay; if
//   set, it returns DECODE_HOTSET_MISS to the caller. The caller (D9)
//   then runs an eager step (which uploads the missing expert and updates
//   the hot-set), invalidates the captured graph, and re-captures it.
//
// Author:
// Branch: agent-d-mxfp4-cpp-d8
// Stack:  GB10 sm_121a, CUDA 13.2, container dsv4-dev
//
// Build:  nvcc -O3 -arch=sm_121a -std=c++17 -Xcompiler -fPIC -shared \
//             graph.cu -o libcppgraph.so
// Test:   nvcc -O3 -arch=sm_121a -std=c++17 -DD8_TEST_MAIN \
//             graph.cu -o D8_graph_test

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

// ─── Public C ABI ───────────────────────────────────────────────────────

extern "C" {

// Status codes returned by replay_decode_graph()
enum DecodeStatus : int32_t {
    DECODE_OK            = 0,
    DECODE_HOTSET_MISS   = 1,   // re-capture required
    DECODE_GRAPH_INVALID = 2,   // capture handle stale or null
    DECODE_LAUNCH_ERR    = 3,   // cudaGraphLaunch returned non-success
    DECODE_SYNC_ERR      = 4,   // post-launch sync failed
};

// Engine state -- opaque to the graph runner; the cpp-engine module fills
// the pointers below at construction time. The graph runner only reads
// h_hotset_miss_flag and the d_token_in/out + d_kv_pos slots.
struct EngineState {
    // Pinned host buffers (zero-copy, mapped to device via UVA)
    uint32_t* h_hotset_miss_flag;   // 4B, set by engine on miss
    int32_t*  h_token_out_mirror;   // 4B, optional readback slot (unused by graph)

    // Device buffers (updated by the captured graph; host writes via async memcpy)
    int32_t*  d_token_in;           // 4B, input token id (host writes)
    int32_t*  d_token_out;          // 4B, output token id (graph writes)
    int32_t*  d_kv_pos;             // 4B, current kv index (graph increments)

    // Static KV cache (pre-allocated to max size, owned by engine)
    void*     d_kv_cache;           // [n_layers, max_slot, max_context, head_dim]
    size_t    kv_bytes;             // total bytes of d_kv_cache
    int32_t   max_context;          // 256000
    int32_t   max_slot;             // 16
    int32_t   n_layers;             // 43
    int32_t   hidden;               // 4096

    // User-provided decode step entry. Captured by cudaStreamBeginCapture.
    // Must be re-entrant on stream and contain ZERO host branches.
    void (*decode_step_fn)(struct EngineState* st, cudaStream_t stream);

    // Internal: graph handle (filled by capture_decode_graph)
    cudaGraph_t      _graph;
    cudaGraphExec_t  _graph_exec;
    int32_t          _captured;     // bool 0/1
    int32_t          _capture_gen;  // bumped on every successful capture
};

// ─── Error helper ───────────────────────────────────────────────────────

#define D8_CK(call) do {                                                  \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr, "[D8] CUDA error at %s:%d: %s (%s)\n",            \
                __FILE__, __LINE__, cudaGetErrorString(_e),               \
                cudaGetErrorName(_e));                                    \
        return -1;                                                        \
    }                                                                     \
} while (0)

#define D8_CK_NORET(call) do {                                            \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr, "[D8] CUDA error at %s:%d: %s (%s)\n",            \
                __FILE__, __LINE__, cudaGetErrorString(_e),               \
                cudaGetErrorName(_e));                                    \
    }                                                                     \
} while (0)

// ─── Capture ────────────────────────────────────────────────────────────

// capture_decode_graph
//   Captures a single decode_step into a cudaGraphExec_t. Caller must have
//   already populated st->decode_step_fn and the buffer pointers, and must
//   have done at least one warmup eager call so all lazy allocations have
//   happened.
//
// Returns: 0 on success, negative on failure.
int capture_decode_graph(EngineState* st)
{
    if (!st || !st->decode_step_fn) {
        fprintf(stderr, "[D8] capture_decode_graph: null state or fn\n");
        return -1;
    }

    // Tear down any previous capture first.
    if (st->_captured) {
        D8_CK_NORET(cudaGraphExecDestroy(st->_graph_exec));
        D8_CK_NORET(cudaGraphDestroy(st->_graph));
        st->_captured = 0;
    }

    // Create dedicated capture stream.
    cudaStream_t capture_stream;
    D8_CK(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking));

    // Warm side stream sync per PyTorch capture pattern.
    D8_CK(cudaStreamSynchronize(0));

    // Begin capture in ThreadLocal mode -- only ops on this stream get
    // recorded, no global sync requirement.
    D8_CK(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal));

    // Invoke the decode step on the capture stream.
    st->decode_step_fn(st, capture_stream);

    cudaGraph_t graph = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(capture_stream, &graph);
    if (end_err != cudaSuccess) {
        fprintf(stderr,
            "[D8] cudaStreamEndCapture FAILED: %s (%s) -- "
            "decode_step_fn must contain ZERO host branches and ZERO "
            "stream-foreign ops\n",
            cudaGetErrorString(end_err), cudaGetErrorName(end_err));
        D8_CK_NORET(cudaStreamDestroy(capture_stream));
        return -2;
    }

    // Instantiate the executable graph.
    cudaGraphExec_t graph_exec = nullptr;
    D8_CK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    st->_graph       = graph;
    st->_graph_exec  = graph_exec;
    st->_captured    = 1;
    st->_capture_gen += 1;

    D8_CK_NORET(cudaStreamDestroy(capture_stream));
    return 0;
}

// ─── Replay ─────────────────────────────────────────────────────────────

// replay_decode_graph
//   Pushes a new (token_id, kv_position) into the device input slots and
//   replays the captured graph. After sync, reads the sampled token from
//   d_token_out and the hot-set miss flag from h_hotset_miss_flag.
//
// Inputs:
//   token_id:    next input token (host int32)
//   kv_position: current KV write index (host int32)
//   stream:      replay stream (caller-owned; pass 0 for default)
//
// Outputs:
//   *out_token:  sampled token id (only valid if return == DECODE_OK)
//
// Returns: DecodeStatus code.
int replay_decode_graph(EngineState* st,
                        int32_t token_id,
                        int32_t kv_position,
                        cudaStream_t stream,
                        int32_t* out_token)
{
    if (!st || !st->_captured || !st->_graph_exec) {
        return DECODE_GRAPH_INVALID;
    }

    // Reset hot-set miss flag (host pinned, mapped).
    if (st->h_hotset_miss_flag) {
        *st->h_hotset_miss_flag = 0u;
    }

    // Push token + kv_pos into device input slots.
    cudaError_t e;
    e = cudaMemcpyAsync(st->d_token_in,  &token_id,     sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return DECODE_LAUNCH_ERR;
    e = cudaMemcpyAsync(st->d_kv_pos,    &kv_position,  sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return DECODE_LAUNCH_ERR;

    // Launch the captured graph.
    e = cudaGraphLaunch(st->_graph_exec, stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[D8] cudaGraphLaunch FAILED: %s\n",
                cudaGetErrorString(e));
        return DECODE_LAUNCH_ERR;
    }

    // Sync and pull output token + miss flag.
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[D8] post-replay sync FAILED: %s\n",
                cudaGetErrorString(e));
        return DECODE_SYNC_ERR;
    }

    int32_t tok_host = -1;
    e = cudaMemcpy(&tok_host, st->d_token_out, sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return DECODE_SYNC_ERR;
    if (out_token) *out_token = tok_host;

    // Check hot-set miss flag (set by engine inside captured kernels via
    // an atomic store to the pinned host buffer; safe under capture).
    if (st->h_hotset_miss_flag && *st->h_hotset_miss_flag) {
        return DECODE_HOTSET_MISS;
    }

    return DECODE_OK;
}

// ─── Teardown ───────────────────────────────────────────────────────────

int destroy_decode_graph(EngineState* st)
{
    if (!st || !st->_captured) return 0;
    D8_CK_NORET(cudaGraphExecDestroy(st->_graph_exec));
    D8_CK_NORET(cudaGraphDestroy(st->_graph));
    st->_captured   = 0;
    st->_graph      = nullptr;
    st->_graph_exec = nullptr;
    return 0;
}

// ─── KV cache lifecycle helpers ─────────────────────────────────────────

// Pre-allocate the static KV cache for the maximum context size that will
// ever be replayed. Allocating once up-front (vs growing per-layer) is the
// key invariant that makes the captured graph valid for every decode step.
//
// Sizing: n_layers * max_slot * max_context * head_dim * sizeof(bf16)
// Default DSv4 config: 43 * 16 * 256000 * 128 * 2 = ~45 GiB (matches
// runtime KV peak from R2 smoke). Caller can pass head_dim_total to
// override (e.g. compressed MLA latent = 512 instead of full 16384).
int alloc_static_kv_cache(EngineState* st, int32_t head_dim_total)
{
    if (!st) return -1;
    if (st->n_layers <= 0 || st->max_context <= 0 || st->max_slot <= 0 ||
        head_dim_total <= 0) {
        fprintf(stderr,
            "[D8] alloc_static_kv_cache: bad dims n_layers=%d max_slot=%d "
            "max_context=%d head_dim=%d\n",
            st->n_layers, st->max_slot, st->max_context, head_dim_total);
        return -1;
    }
    size_t bytes = (size_t)st->n_layers * (size_t)st->max_slot *
                   (size_t)st->max_context * (size_t)head_dim_total *
                   sizeof(uint16_t);  // bf16 = 2 bytes
    D8_CK(cudaMalloc(&st->d_kv_cache, bytes));
    D8_CK(cudaMemset(st->d_kv_cache, 0, bytes));
    st->kv_bytes = bytes;
    fprintf(stderr,
        "[D8] static KV cache allocated: %.2f GiB (n_layers=%d max_slot=%d "
        "max_context=%d head_dim=%d)\n",
        bytes / (1024.0 * 1024.0 * 1024.0),
        st->n_layers, st->max_slot, st->max_context, head_dim_total);
    return 0;
}

int alloc_io_slots(EngineState* st)
{
    if (!st) return -1;
    D8_CK(cudaMalloc(&st->d_token_in,  sizeof(int32_t)));
    D8_CK(cudaMalloc(&st->d_token_out, sizeof(int32_t)));
    D8_CK(cudaMalloc(&st->d_kv_pos,    sizeof(int32_t)));
    D8_CK(cudaMemset(st->d_token_in,  0, sizeof(int32_t)));
    D8_CK(cudaMemset(st->d_token_out, 0, sizeof(int32_t)));
    D8_CK(cudaMemset(st->d_kv_pos,    0, sizeof(int32_t)));

    // Allocate pinned (mapped) host slots
    D8_CK(cudaHostAlloc(&st->h_hotset_miss_flag,  sizeof(uint32_t),
                        cudaHostAllocMapped));
    *st->h_hotset_miss_flag = 0;
    D8_CK(cudaHostAlloc(&st->h_token_out_mirror, sizeof(int32_t),
                        cudaHostAllocMapped));
    *st->h_token_out_mirror = 0;
    return 0;
}

int free_engine_state(EngineState* st)
{
    if (!st) return 0;
    if (st->_captured) destroy_decode_graph(st);
    if (st->d_kv_cache)            D8_CK_NORET(cudaFree(st->d_kv_cache));
    if (st->d_token_in)            D8_CK_NORET(cudaFree(st->d_token_in));
    if (st->d_token_out)           D8_CK_NORET(cudaFree(st->d_token_out));
    if (st->d_kv_pos)              D8_CK_NORET(cudaFree(st->d_kv_pos));
    if (st->h_hotset_miss_flag)    D8_CK_NORET(cudaFreeHost(st->h_hotset_miss_flag));
    if (st->h_token_out_mirror)    D8_CK_NORET(cudaFreeHost(st->h_token_out_mirror));
    memset(st, 0, sizeof(EngineState));
    return 0;
}

}  // extern "C"

// ─── Test main (built only with -DD8_TEST_MAIN) ─────────────────────────
//
// Stub decode kernel that emulates a 43-layer pure-device forward without
// any CPU branch. This is enough to verify the capture/replay invariants:
// if our graph code can capture+replay this stub correctly, then it can
// capture+replay any pure-device decode that D2/D9 produce.

#ifdef D8_TEST_MAIN

// Pure-device "decode" kernel: read token_in, do N rounds of fake compute
// on the KV cache, advance kv_pos, write token_out = (token_in * 7 + 1) % vocab
__global__ void stub_decode_kernel(
    int32_t* d_token_in,
    int32_t* d_token_out,
    int32_t* d_kv_pos,
    uint16_t* d_kv_cache,
    int32_t   layer_stride_u16,
    int32_t   n_layers,
    int32_t   vocab)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;
    int32_t tok = *d_token_in;
    int32_t pos = *d_kv_pos;
    // Fake per-layer compute -- write a deterministic pattern into KV cache
    // at the current slot.  This emulates 43-layer attention writes.
    for (int L = 0; L < n_layers; ++L) {
        size_t off = (size_t)L * (size_t)layer_stride_u16 + (size_t)pos;
        d_kv_cache[off] = (uint16_t)((tok + L) & 0xFFFFu);
    }
    int32_t next = (tok * 7 + 1) % vocab;
    *d_token_out = next;
    *d_kv_pos    = pos + 1;
}

// Decode step entry conforming to the EngineState contract.
extern "C" void stub_decode_step(EngineState* st, cudaStream_t stream)
{
    int32_t layer_stride_u16 = st->max_slot * st->max_context;
    stub_decode_kernel<<<1, 32, 0, stream>>>(
        st->d_token_in, st->d_token_out, st->d_kv_pos,
        (uint16_t*)st->d_kv_cache,
        layer_stride_u16, st->n_layers, /*vocab*/ 129280);
}

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    fprintf(stderr, "[D8] CUDA Graph capture/replay smoke test\n");

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    fprintf(stderr, "[D8] device: %s sm_%d%d\n",
            prop.name, prop.major, prop.minor);

    EngineState st;
    memset(&st, 0, sizeof(st));
    // Realistic-ish DSv4 dims (smaller KV for test to stay <2 GiB)
    st.n_layers       = 43;
    st.max_slot       = 16;
    st.max_context    = 8192;          // smoke uses small context
    st.hidden         = 4096;
    st.decode_step_fn = stub_decode_step;

    if (alloc_io_slots(&st) != 0) { fprintf(stderr, "[D8] io alloc fail\n"); return 1; }
    if (alloc_static_kv_cache(&st, /*head_dim_total*/ 128) != 0) {
        fprintf(stderr, "[D8] kv alloc fail\n"); return 1;
    }

    // Eager warmup -- 1 step to settle any lazy CUDA init (ctx, JIT).
    {
        int32_t tok0 = 1;
        cudaMemcpy(st.d_token_in, &tok0, sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemset(st.d_kv_pos, 0, sizeof(int32_t));
        st.decode_step_fn(&st, 0);
        cudaStreamSynchronize(0);
        int32_t tok_warm = -1;
        cudaMemcpy(&tok_warm, st.d_token_out, sizeof(int32_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[D8] eager warmup token: %d (expect 8 = 1*7+1)\n", tok_warm);
        if (tok_warm != 8) {
            fprintf(stderr, "[D8] WARN warmup mismatch (got %d)\n", tok_warm);
        }
    }

    // ─── Capture ────────────────────────────────────────────────
    fprintf(stderr, "[D8] capturing graph...\n");
    int rc = capture_decode_graph(&st);
    if (rc != 0) {
        fprintf(stderr, "[D8] capture FAIL rc=%d\n", rc);
        free_engine_state(&st);
        return 2;
    }
    fprintf(stderr, "[D8] capture OK (gen=%d)\n", st._capture_gen);

    // ─── Replay 10 times ────────────────────────────────────────
    int32_t tok = 3;
    int32_t pos = 1;
    bool replay_ok = true;
    std::vector<int32_t> graph_tokens;
    double t0 = now_ms();
    for (int i = 0; i < 10; ++i) {
        int32_t out = -1;
        int rstat = replay_decode_graph(&st, tok, pos, 0, &out);
        if (rstat != DECODE_OK) {
            fprintf(stderr, "[D8] replay step %d FAIL status=%d\n", i, rstat);
            replay_ok = false;
            break;
        }
        graph_tokens.push_back(out);
        tok = out;
        pos += 1;
    }
    double t_graph = now_ms() - t0;
    fprintf(stderr, "[D8] replay loop: %d tokens in %.2f ms (%.2f tok/s)\n",
            (int)graph_tokens.size(), t_graph,
            graph_tokens.size() * 1000.0 / t_graph);

    // ─── Reference: same loop without capture (eager) ───────────
    cudaMemset(st.d_kv_pos, 0, sizeof(int32_t));
    int32_t tok_e = 3;
    int32_t pos_e = 1;
    std::vector<int32_t> eager_tokens;
    double t1 = now_ms();
    for (int i = 0; i < 10; ++i) {
        cudaMemcpy(st.d_token_in, &tok_e, sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(st.d_kv_pos,   &pos_e, sizeof(int32_t), cudaMemcpyHostToDevice);
        stub_decode_step(&st, 0);
        cudaStreamSynchronize(0);
        int32_t out_e = -1;
        cudaMemcpy(&out_e, st.d_token_out, sizeof(int32_t), cudaMemcpyDeviceToHost);
        eager_tokens.push_back(out_e);
        tok_e = out_e;
        pos_e += 1;
    }
    double t_eager = now_ms() - t1;
    fprintf(stderr, "[D8] eager loop: %d tokens in %.2f ms (%.2f tok/s)\n",
            (int)eager_tokens.size(), t_eager,
            eager_tokens.size() * 1000.0 / t_eager);

    // ─── Compare token streams ──────────────────────────────────
    bool match = (graph_tokens.size() == eager_tokens.size());
    if (match) {
        for (size_t i = 0; i < graph_tokens.size(); ++i) {
            if (graph_tokens[i] != eager_tokens[i]) {
                fprintf(stderr,
                    "[D8] MISMATCH at i=%zu: graph=%d eager=%d\n",
                    i, graph_tokens[i], eager_tokens[i]);
                match = false;
                break;
            }
        }
    }
    fprintf(stderr, "[D8] graph-vs-eager token match: %s\n",
            match ? "YES" : "NO");

    double speedup = (t_eager > 0.0) ? (t_eager / t_graph) : 0.0;
    fprintf(stderr, "[D8] speedup vs eager: %.2fx\n", speedup);

    // ─── Re-capture after simulated hot-set miss ────────────────
    fprintf(stderr, "[D8] simulating hot-set miss + re-capture...\n");
    int prev_gen = st._capture_gen;
    int rc2 = capture_decode_graph(&st);  // implicit destroy + re-instantiate
    if (rc2 != 0) {
        fprintf(stderr, "[D8] re-capture FAIL rc=%d\n", rc2);
        free_engine_state(&st);
        return 3;
    }
    fprintf(stderr, "[D8] re-capture OK (gen=%d, prev=%d, +1=%s)\n",
            st._capture_gen, prev_gen,
            (st._capture_gen == prev_gen + 1) ? "YES" : "NO");

    // 1 replay after re-capture
    int32_t out_pc = -1;
    int rs = replay_decode_graph(&st, 5, 11, 0, &out_pc);
    fprintf(stderr, "[D8] post-recapture replay: status=%d token=%d "
            "(expect 36 = 5*7+1)\n", rs, out_pc);
    bool postcap_ok = (rs == DECODE_OK && out_pc == 36);

    free_engine_state(&st);

    bool overall_pass = replay_ok && match && postcap_ok;
    fprintf(stderr, "[D8] OVERALL: %s\n", overall_pass ? "PASS" : "FAIL");

    // Emit JSON one-liner for the orchestrator (parsed by D8_graph_test.py)
    printf(
        "{\"capture_ok\":%s,\"replay_ok\":%s,\"match\":%s,"
        "\"recapture_ok\":%s,\"speedup\":%.3f,\"graph_ms\":%.3f,"
        "\"eager_ms\":%.3f,\"n_steps\":10,\"overall\":%s}\n",
        st._captured ? "true" : "true",   // captured before free
        replay_ok ? "true" : "false",
        match ? "true" : "false",
        postcap_ok ? "true" : "false",
        speedup, t_graph, t_eager,
        overall_pass ? "true" : "false");
    return overall_pass ? 0 : 4;
}

#endif  // D8_TEST_MAIN
