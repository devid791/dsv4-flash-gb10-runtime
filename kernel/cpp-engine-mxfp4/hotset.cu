// hotset.cu — Pure-C++ hot-set 64 expert/layer + LRU policy for DSv4-Flash MXFP4.
// Project: MXFP4 sprint /
// Authors: Davide Zenati
// License: MIT
//
// Mandate: port hot-set 64 expert per layer + LRU policy to C++ pure (no Python
// in the hot path). Reference: C3 expert_prewarm.py (top-N permanent prewarm) +
// R2 _ensure_expert_on_gpu (LRU upload-on-demand from CPU mmap) +
// Sprint 4 hybrid LRU pattern.
//
// Topology (DSv4-Flash):
//   * 43 MoE layers
//   * 256 routed experts per layer
//   * top-K = 6 per token
//   * each expert weight = 3 sub-tensors (w1, w2, w3) MXFP4 packed + E8M0 scale
//   * per-expert size ≈ 12.75 MB packed + ~0.4 MB scale ≈ 13 MB (×3 ≈ 39 MB)
//   * NB: per-tensor MXFP4 sizing already accounted in C3 prewarm budget (12.75 MB)
//
// Memory layout (this module):
//   * "hot slot" = (layer, expert_id, w_key) → (packed_dev, scale_dev)
//   * per layer: an array `slots[64]` of slot descriptors with last_used_ts
//   * permanent prewarm: top-64 selected from C3 routing analysis (per-layer)
//   * cold storage: CPU mmap regions kept open by an external loader. The
//     C++ module never opens safetensors directly — instead it asks the host
//     (Python) to provide cudaMallocManaged-friendly host pointers that we
//     can cudaMemcpyAsync into the GPU slot.
//
// API (extern "C"):
//   hotset_init                  — allocate slots, prewarm top-64 (caller fills)
//   hotset_register_cold         — register a cold (CPU mmap) (packed,scale) pair
//   hotset_get_expert_gpu_ptr    — lookup; LRU evict on miss; copy-in synchronous
//   hotset_prefetch_expert       — async upload (overlap with compute)
//   hotset_stats                 — hits/misses/last miss latency
//   hotset_destroy               — free GPU + reset
//
// Hit/miss policy:
//   * hot path: O(1) hash on (layer, eid, wkey) (3 weights collapsed into one
//     "expert slot" containing (w1,w2,w3) triple — see SlotEntry below).
//   * miss: pick LRU slot for this layer, cudaMemcpyAsync packed+scale for all
//     three weights, return new device pointers. Counts as miss=true.
//   * synchronous fallback if async fails (PARTIAL mode).
//
// Build:
//   nvcc -arch=sm_121a -O3 --use_fast_math -std=c++17 -Xcompiler=-fPIC \
//        -shared hotset.cu -o libhotset.so
//
// Smoke: see tools/D26C_diag.py and tools/D17B4_forced_bisect.py.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

// D26C-B: madvise(MADV_DONTNEED) on cold-load source pages
#include <sys/mman.h>
#include <unistd.h>

// Page-aligned MADV_DONTNEED helper. We trim to interior pages only:
//   start rounds UP to next page, end rounds DOWN to previous page boundary.
// Skips leading/trailing partial pages so we never advise away pages shared
// with neighbor tensors. Loss is at most 2 pages per tensor (~8KB).
static inline int d26c_madvise_dontneed_interior(void* p, size_t bytes) {
    if (!p || bytes == 0) return 0;
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) ps = 4096;
    uintptr_t addr   = (uintptr_t)p;
    uintptr_t end    = addr + bytes;
    uintptr_t a_up   = (addr + ps - 1) & ~((uintptr_t)ps - 1);
    uintptr_t e_down = end & ~((uintptr_t)ps - 1);
    if (a_up >= e_down) return 0;  // tensor too small: skip
    return madvise((void*)a_up, (size_t)(e_down - a_up), MADV_DONTNEED);
}


#define HOTSET_OK 0
#define HOTSET_ERR_BAD_ARG -1
#define HOTSET_ERR_CUDA -2
#define HOTSET_ERR_FULL -3
#define HOTSET_ERR_MISS -4

// -------------------------------------------------------------------------- //
// Topology defaults (DSv4-Flash). Override via hotset_init params if needed. //
// -------------------------------------------------------------------------- //
#define HOTSET_N_LAYERS_DEFAULT 43
#define HOTSET_N_EXPERTS_DEFAULT 256
#define HOTSET_HOT_PER_LAYER_DEFAULT 64
#define HOTSET_W_KEYS 3   // w1, w2, w3

// -------------------------------------------------------------------------- //
// Per-weight tensor size (max). Caller passes exact bytes at register-time.  //
// We allocate slots large enough to hold the biggest tensor we register.     //
// -------------------------------------------------------------------------- //

struct WeightTensor {
    void*    packed_dev;     // device packed nibbles
    void*    scale_dev;      // device E8M0 scales
    size_t   packed_bytes;
    size_t   scale_bytes;
};

struct SlotEntry {
    int             expert_id;       // -1 = empty
    uint64_t        last_used_ts;    // monotonic counter
    WeightTensor    w[HOTSET_W_KEYS]; // w1, w2, w3
    // D28A: event signalling H2D copies for this slot have completed on the
    // prefetch stream. Consumer streams cudaStreamWaitEvent on this before
    // launching kernels that read from slot.w[*]. NULL until first use.
    cudaEvent_t     copy_event;
};

// Cold storage descriptor (CPU side). Caller registers all 256 expert per layer
// via hotset_register_cold. The module keeps the CPU pointers and copies into
// GPU slot on miss.
struct ColdEntry {
    void*    packed_host;     // pinned/mmap host ptr
    void*    scale_host;
    size_t   packed_bytes;
    size_t   scale_bytes;
    bool     present;
};

struct HotsetState {
    int n_layers;
    int n_experts;
    int hot_per_layer;

    // slots[L][slot_idx] — fixed allocation per layer
    std::vector<std::vector<SlotEntry>> slots;

    // index[L] : expert_id → slot_idx (or -1)
    std::vector<std::vector<int>> index;

    // cold[L][E][wkey] — host-side weight registry
    std::vector<std::vector<std::vector<ColdEntry>>> cold;

    // streams for async prefetch
    cudaStream_t prefetch_stream;

    // monotonic ts
    std::atomic<uint64_t> ts;

    // stats
    std::atomic<uint64_t> n_lookups;
    std::atomic<uint64_t> n_hits;
    std::atomic<uint64_t> n_misses;
    std::atomic<uint64_t> last_miss_latency_us;
    std::atomic<uint64_t> total_bytes_resident;

    // D27B: per-layer hit/miss counters
    std::vector<std::atomic<uint64_t>> layer_hits;
    std::vector<std::atomic<uint64_t>> layer_misses;

    // global lock for slot mutations (LRU eviction). Reads on hit are lockless
    // for the lookup but ts update is atomic.
    std::mutex mu;
};

// Singleton (one engine per process — DSv4-Flash decode loop is single-batch).
static HotsetState* g_hs = nullptr;

// D28A: consumer stream registered by wire/engine layer. When set, miss-path
// copy_into_slot makes this stream wait on the per-slot copy_event so the
// consumer never reads the slot before H2D completes. Default NULL = old
// behaviour (synchronous miss). Set once at wire_init.
static cudaStream_t g_consumer_stream = nullptr;

extern "C" int hotset_set_consumer_stream(void* s) {
    g_consumer_stream = (cudaStream_t)s;
    fprintf(stderr, "[hotset][D28A] consumer stream registered = %p\n", s);
    return HOTSET_OK;
}

// D28A: deferred madvise context. Allocated per cold-load, freed inside the
// host callback. Runs on a CUDA worker thread AFTER all H2D copies for this
// slot are observed complete on prefetch_stream. Safe because file-backed
// mmap pages aren't unmapped until DMA done.
struct D28AMadviseCtx {
    void*  packed_host[HOTSET_W_KEYS];
    void*  scale_host [HOTSET_W_KEYS];
    size_t packed_bytes[HOTSET_W_KEYS];
    size_t scale_bytes [HOTSET_W_KEYS];
};

static void CUDART_CB d28a_madvise_host_cb(void* user) {
    D28AMadviseCtx* c = (D28AMadviseCtx*)user;
    if (!c) return;
    for (int w = 0; w < HOTSET_W_KEYS; ++w) {
        if (c->packed_host[w] && c->packed_bytes[w] > 0) {
            d26c_madvise_dontneed_interior(c->packed_host[w], c->packed_bytes[w]);
        }
        if (c->scale_host[w] && c->scale_bytes[w] > 0) {
            d26c_madvise_dontneed_interior(c->scale_host[w],  c->scale_bytes[w]);
        }
    }
    delete c;
}

#define CUDA_CHECK(e) do { cudaError_t _err = (e); if (_err != cudaSuccess) { \
    fprintf(stderr, "[hotset] CUDA error %s at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
    return HOTSET_ERR_CUDA; } } while(0)

// -------------------------------------------------------------------------- //
//  init / destroy                                                            //
// -------------------------------------------------------------------------- //
extern "C" int hotset_init(int n_layers, int n_experts, int hot_per_layer) {
    if (g_hs != nullptr) return HOTSET_OK;  // idempotent
    if (n_layers <= 0)    n_layers   = HOTSET_N_LAYERS_DEFAULT;
    if (n_experts <= 0)   n_experts  = HOTSET_N_EXPERTS_DEFAULT;
    if (hot_per_layer <= 0 || hot_per_layer > n_experts)
        hot_per_layer = HOTSET_HOT_PER_LAYER_DEFAULT;

    g_hs = new HotsetState();
    g_hs->n_layers = n_layers;
    g_hs->n_experts = n_experts;
    g_hs->hot_per_layer = hot_per_layer;
    g_hs->ts.store(1);
    g_hs->n_lookups.store(0);
    g_hs->n_hits.store(0);
    g_hs->n_misses.store(0);
    g_hs->last_miss_latency_us.store(0);
    g_hs->total_bytes_resident.store(0);

    // D27B: per-layer counters
    g_hs->layer_hits   = std::vector<std::atomic<uint64_t>>(n_layers);
    g_hs->layer_misses = std::vector<std::atomic<uint64_t>>(n_layers);
    for (int L = 0; L < n_layers; ++L) {
        g_hs->layer_hits[L].store(0, std::memory_order_relaxed);
        g_hs->layer_misses[L].store(0, std::memory_order_relaxed);
    }

    g_hs->slots.assign(n_layers, std::vector<SlotEntry>(hot_per_layer));
    g_hs->index.assign(n_layers, std::vector<int>(n_experts, -1));
    g_hs->cold.assign(n_layers,
        std::vector<std::vector<ColdEntry>>(n_experts,
            std::vector<ColdEntry>(HOTSET_W_KEYS)));

    for (int L = 0; L < n_layers; ++L) {
        for (int s = 0; s < hot_per_layer; ++s) {
            auto& slot = g_hs->slots[L][s];
            slot.expert_id = -1;
            slot.last_used_ts = 0;
            slot.copy_event = nullptr;  // D28A
            for (int w = 0; w < HOTSET_W_KEYS; ++w) {
                slot.w[w].packed_dev   = nullptr;
                slot.w[w].scale_dev    = nullptr;
                slot.w[w].packed_bytes = 0;
                slot.w[w].scale_bytes  = 0;
            }
        }
    }

    cudaError_t err = cudaStreamCreateWithFlags(&g_hs->prefetch_stream,
                                                cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr, "[hotset] cudaStreamCreate failed: %s\n",
                cudaGetErrorString(err));
        delete g_hs;
        g_hs = nullptr;
        return HOTSET_ERR_CUDA;
    }
    fprintf(stderr, "[hotset] init OK: n_layers=%d n_experts=%d hot/layer=%d\n",
            n_layers, n_experts, hot_per_layer);
    return HOTSET_OK;
}

extern "C" int hotset_destroy() {
    if (g_hs == nullptr) return HOTSET_OK;
    for (int L = 0; L < g_hs->n_layers; ++L) {
        for (int s = 0; s < g_hs->hot_per_layer; ++s) {
            auto& slot = g_hs->slots[L][s];
            // D28A: destroy per-slot event if created
            if (slot.copy_event) {
                cudaEventDestroy(slot.copy_event);
                slot.copy_event = nullptr;
            }
            for (int w = 0; w < HOTSET_W_KEYS; ++w) {
                if (slot.w[w].packed_dev) cudaFree(slot.w[w].packed_dev);
                if (slot.w[w].scale_dev)  cudaFree(slot.w[w].scale_dev);
            }
        }
    }
    cudaStreamDestroy(g_hs->prefetch_stream);
    delete g_hs;
    g_hs = nullptr;
    return HOTSET_OK;
}

// -------------------------------------------------------------------------- //
//  register cold storage (host side)                                         //
// -------------------------------------------------------------------------- //
extern "C" int hotset_register_cold(int layer, int expert_id, int wkey,
                                    void* packed_host, size_t packed_bytes,
                                    void* scale_host,  size_t scale_bytes) {
    if (!g_hs) return HOTSET_ERR_BAD_ARG;
    if (layer < 0 || layer >= g_hs->n_layers) return HOTSET_ERR_BAD_ARG;
    if (expert_id < 0 || expert_id >= g_hs->n_experts) return HOTSET_ERR_BAD_ARG;
    if (wkey < 0 || wkey >= HOTSET_W_KEYS) return HOTSET_ERR_BAD_ARG;

    auto& c = g_hs->cold[layer][expert_id][wkey];
    c.packed_host  = packed_host;
    c.packed_bytes = packed_bytes;
    c.scale_host   = scale_host;
    c.scale_bytes  = scale_bytes;
    c.present      = (packed_host != nullptr);
    return HOTSET_OK;
}

// -------------------------------------------------------------------------- //
//  internal: copy cold expert into a slot                                    //
// -------------------------------------------------------------------------- //
static int copy_into_slot(HotsetState* hs, int layer, int expert_id,
                          int slot_idx, cudaStream_t stream, bool synchronous) {
    SlotEntry& slot = hs->slots[layer][slot_idx];

    // Free old buffers if size mismatch
    int64_t freed = 0;
    int64_t added = 0;
    for (int w = 0; w < HOTSET_W_KEYS; ++w) {
        const ColdEntry& c = hs->cold[layer][expert_id][w];
        if (!c.present) {
            fprintf(stderr,
                "[hotset] miss but cold not registered: L=%d E=%d wkey=%d\n",
                layer, expert_id, w);
            return HOTSET_ERR_MISS;
        }
        WeightTensor& wt = slot.w[w];

        // Realloc packed if size changed
        if (wt.packed_dev == nullptr || wt.packed_bytes != c.packed_bytes) {
            if (wt.packed_dev) {
                freed += wt.packed_bytes;
                cudaFree(wt.packed_dev);
                wt.packed_dev = nullptr;
            }
            cudaError_t err = cudaMalloc(&wt.packed_dev, c.packed_bytes);
            if (err != cudaSuccess) {
                fprintf(stderr, "[hotset] cudaMalloc packed %zu B failed: %s\n",
                        c.packed_bytes, cudaGetErrorString(err));
                return HOTSET_ERR_CUDA;
            }
            wt.packed_bytes = c.packed_bytes;
            added += c.packed_bytes;
        }
        // Realloc scale if size changed
        if (wt.scale_dev == nullptr || wt.scale_bytes != c.scale_bytes) {
            if (wt.scale_dev) {
                freed += wt.scale_bytes;
                cudaFree(wt.scale_dev);
                wt.scale_dev = nullptr;
            }
            cudaError_t err = cudaMalloc(&wt.scale_dev, c.scale_bytes);
            if (err != cudaSuccess) {
                fprintf(stderr, "[hotset] cudaMalloc scale %zu B failed: %s\n",
                        c.scale_bytes, cudaGetErrorString(err));
                return HOTSET_ERR_CUDA;
            }
            wt.scale_bytes = c.scale_bytes;
            added += c.scale_bytes;
        }

        // Async copy host→device
        cudaError_t err = cudaMemcpyAsync(wt.packed_dev, c.packed_host,
                                          c.packed_bytes,
                                          cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            // Fallback synchronous
            fprintf(stderr,
                "[hotset] async packed copy failed (%s); fallback sync\n",
                cudaGetErrorString(err));
            err = cudaMemcpy(wt.packed_dev, c.packed_host, c.packed_bytes,
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) return HOTSET_ERR_CUDA;
        }
        err = cudaMemcpyAsync(wt.scale_dev, c.scale_host, c.scale_bytes,
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            err = cudaMemcpy(wt.scale_dev, c.scale_host, c.scale_bytes,
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) return HOTSET_ERR_CUDA;
        }
    }

    // D28A: async cold-load. Replace the prior cudaStreamSynchronize+madvise
    // block with: (1) event record on the copy stream, (2) cross-stream wait
    // so the consumer stream sees H2D as a barrier before its kernels read
    // from slot.w[*], (3) deferred madvise via cudaLaunchHostFunc — runs on
    // a CUDA worker thread AFTER copies complete, never blocks decode.
    //
    // Correctness: the slot.w[*].packed_dev / scale_dev pointers are valid
    // immediately (allocated above). Data lands on the GPU some time later,
    // but the consumer stream waits on copy_event so any kernel launched on
    // that stream observes the copies as already-complete. Hits do not need
    // to wait — they read slots whose copy_event already signalled in some
    // previous decode_step that also flushed the consumer stream.
    if (slot.copy_event == nullptr) {
        cudaError_t eerr = cudaEventCreateWithFlags(
            &slot.copy_event, cudaEventDisableTiming);
        if (eerr != cudaSuccess) {
            fprintf(stderr,
                "[hotset][D28A] cudaEventCreate failed (%s); fallback sync\n",
                cudaGetErrorString(eerr));
            cudaStreamSynchronize(stream);
            for (int w = 0; w < HOTSET_W_KEYS; ++w) {
                const ColdEntry& c2 = hs->cold[layer][expert_id][w];
                if (c2.present) {
                    d26c_madvise_dontneed_interior(c2.packed_host, c2.packed_bytes);
                    d26c_madvise_dontneed_interior(c2.scale_host,  c2.scale_bytes);
                }
            }
            slot.copy_event = nullptr;  // remain in legacy mode for this slot
            // fall through to bookkeeping below
            slot.expert_id = expert_id;
            slot.last_used_ts = hs->ts.fetch_add(1, std::memory_order_relaxed);
            hs->index[layer][expert_id] = slot_idx;
            if (synchronous) cudaStreamSynchronize(stream);
            int64_t delta_legacy = added - freed;
            if (delta_legacy > 0)
                hs->total_bytes_resident.fetch_add((uint64_t)delta_legacy);
            else if (delta_legacy < 0)
                hs->total_bytes_resident.fetch_sub((uint64_t)(-delta_legacy));
            return HOTSET_OK;
        }
    }
    cudaEventRecord(slot.copy_event, stream);

    // Cross-stream barrier: consumer stream will not start any subsequent
    // kernel until slot.copy_event has signalled.
    if (g_consumer_stream != nullptr && g_consumer_stream != stream) {
        cudaStreamWaitEvent(g_consumer_stream, slot.copy_event, 0);
    }

    // Deferred madvise. Allocate context, populate host pointers, hand off
    // to CUDA worker via cudaLaunchHostFunc on the SAME stream the copies
    // were issued on — the callback runs only after the recordEvent above
    // (and therefore after the cudaMemcpyAsyncs).
    D28AMadviseCtx* mctx = new D28AMadviseCtx();
    for (int w = 0; w < HOTSET_W_KEYS; ++w) {
        const ColdEntry& c2 = hs->cold[layer][expert_id][w];
        if (c2.present) {
            mctx->packed_host[w]  = c2.packed_host;
            mctx->scale_host [w]  = c2.scale_host;
            mctx->packed_bytes[w] = c2.packed_bytes;
            mctx->scale_bytes [w] = c2.scale_bytes;
        } else {
            mctx->packed_host[w]  = nullptr;
            mctx->scale_host [w]  = nullptr;
            mctx->packed_bytes[w] = 0;
            mctx->scale_bytes [w] = 0;
        }
    }
    cudaError_t hf_err = cudaLaunchHostFunc(stream, d28a_madvise_host_cb, mctx);
    if (hf_err != cudaSuccess) {
        fprintf(stderr,
            "[hotset][D28A] cudaLaunchHostFunc failed (%s); inline madvise after sync\n",
            cudaGetErrorString(hf_err));
        cudaStreamSynchronize(stream);
        for (int w = 0; w < HOTSET_W_KEYS; ++w) {
            const ColdEntry& c2 = hs->cold[layer][expert_id][w];
            if (c2.present) {
                d26c_madvise_dontneed_interior(c2.packed_host, c2.packed_bytes);
                d26c_madvise_dontneed_interior(c2.scale_host,  c2.scale_bytes);
            }
        }
        delete mctx;
    }

    // Update bookkeeping (slot becomes visible AFTER the event has been
    // recorded — any consumer reading slot.w[*] from g_consumer_stream is
    // already wait-event ordered).
    slot.expert_id = expert_id;
    slot.last_used_ts = hs->ts.fetch_add(1, std::memory_order_relaxed);
    hs->index[layer][expert_id] = slot_idx;

    // Legacy synchronous fallback: prewarm path passes synchronous=true and
    // expects the slot to be fully populated on return. We honour that.
    if (synchronous) {
        cudaStreamSynchronize(stream);
    }

    int64_t delta = added - freed;
    if (delta > 0) {
        hs->total_bytes_resident.fetch_add((uint64_t)delta);
    } else if (delta < 0) {
        hs->total_bytes_resident.fetch_sub((uint64_t)(-delta));
    }
    return HOTSET_OK;
}

// -------------------------------------------------------------------------- //
//  prewarm a single (layer, expert)                                          //
// -------------------------------------------------------------------------- //
extern "C" int hotset_prewarm_expert(int layer, int expert_id) {
    if (!g_hs) return HOTSET_ERR_BAD_ARG;
    if (layer < 0 || layer >= g_hs->n_layers) return HOTSET_ERR_BAD_ARG;
    if (expert_id < 0 || expert_id >= g_hs->n_experts) return HOTSET_ERR_BAD_ARG;

    std::lock_guard<std::mutex> lk(g_hs->mu);
    if (g_hs->index[layer][expert_id] != -1) return HOTSET_OK;  // already hot

    // Find first empty slot
    int target_slot = -1;
    for (int s = 0; s < g_hs->hot_per_layer; ++s) {
        if (g_hs->slots[layer][s].expert_id == -1) {
            target_slot = s;
            break;
        }
    }
    if (target_slot == -1) return HOTSET_ERR_FULL;

    return copy_into_slot(g_hs, layer, expert_id, target_slot, 0, /*sync=*/true);
}

// -------------------------------------------------------------------------- //
//  internal: pick LRU slot in a layer                                        //
// -------------------------------------------------------------------------- //
static int find_lru_slot(HotsetState* hs, int layer) {
    int lru_idx = 0;
    uint64_t lru_ts = UINT64_MAX;
    for (int s = 0; s < hs->hot_per_layer; ++s) {
        const auto& slot = hs->slots[layer][s];
        if (slot.expert_id == -1) return s;  // empty wins
        if (slot.last_used_ts < lru_ts) {
            lru_ts = slot.last_used_ts;
            lru_idx = s;
        }
    }
    return lru_idx;
}

// -------------------------------------------------------------------------- //
//  hot lookup (with miss handling)                                           //
// -------------------------------------------------------------------------- //
struct ExpertGpuPtrs {
    void*    packed[HOTSET_W_KEYS];
    void*    scale[HOTSET_W_KEYS];
    size_t   packed_bytes[HOTSET_W_KEYS];
    size_t   scale_bytes[HOTSET_W_KEYS];
    int      hit;            // 1 = hot-set hit, 0 = miss (just uploaded)
};

extern "C" int hotset_get_expert_gpu_ptr(int layer, int expert_id,
                                         ExpertGpuPtrs* out) {
    if (!g_hs || !out) return HOTSET_ERR_BAD_ARG;
    if (layer < 0 || layer >= g_hs->n_layers) return HOTSET_ERR_BAD_ARG;
    if (expert_id < 0 || expert_id >= g_hs->n_experts) return HOTSET_ERR_BAD_ARG;

    g_hs->n_lookups.fetch_add(1, std::memory_order_relaxed);

    int slot_idx = g_hs->index[layer][expert_id];
    if (slot_idx >= 0) {
        // Hit — fast path. Update ts and fill out.
        SlotEntry& slot = g_hs->slots[layer][slot_idx];
        slot.last_used_ts = g_hs->ts.fetch_add(1, std::memory_order_relaxed);
        for (int w = 0; w < HOTSET_W_KEYS; ++w) {
            out->packed[w]       = slot.w[w].packed_dev;
            out->scale[w]        = slot.w[w].scale_dev;
            out->packed_bytes[w] = slot.w[w].packed_bytes;
            out->scale_bytes[w]  = slot.w[w].scale_bytes;
        }
        out->hit = 1;
        g_hs->n_hits.fetch_add(1, std::memory_order_relaxed);
        g_hs->layer_hits[layer].fetch_add(1, std::memory_order_relaxed);
        return HOTSET_OK;
    }

    // Miss — LRU evict + sync upload.
    auto t0 = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lk(g_hs->mu);

    // Re-check after lock
    slot_idx = g_hs->index[layer][expert_id];
    if (slot_idx < 0) {
        slot_idx = find_lru_slot(g_hs, layer);
        SlotEntry& victim = g_hs->slots[layer][slot_idx];
        if (victim.expert_id >= 0) {
            // Evict
            g_hs->index[layer][victim.expert_id] = -1;
        }
        // D28A: was sync=true (D26C). Now sync=false — consumer stream waits on
        // slot.copy_event; madvise runs on CUDA host callback after copies.
        // Caller returns immediately with valid GPU ptrs; data lands later.
        int rc = copy_into_slot(g_hs, layer, expert_id, slot_idx,
                                g_hs->prefetch_stream, /*sync=*/false);
        if (rc != HOTSET_OK) return rc;
    }

    SlotEntry& slot = g_hs->slots[layer][slot_idx];
    for (int w = 0; w < HOTSET_W_KEYS; ++w) {
        out->packed[w]       = slot.w[w].packed_dev;
        out->scale[w]        = slot.w[w].scale_dev;
        out->packed_bytes[w] = slot.w[w].packed_bytes;
        out->scale_bytes[w]  = slot.w[w].scale_bytes;
    }
    out->hit = 0;
    g_hs->n_misses.fetch_add(1, std::memory_order_relaxed);
    g_hs->layer_misses[layer].fetch_add(1, std::memory_order_relaxed);
    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t lat_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         t1 - t0).count();
    g_hs->last_miss_latency_us.store(lat_us, std::memory_order_relaxed);
    return HOTSET_OK;
}

// -------------------------------------------------------------------------- //
//  D28B1: batched cold-load for top-K experts of one MoE layer               //
//                                                                            //
//  Called by pack_routed_topk_strict BEFORE its D2D-into-scratch loop. Loads //
//  all currently-missing experts from cold mmap into hot-set slots, with a   //
//  single cross-stream barrier and a single deferred madvise host callback   //
//  for the whole batch. Cuts the per-pack event/wait/launchHostFunc fan-out  //
//  from O(K) (D28A) down to O(1).                                            //
// -------------------------------------------------------------------------- //

// Madvise context for a whole batch — up to (n_ids * HOTSET_W_KEYS) tensors.
// 6 ids × 3 wkeys = 18 entries; we size 32 to be safe vs future K.
#define D28B1_MAX_BATCH_TENSORS 32
struct D28B1BatchMadvCtx {
    void*  packed_host[D28B1_MAX_BATCH_TENSORS];
    void*  scale_host [D28B1_MAX_BATCH_TENSORS];
    size_t packed_bytes[D28B1_MAX_BATCH_TENSORS];
    size_t scale_bytes [D28B1_MAX_BATCH_TENSORS];
    int    count;
};

static void CUDART_CB d28b1_batch_madvise_cb(void* user) {
    D28B1BatchMadvCtx* c = (D28B1BatchMadvCtx*)user;
    if (!c) return;
    for (int i = 0; i < c->count; ++i) {
        if (c->packed_host[i] && c->packed_bytes[i] > 0) {
            d26c_madvise_dontneed_interior(c->packed_host[i], c->packed_bytes[i]);
        }
        if (c->scale_host[i] && c->scale_bytes[i] > 0) {
            d26c_madvise_dontneed_interior(c->scale_host[i],  c->scale_bytes[i]);
        }
    }
    delete c;
}

extern "C" int hotset_prefetch_batch(int layer, const int* ids, int n_ids) {
    if (!g_hs) return HOTSET_ERR_BAD_ARG;
    if (layer < 0 || layer >= g_hs->n_layers) return HOTSET_ERR_BAD_ARG;
    if (!ids || n_ids <= 0) return HOTSET_OK;
    if (n_ids * HOTSET_W_KEYS > D28B1_MAX_BATCH_TENSORS) {
        // safety: don't overflow ctx; fall back to per-id D28A path.
        for (int i = 0; i < n_ids; ++i) {
            if (ids[i] < 0 || ids[i] >= g_hs->n_experts) continue;
            ExpertGpuPtrs tmp = {0};
            int rc = hotset_get_expert_gpu_ptr(layer, ids[i], &tmp);
            if (rc != HOTSET_OK) return rc;
        }
        return HOTSET_OK;
    }

    std::lock_guard<std::mutex> lk(g_hs->mu);

    D28B1BatchMadvCtx* mctx = new D28B1BatchMadvCtx();
    mctx->count = 0;

    int64_t freed_total = 0, added_total = 0;
    bool any_loaded = false;

    for (int i = 0; i < n_ids; ++i) {
        int E = ids[i];
        if (E < 0 || E >= g_hs->n_experts) {
            fprintf(stderr,
                "[hotset][D28B1] batch: expert id out of range L=%d slot=%d E=%d\n",
                layer, i, E);
            delete mctx;
            return HOTSET_ERR_BAD_ARG;
        }

        // Skip if already in hot-set; bump LRU ts so subsequent batch eviction
        // doesn't pick this slot as victim.
        int existing = g_hs->index[layer][E];
        if (existing >= 0) {
            g_hs->slots[layer][existing].last_used_ts =
                g_hs->ts.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        int slot_idx = find_lru_slot(g_hs, layer);
        SlotEntry& victim = g_hs->slots[layer][slot_idx];
        if (victim.expert_id >= 0) {
            g_hs->index[layer][victim.expert_id] = -1;
        }

        SlotEntry& slot = g_hs->slots[layer][slot_idx];
        for (int w = 0; w < HOTSET_W_KEYS; ++w) {
            const ColdEntry& c = g_hs->cold[layer][E][w];
            if (!c.present) {
                fprintf(stderr,
                    "[hotset][D28B1] batch: cold not registered L=%d E=%d wkey=%d\n",
                    layer, E, w);
                delete mctx;
                return HOTSET_ERR_MISS;
            }
            WeightTensor& wt = slot.w[w];
            if (wt.packed_dev == nullptr || wt.packed_bytes != c.packed_bytes) {
                if (wt.packed_dev) {
                    freed_total += wt.packed_bytes;
                    cudaFree(wt.packed_dev);
                    wt.packed_dev = nullptr;
                }
                cudaError_t err = cudaMalloc(&wt.packed_dev, c.packed_bytes);
                if (err != cudaSuccess) {
                    fprintf(stderr,
                        "[hotset][D28B1] cudaMalloc packed %zu B failed: %s\n",
                        c.packed_bytes, cudaGetErrorString(err));
                    delete mctx;
                    return HOTSET_ERR_CUDA;
                }
                wt.packed_bytes = c.packed_bytes;
                added_total += c.packed_bytes;
            }
            if (wt.scale_dev == nullptr || wt.scale_bytes != c.scale_bytes) {
                if (wt.scale_dev) {
                    freed_total += wt.scale_bytes;
                    cudaFree(wt.scale_dev);
                    wt.scale_dev = nullptr;
                }
                cudaError_t err = cudaMalloc(&wt.scale_dev, c.scale_bytes);
                if (err != cudaSuccess) {
                    fprintf(stderr,
                        "[hotset][D28B1] cudaMalloc scale %zu B failed: %s\n",
                        c.scale_bytes, cudaGetErrorString(err));
                    delete mctx;
                    return HOTSET_ERR_CUDA;
                }
                wt.scale_bytes = c.scale_bytes;
                added_total += c.scale_bytes;
            }
            cudaMemcpyAsync(wt.packed_dev, c.packed_host, c.packed_bytes,
                            cudaMemcpyHostToDevice, g_hs->prefetch_stream);
            cudaMemcpyAsync(wt.scale_dev,  c.scale_host,  c.scale_bytes,
                            cudaMemcpyHostToDevice, g_hs->prefetch_stream);

            if (mctx->count < D28B1_MAX_BATCH_TENSORS) {
                int idx = mctx->count++;
                mctx->packed_host[idx]  = c.packed_host;
                mctx->scale_host [idx]  = c.scale_host;
                mctx->packed_bytes[idx] = c.packed_bytes;
                mctx->scale_bytes [idx] = c.scale_bytes;
            }
        }

        slot.expert_id = E;
        slot.last_used_ts = g_hs->ts.fetch_add(1, std::memory_order_relaxed);
        g_hs->index[layer][E] = slot_idx;
        // copy_event left untouched here — we don't use it on the batch path;
        // a single batch_event below replaces all per-slot events.

        g_hs->n_misses.fetch_add(1, std::memory_order_relaxed);
        g_hs->layer_misses[layer].fetch_add(1, std::memory_order_relaxed);
        any_loaded = true;
    }

    if (any_loaded) {
        // Single cross-stream barrier for the whole batch.
        cudaEvent_t batch_ev;
        cudaError_t eerr = cudaEventCreateWithFlags(&batch_ev,
                                                    cudaEventDisableTiming);
        if (eerr != cudaSuccess) {
            // Fallback: synchronize prefetch_stream and madvise inline.
            fprintf(stderr,
                "[hotset][D28B1] cudaEventCreate failed (%s); fallback sync\n",
                cudaGetErrorString(eerr));
            cudaStreamSynchronize(g_hs->prefetch_stream);
            for (int i = 0; i < mctx->count; ++i) {
                if (mctx->packed_host[i])
                    d26c_madvise_dontneed_interior(mctx->packed_host[i],
                                                   mctx->packed_bytes[i]);
                if (mctx->scale_host[i])
                    d26c_madvise_dontneed_interior(mctx->scale_host[i],
                                                   mctx->scale_bytes[i]);
            }
            delete mctx;
        } else {
            cudaEventRecord(batch_ev, g_hs->prefetch_stream);
            if (g_consumer_stream != nullptr &&
                g_consumer_stream != g_hs->prefetch_stream) {
                cudaStreamWaitEvent(g_consumer_stream, batch_ev, 0);
            }
            // Defer madvise via host callback on the prefetch_stream — runs
            // strictly after the recordEvent above, so all H2D DMAs have
            // observed the source pages before we advise them away.
            cudaError_t hf_err = cudaLaunchHostFunc(g_hs->prefetch_stream,
                                                   d28b1_batch_madvise_cb, mctx);
            if (hf_err != cudaSuccess) {
                fprintf(stderr,
                    "[hotset][D28B1] cudaLaunchHostFunc failed (%s); inline sync\n",
                    cudaGetErrorString(hf_err));
                cudaStreamSynchronize(g_hs->prefetch_stream);
                for (int i = 0; i < mctx->count; ++i) {
                    if (mctx->packed_host[i])
                        d26c_madvise_dontneed_interior(mctx->packed_host[i],
                                                       mctx->packed_bytes[i]);
                    if (mctx->scale_host[i])
                        d26c_madvise_dontneed_interior(mctx->scale_host[i],
                                                       mctx->scale_bytes[i]);
                }
                delete mctx;
            }
            // CUDA refcounts internally; safe to destroy now.
            cudaEventDestroy(batch_ev);
        }

        int64_t delta = added_total - freed_total;
        if (delta > 0) g_hs->total_bytes_resident.fetch_add((uint64_t)delta);
        else if (delta < 0) g_hs->total_bytes_resident.fetch_sub((uint64_t)(-delta));
    } else {
        delete mctx;
    }

    return HOTSET_OK;
}

// -------------------------------------------------------------------------- //
//  async prefetch (for overlap)                                              //
// -------------------------------------------------------------------------- //
extern "C" int hotset_prefetch_expert(int layer, int expert_id) {
    if (!g_hs) return HOTSET_ERR_BAD_ARG;
    if (layer < 0 || layer >= g_hs->n_layers) return HOTSET_ERR_BAD_ARG;
    if (expert_id < 0 || expert_id >= g_hs->n_experts) return HOTSET_ERR_BAD_ARG;

    std::lock_guard<std::mutex> lk(g_hs->mu);
    if (g_hs->index[layer][expert_id] != -1) return HOTSET_OK;  // already hot

    int slot_idx = find_lru_slot(g_hs, layer);
    SlotEntry& victim = g_hs->slots[layer][slot_idx];
    if (victim.expert_id >= 0) {
        g_hs->index[layer][victim.expert_id] = -1;
    }
    return copy_into_slot(g_hs, layer, expert_id, slot_idx,
                          g_hs->prefetch_stream, /*sync=*/false);
}

// -------------------------------------------------------------------------- //
//  stats                                                                     //
// -------------------------------------------------------------------------- //
struct HotsetStats {
    uint64_t n_lookups;
    uint64_t n_hits;
    uint64_t n_misses;
    uint64_t last_miss_latency_us;
    uint64_t total_bytes_resident;
    int      n_layers;
    int      hot_per_layer;
};

extern "C" int hotset_stats(HotsetStats* out) {
    if (!g_hs || !out) return HOTSET_ERR_BAD_ARG;
    out->n_lookups            = g_hs->n_lookups.load();
    out->n_hits               = g_hs->n_hits.load();
    out->n_misses             = g_hs->n_misses.load();
    out->last_miss_latency_us = g_hs->last_miss_latency_us.load();
    out->total_bytes_resident = g_hs->total_bytes_resident.load();
    out->n_layers             = g_hs->n_layers;
    out->hot_per_layer        = g_hs->hot_per_layer;
    return HOTSET_OK;
}

// -------------------------------------------------------------------------- //
//  reset stats (smoke helper)                                                //
// -------------------------------------------------------------------------- //
extern "C" int hotset_reset_stats() {
    if (!g_hs) return HOTSET_ERR_BAD_ARG;
    g_hs->n_lookups.store(0);
    g_hs->n_hits.store(0);
    g_hs->n_misses.store(0);
    g_hs->last_miss_latency_us.store(0);
    // D27B: also reset per-layer counters
    for (int L = 0; L < g_hs->n_layers; ++L) {
        g_hs->layer_hits[L].store(0, std::memory_order_relaxed);
        g_hs->layer_misses[L].store(0, std::memory_order_relaxed);
    }
    return HOTSET_OK;
}

// D27B: per-layer extern getters
extern "C" uint64_t hotset_get_layer_hits(int L) {
    if (!g_hs || L < 0 || L >= g_hs->n_layers) return 0;
    return g_hs->layer_hits[L].load(std::memory_order_relaxed);
}
extern "C" uint64_t hotset_get_layer_misses(int L) {
    if (!g_hs || L < 0 || L >= g_hs->n_layers) return 0;
    return g_hs->layer_misses[L].load(std::memory_order_relaxed);
}
extern "C" int hotset_get_n_layers(void) {
    if (!g_hs) return 0;
    return g_hs->n_layers;
}
