# dsv4-flash-gb10-runtime

Experimental proof-of-life custom C++ runtime for **DeepSeek V4 Flash** (43L MoE,
MLA, MXFP4 routed experts) on a single **NVIDIA GB10 / GX10**-class node with
128 GB unified memory.

> **This is not a vLLM replacement and not production-ready.** It exists to
> validate that DeepSeek V4 Flash MXFP4 can be made to run correctly end-to-end
> on a single 128 GB unified-memory node, with no sharding / pipeline / tensor
> parallelism. All 43 layers, 256 routed experts per layer, KV cache, and
> workspace fit in unified memory at the same time. Throughput optimization is
> staged after correctness; current absolute TPS is **not** competitive.

Discussion thread on the NVIDIA Developer Forums:
[DeepSeek V4 Flash MXFP4 proof-of-life on a single GB10/GX10](https://forums.developer.nvidia.com/t/deepseek-v4-flash-mxfp4-proof-of-life-on-a-single-gb10-gx10/369131)

## Hardware target

- NVIDIA GB10 / GX10-class system, single node
- 128 GB unified memory (CPU + GPU)
- CUDA 13.x, sm_121a (Blackwell GB10)
- ~160 GB free disk for the BF16 HuggingFace snapshot

## What's included

- `kernel/cpp-engine-mxfp4/` — main C++ engine (forward pass, HC, MLA wire,
  routed dispatch, hot-set, head)
- `kernel/{fp8-dense,mxfp4-dense,mxfp4-routed,rmsnorm-fuse}/` — supporting
  CUDA kernels (FP8 dense GEMV, MXFP4 dense GEMV, MXFP4 grouped GEMV for top-K
  expert dispatch, fused RMSNorm)
- `runtime/` — Python wrapper that loads the BF16 HF snapshot, MXFP4-quantizes
  routed experts on the fly, and drives the C++ engine via ctypes
- `tools/D17B4_forced_bisect.py` — forced-token R2-vs-C++ correctness gate
- `tools/D26C_diag.py` — same-process multi-prompt smoke + memory diagnostics
- `tools/D27_baseline.py` — performance / timing harness

## What's NOT included

- Model weights — download from
  [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- A serving interface (no OpenAI-compatible HTTP server, no streaming)
- CUDA Graph capture path
- MTP / speculative decoding
- Request batching
- Prebuilt `.so` binaries

## Current correctness status

**Forced-token R2-vs-C++ equivalence** (each step the C++ engine is fed the
*previous* token from the Python reference, not its own output, and the
predicted argmax + full logits are compared):

- 8/8 forced-token argmax match against R2
- logits cosine 0.9994 – 0.9999, max abs error 0.6 – 1.1

**Per-layer R2-vs-C++ cosine bisect** (intermediate tensors compared layer by
layer: HC pre/post, RMS, MLA, MoE, post-FFN):

- No `first_bad_layer` found in 0..42 with all bug fixes applied

**Same-process multi-prompt smoke** (max_new=16, temp=0 greedy):

- 5 / 5 prompts complete without OOM/kill, single Python process
- ≥ 3 / 5 intelligible

Example outputs (greedy decode):

| prompt | output |
|---|---|
| `The capital of France is` | ` Paris. It is one of the most famous cities in the world. It is` |
| `Ciao, come stai?` | ` Spero tutto bene. Oggi voglio parlarti di un argomento` |
| `Hello` | `  <%= @user.name %>,\n\nYou have requested to reset your password.\n\nTo` |

## Quick start

### 1. Install host requirements

- CUDA 13.x toolkit with `nvcc` reachable as `/usr/local/cuda/bin/nvcc`
  (override via `NVCC` make variable)
- Python 3.12 with `torch >= 2.11.0+cu130`, `numpy`, `safetensors`,
  `huggingface_hub`, `tokenizers`

### 2. Download the BF16 snapshot

```sh
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir $HOME/DeepSeek-V4-Flash
```

### 3. Configure environment

```sh
cp env.example .env
# edit .env so DSV4_HOME, DSV4_WEIGHTS point at your paths, then:
source .env
```

### 4. Build all kernels

```sh
make -C kernel/fp8-dense
make -C kernel/mxfp4-dense
make -C kernel/mxfp4-routed
make -C kernel/rmsnorm-fuse
make -C kernel/cpp-engine-mxfp4
```

This produces the `.so` files the Python runtime loads via ctypes.

### 5. Run the validation gates

**Math correctness gate (D17B4 forced-token bisect):**

```sh
python3 tools/D17B4_forced_bisect.py
```

Expected: `=== DONE: status=PASS ===`, 8 / 8 argmax MATCH.

**Lifecycle / quality gate (D26C same-process smoke, max_new=16):**

```sh
python3 tools/D26C_diag.py
```

Expected: 5 / 5 prompts `status=ok`, ≥ 3 / 5 `cat=INTELLIGIBILE`, no OOM/kill,
`MemAvailable` plateau (no Cached collapse, no SwapFree exhaustion).

**Performance baseline (D27, optional):**

```sh
python3 tools/D27_baseline.py --top_n 32 --max_new 16 --tag mybaseline
```

Output JSON in `$DSV4_OUT/D27_mybaseline.json` with prewarm/prefill/decode
timing breakdown.

## Bug fixes during bring-up

A few non-trivial issues had to be resolved to get from token noise to coherent
output. They're documented here because they may bite anyone else porting
DeepSeek V4 Flash to a custom runtime.

### 1. HC residual broadcast

The token embedding initially populated only HC slot 0 of the
`[N_HC=4, HIDDEN]` residual stream. The other 3 slots were garbage, which
propagated into the HyperConnection mixing block and produced orthogonal hidden
states from layer 0 onward. Fixed by broadcasting embed slot 0 across all HC
slots.

### 2. MoE W2/W3 enum ordering

The C++ enum mapping `{W1=0, W3=1, W2=2}` did not match the Python loader's
registration order `(w1, w2, w3)`. The routed expert output was nearly
orthogonal (cos ≈ 0) before the fix. Re-ordering the enum to
`{W1=0, W2=1, W3=2}` was a one-line fix that brought routed expert cosine to
0.999989 and step-0 argmax to the correct token.

### 3. YARN compressed RoPE — piecewise NTK ramp

The C++ initially divided all compressed RoPE dimensions uniformly by the YARN
factor. The DeepSeek reference uses a piecewise NTK ramp:

```
freqs = freqs/factor*(1-smooth) + freqs*smooth
where smooth = 1 - linear_ramp(low, high)
```

with `low=15, high=25` for our config (dim_full=64, theta=160000, factor=16,
original=65536). After porting the exact ramp 1:1 from the Python reference,
L2 `q_post_rope` cosine went from 0.921 to 0.999982 vs the reference, and
downstream layers stopped diverging.

### 4. mmap / hot-set memory pressure

Same-process multi-prompt smoke originally died around prompt 3. Root cause was
routed-expert cold-loads touching mmap-backed safetensors pages and creating
unbounded page-cache + swap pressure (≈ 15 GB of routed pages touched per
prompt at top-32 hot-set). Fixed with `madvise(MADV_DONTNEED)` on interior
pages after H2D copy completion. Same-process smoke now plateaus at
`MemAvailable` ≈ 33–40 GB and `SwapFree` stable across all 5 prompts.

### 5. Cold-load async + batched infrastructure

Per-slot `cudaEvent_t` + `cudaStreamWaitEvent` from the consumer stream onto
the prefetch stream + `cudaLaunchHostFunc` for deferred madvise so the decode
thread never blocks on copy completion. Then collapsed the per-expert
event/wait/host-function fan-out down to one event + one wait + one
host-function per pack call (top-K = 6 experts batched together). Outputs
remained bit-identical across these infrastructure changes.

## Performance notes

This is **not** a tuned engine. Cumulative effect of the async + batched
cold-load infrastructure (top-32 hot-set, max_new=16, 5 greedy prompts):

- prompts wall total: **−6.1 %**
- effective decode TPS (excl. init/prewarm): **+4.7 %**
- end-to-end TPS (incl. init/prewarm): **+5.7 %**

Throughput is currently **decode-bandwidth-bound** by routed-expert cold-loads
from mmap, not by compute. Total cold-load miss volume is unchanged across
these changes — they reduce the per-miss coordination cost, not the H2D bytes.

A per-layer miss histogram (5 prompts × 16 decode tokens, top-32 prewarm):

- 6 110 routed-expert cold-loads
- top-10 layers cover only 28.8 % of misses (vs 23 % uniform, 50 % concentrated)
- spread max/min ≈ 2.7 ×

Routing is dispersed both cross-layer within a prompt and dynamically across
prompts. **Static prewarm scaling does not help further** (verified: top-64
prewarm raises memory cost 18 GB and prewarm time 60 % while reducing miss
count by only 1.5 %).

The next perf gain has to come from prefetch / overlap, not from a larger
static hot-set. That work is ongoing internally and not part of this release.

## Known limitations

- No CUDA Graph capture path
- No MTP / speculative decode
- No request batching
- No OpenAI-compatible server / no streaming
- Hot-set routing policy is basic LRU + static prewarm
- Absolute throughput is not yet competitive; this is a correctness +
  feasibility proof, not a serving engine
- `max_seq` capped at 128 in current harnesses
- Tokenizer wrapper uses raw HF `tokenizers.Tokenizer` to avoid `transformers`
  drift on `rope_scaling` config fields

## Directory layout

```
.
├── README.md
├── LICENSE                                 # Apache-2.0
├── env.example                             # copy to .env, edit, source it
├── kernel/
│   ├── cpp-engine-mxfp4/                   # main engine
│   ├── fp8-dense/                          # FP8 E4M3 dense GEMV (M=1)
│   ├── mxfp4-dense/                        # MXFP4 dense GEMV (M=1)
│   ├── mxfp4-routed/                       # MXFP4 grouped GEMV (top-K dispatch)
│   ├── rmsnorm-fuse/                       # fused RMSNorm
├── runtime/                                # Python wrapper + bindings
└── tools/
    ├── D17B4_forced_bisect.py              # math correctness gate
    ├── D26C_diag.py                        # same-process smoke + memory diag
    └── D27_baseline.py                     # performance / timing harness
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The DeepSeek V4 Flash model weights are released under their own license by
DeepSeek; this repository does not redistribute weights.

## Status & contributions

This is an experimental research artifact. Issues and pull requests are
welcome but the maintainer's primary focus is upstream correctness and
performance work, not feature parity with full inference frameworks. For
production use, watch
[vLLM #41063](https://github.com/vllm-project/vllm/issues/41063) and the
official [SGLang DeepSeek V4 roadmap](https://github.com/sgl-project/sglang/issues/23602).
