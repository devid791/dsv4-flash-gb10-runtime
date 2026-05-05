# Bring-up changelog

The runtime went through six logical phases, each captured as one commit:

1. **Phase 1** — scaffold + dependent CUDA kernels (FP8 dense, MXFP4 dense
   + grouped, fused RMSNorm, MLA / KV cache / Lightning Indexer) and Python
   runtime bindings.
2. **Phase 2** — HC residual broadcast correctness fix.
3. **Phase 3** — MoE W2 / W3 enum ordering correctness fix.
4. **Phase 4** — YARN compressed RoPE piecewise NTK ramp port +
   idempotent `alloc_kv_cache`.
5. **Phase 5** — same-process lifecycle `madvise(MADV_DONTNEED)` for
   mmap-backed cold-loads + async cold-load (`cudaEvent` +
   `cudaStreamWaitEvent` + `cudaLaunchHostFunc` for deferred madvise).
6. **Phase 6** — batched cold-load (one event / wait / host-func per pack
   call instead of per-expert) + validation harnesses
   (`D17B4_forced_bisect.py`, `D26C_diag.py`, `D27_baseline.py`) +
   final README.

See README.md for a fuller technical writeup of each fix.
