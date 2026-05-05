#!/usr/bin/env python3
"""D26C diagnostic — multi-prompt single-process with full /proc/meminfo logging.
import os

Goal: confirm hypothesis that sys_avail collapse ≈ Cached growth ≈ delta_misses × expert_bytes.

Logs per prompt (pre/post):
  /proc/meminfo: MemFree, MemAvailable, Cached, Buffers, SReclaimable,
                 SUnreclaim, Unevictable, Mlocked, SwapFree, Dirty, Writeback
  /proc/self/status: VmRSS, VmHWM, VmSize, VmData, VmPeak
  torch: memory_allocated, memory_reserved
  hotset: lookups, hits, misses, total_bytes_resident

Per-expert bytes (architecture constants):
  w1+w3 packed:   2 × EXPERT_FF(2048) × HIDDEN(4096)/2 = 16 MB
  w2 packed:      HIDDEN(4096) × EXPERT_FF(2048)/2     =  4 MB
  w1+w3 scale:    2 × EXPERT_FF × HIDDEN/32            = 0.5 MB
  w2 scale:       HIDDEN × EXPERT_FF/32                = 0.25 MB
  total ≈ 20.75 MB per expert (mmap pages touched per cold-load)

NO math touch. NO C++ touch. Only Python diagnostic + 1 prompt-per-prompt smoke.
"""
from __future__ import annotations
import ctypes, gc, json, os, sys, time
from collections import Counter
from pathlib import Path

KERNEL_DEPS_WT = os.environ.get("DSV4_KERNEL_DEPS", "")
KERNEL_BUILD_WT = os.environ.get("DSV4_HOME", "")
sys.path.insert(0, f"{KERNEL_BUILD_WT}/runtime")
sys.path.insert(0, f"{KERNEL_DEPS_WT}/runtime")
KDIR = f"{KERNEL_DEPS_WT}/kernel"
DEPS = [f"{KDIR}/fp8-dense/libfp8_e4m3_gemv.so",
        f"{KDIR}/mxfp4-dense/libmxfp4_gemv.so",
        f"{KDIR}/mxfp4-routed/libmxfp4_grouped_gemv.so",
        f"{KDIR}/rmsnorm-fuse/librmsnorm_fuse.so"]
for p in DEPS: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
LIB_PATH = f"{KERNEL_BUILD_WT}/kernel/cpp-engine-mxfp4/libdsv4_engine_mxfp4.so"
os.environ["DSV4_MXFP4_CPP_LIB"] = LIB_PATH
for k, v in [("DSV4_KERNEL_FP8_DENSE", DEPS[0]), ("DSV4_KERNEL_MXFP4_DENSE", DEPS[1]),
             ("DSV4_KERNEL_MXFP4_ROUTED", DEPS[2]), ("DSV4_KERNEL_RMSNORM", DEPS[3]),
             ("FP8_E4M3_GEMV_LIB", DEPS[0]), ("MXFP4_GEMV_LIB", DEPS[1]),
             ("MXFP4_GROUPED_LIB", DEPS[2]), ("RMSNORM_FUSE_LIB", DEPS[3])]:
    os.environ[k] = v

import torch  # noqa: E402
from dsv4_engine_mxfp4 import DSv4EngineMXFP4, N_EXPERTS  # noqa: E402
from dsv4_engine_mxfp4_cpp import DSv4EngineMxfp4Cpp, decode_step  # noqa: E402
from tokenizer_raw import RawTokenizer  # noqa: E402

PROMPTS = [
    ("P1_BOS",            ""),
    ("P2_Hello",          "Hello"),
    ("P3_capital_France", "The capital of France is"),
    ("P4_2plus2",         "2+2="),
    ("P5_Ciao",           "Ciao, come stai?"),
]
MAX_NEW = 16

# Per-expert byte estimate (sum w1+w3+w2 packed + their scales)
EXPERT_FF = 2048
HIDDEN = 4096
W13_PACKED = EXPERT_FF * (HIDDEN // 2)        # 4 MB packed each
W2_PACKED  = HIDDEN * (EXPERT_FF // 2)         # 4 MB packed
W13_SCALE  = EXPERT_FF * (HIDDEN // 32)        # 256 KB each
W2_SCALE   = HIDDEN * (EXPERT_FF // 32)        # 256 KB
EXPERT_BYTES = (2*W13_PACKED + W2_PACKED) + (2*W13_SCALE + W2_SCALE)
EXPERT_MB = EXPERT_BYTES / 1e6
print(f"[D26C-diag] estimated bytes per expert (w1+w2+w3 packed+scale) = {EXPERT_MB:.2f} MB")


class HotsetStats(ctypes.Structure):
    _fields_ = [
        ("n_lookups",            ctypes.c_uint64),
        ("n_hits",               ctypes.c_uint64),
        ("n_misses",             ctypes.c_uint64),
        ("last_miss_latency_us", ctypes.c_uint64),
        ("total_bytes_resident", ctypes.c_uint64),
        ("n_layers",             ctypes.c_int),
        ("hot_per_layer",        ctypes.c_int),
    ]


def cycling_pct(ids):
    if len(ids) <= 1: return 0.0
    return 100.0 * Counter(ids).most_common(1)[0][1] / len(ids)


def categorize(ids, text):
    cyc = cycling_pct(ids)
    if cyc >= 90: return "CYCLING_TOTAL"
    if cyc >= 60: return "CYCLING_PARZIALE"
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    if len(text) > 0 and alpha / len(text) < 0.30: return "TOKEN_NOISE"
    words = text.split()
    if not words: return "EMPTY"
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len > 15: return "TOKEN_NOISE"
    if len(words) <= 1 and len(text) < 4: return "BALBETTANTE_BREVE"
    return "INTELLIGIBILE"


def read_meminfo_full():
    keys = ["MemFree", "MemAvailable", "Cached", "Buffers", "SReclaimable",
            "SUnreclaim", "Unevictable", "Mlocked", "SwapFree", "Dirty", "Writeback"]
    d = {}
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    name = parts[0].rstrip(":")
                    if name in keys:
                        d[name] = int(parts[1]) / 1e6  # GB
    except Exception:
        pass
    return d


def read_status():
    keys = ["VmRSS", "VmHWM", "VmSize", "VmData", "VmPeak"]
    d = {}
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    name = parts[0].rstrip(":")
                    if name in keys:
                        d[name] = int(parts[1]) / 1e6  # GB
    except Exception:
        pass
    return d


def torch_mem():
    try:
        return {"alloc_gb": round(torch.cuda.memory_allocated()/1e9, 2),
                "resv_gb":  round(torch.cuda.memory_reserved()/1e9, 2)}
    except Exception:
        return {"alloc_gb": -1, "resv_gb": -1}


def full_snap():
    return {
        "meminfo_gb":  {k: round(v, 2) for k, v in read_meminfo_full().items()},
        "vm_gb":       {k: round(v, 2) for k, v in read_status().items()},
        "torch_gb":    torch_mem(),
    }


def main():
    t_global = time.time()
    print("=" * 80)
    print(f"[D26C-diag] multi-prompt single-process diag, max_new={MAX_NEW}")
    print("=" * 80)

    print("[D26C-diag] R2 init ...")
    r2 = DSv4EngineMXFP4(max_layers=43)
    print("[D26C-diag] C++ init + wire ...")
    cpp = DSv4EngineMxfp4Cpp(n_layers=43, max_seq=128)
    cpp.wire_weights(r2)
    cpp.lib.dsv4_mxfp4_alloc_kv_cache(cpp._state, 128)  # ONCE
    lib = cpp.lib

    for fn, ret, args_ in [
        ("hotset_init",            ctypes.c_int, [ctypes.c_int, ctypes.c_int, ctypes.c_int]),
        ("hotset_register_cold",   ctypes.c_int, [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_void_p, ctypes.c_size_t,
                                                  ctypes.c_void_p, ctypes.c_size_t]),
        ("hotset_prewarm_expert",  ctypes.c_int, [ctypes.c_int, ctypes.c_int]),
        ("hotset_stats",           ctypes.c_int, [ctypes.POINTER(HotsetStats)]),
        ("hotset_reset_stats",     ctypes.c_int, []),
        ("dsv4_mxfp4_reset_kv",    None,         [ctypes.c_void_p]),
    ]:
        f = getattr(lib, fn)
        if ret is not None: f.restype = ret
        f.argtypes = args_

    lib.hotset_init(43, N_EXPERTS, 64)
    print("[D26C-diag] hot-set register_cold + prewarm top-32 ...")
    t0 = time.time()
    for L in range(43):
        d_layer = r2._routed_meta[L]
        for E in range(N_EXPERTS):
            d_e = d_layer[E]
            for wk_idx, wname in enumerate(("w1", "w2", "w3")):
                p_t, s_t = d_e[wname]
                lib.hotset_register_cold(L, E, wk_idx,
                    int(p_t.data_ptr()), int(p_t.numel()*p_t.element_size()),
                    int(s_t.data_ptr()), int(s_t.numel()*s_t.element_size()))
    for L in range(43):
        for E in range(32):
            lib.hotset_prewarm_expert(L, E)
    print(f"[D26C-diag] hot-set {time.time()-t0:.1f}s")

    tok = RawTokenizer(os.path.join(os.environ.get("DSV4_WEIGHTS", ""), "tokenizer.json"))
    BOS_ID = 0

    snap_after_init = full_snap()
    print(f"[D26C-diag] snap after_init: {json.dumps(snap_after_init)}")

    results = []
    prev_snap = snap_after_init
    for name, prompt in PROMPTS:
        print(f"\n[D26C-diag] === {name} === prompt={prompt!r}")

        if prompt == "":
            prefill_ids = [BOS_ID]
        else:
            prefill_ids = tok.encode(prompt, add_bos=True)

        lib.dsv4_mxfp4_reset_kv(cpp._state)
        lib.hotset_reset_stats()

        snap_pre = full_snap()
        hs_pre = HotsetStats(); lib.hotset_stats(ctypes.byref(hs_pre))

        # prefill
        t_p = time.time(); last_tok = None; status = "ok"
        for tid in prefill_ids:
            rc = decode_step(cpp, int(tid))
            if rc < 0:
                status = f"prefill_err_rc={rc}"; break
            last_tok = rc
        prefill_s = time.time() - t_p

        out_ids = []
        decode_tps = 0.0
        if status == "ok":
            cur_tok = last_tok
            t_d = time.time()
            for step in range(MAX_NEW):
                if cur_tok is None or cur_tok < 0 or cur_tok >= 129280:
                    status = f"decode_stop_step={step}_bad_tok={cur_tok}"; break
                out_ids.append(int(cur_tok))
                rc = decode_step(cpp, int(cur_tok))
                if rc < 0:
                    status = f"decode_err_rc={rc}_step={step}"; break
                cur_tok = rc
            decode_s = time.time() - t_d
            decode_tps = len(out_ids) / max(decode_s, 1e-6)

        text = tok.decode(out_ids) if out_ids else ""
        cyc = cycling_pct(out_ids)
        cat = categorize(out_ids, text) if out_ids else "EMPTY_NO_OUT"

        snap_post = full_snap()
        hs_post = HotsetStats(); lib.hotset_stats(ctypes.byref(hs_post))
        delta_misses = hs_post.n_misses - hs_pre.n_misses
        touched_gb_estimate = delta_misses * EXPERT_BYTES / 1e9

        # Diff meminfo for hypothesis
        m_pre = snap_pre["meminfo_gb"]
        m_post = snap_post["meminfo_gb"]
        avail_drop = m_pre.get("MemAvailable", 0) - m_post.get("MemAvailable", 0)
        cached_growth = m_post.get("Cached", 0) - m_pre.get("Cached", 0)
        free_drop = m_pre.get("MemFree", 0) - m_post.get("MemFree", 0)

        result = {
            "name": name, "prompt": prompt, "status": status,
            "prefill_ids": prefill_ids, "prefill_s": round(prefill_s, 2),
            "out_ids": out_ids, "out_text": text, "cycling_pct": round(cyc, 2),
            "category": cat, "decode_tps": round(decode_tps, 3),
            "snap_pre": snap_pre, "snap_post": snap_post,
            "delta_misses": delta_misses,
            "touched_bytes_estimate_gb": round(touched_gb_estimate, 2),
            "avail_drop_gb": round(avail_drop, 2),
            "cached_growth_gb": round(cached_growth, 2),
            "free_drop_gb": round(free_drop, 2),
        }
        results.append(result)

        print(f"[D26C-diag] {name}: status={status} cat={cat} cyc={cyc:.1f}% out={text[:40]!r}")
        print(f"[D26C-diag]   delta_misses={delta_misses}  touched_estim={touched_gb_estimate:.2f} GB")
        print(f"[D26C-diag]   MemAvail_drop={avail_drop:.2f} GB  Cached_growth={cached_growth:.2f} GB  MemFree_drop={free_drop:.2f} GB")
        print(f"[D26C-diag]   meminfo_post = {json.dumps(snap_post['meminfo_gb'])}")
        print(f"[D26C-diag]   torch_post = {json.dumps(snap_post['torch_gb'])}")
        prev_snap = snap_post

    # Summary table
    print("\n" + "=" * 100)
    print("D26C-DIAG SUMMARY (hypothesis: avail_drop ≈ cached_growth ≈ delta_misses × expert_bytes)")
    print("=" * 100)
    print(f"{'name':<22} {'status':<28} {'misses':>7} {'estim_GB':>9} {'avail_drop':>11} {'cached_grow':>12} {'free_drop':>10}")
    for r in results:
        print(f"{r['name']:<22} {r['status']:<28} {r['delta_misses']:>7} "
              f"{r['touched_bytes_estimate_gb']:>9.2f} {r['avail_drop_gb']:>11.2f} "
              f"{r['cached_growth_gb']:>12.2f} {r['free_drop_gb']:>10.2f}")

    out_p = Path(os.environ.get("DSV4_OUT", "/tmp") + "/D26C_diag.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w") as f:
        json.dump({"max_new": MAX_NEW, "results": results,
                   "expert_bytes": EXPERT_BYTES,
                   "snap_after_init": snap_after_init}, f, indent=2)
    print(f"\nreport -> {out_p}")
    print(f"wall: {time.time()-t_global:.1f}s")


if __name__ == "__main__":
    main()
