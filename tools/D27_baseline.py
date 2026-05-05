#!/usr/bin/env python3
"""D27 baseline — performance profile, no math changes.
import os

Same-process smoke max_new=16 with full timing breakdown:
  - R2 load
  - C++ init (init_engine + alloc_kv_cache + alloc_hotset_banks + precompute_rope)
  - wire_weights
  - hotset_register_cold
  - hotset_prewarm (top-N)
  - per-prompt prefill_s
  - per-token decode latency (ms array)
  - hotset stats per prompt (lookups/hits/misses + delta_misses)
  - effective TPS excluding init/prewarm
  - end-to-end TPS including init/prewarm
  - mem snapshots from /proc/meminfo per prompt

Usage:
  python3 D27_baseline.py --top_n 32 --max_new 16 --tag baseline_top32
  python3 D27_baseline.py --top_n 64 --max_new 16 --tag baseline_top64

Output: ${DSV4_OUT}/D27_<tag>.json
"""
from __future__ import annotations
import argparse, ctypes, json, os, sys, time
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

import torch
from dsv4_engine_mxfp4 import DSv4EngineMXFP4, N_EXPERTS
from dsv4_engine_mxfp4_cpp import DSv4EngineMxfp4Cpp, decode_step
from tokenizer_raw import RawTokenizer

PROMPTS = [
    ("P1_BOS",            ""),
    ("P2_Hello",          "Hello"),
    ("P3_capital_France", "The capital of France is"),
    ("P4_2plus2",         "2+2="),
    ("P5_Ciao",           "Ciao, come stai?"),
]

EXPERT_BYTES = (2*2048*2048 + 4096*1024) + (2*2048*128 + 4096*64)


class HotsetStats(ctypes.Structure):
    _fields_ = [("n_lookups",            ctypes.c_uint64),
                ("n_hits",               ctypes.c_uint64),
                ("n_misses",             ctypes.c_uint64),
                ("last_miss_latency_us", ctypes.c_uint64),
                ("total_bytes_resident", ctypes.c_uint64),
                ("n_layers",             ctypes.c_int),
                ("hot_per_layer",        ctypes.c_int)]


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


def read_meminfo():
    keys = ["MemFree", "MemAvailable", "Cached", "SwapFree"]
    d = {}
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    name = parts[0].rstrip(":")
                    if name in keys:
                        d[name] = round(int(parts[1])/1e6, 2)
    except Exception:
        pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_n",  type=int, default=32, help="prewarm top-N expert per layer")
    ap.add_argument("--bank",   type=int, default=64, help="hotset bank size per layer (>= top_n)")
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--tag",    type=str, required=True)
    _repo_root = str(Path(__file__).resolve().parent.parent)
    _out_default = os.environ.get("DSV4_OUT", os.path.join(_repo_root, "out"))
    ap.add_argument("--out_dir", type=str, default=_out_default)
    args = ap.parse_args()

    if args.bank < args.top_n:
        args.bank = args.top_n

    out = {
        "tag": args.tag, "top_n": args.top_n, "bank": args.bank,
        "max_new": args.max_new, "expert_bytes": EXPERT_BYTES,
        "phases": {}, "prompts": [],
    }

    t_global = time.time()
    print("=" * 80)
    print(f"[D27] tag={args.tag} top_n={args.top_n} bank={args.bank} max_new={args.max_new}")
    print("=" * 80)

    # Phase: R2 load
    t0 = time.time()
    print("[D27] phase: R2 load ...")
    r2 = DSv4EngineMXFP4(max_layers=43)
    out["phases"]["r2_load_s"] = round(time.time() - t0, 2)
    out["phases"]["mem_after_r2"] = read_meminfo()

    # Phase: C++ init
    t0 = time.time()
    print("[D27] phase: C++ init ...")
    cpp = DSv4EngineMxfp4Cpp(n_layers=43, max_seq=128)
    out["phases"]["cpp_init_s"] = round(time.time() - t0, 2)

    # Phase: wire weights
    t0 = time.time()
    print("[D27] phase: wire weights ...")
    cpp.wire_weights(r2)
    cpp.lib.dsv4_mxfp4_alloc_kv_cache(cpp._state, 128)
    out["phases"]["wire_s"] = round(time.time() - t0, 2)

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

    # Phase: hotset_init + register_cold
    t0 = time.time()
    print(f"[D27] phase: hotset_init bank={args.bank} ...")
    lib.hotset_init(43, N_EXPERTS, args.bank)
    print(f"[D27] phase: register_cold (43 × 256 × 3 = 33024 entries) ...")
    for L in range(43):
        d_layer = r2._routed_meta[L]
        for E in range(N_EXPERTS):
            d_e = d_layer[E]
            for wk_idx, wname in enumerate(("w1", "w2", "w3")):
                p_t, s_t = d_e[wname]
                lib.hotset_register_cold(L, E, wk_idx,
                    int(p_t.data_ptr()), int(p_t.numel()*p_t.element_size()),
                    int(s_t.data_ptr()), int(s_t.numel()*s_t.element_size()))
    out["phases"]["hotset_register_cold_s"] = round(time.time() - t0, 2)

    # Phase: prewarm top-N
    t0 = time.time()
    print(f"[D27] phase: prewarm top-{args.top_n} × 43 layer = {args.top_n*43} loads ...")
    for L in range(43):
        for E in range(args.top_n):
            lib.hotset_prewarm_expert(L, E)
    out["phases"]["prewarm_s"] = round(time.time() - t0, 2)
    out["phases"]["mem_after_prewarm"] = read_meminfo()

    tok = RawTokenizer(os.path.join(os.environ.get("DSV4_WEIGHTS", ""), "tokenizer.json"))
    BOS_ID = 0

    decode_total_tokens = 0
    decode_total_s = 0.0
    init_done_t = time.time() - t_global
    out["phases"]["init_done_at_s"] = round(init_done_t, 2)

    prompts_t0 = time.time()
    for name, prompt in PROMPTS:
        if prompt == "":
            prefill_ids = [BOS_ID]
        else:
            prefill_ids = tok.encode(prompt, add_bos=True)

        lib.dsv4_mxfp4_reset_kv(cpp._state)
        lib.hotset_reset_stats()

        snap_pre = read_meminfo()
        hs_pre = HotsetStats(); lib.hotset_stats(ctypes.byref(hs_pre))

        # prefill — measure as a single block (multiple decode_step but no output)
        t_p = time.time()
        last_tok = None
        status = "ok"
        for tid in prefill_ids:
            rc = decode_step(cpp, int(tid))
            if rc < 0:
                status = f"prefill_err_rc={rc}"; break
            last_tok = rc
        prefill_s = time.time() - t_p

        # decode — record per-token latency
        out_ids = []
        token_lat_ms = []
        if status == "ok":
            cur_tok = last_tok
            for step in range(args.max_new):
                if cur_tok is None or cur_tok < 0 or cur_tok >= 129280:
                    status = f"decode_stop_step={step}_bad_tok={cur_tok}"; break
                out_ids.append(int(cur_tok))
                t_tok = time.time()
                rc = decode_step(cpp, int(cur_tok))
                lat_ms = (time.time() - t_tok) * 1000.0
                token_lat_ms.append(round(lat_ms, 1))
                if rc < 0:
                    status = f"decode_err_rc={rc}_step={step}"; break
                cur_tok = rc

        decode_s = sum(token_lat_ms) / 1000.0 if token_lat_ms else 0.0
        decode_tps = len(out_ids) / max(decode_s, 1e-6)
        decode_total_tokens += len(out_ids)
        decode_total_s += decode_s

        text = tok.decode(out_ids) if out_ids else ""
        cyc = cycling_pct(out_ids)
        cat = categorize(out_ids, text) if out_ids else "EMPTY_NO_OUT"

        snap_post = read_meminfo()
        hs_post = HotsetStats(); lib.hotset_stats(ctypes.byref(hs_post))
        delta_lookups = hs_post.n_lookups - hs_pre.n_lookups
        delta_hits    = hs_post.n_hits    - hs_pre.n_hits
        delta_misses  = hs_post.n_misses  - hs_pre.n_misses
        miss_rate = (delta_misses / max(delta_lookups, 1)) * 100.0
        touched_gb = delta_misses * EXPERT_BYTES / 1e9

        prompt_result = {
            "name": name, "prompt": prompt, "status": status,
            "n_prefill_tokens": len(prefill_ids),
            "prefill_s": round(prefill_s, 3),
            "out_ids": out_ids, "out_text": text,
            "cycling_pct": round(cyc, 2), "category": cat,
            "n_decode_tokens": len(out_ids),
            "decode_s": round(decode_s, 3),
            "decode_tps": round(decode_tps, 3),
            "token_lat_ms": token_lat_ms,
            "miss_rate_pct": round(miss_rate, 2),
            "delta_lookups": delta_lookups,
            "delta_hits": delta_hits,
            "delta_misses": delta_misses,
            "touched_estim_gb": round(touched_gb, 2),
            "snap_pre": snap_pre, "snap_post": snap_post,
        }
        out["prompts"].append(prompt_result)
        print(f"[D27] {name}: status={status} cat={cat} cyc={cyc:.1f}% tps={decode_tps:.2f} "
              f"misses={delta_misses}/{delta_lookups} ({miss_rate:.1f}%)")
        print(f"[D27]   prefill={prefill_s:.2f}s decode={decode_s:.2f}s tokens={len(out_ids)}")
        print(f"[D27]   tok_lat ms (first 8): {token_lat_ms[:8]}")
        print(f"[D27]   tok_lat ms (last 8):  {token_lat_ms[-8:]}")
        print(f"[D27]   out_text = {text!r}")

    prompts_total_s = time.time() - prompts_t0
    out["phases"]["prompts_total_s"] = round(prompts_total_s, 2)

    # End-to-end stats
    completed = sum(1 for p in out["prompts"] if p.get("status") == "ok")
    intel = sum(1 for p in out["prompts"] if p.get("category") == "INTELLIGIBILE")
    eff_tps = decode_total_tokens / max(decode_total_s, 1e-6)
    e2e_s   = time.time() - t_global
    e2e_tps = decode_total_tokens / max(e2e_s, 1e-6)

    out["summary"] = {
        "completed": completed, "total": len(PROMPTS),
        "intelligible": intel,
        "decode_total_tokens": decode_total_tokens,
        "decode_total_s": round(decode_total_s, 2),
        "effective_decode_tps_excl_init": round(eff_tps, 3),
        "e2e_total_s": round(e2e_s, 2),
        "e2e_tps_incl_init": round(e2e_tps, 3),
    }

    print("\n" + "=" * 80)
    print(f"[D27] === SUMMARY tag={args.tag} ===")
    print(f"[D27] phases:")
    for k, v in out["phases"].items():
        if isinstance(v, dict):
            print(f"  {k}: {json.dumps(v)}")
        else:
            print(f"  {k}: {v}")
    print(f"[D27] decode total tokens: {decode_total_tokens}, decode total s: {decode_total_s:.2f}")
    print(f"[D27] effective decode TPS (excl init/prewarm): {eff_tps:.3f}")
    print(f"[D27] end-to-end TPS (incl init/prewarm):       {e2e_tps:.3f}")
    print(f"[D27] e2e wall total: {e2e_s:.2f}s")
    print(f"[D27] {completed}/{len(PROMPTS)} ok, {intel}/{len(PROMPTS)} INTELLIGIBILE")

    out_p = Path(args.out_dir) / f"D27_{args.tag}.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[D27] report -> {out_p}")


if __name__ == "__main__":
    main()
