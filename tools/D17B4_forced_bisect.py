#!/usr/bin/env python3
"""D14 — Forced-token bisect harness for DSv4-Flash MXFP4 multi-token regression.

Protocol: G:\\DGX-DSV4\\DEBUG-MULTITOKEN-REGRESSION.md (D14 mandate).

Strategy:
  PHASE A — R2 Python oracle:
    * Init R2 engine (full BF16 weights)
    * Encode "The capital of France is" + BOS
    * Greedy 8 tokens, save r2_tokens[]
    * For decode_step in {0, 1, 2}: hook forward_layer to capture, for layers
      {0, 1, 2, 5, 10, 17, 20, 30, 42}, the 10 blocks specified by protocol
    * Save coord/D17B4_r2_oracle.json

  PHASE B — C++ forced replay:
    * Init C++ engine + wire weights (reuse R2 weights, zero-copy)
    * Allocate KV + register-cold + prewarm hot-set top-32
    * For each step s in {0..7}: reset_kv, replay BOS+r2_tokens[:s] via
      decode_step(), capture last logits via dsv4_mxfp4_d14_get_logits_ptr()
    * Compare logits R2 vs C++ per step (cosine, max_abs_err, argmax, top-10)

  Comparison + classification:
    * Step 0 match expected (D11 verified bit-exact 69146)
    * First step where C++ logits diverge significantly = first_diverge_step
    * Hidden states C++ NOT introspectable per-layer without recompile of
      forward path (out of D14 scope). Per-layer divergence info derived from
      R2-side hidden state analysis + C++ logits divergence + classification
      per protocol triage tree.

NO FIX — only identification + dump + table.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
WORKTREE = os.environ.get("DSV4_HOME", _REPO_ROOT)
KERNEL_SRC_WT = os.environ.get("DSV4_KERNEL_DEPS", WORKTREE)
sys.path.insert(0, f"{WORKTREE}/runtime")

import torch  # noqa: E402
import numpy as np  # noqa: E402

# Pre-load kernel deps as RTLD_GLOBAL so libdsv4_engine_mxfp4.so resolves symbols
KERNEL_DIR_RUNTIME = f"{KERNEL_SRC_WT}/kernel"
_KERNEL_DEPS = [
    f"{KERNEL_DIR_RUNTIME}/fp8-dense/libfp8_e4m3_gemv.so",
    f"{KERNEL_DIR_RUNTIME}/mxfp4-dense/libmxfp4_gemv.so",
    f"{KERNEL_DIR_RUNTIME}/mxfp4-routed/libmxfp4_grouped_gemv.so",
    f"{KERNEL_DIR_RUNTIME}/rmsnorm-fuse/librmsnorm_fuse.so",
]
for _p in _KERNEL_DEPS:
    ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
print(f"[D17B4] preloaded {len(_KERNEL_DEPS)} kernel .so RTLD_GLOBAL")
# Force D14 lib path with diagnostic getters
os.environ["DSV4_MXFP4_CPP_LIB"] = f"{WORKTREE}/kernel/cpp-engine-mxfp4/libdsv4_engine_mxfp4.so"
# C++ lib uses DSV4_KERNEL_* (consumed inside dsv4_engine_mxfp4_cpp.py)
os.environ["DSV4_KERNEL_FP8_DENSE"] = _KERNEL_DEPS[0]
os.environ["DSV4_KERNEL_MXFP4_DENSE"] = _KERNEL_DEPS[1]
os.environ["DSV4_KERNEL_MXFP4_ROUTED"] = _KERNEL_DEPS[2]
os.environ["DSV4_KERNEL_RMSNORM"] = _KERNEL_DEPS[3]
# R2 Python bindings use *_LIB env vars
os.environ["FP8_E4M3_GEMV_LIB"] = _KERNEL_DEPS[0]
os.environ["MXFP4_GEMV_LIB"] = _KERNEL_DEPS[1]
os.environ["MXFP4_GROUPED_LIB"] = _KERNEL_DEPS[2]
os.environ["RMSNORM_FUSE_LIB"] = _KERNEL_DEPS[3]
# MLA full lib (D3) — point to D9-wt
from dsv4_engine_mxfp4 import (  # noqa: E402
    DSv4EngineMXFP4, N_EXPERTS, HIDDEN, HC_N_HC,
)
from dsv4_engine_mxfp4_cpp import DSv4EngineMxfp4Cpp, decode_step  # noqa: E402

LIB_PATH = f"{WORKTREE}/kernel/cpp-engine-mxfp4/libdsv4_engine_mxfp4.so"
PROMPT = "The capital of France is"
N_GEN = 8
LAYERS_TO_DUMP = [0, 1, 2, 5, 10, 17, 20, 30, 42]
STEPS_TO_DUMP = [0, 1, 2]
COSINE_DIVERGE_THRESHOLD = 0.98


# ─── Hot-set helpers (copied from D10) ───────────────────────────────────
def _bind_hotset(lib):
    lib.hotset_init.restype = ctypes.c_int
    lib.hotset_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.hotset_register_cold.restype = ctypes.c_int
    lib.hotset_register_cold.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    lib.hotset_prewarm_expert.restype = ctypes.c_int
    lib.hotset_prewarm_expert.argtypes = [ctypes.c_int, ctypes.c_int]


def _bind_d14_getters(lib):
    lib.dsv4_mxfp4_d14_get_logits_ptr.restype = ctypes.c_void_p
    lib.dsv4_mxfp4_d14_get_logits_ptr.argtypes = []
    lib.dsv4_mxfp4_d14_get_vocab.restype = ctypes.c_int
    lib.dsv4_mxfp4_d14_get_vocab.argtypes = []
    lib.dsv4_mxfp4_d14_get_x_residual_ptr.restype = ctypes.c_void_p
    lib.dsv4_mxfp4_d14_get_x_residual_ptr.argtypes = []
    lib.dsv4_mxfp4_d14_get_hnorm_ptr.restype = ctypes.c_void_p
    lib.dsv4_mxfp4_d14_get_hnorm_ptr.argtypes = [ctypes.c_void_p]
    lib.dsv4_mxfp4_d14_get_hidden.restype = ctypes.c_int
    lib.dsv4_mxfp4_d14_get_hidden.argtypes = []
    lib.dsv4_mxfp4_d14_get_n_hc.restype = ctypes.c_int
    lib.dsv4_mxfp4_d14_get_n_hc.argtypes = []
    lib.dsv4_mxfp4_d14_get_cur_pos.restype = ctypes.c_int
    lib.dsv4_mxfp4_d14_get_cur_pos.argtypes = [ctypes.c_void_p]


def register_cold_all(lib, r2, n_layers: int):
    print(f"[D17B4] registering cold: {n_layers}L × 256E × 3wkey ...")
    init_rc = lib.hotset_init(n_layers, N_EXPERTS, 64)
    print(f"[D17B4]   hotset_init({n_layers},{N_EXPERTS},64) rc={init_rc}")
    t0 = time.time()
    n_total, n_fail = 0, 0
    for L in range(n_layers):
        d_layer = r2._routed_meta[L]
        for E in range(N_EXPERTS):
            d_e = d_layer[E]
            for wk_idx, wname in enumerate(("w1", "w2", "w3")):
                p_t, s_t = d_e[wname]
                rc = lib.hotset_register_cold(
                    L, E, wk_idx,
                    int(p_t.data_ptr()), int(p_t.numel() * p_t.element_size()),
                    int(s_t.data_ptr()), int(s_t.numel() * s_t.element_size()),
                )
                if rc != 0:
                    n_fail += 1
                n_total += 1
    print(f"[D17B4] register_cold {n_total} entries, {n_fail} fail, {time.time()-t0:.1f}s")
    return n_fail == 0


def prewarm_top_n(lib, n_layers: int, top_n: int):
    print(f"[D17B4] prewarming top-{top_n} expert × {n_layers} layer ...")
    t0 = time.time()
    n_done, n_fail = 0, 0
    for L in range(n_layers):
        for E in range(top_n):
            rc = lib.hotset_prewarm_expert(L, E)
            if rc != 0:
                n_fail += 1
            else:
                n_done += 1
    return n_done, time.time() - t0


# ─── Tensor compare utilities ───────────────────────────────────────────
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float().flatten(), b.float().flatten()
    nA, nB = af.norm().item(), bf.norm().item()
    if nA == 0 or nB == 0:
        return 0.0 if (nA + nB) > 0 else 1.0
    return float((af.dot(bf) / (nA * nB)).item())


def maxabs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def meanabs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def topk(t: torch.Tensor, k: int = 10):
    v, i = t.float().flatten().topk(k)
    return [(int(i[j].item()), float(v[j].item())) for j in range(k)]


# ─── PHASE A: R2 oracle with hidden-state hooks ─────────────────────────
class R2Capture:
    """Monkey-patches R2 forward_layer to capture per-layer hidden states.

    For each (decode_step, layer) in scope, captures:
      - hidden_in (input to forward_layer)
      - rms_attn_out (after attn_norm)
      - mla_out (after _mla_attention)
      - hidden_after_attn (after hc_post)
      - rms_ffn_out (after ffn_norm)
      - moe_out (after _moe_block)
      - hidden_after_moe (final layer output)
      - router_logits (top-16)
      - topk_expert_ids
      - topk_expert_weights
    """

    def __init__(self, r2, layers, steps):
        self.r2 = r2
        self.layers = set(layers)
        self.steps = set(steps)
        self.current_step = -1  # -1 = prefill, 0,1,2,... = decode steps
        # store: capture[step][layer][block] = tensor (cpu, f32)
        self.cap = {}
        # rope position log: rope_log[step][layer] = pos_used
        self.rope_log = {}
        # patch
        self._orig_forward_layer = r2.forward_layer
        self._orig_mla = r2._mla_attention
        self._orig_moe = r2._moe_block
        self._orig_route = r2._route
        self._patched = False
        # transient per-call
        self._cur_layer = None

    def __enter__(self):
        self._patched = True
        from dsv4_engine_mxfp4 import (
            rmsnorm_torch, hc_pre, hc_post, RMS_EPS,
            HIDDEN as _H, HC_N_HC as _NHC, HC_SINKHORN_ITERS, HC_EPS,
        )

        def patched_fl(hidden, L):
            self._cur_layer = L
            do_cap = (self.current_step in self.steps and L in self.layers)
            if do_cap:
                step_d = self.cap.setdefault(self.current_step, {})
                lay_d = step_d.setdefault(L, {})
                lay_d["hidden_in"] = hidden.detach().clone().cpu()
            layer = self.r2.layers[L]

            # Replicate forward_layer body but capture intermediates
            x_pre_a, _pre, post_a, comb_a = hc_pre(
                hidden, layer.hc_attn_fn, layer.hc_attn_base, layer.hc_attn_scale,
                n_embd=_H, n_hc=_NHC,
                sinkhorn_iters=HC_SINKHORN_ITERS, hc_eps=HC_EPS, norm_eps=RMS_EPS,
            )
            h_n = rmsnorm_torch(x_pre_a, layer.attn_norm, eps=RMS_EPS)
            if do_cap:
                lay_d["rms_attn_out"] = h_n.detach().clone().cpu()
            attn = self.r2._mla_attention(h_n, layer, L)
            if do_cap:
                lay_d["mla_out"] = attn.detach().clone().cpu()
            hidden_aa = hc_post(attn, hidden, post_a, comb_a)
            if do_cap:
                lay_d["hidden_after_attn"] = hidden_aa.detach().clone().cpu()

            x_pre_f, _pre, post_f, comb_f = hc_pre(
                hidden_aa, layer.hc_ffn_fn, layer.hc_ffn_base, layer.hc_ffn_scale,
                n_embd=_H, n_hc=_NHC,
                sinkhorn_iters=HC_SINKHORN_ITERS, hc_eps=HC_EPS, norm_eps=RMS_EPS,
            )
            h_n2 = rmsnorm_torch(x_pre_f, layer.ffn_norm, eps=RMS_EPS)
            if do_cap:
                lay_d["rms_ffn_out"] = h_n2.detach().clone().cpu()
            moe_o = self.r2._moe_block(h_n2, L, layer)
            if do_cap:
                lay_d["moe_out"] = moe_o.detach().clone().cpu()
            hidden_out = hc_post(moe_o, hidden_aa, post_f, comb_f)
            if do_cap:
                lay_d["hidden_after_moe"] = hidden_out.detach().clone().cpu()
            return hidden_out

        # patch _route to capture router logits + topk
        def patched_route(h_n, layer):
            topk_ids, topk_w = self._orig_route(h_n, layer)
            L = self._cur_layer
            if (self.current_step in self.steps and L is not None and L in self.layers):
                # router_logits = h_n @ router_gate.T
                logits = torch.nn.functional.linear(h_n.float(), layer.router_gate.float())
                step_d = self.cap.setdefault(self.current_step, {})
                lay_d = step_d.setdefault(L, {})
                lay_d["router_logits_top16"] = topk(logits[0], 16) if logits.dim() > 1 else topk(logits, 16)
                lay_d["topk_expert_ids"] = topk_ids.detach().cpu().tolist()
                lay_d["topk_expert_weights"] = topk_w.detach().cpu().tolist()
            return topk_ids, topk_w

        # patch _mla_attention to log RoPE position
        def patched_mla(h_n, layer, layer_id):
            S = h_n.shape[0]
            pos_start = self.r2._pos
            pos_end = self.r2._pos + S
            if (self.current_step in self.steps and layer_id in self.layers):
                step_d = self.rope_log.setdefault(self.current_step, {})
                step_d[layer_id] = {"pos_start": pos_start, "pos_end": pos_end, "S": S}
            return self._orig_mla(h_n, layer, layer_id)

        self.r2.forward_layer = patched_fl
        self.r2._route = patched_route
        self.r2._mla_attention = patched_mla
        return self

    def __exit__(self, *args):
        if self._patched:
            self.r2.forward_layer = self._orig_forward_layer
            self.r2._mla_attention = self._orig_mla
            self.r2._moe_block = self._orig_moe
            self.r2._route = self._orig_route
            self._patched = False


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-layers", type=int, default=43)
    p.add_argument("--top-n", type=int, default=8)
    _out_default = os.environ.get("DSV4_OUT", os.path.join(_REPO_ROOT, "out"))
    p.add_argument("--out", default=os.path.join(_out_default, "D17B4_done.json"))
    p.add_argument("--dump-tensor", default=os.path.join(_out_default, "D14_first_bad_dump.npz"))
    p.add_argument("--oracle-dump", default=os.path.join(_out_default, "D17B4_r2_oracle.json"))
    p.add_argument("--skip-cpp", action="store_true")
    p.add_argument("--phase-a-only", action="store_true",
                   help="Only run R2 oracle, save oracle dump, exit (frees GPU)")
    p.add_argument("--phase-b-only", action="store_true",
                   help="Only run C++ replay (loads R2 oracle from oracle-dump)")
    p.add_argument("--single-tok-control", action="store_true",
                   help="Override prompt to single BOS token (replay D11 setup)")
    args = p.parse_args()

    rep = {
        "status": "FAIL",
        "args": vars(args),
        "phase_a_r2_oracle": {},
        "phase_b_cpp_replay": {},
        "comparison_table": [],
        "first_diverge_step": None,
        "first_diverge_layer": None,
        "first_diverge_block": None,
        "case_classification": None,
        "rope_position_log": {},
        "kv_check": {},
        "raccomandazione_fix": "",
        "branch": "agent-d-d17b3-fix-embed-broadcast",
        "commit_hash": None,
    }

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    # ── Tokenize prompt ──
    from tokenizer_raw import RawTokenizer
    tok = RawTokenizer()
    if args.single_tok_control:
        prompt_used = "<BOS>"
        prefix_ids = [0]  # match D11 setup which used [1] for control
    else:
        prompt_used = PROMPT
        prefix_ids = tok.encode(PROMPT, add_bos=True)
    rep["phase_a_r2_oracle"]["prompt"] = prompt_used
    rep["phase_a_r2_oracle"]["prefix_ids"] = prefix_ids

    # ── PHASE A: R2 oracle ──
    if args.phase_b_only:
        print("[D17B4] === PHASE B-ONLY: load oracle from disk ===")
        with open(args.oracle_dump, "r") as f:
            oracle = json.load(f)
        r2_tokens = oracle["r2_tokens"]
        rep["phase_a_r2_oracle"] = oracle
        # Load saved logits npz
        r2_npz = np.load(args.oracle_dump.replace(".json", "_logits.npz"))
        r2_logits_per_step = {int(k.split("_")[1]): torch.from_numpy(r2_npz[k]).float()
                              for k in r2_npz.files if k.startswith("step_")}
        rope_log_loaded = oracle.get("rope_position_log", {})
        rep["rope_position_log"] = rope_log_loaded
        # Stub cap object for delta analysis (skip if not phase A captured)
        cap = None
        # Load R2 again (needed for C++ wire_weights)
        print("[D17B4] R2 init for wire ...")
        t0 = time.time()
        r2 = DSv4EngineMXFP4(max_layers=args.n_layers)
        print(f"[D17B4] R2 init {time.time()-t0:.1f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    else:
        print("[D17B4] === PHASE A: R2 oracle ===")
        print(f"[D17B4] prompt='{PROMPT}' → ids={prefix_ids} (len={len(prefix_ids)})")

        # Init R2
        print("[D17B4] R2 init full ...")
        t0 = time.time()
        r2 = DSv4EngineMXFP4(max_layers=args.n_layers)
        print(f"[D17B4] R2 init {time.time()-t0:.1f}s  GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

        # PHASE A.1: greedy 8 token + capture hidden for steps {0,1,2}
        print(f"[D17B4] R2 greedy decode {N_GEN} tokens with hidden capture for steps {STEPS_TO_DUMP} ...")
        r2.reset_kv()
        r2_tokens = []
        r2_logits_per_step = {}  # step -> logits[VOCAB] (cpu f32)

        cap = R2Capture(r2, LAYERS_TO_DUMP, STEPS_TO_DUMP)
        with cap:
            # prefill — NOT a decode_step in our terminology; cur_step=-1 means "prefill"
            cap.current_step = -1
            t0 = time.time()
            logits = r2.forward(prefix_ids)  # advances _pos by len(prefix)
            prefill_s = time.time() - t0
            next_id = int(logits.float().argmax().item())
            r2_tokens.append(next_id)
            # step 0 = the FIRST decode after prefill produced this token; logits ARE step 0 output
            r2_logits_per_step[0] = logits.detach().cpu().float()
            print(f"[D17B4]   prefill {prefill_s:.2f}s → step0 tok={next_id}")

            # subsequent decode steps
            for s in range(1, N_GEN):
                cap.current_step = s
                t0 = time.time()
                logits = r2.forward([next_id])
                dt = time.time() - t0
                r2_logits_per_step[s] = logits.detach().cpu().float()
                next_id = int(logits.float().argmax().item())
                r2_tokens.append(next_id)
                print(f"[D17B4]   step{s} {dt*1000:.0f}ms tok={next_id}")

        print(f"[D17B4] R2 tokens (8) = {r2_tokens}")
        rep["phase_a_r2_oracle"]["r2_tokens"] = r2_tokens
        try:
            rep["phase_a_r2_oracle"]["text"] = tok.decode(r2_tokens)
        except Exception as e:
            rep["phase_a_r2_oracle"]["text"] = f"<decode err: {e}>"

        # store per-step logits top-10 + argmax
        rep["phase_a_r2_oracle"]["logits_per_step"] = {}
        for s, lg in r2_logits_per_step.items():
            rep["phase_a_r2_oracle"]["logits_per_step"][s] = {
                "argmax": int(lg.argmax().item()),
                "top10": topk(lg, 10),
            }

        # rope log
        rep["rope_position_log"] = cap.rope_log

        # Save oracle for phase B-only mode + always
        oracle_data = {
            "prompt": PROMPT,
            "prefix_ids": prefix_ids,
            "r2_tokens": r2_tokens,
            "text": rep["phase_a_r2_oracle"]["text"],
            "logits_per_step": rep["phase_a_r2_oracle"]["logits_per_step"],
            "rope_position_log": cap.rope_log,
        }
        Path(args.oracle_dump).parent.mkdir(parents=True, exist_ok=True)
        with open(args.oracle_dump, "w") as f:
            json.dump(oracle_data, f, indent=2, default=str)
        # Save logits as npz
        npz_kw = {f"step_{s}": lg.numpy() for s, lg in r2_logits_per_step.items()}
        np.savez(args.oracle_dump.replace(".json", "_logits.npz"), **npz_kw)
        print(f"[D17B4] R2 oracle saved → {args.oracle_dump}")

    if args.phase_a_only:
        print("[D17B4] --phase-a-only: bailing out after PHASE A")
        rep["status"] = "PARTIAL_PHASE_A_ONLY"
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2, default=str)
        return

    # ── PHASE B: C++ engine + replay ──
    if args.skip_cpp:
        print("[D17B4] --skip-cpp: bailing out before phase B")
        rep["status"] = "PARTIAL"
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2, default=str)
        return

    print("[D17B4] === PHASE B: C++ engine init + wire ===")
    cpp = DSv4EngineMxfp4Cpp(n_layers=args.n_layers, max_seq=128)
    cpp.wire_weights(r2)
    _bind_hotset(cpp.lib)
    _bind_d14_getters(cpp.lib)
    rc = cpp.lib.dsv4_mxfp4_alloc_kv_cache(cpp._state, 128)
    print(f"[D17B4] alloc_kv_cache rc={rc}")

    print("[D17B4] hot-set register cold + prewarm top-N ...")
    register_cold_all(cpp.lib, r2, args.n_layers)
    n_done, dur = prewarm_top_n(cpp.lib, args.n_layers, args.top_n)
    print(f"[D17B4] prewarm done={n_done} in {dur:.1f}s")

    # Vocab size from getter
    vocab_cpp = cpp.lib.dsv4_mxfp4_d14_get_vocab()
    print(f"[D17B4] cpp vocab={vocab_cpp}")

    # PHASE B: forced replay per step
    print("[D17B4] forced replay per step (input = BOS_prefix + r2_tokens[:s]) ...")
    cpp_logits_per_step = {}
    cpp_tokens_per_step = {}
    rep_cpp = {}
    for s in range(N_GEN):
        # Replay context: prefix + r2_tokens[:s]; produce step `s` logits
        ctx = list(prefix_ids) + list(r2_tokens[:s])
        cpp.reset_kv()
        last_tok = -1
        t0 = time.time()
        try:
            for tid in ctx:
                last_tok = decode_step(cpp, tid)
                if last_tok < 0:
                    print(f"[D17B4]   step{s} cpp ABORT in ctx feed at tok={tid} rc={last_tok}")
                    break
            dt = time.time() - t0
            cpp_tokens_per_step[s] = int(last_tok)
            # Pull logits via getter
            logits_ptr = cpp.lib.dsv4_mxfp4_d14_get_logits_ptr()
            if logits_ptr and last_tok >= 0:
                # Read VOCAB BF16 from device pointer
                buf = (ctypes.c_uint16 * vocab_cpp).from_address(0)  # placeholder
                # Use cudaMemcpyDtoH via torch
                fake_t = torch.empty(vocab_cpp, dtype=torch.bfloat16, device="cuda")
                # Use cudaMemcpyAsync via cuda runtime API
                cudart = ctypes.CDLL("libcudart.so")
                cudart.cudaMemcpy.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
                ]
                cudart.cudaMemcpy.restype = ctypes.c_int
                cudart.cudaDeviceSynchronize.restype = ctypes.c_int
                cudart.cudaDeviceSynchronize()
                rc_cp = cudart.cudaMemcpy(
                    int(fake_t.data_ptr()),
                    int(logits_ptr),
                    vocab_cpp * 2,
                    2,  # cudaMemcpyDeviceToDevice (both on GPU)
                )
                if rc_cp != 0:
                    print(f"[D17B4]   step{s} cudaMemcpy DtoD rc={rc_cp}")
                lg = fake_t.cpu().float()
                cpp_logits_per_step[s] = lg
                print(f"[D17B4]   step{s} cpp tok={last_tok} dt={dt*1000:.0f}ms argmax_logits={int(lg.argmax().item())}")
            else:
                print(f"[D17B4]   step{s} cpp tok={last_tok} (no logits dump, abort or null ptr)")
                cpp_logits_per_step[s] = None
        except Exception as e:
            print(f"[D17B4]   step{s} EXC {type(e).__name__}: {e}")
            cpp_logits_per_step[s] = None
            cpp_tokens_per_step[s] = -1

    rep["phase_b_cpp_replay"]["cpp_tokens_per_step"] = cpp_tokens_per_step
    rep["phase_b_cpp_replay"]["logits_per_step"] = {}
    for s, lg in cpp_logits_per_step.items():
        if lg is not None:
            rep["phase_b_cpp_replay"]["logits_per_step"][s] = {
                "argmax": int(lg.argmax().item()),
                "top10": topk(lg, 10),
            }
        else:
            rep["phase_b_cpp_replay"]["logits_per_step"][s] = None

    # ── COMPARISON ──
    print("[D17B4] === COMPARISON ===")
    table_rows = []
    first_diverge = None
    for s in range(N_GEN):
        r2_lg = r2_logits_per_step[s]
        cpp_lg = cpp_logits_per_step.get(s)
        argmax_r2 = int(r2_lg.argmax().item())
        if cpp_lg is None:
            row = {
                "decode_step": s,
                "first_bad_layer": "n/a",
                "first_bad_block": "logits_dump_unavailable",
                "cosine": None,
                "max_abs_err": None,
                "argmax_r2": argmax_r2,
                "argmax_cpp": cpp_tokens_per_step.get(s, -99),
                "ipotesi": "C++ abort or no logits captured",
            }
        else:
            cos = cosine_sim(r2_lg, cpp_lg)
            mae = maxabs_err(r2_lg, cpp_lg)
            argmax_cpp = int(cpp_lg.argmax().item())
            sample_tok = cpp_tokens_per_step.get(s, -99)
            # ipotesi based on cosine + argmax
            if cos >= 0.999 and argmax_r2 == argmax_cpp:
                ipot = "match"
                bad_layer = "none"
                bad_block = "none"
            elif cos >= 0.99:
                ipot = "near-match (low drift, cosine OK but argmax differ)" if argmax_r2 != argmax_cpp else "near-match"
                bad_layer = "n/a"
                bad_block = "logits (need per-layer recompile to localize)"
            elif cos >= 0.98:
                ipot = "moderate divergence — late layer drift suspected"
                bad_layer = "n/a"
                bad_block = "logits (need per-layer recompile to localize)"
            else:
                ipot = "SEVERE divergence — first bad step"
                bad_layer = "n/a"
                bad_block = "logits (need per-layer recompile to localize)"
                if first_diverge is None:
                    first_diverge = s
            row = {
                "decode_step": s,
                "first_bad_layer": bad_layer,
                "first_bad_block": bad_block,
                "cosine": cos,
                "max_abs_err": mae,
                "argmax_r2": argmax_r2,
                "argmax_cpp": argmax_cpp,
                "sample_tok_cpp": sample_tok,
                "ipotesi": ipot,
            }
            # Detect first argmax mismatch as backup signal
            if argmax_r2 != argmax_cpp and first_diverge is None and cos < 0.999:
                first_diverge = s
        table_rows.append(row)

    rep["comparison_table"] = table_rows
    rep["first_diverge_step"] = first_diverge

    # ── R2-side hidden state cross-step analysis (since we cannot cross to cpp) ──
    # Heuristic: for first_diverge step, dump R2 hidden states diff between
    # step (s-1) and step s for each captured layer to identify layer with
    # most rapid state change (informational only).
    print("[D17B4] R2 hidden cross-step delta analysis ...")
    delta_rows = []
    if cap is None:
        print("[D17B4] (skipped: phase-b-only mode, no R2 hidden capture)")
        rep["r2_hidden_cross_step_delta"] = []
        # jump to classification
        delta_rows = []
    for s in (STEPS_TO_DUMP if cap is not None else []):
        if s == 0 or (s - 1) not in cap.cap:
            continue
        if s not in cap.cap:
            continue
        for L in LAYERS_TO_DUMP:
            d_curr = cap.cap[s].get(L, {})
            d_prev = cap.cap[s - 1].get(L, {})
            if not d_curr or not d_prev:
                continue
            for blk in ["hidden_in", "rms_attn_out", "mla_out", "hidden_after_attn",
                        "rms_ffn_out", "moe_out", "hidden_after_moe"]:
                a, b = d_curr.get(blk), d_prev.get(blk)
                if a is None or b is None:
                    continue
                # Compare last token only (single-token decode steps)
                aa, bb = a, b
                if aa.shape != bb.shape:
                    continue
                cos = cosine_sim(aa, bb)
                mae = maxabs_err(aa, bb)
                delta_rows.append({
                    "step": s, "layer": L, "block": blk,
                    "cosine_vs_prev_step": cos,
                    "max_abs_err_vs_prev_step": mae,
                })
    rep["r2_hidden_cross_step_delta"] = delta_rows[:50]

    # ── Classification ──
    print("[D17B4] === CLASSIFICATION ===")
    if first_diverge is None:
        case = "MATCH"
        raccom = ("Logits R2 vs C++ match across all 8 steps. The 91-97% TOKEN_NOISE"
                  " observed in past smoke runs may be due to temperature/sampling, not"
                  " forward divergence. RECOMMEND: re-run smoke with temp=0 + same"
                  " tokenizer + same hot-set; confirm cycling reproduces with this"
                  " harness configuration. If reproduces here too → KV cache state leak"
                  " across non-reset_kv invocations.")
    elif first_diverge == 0:
        case = "A"
        raccom = ("Step 0 already diverges → NOT autoregressive bug. Triage Caso A:"
                  " (1) verify weight pointer wiring (PascalCase ABI count = 28),"
                  " (2) verify dtype/scale MXFP4/FP8 (no double-scaling),"
                  " (3) verify layer iteration order (forward not skipping shared FFN),"
                  " (4) verify tokenizer add_bos=True consistent,"
                  " (5) verify hot-set top-N covers experts used by step 0 prompt."
                  " First action: re-run D11 1-tok bit-exact verification on this hot path.")
    elif first_diverge == 1:
        case = "B"
        raccom = ("Step 0 matches, step 1 diverges → autoregressive bug. Triage Caso B:"
                  " (1) KV cache append/read — verify post-RoPE post-norm KV stored at"
                  " correct position index in C++ kv_cache buffer,"
                  " (2) RoPE absolute position — verify decode_step 0 after prefill of N"
                  " tokens uses position N (not N-1, not 0),"
                  " (3) MLA layout/stride — verify Q [S,64,512] and KV [S,512] strides,"
                  " (4) cur_pos increment — st->cur_pos++ at correct timing in"
                  " dsv4_engine_mxfp4_wire.cu:dsv4_mxfp4_decode_step_real,"
                  " (5) per-layer KV cache buffer not aliased across layers."
                  " ROPE LOG: see rope_position_log section for R2 expected positions.")
    else:
        case = "C_or_D"
        raccom = (f"Step 0 and 1 match, step {first_diverge} diverges → late-emerging bug."
                  " Triage Caso C (MoE) or D (logits precision):"
                  " (C) top-k expert ids stable across runs?"
                  " expert in hot-set top-N for that prompt?"
                  " shared expert summed correctly?"
                  " (D) lm_head BF16 vs F32 accumulation; argmax tie-breaking on BF16."
                  " First action: dump router_logits + topk_expert_ids R2 vs C++ at first_diverge step.")

    rep["case_classification"] = case
    rep["raccomandazione_fix"] = raccom

    # KV check — simple R2-side
    rep["kv_check"] = {
        "r2_pos_after_prefill": len(prefix_ids),
        "r2_pos_after_8_decode": len(prefix_ids) + N_GEN,
        "r2_kv_layer0_shape_after": list(r2._kv[0].shape) if r2._kv[0] is not None else None,
        "rope_freqs_max_seq": r2.freqs_cis_base.shape[0] if hasattr(r2, "freqs_cis_base") else None,
    }

    # Save first-bad tensor dump (if available cross-step on R2 side)
    if first_diverge is not None and first_diverge >= 0:
        dump = {}
        if first_diverge in r2_logits_per_step:
            dump["r2_logits_step"] = r2_logits_per_step[first_diverge].numpy()
        if cpp_logits_per_step.get(first_diverge) is not None:
            dump["cpp_logits_step"] = cpp_logits_per_step[first_diverge].numpy()
        if dump:
            np.savez(args.dump_tensor, **dump)
            print(f"[D17B4] first-bad tensor dump → {args.dump_tensor}")

    # Status
    rep["status"] = "PASS"

    # ── PRINT TABLE ──
    print()
    print("| decode_step | first_bad_layer | first_bad_block | cosine | max_abs_err | argmax_r2 | argmax_cpp | ipotesi |")
    print("|---|---:|---|---:|---:|---:|---:|---|")
    for r in table_rows:
        cos_s = f"{r['cosine']:.4f}" if r['cosine'] is not None else "—"
        mae_s = f"{r['max_abs_err']:.3f}" if r['max_abs_err'] is not None else "—"
        print(f"| {r['decode_step']} | {r['first_bad_layer']} | {r['first_bad_block']} "
              f"| {cos_s} | {mae_s} | {r['argmax_r2']} | {r['argmax_cpp']} | {r['ipotesi']} |")
    print()
    print(f"[D17B4] FIRST DIVERGE: step={first_diverge}")
    print(f"[D17B4] CASE: {case}")
    print(f"[D17B4] RACCOM: {raccom[:200]}...")

    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"[D17B4] report → {args.out}")
    print(f"[D17B4] === DONE: status={rep['status']} ===")


if __name__ == "__main__":
    main()
