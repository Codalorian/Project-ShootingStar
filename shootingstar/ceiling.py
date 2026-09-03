#!/usr/bin/env python3
"""
Phase 0 deliverable: the implied bytes-per-token compression ceiling.

Combines the measured roofline (hw.py) with the measured redundancy atlas
to answer the only question Phase 0 exists to answer:

    given how much redundancy this model actually has, on this hardware,
    how many bytes per token could we stop moving -- and is 10x in range?

Every lever is priced in BYTES NOT READ PER DECODE STEP, because that is what
bounds single-stream decoding on this machine (89% of its arithmetic sits idle
during decode; see results/roofline_raw.json).

READ THE CAVEAT AT THE BOTTOM OF THE OUTPUT. The per-lever numbers are
measured. Their PRODUCT is not: no experiment here shows that these compose.
Establishing the interaction matrix is Phase 1, and it is the actual research
contribution of this project.
"""
from __future__ import annotations

import argparse, json, sys
from dataclasses import dataclass
from pathlib import Path

from shootingstar.hw import Machine, ModelShape, decode_bound, KNOWN_SHAPES


@dataclass
class Lever:
    name: str
    factor: float          # bytes_after / bytes_before  (lower is better)
    basis: str             # what measurement it came from
    lossless: bool = False
    note: str = ""

    @property
    def speedup(self) -> float:
        return 1.0 / self.factor if self.factor > 0 else float("inf")


def _keep_frac_for_budget(trunc: dict, budget: float, key: str) -> float:
    """Smallest keep-fraction whose error stays within budget."""
    best = 1.0
    for frac_s, entry in sorted(trunc.items(), key=lambda kv: float(kv[0])):
        v = entry.get(key)
        if v is not None and v <= budget:
            return float(frac_s)
        best = float(frac_s)
    return 1.0


def build_levers(atlas: dict, shape: ModelShape, resid_budget: float,
                 depth_cos_threshold: float, spec_threshold: float,
                 draft_cost_frac: float) -> list[Lever]:
    levers: list[Lever] = []
    P = shape.params_breakdown()
    total = sum(P.values())
    mlp_share = sum(v for k, v in P.items() if k.startswith("mlp.")) / total
    qo_share = (P["attn.q_proj"] + P["attn.o_proj"]) / total

    # ---- 1. MLP contextual sparsity ----
    keeps = [_keep_frac_for_budget(L["truncation"], resid_budget,
                                   "rel_err_vs_resid_mean")
             for L in atlas["mlp_width"]]
    mk = sum(keeps) / len(keeps)
    levers.append(Lever(
        "mlp_contextual_sparsity", 1 - mlp_share * (1 - mk),
        f"exact truncation error <= {resid_budget:.0%} of residual norm; "
        f"mean keep-fraction {mk:.2f} of {atlas['mlp_width'][0]['n_neurons']} neurons",
        note="requires a predictor for WHICH neurons, and a gather-matmul kernel; "
             "neither is priced in here"))

    # ---- 2. attention head sparsity ----
    hkeeps = [_keep_frac_for_budget(L["truncation"], resid_budget,
                                    "rel_err_vs_resid_mean")
              for L in atlas["attn_heads"]]
    hk = sum(hkeeps) / len(hkeeps)
    # Under MHA every head owns its own k and v, so dropping a head drops all
    # four projections. Under GQA k/v are shared across a group and survive
    # unless the whole group goes, so only q and o are credited.
    mha = shape.n_kv_heads == shape.n_heads
    head_share = (P["attn.q_proj"] + P["attn.o_proj"]
                  + (P["attn.k_proj"] + P["attn.v_proj"] if mha else 0)) / total
    levers.append(Lever(
        "attn_head_sparsity", 1 - head_share * (1 - hk),
        f"exact truncation error <= {resid_budget:.0%} of residual norm; "
        f"mean keep-fraction {hk:.2f} of heads",
        note=("MHA: q+k+v+o all freed per dropped head"
              if mha else
              "GQA: q+o only; k/v are shared across a group and survive")))

    # ---- 3. depth ----
    drop = [L for L in atlas["depth"] if L["cos_in_out"] >= depth_cos_threshold]
    n = len(atlas["depth"])
    layer_share = 1 - P["lm_head"] / total
    levers.append(Lever(
        "layer_dropping", 1 - layer_share * (len(drop) / n),
        f"{len(drop)}/{n} layers with cos(in,out) >= {depth_cos_threshold}: "
        f"{[L['layer'] for L in drop]}",
        note="quality cost is measured independently in results/gate_selftest.json"))

    # ---- 4. speculative decoding (LOSSLESS) ----
    td = atlas["token_difficulty"]
    run = td["mean_easy_run"][str(spec_threshold)]
    accepted = 1.0 + run
    levers.append(Lever(
        "speculative_decoding", (1.0 / accepted) + draft_cost_frac,
        f"mean run of {run:.2f} consecutive tokens with top-1 prob > "
        f"{spec_threshold} (easy fraction {td['easy_fraction'][str(spec_threshold)]:.2f})",
        lossless=True,
        note="ORACLE UPPER BOUND: assumes the drafter guesses every 'easy' token. "
             f"A real drafter is worse. Draft cost charged at {draft_cost_frac:.0%}."))
    return levers


def report(machine: Machine, shape: ModelShape, atlas: dict, args) -> dict:
    base = decode_bound(machine, shape, bits=args.bits)
    levers = build_levers(atlas, shape, args.resid_budget, args.depth_cos,
                          args.spec_threshold, args.draft_cost)

    lines = []
    W = 78
    lines.append("=" * W)
    lines.append("SHOOTINGSTAR PHASE 0 -- IMPLIED COMPRESSION CEILING")
    lines.append("=" * W)
    lines.append(f"machine     : {machine.read_bw_gbps:.1f} GB/s read "
                 f"({machine.best_bw_threads} threads), "
                 f"{machine.int8_gops:.0f} GOP/s int8")
    lines.append(f"              machine balance {machine.balance_int8:.1f} ops/byte")
    lines.append(f"model       : {shape.name}, {shape.decode_params()/1e9:.2f}B "
                 f"decode params @ {args.bits:g}-bit")
    lines.append(f"atlas       : {atlas['meta']['model']} / {atlas['meta']['split']} / "
                 f"{atlas['meta']['n_tokens']} tokens")
    lines.append("")
    lines.append(f"BASELINE    {base.bytes_per_token/1e6:8.1f} MB/token  ->  "
                 f"{base.tok_s_ceiling:6.1f} tok/s roofline ceiling")
    lines.append(f"            arithmetic intensity {base.arithmetic_intensity:.1f} ops/byte "
                 f"vs balance {base.machine_balance:.1f}")
    lines.append(f"            => {base.idle_compute_frac*100:.0f}% of this machine's "
                 f"arithmetic is IDLE during decode")
    lines.append("")
    lines.append(f"LEVERS (error budget: {args.resid_budget:.0%} of residual norm)")
    lines.append("-" * W)
    lines.append(f"{'lever':<28}{'bytes x':>9}{'speedup':>9}   basis")
    for L in levers:
        tag = " [lossless]" if L.lossless else ""
        lines.append(f"{L.name:<28}{L.factor:>9.3f}{L.speedup:>9.2f}x  {L.basis[:60]}{tag}")
        for chunk in (L.note[i:i+64] for i in range(0, len(L.note), 64)):
            lines.append(f"{'':<48}{chunk}")
    lines.append("-" * W)

    prod = 1.0
    for L in levers:
        prod *= L.factor
    lossless_prod = 1.0
    for L in levers:
        if L.lossless:
            lossless_prod *= L.factor

    bytes_after = base.bytes_per_token * prod
    tok_s_naive = machine.read_bw_gbps * 1e9 / bytes_after
    # if the per-token working set collapses far enough it moves up a cache tier
    bw_ws = machine.bw_at_working_set(int(bytes_after))
    tok_s_cacheaware = bw_ws * 1e9 / bytes_after

    lines.append(f"{'LOSSLESS ONLY':<28}{lossless_prod:>9.3f}"
                 f"{1/lossless_prod:>9.2f}x  <- safe to bank")
    lines.append(f"{'NAIVE PRODUCT (all)':<28}{prod:>9.3f}{1/prod:>9.2f}x")
    lines.append("")
    lines.append(f"  bytes/token  {base.bytes_per_token/1e6:.1f} MB -> {bytes_after/1e6:.1f} MB")
    lines.append(f"  ceiling      {base.tok_s_ceiling:.1f} -> {tok_s_naive:.1f} tok/s "
                 f"(DRAM bandwidth)")
    if bw_ws > machine.read_bw_gbps * 1.05:
        lines.append(f"  cache-aware  -> {tok_s_cacheaware:.1f} tok/s if the working set "
                     f"({bytes_after/1e6:.1f} MB) stays resident")
        lines.append(f"               at {bw_ws:.0f} GB/s instead of "
                     f"{machine.read_bw_gbps:.0f} GB/s")
    if args.baseline_tok_s:
        lines.append("")
        lines.append(f"  measured baseline (llama.cpp): {args.baseline_tok_s:.2f} tok/s")
        lines.append(f"  10x target                   : {args.baseline_tok_s*10:.2f} tok/s")
        lines.append(f"  implied ceiling vs target    : "
                     f"{'REACHABLE' if tok_s_naive >= args.baseline_tok_s*10 else 'SHORT'} "
                     f"({tok_s_naive/(args.baseline_tok_s*10):.2f}x of target)")
    lines.append("")
    lines.append("!" * W)
    lines.append("CAVEAT. Each lever above is measured. Their PRODUCT is not.")
    lines.append("Nothing here shows these compose: contextual sparsity may not survive")
    lines.append("aggressive quantisation, and layer dropping and speculative decoding")
    lines.append("may be eating the same redundancy (both exploit 'this token was easy').")
    lines.append("Treat the product as an UPPER BOUND to be falsified in Phase 1.")
    lines.append("Also unpriced: every lever needs a kernel that actually skips the")
    lines.append("memory traffic. A technique with no kernel is a paper, not a speedup.")
    lines.append("!" * W)

    text = "\n".join(lines)
    return {
        "text": text,
        "baseline": base.__dict__,
        "levers": [L.__dict__ | {"speedup": L.speedup} for L in levers],
        "naive_product_factor": prod,
        "lossless_product_factor": lossless_prod,
        "bytes_per_token_after": bytes_after,
        "tok_s_ceiling_after": tok_s_naive,
        "tok_s_ceiling_after_cache_aware": tok_s_cacheaware,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roofline", default="results/roofline_raw.json")
    ap.add_argument("--atlas", default="results/atlas_tinyllama_heldout.json")
    ap.add_argument("--shape", default="tinyllama-1.1b", choices=list(KNOWN_SHAPES))
    ap.add_argument("--bits", type=float, default=4.0)
    ap.add_argument("--resid-budget", type=float, default=0.05,
                    help="allowed sublayer error as a fraction of residual norm")
    ap.add_argument("--depth-cos", type=float, default=0.95,
                    help="cos(in,out) at or above which a layer counts as near-identity")
    ap.add_argument("--spec-threshold", type=float, default=0.8)
    ap.add_argument("--draft-cost", type=float, default=0.10)
    ap.add_argument("--baseline-tok-s", type=float, default=None)
    ap.add_argument("--out", default="results/ceiling.json")
    a = ap.parse_args()

    machine = Machine.from_roofline_json(a.roofline)
    atlas = json.loads(Path(a.atlas).read_text())
    shape = KNOWN_SHAPES[a.shape]
    # Redundancy is a function of over-parameterisation, so it does NOT transfer
    # across model scales. Applying a 1B model's atlas to a 7B budget is the
    # single easiest way to manufacture a fake 10x.
    n_atlas = atlas["meta"]["config"]["num_hidden_layers"]
    if n_atlas != shape.n_layers:
        print(f"WARNING: atlas is from a {n_atlas}-layer model "
              f"({atlas['meta']['model']}) but --shape {a.shape} has "
              f"{shape.n_layers} layers.\n"
              f"         Redundancy does not transfer across scale. Run the "
              f"atlas on the target model.\n", file=sys.stderr)
    r = report(machine, shape, atlas, a)
    print(r["text"])
    Path(a.out).write_text(json.dumps(r, indent=2))
    Path(a.out).with_suffix(".txt").write_text(r["text"] + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
