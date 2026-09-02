#!/usr/bin/env python3
"""
Line up two or more redundancy atlases.

This is the instrument for ROADMAP.md §3: does redundancy scale with model
size? Feed it atlases from models of increasing size and read the four signals
across the row. Also serves as a control comparison between same-size models
that were trained differently.

  python -m shootingstar.compare results/atlas_*.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path


def keep_frac(trunc: dict, budget: float) -> float:
    """Smallest fraction of components that stays within the error budget."""
    for f, e in sorted(trunc.items(), key=lambda kv: float(kv[0])):
        v = e.get("rel_err_vs_resid_mean")
        if v is not None and v <= budget:
            return float(f)
    return 1.0


def signals(a: dict, budget: float) -> dict:
    cfg = a["meta"]["config"]
    depth = a["depth"]
    cos = [L["cos_in_out"] for L in depth]
    mlp = [keep_frac(L["truncation"], budget) for L in a["mlp_width"]]
    att = [keep_frac(L["truncation"], budget) for L in a["attn_heads"]]
    td = a["token_difficulty"]
    rr = a.get("residual_rank") or []
    return {
        "model": a["meta"]["model"].split("/")[-1],
        "split": a["meta"]["split"],
        "tokens": a["meta"]["n_tokens"],
        "layers": cfg["num_hidden_layers"],
        "hidden": cfg["hidden_size"],
        "n_droppable": sum(c >= 0.95 for c in cos),
        "cos_max": max(cos),
        "cos_mean": sum(cos) / len(cos),
        "mlp_keep": sum(mlp) / len(mlp),
        "mlp_keep_min": min(mlp),
        "attn_keep": sum(att) / len(att),
        "easy_run": td["mean_easy_run"]["0.8"],
        "easy_frac": td["easy_fraction"]["0.8"],
        "top1_mean": td["top1_prob_mean"],
        "d99_frac": (sum(L["centred"]["dims_for_99pct"] / L["dim"] for L in rr) / len(rr)) if rr else float("nan"),
    }


ROWS = [
    ("layers x hidden",            lambda s: f"{s['layers']} x {s['hidden']}",      None),
    ("eval tokens",                lambda s: f"{s['tokens']:,}",                    None),
    ("",                           lambda s: "",                                    None),
    ("max cos(in, out)",           lambda s: f"{s['cos_max']:.3f}",  "-> 0.99+ if redundancy scales"),
    ("mean cos(in, out)",          lambda s: f"{s['cos_mean']:.3f}", None),
    ("layers with cos >= 0.95",    lambda s: f"{s['n_droppable']} of {s['layers']}", "-> several"),
    ("",                           lambda s: "",                                    None),
    ("MLP neurons needed",         lambda s: f"{s['mlp_keep']:.0%}", "-> 20-30%"),
    ("  best layer",               lambda s: f"{s['mlp_keep_min']:.0%}",            None),
    ("attn heads needed",          lambda s: f"{s['attn_keep']:.0%}", "-> ~30%"),
    ("residual dims for 99%",      lambda s: f"{s['d99_frac']:.0%}", None),
    ("",                           lambda s: "",                                    None),
    ("mean easy-run (p>0.8)",      lambda s: f"{s['easy_run']:.2f}", "-> 2+"),
    ("easy fraction (p>0.8)",      lambda s: f"{s['easy_frac']:.2f}",               None),
    ("mean top-1 probability",     lambda s: f"{s['top1_mean']:.3f}",               None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("atlases", nargs="+")
    ap.add_argument("--budget", type=float, default=0.05,
                    help="error budget as a fraction of residual norm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    S = [signals(json.loads(Path(p).read_text()), a.budget) for p in a.atlases]
    S.sort(key=lambda s: (s["layers"] * s["hidden"]))

    w = max(24, max(len(s["model"]) for s in S) + 2)
    W = 26 + w * len(S) + 32
    out = ["=" * W,
           f"REDUNDANCY ATLAS COMPARISON   (error budget {a.budget:.0%} of residual norm)",
           "=" * W,
           f"{'signal':<26}" + "".join(f"{s['model']:<{w}}" for s in S) + "thesis needs"]
    out.append("-" * W)
    for label, fn, target in ROWS:
        if not label:
            out.append("")
            continue
        out.append(f"{label:<26}" + "".join(f"{fn(s):<{w}}" for s in S)
                   + (target or ""))
    out.append("-" * W)
    if len({s["split"] for s in S}) > 1:
        out.append("WARNING: atlases use different corpora -- not comparable.")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
