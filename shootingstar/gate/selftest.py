#!/usr/bin/env python3
"""
Validate the quality gate, and get the first real depth-redundancy datapoint.

Bypasses the k layers the atlas rates most redundant (lowest angular distance
between layer input and output) and scores the damage three ways. A control
arm bypasses the k LEAST redundant layers.

Two things this establishes:
  1. the atlas's redundancy ranking predicts damage (redundant arm << control)
  2. KL detects damage that rank statistics (accuracy, and by extension
     multiple-choice benchmarks) report as almost nothing

  python -m shootingstar.gate.selftest --atlas results/atlas_tinyllama_heldout.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

from shootingstar import corpus
from shootingstar.atlas.run import build_batches
from shootingstar.gate.kl import (ReferenceLogits, bypass_layers, restore_layers,
                                  detect_layer_output_shape, evaluate,
                                  sample_generation)

PROBE_PROMPT = "The main reason large language models are slow on laptops is"


def rank_layers(atlas_path: str) -> tuple[list[int], list[int]]:
    d = json.loads(Path(atlas_path).read_text())["depth"]
    order = sorted(d, key=lambda x: x["angular_distance"])
    most = [x["layer"] for x in order]          # most redundant first
    return most, most[::-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--atlas", default="results/atlas_tinyllama_heldout.json")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-drop", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/gate_selftest.json")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    n_layers = len(model.model.layers)

    batches = build_batches(tok, corpus.get("held_out", n_docs=2048),
                            a.n_seqs, a.seq_len)
    detect_layer_output_shape(model, batches)
    print(f"[gate] {n_layers} layers, layer-returns-tuple="
          f"{model._ss_tuple_out}, {batches.numel()} eval tokens", file=sys.stderr)

    t0 = time.time()
    ref = ReferenceLogits(model, batches, a.batch_size)
    print(f"[gate] reference logits cached in {time.time()-t0:.1f}s", file=sys.stderr)

    base = evaluate(model, ref, batches, a.batch_size)
    assert base.kl_mean < 1e-6, f"identity check failed: KL={base.kl_mean}"
    print(f"[gate] identity check OK (KL={base.kl_mean:.2e}, "
          f"acc={base.acc_ref:.3f}, ppl={base.ppl_ref:.2f})", file=sys.stderr)

    most, least = rank_layers(a.atlas)
    print(f"[gate] most-redundant order: {most[:8]}", file=sys.stderr)

    out = {"model": a.model, "n_layers": n_layers,
           "eval_tokens": int(batches.numel()),
           "baseline": base.dict(),
           "baseline_generation": sample_generation(model, tok, PROBE_PROMPT),
           "arms": {}}

    for arm, order in (("redundant_first", most), ("control_least_redundant", least)):
        out["arms"][arm] = []
        for k in range(1, a.max_drop + 1):
            dropped = sorted(order[:k])
            saved = bypass_layers(model, dropped)
            try:
                r = evaluate(model, ref, batches, a.batch_size)
                gen = sample_generation(model, tok, PROBE_PROMPT) \
                    if arm == "redundant_first" else None
            finally:
                restore_layers(model, saved)
            rec = {"k": k, "dropped": dropped, "frac_layers": k / n_layers, **r.dict()}
            if gen is not None:
                rec["generation"] = gen
            out["arms"][arm].append(rec)
            print(f"[gate] {arm:24s} k={k} drop={dropped} "
                  f"KL={r.kl_mean:7.4f}  top1agree={r.top1_agreement:.3f}  "
                  f"acc {r.acc_ref:.3f}->{r.acc_mod:.3f}  "
                  f"ppl x{r.ppl_ratio:.2f}", file=sys.stderr)

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[gate] wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
