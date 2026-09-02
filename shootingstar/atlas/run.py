#!/usr/bin/env python3
"""
Build the redundancy atlas for a model.

  python -m shootingstar.atlas.run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --n-seqs 32 --seq-len 512 --out results/atlas_tinyllama.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import torch

from shootingstar import corpus
from shootingstar.atlas.probes import Atlas


def build_batches(tok, docs, n_seqs, seq_len):
    """Pack the corpus into fixed-length sequences (no padding, no masking games)."""
    ids: list[int] = []
    for d in docs:
        ids += tok(d, add_special_tokens=False)["input_ids"]
        if len(ids) >= n_seqs * seq_len + 1:
            break
    if len(ids) < seq_len + 1:
        raise SystemExit(f"corpus too small: {len(ids)} tokens < {seq_len + 1}")
    n = min(n_seqs, len(ids) // seq_len)
    t = torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--split", default="held_out", choices=["held_out", "workload"])
    ap.add_argument("--n-seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--cap", type=int, default=192,
                    help="token-vectors retained per layer for exact truncation curves")
    ap.add_argument("--threads", type=int, default=4,
                    help="4 beats 8 here: SMT oversubscription costs memory bandwidth")
    ap.add_argument("--no-rank", action="store_true", help="skip the Gram/SVD probe")
    ap.add_argument("--out", default="results/atlas.json")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    print(f"[atlas] loading {a.model} (fp32, {a.threads} threads)", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model.eval()
    print(f"[atlas] loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    docs = corpus.get(a.split, n_docs=4096)
    batches = build_batches(tok, docs, a.n_seqs, a.seq_len)
    n_tokens = batches.numel()
    print(f"[atlas] {batches.shape[0]} seqs x {a.seq_len} tok = {n_tokens} tokens "
          f"from '{a.split}'", file=sys.stderr)

    atlas = Atlas(model, cap=a.cap, rank=not a.no_rank)
    t0 = time.time()
    try:
        for bi in range(0, batches.shape[0], a.batch_size):
            chunk = batches[bi: bi + a.batch_size]
            out = model(chunk)
            for row in range(chunk.shape[0]):
                # difficulty of predicting token t+1 given prefix -- drop last
                atlas.tok.observe(out.logits[row, :-1, :])
            done = min(bi + a.batch_size, batches.shape[0])
            el = time.time() - t0
            print(f"[atlas] {done}/{batches.shape[0]} seqs  "
                  f"{el:6.1f}s  ({done*a.seq_len/el:.0f} tok/s prefill)", file=sys.stderr)
    finally:
        atlas.remove()

    print("[atlas] computing truncation curves and spectra...", file=sys.stderr)
    res = atlas.result()
    res["meta"] = {
        "model": a.model, "split": a.split, "n_seqs": int(batches.shape[0]),
        "seq_len": a.seq_len, "n_tokens": int(n_tokens), "cap": a.cap,
        "threads": a.threads, "dtype": "float32",
        "config": {k: getattr(model.config, k, None) for k in
                   ("num_hidden_layers", "hidden_size", "intermediate_size",
                    "num_attention_heads", "num_key_value_heads", "vocab_size",
                    "head_dim")},
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[atlas] wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
