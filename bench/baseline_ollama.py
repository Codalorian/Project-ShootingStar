#!/usr/bin/env python3
"""
The number ShootingStar has to beat by 10x.

Measures decode and prefill throughput of the local ollama (llama.cpp) stack.
This is a *strong* baseline -- it already includes 4-bit quantisation, tuned
AVX kernels and a real KV cache. Beating fp16 HF transformers is not a result;
beating this is.

Usage: python bench/baseline_ollama.py [--model NAME] [--repeats N]
"""
from __future__ import annotations

import argparse, json, statistics, sys, time, urllib.request

HOST = "http://localhost:11434"

PROMPTS = [
    ("short", "Explain in one paragraph why memory bandwidth limits language model inference."),
    ("medium", "Write a detailed technical comparison of speculative decoding and "
               "contextual sparsity as strategies for accelerating transformer inference "
               "on CPUs. Cover the mechanism, the expected speedup, and the failure modes."),
]


def _post(path: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def list_models() -> list[str]:
    with urllib.request.urlopen(HOST + "/api/tags", timeout=10) as r:
        return [m["name"] for m in json.loads(r.read())["models"]]


def run_one(model: str, prompt: str, n_predict: int) -> dict:
    """ollama reports its own nanosecond timings; trust those over wall clock."""
    r = _post("/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": n_predict, "temperature": 0.0, "seed": 0},
    })
    ev, evd = r.get("eval_count", 0), r.get("eval_duration", 0)
    pv, pvd = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0)
    return {
        "decode_tok_s": ev / (evd / 1e9) if evd else float("nan"),
        "prefill_tok_s": pv / (pvd / 1e9) if pvd else float("nan"),
        "eval_count": ev,
        "prompt_tokens": pv,
        "load_ms": r.get("load_duration", 0) / 1e6,
        "total_s": r.get("total_duration", 0) / 1e9,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--out", default="results/baseline_ollama.json")
    a = ap.parse_args()

    try:
        models = list_models()
    except Exception as e:
        print(f"ollama not reachable at {HOST}: {e}", file=sys.stderr)
        return 1
    targets = [a.model] if a.model else models
    print(f"models: {models}", file=sys.stderr)

    out = {"host": HOST, "n_predict": a.n_predict, "repeats": a.repeats, "runs": {}}
    for model in targets:
        out["runs"][model] = {}
        for label, prompt in PROMPTS:
            samples = []
            for i in range(a.repeats):
                try:
                    s = run_one(model, prompt, a.n_predict)
                except Exception as e:
                    print(f"  {model}/{label} failed: {e}", file=sys.stderr)
                    break
                samples.append(s)
                print(f"  {model:22s} {label:7s} rep{i}: "
                      f"decode {s['decode_tok_s']:6.2f} tok/s  "
                      f"prefill {s['prefill_tok_s']:7.2f} tok/s  "
                      f"({s['eval_count']} tok)", file=sys.stderr)
            if not samples:
                continue
            out["runs"][model][label] = {
                "decode_tok_s_median": statistics.median(s["decode_tok_s"] for s in samples),
                "decode_tok_s_max": max(s["decode_tok_s"] for s in samples),
                "prefill_tok_s_median": statistics.median(s["prefill_tok_s"] for s in samples),
                "prompt_tokens": samples[0]["prompt_tokens"],
                "samples": samples,
            }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
