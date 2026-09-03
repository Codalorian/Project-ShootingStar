# Project-ShootingStar

Aiming to improve LLM speed on consumer hardware by 90% while retaining 95% intelligence.

## What this is

An attempt to answer, with measurements rather than intuition: **do modern LLMs
carry enough redundancy that a consumer machine could run one on a tenth of the
resources?**

The framing that makes the question tractable: at batch size 1, decoding is
bound by **bytes moved per token**, not FLOPs. On the reference device, 89% of
the machine's arithmetic sits idle waiting on memory. So the question is not
"which computations are redundant" but **"which bytes did we not need to read?"**

## Status: Phase 0 complete

Measured across three models, the implied compression ceiling is **1.43x → 1.80x →
2.77x** (Llama-3.2-1B, TinyLlama-1.1B, Llama-2-7B) — not 10x. Redundancy is real and
it **does** scale with model size, but not steeply enough, and it is largest exactly
where the machine can least afford to run the model. Full write-up:
**[docs/PHASE0-FINDINGS.md](docs/PHASE0-FINDINGS.md)**.

The load-bearing question — *does redundancy scale with size?* — is now answered, and
the follow-on work is set out in **[docs/ROADMAP.md](docs/ROADMAP.md)**.

## Run it

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
./scripts/phase0.sh          # ~25 min on a 4-core laptop
```

| component | what it measures |
|---|---|
| `bench/roofline.c` | memory bandwidth, cache tiers, fp32/int8 peaks |
| `bench/baseline_ollama.py` | the llama.cpp number we have to beat by 10x |
| `shootingstar/atlas/` | depth, MLP width, head, rank and token-difficulty redundancy |
| `shootingstar/gate/` | the KL quality gate, and a self-test that validates it |
| `shootingstar/compare.py` | lines up N atlases — the scaling-law instrument |
| `shootingstar/ceiling.py` | roofline × atlas → implied bytes/token ceiling |
| `shootingstar/hw.py` | byte accounting and roofline bounds for any model shape |

Put your own text in `data/` and pass `--split workload`: redundancy is a
property of a model *on a distribution*, and measuring it on the wrong one is
how published pruning results fail to reproduce.
