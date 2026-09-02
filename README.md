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

The implied compression ceiling on TinyLlama-1.1B is **1.8x**, not 10x — and
only 5.1x even at an error budget that destroys the model. Full write-up:
**[docs/PHASE0-FINDINGS.md](docs/PHASE0-FINDINGS.md)**.

That is a measurement on the *least* redundant plausible subject, so it does not
close the question — it sharpens it into one cheap experiment, described in
**[docs/ROADMAP.md](docs/ROADMAP.md)** §3: *does redundancy scale with model
size?*

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
| `shootingstar/ceiling.py` | roofline × atlas → implied bytes/token ceiling |
| `shootingstar/hw.py` | byte accounting and roofline bounds for any model shape |

Put your own text in `data/` and pass `--split workload`: redundancy is a
property of a model *on a distribution*, and measuring it on the wrong one is
how published pruning results fail to reproduce.
