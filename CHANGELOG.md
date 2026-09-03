# Changelog

## v0.1.0 — 2026-09-02

First release. Phase 0 (instruments) and Phase 1 (the scaling experiment) are complete.

### The question

Do modern LLMs carry enough redundancy that a consumer machine could run one on a tenth
of the resources?

### The answer

**No — but redundancy is real, it scales with model size, and it is predictable.**

| | Llama-3.2-1B | TinyLlama-1.1B | Llama-2-7B |
|---|---|---|---|
| layers with cos(in,out) ≥ 0.95 | 0 of 16 | 0 of 22 | **7 of 32** |
| MLP neurons needed @ 5% error | 72% | 62% | **53%** |
| attention heads needed @ 5% | 89% | 75% | **57%** |
| implied ceiling @ 5% | 1.43x | 1.80x | **2.77x** |

Against the measured llama.cpp baseline that is **5.5 → ~15 tok/s** on a 7B: it crosses
from slower than reading speed to comfortable, and stops. Extrapolating the slope puts a
70B at 4–5x, on hardware that cannot hold a 70B. The redundancy is largest exactly where
the machine can least afford to run the model.

### Findings

- **The bottleneck is bytes, not FLOPs.** At batch size 1, decode runs at 4.0 ops/byte
  against a measured machine balance of 36.3 — **89% of the machine's arithmetic is idle**
  waiting on memory. Measured end-to-end: 350 tok/s prefill vs 5.5 tok/s decode on
  identical weights, a 64x gap.
- **A redundancy scaling law**, measured across three models and 6x of parameters, with a
  mechanism and a predictor. Not found in the literature.
- **Depth redundancy lives at 72–91% depth** (layers 23–29 of 32 in Llama-2-7B) and does
  not exist at all in either 1B model.
- **GQA is redundancy already harvested.** The 1B models need 75–89% of their attention
  heads because their architecture pre-spent the lever; Llama-2's older MHA still has it,
  and it is worth double the bytes there.
- **Training tokens per parameter predicts the ordering** (7300 / 2700 / 300). Redundancy
  is what over-training squeezes out.
- **Rank statistics under-report damage.** Dropping one layer moved 22% of argmaxes while
  ground-truth accuracy fell 8%, and generations stayed fluent with 27% of layers removed.
  Gate on KL divergence, not accuracy or eyeballed output.

### Tools

- `bench/roofline.c` — measured memory bandwidth, cache tiers, fp32/int8 peaks
- `bench/baseline_ollama.py` — the llama.cpp number to beat
- `shootingstar/atlas/` — five redundancy probes using exact reconstruction error
- `shootingstar/gate/` — KL quality gate plus a self-test that validates it
- `shootingstar/ceiling.py` — roofline × atlas → implied bytes/token ceiling
- `shootingstar/compare.py` — lines up N atlases; the scaling-law instrument
- `shootingstar/hw.py` — byte accounting and roofline bounds for any model shape

### Corrections made before release

- **Cache residency is not a harvestable 10.7x.** Earlier drafts presented the measured
  L2/L3-vs-DRAM ratio as the largest unexploited lever. At this model size it is not: L3
  is 12 MB and a 7B at 4-bit is 3.8 GB, ~300x too large, and at batch size 1 there is no
  reuse to exploit. The exploitable form of the same physics is reuse, not residency —
  speculative decoding.
- **Head-sparsity accounting assumed GQA.** Under MHA a dropped head also frees k and v.
  This under-counted Llama-2-7B; corrected, its ceiling moves 2.56x → 2.77x.

### Known limits

- Levers are measured individually; that they **compose** is untested. Treat the product
  as an upper bound to be falsified.
- No lever has a kernel that actually skips the memory traffic. A technique without a
  kernel is a paper, not a speedup.
- All measurements are on wikitext. Redundancy is a property of a model *on a
  distribution* — use `--split workload` on your own traffic.
- The 7B run used `--no-rank --dtype bfloat16` to fit in 31 GB, so it has no
  residual-dimension figure.
