# Phase 0 findings

**Date:** 2026-09-01 · **Reference device:** Intel i7-1165G7 (4c/8t, AVX-512+VNNI),
31 GB LPDDR4x, no discrete GPU · **Subject:** TinyLlama-1.1B-Chat-v1.0, fp32,
16 384 tokens of wikitext-2 test.

Reproduce with `scripts/phase0.sh`. Raw records in `results/`.

---

## 1. The bottleneck is bytes, not FLOPs

Generating one token at batch size 1 reads every weight exactly once and does
~2 ops per weight. At 4-bit that fixes arithmetic intensity at **4.0 ops/byte**,
independent of model size.

Measured on this machine (`results/roofline_raw.json`):

| quantity | measured |
|---|---|
| DRAM sequential read, 4 threads | **33.4 GB/s** |
| DRAM sequential read, 8 threads | 27.7 GB/s (SMT *hurts*) |
| L2-resident read (1–4 MB) | 319–357 GB/s |
| L3 boundary (12 MB) | 132 GB/s |
| fp32 FMA peak | 298 GFLOP/s |
| int8 VNNI peak | **1211 GOP/s** |
| **machine balance** | **36.3 ops/byte** |

Decode runs at 4.0 ops/byte against a balance of 36.3, so **89% of this
machine's arithmetic sits idle during decoding.** Cutting FLOPs buys nothing;
cutting bytes-read-per-token buys everything.

The same fact, measured end-to-end on llama.cpp with llama2-7B-Q4:

| phase | throughput |
|---|---|
| prefill (compute-bound, batched) | 350 tok/s |
| decode (bandwidth-bound, bs=1) | **5.5 tok/s** |

A **64x gap on identical weights.** That gap is the project's entire
opportunity, and it is not made of FLOPs.

### Two structural findings

- **Use 4 threads, not 8.** Read bandwidth peaks at 4 and *drops* 17% at 8;
  SMT siblings contend for the same load ports and memory queue.
- **The cache hierarchy already contains a 10x.** A working set resident in
  L2/L3 streams at 319–357 GB/s versus 33 GB/s from DRAM — a **10.7x** ratio.
  The cheapest route to a 10x is not a smaller model, it is a per-token
  *working set* small enough to stay resident.

## 2. The honest baseline

**llama2-7B Q4_0 via ollama/llama.cpp: 5.5 tok/s warm decode** (128 tok,
median of 2 warm runs; cold runs 2.8–4.1 while the 3.8 GB file pages in).

Roofline ceiling for that model is 8.8 tok/s, so llama.cpp already achieves
**63% of the memory roofline**. There is no free systems overhead to reclaim —
only ~1.6x — which means any 10x must come from moving fewer bytes.

**The target is 55 tok/s on a 7B.** Beating fp16 HF `transformers` is not a
result; this is the number that counts.

## 3. Measured redundancy

All truncation numbers are **exact reconstruction errors** — the sublayer output
rebuilt from its top-k components, measured against the true output, expressed
as a fraction of the residual-stream norm (what actually propagates). Proxies
such as activation magnitude overstate prunability by ignoring cancellation.

### Depth — essentially none

| | |
|---|---|
| max cos(layer input, layer output) | **0.897** (layer 17) |
| layers with cos ≥ 0.95 | **0 of 22** |

The literature's droppable layers (ShortGPT, Gromov et al.) show cos > 0.99 in
7B–70B models. **TinyLlama has no near-identity layers at all.** Every layer is
doing work.

### MLP contextual sparsity — real but small

| error budget (of residual norm) | neurons that must be read |
|---|---|
| 2% | 79% |
| 5% | **62%** |
| 10% | 39% |

Raw activation sparsity (|h| below 1% of max) is 31–63%, but the neurons below
that threshold still carry output energy. The "80–95% contextual sparsity"
result comes from ReLU-family models; **SwiGLU models are far denser.**

### Attention heads — dense

Keeping 65–90% of heads is needed for 5% error. The top 4 of 32 heads hold only
19–32% of head energy: contribution is spread, not concentrated.

### Residual stream — not low-rank

Centred covariance needs **546–1785 of 2048 dimensions** for 99% of energy.
Middle layers show participation ratio ≈ 1.1 with one direction holding 95% of
variance, but that is the known massive-activation outlier feature, not
compressible structure. *(The uncentred spectrum reports "rank 1" and is
misleading — it is finding the DC offset. Both are recorded.)*

### Token difficulty — modest speculative headroom

| top-1 prob threshold | easy fraction | mean consecutive run |
|---|---|---|
| > 0.5 | 0.50 | 0.98 |
| > 0.8 | 0.29 | **0.41** |
| > 0.95 | 0.18 | 0.21 |

Mean top-1 probability 0.542, entropy 2.15 nats. Even an *oracle* drafter
averages 1.41 tokens per verification pass — **~1.4x, not the 3x reported on
GPUs.** Caveat: wikitext is high-entropy prose. Chat and code are far more
predictable, and this lever is the one most likely to improve on a real
workload — which is why `data/` exists.

## 4. The quality gate works

`results/gate_selftest.json`. Identity check passes (KL = -7.8e-7).

**The atlas's redundancy ranking predicts damage.** Bypassing one layer:

| arm | layer | KL(ref‖mod) |
|---|---|---|
| most redundant | 17 | **0.315** |
| least redundant | 0 | **4.308** |

A 14x difference — the angular-distance metric is validated as a predictor.
But even the *best* single layer to drop (4.5% of the model) costs KL 0.315
and moves 22% of argmaxes. There is no free layer.

**Rank statistics under-report damage**, as argued in `shootingstar/gate/kl.py`:

| layers dropped | KL | top-1 agreement | ground-truth accuracy | perplexity |
|---|---|---|---|---|
| 1 | 0.315 | 0.780 | 0.530 → 0.489 (−8%) | ×1.26 |
| 4 | 0.944 | 0.580 | 0.530 → 0.405 (−24%) | ×2.26 |
| 6 | 1.628 | 0.434 | 0.530 → 0.301 (−43%) | ×4.65 |

At k=1, **22% of tokens changed their argmax while ground-truth accuracy fell
only 8% relative.** A multiple-choice benchmark — the same kind of rank
statistic, over 4 options instead of 32 000 — would have reported almost
nothing. And the generations stay *fluent* at k=6 with 27% of layers gone:

> k=0: "that they are trained on a large dataset, which requires a lot of computational resources…"
> k=6: "that the training of the model is not optimized for the task. This means that the model is not trained to solve the problem."

Eyeballing output would have passed this model. KL says its distribution has
been rearranged beyond recognition. **Gate on KL.**

## 5. Verdict

Implied ceiling from the naive product of all measured levers
(`results/ceiling.txt`):

| error budget | implied speedup |
|---|---|
| 2% | 1.47x |
| **5%** | **1.80x** |
| 10% | 2.48x |
| 20% | 3.74x |
| 35% (model destroyed) | **5.10x** |

**10x is not reachable on TinyLlama-1.1B at any error budget, by any
combination of these levers.** And 1.80x is an upper bound twice over: the
levers are assumed to compose (untested), and every one is assumed to have a
kernel that actually skips the memory traffic (none is written).

Banked lossless component: **1.24x** (speculative decoding, oracle bound).

### What this does and does not say

It does **not** say the moonshot is dead. It says the experiment was run on the
worst plausible subject. TinyLlama-1.1B is trained on 3T tokens for 1.1B
parameters — extraordinarily over-*trained* and therefore minimally
over-*parameterised*. Redundancy is a consequence of over-parameterisation, so
the model with the least of it is exactly the one we measured.

**The load-bearing question is now sharp and cheap to answer: does redundancy
scale with model size?** Every number above is a point at 1.1B. If the same
atlas on a 7B and a 13B shows the curves opening up — cos(in,out) climbing
toward 0.99, MLP keep-fraction falling toward 0.2 — the thesis survives and
the roadmap is a systems project. If the curves are flat in scale, the
redundancy is not there and ShootingStar should become a different project
(see `ROADMAP.md` §4).

One measurement decides it. That is what Phase 0 bought.
