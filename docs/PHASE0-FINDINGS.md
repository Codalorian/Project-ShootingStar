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

## 5. Scaling: does redundancy grow with model size?

**Yes, monotonically — and the mechanism is depth.** Three models, identical corpus,
identical 16 384 tokens, identical settings. Reproduce with
`python -m shootingstar.compare results/atlas_*.json`.

| signal | Llama-3.2-1B | TinyLlama-1.1B | **Llama-2-7B** | thesis needs |
|---|---|---|---|---|
| layers × hidden | 16 × 2048 | 22 × 2048 | 32 × 4096 | |
| max cos(layer in, out) | 0.880 | 0.897 | **0.963** | 0.99+ |
| layers with cos ≥ 0.95 | 0 of 16 | 0 of 22 | **7 of 32** | several |
| MLP neurons needed @ 5% | 72% | 62% | **53%** | 20–30% |
| attention heads needed @ 5% | 89% | 75% | **57%** | ~30% |
| mean easy-run, p > 0.8 | 0.30 | 0.41 | **0.49** | 2+ |
| **implied ceiling @ 5%** | **1.43x** | **1.80x** | **2.77x** | 10x |

Every signal moves the right way with scale. Two findings carry the result:

**Depth redundancy appears exactly where predicted.** The seven near-identity layers
are 23–29 of 32 — **72–91% of the way down** — the deeper-middle band the layer-pruning
literature reports, and a band that does not exist in either 1B model. The per-layer
curve rises monotonically from 0.839 at layer 2 to 0.963 at layer 27, then collapses to
0.694 at the final layer.

**GQA is redundancy already harvested.** Head keep-fraction is 57% here against 75% and
89% in the two 1B models — and those two use GQA, the architectural change that removes
head redundancy by construction. Llama-2-7B predates it at this scale, so the redundancy
is still present *and* worth double the bytes: under MHA a dropped head frees q, k, v and
o, where under GQA only q and o can go (`shootingstar/ceiling.py` accounts for both).

**The mechanism is over-parameterisation, and tokens-per-parameter tracks it:**

| model | tokens/param | vs Chinchilla (~20) | ceiling |
|---|---|---|---|
| Llama-3.2-1B | ~7,300 | 365x past | 1.43x |
| TinyLlama-1.1B | ~2,700 | 135x past | 1.80x |
| Llama-2-7B | ~300 | 15x past | 2.77x |

Redundancy is what over-training squeezes out. The most heavily over-trained model has
the least of it.

*Caveat: the 7B run used `--no-rank` and `--dtype bfloat16` to fit in 31 GB, so it has no
residual-dimension figure. Every other signal is directly comparable.*

## 6. Verdict

| error budget | Llama-3.2-1B | TinyLlama-1.1B | Llama-2-7B |
|---|---|---|---|
| 2% | 1.31x | 1.47x | 2.12x |
| **5%** | **1.43x** | **1.80x** | **2.77x** |
| 10% | 1.68x | 2.48x | 3.82x |
| 20% | 2.17x | 3.74x | 5.50x |

Applied to the measured baseline: **5.5 → ~15 tok/s** on llama2-7B, with a roofline
ceiling moving 10.1 → 28.0 tok/s. That crosses from "slower than you can read" to
comfortable. It is not 10x.

And 2.77x remains an upper bound twice over: the levers are assumed to compose, which no
experiment here has tested, and each is assumed to have a kernel that actually skips the
memory traffic, which none has. Banked lossless component: **1.30x** (speculative
decoding, oracle bound).

### What this settles

**Redundancy is real and it scales — but not steeply enough, and not where you need it.**
1.43 → 1.80 → 2.77 across 6x of parameters. Extrapolating that slope puts a 70B at
perhaps 4–5x, on hardware that cannot hold a 70B. *The redundancy is largest exactly
where the machine can least afford to run the model.*

The original goal — 10x on consumer hardware by exploiting redundancy — is not reachable
by these levers on models that fit on this laptop.

### What is worth more

The **10.7x still sitting in the cache hierarchy** (§1): L2/L3 at 357 GB/s against DRAM
at 33 GB/s, requiring no redundancy at all, unexploited by llama.cpp. Combined with the
1.6x of implementation slack and lossless speculative decoding, a redundancy-free path
plausibly reaches 15–22 tok/s — comparable to the redundancy path, with none of its
quality cost.

### What is publishable

A **redundancy scaling law**, measured across three models and 6x of parameters, with a
mechanism (depth relative to task difficulty, plus un-harvested architecture) and a
predictor (tokens per parameter). We have not found this published. It is a result
independent of whether ShootingStar's 10x goal survives — and it did not.
