# ShootingStar roadmap

> Aiming to improve LLM speed on consumer hardware by 90% while retaining 95%
> intelligence.

## 0. The reframing that makes the goal measurable

"90% less compute" is the wrong target. At batch size 1, decoding reads every
weight once per token and does ~2 ops per weight — **4.0 ops/byte at 4-bit**,
against a measured machine balance of **36.3 ops/byte** on the reference
device. 89% of the machine's arithmetic is already idle. Removing FLOPs removes
idle time.

**The metric is bytes moved per generated token.** Every experiment in this
repo is priced in bytes. FLOP counts and "% of parameters pruned" are vanity
metrics and are not accepted as evidence.

## 1. Definitions, fixed before we have a stake in the answer

**Baseline.** ollama/llama.cpp Q4_K_M on the reference device, measured:
**5.5 tok/s** on llama2-7B. Not fp16 `transformers` — beating that is free and
proves nothing. **Target: 55 tok/s on a 7B.**

**"95% intelligence."** Primary gate is **mean KL(P_reference ‖ P_modified) per
token ≤ 0.05 nats** with **top-1 agreement ≥ 0.95**, on held-out text *and* on
workload text. Rationale and evidence in `shootingstar/gate/kl.py` and
`docs/PHASE0-FINDINGS.md` §4: rank statistics (accuracy, multiple-choice
benchmarks) miss damage that has visibly rearranged the distribution, and
fluent-looking generations survive damage that KL catches immediately.

**Reference device.** i7-1165G7, 4c/8t AVX-512+VNNI, 31 GB LPDDR4x, no dGPU.
Use **4 threads**: read bandwidth peaks there and drops 17% at 8.

## 2. Phase 0 — measure before optimising ✅ complete

Deliverable was one number: the implied bytes-per-token ceiling.
**Answer: 1.80x at a 5% error budget, 5.10x at a model-destroying 35%.**
Full result in `docs/PHASE0-FINDINGS.md`. Built:

- `bench/roofline.c` — measured bandwidth, cache tiers, fp32/int8 peaks
- `bench/baseline_ollama.py` — the number to beat
- `shootingstar/atlas/` — five redundancy probes, exact reconstruction errors
- `shootingstar/gate/` — the KL quality gate, plus a self-test that validates it
- `shootingstar/ceiling.py` — roofline × atlas → implied ceiling

## 3. Phase 1 — the deciding question, answered ✅

**Does redundancy scale with model size? Yes, monotonically — but not steeply enough.**
Measured on three models with an identical corpus and token budget:

| signal | Llama-3.2-1B | TinyLlama-1.1B | Llama-2-7B |
|---|---|---|---|
| layers with cos ≥ 0.95 | 0 of 16 | 0 of 22 | **7 of 32** (layers 23–29) |
| MLP neurons needed @ 5% | 72% | 62% | **53%** |
| attention heads needed @ 5% | 89% | 75% | **57%** |
| implied ceiling @ 5% | 1.43x | 1.80x | **2.77x** |

Predicted before the run and scored after: max cos 0.95–0.98 (got 0.963), 3–6 droppable
layers (got 7), MLP keep 45–60% (got 53%), head keep 50–70% (got 57%), ceiling ~2.8x
(got 2.77x). The one miss was speculative easy-run, predicted 0.6–0.9, measured 0.49.

Full analysis in `docs/PHASE0-FINDINGS.md` §5. **The 10x goal is not reachable by these
levers on models that fit on consumer hardware**, and the honest extrapolation puts even
a 70B at 4–5x.

## 4. What the project should do now

In priority order, by measured size and by how little each depends on unproven claims:

1. **Cache residency — 10.7x, measured, unexploited.** L2/L3 at 357 GB/s against DRAM at
   33 GB/s on the reference device. Needs no redundancy, no quality budget, and no
   retraining. llama.cpp does not exploit it. This is now the largest single lever the
   project has found and it is a data-layout and scheduling problem.
2. **Speculative decoding — lossless.** 1.30x oracle bound on wikitext at 7B, which is
   the pessimistic case; re-measure on workload text (`--split workload`) before judging.
3. **The interaction matrix.** Every lever in `ceiling.py` is individually measured; that
   they *compose* is assumed by everyone and demonstrated by no one. Method: for each
   pair, measure KL at matched byte-savings for A alone, B alone, and A+B. If
   KL(A+B) > KL(A) + KL(B), they conflict. Layer dropping and speculative decoding are
   the prime suspects — both cash in "this token was easy".
4. **Publish the scaling law.** Three models, 6x of parameters, a mechanism (depth
   relative to task difficulty; GQA as already-harvested head redundancy) and a predictor
   (tokens per parameter). We have not found this in the literature. It stands whether or
   not ShootingStar's own goal survived — and it did not.

## 5. Rules of method

These exist because they are the ways this kind of project fools itself.

1. **Price everything in bytes/token.** Not FLOPs, not parameter count.
2. **Wall-clock or it did not happen.** Every claimed win must show up as end-to-end
   tok/s on the reference device. A technique without a kernel that actually skips the
   memory traffic is a paper, not a speedup.
3. **Gate on KL, never on accuracy or eyeballed fluency.** Phase 0 produced fluent text
   from a model with 27% of its layers removed and its distribution rearranged beyond
   recognition.
4. **Exact reconstruction error, never importance proxies.** Magnitude and weight-norm
   heuristics ignore cancellation and overstate prunability.
5. **Never transfer an atlas across model scales or corpora.** `ceiling.py` warns when
   you try. It is the easiest way to manufacture a fake 10x.
6. **State predictions before the run.** Phase 1's were recorded with an explicit
   falsifier, then scored. Do that every time.
7. **Report the product of levers as an upper bound to be falsified,** until the
   interaction matrix says otherwise.
8. **Budget memory for the page cache, not just for allocations.** The first 7B attempt
   allocated 16 GB of 23 GB and looked safe, but transformers mmaps weights: once the
   cache was squeezed, every forward pass re-read 13.5 GB from disk and throughput fell
   37x. Check that the model still fits in cache *after* everything else.
