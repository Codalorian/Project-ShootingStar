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

## 3. Phase 1 — the one question that decides the project

Phase 0 measured a 1.1B model that is over-*trained* (3T tokens / 1.1B params)
and therefore minimally over-*parameterised*. Redundancy is a consequence of
over-parameterisation. **We measured the worst plausible subject.**

**Experiment 1 (do this first, it is one command per model): run the atlas on
7B and 13B.** Everything hinges on whether these curves open up with scale:

| signal | 1.1B (measured) | thesis survives if 7B/13B shows |
|---|---|---|
| max cos(layer in, out) | 0.897 | → 0.99+, several layers |
| MLP keep-frac @ 5% err | 0.62 | → 0.2–0.3 |
| attn head keep-frac @ 5% | 0.65–0.90 | → 0.3 |
| mean easy-run @ p>0.8 | 0.41 | → 2+ |

`python -m shootingstar.atlas.run --model <7B> --n-seqs 16 --seq-len 512`.
Budget ~2 hours per model on the reference device. Plot each signal against
parameter count. **This is a scaling law for redundancy, and as far as we can
tell nobody has published it.** It is a result whether the answer is yes or no.

**Experiment 2: rerun the atlas on `--split workload`.** Redundancy is a
property of a model *on a distribution*. The speculative lever in particular
was measured on wikitext, which is close to worst case; chat and code are far
more predictable. Drop your real traffic into `data/`.

**Experiment 3 — the interaction matrix.** Every lever in `ceiling.py` is
individually published; that they *compose* is assumed by everyone and
demonstrated by no one. The 10x thesis is precisely a multiplicativity claim.
Measure the off-diagonal:

- does MLP contextual sparsity survive 2–3 bit quantisation, or does quant
  noise swamp the neuron-importance signal it depends on?
- does layer dropping degrade speculative acceptance rate? Both cash in "this
  token was easy" — they may be eating the same lunch.
- do KV eviction and head pruning compound the same error twice?

Method: for each pair, measure KL at matched byte-savings for lever A alone,
B alone, and A+B. If KL(A+B) > KL(A) + KL(B), they conflict. **This is the
publishable contribution and it is sized for a small team.**

## 4. If redundancy does not scale

Then the premise is wrong and the project should change, not grind. In
priority order, the levers Phase 0 found that *do not* depend on redundancy:

1. **Cache residency.** Measured 10.7x between DRAM (33 GB/s) and L2/L3
   (319–357 GB/s) on this machine. The 10x is sitting in the memory hierarchy.
   A decode step whose working set stays resident wins without removing
   anything. This is a scheduling and layout problem, not a model problem.
2. **Speculative decoding.** Lossless, and its 1.24x oracle bound on wikitext
   is the pessimistic case — re-measure on real workload text first.
3. **Sub-4-bit weights with a VNNI kernel.** Orthogonal to all redundancy
   findings; the reference device has AVX-512 VNNI and VPOPCNTDQ, which is a
   good instruction set for bit-packed weights that llama.cpp does not
   currently exploit on this class of chip.

## 5. Rules of method

These exist because they are the ways this kind of project fools itself.

1. **Price everything in bytes/token.** Not FLOPs, not parameter count.
2. **Wall-clock or it did not happen.** Every claimed win must show up as
   end-to-end tok/s on the reference device. A technique without a kernel that
   actually skips the memory traffic is a paper, not a speedup.
3. **Gate on KL, never on accuracy or eyeballed fluency.** Phase 0 produced
   fluent text from a model with 27% of its layers removed and its output
   distribution rearranged beyond recognition.
4. **Exact reconstruction error, never importance proxies.** Magnitude and
   weight-norm heuristics ignore cancellation and overstate prunability.
5. **Never transfer an atlas across model scales or corpora.** `ceiling.py`
   warns when you try. It is the easiest way to manufacture a fake 10x.
6. **Report the product of levers as an upper bound to be falsified,** until
   the interaction matrix says otherwise.
