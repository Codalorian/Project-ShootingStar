# ShootingStar roadmap

> Build a storage-native sparse LLM: 500B total parameters, ~2B active per token, on a
> machine with 31 GB of RAM. See [WORKFLOW.md](WORKFLOW.md) for the design.

## 0. The reframing that makes the goal measurable

The previous incarnation of this project learned that at batch size 1 decoding is bound
by **bytes moved**, not FLOPs. Moving the cold weights to NVMe does not repeal that — it
makes it sharper, because it swaps a 33 GB/s bus for a 3.5 GB/s one.

The tempting framing is "hide SSD latency behind computation." That framing is wrong and
it will waste months. Price it:

| | per token, 2B active @ 4-bit |
|---|---|
| bytes to touch | 1.0 GB |
| compute | ~4 GFLOP → **~3 ms** (≈300 tok/s if compute-bound) |
| I/O at 3.5 GB/s, every byte a miss | **~285 ms** (≈3.5 tok/s) |

I/O dominates compute by roughly **85x**. There is no amount of idle arithmetic to hide
285 ms behind. Prefetching converts *latency* into *throughput*, and this problem is not
latency-bound — it is **bandwidth**-bound.

**The metric is bytes read from storage per generated token.** Prefetch depth, queue
depth and IOPS are means; they are not evidence. Only two things actually reduce SSD
bytes:

1. **Cache hit rate** — the block was already in RAM.
2. **Bits per weight** — the block was smaller.

Everything in this roadmap is priced against those two. `tok/s` claims that do not
decompose into them are not accepted.

## 1. The master equation

```
                          BW_ssd
tok/s  ≤  ─────────────────────────────────────
           active_bytes  ×  miss_rate
```

with `T_compute` (~3 ms) small enough to overlap away entirely. Everything the project
builds moves one of three terms. On the reference device (`BW_ssd` = 3.5 GB/s, to be
measured in Phase 0):

| active params (4-bit) | miss 100% | miss 35% | miss 10% | miss 3% |
|---|---|---|---|---|
| **2B** (1.0 GB/token) — the target | 3.5 tok/s | **10 tok/s** | 35 tok/s | 117 tok/s |
| 3B (1.5 GB/token) — Qwen3-30B-A3B | 2.3 | 6.7 | 23 | 78 |
| 22B (11 GB/token) — Qwen3-235B-A22B | 0.3 | 0.9 | 3.2 | 10.6 |

Read off the two things that matter:

- **The 2B-active target is not crazy.** It needs a miss rate ≤ 35% to clear 10 tok/s.
  That is a demanding but not absurd cache requirement.
- **Active volume, not total size, is what kills you.** A 22B-active model needs a *3%*
  miss rate for the same result. Worse, 11 GB of active weights per token cannot fit in a
  RAM cache on this machine *at all*, so a high hit rate is not merely unachieved, it is
  arithmetically unavailable.

**Therefore the architectural constraint is: `active_bytes` must fit comfortably inside
the RAM cache, with room for the working set to drift across neighbouring tokens.** At
~14 GB of usable cache that means active ≤ ~1–2 GB, i.e. **≤ 2–4B active params at
4-bit.** The WORKFLOW target of 2B active is the right number. It is existing MoE models
that are the wrong shape for this, not the target.

## 2. Reference device

Measured, not assumed — except where flagged.

| | |
|---|---|
| CPU | i7-1165G7, 4c/8t, AVX-512 + VNNI. Use **4 threads** (read bandwidth peaks there, −17% at 8) |
| RAM | 31 GB LPDDR4x, **33 GB/s** measured read. ~20 GB usable after OS; budget **~14 GB** for the block cache after activations and KV |
| L2 / L3 | 5 MB (4×1.25) / 12 MB — irrelevant at this scale, see the correction in §9 |
| SSD | **Intel SSDPEKNU020TZ (670p), 2 TB, QLC**, PCIe **3.0 x4** (8.0 GT/s ×4 measured) |
| SSD ceiling | link cap ~3.94 GB/s; vendor sequential read ~3.5 GB/s — **Phase 0 must measure the real figure under the access pattern a block loader actually issues** |
| **Free disk** | **187 GB** — the binding constraint on model size |
| `max_sectors_kb` | **128** — single I/Os are split at 128 KB; raise toward `max_hw_sectors_kb` and re-measure |
| scheduler | `none` (correct for NVMe) |

Two consequences that are easy to miss:

- **RAM is only ~9.4x faster than this SSD** sequentially. That is the whole budget the
  project is spending. It is not 100x, and it is not 2x.
- **187 GB free caps the model.** A 500B model at 4-bit is 335 GB and **does not fit**.
  At 4-bit the ceiling is ~370B params; reaching 500B on this disk requires ~3-bit or
  clearing space. Decide this before downloading anything.

QLC specifics to characterise rather than trust: dynamic SLC cache behaviour, sustained
large-read throughput as the drive heats, and whether read performance degrades once the
resident data exceeds the SLC region. Reads do not meaningfully wear the drive, so
endurance is not a concern; sustained throughput is.

## 3. The sparsity gap — state it before starting

The target is 500B/2B = **250x** sparsity. Nothing open comes close:

| model | total / active | sparsity |
|---|---|---|
| Mixtral 8x7B | 46.7B / 12.9B | 3.6x |
| OLMoE-1B-7B | 6.9B / 1.3B | 5.3x |
| Qwen3-30B-A3B | 30B / 3B | 10x |
| DeepSeek-V2 | 236B / 21B | 11x |
| DeepSeek-V3 | 671B / 37B | 18x |
| **ShootingStar target** | **500B / 2B** | **250x** |

So there is a **~14x gap** between the sparsest thing that exists and what this design
needs. That gap is the actual research contribution, and it is an *architecture and
training* problem, not a runtime problem. Which forces an honest split:

- **The runtime can be built and validated now**, on existing models, without training
  anything. This is most of the engineering and all of the near-term results.
- **The 250x-sparse 500B model cannot be trained on this hardware.** A run at that scale
  is eight figures. The deliverable for that half is *an architecture plus small-scale
  evidence that it works*, not the model.

Write that down now, because the failure mode is drifting into believing a 500B model is
a milestone rather than someone else's budget.

## 4. Phase 0 — instruments

Deliverable: **one number** — the implied tok/s ceiling for a given (model, cache size,
SSD), computed as `storage roofline × routing atlas`. This deliberately mirrors the
previous project's `roofline × atlas → ceiling`, which worked.

### 4.1 `bench/ssd_roofline.c` — the storage roofline

Measure achievable read bandwidth across the space a block loader can actually occupy:

- block size 4 KB → 16 MB (note the 128 KB split at `max_sectors_kb`)
- queue depth 1 → 256
- `O_DIRECT` vs buffered, sequential vs random, cold vs warm
- `io_uring` vs `pread` vs `mmap` + `MADV_WILLNEED`
- sustained multi-minute reads, to catch thermal and QLC effects

Expected shape, to be confirmed or falsified: QD1 4 KB random is latency-bound near
~40 MB/s and useless; QD1 at 128 KB is ~0.9 GB/s; saturation needs QD ≥ 4–8 with large
blocks. **This directly sets the minimum viable block granularity** — if 3.5 GB/s needs
≥ 512 KB reads, then a "block" must be ≥ 512 KB of contiguous weights, which constrains
the architecture in §8.

**`O_DIRECT` is not optional.** With buffered I/O the kernel page cache holds a second
copy of every block, competing with your own cache on a 31 GB machine for a 118 GB
model. The previous project already lost 37x throughput to exactly this mistake (§9 rule
8). Own the cache, bypass the kernel's.

### 4.2 `atlas/routing.py` — the routing atlas

Instrument real MoE models, dump per-token expert selections over a fixed corpus, and
compute:

- **Skew** — activation frequency distribution per layer. Power-law, or flat? Flat is
  fatal; LRU cannot beat a uniform distribution.
- **Temporal reuse** — P(expert active at t+1 | active at t). This is what a cache
  converts into hit rate.
- **Hit-rate curves** — simulated LRU / LFU / ARC hit rate vs cache size, with cache size
  expressed as a **fraction of model size**, never in GB. Fractions transfer to the 500B
  regime; absolute sizes do not.
- **Co-activation structure** — which experts fire together. Feeds block placement in §8.
- **Predictability** — can token t+1's experts be predicted from the layer-`n` hidden
  state at token t? Baselines: marginal frequency, bigram, small MLP probe. This is the
  prefetcher's ceiling.

Measure on ≥3 models spanning routing granularity, chosen so the *shape* varies:
**OLMoE-1B-7B** (64 fine-grained experts, top-8), **DeepSeek-V2-Lite** (fine-grained +
shared experts), **Qwen3-30B-A3B** (10x sparse). None of them needs storage to run — that
is fine and in fact the point: constrain the simulated cache to a *fraction* of model
size and the results generalise upward.

### 4.3 `hw.py` — byte accounting

Given (n_params, n_active, bits, cache fraction, measured hit rate, measured `BW_ssd`),
emit the implied tok/s and the byte budget per token. Every later claim gets checked
against this.

## 5. Phase 1 — the deciding question

> **Is routing skewed and temporally correlated enough that a RAM cache holding a small
> fraction of the model gets a high hit rate?**

This is the load-bearing question, exactly as "does redundancy scale with size?" was
before. Everything downstream is dead if the answer is no, and it is cheap to answer —
Phase 0's atlas already produces it.

**State predictions before running, with a falsifier, then score them.** Mine, to be
scored honestly:

- Activation frequency is power-law, not uniform, in every layer.
- Hit rate at a **10% cache fraction** lands in **50–75%** under plain LRU.
- Fine-grained routing (OLMoE, 64 experts) caches *better* than coarse (Mixtral, 8), per
  expert byte — smaller blocks mean less wasted transfer per useful parameter.
- Next-token expert prediction beats the frequency baseline by a wide margin, because
  routing is largely driven by slow-moving topical context.
- Early layers are more predictable than late ones.

**Falsifier / kill criterion:** if LRU hit rate at a 10% cache fraction is **below ~40%**
across all three models, plain caching cannot get the miss rate under 35% and this
architecture does not work with off-the-shelf routing. That outcome does not end the
project — it *promotes* §8 (learned locality) from optimisation to prerequisite, which is
a much more expensive path and should be entered deliberately, not by accident.

Note carefully: **the runtime is exactly lossless.** Paging weights from NVMe computes
precisely the same function as holding them in RAM. There is no quality gate on Phases
0–2 because there is no approximation. The KL gate in §9 binds only from §7 onward, where
the project starts *skipping* or *degrading* blocks. Keep that boundary clean — it is the
project's main defence against fooling itself.

## 6. Phase 2 — the runtime

Where "wall-clock or it did not happen" applies. Build, in order:

1. **Block store.** Convert a model to a flat file of contiguous, independently loadable
   blocks (one expert = one block, ≥ the minimum size from §4.1), plus an index. Blocks
   must be aligned and sized for `O_DIRECT`.
2. **Loader.** `io_uring` + `O_DIRECT`, deep queues, large reads. Must reach the §4.1
   measured bandwidth on the real access pattern, not just on a synthetic sweep.
3. **Cache.** Explicit, user-managed, byte-budgeted. Start with LRU as the baseline the
   smarter policies must beat; the atlas says which are worth trying.
4. **Prefetcher.** Issue block reads for token t+1 during token t's compute, driven by the
   §4.2 predictor. Remember its job is hiding the ~3 ms of compute and the queue latency —
   it cannot manufacture bandwidth.
5. **Miss policy.** The interesting design question. On a miss: (a) stall and load,
   (b) skip the block and accept the error, or (c) use a resident low-precision copy and
   upgrade on arrival. (c) — *precision follows temperature* — is the elegant one and
   feeds §7. (a) is the lossless baseline. (b) needs the KL gate.

### The baseline this must beat

**`llama.cpp` with `mmap` on the same model.** This is not a strawman — the kernel page
cache already *is* an LRU cache over the weight file, with readahead, and it is very
good. The project's entire claim reduces to:

> **routing-aware, block-granular caching with prediction beats LRU-over-4 KB-pages.**

If the runtime cannot beat `mmap` + page cache end to end, there is no result, however
good the hit-rate curves look. Measure both on identical weights, on the reference
device, in tok/s.

**Milestone:** a model whose on-disk size exceeds RAM, decoding at a measured tok/s, with
a byte-per-token accounting that matches `hw.py`'s prediction. Pick the flagship by
§2's 187 GB limit — **Qwen3-235B-A22B at 4-bit (~118 GB)** fits and is genuinely 4x RAM,
but note from §1 that its 22B active volume caps it near 1 tok/s. It is a *correctness and
bandwidth* demonstration, not a speed one. Say so in the write-up rather than quietly
choosing a friendlier model.

## 7. Phase 3 — bits per weight, and layout

The second of the only two levers. Now the KL gate binds.

- **Precision follows temperature.** Hot blocks at 4-bit, cold blocks at 2–3 bit. A miss
  on a cold block then costs half the bytes. Because cold blocks are by definition rarely
  used, the quality cost should be far below uniform quantisation at the same mean bit
  width — *should*, so measure it against that exact control.
- **Co-activation placement.** Order blocks on disk so experts that fire together are
  physically adjacent, turning several random reads into one sequential read. The §4.2
  co-activation matrix gives the clustering objective. This is pure win — no quality cost.
- **Block size vs waste.** Bigger blocks hit the measured bandwidth sweet spot but
  transfer parameters you did not need. There is an optimum; find it empirically.

## 8. Phase 4 — learned locality (the training half)

Only after §5 says how much the runtime can get from *existing* routing. The thesis in
WORKFLOW.md is that a model can be *trained* to be cache-friendly. Auxiliary objectives
worth testing, at 100M–1B scale as a proxy:

- **Stickiness** — reward reusing the previous token's experts, so temporal reuse rises.
- **Clustering** — reward co-activation of experts that are placed together.
- **Extreme sparsity** — push toward the §3 gap: many more, much smaller experts. The
  fine-grained-expert and million-expert lines of work are the relevant prior art.

Deliverable: a small model whose *routing atlas* is measurably better than an
architecturally matched control trained without these losses, at equal quality. Report
hit rate at matched cache fraction, and KL against the control. That is a real result at
1B and it is the honest evidence for the 500B design — which this hardware will not train.

## 9. Rules of method

Carried over from the previous phase of this project, where each was learned the hard way,
plus the ones specific to storage.

1. **Price everything in bytes read from storage per token.** Not IOPS, not queue depth,
   not prefetch hit count, not FLOPs.
2. **Wall-clock or it did not happen.** Every claimed win shows up as end-to-end tok/s on
   the reference device. A technique without a loader that actually skips the reads is a
   paper, not a speedup.
3. **Beat `mmap` + page cache, or admit there is no result.** The naive baseline is
   strong; most of the literature's wins shrink when measured against it properly.
4. **Gate on KL, never on accuracy or eyeballed fluency** — mean KL(reference ‖ modified)
   ≤ 0.05 nats/token with top-1 agreement ≥ 0.95, on held-out *and* workload text. The
   previous phase produced fluent text from a model with 27% of its layers removed and
   its output distribution rearranged beyond recognition.
5. **Keep the lossless/lossy boundary explicit.** Phases 0–2 are exact and need no gate.
   The moment a miss is skipped or a block degraded, the gate binds. Never let an
   approximation slip in unlabelled.
6. **Express cache size as a fraction of model size, never in GB.** Absolute sizes do not
   transfer across scales, and transferring them is the easiest way to manufacture a fake
   result.
7. **Never transfer an atlas across models or corpora.** Routing is a property of a model
   *on a distribution*.
8. **Budget memory for the page cache, not just allocations.** The first 7B attempt of the
   previous phase allocated 16 GB of 23 GB and looked safe, but `transformers` mmaps
   weights: once the cache was squeezed, every forward pass re-read 13.5 GB from disk and
   throughput fell **37x**. This project is that failure mode by design — use `O_DIRECT`
   and own the cache.
9. **State predictions before the run, with a falsifier, then score them.**
10. **Report composed levers as an upper bound to be falsified.** Hit rate and
    quantisation interact: cheaper cold blocks mean more of them fit, which changes the
    hit rate. Measure the product, do not multiply the parts.

### Correction inherited from the previous phase

Cache *residency* is not a harvestable lever at these model sizes, and the measured
357 GB/s (L2/L3) vs 33 GB/s (DRAM) ratio must not be presented as available speedup. L3
is 12 MB; the working set is gigabytes. The exploitable form of that physics is **reuse**,
not residency. The same discipline applies one level down: the RAM-vs-NVMe ratio here is
~9.4x and it is real, but only for bytes that are actually *reused*.

## 10. Prior art to triage first

This is a crowded field and several groups have built large parts of this. Read before
writing code; the goal is to find the remaining gap, not to rediscover the field.

- **"LLM in a flash: Efficient LLM Inference with Limited Memory"** (Apple) — the closest
  prior work. Flash-aware windowing and row-column bundling. Read this first.
- **PowerInfer** and **PowerInfer-2** — hot/cold neuron split exploiting power-law
  activation skew; the second targets flash on phones. Directly overlapping.
- **MoE expert offloading** — Mixtral-offloading (LRU + speculative expert prefetch),
  MoE-Infinity, Pre-gated MoE, EdgeMoE, SiDA-MoE, AdapMoE, Fiddler. Several already
  implement §6's cache-plus-prefetcher.
- **DejaVu** — contextual sparsity with learned predictors; the §4.2 predictor question.
- **FlexGen**, **DeepSpeed ZeRO-Inference** — offload scheduling, but throughput- and
  batch-oriented rather than latency at batch 1.
- **DeepSeekMoE** (fine-grained experts + shared experts) and the **million-experts /
  product-key-memory** line — the architecture direction for closing §3's 14x gap.
- **`llama.cpp` mmap**, **ktransformers** — the baselines of §6.

Expected outcome of the triage: the *runtime* is largely known art, and the genuinely open
question is §3 — **whether a model can be trained to be 100–250x sparse with routing
locality good enough to make storage-native decoding fast.** Aim the novelty claim there,
and let the runtime be solid engineering rather than a contribution.

## 11. Kill criteria

Decide these now, while there is no stake in the answer.

| check | kill / pivot condition |
|---|---|
| §4.1 storage roofline | sustained large-block read < ~2 GB/s → the bandwidth budget is too thin; the target drops accordingly |
| §5 hit rate | LRU < 40% at a 10% cache fraction on all models → plain caching fails; §8 becomes a prerequisite, not an optimisation |
| §6 vs baseline | cannot beat `mmap` + page cache end to end → no result; fix the loader or stop |
| §7 quantisation | temperature-tiered quantisation fails to beat uniform quantisation at matched mean bit width → drop the idea, it is just quantisation |
| §8 learned locality | no hit-rate gain over an architecturally matched control at equal KL → the central WORKFLOW thesis is false; report it and stop |

The previous phase of this project asked whether weight redundancy could deliver 10x,
measured carefully, and found 2.77x. It published the negative result and changed
direction. Hold this phase to the same standard.
