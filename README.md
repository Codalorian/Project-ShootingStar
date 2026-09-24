# Project-ShootingStar

A **storage-native sparse LLM**. Unlike a typical transformer, ShootingStar does not
require the whole model to fit in RAM.

## The idea

A traditional LLM has to hold every parameter in memory to run at all:

```
Traditional LLM          Storage-native LLM
     ↓                          ↓
    RAM                        SSD
     ↓                          ↓
500B parameters            500B parameters
     ↓                          ↓
    CPU                  Intelligent router
                                ↓
                         only needed blocks
                                ↓
                               RAM
                                ↓
                               CPU
```

The model learns which parts of itself are needed for each token. The SSD becomes the
model's cold memory; RAM becomes its working set.

## The core insight

The model doesn't need to *store* all 500B parameters in RAM. It only needs **fast access
to the small subset required for the current computation.**

The challenge is making parameter loading fast enough that SSD access doesn't become the
bottleneck.

## The fundamental goal

Instead of `500B params → every token`:

```
500B total
    ↓
2B active
    ↓
only required blocks
    ↓
RAM
    ↓
CPU
```

## Key technologies

| | |
|---|---|
| **Sparse / conditional computation** | don't execute all parameters |
| **Modular parameter blocks** | make parts independently loadable |
| **SSD parameter storage** | cold parameters remain on NVMe |
| **Predictive prefetching** | load blocks before the CPU needs them |
| **RAM caching** | keep frequently used blocks resident |
| **Quantization** | minimize bytes transferred |
| **Learned locality** | train the model to reuse parameters, and keep commonly co-used blocks together |

## What ShootingStar needs to solve

- **Routing** — which parameter blocks does the current token need?
- **Prefetching** — which blocks will the next tokens need?
- **Caching** — which blocks should remain in RAM?
- **Locality** — how do we make commonly used blocks physically close together?
- **I/O efficiency** — how do we minimize random SSD reads and bytes transferred?
- **Sparsity** — how do we keep the active parameter count extremely small without
  destroying model quality?

## The ideal runtime

```
Token
  ↓
Router
  ↓
Predict required blocks
  ↓
Check RAM cache
  ↓
 ┌──────────────┐
 │ Cache hit    │ → use immediately
 │ Cache miss   │ → load from SSD
 └──────────────┘
  ↓
Execute active parameters
  ↓
Predict next blocks
  ↓
Prefetch while computing
  ↓
Next token
```

Ideally, SSD reads happen *before* the parameters are needed, so storage latency hides
behind computation.

## The long-term goal

```
Total parameters:     500B
Active parameters:    ~2B/token
RAM:                  far smaller than model
Storage:              NVMe
```

The important metric isn't how many parameters the model has. It's:

> **How much useful computation can we get per byte loaded from storage?**

If successful, ShootingStar would make it possible to experiment with models whose total
parameter count is much larger than the machine's available RAM, while only keeping a
small, intelligently selected working set in memory.

## Repository layout

```
WORKFLOW.md     the design this repo is building toward — the canonical statement of it
pytorch/        PyTorch source (v2.15.0a0), vendored
transformers/   Hugging Face Transformers source (v5.18.0.dev0), vendored
```

Both dependencies are vendored as source rather than installed as wheels: block-granular
weight loading and kernels that actually skip memory traffic live below the Python API,
so they have to be modified and rebuilt.

> **Note:** `pytorch` and `transformers` are recorded as gitlinks but the repo has no
> `.gitmodules`, so a fresh `git clone` leaves both directories empty. Add submodule
> entries or clone the two trees in place before building.

## Prior work

ShootingStar began as an attempt to hit the same goal by exploiting redundancy already
inside the weights. That was measured across three models and **falsified** — the implied
bytes-per-token ceiling was 1.43x / 1.80x / 2.77x (Llama-3.2-1B, TinyLlama-1.1B,
Llama-2-7B), not 10x. Redundancy scales with model size, but not steeply enough, and it
is largest exactly where the machine can least afford to run the model. That result is
why the project moved to the storage-native design above.

The instruments and write-ups live in git history at `bb002cc`:

```bash
git show bb002cc:docs/PHASE0-FINDINGS.md   # the full write-up
git show bb002cc:docs/ROADMAP.md           # phase plan, and the corrections made to it
git checkout bb002cc -- shootingstar bench results   # restore the measurement tools
```

Two results from that work still constrain the design here. At batch size 1 decode runs
at 4.0 ops/byte against a measured machine balance of 36.3, so **89% of the machine's
arithmetic is already idle waiting on memory** — which is the headroom a storage-native
model spends on routing and prefetch. And quality must be gated on **KL divergence**, not
accuracy or eyeballed fluency: generations stayed fluent with 27% of layers removed and
the output distribution rearranged beyond recognition.
