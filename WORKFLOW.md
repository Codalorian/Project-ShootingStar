Project-ShootingStar

Unlike the typical transformer, Project-ShootingStar attempts to build a storage-native sparse LLM.

The Idea

Instead of requiring the entire model to fit into RAM:

Traditional LLM
    ↓
RAM
    ↓
500B parameters
    ↓
CPU


You build a model where:

Storage-native LLM
    ↓
SSD
    ↓
500B params
    ↓
Intelligent router
    ↓
Only needed blocks
    ↓
RAM
    ↓
CPU


The model learns which parts of itself are needed for each token.

The key technologies:

Sparse/conditional computation — don't execute all parameters.

Modular parameter blocks — make parts independently loadable.

SSD parameter storage — cold parameters remain on NVMe.

Predictive prefetching — load blocks before the CPU needs them.

RAM caching — keep frequently used blocks resident.

Quantization — minimize bytes transferred.

Learned locality — train the model to reuse parameters and keep commonly co-used blocks together.

The Fundamental Goal

Instead of:

500B params → every token


you want something like:

500B total
    ↓
2B active
    ↓
only required blocks
    ↓
RAM
    ↓
CPU

The Core Insight

The model doesn't need to store all 500B parameters in RAM.

It only needs fast access to the small subset of parameters required for the current computation.

The SSD becomes the model's cold memory, while RAM becomes its working set.

The challenge is making parameter loading fast enough that SSD access doesn't become the bottleneck.

What ShootingStar Needs to Solve

Routing — Which parameter blocks does the current token need?

Prefetching — Which blocks will the next tokens need?

Caching — Which blocks should remain in RAM?

Locality — How do we make commonly used blocks physically close together?

I/O efficiency — How do we minimize random SSD reads and bytes transferred?

Sparsity — How do we keep the active parameter count extremely small without destroying model quality?

The Ideal Runtime
Token
  ↓
Router
  ↓
Predict required blocks
  ↓
Check RAM cache
  ↓
 ┌───────────────┐
 │ Cache hit      │ → use immediately
 │ Cache miss     │ → load from SSD
 └───────────────┘
  ↓
Execute active parameters
  ↓
Predict next blocks
  ↓
Prefetch while computing
  ↓
Next token


Ideally, SSD reads happen before the parameters are needed, so storage latency is hidden behind computation.

The Long-Term Goal

Build a model where:

Total parameters:     500B
Active parameters:      ~2B/token
RAM:                   far smaller than model
Storage:               NVMe


The important metric isn't just how many parameters the model has.

It's:

How much useful computation can we get
per byte loaded from storage?


If successful, ShootingStar would make it possible to experiment with models whose total parameter count is much larger than the machine's available RAM, while only keeping a small, intelligently selected working set in memory.