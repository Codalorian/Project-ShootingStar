# Phase 0 — storage roofline (measured)

Instrument: `bench/ssd_roofline.c`. Raw data: `results/ssd_roofline.json`.
Device: Intel SSDPEKNU020TZ (670p, 2 TB, QLC), PCIe 3.0 x4, ext4, `O_DIRECT`,
32 GiB test file, queue depth via concurrent `pread` threads, 2 s per point.

## Headline

| | measured |
|---|---|
| **peak read bandwidth** | **3.1 GB/s** (3.13 seq / 3.10 random, 4 MB blocks, QD ≥ 4) |
| fraction of PCIe 3.0 x4 link (~3.94 GB/s) | 79% — we are near the drive's limit, not the link's |
| **minimum block size for 94% of peak** | **256 KB** |
| **minimum queue depth to saturate** | **4** |
| random-vs-sequential penalty at ≥ 256 KB | **~0–2%** |
| random-vs-sequential penalty at 4 KB, QD1 | **73%** (0.33 → 0.09 GB/s) |
| write bandwidth (block-store build) | 1.85 GB/s |

**The vendor figure of ~3.5 GB/s is 13% optimistic.** Every projection in
`ROADMAP.md` now uses **3.1 GB/s**.

## Bandwidth vs block size and queue depth

Sequential (GB/s):

| bs\qd | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 4K | 0.33 | 0.16 | 0.32 | 0.45 | 0.73 | 1.24 | 1.25 |
| 16K | 0.77 | 0.42 | 0.70 | 1.23 | 1.85 | 2.05 | 1.84 |
| 64K | 1.53 | 0.96 | 1.67 | 2.50 | 2.73 | 2.58 | 2.51 |
| 128K | 1.74 | 1.50 | 2.29 | 2.86 | 2.75 | 2.69 | 2.59 |
| 256K | 2.15 | 2.26 | 2.85 | 2.90 | 2.80 | 2.78 | 2.79 |
| 1M | 2.73 | 1.84 | 2.14 | 2.17 | 3.07 | 2.93 | 2.97 |
| 4M | 2.75 | 2.37 | **3.13** | 2.97 | 2.86 | 3.07 | 3.04 |

Random (GB/s):

| bs\qd | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 4K | 0.09 | 0.16 | 0.28 | 0.43 | 0.69 | 1.14 | 1.21 |
| 16K | 0.23 | 0.42 | 0.69 | 1.10 | 1.56 | 1.77 | 1.71 |
| 64K | 0.50 | 0.88 | 1.50 | 2.12 | 2.25 | 2.21 | 2.18 |
| 128K | 0.73 | 1.45 | 2.16 | 2.49 | 2.49 | 2.48 | 2.45 |
| 256K | 1.23 | 2.21 | 2.92 | 2.81 | 2.83 | 2.78 | 2.76 |
| 1M | 1.68 | 1.88 | 2.02 | 2.11 | 2.99 | 3.01 | 2.96 |
| 4M | 2.23 | 2.40 | 3.08 | 3.03 | **3.10** | 3.09 | 3.07 |

Fraction of peak by block size (best QD):

| block | best | % of 3.1 GB/s |
|---|---|---|
| 4K | 1.25 | 40% |
| 16K | 2.05 | 66% |
| 64K | 2.73 | 88% |
| 128K | 2.86 | 92% |
| **256K** | **2.92** | **94%** |
| 1M–4M | 3.10 | 100% |

## QD1 latency — the price of a cache miss on the critical path

| block | mean | p50 | p99 | max |
|---|---|---|---|---|
| 4K | 46 µs | 44 µs | 66 µs | 188 µs |
| 16K | 71 | 78 | 105 | 2,407 |
| 64K | 131 | 129 | 188 | 3,301 |
| 128K | 170 | 165 | 228 | 3,239 |
| 256K | 212 | 205 | 292 | 3,292 |
| 1M | 655 | 611 | 927 | 3,835 |
| 4M | 2,326 | 2,320 | 2,985 | 4,608 |

**Tail latency is the hazard, not mean latency.** Every block size above 4 KB shows
multi-millisecond maxima (2.4–4.6 ms). A blocking miss is not priced by its mean.

## Findings

### 1. Block size is the lever; queue depth barely matters

Going 4 KB → 256 KB buys **2.3x**. Going QD4 → QD64 buys nothing (often slightly
negative). This inverts the usual NVMe tuning instinct and it simplifies §6 of the
roadmap: **QD 4–8 with large reads is sufficient**, so threads + `pread` reaches the
roofline and `io_uring` is an optimisation rather than a prerequisite.

**Architectural constraint: a loadable block must be ≥ 256 KB contiguous.** At 4-bit that
is ≥ 512K parameters per block. For a 30B model with ~20k total experts each expert is
~1.4M params ≈ 0.7 MB — comfortably above the floor, so the constraint does not bind on
plausible designs. It *would* bind on very fine-grained designs below ~512K params per
expert, which is now a known boundary rather than a surprise.

### 2. Random access is free, if the blocks are large

At 256 KB and above, random reads land within **0–2%** of sequential. At 4 KB QD1 the
penalty is 73%.

This is the most consequential result, and it **demotes a planned optimisation**: §7's
co-activation placement — reordering blocks on disk so co-fired experts sit adjacent —
was described as "pure win." It is worth approximately **nothing** on this device, because
scattered 256 KB+ reads already achieve full bandwidth. Do not build it. The engineering
effort belongs in hit rate and bits-per-weight, which remain the only two real levers.

Caveat worth keeping: this holds for *this* drive. A drive with weaker random performance,
or a heavily fragmented filesystem, could reintroduce the penalty. Re-measure before
generalising the claim.

### 3. `max_sectors_kb` = 128 is not a barrier

256 KB requests are split into two 128 KB I/Os, yet 256 KB still outperforms 128 KB
(2.92 vs 2.86). The kernel pipelines the split. Raising `max_sectors_kb` is not on the
critical path.

### 4. Revised system arithmetic

| | was assumed | measured |
|---|---|---|
| `BW_ssd` | 3.5 GB/s | **3.1 GB/s** |
| DRAM : SSD ratio | 9.4x | **10.6x** |
| miss rate where SSD and DRAM balance | 10.6% | **9.4%** |

Implied ceilings for 2B active @ 4-bit (1.0 GB/token):

| miss rate | roofline | ×0.63 realistic |
|---|---|---|
| 100% | 3.1 tok/s | 2.0 |
| 35% | 8.9 | 5.6 |
| 10% | 31 | 20 |
| 0% (DRAM-bound) | 33 | **21 — the hard cap** |

The first milestone (Qwen3-30B-A3B, ~75% cache fraction) is DRAM-bound and therefore
**unaffected** by the 3.5 → 3.1 correction: still ~14 tok/s. Only the large-model,
SSD-bound projections move, and they move down by ~11%.

## What Phase 0 still owes

- **Sustained-read behaviour.** Every point here is a 2 s burst. QLC drives and laptop
  thermals both degrade under multi-minute load; a real decode session reads continuously.
  Measure a 10-minute sustained run before trusting 3.1 GB/s as a steady-state figure.
- **Buffered vs `O_DIRECT` head-to-head**, which needs `drop_caches` and therefore root.
  The roadmap's claim that `O_DIRECT` is mandatory is currently argued from the previous
  phase's 37x page-cache collapse, not measured here.
- **Concurrent read+compute.** These numbers are read-only. Whether 4 threads of AVX-512
  GEMM steal bandwidth from the NVMe path is untested and directly affects whether prefetch
  actually overlaps.
- **The routing atlas** (§4.2) — the other half of `roofline × atlas → ceiling`, and the
  input to Phase 1's deciding question.
