# metalsw

Exact Smith-Waterman protein database search on Apple Silicon GPUs, via Metal.

## Positioning

Published GPU-accelerated Smith-Waterman aligners — CUDASW++4.0, GASAL2, ADEPT, Accelign — target CUDA. A search for a working, verified Metal implementation of exact Smith-Waterman (not an approximation, not a stub) turned up nothing: no project this search could find and verify actually runs, is correctness-checked against a reference, and reports measured throughput on Apple Silicon.

**metalsw is not claiming to be first** — that's an unfalsifiable claim about the entire universe of unpublished code, and this project has no way to verify it. The claim is narrower and checkable: *this is a Metal Smith-Waterman implementation whose correctness is verified against a scalar CPU oracle on tens of thousands of real protein sequences, and whose performance is measured and reported honestly, including the cases where it loses to a CPU baseline.*

It is not trying to compete with data-center GPU throughput. An H100 has ~50x the compute of a base M2 and hardware DPX instructions this project's GPU has no equivalent for (see [Comparative analysis](#comparative-analysis-v05) below). The goal is proving exact GPU-accelerated protein alignment works on Apple Silicon, with every number in this README backed by a measurement or a cited, verified source — not by "expected" or "should" language.

## What the tool does

Input: one protein query, one FASTA database of many sequences.
Output: top-N local alignment scores, ranked.
Method: exact Smith-Waterman, affine (Gotoh) gaps, BLOSUM62 substitution matrix, score only (no traceback).

```
query.fasta ──┐
              ├──> [metalsw] ──> ranked list of (score, sequence_id)
db.fasta ─────┘
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Host (C++17)                                                 │
│  ─ parses FASTA (query + database)                            │
│  ─ builds the query profile (host-side precompute against      │
│    BLOSUM62, one row per residue)                              │
│  ─ sorts DB sequences by length before dispatch (SIMD-group     │
│    balance — see Stage 2 findings below)                       │
│  ─ GpuRunner: owns device/pipeline/buffers, caches and reuses   │
│    them across calls keyed on query+corpus identity             │
│  ─ collects scores, unpermutes back to original order, sorts,   │
│    prints top-N                                                 │
└───────────────────────┬─────────────────────────────────────────┘
                         │  metal-cpp (Metal API bindings)
┌───────────────────────▼─────────────────────────────────────────┐
│  Metal API layer                                                 │
│  MTLDevice → MTLCommandQueue → MTLCommandBuffer                   │
│  → MTLComputeCommandEncoder → dispatchThreads                     │
│  MTLBuffer, ResourceStorageModeShared (unified memory —            │
│  no explicit host/device copy; CPU writes go straight into          │
│  what the GPU reads)                                                │
└───────────────────────┬─────────────────────────────────────────────┘
                         │  compiled .metallib
┌───────────────────────▼─────────────────────────────────────────────┐
│  GPU kernels (Metal Shading Language, metal/smith_waterman.metal)    │
│  one database sequence per thread, inter-sequence parallelism only    │
│  (no intra-sequence/wavefront parallelism — out of scope for v1)       │
│                                                                          │
│  Pass 1 (smith_waterman_score_int8): H/E/F run in int8_t, 4x the         │
│  per-thread footprint of int16. Flags per-thread saturation.             │
│  Pass 2 (smith_waterman_score): int16, re-dispatched only over the        │
│  subset of sequences that overflowed int8 in pass 1.                       │
└──────────────────────────────────────────────────────────────────────────┘
```

**Correctness path, not optional:** every GPU score is checked against the in-repo scalar CPU oracle (`sw_reference.cpp`) and, when Parasail is available, against Parasail too. See [Correctness](#correctness) for what's actually been checked.

## Correctness

GPU output diffed directly against the scalar CPU oracle, same query/gap-penalty inputs:

| Test set | Sequences | Mismatches |
|---|---|---|
| Hand-built edge cases (identical / point mutation / partial match / shuffled / reversed / unrelated) | 6 | 0 |
| Real Swiss-Prot subset (pinned release, see [Benchmark corpus](#benchmark-corpus)) | 750 | 0 |
| Real Swiss-Prot data, larger subsets | 5,000 / 20,000 / 50,000 | 0 / 0 / 0 |
| Real Swiss-Prot subset, 4 different query lengths (60–502 residues) | 750 each | 0 |

**78,006 total sequence-score comparisons, 0 mismatches**, gap open=11/extend=1, BLOSUM62. The oracle itself was cross-checked against Parasail (0/6 mismatches) before any GPU code was written, to confirm the gap-penalty convention was correct.

## Benchmark corpus

`data/swissprot_subset.fasta` — 750 sequences, length ≤ 2000, deterministic first-N-after-length-filter selection from a specific, dated Swiss-Prot release (provenance recorded in `data/prepare_swissprot_subset.py`: source URL, download date, ETag). Query: `data/query_hemoglobin.fasta` (hemoglobin subunit alpha, 142 residues). Gap penalties: open=11, extend=1.

Additional DB sizes (5,000 / 20,000 / 50,000 sequences) and query lengths (60 / 256 / 502 residues) used in the benchmark sweep below were generated from the same pinned release with the same recipe but live in a scratch directory, not committed to the repo — only the 750-sequence corpus and hemoglobin query ship with the repo.

## Benchmark results

Hardware: Apple M2 (base, `Mac14,2` MacBook Air, fanless — no active cooling), 10-core GPU, 8 CPU cores (4P+4E), 8GB unified memory, macOS 26.5.2. Methodology: 20 warm-up iterations (discarded) + ≥15 measured iterations per data point, mean ± stddev reported (not best-of-N), cold-start GCUPS reported separately from steady-state. Parasail 2.6.2 (real SIMD CPU baseline, not scalar) via Homebrew.

**GPU vs. Parasail, by DB size** (query = hemoglobin, 142 residues):

| DB size | GPU GCUPS (mean±stddev) | Parasail GCUPS (mean±stddev) | GPU/Parasail |
|---|---|---|---|
| 750 | 0.407 ± 0.0075 | 0.407 ± 0.0009 | ~1.0x (roughly tied) |
| 5,000 | 2.217 ± 0.0062 | 0.370 ± 0.0295 | 5.99x |
| 20,000 | 5.224 ± 0.0322 | 0.393 ± 0.0067 | 13.30x |
| 50,000 | 5.309 ± 0.0284 | 0.392 ± 0.0043 | 13.54x |

**There is no single "GPU is Nx faster" number.** At small corpus sizes, fixed per-dispatch overhead (`commandBuffer` → encode → commit → `waitUntilCompleted`, a synchronous CPU/GPU round-trip) dominates and the GPU doesn't clearly beat a CPU SIMD baseline. That overhead gets amortized as the corpus grows, and sequence-length binning (sorting DB sequences by length before dispatch, so threads in the same SIMD group do similar-sized work) adds a further 17–55% on top of that scaling effect — see the full experimental history in `PROJECT_BLUEPRINT.md` for the failed hypotheses along the way (buffer-reuse was tried first and found to make no difference; the actual bottleneck was dispatch count and SIMD divergence, not memory allocation).

**Power / thermal** (target Mac, 50,000-sequence corpus, 480 measured iterations, ~244s sustained, `powermetrics` sampled every 1s): combined power 6.46W mean (CPU 664.6mW + GPU 5800.0mW), **0.816 GCUPS/Watt**. GPU ran at ~99.9% HW active residency and max clock (1398 MHz) for nearly the entire run. No thermal or performance warnings (`pmset -g therm`) before or after; GCUPS held flat/trended slightly upward across the full run — no throttling observed at this power draw on this fanless machine.

## Comparative analysis (v0.5)

| System | GCUPS/Watt | Source |
|---|---|---|
| CUDASW++4.0 on H100 | 15.7 | [cited](https://pmc.ncbi.nlm.nih.gov/articles/PMC11531700/), synthetic equal-length DB, s16x2+DPX |
| CUDASW++4.0 on L40S | 15.2 | [cited](https://pmc.ncbi.nlm.nih.gov/articles/PMC11531700/), synthetic equal-length DB, s16x2+DPX |
| Accelign on RTX PRO 6000 Blackwell | not reported | [cited](https://www.biorxiv.org/content/10.64898/2025.12.17.694868v1.full) (peak 9.1–16.1 TCUPS reported; no energy figure) |
| **metalsw on M2 (fanless)** | **0.816** | **measured, this work** |

metalsw is ~19x less energy-efficient than CUDASW++4.0's cited H100/L40S figures. This is not a like-for-like comparison and shouldn't be read as one: those are 200–300W data-center GPUs with hardware DPX fused add-max instructions, benchmarked against a synthetic equal-length database that removes length-imbalance overhead entirely. metalsw draws ~6.5W combined against a real, length-varied Swiss-Prot subset on a fanless laptop chip.

Three findings, from measurements taken during this project (not speculation):

1. **Unified memory removes host/device transfer cost, but that's not where metalsw's bottleneck was.** `MTLResourceStorageModeShared` buffers are the same physical RAM the CPU touches — no PCIe-style transfer step exists to eliminate on this architecture. Confirmed empirically: a buffer-reuse experiment that removed ~12 allocations per call found *zero* GCUPS change, because there was no transfer cost being paid. The real bottleneck was fixed per-dispatch synchronization overhead, not data movement.
2. **The M2's lack of a DPX-style fused add-max instruction is a binding, ISA-level disadvantage against Hopper's fastest path**, not something kernel tuning on our side can close.
3. **The fanless thermal ceiling never showed up in these measurements** — the GPU ran essentially saturated (99.9% active residency, max clock) throughout, drawing ~6.5W combined, well within what a passive chassis dissipates continuously. Whether a larger or longer workload eventually hits it is untested.

## Limitations

- Score only — no traceback / alignment path reconstruction.
- Protein sequences only — BLOSUM62, no DNA/RNA scoring matrices.
- Inter-sequence parallelism only (one GPU thread per database sequence) — no intra-sequence/wavefront parallelism.
- Query length capped at 512 residues (`MAX_QUERY_LEN` in the kernel).
- Benchmarked on exactly one machine: a base (non-Pro/Max) M2, fanless MacBook Air. No data on Pro/Max/Ultra chips or actively-cooled machines.
- The int8 fast path handled 100% of every real corpus tested in this project (0 sequences ever overflowed to the int16 fallback) — the fallback path is exercised and correct (verified on hand-built high-scoring edge cases) but has not been exercised by a real-world corpus in this project's own testing.

## Build

Requires macOS with Xcode's Metal toolchain (`xcodebuild -downloadComponent MetalToolchain` if not already installed) for the GPU targets; the CPU-only oracle also builds and runs on Linux.

```bash
cmake -S . -B build
cmake --build build -j
```

Optional Parasail cross-check/baseline (requires `brew install brewsci/bio/parasail` — see `brew tap brewsci/bio` first, it's not in homebrew-core):

```bash
cmake -S . -B build -DMETALSW_USE_PARASAIL=ON -DCMAKE_PREFIX_PATH="$(brew --prefix parasail)"
cmake --build build -j
```

Binaries land in `build/`: `metalsw_baseline` (CPU oracle, always built), `metalsw_gpu`/`metalsw_bench` (Metal targets, macOS only), `metalsw_parasail_bench` (only when `-DMETALSW_USE_PARASAIL=ON`).

## Run

```bash
# GPU: score + rank, print top N
./metalsw_gpu <query.fasta> <db.fasta> [topN]

# CPU oracle, same interface, optional repeat count for timing
./metalsw_baseline <query.fasta> <db.fasta> [topN] [--repeat N]

# GPU benchmark harness: warm-up + measured iterations, mean±stddev GCUPS, per-iteration CSV
./metalsw_bench <query.fasta> <db.fasta> [measured_iters] [metallib_path]

# Parasail CPU baseline, same methodology/report format as metalsw_bench
./metalsw_parasail_bench <query.fasta> <db.fasta> [measured_iters]
```

Power/thermal protocol (needs root for `powermetrics`):

```bash
cd build && sudo ../scripts/run_thermal_power_protocol.sh [query.fasta] [db.fasta] [iterations]
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Kernel language | Metal Shading Language (MSL) | Only GPU-compute language Apple Silicon supports |
| Host language | C++17 | Matches metal-cpp, no Objective-C required |
| GPU binding | metal-cpp (Apple's header-only C++ wrapper) | Direct Metal API access without Objective-C |
| Build system | CMake | Configures on Linux too (CPU oracle only; GPU targets are `if(APPLE)`-gated) |
| Correctness oracle | Hand-written scalar SW (Gotoh recurrence), in-repo | No external dependency required to prove correctness |
| CPU baseline (optional) | Parasail (striped SIMD, ARM NEON on Apple CPU) | Real SIMD throughput number, not scalar, for an honest comparison |
| Substitution matrix | BLOSUM62 | Standard for protein alignment |
| Benchmarking metric | GCUPS (giga cell updates per second) | Standard unit in the GPU-aligner literature |

## Project history and methodology notes

`PROJECT_BLUEPRINT.md` in this repo is the full working log: every stage gate, every measurement, every dead end (buffer-reuse not helping, an inverted reading of an early int8-overflow stat that got corrected, an earlier draft of this README's prior-art section that cited two projects which couldn't be re-verified and were dropped). It's kept append-only on purpose — corrections are added, not silently edited in, so the record of what was tried and what turned out to be wrong stays visible.
