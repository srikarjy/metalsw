# metalsw

## Problem we're solving

Every GPU-accelerated Smith-Waterman protein aligner that exists today — CUDASW++4.0, GASAL2, ADEPT, WFA-GPU — runs on CUDA only. If you own a Mac and want to run exact local sequence alignment on your GPU instead of falling back to CPU SIMD tools (Parasail, SSW), there is nothing. No Metal implementation exists.

That's the gap. **metalsw is a native Apple Silicon GPU implementation of exact Smith-Waterman protein database search** — the same problem CUDASW++4.0 solves, solved with Metal instead of CUDA, validated for correctness and benchmarked honestly against CPU SIMD on the same machine.

It is not trying to beat CUDASW++4.0's throughput. An H100 has hardware fused add-max instructions and ~50x the compute of a base M2 GPU. The problem being solved is narrower and real: **prove exact GPU-accelerated protein alignment is possible on Apple Silicon, with correctness and performance both measured and reported honestly.**

## What the tool does

Input: one protein query, one FASTA database of many sequences.
Output: top-N local alignment scores, ranked.
Method: exact Smith-Waterman, affine (Gotoh) gaps, BLOSUM62 substitution matrix, score only (no traceback in v1).

```
query.fasta ──┐
              ├──> [metalsw] ──> ranked list of (score, sequence_id)
db.fasta ─────┘
```

## Architecture

Three layers, each independently testable:

```
┌─────────────────────────────────────────────────────────┐
│  Host (C++17)                                            │
│  ─ parses FASTA (query + database)                       │
│  ─ builds BLOSUM62 lookup + query profile                │
│  ─ allocates buffers, dispatches GPU work                │
│  ─ collects scores, sorts, prints top-N                  │
└───────────────────────┬───────────────────────────────────┘
                         │  metal-cpp (Metal API bindings)
┌───────────────────────▼───────────────────────────────────┐
│  Metal API layer                                          │
│  MTLDevice → MTLCommandQueue → MTLCommandBuffer             │
│  → MTLComputeCommandEncoder → dispatch(threads, threadgroup)│
│  MTLBuffer (unified memory — no host/device copy)           │
└───────────────────────┬───────────────────────────────────┘
                         │  compiled .metallib
┌───────────────────────▼───────────────────────────────────┐
│  GPU kernel (Metal Shading Language)                       │
│  one database sequence per thread                          │
│  BLOSUM62 + query held in constant memory                   │
│  INT16 running scores, previous column in registers          │
│  simd_shuffle for any cross-lane cooperation                 │
└─────────────────────────────────────────────────────────────┘
```

**Data flow per run:**
1. Host reads query + database FASTA, builds index tables.
2. Query, BLOSUM62 matrix, and all database sequences copied into `MTLBuffer`s (shared storage mode — CPU and GPU both see this memory, no explicit transfer).
3. One kernel dispatch: grid size = number of database sequences, one thread per sequence.
4. Each thread runs the full SW-Gotoh recurrence for its assigned sequence against the query, tracking the running max score in a register.
5. Scores written back to an output buffer; host reads it directly (still unified memory), sorts, prints top-N.

**Correctness path (parallel, not sequential):** every score above is checked against the in-repo scalar CPU oracle (`sw_reference.cpp`, already built and verified) and, optionally, against Parasail. A run is only trusted once GPU scores == oracle scores exactly.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Kernel language | Metal Shading Language (MSL), C++14 dialect | Only GPU-compute language Apple Silicon supports |
| Host language | C++17 | Matches metal-cpp, no ObjC required |
| GPU binding | metal-cpp (Apple's official header-only C++ wrapper) | Direct Metal API access without writing Objective-C |
| Build system | CMake | Cross-platform-honest; configures on Linux for CI even though the GPU path is `if(APPLE)`-gated |
| Correctness oracle | Hand-written scalar SW (Gotoh recurrence), in-repo | No external dependency required to prove correctness |
| CPU baseline (optional) | Parasail (striped SIMD, ARM NEON on Apple CPU) | The throughput number the GPU kernel must be honestly compared against |
| Substitution matrix | BLOSUM62 (hard-coded, NCBI standard ordering) | Standard for protein alignment; matches CUDASW++4.0 and Parasail defaults |
| Test data | Hand-built FASTA set with known score ordering | Makes correctness checkable by eye, not just by diff |
| Benchmarking metric | GCUPS (giga cell updates per second) | Standard unit in the GPU-aligner literature (CUDASW++, Parasail, GASAL2 all report it) |

## What's deliberately NOT in the stack (v1)

- No traceback / alignment path reconstruction — score only.
- No DNA support — protein/BLOSUM62 only.
- No MLX, no Python — pure C++/Metal, for the systems-engineering story.
- No multi-GPU, no distributed anything — single Mac, single GPU.
- No DPX-style fused add-max — Apple hardware has no equivalent; implemented as separate add + max.

## Definition of done for v1

1. `metalsw_gpu` produces scores that exactly match `metalsw_baseline` (scalar oracle) on the full test set.
2. A benchmark table exists: GCUPS for scalar oracle, Parasail (CPU SIMD), and the Metal kernel, all measured on the same M2, same database (Swiss-Prot subset).
3. README states the honest result — including if Parasail wins on this hardware — with the reasoning for why, not just the numbers.
