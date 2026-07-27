# metalsw: a correctness-verified, benchmarked Smith-Waterman protein database search implementation for Apple Silicon GPUs

**srikar jy**, independent researcher

## Abstract

**Summary:** Published GPU-accelerated Smith-Waterman aligners (CUDASW++4.0, GASAL2, ADEPT, Accelign) target CUDA; no verified, benchmarked, working Metal implementation was found in a direct search of the published literature and GitHub. We present metalsw, an exact Smith-Waterman (Gotoh affine-gap) protein database search tool implemented in Metal for Apple Silicon GPUs, using an inter-sequence parallelization strategy (one GPU thread per database sequence) with an int8 fast-path/int16-overflow-fallback scoring scheme and length-sorted dispatch for SIMD-group balance. Correctness is verified against a scalar CPU oracle across 78,006 sequence-score comparisons (0 mismatches) on real Swiss-Prot data up to 50,000 sequences. On a base (fanless) Apple M2, throughput scales from parity with a SIMD CPU baseline (Parasail) at small database sizes to 13.5x faster at 50,000 sequences, with measured energy efficiency of 0.816 GCUPS/Watt — roughly 19x below cited data-center GPU figures for the same algorithm class, a gap attributable primarily to the absence of a DPX-equivalent fused add-max instruction on Apple's GPU ISA. All reported numbers are either directly measured on the evaluated hardware or cited to a verifiable source; no performance claim in this note is projected or estimated.

**Availability and implementation:** Source code (C++17/metal-cpp host, Metal Shading Language kernels, CMake build), benchmark corpus with recorded provenance, and the complete experimental log (including corrected mistakes) are available at https://github.com/srikarjy/metalsw under the MIT license.

## 1. Introduction

Exact local sequence alignment via Smith-Waterman (Smith and Waterman, 1981) is computationally expensive at database scale, motivating decades of hardware acceleration work. GPU implementations targeting NVIDIA CUDA are well established and actively developed: CUDASW++4.0 reports up to 5.71 TCUPS on an H100 (Schmidt et al., 2024), and Accelign reports up to 16.1 TCUPS on an RTX PRO 6000 Blackwell for global alignment (Kallenborn et al., 2025). We searched for an equivalent, verified implementation targeting Metal — the compute API for Apple Silicon GPUs, which ship in every current Mac and iPad — and found none that could be confirmed to work, be correctness-checked against a reference, and report measured (not projected) throughput. This is not a claim that no such project has ever existed privately or unpublished; it is a statement about what a direct search of GitHub and the published literature could verify as of 2026-07-27.

metalsw fills that specific, narrow gap: a working Metal Smith-Waterman implementation, correctness-verified against a scalar reference implementation, with performance measured and reported honestly — including the regimes where it does not outperform a CPU baseline.

## 2. Implementation

**Algorithm.** Exact Smith-Waterman with Gotoh affine gap penalties (gap open = 11, gap extend = 1) and the BLOSUM62 substitution matrix, following the standard convention where the open penalty is inclusive of the first gap position. Output is a local alignment score only; traceback/alignment-path reconstruction is out of scope for this version.

**Parallelization strategy.** One GPU thread per database sequence (inter-sequence parallelism); no intra-sequence (wavefront) parallelism is implemented. Each thread computes the full DP recurrence for its assigned sequence against a shared, host-precomputed query profile held in `constant` address space.

**Two-pass scoring.** A fast pass runs the recurrence in `int8_t` (`smith_waterman_score_int8`), 4x the per-thread register/array footprint of `int16_t`, flagging any thread whose running score would saturate the int8 range. A second pass re-dispatches only the flagged subset through an unmodified `int16_t` kernel (`smith_waterman_score`). On every real corpus evaluated in this work (up to 50,000 Swiss-Prot sequences), 0 sequences required the fallback pass — the int8 path handled 100% of the workload — though the fallback path is independently exercised and verified correct on hand-built high-scoring edge cases.

**Sequence-length binning.** Database sequences are sorted by length on the host before packing into the dispatch buffer, so consecutive grid positions (the same threadgroup/SIMD group) hold similarly-sized sequences. This addresses SIMD divergence: without it, a 32-wide SIMD group containing both short and long sequences stalls every thread until the longest sequence in the group finishes. This single change improved measured throughput by 17-55% depending on database size (Section 3), with no kernel-code change required.

**Host/device interface.** metal-cpp provides the Metal API bindings. All buffers use `MTLResourceStorageModeShared` (Apple Silicon's unified memory) — CPU writes into buffer contents are directly visible to the GPU with no explicit transfer step. A `GpuRunner` class amortizes one-time device/pipeline/library setup and caches per-corpus buffers across repeated calls with the same query/database, keyed on object identity.

## 3. Results

**Correctness.** GPU output was diffed exactly against a scalar CPU oracle (identical recurrence, sequential C++) across: a 6-case hand-built edge-case set (identical/point-mutation/partial-match/shuffled/reversed/unrelated sequences); a 750/5,000/20,000/50,000-sequence pinned Swiss-Prot subset (single query); and the 750-sequence subset against four additional query lengths (60-502 residues). **78,006 total score comparisons, 0 mismatches.** The oracle itself was independently cross-checked against Parasail (Daily, 2016) before any GPU code was written (0/6 mismatches), confirming the gap-penalty convention.

**Throughput.** Measured on an Apple M2 (base, non-Pro/Max, 10-core GPU, 8 CPU cores, 8GB unified memory, fanless MacBook Air), 15+ measured iterations per data point after a discarded warm-up phase, mean±stddev reported:

| DB size | GPU GCUPS | Parasail GCUPS (CPU SIMD) | Ratio |
|---|---|---|---|
| 750 | 0.407 ± 0.0075 | 0.407 ± 0.0009 | 1.0x |
| 5,000 | 2.217 ± 0.0062 | 0.370 ± 0.0295 | 6.0x |
| 20,000 | 5.224 ± 0.0322 | 0.393 ± 0.0067 | 13.3x |
| 50,000 | 5.309 ± 0.0284 | 0.392 ± 0.0043 | 13.5x |

GPU throughput does not exceed the CPU SIMD baseline at small database sizes; fixed per-dispatch overhead (command-buffer submission and synchronous completion) dominates until enough parallel work amortizes it. This was established directly, not assumed: a buffer-reuse experiment intended to reduce per-call allocation overhead produced no measurable throughput change, isolating dispatch/synchronization count — not memory allocation or host/device transfer — as the actual small-scale bottleneck.

**Power and energy efficiency.** At 50,000 sequences (480 measured iterations, ~244s sustained, `powermetrics` sampled at 1Hz): combined power draw 6.46W mean (CPU 0.66W + GPU 5.80W), yielding **0.816 GCUPS/Watt**. The GPU ran at ~99.9% hardware-active residency and maximum clock (1398MHz) for nearly the entire sustained run, with no thermal or performance-state warnings recorded before or after (`pmset -g therm`) and no measurable throughput degradation over the run — this fanless machine's passive cooling was not challenged at this power draw.

**Comparison to cited GPU figures.** metalsw's 0.816 GCUPS/Watt is compared, not benchmarked head-to-head, against cited figures for CUDASW++4.0 (Schmidt et al., 2024): 15.7 GCUPS/Watt on an H100 and 15.2 GCUPS/Watt on an L40S, both measured against a synthetic equal-length database at rated TDP (200-300W) using hardware DPX fused add-max instructions. Accelign (Kallenborn et al., 2025) reports peak throughput (9.1-16.1 TCUPS on an RTX PRO 6000 Blackwell) but no energy-efficiency figure. metalsw is approximately 19x less energy-efficient than the cited H100/L40S figures — expected, not a claim of competitiveness: those are 200-300W data-center accelerators with an ISA-level instruction this work's target hardware lacks, evaluated against a database constructed to remove length-imbalance overhead, versus metalsw's ~6.5W combined draw against real, length-varied protein data.

## 4. Discussion and limitations

Unified memory removes a transfer-cost concern that does not apply to this architecture in the first place, rather than providing a benchmarked speedup — Apple Silicon's shared CPU/GPU memory means there is no PCIe-style host/device copy for this workload to eliminate; the bottleneck that does exist (small-scale dispatch overhead, Section 3) is orthogonal to memory architecture. The absence of a DPX-equivalent fused add-max instruction on Apple's GPU ISA is a binding, architecture-level constraint on this algorithm's peak throughput, not a tuning gap addressable by further kernel optimization within this design.

**Limitations, stated explicitly:** score-only output, no traceback; protein sequences and BLOSUM62 only, no DNA/RNA scoring; inter-sequence parallelism only, no intra-sequence/wavefront strategy; query length capped at 512 residues; evaluated on exactly one machine (a base, fanless Apple M2) — no data on Pro/Max/Ultra chips or actively-cooled systems; and the int8 fast path, while independently verified correct, was never exercised by its int16 fallback on any real corpus tested in this work, so the fallback's real-world scoring-overflow behavior on datasets scoring routinely above 127 remains unmeasured.

## Availability

https://github.com/srikarjy/metalsw, MIT license. Tagged pre-publication snapshot: `v1.0-rc1`. `PROJECT_BLUEPRINT.md` in the repository is the complete, append-only working log of every stage gate, measurement, and correction made during this project, including two prior-art claims from an early planning note that could not be re-verified and are not repeated here.

## Acknowledgments

None.

## Funding

None.

## Conflict of interest

None declared.

## References

Daily, J. (2016) Parasail: SIMD C library for global, semi-global, and local pairwise sequence alignments. *BMC Bioinformatics*, 17, 81.

Kallenborn, F., Dabbaghie, F., Steinegger, M., Schmidt, B. (2025) Accelign: a GPU-based Library for Accelerating Pairwise Sequence Alignment. *bioRxiv* 2025.12.17.694868.

Schmidt, B., Kallenborn, F., Chacon, A., Hundt, C. (2024) CUDASW++4.0: ultra-fast GPU-based Smith-Waterman protein sequence database search. *BMC Bioinformatics*, 25, 342.

Smith, T.F., Waterman, M.S. (1981) Identification of common molecular subsequences. *Journal of Molecular Biology*, 147, 195-197.
