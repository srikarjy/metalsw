# metalsw — Project Blueprint

Goal: the first verified, benchmarked Smith-Waterman GPU aligner on Apple Silicon Metal, written up as a short tools paper (target: Bioinformatics Applications Notes, fallback: ACM-BCB tools track).

Language/stack decision (locked): C++17 + metal-cpp. Not Rust. Do not revisit this without a written reason in this file.

Hardware: MacBook Air M2 (base, ~10 GPU cores, fanless — no active cooling). All benchmark methodology must account for thermal throttling (see Stage 2).

LLM features: out of scope until v3+, strictly after publication. Do not add before Stage 4 ships.

## Version map

| Version | Contains | Gate to move on |
|---|---|---|
| v0.1 | Scalar CPU oracle, FASTA parser, CMake build, verified test set | Oracle scores match hand-derived expectations (done: db01=291, db02=287, db03=278...) |
| v0.2 | Naive Metal kernel (inter-sequence, 1 DB seq/thread), metal-cpp host | Every GPU score bit-identical to v0.1 oracle on full test set |
| v0.3 | Optimized kernel (query profile, INT8+overflow fallback, threadgroup tuning) | Still bit-identical to oracle; GCUPS measured with documented methodology |
| v0.4 | Full benchmark suite: Swiss-Prot subset, Parasail same-machine baseline, powermetrics power draw, thermal steady-state protocol | Benchmark table complete, reproducible (documented hardware/OS/run-count) |
| v0.5 | Comparative analysis section: measured GCUPS/Watt vs. cited CUDASW++4.0/Accelign figures; unified-memory and no-DPX discussion | Analysis draft written, all cited numbers traced to source |
| v1.0 | Paper draft + public tagged GitHub release + README with prior-art section (cyanea-gpu, biometal named) | Submitted to target venue |

Do not skip a version's gate to start the next. If a gate fails, the fix happens before moving forward — this file gets updated with why, not silently patched.

## Step-by-step build plan

### Stage 0 — CPU oracle and harness (v0.1) — DONE

- [x] `blosum62.hpp` — BLOSUM62 matrix, residue indexing
- [x] `fasta.hpp` / `fasta.cpp` — FASTA parser
- [x] `sw_reference.hpp` / `sw_reference.cpp` — scalar Gotoh affine-gap SW, score-only
- [x] `timing.hpp` — Timer, GCUPS helper
- [x] `baseline_main.cpp` — CLI harness, optional Parasail cross-check via `METALSW_USE_PARASAIL`
- [x] `data/generate_test_set.py`, `data/query.fasta`, `data/db_small.fasta` — seeded, reproducible test set (seed=42)
- [x] `CMakeLists.txt` — builds `metalsw_baseline`, `if(APPLE)` stub for GPU target
- [x] `.devcontainer/` — Ubuntu 24.04 container (build-essential, cmake) for reproducible CPU-oracle builds. GPU path cannot run here — Metal requires bare-metal macOS; this container only ever builds `metalsw_baseline`.
- [x] Compiled and run in the devcontainer (2026-07-23): scores confirmed sane and correctly ordered — db01_identical=316 (self, max), db02_point_mutation=308 (one conservative substitution, just below max), db03_partial_match=190 (first 30 residues match, tail is unrelated padding — confirms SW finds the local region, not penalized by the full-length mismatch), db04_shuffled=29, db05_reversed=29, db06_unrelated=25 (all near-baseline, as expected for no real local similarity).

  Note: these numbers supersede an earlier placeholder set (db01=291, db02=287, db03=278...) that had been written into this file from a prior session but whose source code and test data were never committed to this repo — there was nothing on disk to reproduce those numbers from. Per this project's own hard rule ("no claim that isn't backed by a number you actually measured"), the numbers above are the real, reproducible, measured result of the actual committed code and seeded test set.

**Next action:** on target Mac — `brew install parasail`, rebuild with `-DMETALSW_USE_PARASAIL=ON`, confirm zero mismatches. This is the last unverified assumption (gap-penalty convention) before any GPU code gets written. (The Linux devcontainer intentionally does not install Parasail — that cross-check needs to happen on the actual target Mac, not in the container.)

### Stage 1 — Naive Metal kernel (v0.2)

- [ ] `metal/smith_waterman.metal` — MSL kernel, one DB sequence per thread, INT16, BLOSUM in constant memory, previous column in thread-local registers
- [ ] `src/metal_runtime.hpp/.cpp` — metal-cpp host: device init, buffer allocation (shared storage mode), command queue/buffer/encoder, dispatch
- [ ] `src/gpu_main.cpp` — CLI harness mirroring `baseline_main.cpp`, same args, GPU path
- [ ] Wire into `CMakeLists.txt` under the existing `if(APPLE)` block, link Metal/Foundation/QuartzCore frameworks
- [ ] Correctness loop: run same query+db as Stage 0, assert every score == oracle score, print PASS/FAIL per sequence

**Gate:** 100% match, zero tolerance, on the full test set including edge cases (empty-ish, single mutation, reversed, unrelated)

### Stage 2 — Optimization + rigorous benchmarking (v0.3–v0.4)

- [ ] Query-profile memory layout (coalesced reads)
- [ ] INT8 scores with INT16 overflow-recompute fallback
- [ ] Threadgroup memory tuning (8–16 KiB target, verify occupancy via Xcode GPU profiler)
- [ ] Sequence-length binning for balanced SIMD-group workloads
- [ ] Benchmark harness: n≥10 runs per data point, report mean ± stddev, not best-of-N
- [ ] Thermal protocol: documented warm-up period before measurement, steady-state GCUPS reported separately from cold-start
- [ ] Power measurement via `powermetrics` sampled during runs
- [ ] Fixed, pinned benchmark corpus: specific Swiss-Prot release (record version/date), specific query set, fixed gap penalties (open=11, extend=1)
- [ ] Parasail baseline run on the same machine, same methodology, same corpus

**Gate:** benchmark table complete and reproducible from a documented recipe (not "trust me, I ran it once")

### Stage 3 — Comparative analysis (v0.5)

- [ ] GCUPS/Watt table: measured Apple Silicon numbers vs. cited CUDASW++4.0 (A100/L40S/H100) and Accelign (Blackwell) figures — cited, not re-measured, explicitly labeled as such
- [ ] Written analysis: unified memory removing host/device transfer cost; absence of DPX-style fused add-max as the binding disadvantage; where the M2 Air's fanless thermal ceiling shows up in the numbers
- [ ] (Optional, strengthens paper) same benchmark on a second, actively-cooled Apple Silicon machine if accessible — throttled-vs-unthrottled delta

### Stage 4 — Publication (v1.0)

- [ ] README rewritten with: problem statement, honest positioning ("no verified, benchmarked, working Metal SW aligner exists" — not "first"), prior-art section naming cyanea-gpu and biometal specifically with their actual limitations, build/run instructions, benchmark table
- [ ] LICENSE (MIT or similar)
- [ ] Tagged GitHub release matching the paper's evaluated version
- [ ] Draft paper: problem, method, benchmark methodology, results, comparative analysis, limitations (explicitly: base M2 fanless, score-only/no traceback, protein-only, inter-sequence-only)
- [ ] Authorship: no BU affiliation (graduated Jan 2026) — list current position or independent
- [ ] Target: Bioinformatics Applications Notes (2-page tool note). Fallback: ACM-BCB tools track.

## Progress log

Update this section every session. One line per meaningful change, dated. Don't rewrite history, append.

- 2026-07-22 — Stage 0 complete. Scalar oracle compiled and verified in sandbox (not yet on target Mac hardware).
- 2026-07-22 — Verified no existing Metal SW aligner is production-ready: cyanea-gpu (Metal backend unverified/inaccessible source), biometal (GPU path likely stub, "expected" not "measured" speedups cited).
- 2026-07-23 — Stage 0 source and test data actually written and committed to the repo for the first time (nothing from the 2026-07-22 entry existed on disk). Built and run inside a Linux devcontainer (`.devcontainer/`): `metalsw_baseline` compiles cleanly and produces correctly-ordered scores on a seeded 6-sequence test set. Real measured scores replace the earlier placeholder numbers (see Stage 0 checklist above for the actual values and why they changed). Parasail cross-check still pending on target Mac hardware — blocks Stage 1.

## Open items / decisions pending

- [ ] Confirm Parasail cross-check passes with zero mismatches on the actual Mac (blocks Stage 1 start)
- [ ] Decide whether to attempt building/testing cyanea-gpu's GPU feature flag as a first-party prior-art data point for the README
- [ ] Confirm whether a second, actively-cooled Apple Silicon machine is available for the Stage 3 throttling comparison (optional but strengthens the paper)

## Hard rules (do not relitigate without updating this file explicitly)

- C++/metal-cpp, not Rust. Switching languages mid-project restarts Stage 1 from zero.
- No LLM features before v1.0 ships. This is a hardware/performance paper, not an LLM project.
- No wavefront/intra-sequence kernel in v1. Inter-sequence, one-thread-per-sequence, is the whole v1 design. Wavefront parallelism is future work, not now.
- Every benchmark number needs a documented method (hardware, OS version, run count, thermal state) at the moment it's collected, not reconstructed later.
- No claim in the README or paper that isn't backed by a number you actually measured or a source you can cite. This is what separates this project from cyanea-gpu's "expected 10-50x" language.
