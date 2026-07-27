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

- [x] Parasail cross-check on target Mac (M2, 2026-07-23): `brew tap brewsci/bio && brew install brewsci/bio/parasail` (not in homebrew-core), rebuilt with `-DMETALSW_USE_PARASAIL=ON -DCMAKE_PREFIX_PATH="$(brew --prefix parasail)"`. Result: **0/6 mismatches** — the gap-penalty convention (open=11 inclusive of first gap position, extend=1 per additional position) matches Parasail's `parasail_sw` exactly. This was the last unverified assumption before GPU code. Gate cleared — Stage 1 (v0.2, naive Metal kernel) can start.

### Stage 1 — Naive Metal kernel (v0.2) — DONE

- [x] `metal/smith_waterman.metal` — MSL kernel, one DB sequence per thread, INT16 running scores, BLOSUM62 + residue index in `constant` buffers, DB sequences packed into one `device` buffer + offsets/lengths, per-thread `H`/`E`/`F` arrays sized to `MAX_QUERY_LEN` (512) — a near-line-for-line port of `sw_reference.cpp`'s recurrence
- [x] `src/metal_runtime.hpp/.cpp` — metal-cpp host: device init, `.metallib` load, compute pipeline state, buffers in shared storage mode (unified memory), command queue/buffer/encoder, `dispatchThreads`
- [x] `src/gpu_main.cpp` — CLI harness mirroring `baseline_main.cpp`'s args (`query.fasta db.fasta [topN] [metallib_path]`), runs GPU scores then cross-checks each against the CPU oracle, printing PASS/FAIL per sequence
- [x] Wired into `CMakeLists.txt` under the existing `if(APPLE)` block: metal-cpp vendored via `FetchContent` (pinned to commit `c9727bc`, bkaradzic/metal-cpp mirror), custom build step compiles `.metal` → `.air` → `.metallib` via `xcrun metal`/`xcrun metallib`, links Metal/Foundation/QuartzCore frameworks
- [x] Correctness loop run on target Mac (M2, 2026-07-23): **6/6 PASS, 0 mismatches** — db01_identical=316, db02_point_mutation=308, db03_partial_match=190, db04_shuffled=29, db05_reversed=29, db06_unrelated=25 — bit-identical to the Stage 0 oracle on every sequence, including all edge cases (identical, point mutation, partial/local match, shuffled, reversed, unrelated)

**Gate:** 100% match, zero tolerance, on the full test set including edge cases — **cleared**. Note: the Linux devcontainer from Stage 0 cannot build `metalsw_gpu` (Metal requires bare-metal macOS); it was used only to regression-test that the CMake changes didn't break `metalsw_baseline`.

Setup note: building this target required downloading Xcode's Metal Toolchain component (`xcodebuild -downloadComponent MetalToolchain`, ~688 MB) — it wasn't installed on this machine yet, and turned out to be needed. GCUPS numbers from this stage are informational only (dominated by per-run device/library setup overhead on this tiny 6-sequence test set) — Stage 2 owns rigorous, methodologically-sound benchmarking.

### Stage 2 — Optimization + rigorous benchmarking (v0.3–v0.4)

- [x] Query-profile memory layout — `buildQueryProfile()` in `include/blosum62.hpp`, precomputes `profile[residueIdx * queryLen + j]`; replaces the kernel's residue-index + 2D-matrix lookup with one indexed read per DP cell. Validated in the Linux devcontainer with a standalone test (`tests/test_query_profile.cpp`, 1944/1944 checks pass) before any kernel code changed.
- [x] INT8 scores with INT16 overflow-recompute fallback — `smith_waterman_score_int8` (new kernel, `metal/smith_waterman.metal`) runs H/E/F in `int8_t`, flags per-thread saturation; host (`src/metal_runtime.cpp`) dispatches it over all sequences first, then re-dispatches the existing `smith_waterman_score` (int16) kernel only over the flagged subset via filtered offset/length buffers, and merges results. Verified on target Mac: 3/6 test sequences overflowed int8 (the three scoring >127) and correctly fell back; the other 3 were computed directly by the int8 path — both paths exercised and correct.
- [~] Threadgroup memory tuning (8–16 KiB target, verify occupancy via Xcode GPU profiler) — partially done. Threadgroup size is now chosen from `threadExecutionWidth()` (32 on this M2) rather than just `maxTotalThreadsPerThreadgroup()` (1024). `staticThreadgroupMemoryLength` is currently 0 for both kernels (no explicit `threadgroup`-address-space buffers) — moving the query profile there was evaluated but not implemented: at `MAX_QUERY_LEN=512`, the full profile is 24 KiB, over budget without a chunked/streaming scheme, which is a bigger change than this pass scoped. Actual Xcode GPU Frame Capture / Instruments occupancy verification is an interactive GUI step this assistant can't drive headlessly — still open if you want to do it yourself.
- [x] Sequence-length binning for balanced SIMD-group workloads — `GpuRunner::run()` (`src/metal_runtime.cpp`) now sorts DB sequences by length before packing them into the dispatch buffer (cached alongside the other corpus buffers, only recomputed when the corpus changes), then unpermutes the output scores back to the caller's original order. Since consecutive grid positions land in the same threadgroup/SIMD group, this keeps sequence lengths similar within a group instead of dispatch order following arbitrary FASTA file order. Re-verified correctness on target Mac: 6/6 PASS on the small set, 0 mismatches against the CPU oracle across 750/5,000/20,000/50,000-sequence corpora and 3 additional query lengths. Measured effect (same hemoglobin-query benchmark as the sweep below): 750 seqs 0.349→0.407 GCUPS (+17%), 5,000 seqs 2.225→2.217 (flat, within noise), 20,000 seqs 3.379→5.224 GCUPS (+55%), 50,000 seqs 3.504→5.309 GCUPS (+51%) — binning helps most at larger corpora, where length variance within a threadgroup was previously large enough to cause real SIMD-divergence stalls.
- [x] Benchmark harness: n≥10 runs per data point, report mean ± stddev, not best-of-N — `metalsw_bench` (15 measured, 20 warm-up) and `metalsw_parasail_bench` (15 measured, 3 warm-up).
- [x] Thermal protocol: documented warm-up period before measurement, steady-state GCUPS reported separately from cold-start — both harnesses report `cold-start GCUPS (1st warm-up iteration)` separately from `steady-state GCUPS: mean±stddev`.
- [x] Power measurement via `powermetrics` sampled during runs — `scripts/run_thermal_power_protocol.sh` run on target Mac (M2 Air, 2026-07-27, user-run under sudo since the assistant can't hold the password prompt): 2,000 measured iterations (~182s sustained load) of `metalsw_bench` against the pinned `data/swissprot_subset.fasta` corpus, `powermetrics` sampled every 1s throughout (183 samples, `results/powermetrics_20260727_174444.txt`). Results: CPU power mean 414.6 mW (min 141.0, max 4087.0 -- brief spike, not sustained), GPU power mean 885.6 mW (min 627.0, max 1064.0), combined mean 1300.1 mW. `pmset -g therm` reported no thermal or performance warnings before or after the run. Per-iteration GCUPS stayed flat at 0.348-0.350 from iteration 0 through iteration 1999 (`results/benchmark_20260727_174447_iterations.csv`) -- no thermal-throttling drift detected over the full sustained run, and steady-state GCUPS (0.3476±0.0016 over 2,000 iterations) matches the earlier 15-iteration measurement (0.349±0.0017) almost exactly.
- [x] Fixed, pinned benchmark corpus: `data/swissprot_subset.fasta` (750 sequences, len<=2000, deterministic first-N-after-filter selection from Swiss-Prot release downloaded 2026-07-23, provenance recorded in `data/prepare_swissprot_subset.py`), query `data/query_hemoglobin.fasta`, gap penalties open=11/extend=1.
- [x] Parasail baseline run infrastructure: `metalsw_parasail_bench` (`src/parasail_bench_main.cpp`), built when `-DMETALSW_USE_PARASAIL=ON`. Same corpus, same gap penalties, same warm-up/measured-iteration/mean+-stddev methodology and results/*.txt+*.csv format as `metalsw_bench` (the GPU harness), so the two reports are directly comparable. Parasail scoring itself was factored out of `baseline_main.cpp` into shared `include/parasail_score.hpp` / `src/parasail_score.cpp` to avoid duplicating it.

**First full-corpus run (target Mac, 2026-07-24), query=hemoglobin (142 residues) vs. `swissprot_subset.fasta` (750 sequences, 222,978 total residues), gap open=11/extend=1, 15 measured iterations each:**

| Path | Steady-state GCUPS (mean±stddev) | Cold-start GCUPS |
|---|---|---|
| metalsw GPU (M2, int8-fast-path + int16 fallback) | 0.348 ± 0.0015 | 0.230 |
| Parasail (scalar CPU baseline) | 0.405 ± 0.0044 | 0.334 |

**Honest finding, not yet explained away:** Parasail's scalar CPU baseline currently beats the metalsw GPU kernel on this corpus. Correction (2026-07-24): an earlier version of this note had the int8 stat backwards — "0/750 sequences overflowed" means 0 sequences *fell back* to int16, i.e. the int8 fast path handled 100% of this corpus directly. Re-confirmed at 750/5,000/20,000/50,000 sequences (hemoglobin query): 0 overflow at every size — the int16 fallback kernel is never dispatched at all for a hemoglobin-vs-Swiss-Prot workload on this corpus.

**Buffer-reuse experiment (target Mac, 2026-07-24):** `GpuRunner` was rewritten to cache and reuse Metal buffers across `run()` calls (query-profile, DB-corpus, and output buffers persist and are only re-uploaded/reallocated when the query or corpus actually changes; fallback-pass buffers are pre-sized for the worst case) instead of allocating ~12 new buffers and releasing them every call. Correctness re-verified: 6/6 PASS on the small test set and 0 mismatches against the CPU oracle across the full 750-sequence corpus. **Result: no meaningful GCUPS change** (0.345 vs. 0.348 before, within noise) — buffer allocation was not the bottleneck. Per-iteration CSV timing is a flat ~90-93ms regardless of buffer caching, which is the actual signal: at 750 threads doing ~142-residue-query DP (microseconds of real compute), the fixed cost of `commandBuffer→encode→commit→waitUntilCompleted` per dispatch — a synchronous CPU/GPU round-trip — dominates total time. This reframes the next fix: it's not allocation overhead, it's dispatch-count/synchronization overhead. Candidates, not yet tried: batching more work per dispatch (larger corpus, or fusing int8+int16 passes), and/or overlapping dispatches instead of blocking on `waitUntilCompleted` each time. Sequence-length binning may also help SIMD-group efficiency but won't address the fixed per-dispatch floor.

**Full DB-size / query-length sweep (target Mac, 2026-07-24, pre-sequence-length-binning), 15 measured iterations each, gap open=11/extend=1, BLOSUM62.** Correctness re-verified first: 0 mismatches across 78,006 GPU-vs-CPU-oracle score comparisons (750/5,000/20,000/50,000-sequence corpora and 4 query lengths), on top of the existing 6/6 edge-case set. DB-size subsets and extra query lengths were built from the same pinned Swiss-Prot release (same URL/ETag as `data/prepare_swissprot_subset.py`'s recorded provenance) but generated in a scratch directory, not committed to the repo — only the 750-sequence corpus and hemoglobin query are checked in.

*DB-size sweep (query = hemoglobin, `HBA_HUMAN`, 142 residues):*

| DB size | Total residues | GPU GCUPS (mean±stddev) | GPU iter time | Parasail GCUPS (mean±stddev) | Parasail iter time | GPU/Parasail |
|---|---|---|---|---|---|---|
| 750 | 222,978 | 0.349 ± 0.0017 | 90.8 ms | 0.407 ± 0.0009 | 77.8 ms | 0.86x (GPU slower) |
| 5,000 | 1,850,501 | 2.225 ± 0.0037 | 118.1 ms | 0.370 ± 0.0295 | 715.3 ms | 6.02x |
| 20,000 | 8,023,783 | 3.379 ± 0.0621 | 337.3 ms | 0.393 ± 0.0067 | 2899.8 ms | 8.60x |
| 50,000 | 18,852,813 | 3.504 ± 0.0314 | 764.1 ms | 0.392 ± 0.0043 | 6827.6 ms | 8.93x |

*Query-length sweep (DB = pinned 750-sequence corpus, 222,978 residues):*

| Query length | GPU GCUPS (mean±stddev) | GPU iter time | Parasail GCUPS (mean±stddev) | Parasail iter time | GPU/Parasail |
|---|---|---|---|---|---|
| 60 | 0.337 ± 0.0088 | 39.7 ms | 0.395 ± 0.0095 | 33.9 ms | 0.85x |
| 142 (hemoglobin) | 0.349 ± 0.0017 | 90.8 ms | 0.407 ± 0.0009 | 77.8 ms | 0.86x |
| 256 | 0.305 ± 0.0006 | 187.0 ms | 0.390 ± 0.0154 | 146.7 ms | 0.78x |
| 502 | 0.266 ± 0.0059 | 420.9 ms | 0.398 ± 0.0022 | 281.1 ms | 0.67x |

**Conclusion (pre-binning):** there is no single "GPU is Nx faster/slower" number — it depends entirely on DB size. The GPU crosses over from losing to Parasail at 750 sequences to an 8.93x win at 50,000, consistent with fixed per-dispatch overhead (established above) being amortized over more parallel work as the corpus grows. int8 fast-path handled 100% of every DB size tested (0 overflow to int16 at 750/5,000/20,000/50,000). Longer queries reduce GPU GCUPS somewhat (0.349→0.266 GCUPS from 142→502 residues) while Parasail stays flatter — not yet root-caused. Hardware: Apple M2 (base, `Mac14,2` MacBook Air), 10-core GPU, 8 CPU cores (4P+4E), 8GB unified memory, macOS 26.5.2.

**Post-binning DB-size re-measurement (target Mac, 2026-07-27), same hemoglobin query, same corpora, 15 measured iterations, correctness re-verified (0 mismatches at every size):**

| DB size | GPU GCUPS pre-binning | GPU GCUPS post-binning | Change |
|---|---|---|---|
| 750 | 0.349 ± 0.0017 | 0.407 ± 0.0075 | +17% (now roughly matches Parasail's 0.407, not losing to it) |
| 5,000 | 2.225 ± 0.0037 | 2.217 ± 0.0062 | flat, within noise |
| 20,000 | 3.379 ± 0.0621 | 5.224 ± 0.0322 | +55% |
| 50,000 | 3.504 ± 0.0314 | 5.309 ± 0.0284 | +51% |

Binning helps most at the larger corpora, where length variance within a 32-wide SIMD group was apparently large enough to cause real divergence stalls (short-sequence threads idling while the longest sequence in their group finishes). At 750 and 5,000 the effect is smaller/flat, consistent with fixed per-dispatch overhead still dominating at that scale rather than compute-time divergence.

**Post-binning query-length re-measurement (target Mac, 2026-07-27), DB = pinned 750-sequence corpus, 15 measured iterations each:**

| Query length | GPU GCUPS pre-binning | GPU GCUPS post-binning | Change |
|---|---|---|---|
| 60 | 0.337 ± 0.0088 | 0.431 ± 0.0025 | +28% |
| 142 (hemoglobin) | 0.349 ± 0.0017 | 0.407 ± 0.0075 | +17% |
| 256 | 0.305 ± 0.0006 | 0.350 ± 0.0171 | +15% |
| 502 | 0.266 ± 0.0059 | 0.314 ± 0.0010 | +18% |

Binning helps consistently across all query lengths at this DB size (750 sequences), not just the largest corpora — the SIMD-divergence mechanism applies regardless of query length since it depends on DB sequence length variance within a threadgroup.

**Gate:** benchmark table complete and reproducible from a documented recipe (not "trust me, I ran it once")

**v0.3 correctness re-check (2026-07-23):** after the query-profile and INT8/fallback changes, `metalsw_gpu` still reports 6/6 PASS bit-identical to the CPU oracle on the full test set. Sequence-length binning and the full benchmark harness (v0.4) are separate, not-yet-started follow-up work.

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
- 2026-07-23 — Stage 0 source and test data actually written and committed to the repo for the first time (nothing from the 2026-07-22 entry existed on disk). Built and run inside a Linux devcontainer (`.devcontainer/`): `metalsw_baseline` compiles cleanly and produces correctly-ordered scores on a seeded 6-sequence test set. Real measured scores replace the earlier placeholder numbers (see Stage 0 checklist above for the actual values and why they changed).
- 2026-07-23 — Parasail cross-check run on the actual target Mac (M2, arm64, Darwin 25.5.0): 0/6 mismatches. Gap-penalty convention confirmed correct. Stage 0 gate fully cleared — ready to start Stage 1 (v0.2, naive Metal kernel).
- 2026-07-23 — Stage 1 (v0.2) implemented and gate cleared: naive Metal kernel (`metal/smith_waterman.metal`), metal-cpp host runtime, `metalsw_gpu` CLI, wired into CMake. Built and run on target Mac hardware — 6/6 PASS, bit-identical to CPU oracle on the full test set. Regression-checked the CMake changes in the Stage 0 Linux devcontainer first (CPU oracle build unaffected) before touching the Metal toolchain on the Mac.
- 2026-07-23 — Stage 2 part 1 (v0.3, kernel optimization) done: query-profile layout (validated in devcontainer first via `tests/test_query_profile.cpp`, then wired into both kernels), INT8-with-INT16-fallback scoring (new `smith_waterman_score_int8` kernel + host-side dispatch/merge logic), and threadExecutionWidth-based threadgroup sizing. Re-ran the correctness gate on target Mac: still 6/6 PASS. Threadgroup-memory occupancy tuning only partially done (see checklist note) — sequence-length binning and the full v0.4 benchmark suite are next.
- 2026-07-24 — Stage 2 benchmark infrastructure (v0.4) built: `GpuRunner` class hoists one-time Metal setup out of the timed path, `metalsw_bench` (GPU harness: warm-up + >=10 measured iterations, mean+-stddev GCUPS, per-iteration CSV, cold-start vs steady-state split), `metalsw_parasail_bench` (same methodology/corpus/report format for the Parasail CPU baseline), pinned Swiss-Prot subset corpus (`data/swissprot_subset.fasta`, provenance in `data/prepare_swissprot_subset.py`), and `scripts/run_thermal_power_protocol.sh` (powermetrics + `pmset -g therm` before/after, sustained-load run). Built and smoke-tested on target Mac against the small 6-sequence set: both new binaries run cleanly, `metalsw_baseline`'s 0/6-mismatch Parasail cross-check still passes after factoring `parasailScore` into shared `parasail_score.hpp/.cpp`.
- 2026-07-24 — Bug found and fixed before trusting any benchmark numbers: both `bench_main.cpp` and `parasail_bench_main.cpp` opened `results/*.txt`/`.csv` with `std::ofstream` but never created the `results/` directory and never checked whether `open()` succeeded — after an earlier `rm -rf build_mac/results`, both harnesses printed "wrote results/..." success messages while silently writing nothing. Fixed with `std::filesystem::create_directories("results")` plus an explicit `if (!stream) throw` in both files before ever trusting harness output again.
- 2026-07-24 — First full-corpus benchmark run (target Mac, hemoglobin query vs. 750-sequence Swiss-Prot subset, 15 measured iterations each): GPU steady-state 0.348±0.0015 GCUPS vs. Parasail steady-state 0.405±0.0044 GCUPS — **Parasail currently wins**. Recorded as an honest, unresolved finding in the Stage 2 checklist above, not smoothed over — likely causes are per-iteration Metal buffer alloc/release in `GpuRunner::run()` and/or too little parallel work (750 threads) to saturate the M2 GPU. Sequence-length binning, buffer reuse across iterations, and a larger corpus are the candidate fixes, not yet tried. (Note: this entry originally also claimed the int8 fast path was contributing nothing based on "0/750 overflowed" -- that reading was backwards, see the correction above; int8 was actually handling 100% of this corpus.)
- 2026-07-24 — Downloaded the actual pinned Swiss-Prot release (same URL/ETag as the committed provenance in `data/prepare_swissprot_subset.py`, 575,503 records, 572,462 with length<=2000) to build additional DB-size subsets (750/5,000/20,000/50,000 sequences, first-N-after-length-filter, same recipe) and three additional query lengths (60/256/502 residues, real Swiss-Prot entries) in the scratch directory for a full benchmark sweep, per an explicit request for resume-defensible numbers with no estimation. Correctness re-verified first: metalsw_baseline vs metalsw_gpu score diffs are 0/750, 0/5,000, 0/20,000, 0/50,000 mismatches (hemoglobin query) and 0 mismatches across all three new query lengths against the 750-sequence corpus -- 78,006 total sequence-score comparisons, all exact, on top of the existing 6/6 edge-case set. Then ran metalsw_bench and metalsw_parasail_bench (15 measured iterations each) across the full DB-size x query-length matrix. Key result: **GPU throughput scales sharply with DB size while Parasail's stays flat** -- GPU crosses over from losing to Parasail at 750 sequences (0.349 vs 0.407 GCUPS) to winning by ~8.9x at 50,000 sequences (3.504 vs 0.392 GCUPS), consistent with the buffer-reuse experiment's finding that fixed per-dispatch overhead, not compute, dominates at small corpus sizes. Also corrected the earlier int8-overflow misreading (see above): 0/N overflow at every DB size tested, i.e. the int8 fast path is doing 100% of the work, not 0%. Full numbers (hardware: Apple M2, 10-core GPU, 8GB) reported to the user directly; not yet written into a permanent results table in this file -- that's still open v0.4 work (also still open: sequence-length binning, powermetrics run).
- 2026-07-24 — Tried the buffer-reuse fix: rewrote `GpuRunner` (`src/metal_runtime.cpp`) to cache and reuse Metal buffers across `run()` calls, keyed on query/DB-corpus identity, instead of allocating and releasing ~12 buffers every call. Re-verified correctness on target Mac (6/6 PASS on the small set; 0 mismatches against the CPU oracle across the full 750-sequence corpus via a `metalsw_baseline`/`metalsw_gpu` diff). **Result: essentially no GCUPS change (0.345 vs. 0.348)** — buffer allocation was not the bottleneck. Per-iteration CSV shows a flat ~90-93ms regardless, pointing at fixed per-dispatch `commandBuffer`/encode/commit/`waitUntilCompleted` overhead (a synchronous GPU round-trip) as the real floor at this scale, not buffer setup. Updated the Stage 2 notes to reflect this — the next candidates are batching more work per dispatch or avoiding the synchronous wait, not further buffer optimization.
- 2026-07-27 — Ran the power/thermal protocol (`scripts/run_thermal_power_protocol.sh`) on target Mac under sudo (user-run, since the assistant can't hold an interactive password prompt): 2,000 measured `metalsw_bench` iterations (~182s sustained load) against the pinned 750-sequence corpus, `powermetrics` sampled every 1s. CPU power mean 414.6 mW, GPU power mean 885.6 mW, combined mean 1300.1 mW; no thermal/performance warnings before or after (`pmset -g therm`); GCUPS held flat at 0.348-0.350 across all 2,000 iterations with zero throttling drift. Steady-state GCUPS over 2,000 iterations (0.3476±0.0016) closely matches the earlier 15-iteration measurement (0.349±0.0017), confirming that result wasn't a short-run fluke. This closes the last open item in the Stage 2 (v0.4) checklist — remaining work before the stage gate is sequence-length binning (still not implemented) and folding the DB-size/query-length benchmark matrix from the 2026-07-24 sweep into a permanent results table.
- 2026-07-27 — Folded the 2026-07-24 DB-size/query-length sweep into permanent tables in this file (Stage 2 section above), then implemented sequence-length binning: `GpuRunner::run()` now sorts DB sequences by length before packing the dispatch buffer (cached with the corpus, unpermuted back to caller order on return) so threads in the same threadgroup/SIMD group have similar-length work instead of arbitrary FASTA-file order. Re-verified correctness on target Mac: 6/6 PASS plus 0 mismatches across 750/5,000/20,000/50,000-sequence corpora and 3 query lengths. Re-measured the DB-size sweep: 750 seqs +17% (0.349→0.407 GCUPS, now roughly matching Parasail instead of losing to it), 5,000 seqs flat, 20,000 seqs +55% (3.379→5.224), 50,000 seqs +51% (3.504→5.309). This closes the last unimplemented code item in the Stage 2 checklist — every Stage 2 checkbox is now `[x]` or `[~]` (the `[~]` being threadgroup-memory occupancy tuning, explicitly scoped as a bigger future change). Query-length sweep not yet re-measured post-binning.

## Open items / decisions pending

- [ ] Decide whether to attempt building/testing cyanea-gpu's GPU feature flag as a first-party prior-art data point for the README
- [ ] Confirm whether a second, actively-cooled Apple Silicon machine is available for the Stage 3 throttling comparison (optional but strengthens the paper)

## Hard rules (do not relitigate without updating this file explicitly)

- C++/metal-cpp, not Rust. Switching languages mid-project restarts Stage 1 from zero.
- No LLM features before v1.0 ships. This is a hardware/performance paper, not an LLM project.
- No wavefront/intra-sequence kernel in v1. Inter-sequence, one-thread-per-sequence, is the whole v1 design. Wavefront parallelism is future work, not now.
- Every benchmark number needs a documented method (hardware, OS version, run count, thermal state) at the moment it's collected, not reconstructed later.
- No claim in the README or paper that isn't backed by a number you actually measured or a source you can cite. This is what separates this project from cyanea-gpu's "expected 10-50x" language.
