#!/usr/bin/env bash
# v0.4 thermal + power protocol for metalsw on the target M2 Air (fanless, no
# active cooling). Runs metalsw_bench for a sustained period (long enough to
# actually reach thermal steady state on a fanless chip, not just the
# harness's built-in 20-iteration warm-up) while sampling powermetrics
# alongside it, so per-iteration GCUPS (results/*_iterations.csv) can be
# correlated against CPU/GPU power draw over the same wall-clock window.
#
# Must be run with sudo (powermetrics requires root). Run from the build
# directory that contains metalsw_bench, e.g.:
#   cd build_mac && sudo ../scripts/run_thermal_power_protocol.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "must run as root (powermetrics requires it) -- try: sudo $0" >&2
    exit 1
fi

QUERY="${1:-../data/query_hemoglobin.fasta}"
DB="${2:-../data/swissprot_subset.fasta}"
ITERS="${3:-2000}"   # ~2000 iterations x ~0.1s/iter =~ 3-4 min sustained load
TAG=$(date +%Y%m%d_%H%M%S)

mkdir -p results
POWER_LOG="results/powermetrics_${TAG}.txt"
THERM_BEFORE="results/therm_before_${TAG}.txt"
THERM_AFTER="results/therm_after_${TAG}.txt"

echo "thermal state before run:" | tee "$THERM_BEFORE"
pmset -g therm | tee -a "$THERM_BEFORE"

echo "starting powermetrics sampling (1s interval) -> $POWER_LOG"
powermetrics -i 1000 --samplers cpu_power,gpu_power -o "$POWER_LOG" &
PM_PID=$!

# give powermetrics a moment to attach before load starts
sleep 1

echo "running metalsw_bench: $ITERS measured iterations (this is the sustained load)"
# Drop root for the benchmark itself so it isn't running as root unnecessarily.
sudo -u "${SUDO_USER:-$(id -un)}" ./metalsw_bench "$QUERY" "$DB" "$ITERS"

kill "$PM_PID" 2>/dev/null || true
wait "$PM_PID" 2>/dev/null || true

echo "thermal state after run:" | tee "$THERM_AFTER"
pmset -g therm | tee -a "$THERM_AFTER"

LATEST_CSV=$(ls -t results/benchmark_*_iterations.csv | head -1)
echo
echo "done. Correlate $LATEST_CSV (per-iteration GCUPS + elapsed"
echo "wall time) against $POWER_LOG (per-second CPU/GPU power draw) for the writeup."
