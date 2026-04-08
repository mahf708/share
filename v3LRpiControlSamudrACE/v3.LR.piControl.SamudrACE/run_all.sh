#!/bin/bash
#
# Run all ACE2-EAMv3 piControl analysis scripts.
#
# Usage:
#   Interactive (login node, quick test):
#     bash run_all.sh
#
#   On a Perlmutter CPU node (recommended for 6h scripts):
#     salloc -A <account> -C cpu -q interactive -t 4:00:00 -N 1
#     bash run_all.sh --overwrite
#
#   As a SLURM batch job:
#     sbatch run_all.sh
#
# Environment variable overrides (edit below or export before running):
#   ACE_DATA_ROOT    — root of ACE2-EAMv3 output
#   ACE_SOLIN_PATH   — path to SOLIN forcing file
#   ACE_SPINUP_YEAR  — spinup cutoff year (default: 600)
#   ACE_NWORKERS     — parallel workers for 6h scripts (default: 4)
#   ACE_ENV          — micromamba environment name (default: xgns)
#
# ============================================================================
# SLURM settings (used only when submitted with sbatch)
# ============================================================================
#SBATCH -J ace-picontrol-analysis
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=run_all_%j.log
# #SBATCH -A <your_account>   # ← uncomment and set your NERSC account

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Micromamba environment
# If already inside an active conda/mamba env, just use python directly
# to avoid lock contention when launching parallel processes.
ENV="${ACE_ENV:-xgns}"
if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV" ]]; then
    RUN=""
else
    RUN="micromamba run -n $ENV"
fi

# Flags passed to all scripts (e.g., --overwrite)
FLAGS="${@}"

# ============================================================================
# Phase 1: Monthly scripts (fast, ~5-20 min each)
# These are independent — run all 3 in parallel.
# ============================================================================
echo "========================================"
echo "Phase 1: Monthly diagnostics (parallel)"
echo "========================================"

$RUN python general_diagnostics.py $FLAGS &
PID_GENERAL=$!

$RUN python low_freq_variability.py $FLAGS &
PID_LOWFREQ=$!

$RUN python cross_variable_monthly.py $FLAGS &
PID_CROSSVAR=$!

echo "  Launched: general_diagnostics (PID $PID_GENERAL)"
echo "  Launched: low_freq_variability (PID $PID_LOWFREQ)"
echo "  Launched: cross_variable_monthly (PID $PID_CROSSVAR)"

# Wait for all monthly scripts
wait $PID_GENERAL  && echo "  general_diagnostics: DONE" || echo "  general_diagnostics: FAILED"
wait $PID_LOWFREQ  && echo "  low_freq_variability: DONE" || echo "  low_freq_variability: FAILED"
wait $PID_CROSSVAR && echo "  cross_variable_monthly: DONE" || echo "  cross_variable_monthly: FAILED"

echo ""

# ============================================================================
# Phase 2: 6-hourly scripts (slow, ~1-4 hours each)
# These are I/O and memory heavy (~30 GB per segment).
# Run 2 at a time on a 256 GB node, or 1 at a time on smaller nodes.
#
# If cached statistics exist (from a previous run), the data loading is
# skipped and only plots are regenerated — this is fast (~1 min).
# ============================================================================
echo "========================================"
echo "Phase 2: 6-hourly diagnostics"
echo "========================================"

# Pair 1: extremes + distributions (both read TS, PS, precip — different stats)
$RUN python extremes_6h.py $FLAGS &
PID_EXTREMES=$!

$RUN python distributions_6h.py $FLAGS &
PID_DISTRIB=$!

echo "  Launched: extremes_6h (PID $PID_EXTREMES)"
echo "  Launched: distributions_6h (PID $PID_DISTRIB)"

wait $PID_EXTREMES && echo "  extremes_6h: DONE" || echo "  extremes_6h: FAILED"
wait $PID_DISTRIB  && echo "  distributions_6h: DONE" || echo "  distributions_6h: FAILED"

echo ""

# Pair 2: dynamics (reads PS, U_7, V_7 — heaviest memory usage)
echo "  Running dynamics_6h (solo — highest memory usage) …"
$RUN python dynamics_6h.py $FLAGS
echo "  dynamics_6h: DONE"

echo ""
echo "========================================"
echo "All scripts complete."
echo "========================================"
echo ""
echo "Output directories:"
ls -d figs_*/
echo ""
echo "Figure counts:"
for d in figs_*/; do
    echo "  $d: $(ls "$d"/*.png 2>/dev/null | wc -l) PNGs"
done
