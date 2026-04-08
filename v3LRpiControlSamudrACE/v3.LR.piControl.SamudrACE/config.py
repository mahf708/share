"""
Shared configuration for ACE2-EAMv3 piControl analysis scripts.

All analysis scripts import paths and parameters from this file.
Edit the defaults below when moving to a different machine, or
override at runtime via environment variables.

Scripts:
    general_diagnostics.py   — Monthly climatology, energy balance, vertical structure
    low_freq_variability.py  — Power spectra, AMO/PDO, Hurst exponent, wavelets
    cross_variable_monthly.py — ENSO teleconnections, P-E, albedo, Gregory plot, monsoons
    extremes_6h.py           — Block maxima, return levels, DTR, CC scaling (6-hourly)
    distributions_6h.py      — Temperature/precip PDFs, skewness/kurtosis maps (6-hourly)
    dynamics_6h.py           — Storm tracks, EKE, wind speed distributions (6-hourly)

Running:
    # Single script:
    micromamba run -n xgns python general_diagnostics.py --overwrite

    # All scripts (parallel where possible):
    bash run_all.sh --overwrite

    # On a different machine (override paths):
    ACE_DATA_ROOT=/new/path python general_diagnostics.py

    # As a SLURM batch job:
    sbatch run_all.sh

Environment variables:
    ACE_DATA_ROOT    Root of ACE2-EAMv3 output             (default: /pscratch/sd/m/mahf708/ACE2-EAMv3)
    ACE_SOLIN_PATH   Path to SOLIN forcing file             (default: see below)
    ACE_SPINUP_YEAR  Spinup cutoff year                     (default: 600)
    ACE_NWORKERS     Parallel workers for 6h scripts        (default: 4)
    ACE_CHUNK_SIZE   Dask chunk size for 6h data            (default: 1460 = 1 year)
    ACE_ENV          Micromamba env name for run_all.sh      (default: xgns)

Caching:
    The 6-hourly scripts (extremes, distributions, dynamics) cache their
    accumulated statistics to .npz files (cache_extremes_6h.npz, etc.).
    On re-run, if the cache exists, data processing is skipped and only
    plots are regenerated (~1 min). Use --overwrite to force recomputation.

Data layout expected:
    ACE_DATA_ROOT/
    ├── segment_01/atmosphere/monthly_mean_predictions.nc   (initial segment)
    └── picontrol_run/
        ├── seg_0000/atmosphere/
        │   ├── monthly_mean_predictions.nc
        │   ├── monthly_mean_target.nc            (SOLIN, PHIS, land_fraction)
        │   ├── enso_index_diagnostics.nc
        │   ├── 6h_surface_TS_predictions.nc
        │   ├── 6h_surface_PS_predictions.nc
        │   ├── 6h_surface_T7_predictions.nc
        │   ├── 6h_surface_U7_predictions.nc
        │   ├── 6h_surface_V7_predictions.nc
        │   └── 6h_surface_surface_precipitation_rate_predictions.nc
        ├── seg_0001/atmosphere/...
        └── ...
"""

import os
import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# Data paths — edit these when moving to a different machine
# ---------------------------------------------------------------------------

# Root directory containing the ACE2-EAMv3 run output
ACE_DATA_ROOT = Path(os.environ.get(
    "ACE_DATA_ROOT",
    "/lcrc/globalscratch/ac.ngmahfouz"
))

# Initial segment (before the piControl continuation)
INITIAL_SEGMENT_DIR = ACE_DATA_ROOT / "segment_01" / "atmosphere"

# piControl run directory (contains seg_0000, seg_0001, ...)
PICONTROL_DIR = ACE_DATA_ROOT / "picontrol_run"

# SOLIN forcing file
SOLIN_PATH = Path(os.environ.get(
    "ACE_SOLIN_PATH",
    "/home/ac.ngmahfouz/inference/2025-11-24-E3SMv3-piControl-100yr-coupled-IC/atmosphere-forcing-1yr.nc"
))

# ---------------------------------------------------------------------------
# Derived paths (built from the roots above)
# ---------------------------------------------------------------------------

def get_monthly_prediction_files():
    """Return sorted list of monthly mean prediction NetCDF files."""
    files = []
    init_file = INITIAL_SEGMENT_DIR / "monthly_mean_predictions.nc"
    if init_file.exists():
        files.append(str(init_file))
    files += sorted(glob.glob(
        str(PICONTROL_DIR / "seg_00*" / "atmosphere" / "monthly_mean_predictions.nc")
    ))
    return files


def get_enso_diagnostic_files():
    """Return sorted list of ENSO diagnostic NetCDF files."""
    return sorted(glob.glob(
        str(PICONTROL_DIR / "seg_00*" / "atmosphere" / "enso_index_diagnostics.nc")
    ))


def get_6h_segment_dirs():
    """Return sorted list of segment atmosphere directories with 6h data."""
    return sorted(glob.glob(
        str(PICONTROL_DIR / "seg_00*" / "atmosphere")
    ))


def get_land_fraction_path():
    """Return path to the land fraction target file."""
    return str(PICONTROL_DIR / "seg_0000" / "atmosphere" / "monthly_mean_target.nc")


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------

# Spinup cutoff: years before this are excluded from statistical analyses
SPINUP_CUTOFF_YEAR = int(os.environ.get("ACE_SPINUP_YEAR", "400"))

# ---------------------------------------------------------------------------
# HPC / parallelism settings
# ---------------------------------------------------------------------------

# Number of Dask workers for parallel segment processing (6h scripts)
# Set via environment variable or defaults to 4
# On a Perlmutter CPU node (256 GB, 128 cores): 4-8 workers is good
# On a GPU node (80 GB): use 2 workers
# Each worker loads ~30 GB per segment, so memory = N_WORKERS × 30 GB
N_WORKERS = int(os.environ.get("ACE_NWORKERS", "4"))

# Chunk size for dask arrays when loading 6h data
# Larger chunks = less overhead but more memory per chunk
# 1460 = 1 year of 6-hourly data (365 days × 4)
DASK_CHUNK_SIZE = int(os.environ.get("ACE_CHUNK_SIZE", "1460"))
