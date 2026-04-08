"""
Progressive PDFs: how distributions evolve across the piControl run.

Each segment (~80 years) gets its own PDF curve, so distributional drift
is visible without any sub-segment splitting.

Usage:
    python progressive_pdfs.py                # uses defaults from config.py
    python progressive_pdfs.py --overwrite    # recompute cached histograms
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import get_6h_segment_dirs

# ── Configuration ──────────────────────────────────────────────────────────

OUTDIR = Path("figs_progressive_pdfs")
YEARS_PER_SEGMENT = 80
START_YEAR = 401  # first segment starts at model year 401

# Variables: (file_suffix, var_name, unit, log_x, bins)
VARIABLES = {
    "TS": ("TS", "TS", "K", False, np.linspace(190, 330, 501)),
    "T7": ("T7", "T_7", "K", False, np.linspace(190, 330, 501)),
    "PS": ("PS", "PS", "hPa", False, np.linspace(480, 1080, 501)),
    "precip": (
        "surface_precipitation_rate",
        "surface_precipitation_rate",
        "mm/day",
        True,
        np.concatenate([np.array([0.0]), np.geomspace(0.01, 500, 500)]),
    ),
    "wind_speed": (None, None, "m/s", False, np.linspace(0, 60, 501)),
}

TIMESTEPS_PER_YEAR = 365 * 4  # 6-hourly


# ── Helpers ────────────────────────────────────────────────────────────────

def _ts(msg):
    """Print with timestamp."""
    elapsed = time.time() - _ts.t0
    h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"  [{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)

_ts.t0 = time.time()


def load_variable(seg_dir, varkey):
    """Load a single variable from a segment, return numpy array (time, lat, lon)."""
    if varkey == "wind_speed":
        fu = os.path.join(seg_dir, "6h_surface_U7_predictions.nc")
        fv = os.path.join(seg_dir, "6h_surface_V7_predictions.nc")
        u = xr.open_dataset(fu, decode_timedelta=False)["U_7"].values[0]
        v = xr.open_dataset(fv, decode_timedelta=False)["V_7"].values[0]
        return np.sqrt(u**2 + v**2)

    file_suffix, var_name, _, _, _ = VARIABLES[varkey]
    fpath = os.path.join(seg_dir, f"6h_surface_{file_suffix}_predictions.nc")
    ds = xr.open_dataset(fpath, decode_timedelta=False)
    data = ds[var_name].values[0]  # (time, lat, lon)

    if varkey == "PS":
        data = data / 100.0  # Pa → hPa
    elif varkey == "precip":
        data = data * 86400.0  # kg/m²/s → mm/day

    return data


def compute_segment_histograms(seg_dirs):
    """
    Compute one histogram per segment per variable.

    Returns: {varkey: [(year_start, year_end, bin_edges, counts), ...]}
    """
    results = {k: [] for k in VARIABLES}

    for seg_i, seg_dir in enumerate(seg_dirs):
        _ts(f"Segment {seg_i:04d}: {seg_dir}")

        yr_start = START_YEAR + seg_i * YEARS_PER_SEGMENT

        for varkey in VARIABLES:
            _, _, _, _, bins = VARIABLES[varkey]

            data = load_variable(seg_dir, varkey)
            seg_years = data.shape[0] / TIMESTEPS_PER_YEAR
            yr_end = yr_start + seg_years

            flat = data.ravel()
            flat = flat[np.isfinite(flat)]
            counts, _ = np.histogram(flat, bins=bins)
            results[varkey].append((yr_start, yr_end, bins.copy(), counts))

        _ts(f"  Years {yr_start}–{yr_start + seg_years:.0f}")

    return results


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_progressive_pdf(varkey, segments, outdir):
    """Plot overlaid PDFs for one variable, one curve per segment."""
    _, _, unit, log_x, _ = VARIABLES[varkey]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = cm.get_cmap("coolwarm", len(segments))

    for i, (yr0, yr1, bins, counts) in enumerate(segments):
        widths = np.diff(bins)
        density = counts / (counts.sum() * widths)
        centres = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(centres, density, color=cmap(i), lw=1.2,
                label=f"yr {yr0:.0f}–{yr1:.0f}")

    ax.set_xlabel(f"{varkey} ({unit})", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(f"PDF by 80-yr segment — {varkey}", fontsize=14)
    ax.legend(fontsize=7, title="Segment", ncol=2, loc="upper right")

    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(left=0.01)
    if varkey == "precip":
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    outpath = outdir / f"progressive_pdf_{varkey}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    _ts(f"  → {outpath}")


def plot_pdf_difference(varkey, segments, outdir):
    """Plot PDF difference of each segment vs the last segment."""
    _, _, unit, log_x, _ = VARIABLES[varkey]

    if len(segments) < 2:
        return

    # Reference = last segment
    ref_yr0, ref_yr1, ref_bins, ref_counts = segments[-1]
    ref_widths = np.diff(ref_bins)
    ref_density = ref_counts / (ref_counts.sum() * ref_widths)
    centres = 0.5 * (ref_bins[:-1] + ref_bins[1:])

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = cm.get_cmap("coolwarm", len(segments) - 1)

    for i, (yr0, yr1, bins, counts) in enumerate(segments[:-1]):
        widths = np.diff(bins)
        density = counts / (counts.sum() * widths)
        diff = density - ref_density
        ax.plot(centres, diff, color=cmap(i), alpha=0.8, lw=1.0,
                label=f"yr {yr0:.0f}–{yr1:.0f}")

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel(f"{varkey} ({unit})", fontsize=12)
    ax.set_ylabel("Density difference", fontsize=12)
    ax.set_title(f"PDF difference vs. yr {ref_yr0:.0f}–{ref_yr1:.0f} — {varkey}",
                 fontsize=14)
    ax.legend(fontsize=7, ncol=2)

    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(left=0.01)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    outpath = outdir / f"progressive_pdf_diff_{varkey}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    _ts(f"  → {outpath}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Progressive PDF analysis")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute cached histograms")
    args = parser.parse_args()

    cache_path = Path("cache_progressive_pdfs.npz")
    OUTDIR.mkdir(exist_ok=True)

    seg_dirs = get_6h_segment_dirs()
    print(f"Found {len(seg_dirs)} segments (~{YEARS_PER_SEGMENT} yr each)")
    print(f"Variables: {list(VARIABLES.keys())}")
    print()

    if cache_path.exists() and not args.overwrite:
        _ts(f"Loading cached histograms from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        results = cached["results"].item()
    else:
        _ts("Computing histograms from 6h data …")
        results = compute_segment_histograms(seg_dirs)
        np.savez(cache_path, results=results)
        _ts(f"Cached to {cache_path}")

    print()
    _ts("Generating figures …")
    for varkey in VARIABLES:
        if varkey in results and results[varkey]:
            plot_progressive_pdf(varkey, results[varkey], OUTDIR)
            plot_pdf_difference(varkey, results[varkey], OUTDIR)

    n_figs = len(list(OUTDIR.glob("*.png")))
    print(f"\nDone — saved {n_figs} PNGs to {OUTDIR}/")


if __name__ == "__main__":
    main()
