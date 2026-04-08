"""
Sea ice diagnostics for ACE2-EAMv3 piControl run.

Uses ocean_sea_ice_fraction and sea_ice_volume from the ocean monthly
prediction files. Produces:
  1. Arctic & Antarctic sea ice extent timeseries (monthly + annual)
  2. Arctic & Antarctic sea ice volume timeseries
  3. Seasonal cycle of SIE by century (do the seasons shift?)
  4. Seasonal cycle of SIV by century
  5. September Arctic & March Antarctic SIE timeseries (annual minima)
  6. Sea ice concentration maps: early vs late run (DJF & JJA)
  7. Sea ice concentration drift maps (late minus early)
  8. Sea ice extent interannual variability by century

Usage:
    python sea_ice_diagnostics.py
    python sea_ice_diagnostics.py --overwrite
"""

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import PICONTROL_DIR, SPINUP_CUTOFF_YEAR

# ── Configuration ──────────────────────────────────────────────────────────

OUTDIR = Path("figs_sea_ice")
START_YEAR = 401  # model year of first month
MONTHS_PER_YEAR = 12
R_EARTH = 6.371e6  # m


# ── Helpers ────────────────────────────────────────────────────────────────

def _ts(msg):
    elapsed = time.time() - _ts.t0
    h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"  [{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)

_ts.t0 = time.time()


def get_ocean_monthly_files():
    return sorted(glob.glob(
        str(PICONTROL_DIR / "seg_*" / "ocean" / "monthly_mean_predictions.nc")
    ))


def compute_cell_areas(lat):
    """Cell areas for a regular 1-degree lat-lon grid (m^2)."""
    dlat = np.deg2rad(np.abs(np.diff(lat).mean()))
    dlon = np.deg2rad(1.0)
    return R_EARTH**2 * np.cos(np.deg2rad(lat)) * dlat * dlon  # (nlat,)


def load_all_sea_ice(files):
    """Load SIC and SIV from all segments, concatenate along time.

    Each segment has 961 months (indices 0–960). Month 960 is a partial
    month (counts=1) that overlaps with the next segment. We keep months
    0–959 (exactly 960 = 80 years × 12 months) so the sequential month
    counter stays aligned with the calendar.
    """
    sic_list, siv_list = [], []

    for i, f in enumerate(files):
        _ts(f"Loading {os.path.basename(os.path.dirname(os.path.dirname(f)))}")
        ds = xr.open_dataset(f, decode_times=False)
        sic = ds["ocean_sea_ice_fraction"].values[0]  # (961, lat, lon)
        siv = ds["sea_ice_volume"].values[0]
        # Keep first 960 months (drop partial month 960)
        sic_list.append(sic[:960])
        siv_list.append(siv[:960])
        if i == 0:
            lat = ds["lat"].values
            lon = ds["lon"].values
        ds.close()

    sic_all = np.concatenate(sic_list, axis=0)
    siv_all = np.concatenate(siv_list, axis=0)
    return sic_all, siv_all, lat, lon


# ── Metric computation ────────────────────────────────────────────────────

def compute_sie_timeseries(sic, cell_area, lat, threshold=0.15):
    """Sea ice extent (million km^2) for Arctic (lat>0) and Antarctic (lat<0)."""
    arctic = lat > 0
    antarctic = lat < 0

    ntime = sic.shape[0]
    sie_arctic = np.zeros(ntime)
    sie_antarctic = np.zeros(ntime)

    for t in range(ntime):
        ice_mask = sic[t] > threshold
        # Arctic
        a = ice_mask[arctic, :] * cell_area[arctic, None]
        sie_arctic[t] = np.nansum(a)
        # Antarctic
        a = ice_mask[antarctic, :] * cell_area[antarctic, None]
        sie_antarctic[t] = np.nansum(a)

    return sie_arctic / 1e12, sie_antarctic / 1e12  # million km^2


def compute_siv_timeseries(siv, cell_area, lat):
    """Sea ice volume (10^3 km^3) for Arctic and Antarctic."""
    arctic = lat > 0
    antarctic = lat < 0

    ntime = siv.shape[0]
    siv_arctic = np.zeros(ntime)
    siv_antarctic = np.zeros(ntime)

    for t in range(ntime):
        # SIV is m (ice volume per unit area) → multiply by cell area → m^3
        siv_arctic[t] = np.nansum(siv[t, arctic, :] * cell_area[arctic, None])
        siv_antarctic[t] = np.nansum(siv[t, antarctic, :] * cell_area[antarctic, None])

    return siv_arctic / 1e12, siv_antarctic / 1e12  # 10^3 km^3


# ── Plotting ───────────────────────────────────────────────────────────────

def make_years_months(ntime):
    """Return arrays of year and month-of-year for each timestep."""
    years = np.array([START_YEAR + t // 12 for t in range(ntime)])
    months = np.array([t % 12 + 1 for t in range(ntime)])  # 1-12
    return years, months


def plot_sie_timeseries(sie_arctic, sie_antarctic, years, outdir):
    """1. Monthly SIE timeseries for both hemispheres."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(years + np.arange(len(sie_arctic)) % 12 / 12,
                 sie_arctic, lw=0.3, alpha=0.6, color="steelblue")
    # Annual mean
    n_full_years = len(sie_arctic) // 12
    annual_arctic = sie_arctic[:n_full_years * 12].reshape(-1, 12).mean(axis=1)
    yr_annual = np.arange(n_full_years) + START_YEAR
    axes[0].plot(yr_annual + 0.5, annual_arctic, lw=1.5, color="darkblue",
                 label="Annual mean")
    axes[0].set_ylabel("SIE (million km²)")
    axes[0].set_title("Arctic Sea Ice Extent")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(years + np.arange(len(sie_antarctic)) % 12 / 12,
                 sie_antarctic, lw=0.3, alpha=0.6, color="coral")
    annual_antarctic = sie_antarctic[:n_full_years * 12].reshape(-1, 12).mean(axis=1)
    axes[1].plot(yr_annual + 0.5, annual_antarctic, lw=1.5, color="darkred",
                 label="Annual mean")
    axes[1].set_ylabel("SIE (million km²)")
    axes[1].set_xlabel("Model year")
    axes[1].set_title("Antarctic Sea Ice Extent")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = outdir / "01_sie_timeseries.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    _ts(f"  → {outpath}")


def plot_siv_timeseries(siv_arctic, siv_antarctic, years, outdir):
    """2. Monthly SIV timeseries."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    t_frac = years + np.arange(len(siv_arctic)) % 12 / 12

    axes[0].plot(t_frac, siv_arctic, lw=0.3, alpha=0.6, color="steelblue")
    n_full_years = len(siv_arctic) // 12
    annual = siv_arctic[:n_full_years * 12].reshape(-1, 12).mean(axis=1)
    yr_annual = np.arange(n_full_years) + START_YEAR
    axes[0].plot(yr_annual + 0.5, annual, lw=1.5, color="darkblue",
                 label="Annual mean")
    axes[0].set_ylabel("SIV (10³ km³)")
    axes[0].set_title("Arctic Sea Ice Volume")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_frac, siv_antarctic, lw=0.3, alpha=0.6, color="coral")
    annual = siv_antarctic[:n_full_years * 12].reshape(-1, 12).mean(axis=1)
    axes[1].plot(yr_annual + 0.5, annual, lw=1.5, color="darkred",
                 label="Annual mean")
    axes[1].set_ylabel("SIV (10³ km³)")
    axes[1].set_xlabel("Model year")
    axes[1].set_title("Antarctic Sea Ice Volume")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = outdir / "02_siv_timeseries.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    _ts(f"  → {outpath}")


def plot_seasonal_cycle_by_century(sie_arctic, sie_antarctic, years, months, outdir):
    """3 & 4. Seasonal cycle grouped by century."""
    unique_years = np.unique(years)
    # Group into ~200-year chunks for cleaner plots
    chunk_size = 200
    year_min = unique_years[0]
    year_max = unique_years[-1]
    chunks = []
    y0 = year_min
    while y0 < year_max:
        y1 = min(y0 + chunk_size, year_max + 1)
        chunks.append((y0, y1))
        y0 = y1

    cmap = cm.get_cmap("coolwarm", len(chunks))

    for region, sie, label in [
        ("Arctic", sie_arctic, "arctic"),
        ("Antarctic", sie_antarctic, "antarctic"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ci, (y0, y1) in enumerate(chunks):
            mask = (years >= y0) & (years < y1)
            if mask.sum() == 0:
                continue
            monthly_means = np.zeros(12)
            monthly_counts = np.zeros(12)
            for m in range(1, 13):
                mm = mask & (months == m)
                if mm.sum() > 0:
                    monthly_means[m - 1] = sie[mm].mean()
                    monthly_counts[m - 1] = mm.sum()
            ax.plot(np.arange(1, 13), monthly_means, color=cmap(ci), lw=2,
                    marker="o", ms=4, label=f"yr {y0}–{y1-1}")

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J",
                            "J", "A", "S", "O", "N", "D"])
        ax.set_ylabel("SIE (million km²)")
        ax.set_title(f"{region} Seasonal Cycle of Sea Ice Extent")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        outpath = outdir / f"03_seasonal_cycle_sie_{label}.png"
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        _ts(f"  → {outpath}")


def plot_seasonal_cycle_siv_by_century(siv_arctic, siv_antarctic, years, months, outdir):
    """Seasonal cycle of SIV by century."""
    unique_years = np.unique(years)
    chunk_size = 200
    year_min, year_max = unique_years[0], unique_years[-1]
    chunks = []
    y0 = year_min
    while y0 < year_max:
        y1 = min(y0 + chunk_size, year_max + 1)
        chunks.append((y0, y1))
        y0 = y1

    cmap = cm.get_cmap("coolwarm", len(chunks))

    for region, siv, label in [
        ("Arctic", siv_arctic, "arctic"),
        ("Antarctic", siv_antarctic, "antarctic"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ci, (y0, y1) in enumerate(chunks):
            mask = (years >= y0) & (years < y1)
            if mask.sum() == 0:
                continue
            monthly_means = np.zeros(12)
            for m in range(1, 13):
                mm = mask & (months == m)
                if mm.sum() > 0:
                    monthly_means[m - 1] = siv[mm].mean()
            ax.plot(np.arange(1, 13), monthly_means, color=cmap(ci), lw=2,
                    marker="o", ms=4, label=f"yr {y0}–{y1-1}")

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J",
                            "J", "A", "S", "O", "N", "D"])
        ax.set_ylabel("SIV (10³ km³)")
        ax.set_title(f"{region} Seasonal Cycle of Sea Ice Volume")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        outpath = outdir / f"04_seasonal_cycle_siv_{label}.png"
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        _ts(f"  → {outpath}")


def plot_annual_min_sie(sie_arctic, sie_antarctic, months, outdir):
    """5. September Arctic & February Antarctic SIE (annual minima)."""
    n_full_years = len(sie_arctic) // 12
    yr = np.arange(n_full_years) + START_YEAR

    # Reshape to (nyears, 12)
    arctic_monthly = sie_arctic[:n_full_years * 12].reshape(-1, 12)
    antarctic_monthly = sie_antarctic[:n_full_years * 12].reshape(-1, 12)

    # September = month 9, index 8; March = month 3, index 2
    sept_arctic = arctic_monthly[:, 8]
    march_arctic = arctic_monthly[:, 2]
    feb_antarctic = antarctic_monthly[:, 1]
    sept_antarctic = antarctic_monthly[:, 8]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(yr, sept_arctic, lw=1, color="steelblue", alpha=0.7,
                 label="September (min)")
    axes[0].plot(yr, march_arctic, lw=1, color="navy", alpha=0.7,
                 label="March (max)")
    axes[0].set_ylabel("SIE (million km²)")
    axes[0].set_title("Arctic Sea Ice Extent — Seasonal Extremes")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(yr, feb_antarctic, lw=1, color="coral", alpha=0.7,
                 label="February (min)")
    axes[1].plot(yr, sept_antarctic, lw=1, color="darkred", alpha=0.7,
                 label="September (max)")
    axes[1].set_ylabel("SIE (million km²)")
    axes[1].set_xlabel("Model year")
    axes[1].set_title("Antarctic Sea Ice Extent — Seasonal Extremes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = outdir / "05_annual_extremes_sie.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    _ts(f"  → {outpath}")


def plot_sic_maps(sic, lat, lon, years, months, outdir):
    """6 & 7. SIC maps: early vs late, and drift maps."""
    # Define early (first 100 yr) and late (last 100 yr) periods
    unique_years = np.unique(years)
    early_mask = (years >= unique_years[0]) & (years < unique_years[0] + 100)
    late_mask = (years >= unique_years[-1] - 100) & (years <= unique_years[-1])

    for season, season_months, season_label in [
        ("DJF", [12, 1, 2], "djf"),
        ("JJA", [6, 7, 8], "jja"),
    ]:
        early_sel = early_mask & np.isin(months, season_months)
        late_sel = late_mask & np.isin(months, season_months)

        early_mean = np.nanmean(sic[early_sel], axis=0)
        late_mean = np.nanmean(sic[late_sel], axis=0)
        diff = late_mean - early_mean

        fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                                 subplot_kw={"projection": None})

        for ax, data, title, cmap_name, vmin, vmax in [
            (axes[0], early_mean, f"Early 100yr {season}", "Blues_r", 0, 1),
            (axes[1], late_mean, f"Late 100yr {season}", "Blues_r", 0, 1),
            (axes[2], diff, f"Late − Early {season}", "RdBu_r", -0.5, 0.5),
        ]:
            im = ax.pcolormesh(lon, lat, data, cmap=cmap_name,
                               vmin=vmin, vmax=vmax, shading="auto")
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(f"Sea Ice Concentration — {season}", fontsize=14, y=1.02)
        fig.tight_layout()

        outpath = outdir / f"06_sic_maps_{season_label}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _ts(f"  → {outpath}")


def plot_sie_variability_by_century(sie_arctic, sie_antarctic, years, months, outdir):
    """8. Interannual variability of SIE by century."""
    n_full_years = len(sie_arctic) // 12
    yr = np.arange(n_full_years) + START_YEAR

    # September Arctic, February Antarctic
    arctic_monthly = sie_arctic[:n_full_years * 12].reshape(-1, 12)
    antarctic_monthly = sie_antarctic[:n_full_years * 12].reshape(-1, 12)
    sept_arctic = arctic_monthly[:, 8]
    feb_antarctic = antarctic_monthly[:, 1]

    # Compute running 50-year std
    window = 50
    if len(yr) < window:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for ax, data, title, color in [
        (axes[0], sept_arctic, "Arctic September SIE — 50-yr running σ", "steelblue"),
        (axes[1], feb_antarctic, "Antarctic February SIE — 50-yr running σ", "coral"),
    ]:
        running_std = np.array([
            data[i:i + window].std() for i in range(len(data) - window + 1)
        ])
        yr_std = yr[:len(running_std)] + window // 2
        ax.plot(yr_std, running_std, lw=1.5, color=color)
        ax.set_ylabel("σ (million km²)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Model year")
    fig.tight_layout()

    outpath = outdir / "07_sie_variability.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    _ts(f"  → {outpath}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sea ice diagnostics")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_path = Path("cache_sea_ice.npz")
    OUTDIR.mkdir(exist_ok=True)

    files = get_ocean_monthly_files()
    print(f"Found {len(files)} ocean monthly files")
    print()

    if cache_path.exists() and not args.overwrite:
        _ts(f"Loading cached data from {cache_path}")
        cached = np.load(cache_path)
        sic = cached["sic"]
        siv = cached["siv"]
        lat = cached["lat"]
        lon = cached["lon"]
    else:
        _ts("Loading sea ice data …")
        sic, siv, lat, lon = load_all_sea_ice(files)
        _ts(f"Total months: {sic.shape[0]} ({sic.shape[0]/12:.0f} years)")
        _ts("Caching …")
        np.savez(cache_path, sic=sic, siv=siv, lat=lat, lon=lon)
        _ts(f"Cached to {cache_path}")

    cell_area = compute_cell_areas(lat)
    years, months = make_years_months(sic.shape[0])

    print()
    _ts("Computing sea ice extent …")
    sie_arctic, sie_antarctic = compute_sie_timeseries(sic, cell_area, lat)

    _ts("Computing sea ice volume …")
    siv_arctic, siv_antarctic = compute_siv_timeseries(siv, cell_area, lat)

    print()
    _ts("Generating figures …")

    plot_sie_timeseries(sie_arctic, sie_antarctic, years, OUTDIR)
    plot_siv_timeseries(siv_arctic, siv_antarctic, years, OUTDIR)
    plot_seasonal_cycle_by_century(sie_arctic, sie_antarctic, years, months, OUTDIR)
    plot_seasonal_cycle_siv_by_century(siv_arctic, siv_antarctic, years, months, OUTDIR)
    plot_annual_min_sie(sie_arctic, sie_antarctic, months, OUTDIR)
    plot_sic_maps(sic, lat, lon, years, months, OUTDIR)
    plot_sie_variability_by_century(sie_arctic, sie_antarctic, years, months, OUTDIR)

    n_figs = len(list(OUTDIR.glob("*.png")))
    print(f"\nDone — saved {n_figs} PNGs to {OUTDIR}/")


if __name__ == "__main__":
    main()
