"""
ETCCDI climate extreme indices from ACE2-EAMv3 piControl 6-hourly output.

Resamples 6-hourly TS and precipitation to daily Tmax/Tmin/precip,
then computes standard ETCCDI indices using xclim. Results are saved
as NetCDF per segment and summary plots are generated.

Input:  6h_surface_TS_predictions.nc, 6h_surface_surface_precipitation_rate_predictions.nc
Output: etccdi_results/seg_XXXX_etccdi.nc  (per-segment annual indices)
        figs_etccdi/*.png                   (summary plots)

Usage:
    python compute_etccdi.py                    # all segments
    python compute_etccdi.py --n_segments 3     # first 3 segments (test)
    python compute_etccdi.py --plots_only       # skip computation, just plot
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

DATA_ROOT = Path("/lcrc/globalscratch/ac.ngmahfouz/picontrol_run")
RESULTS_DIR = Path("etccdi_results")
FIGS_DIR = Path("figs_etccdi")
START_YEAR = 401
STEPS_PER_DAY = 4  # 6-hourly


def _ts(msg):
    elapsed = time.time() - _ts.t0
    m, s = int(elapsed // 60), int(elapsed % 60)
    print(f"  [{m:02d}:{s:02d}] {msg}", flush=True)

_ts.t0 = time.time()


def resample_to_daily(seg_dir):
    """Load 6h TS and precip, resample to daily Tmax, Tmin, Tmean, Precip.

    Returns an xarray Dataset with CF-compliant variable names and a
    proper daily time coordinate.
    """
    import cftime

    seg_name = os.path.basename(os.path.dirname(seg_dir))
    seg_idx = int(seg_name.split("_")[1])
    year_offset = START_YEAR + seg_idx * 80

    # Open lazily — don't load full arrays
    ts_f = os.path.join(seg_dir, "6h_surface_TS_predictions.nc")
    ds_ts = xr.open_dataset(ts_f, decode_timedelta=False)
    lat = ds_ts["lat"].values
    lon = ds_ts["lon"].values
    ntime = ds_ts.dims["time"]
    nlat, nlon = len(lat), len(lon)

    pr_f = os.path.join(seg_dir, "6h_surface_surface_precipitation_rate_predictions.nc")
    ds_pr = xr.open_dataset(pr_f, decode_timedelta=False)

    # Process year by year to avoid OOM (~1.5 GB per variable per year)
    steps_per_year = 365 * STEPS_PER_DAY
    n_full_years = ntime // steps_per_year
    days_per_year = 365

    tasmax_list, tasmin_list, tas_list, pr_list = [], [], [], []

    for y in range(n_full_years):
        t0 = y * steps_per_year
        t1 = t0 + steps_per_year

        ts_yr = ds_ts["TS"].values[0, t0:t1].reshape(
            days_per_year, STEPS_PER_DAY, nlat, nlon)
        pr_yr = ds_pr["surface_precipitation_rate"].values[0, t0:t1].reshape(
            days_per_year, STEPS_PER_DAY, nlat, nlon)

        tasmax_list.append(np.nanmax(ts_yr, axis=1))
        tasmin_list.append(np.nanmin(ts_yr, axis=1))
        tas_list.append(np.nanmean(ts_yr, axis=1))
        pr_list.append(np.nanmean(pr_yr, axis=1) * 86400)  # mm/day

    ds_ts.close()
    ds_pr.close()

    if not tasmax_list:
        return None  # segment too short for any full year

    tasmax = np.concatenate(tasmax_list, axis=0)
    tasmin = np.concatenate(tasmin_list, axis=0)
    tas = np.concatenate(tas_list, axis=0)
    pr_daily = np.concatenate(pr_list, axis=0)
    n_days = tasmax.shape[0]

    times = xr.cftime_range(
        start=cftime.DatetimeNoLeap(year_offset, 1, 1),
        periods=n_days, freq="D", calendar="noleap"
    )

    ds = xr.Dataset(
        {
            "tasmax": (["time", "lat", "lon"], tasmax),
            "tasmin": (["time", "lat", "lon"], tasmin),
            "tas":    (["time", "lat", "lon"], tas),
            "pr":     (["time", "lat", "lon"], pr_daily),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )

    ds["tasmax"].attrs = {"units": "K", "cell_methods": "time: maximum"}
    ds["tasmin"].attrs = {"units": "K", "cell_methods": "time: minimum"}
    ds["tas"].attrs = {"units": "K", "cell_methods": "time: mean"}
    ds["pr"].attrs = {"units": "mm/day", "cell_methods": "time: sum"}

    return ds


def compute_etccdi_indices(ds_daily):
    """Compute ETCCDI indices from a daily dataset using xclim.

    Returns an xarray Dataset with annual index fields.
    """
    from xclim import atmos
    from xclim.core.units import convert_units_to

    tasmax = ds_daily["tasmax"]
    tasmin = ds_daily["tasmin"]
    tas = ds_daily["tas"]
    pr = ds_daily["pr"]

    results = {}

    # ── Temperature indices ───────────────────────────────────────────────

    # TXx: Annual max of daily max temperature
    _ts("    TXx …")
    results["TXx"] = atmos.tx_max(tasmax=tasmax, freq="YS")

    # TNn: Annual min of daily min temperature
    _ts("    TNn …")
    results["TNn"] = atmos.tn_min(tasmin=tasmin, freq="YS")

    # TXn: Annual min of daily max temperature
    _ts("    TXn …")
    results["TXn"] = atmos.tx_min(tasmax=tasmax, freq="YS")

    # TNx: Annual max of daily min temperature
    _ts("    TNx …")
    results["TNx"] = atmos.tn_max(tasmin=tasmin, freq="YS")

    # DTR: Mean diurnal temperature range
    _ts("    DTR …")
    results["DTR"] = atmos.daily_temperature_range(tasmax=tasmax, tasmin=tasmin,
                                                    freq="YS")

    # FD: Frost days (Tmin < 0°C)
    _ts("    FD …")
    results["FD"] = atmos.frost_days(tasmin=tasmin, freq="YS")

    # SU: Summer days (Tmax > 25°C)
    _ts("    SU …")
    results["SU"] = atmos.tx_days_above(tasmax=tasmax, thresh="25 degC", freq="YS")

    # ID: Ice days (Tmax < 0°C)
    _ts("    ID …")
    results["ID"] = atmos.ice_days(tasmax=tasmax, freq="YS")

    # TR: Tropical nights (Tmin > 20°C)
    _ts("    TR …")
    results["TR"] = atmos.tropical_nights(tasmin=tasmin, freq="YS")

    # GSL: Growing season length
    _ts("    GSL …")
    try:
        results["GSL"] = atmos.growing_season_length(tas=tas, freq="YS")
    except Exception as e:
        _ts(f"    GSL skipped: {e}")

    # ── Precipitation indices ─────────────────────────────────────────────

    # Convert pr to kg/m²/s for xclim (it expects flux)
    pr_flux = pr / 86400.0  # mm/day → kg/m²/s
    pr_flux.attrs["units"] = "kg m-2 s-1"
    pr_flux.attrs["cell_methods"] = "time: mean"

    # RX1day: Max 1-day precipitation
    _ts("    RX1day …")
    results["RX1day"] = atmos.max_1day_precipitation_amount(pr=pr_flux, freq="YS")

    # RX5day: Max 5-day precipitation
    _ts("    RX5day …")
    results["RX5day"] = atmos.max_n_day_precipitation_amount(pr=pr_flux, window=5,
                                                              freq="YS")

    # SDII: Simple daily intensity index
    _ts("    SDII …")
    results["SDII"] = atmos.daily_pr_intensity(pr=pr_flux, thresh="1 mm/day",
                                                freq="YS")

    # CDD: Max consecutive dry days
    _ts("    CDD …")
    results["CDD"] = atmos.maximum_consecutive_dry_days(pr=pr_flux,
                                                         thresh="1 mm/day",
                                                         freq="YS")

    # CWD: Max consecutive wet days
    _ts("    CWD …")
    results["CWD"] = atmos.maximum_consecutive_wet_days(pr=pr_flux,
                                                         thresh="1 mm/day",
                                                         freq="YS")

    # R10mm: Days with precip >= 10mm
    _ts("    R10mm …")
    results["R10mm"] = atmos.wetdays(pr=pr_flux, thresh="10 mm/day", freq="YS")

    # R20mm: Days with precip >= 20mm
    _ts("    R20mm …")
    results["R20mm"] = atmos.wetdays(pr=pr_flux, thresh="20 mm/day", freq="YS")

    # PRCPTOT: Annual total wet-day precipitation
    _ts("    PRCPTOT …")
    results["PRCPTOT"] = atmos.wet_precip_accumulation(pr=pr_flux, thresh="1 mm/day",
                                                        freq="YS")

    # Merge into one dataset
    ds_out = xr.Dataset(results)
    return ds_out


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_index_timeseries(index_name, data_all_years, lat, years, outdir):
    """Plot global-mean timeseries of an ETCCDI index."""
    cos_w = np.cos(np.deg2rad(lat))
    cos_w = cos_w / cos_w.sum()

    # Area-weighted global mean per year
    gmean = []
    for yr_idx in range(data_all_years.shape[0]):
        d = data_all_years[yr_idx]
        valid = np.isfinite(d)
        if valid.any():
            w = np.broadcast_to(cos_w[:, None], d.shape)
            gmean.append(np.nansum(d * w) / np.nansum(np.where(valid, w, 0)))
        else:
            gmean.append(np.nan)
    gmean = np.array(gmean)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(years, gmean, lw=0.8, alpha=0.7, color="steelblue")

    # Running mean
    if len(gmean) > 20:
        kernel = np.ones(10) / 10
        smooth = np.convolve(gmean, kernel, mode="valid")
        ax.plot(years[4:-5], smooth, lw=2, color="darkblue", label="10-yr running mean")
        ax.legend(fontsize=9)

    ax.set_xlabel("Model Year")
    ax.set_ylabel(index_name)
    ax.set_title(f"ETCCDI: {index_name} — Global Weighted Mean")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"ts_{index_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_index_map(index_name, data_mean, lat, lon, outdir):
    """Plot climatological mean map of an ETCCDI index."""
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(lon, lat, data_mean, shading="auto", cmap="RdYlBu_r")
    plt.colorbar(im, ax=ax, shrink=0.8, label=index_name)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"ETCCDI: {index_name} — Climatological Mean")
    fig.tight_layout()
    fig.savefig(outdir / f"map_{index_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETCCDI indices from 6h data")
    parser.add_argument("--n_segments", default="all",
                        help="Number of segments to process, or 'all'")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip computation, just regenerate plots")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGS_DIR.mkdir(exist_ok=True)

    seg_dirs = sorted(glob.glob(str(DATA_ROOT / "seg_*/atmosphere")))
    if args.n_segments != "all":
        seg_dirs = seg_dirs[:int(args.n_segments)]

    print(f"ETCCDI computation: {len(seg_dirs)} segments")
    print()

    # ── Compute indices per segment ───────────────────────────────────────
    if not args.plots_only:
        for si, sd in enumerate(seg_dirs):
            seg_name = os.path.basename(os.path.dirname(sd))
            out_nc = RESULTS_DIR / f"{seg_name}_etccdi.nc"

            if out_nc.exists():
                _ts(f"{seg_name}: already computed, skipping (use --plots_only or delete)")
                continue

            _ts(f"{seg_name}: resampling to daily …")
            ds_daily = resample_to_daily(sd)
            if ds_daily is None:
                _ts(f"  Skipping {seg_name}: too few timesteps for a full year")
                continue
            _ts(f"  Daily dataset: {ds_daily.dims}")

            _ts(f"  Computing ETCCDI indices …")
            ds_etccdi = compute_etccdi_indices(ds_daily)

            _ts(f"  Saving → {out_nc}")
            ds_etccdi.to_netcdf(out_nc)
            del ds_daily, ds_etccdi

    # ── Load all results and plot ─────────────────────────────────────────
    print()
    _ts("Loading results for plotting …")
    result_files = sorted(glob.glob(str(RESULTS_DIR / "seg_*_etccdi.nc")))
    if not result_files:
        print("  No results found. Run without --plots_only first.")
        return

    ds_all = xr.open_mfdataset(result_files, combine="nested", concat_dim="time")
    lat = ds_all["lat"].values
    lon = ds_all["lon"].values
    n_total_years = ds_all.dims["time"]
    years = np.arange(n_total_years) + START_YEAR

    _ts(f"  {n_total_years} years across {len(result_files)} segments")
    print()
    _ts("Generating plots …")

    indices = [v for v in ds_all.data_vars]
    for idx_name in indices:
        _ts(f"  {idx_name}")
        data = ds_all[idx_name].values  # (time, lat, lon)

        # xclim returns some indices as timedelta64 (e.g. FD, CDD) — convert to float days
        if np.issubdtype(data.dtype, np.timedelta64):
            data = data / np.timedelta64(1, "D")
            data = data.astype(np.float64)

        # Timeseries
        plot_index_timeseries(idx_name, data, lat, years, FIGS_DIR)

        # Map (climatological mean)
        clim_mean = np.nanmean(data, axis=0)
        plot_index_map(idx_name, clim_mean, lat, lon, FIGS_DIR)

    n_figs = len(list(FIGS_DIR.glob("*.png")))
    print(f"\nDone — {n_figs} figures in {FIGS_DIR}/")
    print(f"       NetCDF results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
