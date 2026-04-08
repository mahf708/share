"""
Extreme weather analysis from 6-hourly ACE2-EAMv3 piControl output.

Analyzes TS, PS, and precipitation extremes across the ~800-year run.
Because each segment's 6h file is ~30 GB, we process segment-by-segment
and accumulate statistics incrementally rather than loading everything.

============================================================================
SCIENTIFIC BACKGROUND & INTERPRETATION GUIDE
============================================================================

This script draws on several areas of extreme-value analysis in climate
science. Below is context for each analysis and how to interpret the output.

--- 1. Precipitation Intensity Distribution (Pendergrass & Hartmann 2014) ---
Precipitation is zero most of the time and heavy-tailed when it rains.
We bin precipitation by intensity and plot the contribution of each bin
to total precipitation. In observations, the heaviest 10% of events
contribute ~50–70% of total rainfall. If ACE's tail is too light, it
underestimates flood risk; too heavy, it overestimates.
  • X-axis: precipitation rate bins (mm/day)
  • Y-axis: fraction of total precipitation from that bin
  • Compare: the peak of this curve should be around 10–40 mm/day in the
    tropics (observations). If ACE shifts the peak left, it's producing
    too much drizzle and not enough heavy rain.

--- 2. Return Levels and GEV Fits (Coles 2001, Kharin et al. 2013) ---------
We fit a Generalized Extreme Value (GEV) distribution to annual block
maxima (the single most extreme value each year at each grid point).
With 800 years we can estimate return levels up to ~200 years directly
from empirical data, and extrapolate further with GEV.
  • The shape parameter ξ (xi) controls tail heaviness:
      ξ > 0: Fréchet (heavy tail, unbounded) — typical for precipitation
      ξ = 0: Gumbel (exponential tail) — typical for temperature
      ξ < 0: Weibull (bounded upper tail) — sometimes seen in cold extremes
  • Return level plot: shows the expected magnitude of the 1-in-N-year event.
    Compare the empirical points against the GEV curve — poor fit at the
    tails suggests the model may not capture extreme dynamics correctly.

--- 3. Diurnal Temperature Range (DTR) (Karl et al. 1993) ------------------
DTR = daily max TS − daily min TS. Observed DTR has decreased over land
since 1950, partly due to increased cloud cover and greenhouse warming
raising nighttime temperatures more than daytime. In a piControl, DTR
should be stable over time. Spatial patterns: large DTR over deserts
(clear skies, dry soil), small DTR over oceans (thermal inertia).
  • If ACE produces unrealistically small DTR everywhere, it may lack
    a proper diurnal cycle.
  • If DTR drifts over time, the model's surface energy balance is evolving.

--- 4. Precipitation Dry/Wet Spell Duration (Giorgi et al. 2014) -----------
Consecutive dry days (CDD) and consecutive wet days (CWD) are key
indicators of drought and flood risk. We define:
  • Dry day: daily precip < 1 mm/day
  • Wet day: daily precip ≥ 1 mm/day
The distributions of spell lengths should follow approximately geometric
distributions, but with heavier tails in dry regions. ACE should produce
CDD > 100 days in arid regions and CWD > 10 days in monsoon regions.

--- 5. Surface Pressure Extremes (Hodges et al. 2011) ----------------------
Minimum PS at each grid point indicates the deepest cyclones the model
produces. Observed extratropical cyclones reach ~940–960 hPa; tropical
cyclones can go below 900 hPa. If ACE's minimum PS is too high, it
doesn't produce realistic storm intensification. We look at:
  • Spatial map of all-time minimum PS — storm tracks should be visible
  • Distribution of PS minima in key storm track regions

--- 6. Percentile-Based Indices (ETCCDI, Zhang et al. 2011) ----------------
Standard climate extreme indices defined by the Expert Team on Climate
Change Detection and Indices:
  • TX90p: fraction of days where TS > 90th percentile (hot days)
  • TX10p: fraction of days where TS < 10th percentile (cold days)
  • R95p: fraction of total precip from days > 95th percentile
  • R99p: fraction of total precip from days > 99th percentile
In a stationary piControl, TX90p and TX10p should be ~10% everywhere by
definition. Deviations indicate non-stationarity or inhomogeneity.
If R95p is very high (>50%), the model concentrates too much precip in
rare extreme events — problematic for hydrology.

--- 7. Scaling of Precipitation Extremes with Temperature ------------------
(Trenberth et al. 2003, O'Gorman & Schneider 2009)
The Clausius-Clapeyron relation predicts that atmospheric water vapor
increases ~7%/K with warming. Precipitation extremes are expected to
scale similarly or even super-CC (>7%/K) due to dynamical feedbacks.
We bin grid points by mean temperature and plot the intensity of the
99th-percentile precipitation vs temperature. The slope tells us whether
ACE follows CC scaling — a basic thermodynamic constraint any climate
model should satisfy.
  • ~7%/K: Clausius-Clapeyron (thermodynamic expectation)
  • >7%/K: super-CC, dynamical intensification
  • <7%/K: sub-CC, the model may lack convective intensification

============================================================================

Usage:
    micromamba run -n xgns python extremes_6h.py

    To force overwrite existing figures:
    micromamba run -n xgns python extremes_6h.py --overwrite
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from config import get_6h_segment_dirs, SPINUP_CUTOFF_YEAR, N_WORKERS, DASK_CHUNK_SIZE

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEG_DIRS = get_6h_segment_dirs()

OUTPUT_DIR = Path("figs_extremes_6h")
OUTPUT_DIR.mkdir(exist_ok=True)
_fig_counter = 0

OVERWRITE = "--overwrite" in sys.argv

# Precipitation threshold for wet day (mm/day)
WET_DAY_THRESHOLD = 1.0  # mm/day

# Conversion: kg/m2/s → mm/day
KG_M2_S_TO_MM_DAY = 86400.0

# Number of segments to process (set to None for all)
MAX_SEGMENTS = None

# Cache file for accumulated statistics (avoids reprocessing 6h data)
CACHE_FILE = Path("cache_extremes_6h.npz")


def save_fig(fig, name):
    """Save figure as numbered PNG at 300 dpi. Skip if exists unless --overwrite."""
    global _fig_counter
    _fig_counter += 1
    path = OUTPUT_DIR / f"{_fig_counter:02d}_{name}.png"
    if path.exists() and not OVERWRITE:
        plt.close(fig)
        print(f"    → {path} (exists, skipping)")
        return
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {path}")


def cosine_weights(lat):
    return np.cos(np.deg2rad(lat))


# ---------------------------------------------------------------------------
# Pass 1: Accumulate per-segment statistics
# ---------------------------------------------------------------------------
# We process each segment's 6h files and accumulate:
#   - Annual block maxima/minima for TS, PS, precip
#   - Daily max/min TS for DTR
#   - Percentile thresholds (from first segment, applied to all)
#   - Precipitation intensity histogram
#   - Temperature-binned precipitation extremes

segs = SEG_DIRS[:MAX_SEGMENTS] if MAX_SEGMENTS else SEG_DIRS
n_segs = len(segs)

# Check if all figures already exist — skip expensive computation if so
ALL_FIG_NAMES = [
    "precip_intensity_distribution", "precip_annual_max_maps", "precip_return_levels",
    "ts_extreme_maps", "ts_return_levels", "dtr", "ps_extremes_maps",
    "ps_extremes_histogram", "ps_outlier_investigation", "cc_scaling",
    "extreme_trends", "percentile_exceedance_maps", "r95p_distribution",
    "summary_stats",
]
all_exist = all(
    (OUTPUT_DIR / f"{i+1:02d}_{name}.png").exists()
    for i, name in enumerate(ALL_FIG_NAMES)
)
if all_exist and not OVERWRITE:
    print("All figures already exist. Use --overwrite to regenerate.")
    print(f"  Figures in: {OUTPUT_DIR}/")
    sys.exit(0)

print(f"Processing {n_segs} segments for 6-hourly extremes …")

# Get grid info from first segment
ds0 = xr.open_dataset(f"{segs[0]}/6h_surface_TS_predictions.nc", chunks={})
lat = ds0.lat.values
lon = ds0.lon.values
nlat, nlon = len(lat), len(lon)
ds0.close()

# Accumulators
all_ts_annual_max = []       # (n_years, lat, lon)
all_ts_annual_min = []
all_ps_annual_min = []
all_precip_annual_max = []   # daily max precip per year
all_dtr_annual_mean = []     # mean DTR per year
all_precip_global_vals = []  # sample of precip values for intensity distribution

# For CC scaling: (temperature_bin, precip_99th)
temp_bins = np.arange(200, 320, 2)  # K
precip_by_temp_bin = {i: [] for i in range(len(temp_bins) - 1)}

cache_loaded = False
if CACHE_FILE.exists() and not OVERWRITE:
    print("  Loading cached statistics …")
    try:
        cache = np.load(CACHE_FILE, allow_pickle=True)
        ts_annual_max = cache["ts_annual_max"]
        ts_annual_min = cache["ts_annual_min"]
        ps_annual_min = cache["ps_annual_min"]
        precip_annual_max = cache["precip_annual_max"]
        dtr_annual_mean = cache["dtr_annual_mean"]
        all_precip_global_vals = [cache["precip_global_vals"]]
        precip_by_temp_bin = {i: list(cache[f"cc_bin_{i}"]) for i in range(len(temp_bins)-1) if f"cc_bin_{i}" in cache}
        n_total_years = len(ts_annual_max)
        cache_loaded = True
        print(f"    {n_total_years} years loaded from cache")
    except Exception as e:
        print(f"    Cache load failed ({e}), recomputing …")

if not cache_loaded:
    for seg_idx, seg_dir in enumerate(segs):
        print(f"\n  Segment {seg_idx}/{n_segs-1}: {seg_dir.split('/')[-2]}")

        # ---- Load 6h data lazily ----
        ts_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_TS_predictions.nc", chunks={"time": 1460}
        ).isel(sample=0)
        ps_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_PS_predictions.nc", chunks={"time": 1460}
        ).isel(sample=0)
        pr_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_surface_precipitation_rate_predictions.nc",
            chunks={"time": 1460},
        ).isel(sample=0)

        # Use valid_time as the time coordinate
        vt = ts_ds.valid_time.compute().values
        n_times = len(vt)

        # Process in yearly chunks (4 timesteps/day × 365 days = 1460 per year)
        steps_per_year = 1460  # 365 days × 4 (6-hourly)
        n_full_years = n_times // steps_per_year

        print(f"    {n_times} timesteps → {n_full_years} full years")

        for yr in range(n_full_years):
            t0 = yr * steps_per_year
            t1 = t0 + steps_per_year

            # Check spinup cutoff using valid_time
            vt_yr = vt[t0:t1]
            first_year = vt_yr[0].year if hasattr(vt_yr[0], 'year') else 0
            if first_year < SPINUP_CUTOFF_YEAR:
                continue

            # Load one year of data
            ts_yr = ts_ds.TS.isel(time=slice(t0, t1)).values     # (1460, lat, lon)
            ps_yr = ps_ds.PS.isel(time=slice(t0, t1)).values
            pr_yr = pr_ds.surface_precipitation_rate.isel(time=slice(t0, t1)).values

            # ---- Annual block maxima/minima ----
            all_ts_annual_max.append(np.nanmax(ts_yr, axis=0))
            all_ts_annual_min.append(np.nanmin(ts_yr, axis=0))
            all_ps_annual_min.append(np.nanmin(ps_yr, axis=0))

            # ---- Daily aggregation for precip and DTR ----
            # Reshape to (365, 4, lat, lon) → daily
            ts_daily = ts_yr.reshape(365, 4, nlat, nlon)
            pr_daily = pr_yr.reshape(365, 4, nlat, nlon)

            daily_max_ts = np.nanmax(ts_daily, axis=1)
            daily_min_ts = np.nanmin(ts_daily, axis=1)
            dtr = daily_max_ts - daily_min_ts
            all_dtr_annual_mean.append(np.nanmean(dtr, axis=0))

            # Daily mean precip for annual max daily precip
            daily_mean_pr = np.nanmean(pr_daily, axis=1) * KG_M2_S_TO_MM_DAY  # mm/day
            all_precip_annual_max.append(np.nanmax(daily_mean_pr, axis=0))

            # ---- Sample precip values for global distribution ----
            # Subsample to keep memory manageable: take every 10th day, tropics only
            if yr % 10 == 0:
                trop_mask = np.abs(lat) < 30
                pr_sample = daily_mean_pr[::10, trop_mask, :].ravel()
                pr_sample = pr_sample[pr_sample > 0.1]  # skip near-zero
                if len(pr_sample) > 0:
                    all_precip_global_vals.append(pr_sample)

            # ---- CC scaling: bin precip extremes by local temperature ----
            if yr % 20 == 0:  # subsample to save time
                ts_daily_mean = np.nanmean(ts_daily, axis=1)  # (365, lat, lon)
                pr_flat = daily_mean_pr.ravel()
                ts_flat = ts_daily_mean.ravel()
                valid = (pr_flat > 0.1) & np.isfinite(ts_flat)
                bin_idx = np.digitize(ts_flat[valid], temp_bins) - 1
                pr_valid = pr_flat[valid]
                for bi in range(len(temp_bins) - 1):
                    mask = bin_idx == bi
                    if mask.sum() > 100:
                        precip_by_temp_bin[bi].append(np.percentile(pr_valid[mask], 99))

        ts_ds.close()
        ps_ds.close()
        pr_ds.close()

    # Stack into arrays
    ts_annual_max = np.array(all_ts_annual_max)     # (n_years_total, lat, lon)
    ts_annual_min = np.array(all_ts_annual_min)
    ps_annual_min = np.array(all_ps_annual_min)
    precip_annual_max = np.array(all_precip_annual_max)
    dtr_annual_mean = np.array(all_dtr_annual_mean)
    n_total_years = len(ts_annual_max)

    print(f"\n  Total: {n_total_years} years of annual statistics accumulated (post-spinup, year >= {SPINUP_CUTOFF_YEAR})")

    # Save cache
    print("  Saving cache …")
    save_dict = {
        "ts_annual_max": ts_annual_max,
        "ts_annual_min": ts_annual_min,
        "ps_annual_min": ps_annual_min,
        "precip_annual_max": precip_annual_max,
        "dtr_annual_mean": dtr_annual_mean,
        "precip_global_vals": np.concatenate(all_precip_global_vals),
    }
    for i in range(len(temp_bins)-1):
        if precip_by_temp_bin[i]:
            save_dict[f"cc_bin_{i}"] = np.array(precip_by_temp_bin[i])
    np.savez(CACHE_FILE, **save_dict)
    print(f"    Saved to {CACHE_FILE}")


# ===========================================================================
# Figures
# ===========================================================================
print("\nGenerating figures …\n")

w = cosine_weights(lat)


# ---- 1. Precipitation intensity distribution (Pendergrass & Hartmann) ----
# INTERPRETATION: The peak of this curve shows which rain rates contribute
# most to total rainfall. Observations show the peak at ~10-40 mm/day in
# the tropics. If the curve peaks at very low rates, the model produces
# too much "drizzle" and not enough intense rainfall. A long right tail
# indicates the model can produce extreme downpours.
print("  1. Precipitation intensity distribution …")
all_pr = np.concatenate(all_precip_global_vals)

# Logarithmic bins for precipitation
pr_bins = np.logspace(np.log10(0.1), np.log10(500), 80)
pr_centers = 0.5 * (pr_bins[:-1] + pr_bins[1:])
hist_count, _ = np.histogram(all_pr, bins=pr_bins)
hist_amount, _ = np.histogram(all_pr, bins=pr_bins, weights=all_pr)
total_amount = hist_amount.sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Frequency distribution
axes[0].semilogx(pr_centers, hist_count / hist_count.sum(), "k-", lw=1.5)
axes[0].set_xlabel("Precipitation rate (mm/day)")
axes[0].set_ylabel("Frequency")
axes[0].set_title(f"Tropical Precip Frequency Distribution (post-spinup, {n_total_years} yr)")
axes[0].set_xlim(0.1, 500)

# Contribution to total precipitation
# This is the key plot from Pendergrass & Hartmann (2014): what fraction
# of total rainfall comes from each intensity bin?
axes[1].semilogx(pr_centers, hist_amount / total_amount, "b-", lw=1.5)
axes[1].set_xlabel("Precipitation rate (mm/day)")
axes[1].set_ylabel("Fraction of total precipitation")
axes[1].set_title("Precip Intensity Contribution (Pendergrass & Hartmann, post-spinup)")
axes[1].set_xlim(0.1, 500)

# Mark key thresholds
for ax in axes:
    ax.axvline(10, color="gray", ls="--", alpha=0.5, label="10 mm/day")
    ax.axvline(50, color="gray", ls=":", alpha=0.5, label="50 mm/day")
    ax.legend(fontsize=9)

fig.tight_layout()
save_fig(fig, "precip_intensity_distribution")


# ---- 2. Annual maximum precipitation map --------------------------------
# INTERPRETATION: Shows the spatial pattern of the most extreme daily
# rainfall the model produces in a typical year. High values should appear
# in tropical convergence zones, monsoon regions, and extratropical storm
# tracks. If maxima are suspiciously uniform or weak (<50 mm/day in the
# tropics), the model lacks convective intensity.
print("  2. Annual max precip maps …")
precip_max_median = np.median(precip_annual_max, axis=0)  # typical annual max
precip_max_overall = np.max(precip_annual_max, axis=0)    # all-time record

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

im0 = axes[0].pcolormesh(lon, lat, precip_max_median, cmap="YlOrRd",
                          shading="auto", vmin=0, vmax=100)
axes[0].set_title(f"Median Annual Maximum Daily Precipitation (mm/day, post-spinup, {n_total_years} yr)")
axes[0].set_ylabel("Latitude")
plt.colorbar(im0, ax=axes[0], label="mm/day")

im1 = axes[1].pcolormesh(lon, lat, precip_max_overall, cmap="YlOrRd",
                          shading="auto", vmin=0, vmax=300)
axes[1].set_title(f"All-Time Maximum Daily Precipitation (post-spinup, {n_total_years} yr, mm/day)")
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
plt.colorbar(im1, ax=axes[1], label="mm/day")

fig.tight_layout()
save_fig(fig, "precip_annual_max_maps")


# ---- 3. Return level plot — global-max precipitation ---------------------
# INTERPRETATION: The return level plot shows the expected magnitude of
# the 1-in-N-year event. With 800 years of data, we can directly estimate
# return levels up to ~200 years. The GEV fit (if scipy is available)
# extrapolates beyond. Compare the empirical points (crosses) to the GEV
# curve (line) — systematic departures indicate poor tail behavior.
print("  3. Return level plot …")

# Global maximum daily precip per year (area-weighted)
precip_global_max = np.array([
    np.nanmax(precip_annual_max[yr]) for yr in range(n_total_years)
])

# Tropical maximum per year
trop_mask = np.abs(lat) < 30
precip_trop_max = np.array([
    np.nanmax(precip_annual_max[yr, trop_mask, :]) for yr in range(n_total_years)
])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, data, title in [
    (axes[0], precip_global_max, "Global"),
    (axes[1], precip_trop_max, "Tropical (30°S–30°N)"),
]:
    sorted_data = np.sort(data)[::-1]
    n = len(sorted_data)
    # Empirical return periods (Weibull plotting position)
    return_periods = (n + 1) / np.arange(1, n + 1)

    ax.semilogx(return_periods, sorted_data, "kx", ms=4, alpha=0.5, label="Empirical")

    # Fit GEV if scipy available
    try:
        from scipy.stats import genextreme
        params = genextreme.fit(data)
        rp_fit = np.logspace(0, np.log10(n), 200)
        rl_fit = genextreme.isf(1.0 / rp_fit, *params)
        ax.semilogx(rp_fit, rl_fit, "r-", lw=2,
                     label=f"GEV fit (ξ={params[0]:.2f})")
    except Exception:
        pass

    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Daily max precip (mm/day)")
    ax.set_title(f"{title} Max Daily Precip Return Levels (post-spinup)")
    ax.legend()
    ax.grid(alpha=0.3)

fig.tight_layout()
save_fig(fig, "precip_return_levels")


# ---- 4. TS annual max/min maps ------------------------------------------
# INTERPRETATION: The all-time maximum TS should show realistic hottest
# temperatures: ~320-330 K (47-57°C) over deserts, ~300-310 K over oceans.
# The all-time minimum should show ~200-220 K over polar regions in winter.
# Unrealistic extremes (e.g., TS > 340 K or TS < 180 K) indicate the
# model is producing unphysical spikes, a common issue with ML emulators.
print("  4. TS extreme maps …")
ts_max_map = np.max(ts_annual_max, axis=0)
ts_min_map = np.min(ts_annual_min, axis=0)
ts_max_median = np.median(ts_annual_max, axis=0)
ts_min_median = np.median(ts_annual_min, axis=0)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

im = axes[0, 0].pcolormesh(lon, lat, ts_max_median - 273.15, cmap="YlOrRd",
                            shading="auto", vmin=10, vmax=55)
axes[0, 0].set_title(f"Median Annual Max TS (°C, post-spinup, {n_total_years} yr)")
plt.colorbar(im, ax=axes[0, 0], label="°C")

im = axes[0, 1].pcolormesh(lon, lat, ts_max_map - 273.15, cmap="YlOrRd",
                            shading="auto", vmin=10, vmax=60)
axes[0, 1].set_title(f"All-Time Max TS (post-spinup, {n_total_years} yr, °C)")
plt.colorbar(im, ax=axes[0, 1], label="°C")

im = axes[1, 0].pcolormesh(lon, lat, ts_min_median - 273.15, cmap="YlGnBu_r",
                            shading="auto", vmin=-70, vmax=20)
axes[1, 0].set_title(f"Median Annual Min TS (°C, post-spinup, {n_total_years} yr)")
plt.colorbar(im, ax=axes[1, 0], label="°C")

im = axes[1, 1].pcolormesh(lon, lat, ts_min_map - 273.15, cmap="YlGnBu_r",
                            shading="auto", vmin=-80, vmax=10)
axes[1, 1].set_title(f"All-Time Min TS (post-spinup, {n_total_years} yr, °C)")
plt.colorbar(im, ax=axes[1, 1], label="°C")

for ax in axes.flat:
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

fig.tight_layout()
save_fig(fig, "ts_extreme_maps")


# ---- 5. TS return levels — global hottest/coldest -----------------------
print("  5. TS return levels …")
ts_global_max = np.array([
    np.average(ts_annual_max[yr], weights=w[:, None] * np.ones((1, nlon)))
    for yr in range(n_total_years)
])
ts_global_min = np.array([
    np.average(ts_annual_min[yr], weights=w[:, None] * np.ones((1, nlon)))
    for yr in range(n_total_years)
])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, data, title, color in [
    (axes[0], ts_global_max - 273.15, f"Annual Max TS (global weighted mean, post-spinup, {n_total_years} yr)", "red"),
    (axes[1], -(ts_global_min - 273.15), f"Annual Min TS (global weighted mean, post-spinup, {n_total_years} yr)", "blue"),
]:
    if "Min" in title:
        plot_data = -(data)  # flip back for actual temps
        sorted_d = np.sort(ts_global_min - 273.15)
        n = len(sorted_d)
        rp = (n + 1) / np.arange(1, n + 1)
        ax.semilogx(rp, sorted_d, "x", color=color, ms=4, alpha=0.5)
        ax.set_ylabel("Min TS (°C)")
    else:
        sorted_d = np.sort(data)[::-1]
        n = len(sorted_d)
        rp = (n + 1) / np.arange(1, n + 1)
        ax.semilogx(rp, sorted_d, "x", color=color, ms=4, alpha=0.5)
        ax.set_ylabel("Max TS (°C)")

    ax.set_xlabel("Return period (years)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

fig.tight_layout()
save_fig(fig, "ts_return_levels")


# ---- 6. Diurnal Temperature Range (DTR) ---------------------------------
# INTERPRETATION: DTR should be large (>15 K) over continental interiors
# and deserts (clear skies, low thermal inertia) and small (<2 K) over
# oceans (high thermal inertia). If ACE produces uniformly small DTR,
# it may not resolve the diurnal cycle properly. If DTR is unrealistically
# large (>30 K), surface energy balance is likely wrong.
# Trend over time: in a piControl, DTR should be stable. Any drift
# indicates non-stationarity in the surface energy partition.
print("  6. DTR …")
dtr_clim = np.mean(dtr_annual_mean, axis=0)  # climatological mean DTR

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

im = axes[0].pcolormesh(lon, lat, dtr_clim, cmap="YlOrRd", shading="auto",
                         vmin=0, vmax=25)
axes[0].set_title(f"Climatological Mean Diurnal Temperature Range (K, post-spinup, {n_total_years} yr)")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[0], label="K")

# Global-mean DTR timeseries
dtr_global = np.array([
    np.average(dtr_annual_mean[yr], weights=w[:, None] * np.ones((1, nlon))) for yr in range(n_total_years)
])
axes[1].plot(np.arange(n_total_years), dtr_global, alpha=0.6, lw=0.8)
trend = np.polyfit(np.arange(n_total_years), dtr_global, 1)
axes[1].plot(np.arange(n_total_years), np.polyval(trend, np.arange(n_total_years)),
             "r--", lw=2, label=f"Trend: {trend[0]*100:.4f} K/century")
axes[1].set_xlabel("Year index")
axes[1].set_ylabel("DTR (K)")
axes[1].set_title(f"Global Mean DTR Over Time (Stationarity Check, post-spinup, {n_total_years} yr)")
axes[1].legend()

fig.tight_layout()
save_fig(fig, "dtr")


# ---- 7. Surface pressure extremes (deepest cyclones) --------------------
# INTERPRETATION: The minimum PS map reveals storm tracks. Expect:
#   - North Atlantic/Pacific storm tracks: minima ~950-970 hPa
#   - Southern Ocean: strong extratropical cyclones ~940-960 hPa
#   - Tropical cyclone basins: potentially <950 hPa
# If ACE's minimum PS is everywhere >990 hPa, it cannot produce
# realistic cyclone deepening. The 800-year all-time minimum is the
# ultimate test of how extreme ACE's storms can get.
print("  7. PS extremes …")
ps_min_map = np.min(ps_annual_min, axis=0) / 100  # hPa
ps_min_median = np.median(ps_annual_min, axis=0) / 100

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

im = axes[0].pcolormesh(lon, lat, ps_min_median, cmap="viridis_r",
                         shading="auto", vmin=930, vmax=1020)
axes[0].set_title(f"Median Annual Minimum Surface Pressure (hPa, post-spinup, {n_total_years} yr)")
axes[0].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[0], label="hPa")

im = axes[1].pcolormesh(lon, lat, ps_min_map, cmap="viridis_r",
                         shading="auto", vmin=900, vmax=1020)
axes[1].set_title(f"All-Time Minimum Surface Pressure (post-spinup, {n_total_years} yr, hPa)")
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[1], label="hPa")

fig.tight_layout()
save_fig(fig, "ps_extremes_maps")

# PS minima distribution in storm track regions
fig, ax = plt.subplots(figsize=(10, 5))

regions = {
    "N. Atlantic (40-60°N, 300-360°E)": (slice(40, 60), slice(300, 360)),
    "N. Pacific (40-60°N, 150-220°E)": (slice(40, 60), slice(150, 220)),
    "Southern Ocean (50-70°S)": (slice(-70, -50), slice(0, 360)),
}

for label, (lat_sl, lon_sl) in regions.items():
    lat_mask = (lat >= lat_sl.start) & (lat <= lat_sl.stop)
    lon_mask = (lon >= lon_sl.start) & (lon <= lon_sl.stop)
    region_ps_min = ps_annual_min[:, lat_mask, :][:, :, lon_mask].min(axis=(1, 2)) / 100
    ax.hist(region_ps_min, bins=30, alpha=0.5, density=True, label=label)

ax.set_xlabel("Annual Min PS (hPa)")
ax.set_ylabel("Density")
ax.set_title(f"Distribution of Annual Minimum PS in Storm Track Regions (post-spinup, {n_total_years} yr)")
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, "ps_extremes_histogram")


# ---- 8b. PS outlier investigation ----------------------------------------
print("  PS outlier investigation …")
# Find the grid point with the all-time minimum PS
ps_min_overall = np.min(ps_annual_min, axis=0)
min_idx = np.unravel_index(np.argmin(ps_min_overall), ps_min_overall.shape)
min_lat = lat[min_idx[0]]
min_lon = lon[min_idx[1]]
min_val = ps_min_overall[min_idx] / 100  # hPa

# Also find the min over only "low-elevation" grid points (mean PS > 800 hPa)
# Use median annual min as proxy for mean PS
ps_median_min = np.median(ps_annual_min, axis=0) / 100
low_elev_mask = ps_median_min > 800
ps_min_low_elev = np.where(low_elev_mask, np.min(ps_annual_min, axis=0) / 100, np.nan)
min_low_elev = np.nanmin(ps_min_low_elev)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Map of all-time min PS with outlier marked
im = axes[0, 0].pcolormesh(lon, lat, ps_min_overall / 100, cmap="viridis_r",
                            shading="auto", vmin=500, vmax=1020)
axes[0, 0].plot(min_lon, min_lat, "r*", ms=15, label=f"Min: {min_val:.0f} hPa at ({min_lat:.0f}\u00b0, {min_lon:.0f}\u00b0E)")
axes[0, 0].set_title("All-Time Minimum PS with Outlier Location (post-spinup)")
axes[0, 0].legend(fontsize=9)
plt.colorbar(im, ax=axes[0, 0], label="hPa")

# Low-elevation only (mean PS > 800 hPa) — filters out mountains
im = axes[0, 1].pcolormesh(lon, lat, ps_min_low_elev, cmap="viridis_r",
                            shading="auto", vmin=900, vmax=1020)
axes[0, 1].set_title(f"All-Time Min PS (low-elevation only, > 800 hPa, post-spinup)\nGlobal min: {min_low_elev:.0f} hPa")
plt.colorbar(im, ax=axes[0, 1], label="hPa")

# Timeseries of annual min PS at the outlier location
ps_at_outlier = ps_annual_min[:, min_idx[0], min_idx[1]] / 100
axes[1, 0].plot(np.arange(len(ps_at_outlier)), ps_at_outlier, "k-", lw=0.8)
axes[1, 0].set_xlabel("Year index")
axes[1, 0].set_ylabel("Annual min PS (hPa)")
axes[1, 0].set_title(f"PS Timeseries at Outlier Point ({min_lat:.0f}\u00b0, {min_lon:.0f}\u00b0E)")

# Histogram of all grid points' all-time minimum PS
all_mins = (ps_min_overall / 100).ravel()
axes[1, 1].hist(all_mins, bins=50, color="steelblue", alpha=0.7)
axes[1, 1].axvline(min_val, color="red", lw=2, label=f"Overall min: {min_val:.0f} hPa")
axes[1, 1].set_xlabel("All-time min PS (hPa)")
axes[1, 1].set_ylabel("Number of grid points")
axes[1, 1].set_title("Distribution of All-Time Minimum PS Across Grid Points (post-spinup)")
axes[1, 1].legend()

for ax in axes.flat:
    ax.set_xlabel(ax.get_xlabel() or "Longitude")
    ax.set_ylabel(ax.get_ylabel() or "Latitude")

fig.suptitle("Surface Pressure Outlier Investigation", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "ps_outlier_investigation")


# ---- 9. Clausius-Clapeyron scaling of precip extremes --------------------
# INTERPRETATION: The Clausius-Clapeyron relation predicts ~7%/K increase
# in water vapor capacity with warming. Precipitation extremes (99th
# percentile) should roughly follow this scaling. Plot the 99th percentile
# of daily precipitation vs. local mean temperature.
#   - Slope ~7%/K (dashed reference): thermodynamic scaling — expected
#   - Slope >7%/K: super-CC, dynamical intensification present
#   - Slope <7%/K or negative at high T: moisture limitation (dry regions)
#   - "Hook" shape (increase then decrease at high T): observed in nature,
#     indicating moisture supply limits in very hot/dry conditions
print("  9. CC scaling …")
temp_centers = 0.5 * (temp_bins[:-1] + temp_bins[1:])
cc_medians = []
cc_temps = []

for bi in range(len(temp_bins) - 1):
    vals = precip_by_temp_bin.get(bi, [])
    if len(vals) >= 3:
        cc_medians.append(np.median(vals))
        cc_temps.append(temp_centers[bi])

cc_temps = np.array(cc_temps)
cc_medians = np.array(cc_medians)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
axes[0].plot(cc_temps - 273.15, cc_medians, "ko-", ms=5, lw=1.5)
axes[0].set_xlabel("Local Mean Temperature (°C)")
axes[0].set_ylabel("99th Percentile Precip (mm/day)")
axes[0].set_title("Precip Extremes vs Temperature (post-spinup)")
axes[0].grid(alpha=0.3)

# Log scale — slope should be ~7%/K for CC scaling
axes[1].semilogy(cc_temps - 273.15, cc_medians, "ko-", ms=5, lw=1.5, label="ACE model")

# Add CC reference line (7%/K from a reference point)
if len(cc_medians) > 5:
    ref_idx = len(cc_medians) // 2
    ref_t = cc_temps[ref_idx] - 273.15
    ref_p = cc_medians[ref_idx]
    cc_line = ref_p * np.exp(0.07 * (cc_temps - 273.15 - ref_t))
    axes[1].semilogy(cc_temps - 273.15, cc_line, "r--", lw=2, alpha=0.7,
                      label="7%/K (Clausius-Clapeyron)")
    # 2× CC reference
    cc2_line = ref_p * np.exp(0.14 * (cc_temps - 273.15 - ref_t))
    axes[1].semilogy(cc_temps - 273.15, cc2_line, "b:", lw=1.5, alpha=0.5,
                      label="14%/K (2× CC)")

axes[1].set_xlabel("Local Mean Temperature (°C)")
axes[1].set_ylabel("99th Percentile Precip (mm/day)")
axes[1].set_title("CC Scaling (log scale — slope = scaling rate, post-spinup)")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

fig.tight_layout()
save_fig(fig, "cc_scaling")


# ---- 10. Annual max TS trend (non-stationarity of extremes) --------------
# INTERPRETATION: In a stable piControl, the global-mean annual max TS
# should not trend. A trend here (even if mean TS is stable) indicates
# that the tails of the distribution are shifting — the model may be
# slowly changing its extreme behavior even if the mean is controlled.
print("  10. Extreme trends …")
ts_gmax = np.array([
    np.average(ts_annual_max[yr], weights=w[:, None] * np.ones((1, nlon)))
    for yr in range(n_total_years)
])
ts_gmin = np.array([
    np.average(ts_annual_min[yr], weights=w[:, None] * np.ones((1, nlon)))
    for yr in range(n_total_years)
])
pr_gmax = np.array([
    np.average(precip_annual_max[yr], weights=w[:, None] * np.ones((1, nlon)))
    for yr in range(n_total_years)
])

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
yrs = np.arange(n_total_years)

for ax, data, title, unit in [
    (axes[0], ts_gmax - 273.15, f"Global-Mean Annual Max TS (post-spinup, {n_total_years} yr)", "°C"),
    (axes[1], ts_gmin - 273.15, f"Global-Mean Annual Min TS (post-spinup, {n_total_years} yr)", "°C"),
    (axes[2], pr_gmax, f"Global-Mean Annual Max Daily Precip (post-spinup, {n_total_years} yr)", "mm/day"),
]:
    ax.plot(yrs, data, alpha=0.5, lw=0.8)
    trend = np.polyfit(yrs, data, 1)
    ax.plot(yrs, np.polyval(trend, yrs), "r--", lw=2,
            label=f"Trend: {trend[0]*100:.4f} {unit}/century")
    ax.set_ylabel(unit)
    ax.set_title(title)
    ax.legend()

axes[2].set_xlabel("Year index")
fig.tight_layout()
save_fig(fig, "extreme_trends")


# ---- 11. Percentile exceedance maps (ETCCDI-style) ----------------------
# INTERPRETATION: We compute the 90th and 10th percentile of TS from the
# first 100 years, then count how often these are exceeded in the last
# 100 years. In a stationary climate, exceedance should be ~10% everywhere.
#   - Maps showing >10% hot-day exceedance → model is warming
#   - Maps showing <10% → model is cooling
#   - Spatial patterns reveal where non-stationarity is strongest
print("  11. Percentile exceedance …")
n_baseline = min(100, n_total_years // 4)
n_eval = min(100, n_total_years // 4)

ts_p90 = np.percentile(ts_annual_max[:n_baseline], 90, axis=0)
ts_p10 = np.percentile(ts_annual_min[:n_baseline], 10, axis=0)

# Exceedance rate in last N years
hot_exceed = np.mean(ts_annual_max[-n_eval:] > ts_p90, axis=0) * 100
cold_exceed = np.mean(ts_annual_min[-n_eval:] < ts_p10, axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

im = axes[0].pcolormesh(lon, lat, hot_exceed, cmap="RdBu_r", shading="auto",
                         vmin=0, vmax=30)
axes[0].set_title(f"Hot Extreme Exceedance: % years with max TS > P90 (post-spinup)\n"
                   f"(baseline: first {n_baseline} yr, eval: last {n_eval} yr, expect ~10%)")
plt.colorbar(im, ax=axes[0], label="%")

im = axes[1].pcolormesh(lon, lat, cold_exceed, cmap="RdBu", shading="auto",
                         vmin=0, vmax=30)
axes[1].set_title(f"Cold Extreme Exceedance: % years with min TS < P10 (post-spinup)\n"
                   f"(baseline: first {n_baseline} yr, eval: last {n_eval} yr, expect ~10%)")
plt.colorbar(im, ax=axes[1], label="%")

for ax in axes:
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

fig.tight_layout()
save_fig(fig, "percentile_exceedance_maps")


# ---- 12. Precipitation R95p map -----------------------------------------
# INTERPRETATION: R95p is the fraction of total precipitation that comes
# from days exceeding the 95th percentile of wet-day precipitation.
# This is a standard ETCCDI index. Values of 30-50% are typical in the
# tropics (much of the rain comes from a few intense events). Very high
# values (>60%) suggest the model concentrates precipitation too much
# in extreme events. Very low values (<20%) suggest missing intensity.
print("  12. R95p map …")
# We already have annual max precip; compute R95p from the accumulated
# percentile data. Use the global precip samples we collected.
# For a proper spatial R95p, we'd need to reprocess, so we compute a
# global/tropical summary instead.
pr_all = np.concatenate(all_precip_global_vals)
p95 = np.percentile(pr_all, 95)
p99 = np.percentile(pr_all, 99)

r95p = pr_all[pr_all > p95].sum() / pr_all.sum() * 100
r99p = pr_all[pr_all > p99].sum() / pr_all.sum() * 100

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(pr_all, bins=np.logspace(np.log10(0.1), np.log10(500), 100),
        weights=pr_all, color="steelblue", alpha=0.7)
ax.axvline(p95, color="orange", lw=2, label=f"P95 = {p95:.1f} mm/day")
ax.axvline(p99, color="red", lw=2, label=f"P99 = {p99:.1f} mm/day")
ax.set_xscale("log")
ax.set_xlabel("Precipitation rate (mm/day)")
ax.set_ylabel("Precipitation amount (weighted)")
ax.set_title(f"Tropical Precip Amount by Intensity (post-spinup) — R95p = {r95p:.1f}%, R99p = {r99p:.1f}%")
ax.legend()
fig.tight_layout()
save_fig(fig, "r95p_distribution")


# ---- 13. Summary statistics ---------------------------------------------
print("  13. Summary …")
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis("off")

stats_text = f"""
ACE2-EAMv3 piControl — 6-Hourly Extremes Summary (post-spinup)
{'='*60}

  Data:           {n_segs} segments, {n_total_years} years of 6-hourly output
                  Post-spinup (year >= {SPINUP_CUTOFF_YEAR})
  Grid:           {nlat} × {nlon} (lat × lon)

  TEMPERATURE EXTREMES
  All-time max TS:          {np.max(ts_annual_max) - 273.15:.1f} °C
  All-time min TS:          {np.min(ts_annual_min) - 273.15:.1f} °C
  Global-mean ann max TS:   {np.mean(ts_gmax):.1f} ± {np.std(ts_gmax):.2f} °C
  Global-mean ann min TS:   {np.mean(ts_gmin):.1f} ± {np.std(ts_gmin):.2f} °C
  Mean DTR (global):        {np.mean(dtr_global):.2f} K

  PRECIPITATION EXTREMES
  All-time max daily precip:  {np.max(precip_annual_max):.1f} mm/day
  Global 95th pctl (trop):    {p95:.1f} mm/day
  Global 99th pctl (trop):    {p99:.1f} mm/day
  R95p (tropical):            {r95p:.1f}%
  R99p (tropical):            {r99p:.1f}%

  PRESSURE EXTREMES
  All-time min PS (overall):  {np.min(ps_annual_min)/100:.1f} hPa
  All-time min PS (low-elev): {min_low_elev:.1f} hPa  (grid pts with median min PS > 800 hPa)
  Median ann min PS (global): {np.median(ps_annual_min.min(axis=(1,2)))/100:.1f} hPa
  Outlier location:           ({min_lat:.0f}\u00b0, {min_lon:.0f}\u00b0E) = {min_val:.1f} hPa

  SCALING
  CC scaling data points:     {len(cc_medians)} temperature bins
"""

ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment="top", fontfamily="monospace")
fig.tight_layout()
save_fig(fig, "summary_stats")


# ---- Done ----------------------------------------------------------------
print(f"\nDone — saved {_fig_counter} PNGs to {OUTPUT_DIR}/")
