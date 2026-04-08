"""
Probability distribution analysis from 6-hourly ACE2-EAMv3 piControl output.

Focuses on the tails of TS, T_7, and precipitation distributions using the
full 6-hourly resolution across ~1000 years.

============================================================================
SCIENTIFIC BACKGROUND & INTERPRETATION GUIDE
============================================================================

--- Temperature PDFs (Sardeshmukh et al. 2015, J. Climate) ----------------
Atmospheric temperature distributions are NOT Gaussian. They have:
  • Skewness: positive over continents in winter (cold snaps are bounded
    by absolute zero, but warm intrusions can be extreme), negative in
    tropics (bounded above by convective instability threshold)
  • Excess kurtosis: heavy tails relative to Gaussian, especially in
    regions with strong weather variability or regime transitions

We plot PDFs of 6-hourly anomalies (seasonal cycle removed) on a
LOG-SCALE y-axis. This makes the tails visible:
  • Gaussian → parabola on log scale
  • Heavier-than-Gaussian → slower falloff (flatter in tails)
  • Lighter-than-Gaussian → faster falloff (steeper in tails)

Regional decomposition is essential because:
  • Tropics (30°S–30°N): narrow distribution, small σ, nearly Gaussian
  • Mid-latitudes (30–60°): wide, driven by synoptic weather, often skewed
  • Polar (>60°): can be bimodal (ice/no-ice regimes), very heavy tails
  • Continental vs oceanic: land has larger diurnal range and more extremes

TS vs T_7 comparison tests whether the surface skin temperature and the
near-surface atmospheric temperature have consistent variability. T_7 is
the lowest atmospheric layer; TS is the radiative surface temperature.
Over ocean, they should be very similar; over land, TS can decouple
(strong daytime heating, nighttime radiative cooling).

Q-Q plots (quantile-quantile) compare observed quantiles against a
Gaussian reference. Departures from the 1:1 line reveal:
  • S-shaped departure: heavy tails on both sides
  • Upward curve at right end: positive skew (hot tail heavier)
  • Downward curve at left end: cold tail heavier

Skewness and kurtosis maps (Perkins et al. 2012):
  • Skewness > 0: right tail heavier (extreme warm events more intense)
  • Skewness < 0: left tail heavier (extreme cold events more intense)
  • Kurtosis > 0 (excess): heavier tails than Gaussian → more frequent
    extreme events than expected from mean and variance alone
  • Kurtosis < 0: lighter tails, fewer extremes

--- Precipitation PDFs (Wilson & Toumi 2005, Trenberth et al. 2003) --------
Precipitation is zero most of the time and has a heavy right tail.
The standard approaches:

1. Exceedance probability P(precip > x) on LOG-LOG axes:
   This is the complementary CDF (survival function). For precipitation:
   • Exponential tail → straight line on log-linear plot
   • Stretched exponential → curved on log-log
   • Power-law tail → straight line on log-log
   Real precipitation typically shows a stretched exponential
   (Trenberth et al. 2003). If ACE's tail falls off too fast, it
   underestimates extreme rainfall frequency.

2. Gamma distribution fit (Wilks 2011, Statistical Methods in Atmos. Sci.):
   Wet-day precipitation intensity is classically modeled as a gamma
   distribution with shape α and scale β:
   • α < 1: J-shaped (most rain is light drizzle)
   • α ≈ 1: exponential
   • α > 1: peaked, less drizzle
   The fit tells you whether ACE's precipitation statistics are
   physically reasonable.

3. Regional decomposition:
   • Deep tropics (ITCZ): convective, heavy-tailed, high wet-day frequency
   • Subtropics: infrequent but intense events
   • Mid-latitude oceanic: frequent frontal rain, moderate tails
   • Continental interior: seasonal, mixed convective/frontal

============================================================================

Usage:
    micromamba run -n xgns python distributions_6h.py
    micromamba run -n xgns python distributions_6h.py --overwrite
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats as sp_stats

from config import get_6h_segment_dirs, SPINUP_CUTOFF_YEAR, N_WORKERS, DASK_CHUNK_SIZE

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEG_DIRS = get_6h_segment_dirs()

OUTPUT_DIR = Path("figs_distributions_6h")
OUTPUT_DIR.mkdir(exist_ok=True)
_fig_counter = 0

OVERWRITE = "--overwrite" in sys.argv
MAX_SEGMENTS = None

KG_M2_S_TO_MM_DAY = 86400.0
# Convert 6-hourly precip rate to mm/6h for instantaneous intensity
KG_M2_S_TO_MM_6H = 86400.0 / 4.0


def save_fig(fig, name):
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


def skip_fig(name):
    global _fig_counter
    _fig_counter += 1
    path = OUTPUT_DIR / f"{_fig_counter:02d}_{name}.png"
    print(f"    → {path} (exists, skipping computation)")


def section_done(*names):
    if OVERWRITE:
        return False
    for i, name in enumerate(names):
        if not (OUTPUT_DIR / f"{_fig_counter + 1 + i:02d}_{name}.png").exists():
            return False
    return True


def cosine_weights(lat):
    return np.cos(np.deg2rad(lat))


# ---------------------------------------------------------------------------
# Grid info
# ---------------------------------------------------------------------------
ds0 = xr.open_dataset(f"{SEG_DIRS[0]}/6h_surface_TS_predictions.nc", chunks={})
lat = ds0.lat.values
lon = ds0.lon.values
nlat, nlon = len(lat), len(lon)
ds0.close()

segs = SEG_DIRS[:MAX_SEGMENTS] if MAX_SEGMENTS else SEG_DIRS
n_segs = len(segs)

# All figures this script produces
ALL_FIG_NAMES = [
    "ts_raw_pdfs", "t7_raw_pdfs",
    "ts_regional_pdfs", "t7_regional_pdfs", "ts_vs_t7_pdfs",
    "ts_qq_plots", "ts_skewness_map", "ts_kurtosis_map",
    "t7_skewness_map", "t7_kurtosis_map",
    "ts_single_point_pdfs", "ts_seasonal_pdfs",
    "precip_exceedance", "precip_regional_pdfs",
    "precip_gamma_fits", "precip_6h_vs_daily",
    "drizzle_diagnostic",
    "summary",
]
if all((OUTPUT_DIR / f"{i+1:02d}_{n}.png").exists() for i, n in enumerate(ALL_FIG_NAMES)) and not OVERWRITE:
    print("All figures already exist. Use --overwrite to regenerate.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Define regions and single points
# ---------------------------------------------------------------------------
REGIONS = {
    "Deep Tropics\n(10°S–10°N)": ((-10, 10), None),
    "Tropics\n(30°S–30°N)": ((-30, 30), None),
    "NH Mid-lat\n(30–60°N)": ((30, 60), None),
    "SH Mid-lat\n(60–30°S)": ((-60, -30), None),
    "Arctic\n(>60°N)": ((60, 90), None),
    "Antarctic\n(<60°S)": ((-90, -60), None),
}

SINGLE_POINTS = {
    "Sahara (25°N, 10°E)": (25, 10),
    "Amazon (0°, 300°E)": (0, 300),
    "N. Atlantic (50°N, 330°E)": (50, 330),
    "Tibetan Plateau (35°N, 90°E)": (35, 90),
    "S. Ocean (55°S, 180°E)": (-55, 180),
    "Nino-3.4 (0°, 210°E)": (0, 210),
}

# ---------------------------------------------------------------------------
# Pass 1: Accumulate histograms and moments
# ---------------------------------------------------------------------------
# We accumulate:
#   - Per-region histograms of TS and T_7 anomalies (seasonal cycle removed)
#   - Per-gridpoint running moments (mean, var, skew, kurtosis) via Welford
#   - Per-region precipitation histograms (wet-timestep intensity)
#   - Single-point raw value collectors (subsampled)

CACHE_FILE = Path("cache_distributions_6h.npz")

print(f"Processing {n_segs} segments …\n")

# Temperature anomaly bins (K, centered on zero after removing seasonal cycle)
T_BINS = np.linspace(-40, 40, 801)  # 0.1 K resolution
t_centers = 0.5 * (T_BINS[:-1] + T_BINS[1:])

# Raw (absolute) temperature bins (K)
T_RAW_BINS = np.linspace(180, 340, 801)
t_raw_centers = 0.5 * (T_RAW_BINS[:-1] + T_RAW_BINS[1:])

# Precipitation bins (mm/6h, log-spaced for intensity)
P_BINS = np.concatenate([[0], np.logspace(-2, 3, 500)])  # 0.01 to 1000 mm/6h
p_centers = 0.5 * (P_BINS[:-1] + P_BINS[1:])

# Per-region histograms
ts_hist = {name: np.zeros(len(T_BINS) - 1, dtype=np.float64) for name in REGIONS}
t7_hist = {name: np.zeros(len(T_BINS) - 1, dtype=np.float64) for name in REGIONS}
ts_raw_hist = {name: np.zeros(len(T_RAW_BINS) - 1, dtype=np.float64) for name in REGIONS}
t7_raw_hist = {name: np.zeros(len(T_RAW_BINS) - 1, dtype=np.float64) for name in REGIONS}
pr_hist_wet = {name: np.zeros(len(P_BINS) - 1, dtype=np.float64) for name in REGIONS}
pr_total_steps = {name: 0 for name in REGIONS}
pr_wet_steps = {name: 0 for name in REGIONS}

# Per-gridpoint Welford accumulators for skewness/kurtosis
# M1=mean, M2=sum of squared dev, M3=sum of cubed dev, M4=sum of 4th power dev
ts_n = np.zeros((nlat, nlon), dtype=np.float64)
ts_m1 = np.zeros((nlat, nlon), dtype=np.float64)
ts_m2 = np.zeros((nlat, nlon), dtype=np.float64)
ts_m3 = np.zeros((nlat, nlon), dtype=np.float64)
ts_m4 = np.zeros((nlat, nlon), dtype=np.float64)

t7_n = np.zeros((nlat, nlon), dtype=np.float64)
t7_m1 = np.zeros((nlat, nlon), dtype=np.float64)
t7_m2 = np.zeros((nlat, nlon), dtype=np.float64)
t7_m3 = np.zeros((nlat, nlon), dtype=np.float64)
t7_m4 = np.zeros((nlat, nlon), dtype=np.float64)

# Single-point collectors (subsample every 4th timestep = daily)
single_pt_ts = {name: [] for name in SINGLE_POINTS}
single_pt_t7 = {name: [] for name in SINGLE_POINTS}
single_pt_pr = {name: [] for name in SINGLE_POINTS}

# Per-gridpoint drizzle accumulators
drizzle_count = np.zeros((nlat, nlon), dtype=np.float64)  # precip > 0 but < 0.1 mm/6h
wet_count = np.zeros((nlat, nlon), dtype=np.float64)       # precip > 0.01 mm/6h
total_precip_count = np.zeros((nlat, nlon), dtype=np.float64)

# Seasonal climatology: accumulate per-month means (first pass builds this)
ts_month_sum = np.zeros((12, nlat, nlon), dtype=np.float64)
t7_month_sum = np.zeros((12, nlat, nlon), dtype=np.float64)
month_count = np.zeros(12, dtype=np.float64)
ts_clim = np.zeros((12, nlat, nlon), dtype=np.float64)
t7_clim = np.zeros((12, nlat, nlon), dtype=np.float64)

# Pre-compute region masks and single-point indices
region_masks = {}
for name, ((lat0, lat1), _) in REGIONS.items():
    region_masks[name] = (lat >= lat0) & (lat <= lat1)

single_pt_idx = {}
for name, (pt_lat, pt_lon) in SINGLE_POINTS.items():
    j = np.argmin(np.abs(lat - pt_lat))
    i = np.argmin(np.abs(lon - pt_lon))
    single_pt_idx[name] = (j, i)


def welford_update(n, m1, m2, m3, m4, x):
    """Batch Welford update for computing moments from streaming data.
    x is (nlat, nlon) -- one timestep."""
    n_new = n + 1
    delta = x - m1
    delta_n = delta / n_new
    delta_n2 = delta_n * delta_n
    term1 = delta * delta_n * n

    m1_new = m1 + delta_n
    m4_new = m4 + term1 * delta_n2 * (n_new * n_new - 3 * n_new + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3
    m3_new = m3 + term1 * delta_n * (n_new - 2) - 3 * delta_n * m2
    m2_new = m2 + term1

    return n_new, m1_new, m2_new, m3_new, m4_new


cache_loaded = False
if CACHE_FILE.exists() and not OVERWRITE:
    print("  Loading cached statistics …")
    try:
        _cache = np.load(CACHE_FILE, allow_pickle=True)
        reg_names_list = list(REGIONS.keys())
        for idx, rname in enumerate(reg_names_list):
            ts_hist[rname] = _cache[f"ts_hist_{idx}"]
            t7_hist[rname] = _cache[f"t7_hist_{idx}"]
            ts_raw_hist[rname] = _cache[f"ts_raw_hist_{idx}"]
            t7_raw_hist[rname] = _cache[f"t7_raw_hist_{idx}"]
            pr_hist_wet[rname] = _cache[f"pr_hist_wet_{idx}"]
            pr_total_steps[rname] = int(_cache[f"pr_total_{idx}"])
            pr_wet_steps[rname] = int(_cache[f"pr_wet_{idx}"])
        ts_n[:] = _cache["ts_n"]; ts_m1[:] = _cache["ts_m1"]
        ts_m2[:] = _cache["ts_m2"]; ts_m3[:] = _cache["ts_m3"]; ts_m4[:] = _cache["ts_m4"]
        t7_n[:] = _cache["t7_n"]; t7_m1[:] = _cache["t7_m1"]
        t7_m2[:] = _cache["t7_m2"]; t7_m3[:] = _cache["t7_m3"]; t7_m4[:] = _cache["t7_m4"]
        ts_clim[:] = _cache["ts_clim"]; t7_clim[:] = _cache["t7_clim"]
        drizzle_count[:] = _cache["drizzle_count"]
        wet_count[:] = _cache["wet_count"]
        total_precip_count[:] = _cache["total_precip_count"]
        pt_names_list = list(SINGLE_POINTS.keys())
        for idx, pname in enumerate(pt_names_list):
            single_pt_ts[pname] = _cache[f"single_pt_ts_{idx}"]
            single_pt_t7[pname] = _cache[f"single_pt_t7_{idx}"]
            single_pt_pr[pname] = _cache[f"single_pt_pr_{idx}"]
        _cache.close()
        cache_loaded = True
        print(f"  Cache loaded from {CACHE_FILE}")
    except Exception as e:
        print(f"  Cache load failed: {e}, recomputing...")
        cache_loaded = False

if not cache_loaded:
    # === First pass: build seasonal climatology ===
    print("  Pass 1: Building seasonal climatology …")
    for seg_idx, seg_dir in enumerate(segs):
        ts_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_TS_predictions.nc", chunks={"time": 1460}
        ).isel(sample=0)
        t7_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_T7_predictions.nc", chunks={"time": 1460}
        ).isel(sample=0)
        vt = ts_ds.valid_time.compute().values
        n_times = len(vt)
        steps_per_year = 1460

        # Process one year at a time
        for yr in range(n_times // steps_per_year):
            t0 = yr * steps_per_year
            t1 = t0 + steps_per_year
            vt_yr = vt[t0:t1]
            months_yr = np.array([t.month for t in vt_yr])

            # Subsample: use every 10th year for climatology (enough for a smooth cycle)
            if yr % 10 != 0:
                continue

            ts_yr = ts_ds.TS.isel(time=slice(t0, t1)).values
            t7_yr = t7_ds.T_7.isel(time=slice(t0, t1)).values

            for m in range(1, 13):
                mask = months_yr == m
                if mask.sum() > 0:
                    ts_month_sum[m - 1] += np.nanmean(ts_yr[mask], axis=0)
                    t7_month_sum[m - 1] += np.nanmean(t7_yr[mask], axis=0)
                    month_count[m - 1] += 1

        ts_ds.close()
        t7_ds.close()

    # Climatological monthly mean
    ts_clim = ts_month_sum / np.maximum(month_count[:, None, None], 1)
    t7_clim = t7_month_sum / np.maximum(month_count[:, None, None], 1)
    print("    Climatology built.")


    # === Second pass: accumulate anomaly statistics ===
    print("  Pass 2: Accumulating distributions …")

    for seg_idx, seg_dir in enumerate(segs):
        seg_name = seg_dir.split("/")[-2]
        print(f"    Segment {seg_idx}/{n_segs-1}: {seg_name}")

        ts_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_TS_predictions.nc", chunks={"time": 1460}
        ).isel(sample=0)
        t7_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_T7_predictions.nc", chunks={"time": 1460}
        ).isel(sample=0)
        pr_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_surface_precipitation_rate_predictions.nc",
            chunks={"time": 1460},
        ).isel(sample=0)
        vt = ts_ds.valid_time.compute().values
        n_times = len(vt)
        steps_per_year = 1460

        for yr in range(n_times // steps_per_year):
            t0 = yr * steps_per_year
            t1 = t0 + steps_per_year
            vt_yr = vt[t0:t1]

            # Skip years before spinup cutoff
            first_year = vt_yr[0].year if hasattr(vt_yr[0], 'year') else 0
            if first_year < SPINUP_CUTOFF_YEAR:
                continue

            months_yr = np.array([t.month for t in vt_yr])

            ts_yr = ts_ds.TS.isel(time=slice(t0, t1)).values      # (1460, nlat, nlon)
            t7_yr = t7_ds.T_7.isel(time=slice(t0, t1)).values
            pr_yr = pr_ds.surface_precipitation_rate.isel(time=slice(t0, t1)).values * KG_M2_S_TO_MM_6H

            # Compute anomalies by subtracting monthly climatology
            ts_anom = np.empty_like(ts_yr)
            t7_anom = np.empty_like(t7_yr)
            for m in range(1, 13):
                mask = months_yr == m
                if mask.sum() > 0:
                    ts_anom[mask] = ts_yr[mask] - ts_clim[m - 1][np.newaxis]
                    t7_anom[mask] = t7_yr[mask] - t7_clim[m - 1][np.newaxis]

            # --- Regional histograms (subsample every 4th step = daily) ---
            for step in range(0, steps_per_year, 4):
                ts_slice = ts_anom[step]   # (nlat, nlon)
                t7_slice = t7_anom[step]
                pr_slice = pr_yr[step]

                # Raw (absolute) temperature histograms
                ts_raw_slice = ts_yr[step]  # raw, not anomaly
                t7_raw_slice = t7_yr[step]

                for reg_name, lat_mask in region_masks.items():
                    ts_reg = ts_slice[lat_mask, :].ravel()
                    t7_reg = t7_slice[lat_mask, :].ravel()
                    pr_reg = pr_slice[lat_mask, :].ravel()

                    h, _ = np.histogram(ts_reg[np.isfinite(ts_reg)], bins=T_BINS)
                    ts_hist[reg_name] += h
                    h, _ = np.histogram(t7_reg[np.isfinite(t7_reg)], bins=T_BINS)
                    t7_hist[reg_name] += h

                    # Raw temperature histograms
                    ts_raw_reg = ts_raw_slice[lat_mask, :].ravel()
                    h, _ = np.histogram(ts_raw_reg[np.isfinite(ts_raw_reg)], bins=T_RAW_BINS)
                    ts_raw_hist[reg_name] += h
                    t7_raw_reg = t7_raw_slice[lat_mask, :].ravel()
                    h, _ = np.histogram(t7_raw_reg[np.isfinite(t7_raw_reg)], bins=T_RAW_BINS)
                    t7_raw_hist[reg_name] += h

                    pr_total_steps[reg_name] += len(pr_reg)
                    wet = pr_reg[pr_reg > 0.01]  # > 0.01 mm/6h threshold
                    pr_wet_steps[reg_name] += len(wet)
                    if len(wet) > 0:
                        h, _ = np.histogram(wet, bins=P_BINS)
                        pr_hist_wet[reg_name] += h

                # Drizzle accumulation (per-gridpoint)
                wet_mask = pr_slice > 0.01
                drizzle_mask = (pr_slice > 0.01) & (pr_slice < 0.1)
                wet_count += wet_mask
                drizzle_count += drizzle_mask
                total_precip_count += 1

            # --- Per-gridpoint moments (subsample every 8th step) ---
            for step in range(0, steps_per_year, 8):
                ts_n, ts_m1, ts_m2, ts_m3, ts_m4 = welford_update(
                    ts_n, ts_m1, ts_m2, ts_m3, ts_m4, ts_anom[step])
                t7_n, t7_m1, t7_m2, t7_m3, t7_m4 = welford_update(
                    t7_n, t7_m1, t7_m2, t7_m3, t7_m4, t7_anom[step])

            # --- Single-point collectors (every 4th step) ---
            if yr % 5 == 0:  # every 5th year to keep memory manageable
                for pt_name, (j, i) in single_pt_idx.items():
                    single_pt_ts[pt_name].append(ts_anom[::4, j, i].copy())
                    single_pt_t7[pt_name].append(t7_anom[::4, j, i].copy())
                    single_pt_pr[pt_name].append(pr_yr[::4, j, i].copy())

        ts_ds.close()
        t7_ds.close()
        pr_ds.close()

    # Concatenate single-point data
    for name in SINGLE_POINTS:
        single_pt_ts[name] = np.concatenate(single_pt_ts[name]) if single_pt_ts[name] else np.array([])
        single_pt_t7[name] = np.concatenate(single_pt_t7[name]) if single_pt_t7[name] else np.array([])
        single_pt_pr[name] = np.concatenate(single_pt_pr[name]) if single_pt_pr[name] else np.array([])

    # Save cache
    print(f"  Saving cache to {CACHE_FILE} …")
    _cache_dict = {}
    reg_names_list = list(REGIONS.keys())
    for idx, rname in enumerate(reg_names_list):
        _cache_dict[f"ts_hist_{idx}"] = ts_hist[rname]
        _cache_dict[f"t7_hist_{idx}"] = t7_hist[rname]
        _cache_dict[f"ts_raw_hist_{idx}"] = ts_raw_hist[rname]
        _cache_dict[f"t7_raw_hist_{idx}"] = t7_raw_hist[rname]
        _cache_dict[f"pr_hist_wet_{idx}"] = pr_hist_wet[rname]
        _cache_dict[f"pr_total_{idx}"] = np.array(pr_total_steps[rname])
        _cache_dict[f"pr_wet_{idx}"] = np.array(pr_wet_steps[rname])
    _cache_dict["ts_n"] = ts_n; _cache_dict["ts_m1"] = ts_m1
    _cache_dict["ts_m2"] = ts_m2; _cache_dict["ts_m3"] = ts_m3; _cache_dict["ts_m4"] = ts_m4
    _cache_dict["t7_n"] = t7_n; _cache_dict["t7_m1"] = t7_m1
    _cache_dict["t7_m2"] = t7_m2; _cache_dict["t7_m3"] = t7_m3; _cache_dict["t7_m4"] = t7_m4
    _cache_dict["ts_clim"] = ts_clim; _cache_dict["t7_clim"] = t7_clim
    _cache_dict["drizzle_count"] = drizzle_count
    _cache_dict["wet_count"] = wet_count
    _cache_dict["total_precip_count"] = total_precip_count
    pt_names_list = list(SINGLE_POINTS.keys())
    for idx, pname in enumerate(pt_names_list):
        _cache_dict[f"single_pt_ts_{idx}"] = single_pt_ts[pname]
        _cache_dict[f"single_pt_t7_{idx}"] = single_pt_t7[pname]
        _cache_dict[f"single_pt_pr_{idx}"] = single_pt_pr[pname]
    np.savez(CACHE_FILE, **_cache_dict)
    print(f"  Cache saved to {CACHE_FILE}")

# Compute skewness and kurtosis from accumulated moments
with np.errstate(divide="ignore", invalid="ignore"):
    ts_var = ts_m2 / np.maximum(ts_n, 1)
    ts_std = np.sqrt(ts_var)
    ts_skew = (ts_m3 / np.maximum(ts_n, 1)) / np.where(ts_std > 0.01, ts_std**3, np.nan)
    ts_kurt = (ts_m4 / np.maximum(ts_n, 1)) / np.where(ts_var > 0.01, ts_var**2, np.nan) - 3.0

    t7_var = t7_m2 / np.maximum(t7_n, 1)
    t7_std = np.sqrt(t7_var)
    t7_skew = (t7_m3 / np.maximum(t7_n, 1)) / np.where(t7_std > 0.01, t7_std**3, np.nan)
    t7_kurt = (t7_m4 / np.maximum(t7_n, 1)) / np.where(t7_var > 0.01, t7_var**2, np.nan) - 3.0

print(f"\n  Done accumulating. {int(ts_n.mean())} samples per grid point.\n")


# ===========================================================================
# Figures
# ===========================================================================
print("Generating figures …\n")


# ---- 1. Raw TS Temperature PDFs by Region --------------------------------
print("  1. Raw TS PDFs …")
if section_done("ts_raw_pdfs"):
    skip_fig("ts_raw_pdfs")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (reg_name, _) in zip(axes.flat, REGIONS.items()):
        h = ts_raw_hist[reg_name].astype(float)
        total = h.sum()
        if total > 0:
            pdf = h / (total * np.diff(T_RAW_BINS))
            ax.semilogy(t_raw_centers - 273.15, pdf, "k-", lw=1)
            mean_t = np.average(t_raw_centers, weights=h) - 273.15
            std_t = np.sqrt(np.average((t_raw_centers - np.average(t_raw_centers, weights=h))**2, weights=h))
            ax.axvline(mean_t, color="red", ls="--", lw=1, label=f"Mean={mean_t:.1f}\u00b0C")
            ax.set_title(f"{reg_name.replace(chr(10), ' ')}\n\u03c3={std_t:.1f} K", fontsize=11)
        ax.set_xlim(-80, 60)
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel("Surface Temperature (\u00b0C)")
        ax.set_ylabel("PDF")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("TS 6-hourly Absolute Temperature PDFs by Region (post-spinup)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "ts_raw_pdfs")


# ---- 2. Raw T_7 Temperature PDFs by Region ------------------------------
print("  2. Raw T_7 PDFs …")
if section_done("t7_raw_pdfs"):
    skip_fig("t7_raw_pdfs")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (reg_name, _) in zip(axes.flat, REGIONS.items()):
        h = t7_raw_hist[reg_name].astype(float)
        total = h.sum()
        if total > 0:
            pdf = h / (total * np.diff(T_RAW_BINS))
            ax.semilogy(t_raw_centers - 273.15, pdf, "b-", lw=1)
            mean_t = np.average(t_raw_centers, weights=h) - 273.15
            std_t = np.sqrt(np.average((t_raw_centers - np.average(t_raw_centers, weights=h))**2, weights=h))
            ax.axvline(mean_t, color="red", ls="--", lw=1, label=f"Mean={mean_t:.1f}\u00b0C")
            ax.set_title(f"{reg_name.replace(chr(10), ' ')}\n\u03c3={std_t:.1f} K", fontsize=11)
        ax.set_xlim(-80, 60)
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel("Near-Surface Temperature (\u00b0C)")
        ax.set_ylabel("PDF")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("T_7 6-hourly Absolute Temperature PDFs by Region (post-spinup)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "t7_raw_pdfs")


# ---- 3. TS regional PDFs (anomaly, log-scale) ---------------------------
# INTERPRETATION: On a log y-axis, a Gaussian appears as a downward
# parabola. Deviations from this shape in the tails indicate non-Gaussian
# behavior. Heavier tails (flatter on log scale) mean more frequent
# extreme events than a Gaussian model would predict.
print("  3. TS regional PDFs …")
if section_done("ts_regional_pdfs"):
    skip_fig("ts_regional_pdfs")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (reg_name, _) in zip(axes.flat, REGIONS.items()):
        h = ts_hist[reg_name].astype(float)
        total = h.sum()
        if total > 0:
            pdf = h / (total * np.diff(T_BINS))
            ax.semilogy(t_centers, pdf, "k-", lw=1, label="ACE TS")

            # Gaussian reference with same std
            mean_idx = np.average(t_centers, weights=h)
            std_idx = np.sqrt(np.average((t_centers - mean_idx)**2, weights=h))
            gauss = sp_stats.norm.pdf(t_centers, loc=mean_idx, scale=std_idx)
            ax.semilogy(t_centers, gauss, "r--", lw=1.5, alpha=0.7,
                        label=f"Gaussian (\u03c3={std_idx:.1f} K)")

        ax.set_xlim(-25, 25)
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel("TS anomaly (K)")
        ax.set_ylabel("PDF")
        # Add sigma to subtitle
        h_for_title = ts_hist[reg_name].astype(float)
        if h_for_title.sum() > 0:
            sig_val = np.sqrt(np.average((t_centers - np.average(t_centers, weights=h_for_title))**2, weights=h_for_title))
            ax.set_title(f"{reg_name.replace(chr(10), ' ')}\n\u03c3={sig_val:.1f} K", fontsize=11)
        else:
            ax.set_title(reg_name.replace("\n", " "), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("TS 6-hourly Anomaly PDFs (deviation from local monthly climatology, log scale)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "ts_regional_pdfs")


# ---- 4. T_7 regional PDFs -----------------------------------------------
print("  4. T_7 regional PDFs …")
if section_done("t7_regional_pdfs"):
    skip_fig("t7_regional_pdfs")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (reg_name, _) in zip(axes.flat, REGIONS.items()):
        h = t7_hist[reg_name].astype(float)
        total = h.sum()
        if total > 0:
            pdf = h / (total * np.diff(T_BINS))
            ax.semilogy(t_centers, pdf, "b-", lw=1, label="ACE T_7")

            mean_idx = np.average(t_centers, weights=h)
            std_idx = np.sqrt(np.average((t_centers - mean_idx)**2, weights=h))
            gauss = sp_stats.norm.pdf(t_centers, loc=mean_idx, scale=std_idx)
            ax.semilogy(t_centers, gauss, "r--", lw=1.5, alpha=0.7,
                        label=f"Gaussian (\u03c3={std_idx:.1f} K)")

        ax.set_xlim(-25, 25)
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel("T_7 anomaly (K)")
        ax.set_ylabel("PDF")
        h_for_title = t7_hist[reg_name].astype(float)
        if h_for_title.sum() > 0:
            sig_val = np.sqrt(np.average((t_centers - np.average(t_centers, weights=h_for_title))**2, weights=h_for_title))
            ax.set_title(f"{reg_name.replace(chr(10), ' ')}\n\u03c3={sig_val:.1f} K", fontsize=11)
        else:
            ax.set_title(reg_name.replace("\n", " "), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("T_7 6-hourly Anomaly PDFs (deviation from local monthly climatology, log scale)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "t7_regional_pdfs")


# ---- 5. TS vs T_7 comparison -------------------------------------------
# INTERPRETATION: Over ocean, TS and T_7 should be nearly identical.
# Over land, TS (skin temp) can be more extreme (wider tails) due to
# direct solar heating and radiative cooling. If T_7 tails are heavier
# than TS, something may be wrong with the boundary layer coupling.
print("  5. TS vs T_7 comparison …")
if section_done("ts_vs_t7_pdfs"):
    skip_fig("ts_vs_t7_pdfs")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (reg_name, _) in zip(axes.flat, REGIONS.items()):
        h_ts = ts_hist[reg_name].astype(float)
        h_t7 = t7_hist[reg_name].astype(float)
        if h_ts.sum() > 0 and h_t7.sum() > 0:
            pdf_ts = h_ts / (h_ts.sum() * np.diff(T_BINS))
            pdf_t7 = h_t7 / (h_t7.sum() * np.diff(T_BINS))
            ax.semilogy(t_centers, pdf_ts, "k-", lw=1.2, label="TS")
            ax.semilogy(t_centers, pdf_t7, "b-", lw=1.2, alpha=0.7, label="T_7")
        ax.set_xlim(-25, 25); ax.set_ylim(1e-6, 1)
        ax.set_xlabel("Anomaly (K)"); ax.set_ylabel("PDF")
        h_ts_title = ts_hist[reg_name].astype(float)
        h_t7_title = t7_hist[reg_name].astype(float)
        sig_ts = np.sqrt(np.average((t_centers - np.average(t_centers, weights=h_ts_title))**2, weights=h_ts_title)) if h_ts_title.sum() > 0 else 0
        sig_t7 = np.sqrt(np.average((t_centers - np.average(t_centers, weights=h_t7_title))**2, weights=h_t7_title)) if h_t7_title.sum() > 0 else 0
        ax.set_title(f"{reg_name.replace(chr(10), ' ')}\n\u03c3_TS={sig_ts:.1f} K, \u03c3_T7={sig_t7:.1f} K", fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("TS vs T_7: Anomaly from Local Monthly Climatology \u2014 Surface Decoupling", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "ts_vs_t7_pdfs")


# ---- 6. Q-Q plots at single points --------------------------------------
# INTERPRETATION: If the data were Gaussian, points would lie on the
# diagonal. Departures in the tails reveal non-Gaussian behavior.
# Upper-right departure: positive extreme tail heavier than Gaussian.
# Lower-left departure: negative extreme tail heavier than Gaussian.
print("  6. Q-Q plots …")
if section_done("ts_qq_plots"):
    skip_fig("ts_qq_plots")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (pt_name, _) in zip(axes.flat, SINGLE_POINTS.items()):
        data = single_pt_ts[pt_name]
        if len(data) > 100:
            data_sorted = np.sort(data)
            n = len(data_sorted)
            # Theoretical quantiles
            probs = (np.arange(1, n + 1) - 0.5) / n
            theo = sp_stats.norm.ppf(probs) * np.std(data)

            # Subsample for plotting
            step = max(1, n // 2000)
            ax.plot(theo[::step], data_sorted[::step], "k.", ms=1, alpha=0.5)
            lim = max(abs(data_sorted.min()), abs(data_sorted.max()), abs(theo.min()), abs(theo.max()))
            ax.plot([-lim, lim], [-lim, lim], "r-", lw=1.5, label="Gaussian ref")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

        ax.set_xlabel("Gaussian quantiles (K)")
        ax.set_ylabel("TS anomaly quantiles (K)")
        ax.set_title(pt_name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle("TS Q-Q Plots vs Gaussian at Selected Locations", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "ts_qq_plots")


# ---- 7-8. Skewness maps -------------------------------------------------
# INTERPRETATION:
# Skewness > 0 (red): warm extremes more intense than cold extremes
#   Expected: continental winter (cold bounded, warm intrusions unbounded)
# Skewness < 0 (blue): cold extremes more intense
#   Expected: tropical ocean (bounded by convective threshold above)
print("  7-8. Skewness maps …")
if section_done("ts_skewness_map", "ts_kurtosis_map"):
    for name in ["ts_skewness_map", "ts_kurtosis_map"]:
        skip_fig(name)
else:
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(lon, lat, ts_skew, cmap="RdBu_r", shading="auto", vmin=-1.5, vmax=1.5)
    ax.set_title("TS 6-hourly Anomaly Skewness (>0: heavy warm tail, <0: heavy cold tail)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="Skewness")
    fig.tight_layout()
    save_fig(fig, "ts_skewness_map")

    # Kurtosis map
    # INTERPRETATION: Excess kurtosis > 0 means heavier tails than Gaussian.
    # High kurtosis regions produce more frequent extreme events than
    # the mean and variance alone would suggest.
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(lon, lat, ts_kurt, cmap="RdBu_r", shading="auto", vmin=-2, vmax=5)
    ax.set_title("TS 6-hourly Anomaly Excess Kurtosis (>0: heavier tails than Gaussian)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="Excess Kurtosis")
    fig.tight_layout()
    save_fig(fig, "ts_kurtosis_map")


# ---- 9-10. T_7 skewness/kurtosis maps ----------------------------------
print("  9-10. T_7 skewness/kurtosis maps …")
if section_done("t7_skewness_map", "t7_kurtosis_map"):
    for name in ["t7_skewness_map", "t7_kurtosis_map"]:
        skip_fig(name)
else:
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(lon, lat, t7_skew, cmap="RdBu_r", shading="auto", vmin=-1.5, vmax=1.5)
    ax.set_title("T_7 6-hourly Anomaly Skewness")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="Skewness")
    fig.tight_layout()
    save_fig(fig, "t7_skewness_map")

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(lon, lat, t7_kurt, cmap="RdBu_r", shading="auto", vmin=-2, vmax=5)
    ax.set_title("T_7 6-hourly Anomaly Excess Kurtosis")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="Excess Kurtosis")
    fig.tight_layout()
    save_fig(fig, "t7_kurtosis_map")


# ---- 11. Single-point PDFs ----------------------------------------------
print("  11. Single-point PDFs …")
if section_done("ts_single_point_pdfs"):
    skip_fig("ts_single_point_pdfs")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (pt_name, _) in zip(axes.flat, SINGLE_POINTS.items()):
        data = single_pt_ts[pt_name]
        if len(data) > 100:
            h, _ = np.histogram(data, bins=T_BINS)
            pdf = h.astype(float) / (h.sum() * np.diff(T_BINS))
            ax.semilogy(t_centers, pdf, "k-", lw=1, label="TS")

            std = np.std(data)
            gauss = sp_stats.norm.pdf(t_centers, loc=np.mean(data), scale=std)
            ax.semilogy(t_centers, gauss, "r--", lw=1.5, alpha=0.7, label=f"Gaussian (σ={std:.1f})")

            skew = sp_stats.skew(data)
            kurt = sp_stats.kurtosis(data)
            ax.set_title(f"{pt_name}\nγ₁={skew:.2f}, κ={kurt:.2f}", fontsize=10)
        else:
            ax.set_title(pt_name, fontsize=10)

        ax.set_xlim(-25, 25); ax.set_ylim(1e-6, 1)
        ax.set_xlabel("TS anomaly (K)"); ax.set_ylabel("PDF")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("TS PDFs at Selected Locations (γ₁=skewness, κ=excess kurtosis)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "ts_single_point_pdfs")


# ---- 12. Seasonal PDFs (DJF vs JJA) at selected points ------------------
# INTERPRETATION: Temperature distributions shift and change shape with
# season. In winter, distributions are often wider (more variability)
# and more positively skewed (warm intrusions into cold background).
# In summer, distributions can be negatively skewed (cool intrusions).
print("  12. Seasonal PDFs …")
if section_done("ts_seasonal_pdfs"):
    skip_fig("ts_seasonal_pdfs")
else:
    # We don't have season-separated histograms, so use single-point data
    # which has enough samples. We need to know the month for each sample.
    # Since we subsampled every 5th year, every 4th step, the seasonal info
    # is embedded in the order. Approximate: split by index within year.
    # Each year has 365 daily values; DJF ≈ first 60 + last 30, JJA ≈ 150-270
    # This is a rough approximation since years start at different months.
    # For a clean seasonal split we'd need to store months — use what we have.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (pt_name, _) in zip(axes.flat, SINGLE_POINTS.items()):
        data = single_pt_ts[pt_name]
        if len(data) > 365:
            # Approximate seasonal split: assume data repeats yearly with 365 daily samples
            n_years_pt = len(data) // 365
            if n_years_pt > 0:
                data_reshaped = data[:n_years_pt * 365].reshape(n_years_pt, 365)
                # DJF: days 0-59 (Jan-Feb) and 335-365 (Dec)
                djf = np.concatenate([data_reshaped[:, :59], data_reshaped[:, 335:]], axis=1).ravel()
                # JJA: days 152-243
                jja = data_reshaped[:, 152:243].ravel()

                for d, label, color in [(djf, "DJF", "blue"), (jja, "JJA", "red")]:
                    h, _ = np.histogram(d, bins=T_BINS)
                    if h.sum() > 0:
                        pdf = h.astype(float) / (h.sum() * np.diff(T_BINS))
                        ax.semilogy(t_centers, pdf, "-", color=color, lw=1.2, label=label)

        ax.set_xlim(-25, 25); ax.set_ylim(1e-6, 0.5)
        ax.set_xlabel("TS anomaly (K)"); ax.set_ylabel("PDF")
        ax.set_title(pt_name, fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("TS Seasonal PDFs: DJF vs JJA at Selected Locations", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "ts_seasonal_pdfs")


# ---- 13. Precipitation exceedance probability ----------------------------
# INTERPRETATION: This is 1 - CDF (survival function) on log-log axes.
# Shows the probability that precipitation exceeds a given intensity.
# * Exponential tail: straight line on log-linear (curved on log-log)
# * Power-law tail: straight line on log-log
# * Real precip is usually stretched exponential (curves on both)
# The 6-hourly resolution captures more extreme instantaneous rates
# than daily averages would.
print("  13. Precipitation exceedance …")
if section_done("precip_exceedance"):
    skip_fig("precip_exceedance")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (reg_name, _) in zip(axes.flat, REGIONS.items()):
        h = pr_hist_wet[reg_name].astype(float)
        total = h.sum()
        if total > 0:
            # CDF and survival function
            cdf = np.cumsum(h) / total
            survival = 1.0 - cdf
            # Only plot where survival > 0
            valid = survival > 0
            ax.loglog(p_centers[valid], survival[valid], "b-", lw=1.2)

            # Mark key thresholds
            for thresh, label in [(1, "1 mm/6h"), (10, "10"), (50, "50"), (100, "100")]:
                idx = np.argmin(np.abs(p_centers - thresh))
                if valid[idx]:
                    ax.axvline(thresh, color="gray", ls=":", alpha=0.3)

            # Wet fraction
            wet_frac = pr_wet_steps[reg_name] / max(pr_total_steps[reg_name], 1)
            ax.set_title(f"{reg_name.replace(chr(10), ' ')}\n"
                         f"Wet frac: {wet_frac:.2%}", fontsize=10)

        ax.set_xlabel("Precip intensity (mm/6h)")
        ax.set_ylabel("P(precip > x)")
        ax.set_xlim(0.01, 500); ax.set_ylim(1e-8, 1)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("Precipitation Exceedance Probability (6-hourly, wet timesteps only)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "precip_exceedance")


# ---- 14. Precipitation regional PDFs ------------------------------------
# INTERPRETATION: The shape of the wet-day intensity PDF reveals the
# character of precipitation:
# * Steep falloff: mostly light rain, few extremes
# * Long tail: frequent intense events (tropical convection)
# * Bimodal: mixed convective + stratiform regimes
print("  14. Precipitation regional PDFs …")
if section_done("precip_regional_pdfs"):
    skip_fig("precip_regional_pdfs")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (reg_name, _) in zip(axes.flat, REGIONS.items()):
        h = pr_hist_wet[reg_name].astype(float)
        total = h.sum()
        if total > 0:
            bin_widths = np.diff(P_BINS)
            pdf = h / (total * bin_widths)
            ax.loglog(p_centers, pdf, "b-", lw=1.2)

        ax.set_xlabel("Precip intensity (mm/6h)")
        ax.set_ylabel("PDF")
        ax.set_title(reg_name.replace("\n", " "), fontsize=11)
        ax.set_xlim(0.01, 500); ax.set_ylim(1e-8, 10)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("Wet-Timestep Precipitation Intensity PDFs (6-hourly)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "precip_regional_pdfs")


# ---- 15. Precipitation gamma fits ---------------------------------------
# INTERPRETATION: Wet-day precipitation is classically modeled as gamma.
# Shape alpha < 1: J-shaped (drizzle-dominated)
# Shape alpha ~ 1: exponential
# Shape alpha > 1: peaked distribution
# If the gamma fit is poor in the tails, the model's precipitation
# extremes deviate from classical expectations.
print("  15. Precipitation gamma fits …")
if section_done("precip_gamma_fits"):
    skip_fig("precip_gamma_fits")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (reg_name, _) in zip(axes.flat, REGIONS.items()):
        h = pr_hist_wet[reg_name].astype(float)
        total = h.sum()
        if total > 100:
            bin_widths = np.diff(P_BINS)
            pdf = h / (total * bin_widths)

            # Estimate gamma params from histogram moments
            mean_p = np.sum(p_centers * h) / total
            var_p = np.sum((p_centers - mean_p)**2 * h) / total
            if var_p > 0 and mean_p > 0:
                beta = var_p / mean_p  # scale
                alpha = mean_p / beta  # shape

                gamma_pdf = sp_stats.gamma.pdf(p_centers, a=alpha, scale=beta)
                ax.loglog(p_centers, pdf, "b-", lw=1.2, label="ACE data")
                ax.loglog(p_centers, gamma_pdf, "r--", lw=1.5, alpha=0.7,
                          label=f"Gamma (α={alpha:.2f}, β={beta:.1f})")
                ax.legend(fontsize=8)

        ax.set_xlabel("Precip (mm/6h)"); ax.set_ylabel("PDF")
        ax.set_title(reg_name.replace("\n", " "), fontsize=11)
        ax.set_xlim(0.01, 500); ax.set_ylim(1e-8, 10)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("Precipitation PDF with Gamma Distribution Fit", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "precip_gamma_fits")


# ---- 16. 6-hourly vs daily-equivalent precip extremes --------------------
# INTERPRETATION: 6-hourly data captures more extreme instantaneous
# rates than daily averages. The ratio of 6h max to daily max tells
# you how much extreme information is lost by temporal averaging.
# Typical ratio is 2-4x for convective regions.
print("  16. 6h vs daily precip at single points …")
if section_done("precip_6h_vs_daily"):
    skip_fig("precip_6h_vs_daily")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (pt_name, _) in zip(axes.flat, SINGLE_POINTS.items()):
        data = single_pt_pr[pt_name]
        if len(data) > 100:
            # "Daily" = mean of 4 consecutive 6h values
            n_days = len(data) // 4
            if n_days > 0:
                data_6h = data[:n_days * 4]
                data_daily = data_6h.reshape(n_days, 4).mean(axis=1)

                # Exceedance for both
                for d, label, color in [(data_6h, "6-hourly", "blue"), (data_daily, "Daily mean", "red")]:
                    wet = d[d > 0.01]
                    if len(wet) > 10:
                        sorted_w = np.sort(wet)[::-1]
                        n = len(sorted_w)
                        exc = np.arange(1, n + 1) / n
                        step = max(1, n // 1000)
                        ax.loglog(sorted_w[::step], exc[::step], "-", color=color, lw=1.2, label=label)

                # Ratio annotation
                p99_6h = np.percentile(data_6h[data_6h > 0.01], 99) if (data_6h > 0.01).sum() > 100 else 0
                p99_daily = np.percentile(data_daily[data_daily > 0.01], 99) if (data_daily > 0.01).sum() > 100 else 0
                if p99_daily > 0:
                    ax.set_title(f"{pt_name}\nP99 ratio: {p99_6h/p99_daily:.1f}x", fontsize=10)
                else:
                    ax.set_title(pt_name, fontsize=10)
        else:
            ax.set_title(pt_name, fontsize=10)

        ax.set_xlabel("Precip (mm/6h)"); ax.set_ylabel("Exceedance probability")
        ax.set_xlim(0.01, 500); ax.set_ylim(1e-5, 1)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fig.suptitle("6-Hourly vs Daily Precipitation Exceedance", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "precip_6h_vs_daily")


# ---- 17. Drizzle diagnostic ---------------------------------------------
print("  17. Drizzle diagnostic …")
if section_done("drizzle_diagnostic"):
    skip_fig("drizzle_diagnostic")
else:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    with np.errstate(divide="ignore", invalid="ignore"):
        wet_frac = wet_count / np.maximum(total_precip_count, 1)
        drizzle_frac = drizzle_count / np.maximum(wet_count, 1)

    im = axes[0].pcolormesh(lon, lat, wet_frac * 100, cmap="Blues", shading="auto", vmin=0, vmax=100)
    axes[0].set_title("Wet Fraction (% of 6h timesteps with precip > 0.01 mm/6h)")
    axes[0].set_xlabel("Lon"); axes[0].set_ylabel("Lat")
    plt.colorbar(im, ax=axes[0], label="%")

    im = axes[1].pcolormesh(lon, lat, drizzle_frac * 100, cmap="YlOrRd", shading="auto", vmin=0, vmax=100)
    axes[1].set_title("Drizzle Fraction (% of wet timesteps with precip < 0.1 mm/6h)")
    axes[1].set_xlabel("Lon"); axes[1].set_ylabel("Lat")
    plt.colorbar(im, ax=axes[1], label="%")

    fig.suptitle("Precipitation Drizzle Diagnostic (6-hourly, post-spinup)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "drizzle_diagnostic")


# ---- 18. Summary ---------------------------------------------------------
print("  18. Summary …")

fig, ax = plt.subplots(figsize=(12, 10))
ax.axis("off")

w = cosine_weights(lat)
ts_skew_global = np.nanmean(ts_skew * w[:, None]) / np.nanmean(w)
ts_kurt_global = np.nanmean(ts_kurt * w[:, None]) / np.nanmean(w)
t7_skew_global = np.nanmean(t7_skew * w[:, None]) / np.nanmean(w)
t7_kurt_global = np.nanmean(t7_kurt * w[:, None]) / np.nanmean(w)

summary = f"""
ACE2-EAMv3 piControl — 6-Hourly Distribution Analysis
{'='*60}

  Segments:          {n_segs}
  Samples/gridpoint: ~{int(ts_n.mean())}

  TEMPERATURE (TS)
  Global mean skewness:      {ts_skew_global:+.3f}  (0 = symmetric)
  Global mean excess kurt:   {ts_kurt_global:+.3f}  (0 = Gaussian)
  Max skewness:              {np.nanmax(ts_skew):+.2f}
  Min skewness:              {np.nanmin(ts_skew):+.2f}
  Max excess kurtosis:       {np.nanmax(ts_kurt):+.2f}

  TEMPERATURE (T_7)
  Global mean skewness:      {t7_skew_global:+.3f}
  Global mean excess kurt:   {t7_kurt_global:+.3f}

  PRECIPITATION
  Tropical wet fraction:     {pr_wet_steps.get(list(REGIONS.keys())[1], 0) / max(pr_total_steps.get(list(REGIONS.keys())[1], 0), 1):.2%}
  Arctic wet fraction:       {pr_wet_steps.get(list(REGIONS.keys())[4], 0) / max(pr_total_steps.get(list(REGIONS.keys())[4], 0), 1):.2%}
"""

ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
        verticalalignment="top", fontfamily="monospace")
fig.tight_layout()
save_fig(fig, "summary")


# ---- Done ----------------------------------------------------------------
print(f"\nDone — saved {_fig_counter} PNGs to {OUTPUT_DIR}/")
