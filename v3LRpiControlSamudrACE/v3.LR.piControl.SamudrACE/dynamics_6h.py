"""
Dynamical diagnostics from 6-hourly ACE2-EAMv3 piControl output.

Focuses on weather/storm dynamics that require sub-daily resolution:
  1. Storm tracks — bandpass-filtered PS variance (Blackmon 1976)
  2. Eddy kinetic energy (EKE) from U_7/V_7 high-pass anomalies
  3. Sea ice proxy — ocean_fraction seasonal cycle, extent, trends
  4. Wind speed Weibull distribution fits

============================================================================
SCIENTIFIC BACKGROUND & INTERPRETATION GUIDE
============================================================================

--- 1. Storm Tracks (Blackmon 1976, Chang et al. 2002) ---------------------
Extratropical storm tracks are diagnosed by the variance of bandpass-
filtered (2–6 day) sea-level pressure. High variance = frequent passage
of synoptic-scale cyclones and anticyclones.

Expected patterns from observations/reanalysis:
  • North Atlantic storm track: maximum from Newfoundland to Iceland
  • North Pacific storm track: maximum from Japan to Gulf of Alaska
  • Southern Hemisphere: nearly zonally symmetric band at ~50°S
  • Weak or absent in tropics (different dynamics there)

If ACE produces storm tracks that are:
  • Too weak: the model under-represents extratropical weather variability
  • Wrong location: jet streams are displaced (common in coarse models)
  • Too zonally symmetric in NH: model lacks land-sea contrast effects

We also look at seasonal variation: NH storm tracks are stronger in
DJF (winter), SH storm tracks are more seasonally uniform.

--- 2. Eddy Kinetic Energy (Lorenz 1955, Orlanski & Katzfey 1991) ---------
EKE = ½(u'² + v'²) where primes are deviations from the time mean
(here we use a 10-day high-pass to isolate synoptic eddies).

EKE measures the kinetic energy in weather disturbances. It should:
  • Peak in the upper troposphere at jet level (we only have near-surface
    U_7/V_7, so we see the surface signature)
  • Collocate with storm tracks but extend slightly downstream
  • Show a strong seasonal cycle in the NH, weaker in the SH

Zonal-mean EKE should peak at ~40–50° in both hemispheres. If EKE is
unrealistically low, ACE's winds lack synoptic-scale variability — the
model produces a "smooth" atmosphere without realistic weather.

--- 3. Sea Ice from Ocean Fraction (Notz & SIMIP 2020) --------------------
ACE outputs `ocean_fraction` at 6-hourly resolution. Since ice_fraction
= 1 - ocean_fraction - land_fraction, we can infer sea ice. In a
piControl, sea ice extent should be:
  • Stable over centuries (no long-term trend)
  • Arctic: ~6 M km² in September (minimum), ~14 M km² in March (maximum)
  • Antarctic: ~3 M km² in February (minimum), ~18 M km² in September (max)

We plot:
  • Monthly climatology of sea ice extent (seasonal cycle)
  • Timeseries of annual min/max extent (stability check)
  • Spatial map of ice edge (mean and variability)

If ocean_fraction drifts, it indicates the coupled surface energy balance
is not equilibrated — a critical diagnostic for a piControl run.

--- 4. Wind Speed Distribution (Justus et al. 1978) -----------------------
Near-surface wind speed typically follows a Weibull distribution:
  f(v) = (k/c)(v/c)^(k-1) exp(-(v/c)^k)

The shape parameter k indicates:
  • k ≈ 2: Rayleigh distribution (typical over open ocean)
  • k > 3: narrow distribution, steady winds (trade wind regions)
  • k < 1.5: highly variable winds (continental interiors)

The scale parameter c is related to mean wind speed. We fit Weibull
parameters at each grid point and compare the spatial patterns to
expected climatology. This is also directly relevant for wind energy
resource assessment.

============================================================================

Usage:
    micromamba run -n xgns python dynamics_6h.py
    micromamba run -n xgns python dynamics_6h.py --overwrite
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import butter, sosfiltfilt

from config import (get_6h_segment_dirs, get_land_fraction_path,
                    SPINUP_CUTOFF_YEAR, N_WORKERS, DASK_CHUNK_SIZE)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEG_DIRS = get_6h_segment_dirs()
LAND_FRAC_PATH = get_land_fraction_path()

OUTPUT_DIR = Path("figs_dynamics_6h")
OUTPUT_DIR.mkdir(exist_ok=True)
_fig_counter = 0

OVERWRITE = "--overwrite" in sys.argv

# How many segments to process (None = all). Storm tracks and EKE are
# expensive, so you can start with fewer segments for a quick look.
MAX_SEGMENTS = None

# Cache file for accumulated arrays (avoids reprocessing segments)
CACHE_FILE = Path("cache_dynamics_6h.npz")


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


def cosine_weights(lat):
    return np.cos(np.deg2rad(lat))


def bandpass_filter(data, lowcut_days, highcut_days, fs=4.0, order=3):
    """Butterworth bandpass filter. fs = samples/day (4 for 6-hourly).
    Keeps periods between highcut_days and lowcut_days."""
    nyq = 0.5 * fs
    low = 1.0 / lowcut_days / nyq   # low freq edge (long period)
    high = 1.0 / highcut_days / nyq  # high freq edge (short period)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=0)


def highpass_filter(data, cutoff_days, fs=4.0, order=3):
    """Butterworth high-pass filter. Removes periods > cutoff_days."""
    nyq = 0.5 * fs
    freq = 1.0 / cutoff_days / nyq
    sos = butter(order, freq, btype="high", output="sos")
    return sosfiltfilt(sos, data, axis=0)


# ---------------------------------------------------------------------------
# Get grid info and land fraction
# ---------------------------------------------------------------------------
ds0 = xr.open_dataset(f"{SEG_DIRS[0]}/6h_surface_TS_predictions.nc", chunks={})
lat = ds0.lat.values
lon = ds0.lon.values
nlat, nlon = len(lat), len(lon)
ds0.close()

# Load land fraction for sea ice calculation
try:
    land_ds = xr.open_dataset(LAND_FRAC_PATH)
    land_frac = land_ds.land_fraction.isel(time=0).values  # (lat, lon)
    if land_frac.ndim == 3:
        land_frac = land_frac[0]
    land_ds.close()
except Exception:
    print("  ⚠ Could not load land_fraction, using ocean_fraction directly")
    land_frac = None

w = cosine_weights(lat)

segs = SEG_DIRS[:MAX_SEGMENTS] if MAX_SEGMENTS else SEG_DIRS
n_segs = len(segs)

# Check if all figures already exist — skip expensive computation if so
ALL_FIG_NAMES = [
    "storm_tracks", "storm_tracks_zonal", "eke", "storm_vs_eke",
    "sea_ice_climatology", "sea_ice_extent_timeseries", "sea_ice_seasonal_cycle",
    "wind_speed_maps", "wind_speed_distributions", "summary",
]
all_exist = all(
    (OUTPUT_DIR / f"{i+1:02d}_{name}.png").exists()
    for i, name in enumerate(ALL_FIG_NAMES)
)
SKIP_COMPUTATION = all_exist and not OVERWRITE

if SKIP_COMPUTATION:
    print("All figures already exist. Use --overwrite to regenerate.")
    print(f"  Figures in: {OUTPUT_DIR}/")
    sys.exit(0)

print(f"Processing {n_segs} segments …\n")


# ===========================================================================
# Pass 1: Accumulate statistics segment by segment
# ===========================================================================

# Accumulators for storm tracks (running variance of bandpass PS)
storm_track_var_sum = np.zeros((nlat, nlon), dtype=np.float64)
storm_track_var_djf_sum = np.zeros((nlat, nlon), dtype=np.float64)
storm_track_var_jja_sum = np.zeros((nlat, nlon), dtype=np.float64)
storm_track_n = 0
storm_track_n_djf = 0
storm_track_n_jja = 0

# EKE accumulators
eke_sum = np.zeros((nlat, nlon), dtype=np.float64)
eke_n = 0

# Wind speed: accumulate mean/std via running sums (avoids per-gridpoint
# histograms which are extremely slow).  Regional histograms are kept for
# distribution plots.
wind_speed_bins = np.linspace(0, 40, 161)  # 0.25 m/s bins
wspd_sum = np.zeros((nlat, nlon), dtype=np.float64)
wspd_sq_sum = np.zeros((nlat, nlon), dtype=np.float64)
wspd_n = 0

wind_regions = {
    "Tropical Pacific": ((-10, 10), (150, 220)),
    "N. Atlantic Storm Track": ((40, 60), (300, 350)),
    "Southern Ocean": ((-65, -50), (0, 360)),
    "Sahara Desert": ((15, 30), (0, 40)),
    "Trade Wind Belt": ((10, 25), (300, 350)),
    "Southern Plains": ((30, 40), (250, 270)),
}
wind_hist_regional = {name: np.zeros(len(wind_speed_bins) - 1) for name in wind_regions}

# Sea ice: monthly ocean fraction → ice fraction
# Accumulate monthly means per calendar month across all segments
ocean_frac_monthly_sum = np.zeros((12, nlat, nlon), dtype=np.float64)
ocean_frac_monthly_n = np.zeros(12, dtype=int)

# Annual min/max sea ice extent timeseries
all_ice_extent_annual_max = []
all_ice_extent_annual_min = []
all_ice_extent_arctic_max = []
all_ice_extent_arctic_min = []
all_ice_extent_antarctic_max = []
all_ice_extent_antarctic_min = []

# ---------------------------------------------------------------------------
# Check for cached accumulated arrays
# ---------------------------------------------------------------------------
_loaded_cache = False
if CACHE_FILE.exists() and not OVERWRITE:
    print(f"Loading cached accumulators from {CACHE_FILE} …")
    try:
        _cache = np.load(CACHE_FILE, allow_pickle=True)
        storm_track_var_sum = _cache["storm_track_var_sum"]
        storm_track_var_djf_sum = _cache["storm_track_var_djf_sum"]
        storm_track_var_jja_sum = _cache["storm_track_var_jja_sum"]
        storm_track_n = int(_cache["storm_track_n"])
        storm_track_n_djf = int(_cache["storm_track_n_djf"])
        storm_track_n_jja = int(_cache["storm_track_n_jja"])
        eke_sum = _cache["eke_sum"]
        eke_n = int(_cache["eke_n"])
        wspd_sum = _cache["wspd_sum"]
        wspd_sq_sum = _cache["wspd_sq_sum"]
        wspd_n = int(_cache["wspd_n"])
        _regional_dict = _cache["wind_hist_regional"].item()
        for k in wind_hist_regional:
            if k in _regional_dict:
                wind_hist_regional[k] = _regional_dict[k]
        _loaded_cache = True
        print(f"  Cache loaded: {storm_track_n} storm-track yrs, {eke_n} EKE yrs, {wspd_n} wind samples")
        del _cache
    except Exception as e:
        print(f"  Cache load failed ({e}), reprocessing …")
        _loaded_cache = False

if not _loaded_cache:
    for seg_idx, seg_dir in enumerate(segs):
        seg_name = seg_dir.split("/")[-2]
        print(f"  Segment {seg_idx}/{n_segs-1}: {seg_name}")

        # ---- Storm tracks: bandpass-filtered PS variance --------------------
        # Process in chunks of ~1 year (1460 6-hourly steps) with padding
        # for the filter. We need at least ~30 days of padding on each side.
        print("    Loading PS …")
        ps_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_PS_predictions.nc", chunks={"time": 1460}
        ).isel(sample=0)
        vt = ps_ds.valid_time.compute().values
        n_times = len(vt)
        steps_per_year = 1460

        # Check if the entire segment is before spinup cutoff
        last_year = vt[-1].astype("datetime64[Y]").astype(int) + 1970 if hasattr(vt[-1], 'astype') else 9999
        if last_year < SPINUP_CUTOFF_YEAR:
            print(f"    Skipping (before spinup cutoff year {SPINUP_CUTOFF_YEAR})")
            ps_ds.close()
            continue

        # We process storm tracks / EKE in yearly chunks with overlap
        pad = 120  # 30 days × 4 steps/day padding for filter ringing
        n_full_years = n_times // steps_per_year

        for yr in range(n_full_years):
            t0 = yr * steps_per_year
            t1 = t0 + steps_per_year

            # Skip years before spinup cutoff
            vt_yr = vt[t0:t1]
            try:
                yr_val = vt_yr[0].astype("datetime64[Y]").astype(int) + 1970
                if yr_val < SPINUP_CUTOFF_YEAR:
                    continue
            except Exception:
                pass

            # Add padding but clip to data bounds
            t0_pad = max(0, t0 - pad)
            t1_pad = min(n_times, t1 + pad)

            ps_chunk = ps_ds.PS.isel(time=slice(t0_pad, t1_pad)).values  # (T, lat, lon)

            # Bandpass filter 2–6 days (synoptic band)
            ps_bp = bandpass_filter(ps_chunk, lowcut_days=6, highcut_days=2)

            # Trim padding to get the actual year
            trim_start = t0 - t0_pad
            trim_end = trim_start + steps_per_year
            ps_bp_yr = ps_bp[trim_start:trim_end]

            # Variance of bandpass-filtered PS
            storm_track_var_sum += np.var(ps_bp_yr, axis=0)
            storm_track_n += 1

            # Seasonal: DJF = months 12,1,2 → steps 0-360 approx for a year
            # starting in March. For simplicity, split year into quarters.
            # Each quarter: ~365 steps
            q = steps_per_year // 4
            # Determine which months this year starts in from valid_time
            # Approximate: just use quarter splits
            # Q1 (first quarter), Q2, Q3, Q4
            # For DJF-like: use last quarter + first ~quarter
            storm_track_var_djf_sum += np.var(
                np.concatenate([ps_bp_yr[:q], ps_bp_yr[3*q:]]), axis=0
            )
            storm_track_n_djf += 1
            storm_track_var_jja_sum += np.var(ps_bp_yr[q:3*q], axis=0)
            storm_track_n_jja += 1

            del ps_chunk, ps_bp

        ps_ds.close()

        # ---- EKE: high-pass filtered U_7, V_7 ------------------------------
        print("    Loading U_7, V_7 for EKE …")
        u_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_U7_predictions.nc", chunks={"time": 1460}
        ).isel(sample=0)
        v_ds = xr.open_dataset(
            f"{seg_dir}/6h_surface_V7_predictions.nc", chunks={"time": 1460}
        ).isel(sample=0)

        for yr in range(n_full_years):
            t0 = yr * steps_per_year
            t1 = t0 + steps_per_year

            # Skip years before spinup cutoff
            vt_yr = vt[t0:t1]
            try:
                yr_val = vt_yr[0].astype("datetime64[Y]").astype(int) + 1970
                if yr_val < SPINUP_CUTOFF_YEAR:
                    continue
            except Exception:
                pass

            t0_pad = max(0, t0 - pad)
            t1_pad = min(n_times, t1 + pad)

            u_chunk = u_ds.U_7.isel(time=slice(t0_pad, t1_pad)).values
            v_chunk = v_ds.V_7.isel(time=slice(t0_pad, t1_pad)).values

            # High-pass filter: remove periods > 10 days (keep synoptic eddies)
            u_hp = highpass_filter(u_chunk, cutoff_days=10)
            v_hp = highpass_filter(v_chunk, cutoff_days=10)

            trim_start = t0 - t0_pad
            trim_end = trim_start + steps_per_year
            u_hp_yr = u_hp[trim_start:trim_end]
            v_hp_yr = v_hp[trim_start:trim_end]

            # EKE = 0.5 * mean(u'^2 + v'^2)
            eke_sum += 0.5 * np.mean(u_hp_yr**2 + v_hp_yr**2, axis=0)
            eke_n += 1

            del u_chunk, v_chunk, u_hp, v_hp

        # ---- Wind speed statistics (mean and std for Weibull) ----------------
        print("    Wind speed statistics …")
        for yr in range(0, n_full_years, 5):  # every 5th year
            t0 = yr * steps_per_year
            t1 = t0 + steps_per_year

            # Skip years before spinup cutoff
            vt_yr_check = vt[t0:t1]
            try:
                yr_val = vt_yr_check[0].astype("datetime64[Y]").astype(int) + 1970
                if yr_val < SPINUP_CUTOFF_YEAR:
                    continue
            except Exception:
                pass

            u_yr = u_ds.U_7.isel(time=slice(t0, t1, 4)).values  # daily subsample
            v_yr = v_ds.V_7.isel(time=slice(t0, t1, 4)).values
            wspd = np.sqrt(u_yr**2 + v_yr**2)
            wspd_sum += np.nansum(wspd, axis=0)
            wspd_sq_sum += np.nansum(wspd**2, axis=0)
            wspd_n += wspd.shape[0]

            # Regional histograms only (not per-gridpoint)
            for reg_name, ((lat0, lat1), (lon0, lon1)) in wind_regions.items():
                lat_mask = (lat >= lat0) & (lat <= lat1)
                lon_mask = (lon >= lon0) & (lon <= lon1)
                wspd_reg = wspd[:, lat_mask, :][:, :, lon_mask].ravel()
                h, _ = np.histogram(wspd_reg, bins=wind_speed_bins)
                wind_hist_regional[reg_name] += h

            del u_yr, v_yr, wspd

        u_ds.close()
        v_ds.close()

        # ---- Ocean fraction → sea ice extent --------------------------------
        # NOTE: ocean_fraction prediction files are empty (metadata only).
        # Sea ice analysis is skipped.
        print("    Ocean fraction — skipped (files contain no data)")

    # Save cache
    print(f"\nSaving cache to {CACHE_FILE} …")
    np.savez(
        CACHE_FILE,
        storm_track_var_sum=storm_track_var_sum,
        storm_track_var_djf_sum=storm_track_var_djf_sum,
        storm_track_var_jja_sum=storm_track_var_jja_sum,
        storm_track_n=storm_track_n,
        storm_track_n_djf=storm_track_n_djf,
        storm_track_n_jja=storm_track_n_jja,
        eke_sum=eke_sum,
        eke_n=eke_n,
        wspd_sum=wspd_sum,
        wspd_sq_sum=wspd_sq_sum,
        wspd_n=wspd_n,
        wind_hist_regional=wind_hist_regional,
    )

# Compute means
storm_track_var = storm_track_var_sum / max(storm_track_n, 1)
storm_track_var_djf = storm_track_var_djf_sum / max(storm_track_n_djf, 1)
storm_track_var_jja = storm_track_var_jja_sum / max(storm_track_n_jja, 1)
eke_mean = eke_sum / max(eke_n, 1)

# Wind speed mean and std from running sums
wspd_mean = wspd_sum / max(wspd_n, 1)
wspd_std = np.sqrt(np.maximum(wspd_sq_sum / max(wspd_n, 1) - wspd_mean**2, 0.0))

print(f"\n  Accumulated {storm_track_n} years of storm track data")
print(f"  Accumulated {eke_n} years of EKE data")
print(f"  Accumulated {wspd_n} wind speed samples")


# ===========================================================================
# Figures
# ===========================================================================
print("\nGenerating figures …\n")


# ---- 1. Storm track maps ------------------------------------------------
# INTERPRETATION: Variance of 2–6 day bandpass-filtered PS in Pa².
# Typical values: 50–200 Pa² in storm track cores.
# The NH storm tracks should show clear Atlantic and Pacific maxima.
# The SH storm track should be a band at ~50°S.
print("  1. Storm tracks …")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for ax, data, title in [
    (axes[0], storm_track_var / 1e4, "Annual Mean Storm Track (PS bandpass 2–6 day variance)"),
    (axes[1], storm_track_var_djf / 1e4, "DJF-like (boreal winter) Storm Track"),
    (axes[2], storm_track_var_jja / 1e4, "JJA-like (boreal summer) Storm Track"),
]:
    im = ax.pcolormesh(lon, lat, data, cmap="YlOrRd", shading="auto", vmin=0, vmax=2)
    ax.set_title(title + " (×10⁴ Pa²)")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="×10⁴ Pa²")

axes[2].set_xlabel("Longitude")
fig.tight_layout()
save_fig(fig, "storm_tracks")

# Zonal-mean storm tracks
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(lat, np.mean(storm_track_var / 1e4, axis=1), "k-", lw=2, label="Annual")
ax.plot(lat, np.mean(storm_track_var_djf / 1e4, axis=1), "r--", lw=1.5, label="DJF-like")
ax.plot(lat, np.mean(storm_track_var_jja / 1e4, axis=1), "b--", lw=1.5, label="JJA-like")
ax.set_xlabel("Latitude")
ax.set_ylabel("PS bandpass variance (×10⁴ Pa²)")
ax.set_title("Zonal-Mean Storm Track Activity")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, "storm_tracks_zonal")


# ---- 2. Eddy Kinetic Energy ---------------------------------------------
# INTERPRETATION: EKE in m²/s² at the near-surface level. Typical values
# 5–30 m²/s² in storm track regions. EKE should be collocated with but
# slightly downstream of the storm track PS variance maxima.
print("  2. EKE …")

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

im = axes[0].pcolormesh(lon, lat, eke_mean, cmap="inferno", shading="auto", vmin=0, vmax=40)
axes[0].set_title("Near-Surface Eddy Kinetic Energy (m²/s²)")
axes[0].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[0], label="m²/s²")

# Zonal mean
eke_zonal = np.mean(eke_mean, axis=1)
axes[1].plot(lat, eke_zonal, "k-", lw=2)
axes[1].set_xlabel("Latitude")
axes[1].set_ylabel("EKE (m²/s²)")
axes[1].set_title("Zonal-Mean Near-Surface EKE")
axes[1].grid(alpha=0.3)

fig.tight_layout()
save_fig(fig, "eke")

# Storm track vs EKE comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1 = axes[0]
ax1.plot(lat, np.mean(storm_track_var / 1e4, axis=1), "r-", lw=2, label="Storm track (PS var)")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("PS variance (×10⁴ Pa²)", color="r")
ax1.tick_params(axis="y", labelcolor="r")
ax2 = ax1.twinx()
ax2.plot(lat, eke_zonal, "b-", lw=2, label="EKE")
ax2.set_ylabel("EKE (m²/s²)", color="b")
ax2.tick_params(axis="y", labelcolor="b")
ax1.set_title("Storm Track vs EKE (zonal mean)")

# Scatter: per-gridpoint storm track var vs EKE
axes[1].scatter(storm_track_var.ravel() / 1e4, eke_mean.ravel(),
                s=1, alpha=0.1, c="k")
axes[1].set_xlabel("PS bandpass variance (×10⁴ Pa²)")
axes[1].set_ylabel("EKE (m²/s²)")
axes[1].set_title("Storm Track vs EKE (per grid point)")
axes[1].set_xlim(0, 3)
axes[1].set_ylim(0, 60)

fig.tight_layout()
save_fig(fig, "storm_vs_eke")


# ---- 3. Sea ice ----------------------------------------------------------
# NOTE: ocean_fraction prediction files are empty (no data variables).
# Skipping sea ice analysis — bump figure counter to keep numbering consistent.
print("  3. Sea ice — skipped (ocean_fraction files contain no data)")
for name in ["sea_ice_climatology", "sea_ice_extent_timeseries", "sea_ice_seasonal_cycle"]:
    _fig_counter += 1
    print(f"    → skipped {name} (no data)")


# ---- 4. Wind speed distribution -----------------------------------------
# INTERPRETATION: Fit Weibull shape (k) and scale (c) parameters from
# mean and std of wind speed at each grid point (method of moments).
print("  4. Wind speed …")

bin_centers = 0.5 * (wind_speed_bins[:-1] + wind_speed_bins[1:])

# Weibull shape parameter estimate (method of moments)
# k ≈ (σ/μ)^(-1.086), c ≈ μ / Γ(1 + 1/k)
with np.errstate(divide="ignore", invalid="ignore"):
    cv = wspd_std / np.where(wspd_mean > 0.1, wspd_mean, np.nan)
    k_weibull = np.where(cv > 0.01, cv**(-1.086), np.nan)

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Mean wind speed map
im = axes[0].pcolormesh(lon, lat, wspd_mean, cmap="YlGnBu", shading="auto", vmin=0, vmax=15)
axes[0].set_title("Mean Near-Surface Wind Speed (m/s)")
axes[0].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[0], label="m/s")

# Weibull shape parameter map
im = axes[1].pcolormesh(lon, lat, k_weibull, cmap="RdYlBu_r", shading="auto", vmin=1, vmax=5)
axes[1].set_title("Weibull Shape Parameter k (k≈2: Rayleigh, k>3: steady, k<1.5: gusty)")
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[1], label="k")

fig.tight_layout()
save_fig(fig, "wind_speed_maps")

# Wind speed distributions for selected regions (from regional histograms)
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

region_labels = {
    "Tropical Pacific": "Tropical Pacific\n(10S-10N, 150-220E)",
    "N. Atlantic Storm Track": "N. Atlantic Storm Track\n(40-60N, 300-350E)",
    "Southern Ocean": "Southern Ocean\n(50-65S, 0-360E)",
    "Sahara Desert": "Sahara Desert\n(15-30N, 0-40E)",
    "Trade Wind Belt": "Trade Wind Belt\n(10-25N, 300-350E)",
    "Southern Plains": "Southern Plains\n(30-40N, 250-270E)",
}

for ax, reg_name in zip(axes.flat, wind_regions):
    region_hist = wind_hist_regional[reg_name]
    total = region_hist.sum()
    if total > 0:
        region_hist_norm = region_hist / total
    else:
        region_hist_norm = region_hist
    ax.bar(bin_centers, region_hist_norm, width=0.25, color="steelblue", alpha=0.7)
    ax.set_title(region_labels.get(reg_name, reg_name), fontsize=10)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 30)

fig.suptitle("Near-Surface Wind Speed Distributions by Region", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "wind_speed_distributions")


# ---- 5. Summary ---------------------------------------------------------
print("  5. Summary …")
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")

summary = f"""
ACE2-EAMv3 piControl — 6-Hourly Dynamics Summary
{'='*60}

  Segments processed:   {n_segs}
  Years of data:        {storm_track_n} (storm tracks), {eke_n} (EKE)

  STORM TRACKS
  Max PS bandpass variance:  {np.max(storm_track_var)/1e4:.2f} ×10⁴ Pa²
  Lat of NH max:             {lat[np.argmax(np.mean(storm_track_var[:nlat//2+90], axis=1))]:.1f}°N
  Lat of SH max:             {lat[nlat//2+np.argmax(np.mean(storm_track_var[nlat//2:], axis=1))]:.1f}°

  EDDY KINETIC ENERGY
  Max EKE:                   {np.max(eke_mean):.1f} m²/s²
  Global mean EKE:           {np.average(eke_mean, weights=w[:,None]*np.ones((1,nlon))):.2f} m²/s²

  SEA ICE
  Skipped (ocean_fraction files contain no data)

  WIND SPEED
  Global mean wind speed:    {np.average(wspd_mean, weights=w[:,None]*np.ones((1,nlon))):.2f} m/s
  Max mean wind speed:       {np.max(wspd_mean):.1f} m/s
"""

ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
        verticalalignment="top", fontfamily="monospace")
fig.tight_layout()
save_fig(fig, "summary")


# ---- Done ----------------------------------------------------------------
print(f"\nDone — saved {_fig_counter} PNGs to {OUTPUT_DIR}/")
