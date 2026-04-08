"""
Low-frequency variability analysis for the ~800-year ACE2-EAMv3 piControl run.

Saves individual PNGs (300 dpi) to figs_low_freq/ with:
  1. Global-mean TS with low-pass filters (20, 50, 100 yr)
  2. Power spectrum of global-mean TS
  3. AMO-like index (N. Atlantic SST minus global mean)
  4. PDO-like index via EOF of N. Pacific SST
  5. Wavelet scalogram of global-mean TS
  6. Running standard deviation (stationarity check)
  7. Tropical vs extratropical TS variability
  8. ENSO index from pre-computed diagnostics
  9. Zonal-mean TS Hovmöller diagram
 10. Lag-1 autocorrelation map (memory / persistence)
 11. Energy budget low-frequency components
 12. Precipitation low-frequency variability

Usage:
    micromamba run -n xgns python low_freq_variability.py
"""

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from matplotlib.colors import LogNorm
from scipy.signal import butter, filtfilt, welch

from config import (get_monthly_prediction_files, get_enso_diagnostic_files,
                    SOLIN_PATH, SPINUP_CUTOFF_YEAR)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILES = get_monthly_prediction_files()
ENSO_FILES = get_enso_diagnostic_files()

OUTPUT_DIR = Path("figs_low_freq")
OUTPUT_DIR.mkdir(exist_ok=True)
_fig_counter = 0
OVERWRITE = "--overwrite" in sys.argv


def save_fig(fig, name):
    """Save figure as numbered PNG at 300 dpi. Skip if file exists unless --overwrite."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def lowpass(data, cutoff_years, fs=1.0, order=3):
    """Zero-phase Butterworth low-pass filter. cutoff_years = minimum period kept."""
    nyq = 0.5 * fs
    b, a = butter(order, (1.0 / cutoff_years) / nyq)
    return filtfilt(b, a, data)


def detrend(x):
    """Remove linear trend."""
    n = len(x)
    t = np.arange(n, dtype=float)
    coeffs = np.polyfit(t, x, 1)
    return x - np.polyval(coeffs, t)


def cosine_weights(lat):
    """Area weights proportional to cos(lat)."""
    return np.cos(np.deg2rad(lat))


def global_yearly_mean(da, lat):
    """Weighted global mean → yearly mean, returned as computed array."""
    w = cosine_weights(lat)
    return (
        da.weighted(w)
        .mean(("lon", "lat"))
        .groupby("time.year")
        .mean("time")
        .compute()
    )


def filter_full_years(ds):
    """Keep only years with 12 months of data."""
    counts = ds.time.groupby("time.year").count()
    full = counts.year.where(counts == 12, drop=True).values
    return ds.sel(time=ds.time.dt.year.isin(full))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading dataset …")
mds = xr.open_mfdataset(
    DATA_FILES,
    combine="nested",
    concat_dim="time",
    chunks={},
    drop_variables=["init_time"],
).isel(sample=0)

# Replace time coord with valid_time (actual forecast time)
mds["time"] = mds.valid_time
mds = mds.drop_vars("valid_time")

# Drop incomplete years at edges
mds = filter_full_years(mds)

# Keep the full dataset for drift/stability plots
mds_full = mds

# Create post-spinup dataset for statistical analyses
mds = mds.sel(time=mds.time.dt.year >= SPINUP_CUTOFF_YEAR)

lat = mds.lat
lon = mds.lon

# ---------------------------------------------------------------------------
# Pre-compute yearly global means for key variables (do once, reuse)
# ---------------------------------------------------------------------------
print("Computing yearly means …")

# Full timeseries (for drift plots)
ts_yearly_full = global_yearly_mean(mds_full.TS, lat)
years_full = ts_yearly_full.year.values
n_years_full = len(years_full)
ts_vals_full = ts_yearly_full.values

# Post-spinup (for statistical analyses)
ts_yearly = global_yearly_mean(mds.TS, lat)
years = ts_yearly.year.values
n_years = len(years)

ts_vals = ts_yearly.values
ts_detrended = detrend(ts_vals)


# ===========================================================================
# Figures
# ===========================================================================
print("Generating figures …")


# ---- 1. Global-mean TS with low-pass filters (full timeseries) ----------
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(years_full, ts_vals_full, alpha=0.25, color="gray", label="Annual mean")
for cutoff, color in [(20, "C0"), (50, "C1"), (100, "C3")]:
    if n_years_full > 2 * cutoff:
        ax.plot(years_full, lowpass(ts_vals_full, cutoff), label=f"{cutoff}-yr low-pass", lw=2, color=color)
ax.axvline(SPINUP_CUTOFF_YEAR, color="k", ls="--", lw=1.0, label=f"Spinup cutoff (year {SPINUP_CUTOFF_YEAR})")
ax.set_xlabel("Year")
ax.set_ylabel("TS (K)")
ax.set_title("Global Mean Surface Temperature — Low-Frequency Variability (full run)")
ax.legend()
fig.tight_layout()
save_fig(fig, "ts_lowpass_filters")


# ---- 2. Power spectrum --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
freq, psd = welch(ts_detrended, fs=1.0, nperseg=min(256, n_years // 2))
period = 1.0 / freq[1:]
ax.loglog(period, psd[1:], "k-", lw=1.2)

# Reference red-noise spectrum (AR1 fit)
lag1 = np.corrcoef(ts_detrended[:-1], ts_detrended[1:])[0, 1]
var = np.var(ts_detrended)
red_noise = var * (1 - lag1**2) / (1 - 2 * lag1 * np.cos(2 * np.pi * freq[1:]) + lag1**2)
ax.loglog(period, red_noise, "r--", alpha=0.7, label=f"AR(1) red noise (r₁={lag1:.2f})")

ax.axvspan(2, 7, alpha=0.08, color="red", label="ENSO band (2–7 yr)")
ax.axvspan(20, 70, alpha=0.08, color="blue", label="Multidecadal (20–70 yr)")
ax.axvspan(70, 300, alpha=0.08, color="green", label="Centennial (70–300 yr)")
ax.invert_xaxis()
ax.set_xlabel("Period (years)")
ax.set_ylabel("PSD (K² yr)")
ax.set_title(f"Power Spectrum of Global Mean TS (Welch, post-spinup, N={n_years} years)")
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, "ts_power_spectrum")


# ---- 3. AMO-like index --------------------------------------------------
# North Atlantic: 0–60°N, 280–360°E
print("  AMO index …")
na_ts = mds.TS.sel(lat=slice(0, 60), lon=slice(280, 360))
amo_raw = global_yearly_mean(na_ts, na_ts.lat)
amo_index = amo_raw.values - ts_vals  # subtract global mean

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(years, amo_index, alpha=0.25, color="gray", label="Annual")
if n_years > 40:
    ax.plot(years, lowpass(amo_index, 20), color="red", lw=2, label="20-yr low-pass")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("AMO index (K)")
ax.set_title("AMO-like Index (N. Atlantic 0\u201360\u00b0N, 280\u2013360\u00b0E minus global mean, post-spinup)")
ax.legend()
fig.tight_layout()
save_fig(fig, "amo_index")

# AMO power spectrum
fig, ax = plt.subplots(figsize=(10, 5))
amo_det = detrend(amo_index)
freq_a, psd_a = welch(amo_det, fs=1.0, nperseg=min(256, n_years // 2))
ax.loglog(1.0 / freq_a[1:], psd_a[1:], "k-", lw=1.2)
ax.invert_xaxis()
ax.set_xlabel("Period (years)")
ax.set_ylabel("PSD (K² yr)")
ax.set_title(f"Power Spectrum of AMO Index (post-spinup, N={n_years} years)")
ax.axvspan(50, 90, alpha=0.1, color="orange", label="Observed AMO band (50–90 yr)")
ax.legend()
fig.tight_layout()
save_fig(fig, "amo_spectrum")


# ---- 4. PDO-like index via EOF ------------------------------------------
print("  PDO EOF …")
np_ts = mds.TS.sel(lat=slice(20, 60), lon=slice(120, 260))
np_yearly = (
    np_ts.weighted(cosine_weights(np_ts.lat))
    .mean(dim=[])  # keep spatial dims, just weight
    .groupby("time.year")
    .mean("time")
    .compute()
)
np_clim = np_yearly.mean("year")
np_anom = (np_yearly - np_clim) * cosine_weights(np_yearly.lat)

# Reshape to (time, space), do SVD
vals_2d = np_anom.values.reshape(len(np_anom.year), -1)
vals_2d = np.nan_to_num(vals_2d)
U, S, Vt = np.linalg.svd(vals_2d, full_matrices=False)
pcs = U * S  # principal components
explained = (S**2) / (S**2).sum()

fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 1]})

# PC1 timeseries
axes[0].plot(np_anom.year.values, pcs[:, 0], alpha=0.3, color="gray")
if n_years > 40:
    axes[0].plot(np_anom.year.values, lowpass(pcs[:, 0], 20), lw=2, color="C0")
axes[0].set_title(f"PC1 (PDO-like, post-spinup) — {explained[0]:.1%} variance")
axes[0].set_xlabel("Year")

# EOF1 spatial pattern
eof1 = Vt[0].reshape(np_anom.shape[1:])
im = axes[1].pcolormesh(np_anom.lon, np_anom.lat, eof1, cmap="RdBu_r", shading="auto")
axes[1].set_title("EOF1 Spatial Pattern (N. Pacific)")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[1], shrink=0.8)
fig.tight_layout()
save_fig(fig, "pdo_eof")


# ---- 5. Wavelet scalogram (full timeseries) -----------------------------
print("  Wavelet analysis …")
try:
    import pywt

    ts_detrended_full = detrend(ts_vals_full)
    scales = np.arange(2, min(n_years_full // 2, 300))
    coeffs, freqs = pywt.cwt(ts_detrended_full, scales, "morl", sampling_period=1.0)
    periods_wvt = 1.0 / freqs

    power_wvt = np.abs(coeffs) ** 2
    # Clip tiny values to avoid LogNorm issues
    power_wvt = np.clip(power_wvt, power_wvt[power_wvt > 0].min(), None)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.pcolormesh(
        years_full, periods_wvt, power_wvt, cmap="inferno", shading="auto",
        norm=LogNorm(),
    )
    ax.axvline(SPINUP_CUTOFF_YEAR, color="white", ls="--", lw=1.0, label=f"Spinup cutoff (year {SPINUP_CUTOFF_YEAR})")
    ax.set_yscale("log")
    ax.set_ylabel("Period (years)")
    ax.set_xlabel("Year")
    ax.set_title("Wavelet Power Spectrum of Global Mean TS (log color scale, full run)")
    ax.set_ylim(2, min(n_years_full // 2, 300))
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="Power")
    fig.tight_layout()
    save_fig(fig, "wavelet_scalogram")
except ImportError:
    print("  ⚠ pywt not installed, skipping wavelet analysis")


# ---- 6. Running standard deviation (stationarity check, full run) -------
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for window, color in [(30, "C0"), (50, "C1"), (100, "C3")]:
    if n_years_full > 2 * window:
        rolling_std = (
            ts_yearly_full.rolling(year=window, center=True).std().compute()
        )
        axes[0].plot(years_full, rolling_std, label=f"{window}-yr window", color=color, lw=1.5)

axes[0].axvline(SPINUP_CUTOFF_YEAR, color="k", ls="--", lw=1.0, label=f"Spinup cutoff (year {SPINUP_CUTOFF_YEAR})")
axes[0].set_ylabel("Running \u03c3 (K)")
axes[0].set_title("Running Standard Deviation of Global Mean TS (full run)")
axes[0].legend()

# Running mean as well — drift check
for window, color in [(50, "C0"), (100, "C3")]:
    if n_years_full > 2 * window:
        rolling_mean = ts_yearly_full.rolling(year=window, center=True).mean().compute()
        axes[1].plot(years_full, rolling_mean, label=f"{window}-yr running mean", color=color, lw=1.5)

axes[1].axhline(ts_vals_full.mean(), color="k", ls="--", lw=0.5, label="Overall mean")
axes[1].axvline(SPINUP_CUTOFF_YEAR, color="k", ls="--", lw=1.0)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("TS (K)")
axes[1].set_title("Running Mean of Global Mean TS (Drift Check, full run)")
axes[1].legend()
fig.tight_layout()
save_fig(fig, "running_std_and_mean")


# ---- 7. Tropical vs extratropical variability ---------------------------
print("  Tropical vs extratropical …")
ts_trop = global_yearly_mean(mds.TS.sel(lat=slice(-30, 30)), mds.TS.sel(lat=slice(-30, 30)).lat)
ts_nh = global_yearly_mean(mds.TS.sel(lat=slice(30, 90)), mds.TS.sel(lat=slice(30, 90)).lat)
ts_sh = global_yearly_mean(mds.TS.sel(lat=slice(-90, -30)), mds.TS.sel(lat=slice(-90, -30)).lat)

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, data, title in [
    (axes[0], ts_trop, "Tropics (30°S–30°N)"),
    (axes[1], ts_nh, "Northern Extratropics (30°N–90°N)"),
    (axes[2], ts_sh, "Southern Extratropics (90°S–30°S)"),
]:
    vals = data.values
    ax.plot(years, vals, alpha=0.25, color="gray")
    if n_years > 40:
        ax.plot(years, lowpass(vals, 20), lw=2, color="C0")
    ax.set_title(f"{title}  (σ = {np.std(detrend(vals)):.3f} K)")
    ax.set_ylabel("TS (K)")

axes[2].set_xlabel("Year")
fig.suptitle("Regional Surface Temperature — Low-Frequency Variability (post-spinup)", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "tropical_vs_extratropical")

# Cross-spectra: tropical vs NH — coherence
fig, ax = plt.subplots(figsize=(10, 5))
for data, label, color in [
    (detrend(ts_trop.values), "Tropics", "C0"),
    (detrend(ts_nh.values), "NH Extratropics", "C1"),
    (detrend(ts_sh.values), "SH Extratropics", "C2"),
]:
    f, p = welch(data, fs=1.0, nperseg=min(256, n_years // 2))
    ax.loglog(1.0 / f[1:], p[1:], label=label, color=color, lw=1.5)

ax.invert_xaxis()
ax.set_xlabel("Period (years)")
ax.set_ylabel("PSD (K² yr)")
ax.set_title(f"Power Spectra — Tropics vs Extratropics (post-spinup, N={n_years} years)")
ax.legend()
fig.tight_layout()
save_fig(fig, "regional_spectra")


# ---- 8. ENSO index from pre-computed diagnostics -------------------------
print("  ENSO diagnostics …")
if ENSO_FILES:
    enso_ds = xr.open_mfdataset(
        ENSO_FILES, combine="nested", concat_dim="time", use_cftime=True
    ).isel(sample=0)
    enso_ts = enso_ds.TS.compute()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Timeseries
    axes[0].plot(np.arange(len(enso_ts)), enso_ts.values, alpha=0.4, lw=0.5)
    axes[0].set_xlabel("Month index")
    axes[0].set_ylabel("Niño SST (K)")
    axes[0].set_title("ENSO Index (from enso_index_diagnostics)")

    # Power spectrum (monthly data → fs=12 samples/yr)
    enso_det = detrend(enso_ts.values)
    f_e, p_e = welch(enso_det, fs=12.0, nperseg=min(512, len(enso_det) // 2))
    period_e = 1.0 / f_e[1:]  # in years
    axes[1].loglog(period_e, p_e[1:], "k-", lw=1.2)
    axes[1].axvspan(2, 7, alpha=0.1, color="red", label="ENSO band (2–7 yr)")
    axes[1].invert_xaxis()
    axes[1].set_xlim(20, 0.5)
    axes[1].set_xlabel("Period (years)")
    axes[1].set_ylabel("PSD (K² yr)")
    axes[1].set_title("ENSO Power Spectrum (monthly resolution, period 0.5\u201320 yr)")
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, "enso_diagnostics")
else:
    print("  ⚠ No ENSO diagnostic files found, skipping")


# ---- 9. Zonal-mean TS Hovmöller -----------------------------------------
print("  Hovmöller diagram …")
ts_zonal_yearly = (
    mds.TS.mean("lon")
    .groupby("time.year")
    .mean("time")
    .compute()
)
ts_zonal_clim = ts_zonal_yearly.mean("year")
ts_zonal_anom = ts_zonal_yearly - ts_zonal_clim

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.pcolormesh(
    ts_zonal_anom.year, ts_zonal_anom.lat, ts_zonal_anom.T,
    cmap="RdBu_r", shading="auto", vmin=-2, vmax=2,
)
ax.set_xlabel("Year")
ax.set_ylabel("Latitude")
ax.set_title("Zonal-Mean TS Anomaly Hovm\u00f6ller (K, post-spinup)")
plt.colorbar(im, ax=ax, label="K")
fig.tight_layout()
save_fig(fig, "hovmoller_ts")


# ---- 10. Lag-1 autocorrelation map (persistence / memory, post-spinup) ---
print("  Autocorrelation map …")
ts_yearly_map_full = mds_full.TS.groupby("time.year").mean("time").compute()
ts_yearly_map = mds.TS.groupby("time.year").mean("time").compute()
ts_map_anom = ts_yearly_map - ts_yearly_map.mean("year")

# Compute lag-1 correlation at each grid point
a = ts_map_anom.isel(year=slice(None, -1)).values  # (n-1, lat, lon)
b = ts_map_anom.isel(year=slice(1, None)).values
with np.errstate(divide="ignore", invalid="ignore"):
    r1 = (
        np.nanmean(a * b, axis=0)
        / (np.nanstd(a, axis=0) * np.nanstd(b, axis=0))
    )

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.pcolormesh(
    lon, lat, r1, cmap="RdBu_r", shading="auto", vmin=-0.5, vmax=0.8
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Lag-1 Year Autocorrelation of TS (Persistence Map, post-spinup)")
plt.colorbar(im, ax=ax, label="Correlation")
fig.tight_layout()
save_fig(fig, "lag1_autocorrelation_map")


# ---- 11. Energy budget low-frequency components -------------------------
print("  Energy budget …")
restom = -mds.top_of_atmos_upward_shortwave_flux - mds.FLUT
net_toa = mds.net_energy_flux_toa_into_atmosphere
net_sfc = mds.net_energy_flux_sfc_into_atmosphere
net_col = mds.net_energy_flux_into_atmospheric_column

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Panel 1: yearly timeseries
for da, label, color in [
    (restom, "RESTOM (−SW↑ − FLUT)", "C0"),
    (net_toa, "Net TOA into atm", "C1"),
    (net_sfc, "Net SFC into atm", "C2"),
    (net_col, "Net into column", "C3"),
]:
    yearly = global_yearly_mean(da, lat)
    axes[0].plot(yearly.year, yearly.values, alpha=0.5, color=color, label=label)
    if n_years > 40:
        axes[1].plot(
            yearly.year,
            lowpass(yearly.values, 20),
            color=color, lw=2, label=label,
        )

axes[0].set_ylabel("W/m²")
axes[0].set_title("Energy Budget Components \u2014 Annual Mean (post-spinup)")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("W/m²")
axes[1].set_title("Energy Budget Components \u2014 20-yr Low-Pass (post-spinup)")
axes[1].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "energy_budget_annual")

# Water & energy budget residuals
print("  Budget residuals …")
water_resid = global_yearly_mean(mds.total_water_path_budget_residual, lat)

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

axes[0].plot(water_resid.year, water_resid.values, alpha=0.5)
if n_years > 40:
    axes[0].plot(water_resid.year, lowpass(water_resid.values, 20), lw=2, color="C1")
axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_title("Total Water Path Budget Residual (should be ≈ 0)")
axes[0].set_ylabel("kg/m²/s")

energy_tend = global_yearly_mean(mds.total_energy_ace2_path_tendency, lat)
axes[1].plot(energy_tend.year, energy_tend.values, alpha=0.5)
if n_years > 40:
    axes[1].plot(energy_tend.year, lowpass(energy_tend.values, 20), lw=2, color="C1")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_title("Total Energy Path Tendency (drift indicator)")
axes[1].set_ylabel("W/m²")
axes[1].set_xlabel("Year")

fig.tight_layout()
save_fig(fig, "budget_residuals")


# ---- 12. Precipitation low-frequency variability -------------------------
print("  Precipitation …")
precip_yearly = global_yearly_mean(mds.surface_precipitation_rate, lat)

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

# Timeseries
axes[0].plot(precip_yearly.year, precip_yearly.values * 86400, alpha=0.3, color="gray")
if n_years > 40:
    axes[0].plot(
        precip_yearly.year,
        lowpass(precip_yearly.values * 86400, 20),
        lw=2, color="C0", label="20-yr low-pass",
    )
axes[0].set_ylabel("Precip (mm/day)")
axes[0].set_title("Global Mean Precipitation Rate (post-spinup)")
axes[0].legend()

# Tropical precip — where most action is
precip_trop = global_yearly_mean(
    mds.surface_precipitation_rate.sel(lat=slice(-30, 30)),
    mds.surface_precipitation_rate.sel(lat=slice(-30, 30)).lat,
)
axes[1].plot(precip_trop.year, precip_trop.values * 86400, alpha=0.3, color="gray")
if n_years > 40:
    axes[1].plot(
        precip_trop.year,
        lowpass(precip_trop.values * 86400, 20),
        lw=2, color="C1", label="20-yr low-pass",
    )
axes[1].set_ylabel("Precip (mm/day)")
axes[1].set_xlabel("Year")
axes[1].set_title("Tropical (30\u00b0S\u201330\u00b0N) Mean Precipitation Rate (post-spinup)")
axes[1].legend()

fig.tight_layout()
save_fig(fig, "precip_variability")


# ---- 13. Interhemispheric temperature gradient ---------------------------
print("  Interhemispheric gradient …")
ts_nh_full = global_yearly_mean(mds.TS.sel(lat=slice(0, 90)), mds.TS.sel(lat=slice(0, 90)).lat)
ts_sh_full = global_yearly_mean(mds.TS.sel(lat=slice(-90, 0)), mds.TS.sel(lat=slice(-90, 0)).lat)
ih_gradient = ts_nh_full.values - ts_sh_full.values

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(years, ih_gradient, alpha=0.3, color="gray", label="Annual")
if n_years > 40:
    ax.plot(years, lowpass(ih_gradient, 20), lw=2, color="C0", label="20-yr low-pass")
ax.axhline(ih_gradient.mean(), color="k", ls="--", lw=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("ΔT (K)")
ax.set_title("Interhemispheric Temperature Gradient (NH \u2212 SH, post-spinup)")
ax.legend()
fig.tight_layout()
save_fig(fig, "interhemispheric_gradient")


# ---- 14. Wind stress variability (trade wind / jet stream proxy) ---------
print("  Wind stress …")
taux_yearly = global_yearly_mean(mds.TAUX, lat)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(taux_yearly.year, taux_yearly.values, alpha=0.3, color="gray")
if n_years > 40:
    ax.plot(taux_yearly.year, lowpass(taux_yearly.values, 20), lw=2, color="C0", label="20-yr low-pass")
ax.set_xlabel("Year")
ax.set_ylabel("TAUX (Pa)")
ax.set_title("Global Mean Zonal Wind Stress \u2014 Low-Frequency Variability (post-spinup)")
ax.legend()
fig.tight_layout()
save_fig(fig, "wind_stress")


# ---- 15. Cumulative drift — integrated global-mean TS anomaly (full) ----
# A random-walk-like cumulative sum reveals whether the model drifts
# systematically or stays centered. In a well-equilibrated piControl the
# cumulative anomaly should wander around zero without a secular trend.
print("  Cumulative drift …")
ts_anom_full = ts_vals_full - ts_vals_full.mean()
cumulative = np.cumsum(ts_anom_full)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(years_full, cumulative, lw=1.5)
ax.axhline(0, color="k", lw=0.5)
ax.axvline(SPINUP_CUTOFF_YEAR, color="k", ls="--", lw=1.0, label=f"Spinup cutoff (year {SPINUP_CUTOFF_YEAR})")
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative TS anomaly (K\u00b7yr)")
ax.set_title("Cumulative Global Mean TS Anomaly (Drift Diagnostic, full run)")
ax.legend()
fig.tight_layout()
save_fig(fig, "cumulative_drift")


# ---- 16. Decadal-scale maps: difference between first & last centuries (full) ---
# Spatial pattern of any long-term drift — reveals where the model warms
# or cools over the course of the run.
print("  Century-scale spatial drift …")
n_century = min(100, n_years_full // 4)
ts_first = ts_yearly_map_full.isel(year=slice(0, n_century)).mean("year")
ts_last = ts_yearly_map_full.isel(year=slice(-n_century, None)).mean("year")
drift_map = ts_last - ts_first

fig, ax = plt.subplots(figsize=(12, 5))
vmax = max(abs(float(drift_map.min())), abs(float(drift_map.max())), 1.0)
im = ax.pcolormesh(
    lon, lat, drift_map, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(
    f"TS Drift: Last {n_century} yr Mean \u2212 First {n_century} yr Mean (K, full run)"
)
plt.colorbar(im, ax=ax, label="K")
fig.tight_layout()
save_fig(fig, "century_spatial_drift")


# ---- 17. Multi-lag autocorrelation function (decorrelation timescale) ----
# Tells you how long the model "remembers" — the e-folding time of the
# global-mean TS ACF. Real climate has ~4-5 year decorrelation for SST.
print("  Autocorrelation function …")
max_lag = min(100, n_years // 3)
acf = np.array(
    [np.corrcoef(ts_detrended[: n_years - lag], ts_detrended[lag:])[0, 1] for lag in range(max_lag)]
)
# Find e-folding time
efold_idx = np.where(acf < 1.0 / np.e)[0]
efold_time = efold_idx[0] if len(efold_idx) > 0 else max_lag

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(max_lag), acf, width=1.0, color="steelblue", alpha=0.7)
ax.axhline(1.0 / np.e, color="red", ls="--", label=f"e-folding = {efold_time} yr")
ax.axhline(0, color="k", lw=0.5)
# 95% significance for white noise
ax.axhline(1.96 / np.sqrt(n_years), color="gray", ls=":", label="95% significance (white noise)")
ax.axhline(-1.96 / np.sqrt(n_years), color="gray", ls=":")
ax.set_xlabel("Lag (years)")
ax.set_ylabel("Autocorrelation")
ax.set_title(f"Autocorrelation Function of Global Mean TS (post-spinup, e-folding = {efold_time} yr)")
ax.legend()
ax.set_xlim(0, max_lag)
fig.tight_layout()
save_fig(fig, "autocorrelation_function")


# ---- 18. Hurst exponent / DFA (long-range dependence, post-spinup) ------
# H > 0.5 indicates long-range persistence (memory beyond AR1).
# H = 0.5 -> random walk, H < 0.5 -> anti-persistent. Real climate
# typically shows H ~ 0.6-0.8 for surface temperature.
print("  Hurst exponent (R/S + DFA) …")


def estimate_hurst(x):
    """Rescaled range (R/S) estimate of the Hurst exponent."""
    n = len(x)
    sizes = []
    rs_values = []
    for div in range(8, n // 4):
        n_chunks = n // div
        if n_chunks < 2:
            break
        rs_chunk = []
        for i in range(n_chunks):
            chunk = x[i * div : (i + 1) * div]
            mean_c = chunk.mean()
            dev = np.cumsum(chunk - mean_c)
            r = dev.max() - dev.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            sizes.append(div)
            rs_values.append(np.mean(rs_chunk))
    sizes = np.array(sizes, dtype=float)
    rs_values = np.array(rs_values, dtype=float)
    coeffs = np.polyfit(np.log(sizes), np.log(rs_values), 1)
    return coeffs[0], sizes, rs_values


def estimate_dfa(x, min_box=8, max_box=None):
    """Detrended Fluctuation Analysis -- more robust than R/S to trends."""
    n = len(x)
    if max_box is None:
        max_box = n // 4
    y = np.cumsum(x - np.mean(x))  # integrated series
    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 30).astype(int))
    fluct = []
    for bs in box_sizes:
        n_boxes = n // bs
        if n_boxes < 2:
            continue
        rms_list = []
        for i in range(n_boxes):
            segment = y[i * bs : (i + 1) * bs]
            t = np.arange(bs)
            p = np.polyfit(t, segment, 1)
            rms_list.append(np.sqrt(np.mean((segment - np.polyval(p, t)) ** 2)))
        fluct.append(np.mean(rms_list))
    box_sizes = box_sizes[: len(fluct)]
    fluct = np.array(fluct)
    H = np.polyfit(np.log(box_sizes), np.log(fluct), 1)[0]
    return H, box_sizes, fluct


H_rs, sizes_rs, rs_vals = estimate_hurst(ts_detrended)
H_dfa, sizes_dfa, fluct_dfa = estimate_dfa(ts_detrended)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: R/S analysis
axes[0].loglog(sizes_rs, rs_vals, "ko", ms=3, alpha=0.5)
axes[0].loglog(
    sizes_rs,
    np.exp(np.polyval(np.polyfit(np.log(sizes_rs), np.log(rs_vals), 1), np.log(sizes_rs))),
    "r-", lw=2, label=f"R/S: H = {H_rs:.2f}",
)
axes[0].loglog(sizes_rs, sizes_rs ** 0.5, "b--", alpha=0.5, label="H = 0.5 (random walk)")
axes[0].set_xlabel("Window size (years)")
axes[0].set_ylabel("R/S statistic")
axes[0].set_title(f"R/S Hurst (H = {H_rs:.2f}, post-spinup)")
axes[0].legend()

# Right: DFA
axes[1].loglog(sizes_dfa, fluct_dfa, "ko", ms=3, alpha=0.5)
axes[1].loglog(
    sizes_dfa,
    np.exp(np.polyval(np.polyfit(np.log(sizes_dfa), np.log(fluct_dfa), 1), np.log(sizes_dfa))),
    "r-", lw=2, label=f"DFA: H = {H_dfa:.2f}",
)
axes[1].loglog(sizes_dfa, sizes_dfa ** 0.5 * fluct_dfa[0] / sizes_dfa[0] ** 0.5,
               "b--", alpha=0.5, label="H = 0.5 (random walk)")
axes[1].set_xlabel("Box size (years)")
axes[1].set_ylabel("DFA fluctuation F(n)")
axes[1].set_title(f"DFA Hurst (H = {H_dfa:.2f}, post-spinup)")
axes[1].legend()

fig.suptitle("Long-Range Dependence: R/S vs DFA — Global Mean TS", fontsize=13, y=1.02)
fig.tight_layout()
save_fig(fig, "hurst_dfa_exponent")


# ---- 19. Epoch analysis — std dev maps for early/mid/late thirds (full) --
# Checks whether spatial variability patterns change over the run.
# Non-stationarity in spatial patterns is a strong indicator of model drift.
print("  Epoch variability maps …")
n_third = n_years_full // 3
epochs = [
    ("Early", slice(0, n_third)),
    ("Middle", slice(n_third, 2 * n_third)),
    ("Late", slice(2 * n_third, 3 * n_third)),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
for ax, (label, sl) in zip(axes, epochs):
    std_map = ts_yearly_map_full.isel(year=sl).std("year")
    im = ax.pcolormesh(lon, lat, std_map, cmap="magma", shading="auto", vmin=0, vmax=3)
    ax.set_title(f"{label} Third \u2014 TS \u03c3 (K)")
    ax.set_xlabel("Longitude")

axes[0].set_ylabel("Latitude")
plt.colorbar(im, ax=axes, shrink=0.8, label="K")
fig.suptitle("Interannual TS Variability by Epoch (full run)", fontsize=13, y=1.02)
fig.tight_layout()
save_fig(fig, "epoch_variability_maps")


# ---- 20. Surface heat flux low-frequency coupling -----------------------
# Coherence between LHFLX+SHFLX and TS on decadal timescales reveals
# how the atmosphere-surface energy exchange drives long-term variability.
print("  Surface flux coupling …")
lhflx_yearly = global_yearly_mean(mds.LHFLX, lat)
shflx_yearly = global_yearly_mean(mds.SHFLX, lat)
total_sfc_flux = lhflx_yearly.values + shflx_yearly.values

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].plot(years, lhflx_yearly.values, alpha=0.3, color="C0", label="LHFLX")
axes[0].plot(years, shflx_yearly.values, alpha=0.3, color="C1", label="SHFLX")
if n_years > 40:
    axes[0].plot(years, lowpass(lhflx_yearly.values, 20), color="C0", lw=2)
    axes[0].plot(years, lowpass(shflx_yearly.values, 20), color="C1", lw=2)
axes[0].set_ylabel("W/m²")
axes[0].set_title("Surface Heat Fluxes \u2014 Low-Frequency Variability (post-spinup)")
axes[0].legend()

# Scatter: decadal TS vs total surface flux
if n_years > 40:
    ts_lp = lowpass(ts_vals, 20)
    flux_lp = lowpass(total_sfc_flux, 20)
    sc = axes[1].scatter(ts_lp, flux_lp, c=years, cmap="viridis", s=10)
    axes[1].set_xlabel("TS (K, 20-yr low-pass)")
    axes[1].set_ylabel("LHFLX + SHFLX (W/m², 20-yr low-pass)")
    axes[1].set_title("Decadal TS vs Surface Heat Flux")
    plt.colorbar(sc, ax=axes[1], label="Year")

fig.tight_layout()
save_fig(fig, "surface_flux_coupling")


# ---- 21. Total water path — long-term hydrological cycle stability ------
print("  Total water path …")
twp_yearly = global_yearly_mean(mds.total_water_path, lat)

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

axes[0].plot(twp_yearly.year, twp_yearly.values, alpha=0.3, color="gray")
if n_years > 40:
    axes[0].plot(twp_yearly.year, lowpass(twp_yearly.values, 20), lw=2, color="C0", label="20-yr low-pass")
axes[0].set_ylabel("Total Water Path (kg/m²)")
axes[0].set_title("Global Mean Total Water Path (post-spinup)")
axes[0].legend()

# TWP power spectrum
twp_det = detrend(twp_yearly.values)
f_tw, p_tw = welch(twp_det, fs=1.0, nperseg=min(256, n_years // 2))
axes[1].loglog(1.0 / f_tw[1:], p_tw[1:], "k-", lw=1.2)
axes[1].invert_xaxis()
axes[1].set_xlabel("Period (years)")
axes[1].set_ylabel("PSD")
axes[1].set_title(f"Power Spectrum of Total Water Path (post-spinup, N={n_years} years)")

fig.tight_layout()
save_fig(fig, "total_water_path")


# ---- Done ----------------------------------------------------------------
print(f"\nDone — saved {_fig_counter} PNGs to {OUTPUT_DIR}/")
