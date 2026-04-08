"""
General diagnostics for the ~800-year ACE2-EAMv3 piControl run.

Saves individual PNGs (300 dpi) to figs_general/ covering:
  1.  Climate drift — global-mean TS timeseries + linear trend
  2.  TOA energy balance — RESTOM with SOLIN offset
  3.  ENSO — Niño index timeseries + power spectrum from diagnostics
  4.  Precipitation patterns — time-mean map, zonal mean, ITCZ structure
  5.  Hadley/Walker proxy — zonal-mean V and TAUX/TAUY patterns
  6.  Interannual variability maps — std dev of yearly TS
  7.  Water cycle closure — budget residual evolution
  8.  Vertical temperature structure — zonal-mean T_0–T_7
  9.  Vertical moisture structure — zonal-mean specific_total_water_0–7
  10. Surface pressure — global-mean PS + spatial climatology
  11. Radiation budget maps — time-mean SW/LW at TOA and surface
  12. Surface energy balance — LHFLX + SHFLX + radiation maps
  13. Wind climatology — zonal-mean U_0–U_7 (jet structure)
  14. Prediction vs forcing — verify SOLIN consistency
  15. Seasonal cycle amplitude — TS seasonal cycle map

Usage:
    micromamba run -n xgns python general_diagnostics.py
"""

import warnings

import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.signal import welch

from config import (get_monthly_prediction_files, get_enso_diagnostic_files,
                    SOLIN_PATH, SPINUP_CUTOFF_YEAR)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILES = get_monthly_prediction_files()
ENSO_FILES = get_enso_diagnostic_files()

OUTPUT_DIR = Path("figs_general")
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

# Vertical layer labels (0 = top of atmosphere, 7 = near surface)
LAYER_LABELS = [f"Layer {i}" for i in range(8)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cosine_weights(lat):
    return np.cos(np.deg2rad(lat))


def global_yearly_mean(da, lat):
    w = cosine_weights(lat)
    return (
        da.weighted(w)
        .mean(("lon", "lat"))
        .groupby("time.year")
        .mean("time")
        .compute()
    )


def global_mean(da, lat):
    """Weighted global mean over lon/lat, no time reduction."""
    w = cosine_weights(lat)
    return da.weighted(w).mean(("lon", "lat"))


def filter_full_years(ds):
    counts = ds.time.groupby("time.year").count()
    full = counts.year.where(counts == 12, drop=True).values
    return ds.sel(time=ds.time.dt.year.isin(full))


def detrend(x):
    t = np.arange(len(x), dtype=float)
    return x - np.polyval(np.polyfit(t, x, 1), t)


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

mds["time"] = mds.valid_time
mds = mds.drop_vars("valid_time")
mds = filter_full_years(mds)

lat = mds.lat
lon = mds.lon

# Keep full dataset for drift analysis, then filter post-spinup for all other sections
mds_full = mds
mds = mds.sel(time=mds.time.dt.year >= SPINUP_CUTOFF_YEAR)
n_postspin_years = len(np.unique(mds.time.dt.year.values))

# Load SOLIN forcing
solin_ds = xr.open_dataset(SOLIN_PATH)
solin = solin_ds["SOLIN"]
solin_global = solin.weighted(cosine_weights(solin.lat)).mean(("lon", "lat")).mean().item()

print(f"  Time span (full): {mds_full.time.values[0]} → {mds_full.time.values[-1]}")
print(f"  Spinup cutoff year: {SPINUP_CUTOFF_YEAR}")
print(f"  Post-spinup years: {n_postspin_years}")
print(f"  SOLIN global mean: {solin_global:.2f} W/m²")


# ===========================================================================
# Figures
# ===========================================================================
print("Generating figures …")


# ---- 1. Climate drift — global-mean TS ----------------------------------
print("  1. Climate drift …")
ts_yearly = global_yearly_mean(mds_full.TS, lat)
years = ts_yearly.year.values
n_years = len(years)
ts_vals = ts_yearly.values

# Full-run linear trend
trend_coeffs = np.polyfit(np.arange(n_years), ts_vals, 1)
trend_per_century = trend_coeffs[0] * 100

# Post-spinup trend
post_mask = years >= SPINUP_CUTOFF_YEAR
ts_post = ts_vals[post_mask]
years_post = years[post_mask]
n_post = len(ts_post)
if n_post > 1:
    trend_post_coeffs = np.polyfit(np.arange(n_post), ts_post, 1)
    trend_post_per_century = trend_post_coeffs[0] * 100
else:
    trend_post_per_century = float('nan')

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(years, ts_vals, alpha=0.6, lw=0.8)
axes[0].plot(years, np.polyval(trend_coeffs, np.arange(n_years)), "r--", lw=2,
             label=f"Full trend: {trend_per_century:+.3f} K/century")
if n_post > 1:
    post_idx = np.where(post_mask)[0]
    axes[0].plot(years_post, np.polyval(trend_post_coeffs, np.arange(n_post)), "g--", lw=2,
                 label=f"Post-spinup trend: {trend_post_per_century:+.3f} K/century")
axes[0].axvline(SPINUP_CUTOFF_YEAR, color="green", ls="--", lw=2, alpha=0.7,
                label=f"Spinup cutoff (year {SPINUP_CUTOFF_YEAR})")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("TS (K)")
axes[0].set_title(f"Global Mean Surface Temperature -- Full Simulation (N={n_years} years)")
axes[0].legend()

# Detrended
axes[1].plot(years, detrend(ts_vals), alpha=0.6, lw=0.8)
axes[1].axhline(0, color="k", lw=0.5)
axes[1].axvline(SPINUP_CUTOFF_YEAR, color="green", ls="--", lw=2, alpha=0.7)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("TS anomaly (K)")
axes[1].set_title("Detrended Global Mean TS")

fig.tight_layout()
save_fig(fig, "climate_drift")


# ---- 2. TOA energy balance — RESTOM ------------------------------------
print("  2. TOA energy balance …")
restom = -mds.top_of_atmos_upward_shortwave_flux - mds.FLUT
restom_yearly = global_yearly_mean(restom, lat)
net_toa_yearly = global_yearly_mean(mds.net_energy_flux_toa_into_atmosphere, lat)

# Add SOLIN to get absolute RESTOM
restom_abs = solin_global + restom_yearly.values

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(restom_yearly.year, restom_abs, alpha=0.6, lw=0.8, label="RESTOM (SOLIN − SW↑ − FLUT)")
axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_ylabel("W/m²")
axes[0].set_title(f"TOA Radiative Imbalance (SOLIN = {solin_global:.2f} W/m², post-spinup, N={n_postspin_years} yr)")
axes[0].legend()

axes[1].plot(net_toa_yearly.year, net_toa_yearly.values, alpha=0.6, lw=0.8, color="C1",
             label="net_energy_flux_toa_into_atmosphere")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("W/m²")
axes[1].set_title(f"Net Energy Flux at TOA into Atmosphere (post-spinup, N={n_postspin_years} yr)")
axes[1].legend()

fig.tight_layout()
save_fig(fig, "toa_energy_balance")

# TOA maps — time-mean
net_toa_map = restom.mean("time").compute()
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.pcolormesh(lon, lat, net_toa_map + solin_global, cmap="RdBu_r",
                   shading="auto", vmin=-80, vmax=80)
ax.set_title(f"Time-Mean RESTOM Map (W/m², post-spinup, N={n_postspin_years} yr)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.colorbar(im, ax=ax, label="W/m²")
fig.tight_layout()
save_fig(fig, "toa_restom_map")


# ---- 3. ENSO variability ------------------------------------------------
print("  3. ENSO …")
if ENSO_FILES:
    enso_ds = xr.open_mfdataset(
        ENSO_FILES, combine="nested", concat_dim="time", use_cftime=True
    ).isel(sample=0)
    enso_ts = enso_ds.TS.compute()
    enso_vals = enso_ts.values
    # Drop NaNs (can appear at segment boundaries)
    valid = ~np.isnan(enso_vals)
    enso_vals = enso_vals[valid]
    enso_anom = enso_vals - np.nanmean(enso_vals)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Timeseries (use filtered length after NaN removal)
    months = np.arange(len(enso_vals))
    axes[0].plot(months / 12, enso_anom, lw=0.5, alpha=0.6)
    axes[0].fill_between(months / 12, enso_anom, 0,
                         where=enso_anom > 0, color="red", alpha=0.3, label="El Niño")
    axes[0].fill_between(months / 12, enso_anom, 0,
                         where=enso_anom < 0, color="blue", alpha=0.3, label="La Niña")
    axes[0].set_xlabel("Years from start")
    axes[0].set_ylabel("SST anomaly (K)")
    axes[0].set_title("ENSO Index (Niño Region SST Anomaly)")
    axes[0].legend()

    # Power spectrum (monthly → fs=12)
    enso_det = detrend(enso_vals)
    f_e, p_e = welch(enso_det, fs=12.0, nperseg=min(512, len(enso_det) // 2))
    period_e = 1.0 / f_e[1:]
    axes[1].loglog(period_e, p_e[1:], "k-", lw=1.2)
    axes[1].axvspan(2, 7, alpha=0.15, color="red", label="ENSO band (2–7 yr)")
    axes[1].axvline(3.5, color="red", ls=":", alpha=0.5, label="~3.5 yr (typical ENSO)")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Period (years)")
    axes[1].set_ylabel("PSD (K² yr)")
    axes[1].set_title("ENSO Power Spectrum")
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(20, 0.5)

    # Histogram of ENSO events
    axes[2].hist(enso_anom, bins=60, color="steelblue", alpha=0.7, density=True)
    axes[2].axvline(0, color="k", lw=0.5)
    axes[2].set_xlabel("SST anomaly (K)")
    axes[2].set_ylabel("Density")
    axes[2].set_title(f"ENSO Distribution (skewness = {float(np.mean(enso_anom**3) / np.std(enso_anom)**3):.2f})")

    fig.tight_layout()
    save_fig(fig, "enso_variability")
else:
    print("  ⚠ No ENSO diagnostic files found, skipping")


# ---- 4. Precipitation patterns ------------------------------------------
print("  4. Precipitation …")
precip_mean = mds.surface_precipitation_rate.mean("time").compute() * 86400  # mm/day

fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})

# Map
im = axes[0].pcolormesh(lon, lat, precip_mean, cmap="Blues", shading="auto", vmin=0, vmax=12)
axes[0].set_title(f"Time-Mean Precipitation Rate (mm/day, post-spinup, N={n_postspin_years} yr)")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[0], label="mm/day")

# Zonal mean
precip_zonal = precip_mean.mean("lon")
axes[1].plot(lat, precip_zonal, "k-", lw=2)
axes[1].fill_between(lat.values, 0, precip_zonal.values, alpha=0.3)
axes[1].set_xlabel("Latitude")
axes[1].set_ylabel("Precip (mm/day)")
axes[1].set_title(f"Zonal-Mean Precipitation -- ITCZ Structure (post-spinup, N={n_postspin_years} yr)")
axes[1].axhline(0, color="k", lw=0.3)

# Mark ITCZ (latitude of max precip in tropics)
trop_mask = np.abs(lat.values) < 30
trop_precip = precip_zonal.values.copy()
trop_precip[~trop_mask] = 0
itcz_lat = lat.values[np.argmax(trop_precip)]
axes[1].axvline(itcz_lat, color="red", ls="--", label=f"ITCZ peak: {itcz_lat:.1f}°")

# Check for double ITCZ — look for secondary peak in opposite hemisphere
if itcz_lat > 0:
    sh_precip = trop_precip.copy()
    sh_precip[lat.values > 0] = 0
else:
    sh_precip = trop_precip.copy()
    sh_precip[lat.values < 0] = 0
if sh_precip.max() > 0.5 * trop_precip.max():
    secondary_lat = lat.values[np.argmax(sh_precip)]
    axes[1].axvline(secondary_lat, color="orange", ls="--",
                    label=f"Secondary peak: {secondary_lat:.1f}° (possible double ITCZ)")

axes[1].legend(fontsize=9)

# Add precip bias annotation
global_precip = float(precip_mean.weighted(cosine_weights(lat)).mean(('lon', 'lat')))
axes[1].text(0.98, 0.95,
             f"Global mean: {global_precip:.2f} mm/day\n"
             f"Obs. ~2.7 mm/day\n"
             f"Bias: ~{((global_precip / 2.7) - 1) * 100:.0f}% high",
             transform=axes[1].transAxes, fontsize=9, verticalalignment="top",
             horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

fig.tight_layout()
save_fig(fig, "precip_patterns")


# ---- 5. Hadley/Walker circulation proxy ---------------------------------
print("  5. Hadley/Walker proxy …")

# Zonal-mean meridional wind at each vertical layer
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
for i, ax in enumerate(axes.flat):
    v_zonal = mds[f"V_{i}"].mean(("time", "lon")).compute()
    im = ax.pcolormesh(
        [0], lat, v_zonal.values[:, np.newaxis],
        cmap="RdBu_r", shading="auto", vmin=-3, vmax=3,
    )
    # Plot as line instead
    ax.cla()
    ax.plot(v_zonal, lat, lw=1.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"V_{i}", fontsize=11)
    ax.set_xlabel("m/s")
    if i % 4 == 0:
        ax.set_ylabel("Latitude")
fig.suptitle(f"Zonal-Mean Meridional Wind by Layer (Hadley Cell Proxy, post-spinup, N={n_postspin_years} yr)", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "hadley_meridional_wind")

# TAUX / TAUY maps
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, var, title, cmap in [
    (axes[0], "TAUX", "Zonal Wind Stress (TAUX, Pa)", "RdBu_r"),
    (axes[1], "TAUY", "Meridional Wind Stress (TAUY, Pa)", "RdBu_r"),
]:
    mean_map = mds[var].mean("time").compute()
    vmax = float(np.percentile(np.abs(mean_map.values), 98))
    im = ax.pcolormesh(lon, lat, mean_map, cmap=cmap, shading="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="Pa")
fig.tight_layout()
save_fig(fig, "taux_tauy_maps")

# Zonal-mean TAUX — trade winds and westerlies
fig, ax = plt.subplots(figsize=(8, 5))
taux_zonal = mds.TAUX.mean(("time", "lon")).compute()
ax.plot(taux_zonal, lat, "k-", lw=2)
ax.fill_betweenx(lat.values, 0, taux_zonal.values, where=taux_zonal.values > 0,
                 color="red", alpha=0.3, label="Westerly")
ax.fill_betweenx(lat.values, 0, taux_zonal.values, where=taux_zonal.values < 0,
                 color="blue", alpha=0.3, label="Easterly (trades)")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("TAUX (Pa)")
ax.set_ylabel("Latitude")
ax.set_title("Zonal-Mean Zonal Wind Stress — Trade Winds & Westerlies")
ax.legend()
fig.tight_layout()
save_fig(fig, "taux_zonal_mean")


# ---- 6. Interannual variability maps ------------------------------------
print("  6. Interannual variability …")
ts_yearly_map = mds.TS.groupby("time.year").mean("time").compute()
ts_std = ts_yearly_map.std("year")

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.pcolormesh(lon, lat, ts_std, cmap="magma", shading="auto", vmin=0, vmax=3)
ax.set_title(f"Interannual Std Dev of TS (K, post-spinup, N={n_postspin_years} yr)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.colorbar(im, ax=ax, label="K")
fig.tight_layout()
save_fig(fig, "ts_interannual_std")

# Same for precipitation
precip_yearly_map = (mds.surface_precipitation_rate * 86400).groupby("time.year").mean("time").compute()
precip_std = precip_yearly_map.std("year")

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.pcolormesh(lon, lat, precip_std, cmap="YlOrRd", shading="auto", vmin=0, vmax=2)
ax.set_title(f"Interannual Std Dev of Precipitation (mm/day, post-spinup, N={n_postspin_years} yr)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.colorbar(im, ax=ax, label="mm/day")
fig.tight_layout()
save_fig(fig, "precip_interannual_std")


# ---- 7. Water cycle closure ---------------------------------------------
print("  7. Water cycle closure …")
try:
    water_resid = global_yearly_mean(mds.total_water_path_budget_residual, lat)
    water_resid_valid = True
except Exception:
    water_resid_valid = False

try:
    twp_yearly = global_yearly_mean(mds.total_water_path, lat)
    twp_valid = True
except Exception:
    twp_valid = False

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Budget residual timeseries
if water_resid_valid:
    axes[0].plot(water_resid.year, water_resid.values, alpha=0.6, lw=0.8)
    axes[0].axhline(0, color="k", lw=0.5)
    n_water = len(water_resid.values)
    valid_wr = ~np.isnan(water_resid.values)
    if valid_wr.sum() > 1:
        trend_w = np.polyfit(np.arange(valid_wr.sum()), water_resid.values[valid_wr], 1)
        axes[0].set_title(f"Total Water Path Budget Residual (trend: {trend_w[0]*100:.2e}/century, post-spinup, N={n_postspin_years} yr)")
    else:
        axes[0].set_title(f"Total Water Path Budget Residual (trend: N/A, post-spinup, N={n_postspin_years} yr)")
    axes[0].set_ylabel("kg/m²/s")
else:
    axes[0].set_title("Total Water Path Budget Residual -- data unavailable")

# Total water path
if twp_valid:
    axes[1].plot(twp_yearly.year, twp_yearly.values, alpha=0.6, lw=0.8)
    axes[1].set_title(f"Global Mean Total Water Path (post-spinup, N={n_postspin_years} yr)")
    axes[1].set_ylabel("kg/m²")
else:
    axes[1].set_title("Global Mean Total Water Path -- data unavailable")

# Cumulative residual — shows integrated moisture leak
if water_resid_valid:
    wr_vals = np.where(np.isnan(water_resid.values), 0, water_resid.values)
    cumulative_resid = np.cumsum(wr_vals)
    axes[2].plot(water_resid.year, cumulative_resid, lw=1.5)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title(f"Cumulative Water Budget Residual (Integrated Moisture Leak, post-spinup)")
    axes[2].set_ylabel("Cumulative (kg/m²/s · yr)")
    axes[2].set_xlabel("Year")
else:
    axes[2].set_title("Cumulative Water Budget Residual -- data unavailable")

fig.tight_layout()
save_fig(fig, "water_cycle_closure")

# Budget residual map
resid_map = mds.total_water_path_budget_residual.mean("time").compute()
fig, ax = plt.subplots(figsize=(12, 5))
vmax_val = np.abs(resid_map.values)
vmax_val = vmax_val[~np.isnan(vmax_val)]
vmax = float(np.percentile(vmax_val, 98)) if len(vmax_val) > 0 else 1.0
im = ax.pcolormesh(lon, lat, resid_map, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
ax.set_title(f"Time-Mean Water Budget Residual Map (post-spinup, N={n_postspin_years} yr)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.colorbar(im, ax=ax, label="kg/m²/s")
fig.tight_layout()
save_fig(fig, "water_residual_map")


# ---- 8. Vertical temperature structure -----------------------------------
print("  8. Vertical temperature …")
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True, sharex=True)
t_zonal_all = []
for i, ax in enumerate(axes.flat):
    t_zonal = mds[f"T_{i}"].mean(("time", "lon")).compute()
    t_zonal_all.append(t_zonal.values)
    ax.plot(t_zonal, lat, lw=2)
    ax.set_title(f"T_{i}", fontsize=11)
    ax.set_xlabel("K")
    if i % 4 == 0:
        ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3)
fig.suptitle(f"Zonal-Mean Temperature by Vertical Layer (T_0=top, T_7=sfc, post-spinup, N={n_postspin_years} yr)", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "vertical_temperature")

# Vertical profile at equator and poles
fig, ax = plt.subplots(figsize=(8, 6))
layers = np.arange(8)
for lat_sel, label, color in [(0, "Equator (0°)", "C0"), (45, "45°N", "C1"),
                               (-45, "45°S", "C2"), (80, "80°N (Arctic)", "C3")]:
    lat_idx = np.argmin(np.abs(lat.values - lat_sel))
    profile = [mds[f"T_{i}"].mean(("time", "lon")).compute().values[lat_idx] for i in range(8)]
    ax.plot(profile, layers, "o-", label=label, color=color, lw=2)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Layer index (0=top, 7=near sfc)")
ax.set_title("Vertical Temperature Profiles")
ax.invert_yaxis()
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, "vertical_temp_profiles")


# ---- 9. Vertical moisture structure --------------------------------------
print("  9. Vertical moisture …")
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True, sharex=True)
for i, ax in enumerate(axes.flat):
    q_zonal = mds[f"specific_total_water_{i}"].mean(("time", "lon")).compute()
    ax.plot(q_zonal * 1000, lat, lw=2)  # g/kg
    ax.set_title(f"specific_total_water_{i}", fontsize=10)
    ax.set_xlabel("g/kg")
    if i % 4 == 0:
        ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3)
fig.suptitle(f"Zonal-Mean Specific Total Water by Layer (g/kg, post-spinup, N={n_postspin_years} yr)", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "vertical_moisture")

# Moisture profile at equator
fig, ax = plt.subplots(figsize=(8, 6))
for lat_sel, label, color in [(0, "Equator", "C0"), (45, "45°N", "C1"),
                               (-45, "45°S", "C2"), (80, "80°N", "C3")]:
    lat_idx = np.argmin(np.abs(lat.values - lat_sel))
    profile = [mds[f"specific_total_water_{i}"].mean(("time", "lon")).compute().values[lat_idx] * 1000
               for i in range(8)]
    ax.plot(profile, layers, "o-", label=label, color=color, lw=2)
ax.set_xlabel("Specific Total Water (g/kg)")
ax.set_ylabel("Layer index (0=top, 7=near sfc)")
ax.set_title("Vertical Moisture Profiles")
ax.invert_yaxis()
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, "vertical_moisture_profiles")


# ---- 10. Surface pressure -----------------------------------------------
print("  10. Surface pressure …")
ps_yearly = global_yearly_mean(mds.PS, lat)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Timeseries
n_ps = len(ps_yearly.values)
ps_trend = np.polyfit(np.arange(n_ps), ps_yearly.values, 1)
axes[0].plot(ps_yearly.year, ps_yearly.values / 100, alpha=0.6, lw=0.8)  # hPa
axes[0].set_ylabel("PS (hPa)")
axes[0].set_title(f"Global Mean Surface Pressure (trend: {ps_trend[0]*100/100:.4f} hPa/century, post-spinup, N={n_postspin_years} yr)")

# Map
ps_mean = (mds.PS.mean("time").compute()) / 100
im = axes[1].pcolormesh(lon, lat, ps_mean, cmap="viridis", shading="auto")
axes[1].set_title(f"Time-Mean Surface Pressure (hPa, post-spinup, N={n_postspin_years} yr)")
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[1], label="hPa")

fig.tight_layout()
save_fig(fig, "surface_pressure")


# ---- 11. Radiation budget maps ------------------------------------------
print("  11. Radiation maps …")
rad_vars = [
    ("FSDS", "Downwelling SW at Surface"),
    ("FLDS", "Downwelling LW at Surface"),
    ("FLUT", "Outgoing LW at TOA"),
    ("top_of_atmos_upward_shortwave_flux", "Reflected SW at TOA"),
    ("surface_upward_longwave_flux", "Upwelling LW at Surface"),
    ("surface_upward_shortwave_flux", "Upwelling SW at Surface"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for ax, (var, title) in zip(axes.flat, rad_vars):
    mean_map = mds[var].mean("time").compute()
    im = ax.pcolormesh(lon, lat, mean_map, cmap="inferno", shading="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    plt.colorbar(im, ax=ax, label="W/m²", shrink=0.8)
fig.suptitle(f"Time-Mean Radiation Budget Components (W/m², post-spinup, N={n_postspin_years} yr)", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "radiation_maps")

# Zonal-mean radiation profiles
fig, ax = plt.subplots(figsize=(10, 6))
for var, label in rad_vars:
    zonal = mds[var].mean(("time", "lon")).compute()
    ax.plot(lat, zonal, lw=1.5, label=label)
ax.set_xlabel("Latitude")
ax.set_ylabel("W/m²")
ax.set_title("Zonal-Mean Radiation Components")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, "radiation_zonal")


# ---- 12. Surface energy balance -----------------------------------------
print("  12. Surface energy balance …")
# Net surface = FSDS - surface_upward_shortwave + FLDS - surface_upward_longwave - LHFLX - SHFLX
net_sfc = (
    mds.FSDS - mds.surface_upward_shortwave_flux
    + mds.FLDS - mds.surface_upward_longwave_flux
    - mds.LHFLX - mds.SHFLX
).mean("time").compute()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Net surface energy map
vmax = float(np.percentile(np.abs(net_sfc.values), 98))
im = axes[0].pcolormesh(lon, lat, net_sfc, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
axes[0].set_title("Net Surface Energy Balance (W/m²)")
plt.colorbar(im, ax=axes[0], label="W/m²")

# LHFLX map
lhflx_mean = mds.LHFLX.mean("time").compute()
im = axes[1].pcolormesh(lon, lat, lhflx_mean, cmap="YlOrRd", shading="auto", vmin=0, vmax=200)
axes[1].set_title("Latent Heat Flux (W/m²)")
plt.colorbar(im, ax=axes[1], label="W/m²")

# SHFLX map
shflx_mean = mds.SHFLX.mean("time").compute()
im = axes[2].pcolormesh(lon, lat, shflx_mean, cmap="YlOrRd", shading="auto", vmin=0, vmax=80)
axes[2].set_title("Sensible Heat Flux (W/m²)")
plt.colorbar(im, ax=axes[2], label="W/m²")

for ax in axes:
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
fig.tight_layout()
save_fig(fig, "surface_energy_balance")

# Bowen ratio map (SHFLX / LHFLX) — land vs ocean signature
fig, ax = plt.subplots(figsize=(12, 5))
with np.errstate(divide="ignore", invalid="ignore"):
    bowen = shflx_mean.values / np.where(lhflx_mean.values > 5, lhflx_mean.values, np.nan)
im = ax.pcolormesh(lon, lat, bowen, cmap="RdYlBu_r", shading="auto", vmin=0, vmax=2)
ax.set_title("Bowen Ratio (SHFLX / LHFLX) — Land/Ocean Contrast")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.colorbar(im, ax=ax, label="Bowen ratio")
fig.tight_layout()
save_fig(fig, "bowen_ratio")


# ---- 13. Wind climatology — zonal-mean U --------------------------------
print("  13. Zonal wind structure …")
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True, sharex=True)
for i, ax in enumerate(axes.flat):
    u_zonal = mds[f"U_{i}"].mean(("time", "lon")).compute()
    ax.plot(u_zonal, lat, lw=2)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"U_{i}", fontsize=11)
    ax.set_xlabel("m/s")
    if i % 4 == 0:
        ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3)
fig.suptitle(f"Zonal-Mean Zonal Wind by Layer (Jet Stream Structure, post-spinup, N={n_postspin_years} yr)", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "zonal_wind_structure")

# U wind at ~250 hPa equivalent (layer 2 or 3) — jet stream map
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, layer, title in [
    (axes[0], 2, "U_2 (Upper Troposphere — Jet Level)"),
    (axes[1], 7, "U_7 (Near Surface)"),
]:
    u_map = mds[f"U_{layer}"].mean("time").compute()
    vmax = float(np.percentile(np.abs(u_map.values), 98))
    im = ax.pcolormesh(lon, lat, u_map, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="m/s")
fig.suptitle("Time-Mean Zonal Wind Maps", fontsize=13)
fig.tight_layout()
save_fig(fig, "jet_stream_maps")


# ---- 14. Prediction vs forcing — SOLIN consistency ----------------------
print("  14. SOLIN consistency …")
# Compute the implied SOLIN from predictions: SOLIN = top_of_atmos_upward_shortwave + net absorbed SW
# Actually, check the target file for SOLIN directly
solin_monthly = solin.weighted(cosine_weights(solin.lat)).mean(("lon", "lat")).compute()

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

# SOLIN seasonal cycle
axes[0].plot(np.arange(len(solin_monthly)), solin_monthly.values, "k-o", lw=2)
axes[0].set_xlabel("Month")
axes[0].set_ylabel("W/m²")
axes[0].set_title(f"SOLIN Forcing Seasonal Cycle (mean = {solin_global:.2f} W/m²)")
axes[0].grid(alpha=0.3)

# SOLIN spatial map (annual mean)
solin_map = solin.mean("time").compute()
im = axes[1].pcolormesh(solin.lon, solin.lat, solin_map, cmap="YlOrRd", shading="auto")
axes[1].set_title("Annual-Mean SOLIN Map (W/m²)")
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[1], label="W/m²")

fig.tight_layout()
save_fig(fig, "solin_forcing")


# ---- 15. Seasonal cycle amplitude ---------------------------------------
print("  15. Seasonal cycle …")
# Compute monthly climatology and take max - min as seasonal amplitude
ts_monthly_clim = mds.TS.groupby("time.month").mean("time").compute()
seasonal_amp = ts_monthly_clim.max("month") - ts_monthly_clim.min("month")

fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})

# Map
im = axes[0].pcolormesh(lon, lat, seasonal_amp, cmap="YlOrRd", shading="auto", vmin=0, vmax=50)
axes[0].set_title(f"TS Seasonal Cycle Amplitude (max month - min month, K, post-spinup, N={n_postspin_years} yr)")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
plt.colorbar(im, ax=axes[0], label="K")

# Zonal mean seasonal cycle — Hovmöller
im2 = axes[1].pcolormesh(
    np.arange(1, 13), lat,
    ts_monthly_clim.mean("lon").T - ts_monthly_clim.mean(("lon", "month")),
    cmap="RdBu_r", shading="auto", vmin=-15, vmax=15,
)
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Latitude")
axes[1].set_title("Zonal-Mean TS Seasonal Cycle Anomaly (K)")
plt.colorbar(im2, ax=axes[1], label="K")

fig.tight_layout()
save_fig(fig, "seasonal_cycle")


# ---- 16. TS climatology map ---------------------------------------------
print("  16. TS climatology …")
ts_mean = mds.TS.mean("time").compute()

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.pcolormesh(lon, lat, ts_mean - 273.15, cmap="RdYlBu_r", shading="auto", vmin=-40, vmax=35)
ax.set_title(f"Time-Mean Surface Temperature (C, post-spinup, N={n_postspin_years} yr)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.colorbar(im, ax=ax, label="°C")
fig.tight_layout()
save_fig(fig, "ts_climatology")


# ---- 17. Summary statistics table ----------------------------------------
print("  17. Summary …")
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis("off")

# NaN-safe helpers
twp_mean = float(np.nanmean(twp_yearly.values)) if twp_valid else float('nan')
resid_mean = float(np.nanmean(water_resid.values)) if water_resid_valid else float('nan')

# Post-spinup TS stats
ts_post_vals = ts_vals[post_mask]
ts_post_mean = float(np.mean(ts_post_vals)) if len(ts_post_vals) > 0 else float('nan')

def _fmt(val, fmt_str, suffix="", fallback="N/A"):
    """Format a value, returning fallback if NaN."""
    if np.isnan(val):
        return fallback
    return f"{val:{fmt_str}}{suffix}"

lines = []
lines.append("ACE2-EAMv3 piControl -- Summary Statistics")
lines.append("=" * 64)

# --- Simulation info ---
lines.append("")
lines.append("  SIMULATION INFO")
lines.append("  " + "-" * 40)
lines.append(f"  {'N years (full run)':35s} {n_years}")
lines.append(f"  {'Year range (full)':35s} {years[0]}-{years[-1]}")
lines.append(f"  {'Spinup cutoff year':35s} {SPINUP_CUTOFF_YEAR}")
lines.append(f"  {'N years (post-spinup)':35s} {n_postspin_years}")
lines.append(f"  {'SOLIN':35s} {solin_global:.2f} W/m2")

# --- Temperature ---
lines.append("")
lines.append("  TEMPERATURE")
lines.append("  " + "-" * 40)
lines.append(f"  {'Global mean TS (full)':35s} {ts_vals.mean():.2f} K  ({ts_vals.mean()-273.15:.2f} C)")
lines.append(f"  {'Global mean TS (post-spinup)':35s} {_fmt(ts_post_mean, '.2f', ' K')}  ({_fmt(ts_post_mean - 273.15, '.2f', ' C')})")
lines.append(f"  {'TS trend (full run)':35s} {trend_per_century:+.4f} K/century")
lines.append(f"  {'TS trend (post-spinup)':35s} {_fmt(trend_post_per_century, '+.4f', ' K/century')}")
lines.append(f"  {'TS interannual std (post-spinup)':35s} {np.std(detrend(ts_post_vals)):.4f} K" if len(ts_post_vals) > 1 else f"  {'TS interannual std (post-spinup)':35s} N/A")

# --- Energy balance ---
lines.append("")
lines.append("  ENERGY BALANCE")
lines.append("  " + "-" * 40)
lines.append(f"  {'RESTOM mean':35s} {restom_abs.mean():.2f} W/m2")
lines.append(f"  {'RESTOM std':35s} {np.std(restom_abs):.2f} W/m2")

# --- Precipitation ---
lines.append("")
lines.append("  PRECIPITATION")
lines.append("  " + "-" * 40)
lines.append(f"  {'Global mean precip':35s} {float(precip_mean.weighted(cosine_weights(lat)).mean(('lon','lat'))):.2f} mm/day")
lines.append(f"  {'Observed (approx)':35s} ~2.7 mm/day")

# --- Surface pressure ---
lines.append("")
lines.append("  SURFACE PRESSURE")
lines.append("  " + "-" * 40)
lines.append(f"  {'Global mean PS':35s} {ps_yearly.values.mean()/100:.2f} hPa")

# --- Water cycle ---
lines.append("")
lines.append("  WATER CYCLE")
lines.append("  " + "-" * 40)
lines.append(f"  {'Total water path (mean)':35s} {_fmt(twp_mean, '.2f', ' kg/m2')}")
lines.append(f"  {'Water residual (mean)':35s} {_fmt(resid_mean, '.2e', ' kg/m2/s')}")

table_text = "\n".join(lines)
ax.text(0.03, 0.97, table_text,
        transform=ax.transAxes, fontsize=11, verticalalignment="top", fontfamily="monospace")
fig.tight_layout()
save_fig(fig, "summary_stats")


# ---- Done ----------------------------------------------------------------
print(f"\nDone — saved {_fig_counter} PNGs to {OUTPUT_DIR}/")
