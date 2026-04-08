"""
Cross-variable diagnostics from monthly ACE2-EAMv3 piControl output.

Analyzes relationships between variables that reveal model dynamics:
  1. ENSO teleconnection composites (El Niño vs La Niña)
  2. P − E hydrological cycle patterns
  3. Planetary albedo and its variability
  4. Gregory plot — climate feedback parameter
  5. Spatial correlation length scales of TS
  6. Monsoon diagnostics — seasonal cycle and interannual variability

============================================================================
SCIENTIFIC BACKGROUND & INTERPRETATION GUIDE
============================================================================

--- 1. ENSO Teleconnections (Ropelewski & Halpert 1987, Trenberth 1997) ----
ENSO is the dominant mode of interannual climate variability. During
El Niño (warm Niño-3.4 SSTs), canonical responses include:
  • Tropical Pacific: warm TS, increased central Pacific precip
  • Indian Ocean: warm (with 1-season lag)
  • Maritime Continent: dry, reduced convection
  • Western US: wet winters
  • SE Asia/Australia: dry
  • Pacific-North America (PNA): positive height anomaly

We composite model fields for El Niño (top 15% years) vs La Niña
(bottom 15%) and plot the difference. If ACE reproduces these
teleconnections, it has learned realistic tropical-extratropical
coupling — a hard test for ML emulators because it requires correct
representation of wave propagation and energy transport.

Key checks:
  • Is the TS composite symmetric? (Obs show El Niño/La Niña asymmetry)
  • Does precip shift to central Pacific during El Niño?
  • Are extratropical teleconnections (PNA, NAO modulation) present?

--- 2. P − E Hydrological Cycle (Held & Soden 2006) -----------------------
Precipitation minus evaporation (P − E) reveals the net moisture flux
into/out of the atmosphere. Evaporation ≈ LHFLX / L_v where L_v is
the latent heat of vaporization (2.5 × 10⁶ J/kg).

Expected patterns:
  • P − E > 0: ITCZ, mid-latitude storm tracks (moisture convergence)
  • P − E < 0: subtropical ocean deserts (moisture source regions)
  • Net P − E should be ~0 globally (conservation)

The zonal-mean P − E profile directly reflects the Hadley and Ferrel
circulation: the subtropics are dried by descending air, the ITCZ and
mid-latitudes are moistened by ascending/converging air.

If ACE's P − E pattern is wrong, the implied atmospheric moisture
transport is unrealistic, even if P and E separately look reasonable.

--- 3. Planetary Albedo (Stephens et al. 2015) -----------------------------
Albedo α = reflected SW / incoming SW = top_of_atmos_upward_shortwave /
SOLIN. This integrates the effects of clouds, surface (ice, vegetation),
and atmospheric scattering.

Expected values:
  • Global mean: ~0.29 (about 100 W/m² reflected out of 340 W/m²)
  • Bright regions: ice caps (>0.6), thick cloud decks (>0.5), deserts (~0.35)
  • Dark regions: clear-sky ocean (<0.1)

Interannual variability in albedo is dominated by:
  • Cloud changes (especially low clouds over subtropical oceans)
  • Sea ice changes
  • If albedo variability is too low, ACE may not represent cloud feedbacks

--- 4. Gregory Plot (Gregory et al. 2004) ----------------------------------
Scatter global-mean net TOA radiative flux (N) vs global-mean TS.
In a piControl with constant forcing, this scatter reveals the
climate feedback parameter λ = dN/dTS.

  • Slope (λ) < 0: system is stable (radiative damping), expected
  • Typical λ: −1 to −2 W/m²/K for equilibrium climate
  • Tight cluster: strong restoring, small variability
  • Spread along the regression line: forced-like internal variability
  • Off-diagonal scatter: stochastic radiative forcing (clouds)

In piControl, this is purely internal variability, so the Gregory
plot tests whether ACE's TOA response to TS perturbations follows
the expected negative feedback relationship.

--- 5. Spatial Correlation Length Scales (Jones et al. 1999) ----------------
The e-folding decorrelation distance of surface temperature anomalies
measures the spatial coherence of the model's climate. Expected:
  • Ocean: large decorrelation scales (>2000 km) due to ocean mixing
  • Land: shorter scales (500–1500 km) due to heterogeneous surface
  • Tropics: large zonal scales (planetary waves)
  • High latitudes: moderate scales

If ACE produces very short correlation scales, it's too "noisy" —
each grid point varies independently. If scales are too long, the
model is too smooth and lacks realistic mesoscale variability.

--- 6. Monsoon Diagnostics (Wang & Ding 2008, CLIVAR) ----------------------
The global monsoon is diagnosed by the seasonal reversal of:
  • Winds: TAUX/TAUY change sign between DJF and JJA
  • Precipitation: wet season ≫ dry season

Key regions and expected behavior:
  • South Asian monsoon: strongest, most variable (σ ~ 10–15% of mean)
  • West African monsoon: onset in June, retreat by October
  • East Asian monsoon: driven by land-sea thermal contrast
  • American monsoons: North (Jul–Sep) and South (Dec–Feb)

Monsoon strength index = JJA−DJF precip over monsoon domains.
In 800 years, we can examine the full range of monsoon variability
and check whether ACE produces realistic interannual variations
(driven by ENSO, IOD, and internal variability).

============================================================================

Usage:
    micromamba run -n xgns python cross_variable_monthly.py
    micromamba run -n xgns python cross_variable_monthly.py --overwrite
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from config import (get_monthly_prediction_files, get_enso_diagnostic_files,
                    SOLIN_PATH, SPINUP_CUTOFF_YEAR)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILES = get_monthly_prediction_files()
ENSO_FILES = get_enso_diagnostic_files()

OUTPUT_DIR = Path("figs_cross_variable")
OUTPUT_DIR.mkdir(exist_ok=True)
_fig_counter = 0

OVERWRITE = "--overwrite" in sys.argv


def save_fig(fig, name):
    """Save figure as numbered PNG at 300 dpi."""
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
    """Bump counter without saving — used when skipping a section."""
    global _fig_counter
    _fig_counter += 1
    path = OUTPUT_DIR / f"{_fig_counter:02d}_{name}.png"
    print(f"    → {path} (exists, skipping computation)")


def section_done(*names):
    """Check if all figures for a section already exist."""
    if OVERWRITE:
        return False
    for i, name in enumerate(names):
        path = OUTPUT_DIR / f"{_fig_counter + 1 + i:02d}_{name}.png"
        if not path.exists():
            return False
    return True


def cosine_weights(lat):
    return np.cos(np.deg2rad(lat))


def global_mean_ts(da, w):
    """Weighted spatial mean, preserving time dimension."""
    return da.weighted(w).mean(("lon", "lat"))


def filter_full_years(ds):
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

mds["time"] = mds.valid_time
mds = mds.drop_vars("valid_time")
mds = filter_full_years(mds)
mds = mds.sel(time=mds.time.dt.year >= SPINUP_CUTOFF_YEAR)

lat = mds.lat
lon = mds.lon
nlat = len(lat)
nlon = len(lon)
w = cosine_weights(lat)

# SOLIN for albedo calculation
solin_ds = xr.open_dataset(SOLIN_PATH)
solin = solin_ds["SOLIN"]
solin_clim = solin.groupby("time.month").mean("time")  # (12, lat, lon)
solin_global = float(solin.weighted(cosine_weights(solin.lat)).mean(("lon", "lat", "time")).compute())

n_months = len(mds.time)
yr_start = int(mds.time.dt.year.min())
yr_end = int(mds.time.dt.year.max())
n_years = yr_end - yr_start + 1
print(f"  {n_months} months loaded (post-spinup: years {yr_start}–{yr_end}, N={n_years})")

# ===========================================================================
# 1. ENSO Teleconnections
# ===========================================================================
# Always compute Niño-3.4 index — cheap and needed by later sections
print("\n  Computing Niño-3.4 index …")
nino34 = (
    mds.TS.sel(lat=slice(-5, 5), lon=slice(190, 240))
    .weighted(cosine_weights(mds.TS.sel(lat=slice(-5, 5)).lat))
    .mean(("lon", "lat"))
    .compute()
)
nino34_clim = nino34.groupby("time.month").mean("time")
nino34_anom = (nino34.groupby("time.month") - nino34_clim).compute()
nino34_annual = nino34_anom.groupby("time.year").mean("time").compute()
years = nino34_annual.year.values
n_composite = max(int(len(years) * 0.15), 10)
el_nino_years = years[np.argsort(nino34_annual.values)[-n_composite:]]
la_nina_years = years[np.argsort(nino34_annual.values)[:n_composite]]

print("  1. ENSO teleconnections …")
if section_done("enso_teleconnections", "enso_zonal_response", "enso_asymmetry"):
    for name in ["enso_teleconnections", "enso_zonal_response", "enso_asymmetry"]:
        skip_fig(name)
else:
    print(f"    Compositing {n_composite} El Niño and {n_composite} La Niña years")

    vars_to_composite = {
        "TS": ("TS", "K", "RdBu_r", 2),
        "Precip": ("surface_precipitation_rate", "mm/day", "BrBG", 2),
        "LHFLX": ("LHFLX", "W/m²", "RdBu_r", 20),
        "SHFLX": ("SHFLX", "W/m²", "RdBu_r", 10),
        "TAUX": ("TAUX", "Pa", "RdBu_r", 0.03),
    }

    fig, axes = plt.subplots(len(vars_to_composite), 1,
                              figsize=(14, 4 * len(vars_to_composite)))
    for idx, (label, (varname, unit, cmap, vmax)) in enumerate(vars_to_composite.items()):
        ax = axes[idx]
        da_clim = mds[varname].groupby("time.month").mean("time")
        da_anom = (mds[varname].groupby("time.month") - da_clim)
        da_annual = da_anom.groupby("time.year").mean("time").compute()
        diff = da_annual.sel(year=el_nino_years).mean("year") - da_annual.sel(year=la_nina_years).mean("year")
        if "Precip" in label:
            diff = diff * 86400
        im = ax.pcolormesh(lon, lat, diff, cmap=cmap, shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{label}: El Niño − La Niña composite ({unit})")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, label=unit)
    axes[-1].set_xlabel("Longitude")
    fig.suptitle(f"ENSO Teleconnections (top/bottom {n_composite} years)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "enso_teleconnections")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (label, (varname, unit, cmap, vmax)) in zip(axes.flat, list(vars_to_composite.items())[:4]):
        da_clim = mds[varname].groupby("time.month").mean("time")
        da_anom = (mds[varname].groupby("time.month") - da_clim)
        da_annual = da_anom.groupby("time.year").mean("time").mean("lon").compute()
        scale = 86400 if "Precip" in label else 1
        ax.plot(lat, da_annual.sel(year=el_nino_years).mean("year") * scale, "r-", lw=2, label="El Niño")
        ax.plot(lat, da_annual.sel(year=la_nina_years).mean("year") * scale, "b-", lw=2, label="La Niña")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Latitude"); ax.set_ylabel(unit)
        ax.set_title(f"Zonal-Mean {label} Anomaly")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "enso_zonal_response")

    ts_clim = mds.TS.groupby("time.month").mean("time")
    ts_annual = (mds.TS.groupby("time.month") - ts_clim).groupby("time.year").mean("time").compute()
    asymmetry = ts_annual.sel(year=el_nino_years).mean("year") + ts_annual.sel(year=la_nina_years).mean("year")
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(lon, lat, asymmetry, cmap="RdBu_r", shading="auto", vmin=-0.5, vmax=0.5)
    ax.set_title("ENSO Asymmetry: El Niño + La Niña TS composite (0 = symmetric)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="K")
    fig.tight_layout()
    save_fig(fig, "enso_asymmetry")

# ---------------------------------------------------------------------------
# 1b. ENSO Lifecycle Composites
# ---------------------------------------------------------------------------
print("  1b. ENSO lifecycle …")
# Only compute if figures don't exist
if not section_done("enso_lifecycle_hovmoller", "enso_lifecycle_maps"):
    lags = np.arange(-12, 13)  # months

    # Compute TS monthly anomalies
    ts_clim_monthly = mds.TS.groupby("time.month").mean("time")
    ts_anom_monthly = (mds.TS.groupby("time.month") - ts_clim_monthly).compute()
    nino_vals_lc = nino34_anom.values

    # Tropical band for Hovmoller (5S-5N)
    ts_trop = ts_anom_monthly.sel(lat=slice(-5, 5)).mean("lat").values  # (time, lon)

    n_time = len(nino_vals_lc)
    reg_hovmoller = np.zeros((len(lags), len(lon)))

    for li, lag in enumerate(lags):
        if lag >= 0:
            nino_seg = nino_vals_lc[:n_time-lag]
            ts_seg = ts_trop[lag:n_time]
        else:
            nino_seg = nino_vals_lc[-lag:n_time]
            ts_seg = ts_trop[:n_time+lag]
        # Regression coefficient
        nino_std = np.std(nino_seg)
        if nino_std > 0:
            for j in range(len(lon)):
                reg_hovmoller[li, j] = np.corrcoef(nino_seg, ts_seg[:, j])[0, 1] * np.std(ts_seg[:, j]) / nino_std

    # Hovmoller plot: lag vs longitude
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.pcolormesh(lon, lags, reg_hovmoller, cmap="RdBu_r", shading="auto", vmin=-1, vmax=1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Lag (months, positive = TS leads)")
    ax.set_title("ENSO Lifecycle: Lag Regression of Tropical TS (5\u00b0S\u20135\u00b0N) onto Ni\u00f1o-3.4")
    ax.axhline(0, color="k", lw=0.5)
    plt.colorbar(im, ax=ax, label="K per K of Ni\u00f1o-3.4")
    fig.tight_layout()
    save_fig(fig, "enso_lifecycle_hovmoller")

    # Lag regression maps at selected lags
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    selected_lags = [-6, -3, 0, 3, 6, 12]
    ts_full = ts_anom_monthly.values  # (time, lat, lon)

    for ax, lag in zip(axes.flat, selected_lags):
        if lag >= 0:
            nino_seg = nino_vals_lc[:n_time-lag]
            ts_seg = ts_full[lag:n_time]
        else:
            nino_seg = nino_vals_lc[-lag:n_time]
            ts_seg = ts_full[:n_time+lag]

        nino_std = np.std(nino_seg)
        reg_map = np.zeros((nlat, nlon))
        if nino_std > 0:
            for j in range(nlat):
                for i in range(nlon):
                    reg_map[j, i] = np.corrcoef(nino_seg, ts_seg[:, j, i])[0, 1] * np.std(ts_seg[:, j, i]) / nino_std

        im = ax.pcolormesh(lon, lat, reg_map, cmap="RdBu_r", shading="auto", vmin=-1.5, vmax=1.5)
        sign = "+" if lag >= 0 else ""
        ax.set_title(f"Lag {sign}{lag} months", fontsize=11)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        plt.colorbar(im, ax=ax, label="K/K", shrink=0.8)

    fig.suptitle("ENSO Lifecycle: Lag Regression of TS onto Ni\u00f1o-3.4", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "enso_lifecycle_maps")
else:
    for name in ["enso_lifecycle_hovmoller", "enso_lifecycle_maps"]:
        skip_fig(name)


# ===========================================================================
# 2. P − E Hydrological Cycle
# ===========================================================================
print("  2. P − E …")
if section_done("p_minus_e", "p_minus_e_timeseries"):
    for name in ["p_minus_e", "p_minus_e_timeseries"]:
        skip_fig(name)
    p_global = e_global = 0.0  # placeholders for summary
else:
    L_v = 2.5e6
    precip = mds.surface_precipitation_rate
    evap = mds.LHFLX / L_v

    p_minus_e = (precip - evap).mean("time").compute() * 86400

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})
    vmax = 6
    im = axes[0].pcolormesh(lon, lat, p_minus_e, cmap="BrBG", shading="auto", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Time-Mean P \u2212 E (mm/day, post-spinup)")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im, ax=axes[0], label="mm/day")

    p_zonal = precip.mean(("time", "lon")).compute() * 86400
    e_zonal = evap.mean(("time", "lon")).compute() * 86400
    pe_zonal = p_minus_e.mean("lon")
    axes[1].plot(lat, p_zonal, "b-", lw=2, label="P")
    axes[1].plot(lat, e_zonal, "r-", lw=2, label="E")
    axes[1].plot(lat, pe_zonal, "k-", lw=2.5, label="P − E")
    axes[1].fill_between(lat.values, 0, pe_zonal.values,
                         where=pe_zonal.values > 0, alpha=0.2, color="blue", label="Net moisture sink")
    axes[1].fill_between(lat.values, 0, pe_zonal.values,
                         where=pe_zonal.values < 0, alpha=0.2, color="red", label="Net moisture source")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Latitude"); axes[1].set_ylabel("mm/day")
    axes[1].set_title("Zonal-Mean Hydrological Cycle")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, "p_minus_e")

    p_global = float(precip.weighted(w).mean(("lon", "lat", "time")).compute() * 86400)
    e_global = float(evap.weighted(w).mean(("lon", "lat", "time")).compute() * 86400)
    print(f"    Global mean P = {p_global:.3f} mm/day, E = {e_global:.3f} mm/day, "
          f"P−E = {p_global - e_global:.4f} mm/day")

    pe_yearly = (precip - evap).weighted(w).mean(("lon", "lat")).groupby("time.year").mean("time").compute() * 86400
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pe_yearly.year, pe_yearly.values, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Year"); ax.set_ylabel("P − E (mm/day)")
    ax.set_title("Global Mean P − E Over Time (should hover near zero)")
    fig.tight_layout()
    save_fig(fig, "p_minus_e_timeseries")


# ===========================================================================
# 3. Planetary Albedo
# ===========================================================================
print("  3. Planetary albedo …")
# Initialize defaults for summary
albedo_map = None
slope = slope2 = 0.0
efold_dist = np.zeros(nlat)

if section_done("albedo_map", "albedo_timeseries", "albedo_variability"):
    for name in ["albedo_map", "albedo_timeseries", "albedo_variability"]:
        skip_fig(name)
else:
    sw_up = mds.top_of_atmos_upward_shortwave_flux
    solin_map = solin_clim.compute()
    sw_up_mean = sw_up.mean("time").compute()
    solin_annual_mean = solin_map.mean("month").values

    with np.errstate(divide="ignore", invalid="ignore"):
        albedo_map = sw_up_mean.values / np.where(solin_annual_mean > 10, solin_annual_mean, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    im = axes[0].pcolormesh(lon, lat, albedo_map, cmap="Greys_r", shading="auto", vmin=0, vmax=0.8)
    axes[0].set_title("Time-Mean Planetary Albedo (reflected SW / SOLIN)")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im, ax=axes[0], label="Albedo")
    albedo_zonal = np.nanmean(albedo_map, axis=1)
    axes[1].plot(lat, albedo_zonal, "k-", lw=2)
    axes[1].axhline(0.29, color="red", ls="--", label="Observed global mean (~0.29)")
    axes[1].set_xlabel("Latitude"); axes[1].set_ylabel("Albedo")
    axes[1].set_title(f"Zonal-Mean Albedo (global mean = {np.nanmean(albedo_map * w.values[:,None]) / np.nanmean(w.values):.3f})")
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 0.7)
    fig.tight_layout()
    save_fig(fig, "albedo_map")

    sw_up_yearly = sw_up.weighted(w).mean(("lon", "lat")).groupby("time.year").mean("time").compute()
    albedo_yearly = sw_up_yearly / solin_global
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(albedo_yearly.year, albedo_yearly.values, alpha=0.6, lw=0.8)
    trend = np.polyfit(np.arange(len(albedo_yearly)), albedo_yearly.values, 1)
    ax.plot(albedo_yearly.year, np.polyval(trend, np.arange(len(albedo_yearly))),
            "r--", lw=2, label=f"Trend: {trend[0]*100:.6f}/century")
    ax.set_xlabel("Year"); ax.set_ylabel("Planetary Albedo")
    ax.set_title("Global-Mean Planetary Albedo Over Time"); ax.legend()
    fig.tight_layout()
    save_fig(fig, "albedo_timeseries")

    sw_up_yearly_map = sw_up.groupby("time.year").mean("time").compute()
    with np.errstate(divide="ignore", invalid="ignore"):
        albedo_yearly_map = sw_up_yearly_map.values / np.where(
            solin_annual_mean[np.newaxis] > 10, solin_annual_mean[np.newaxis], np.nan)
    albedo_std = np.nanstd(albedo_yearly_map, axis=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(lon, lat, albedo_std, cmap="YlOrRd", shading="auto", vmin=0, vmax=0.03)
    ax.set_title("Interannual Std Dev of Albedo (cloud + ice variability proxy)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="Albedo σ")
    fig.tight_layout()
    save_fig(fig, "albedo_variability")


# ===========================================================================
# 4. Gregory Plot
# ===========================================================================
print("  4. Gregory plot …")
if section_done("gregory_plot", "gregory_regional"):
    for name in ["gregory_plot", "gregory_regional"]:
        skip_fig(name)
else:
    ts_global = mds.TS.weighted(w).mean(("lon", "lat")).groupby("time.year").mean("time").compute()
    n_toa = mds.net_energy_flux_toa_into_atmosphere.weighted(w).mean(("lon", "lat")).groupby("time.year").mean("time").compute()
    restom_yearly = (
        (-mds.top_of_atmos_upward_shortwave_flux - mds.FLUT)
        .weighted(w).mean(("lon", "lat")).groupby("time.year").mean("time").compute()
    )
    restom_abs = solin_global + restom_yearly.values
    ts_vals = ts_global.values
    n_vals = n_toa.values
    if np.all(np.isfinite(ts_vals)) and np.all(np.isfinite(restom_abs)):
        slope, intercept = np.polyfit(ts_vals, restom_abs, 1)
    else:
        slope, intercept = np.nan, np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(ts_vals - 273.15, restom_abs, c=ts_global.year.values, cmap="viridis", s=10, alpha=0.6)
    if np.isfinite(slope):
        axes[0].plot(ts_vals - 273.15, np.polyval([slope, intercept], ts_vals), "r-", lw=2, label=f"λ = {slope:.2f} W/m²/K")
    axes[0].set_xlabel("Global Mean TS (°C)"); axes[0].set_ylabel("RESTOM (W/m²)")
    axes[0].set_title(f"Gregory Plot — Climate Feedback (post-spinup, N={n_years} years)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    plt.colorbar(sc, ax=axes[0], label="Year")

    if np.all(np.isfinite(ts_vals)) and np.all(np.isfinite(n_vals)):
        slope2, intercept2 = np.polyfit(ts_vals, n_vals, 1)
    else:
        slope2, intercept2 = np.nan, np.nan
    sc2 = axes[1].scatter(ts_vals - 273.15, n_vals, c=ts_global.year.values, cmap="viridis", s=10, alpha=0.6)
    if np.isfinite(slope2):
        axes[1].plot(ts_vals - 273.15, np.polyval([slope2, intercept2], ts_vals), "r-", lw=2, label=f"λ = {slope2:.2f} W/m²/K")
    axes[1].set_xlabel("Global Mean TS (°C)"); axes[1].set_ylabel("Net TOA flux (W/m²)")
    axes[1].set_title("Gregory Plot — Net TOA Flux"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.colorbar(sc2, ax=axes[1], label="Year")
    fig.tight_layout()
    save_fig(fig, "gregory_plot")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (label, lat_sl) in zip(axes.flat, {
        "Tropics (30°S–30°N)": slice(-30, 30), "NH Extratropics (30–90°N)": slice(30, 90),
        "SH Extratropics (90–30°S)": slice(-90, -30), "Arctic (60–90°N)": slice(60, 90),
    }.items()):
        ts_reg = mds.TS.sel(lat=lat_sl).weighted(cosine_weights(mds.TS.sel(lat=lat_sl).lat)).mean(("lon", "lat")).groupby("time.year").mean("time").compute()
        n_reg = mds.net_energy_flux_toa_into_atmosphere.sel(lat=lat_sl).weighted(cosine_weights(mds.TS.sel(lat=lat_sl).lat)).mean(("lon", "lat")).groupby("time.year").mean("time").compute()
        ax.scatter(ts_reg.values - 273.15, n_reg.values, s=5, alpha=0.3)
        if np.all(np.isfinite(ts_reg.values)) and np.all(np.isfinite(n_reg.values)):
            s, i = np.polyfit(ts_reg.values, n_reg.values, 1)
            ax.plot(ts_reg.values - 273.15, np.polyval([s, i], ts_reg.values), "r-", lw=2, label=f"λ = {s:.2f} W/m²/K")
        ax.set_title(label); ax.set_xlabel("TS (°C)"); ax.set_ylabel("Net TOA (W/m²)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("Regional Gregory Plots", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "gregory_regional")


# ===========================================================================
# 5. Spatial Correlation Length Scales
# ===========================================================================
print("  5. Spatial correlation length scales …")
if section_done("correlation_length"):
    skip_fig("correlation_length")
else:
    ts_yearly_map = mds.TS.groupby("time.year").mean("time").compute()
    ts_anom_map = ts_yearly_map - ts_yearly_map.mean("year")
    ts_anom_vals = ts_anom_map.values
    n_yrs = ts_anom_vals.shape[0]

    R = 6.371e6
    dlon_deg = np.abs(np.mean(np.diff(lon.values)))
    max_lag_pts = 60

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for ref_lat in [0, 20, 40, 60, -40, -60]:
        lat_idx = np.argmin(np.abs(lat.values - ref_lat))
        actual_lat = lat.values[lat_idx]
        km_per_pt = R * np.cos(np.deg2rad(actual_lat)) * dlon_deg * np.pi / 180 / 1000
        ts_row = ts_anom_vals[:, lat_idx, :]
        autocorr = []
        for lag in range(max_lag_pts):
            c = np.corrcoef(ts_row[:, :nlon - max_lag_pts].ravel(),
                             np.roll(ts_row, -lag, axis=1)[:, :nlon - max_lag_pts].ravel())[0, 1]
            autocorr.append(c)
        axes[0].plot(np.arange(max_lag_pts) * km_per_pt, autocorr, lw=1.5, label=f"{actual_lat:.0f}°")

    axes[0].axhline(1.0 / np.e, color="k", ls="--", lw=1, label="e-folding")
    axes[0].set_xlabel("Zonal distance (km)"); axes[0].set_ylabel("Correlation")
    axes[0].set_title("Zonal Decorrelation of TS Anomalies")
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=0.3); axes[0].set_xlim(0, 6000)

    efold_dist = np.zeros(nlat)
    for j in range(nlat):
        km_per_pt = R * np.cos(np.deg2rad(lat.values[j])) * dlon_deg * np.pi / 180 / 1000
        if km_per_pt < 1:
            efold_dist[j] = np.nan
            continue
        ts_row = ts_anom_vals[:, j, :]
        for lag in range(1, max_lag_pts):
            c = np.corrcoef(ts_row[:, :nlon - max_lag_pts].ravel(),
                             np.roll(ts_row, -lag, axis=1)[:, :nlon - max_lag_pts].ravel())[0, 1]
            if c < 1.0 / np.e:
                efold_dist[j] = lag * km_per_pt
                break
        else:
            efold_dist[j] = max_lag_pts * km_per_pt

    axes[1].plot(lat, efold_dist, "k-", lw=2)
    axes[1].set_xlabel("Latitude"); axes[1].set_ylabel("E-folding distance (km)")
    axes[1].set_title("Zonal E-Folding Decorrelation Distance of TS")
    axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 8000)
    fig.tight_layout()
    save_fig(fig, "correlation_length")


# ===========================================================================
# 6. Monsoon Diagnostics
# ===========================================================================
print("  6. Monsoon diagnostics …")
if section_done("monsoon_precip", "monsoon_winds", "monsoon_variability", "monsoon_enso_correlation"):
    for name in ["monsoon_precip", "monsoon_winds", "monsoon_variability", "monsoon_enso_correlation"]:
        skip_fig(name)
else:
    months = mds.time.dt.month

    precip_djf = mds.surface_precipitation_rate.sel(time=months.isin([12, 1, 2])).mean("time").compute() * 86400
    precip_jja = mds.surface_precipitation_rate.sel(time=months.isin([6, 7, 8])).mean("time").compute() * 86400
    monsoon_idx = precip_jja - precip_djf

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    im = axes[0].pcolormesh(lon, lat, precip_jja, cmap="Blues", shading="auto", vmin=0, vmax=15)
    axes[0].set_title("JJA Mean Precipitation (mm/day)"); axes[0].set_ylabel("Latitude")
    plt.colorbar(im, ax=axes[0], label="mm/day")
    im = axes[1].pcolormesh(lon, lat, precip_djf, cmap="Blues", shading="auto", vmin=0, vmax=15)
    axes[1].set_title("DJF Mean Precipitation (mm/day)"); axes[1].set_ylabel("Latitude")
    plt.colorbar(im, ax=axes[1], label="mm/day")
    im = axes[2].pcolormesh(lon, lat, monsoon_idx, cmap="BrBG", shading="auto", vmin=-10, vmax=10)
    axes[2].set_title("Monsoon Index: JJA − DJF Precipitation (mm/day)")
    axes[2].set_xlabel("Longitude"); axes[2].set_ylabel("Latitude")
    plt.colorbar(im, ax=axes[2], label="mm/day")
    fig.tight_layout()
    save_fig(fig, "monsoon_precip")

    taux_diff = mds.TAUX.sel(time=months.isin([6, 7, 8])).mean("time").compute() - mds.TAUX.sel(time=months.isin([12, 1, 2])).mean("time").compute()
    tauy_diff = mds.TAUY.sel(time=months.isin([6, 7, 8])).mean("time").compute() - mds.TAUY.sel(time=months.isin([12, 1, 2])).mean("time").compute()
    vmax_tau = float(np.percentile(np.abs(taux_diff.values), 95))
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    im = axes[0].pcolormesh(lon, lat, taux_diff, cmap="RdBu_r", shading="auto", vmin=-vmax_tau, vmax=vmax_tau)
    axes[0].set_title("Zonal Wind Stress Reversal: JJA − DJF TAUX (Pa)"); axes[0].set_ylabel("Latitude")
    plt.colorbar(im, ax=axes[0], label="Pa")
    im = axes[1].pcolormesh(lon, lat, tauy_diff, cmap="RdBu_r", shading="auto", vmin=-vmax_tau, vmax=vmax_tau)
    axes[1].set_title("Meridional Wind Stress Reversal: JJA − DJF TAUY (Pa)")
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
    plt.colorbar(im, ax=axes[1], label="Pa")
    fig.tight_layout()
    save_fig(fig, "monsoon_winds")

    monsoon_regions = {
        "South Asian\n(5–30°N, 60–100°E)": (slice(5, 30), slice(60, 100)),
        "West African\n(5–15°N, -20–30°E)": (slice(5, 15), slice(340, 360)),
        "East Asian\n(20–40°N, 100–140°E)": (slice(20, 40), slice(100, 140)),
        "N. American\n(15–35°N, 250–280°E)": (slice(15, 35), slice(250, 280)),
        "S. American\n(20°S–5°N, 290–330°E)": (slice(-20, 5), slice(290, 330)),
        "Australian\n(20°S–5°S, 110–150°E)": (slice(-20, -5), slice(110, 150)),
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    for ax, (label, (lat_sl, lon_sl)) in zip(axes.flat, monsoon_regions.items()):
        season_months = [12, 1, 2] if (lat_sl.start < 0 and lat_sl.stop <= 10) else [6, 7, 8]
        season_label = "DJF" if season_months[0] == 12 else "JJA"
        regional_precip = (
            mds.surface_precipitation_rate.sel(lat=lat_sl, lon=lon_sl)
            .sel(time=months.isin(season_months))
            .weighted(cosine_weights(mds.surface_precipitation_rate.sel(lat=lat_sl).lat))
            .mean(("lon", "lat")).groupby("time.year").mean("time").compute() * 86400
        )
        ax.plot(regional_precip.year, regional_precip.values, alpha=0.5, lw=0.8)
        ax.axhline(float(regional_precip.mean()), color="k", ls="--", lw=1)
        mean_val, std_val = float(regional_precip.mean()), float(regional_precip.std())
        ax.set_title(f"{label} ({season_label})\nμ={mean_val:.1f}, σ={std_val:.2f} mm/day, CV={std_val/mean_val*100:.1f}%", fontsize=10)
        ax.set_ylabel("Precip (mm/day)")
    axes[2, 0].set_xlabel("Year"); axes[2, 1].set_xlabel("Year")
    fig.suptitle(f"Monsoon Precipitation Interannual Variability (years {yr_start}\u2013{yr_end})", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "monsoon_variability")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (label, (lat_sl, lon_sl)) in zip(axes.flat, monsoon_regions.items()):
        season_months = [12, 1, 2] if (lat_sl.start < 0 and lat_sl.stop <= 10) else [6, 7, 8]
        regional_precip = (
            mds.surface_precipitation_rate.sel(lat=lat_sl, lon=lon_sl)
            .sel(time=months.isin(season_months))
            .weighted(cosine_weights(mds.surface_precipitation_rate.sel(lat=lat_sl).lat))
            .mean(("lon", "lat")).groupby("time.year").mean("time").compute() * 86400
        )
        common_years = np.intersect1d(regional_precip.year.values, nino34_annual.year.values)
        precip_vals = regional_precip.sel(year=common_years).values
        nino_vals = nino34_annual.sel(year=common_years).values
        r = np.corrcoef(nino_vals, precip_vals)[0, 1]
        ax.scatter(nino_vals, precip_vals, s=5, alpha=0.3)
        if abs(r) > 0.05:
            z = np.polyfit(nino_vals, precip_vals, 1)
            ax.plot(np.sort(nino_vals), np.polyval(z, np.sort(nino_vals)), "r-", lw=2)
        ax.set_title(f"{label}\nr = {r:.2f}", fontsize=10)
        ax.set_xlabel("Niño-3.4 (K)"); ax.set_ylabel("Precip (mm/day)")
    fig.suptitle("Monsoon–ENSO Relationship", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "monsoon_enso_correlation")


# ===========================================================================
# 7. Summary
# ===========================================================================
print("  7. Summary …")

fig, ax = plt.subplots(figsize=(12, 10))
ax.axis("off")

summary = f"""
ACE2-EAMv3 piControl — Cross-Variable Diagnostics Summary
{'='*60}

  ENSO
  El Niño years composited:   {n_composite}
  La Niña years composited:   {n_composite}
  Niño-3.4 σ:                 {float(nino34_annual.std()):.3f} K
"""

ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
        verticalalignment="top", fontfamily="monospace")
fig.tight_layout()
save_fig(fig, "summary")


# ---- Done ----------------------------------------------------------------
print(f"\nDone — saved {_fig_counter} PNGs to {OUTPUT_DIR}/")
