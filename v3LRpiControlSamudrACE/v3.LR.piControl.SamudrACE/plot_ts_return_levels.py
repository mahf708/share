"""
Standalone: Return level analysis for temperature and precipitation extremes.

Computes annual block maxima/minima from 6-hourly data:
  - TS max/min (global weighted mean)
  - RX1day: max 1-day precipitation (global, tropics, midlat)
  - RX5day: max 5-day precipitation (global, tropics, midlat)

Plots empirical return levels with GEV fits.
Caches annual extremes to avoid reprocessing on rerun.

Output:
    figs_standalone/05a_ts_return_levels.png
    figs_standalone/05b_precip_return_levels_global.png
    figs_standalone/05c_precip_return_levels_regional.png
"""
import glob
import time
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("figs_standalone")
OUTDIR.mkdir(exist_ok=True)
CACHE = Path("cache_return_levels.npz")

DATA_ROOT = Path("/lcrc/globalscratch/ac.ngmahfouz/picontrol_run")
START_YEAR = 401
STEPS_PER_YEAR = 1460   # 365 days × 4 (6-hourly)
STEPS_PER_DAY = 4
KG_M2S_TO_MM_DAY = 86400.0

t_start = time.time()


def _ts(msg):
    elapsed = time.time() - t_start
    m, s = int(elapsed // 60), int(elapsed % 60)
    print(f"  [{m:02d}:{s:02d}] {msg}", flush=True)


# ── Grid ──────────────────────────────────────────────────────────────────
seg_dirs = sorted(glob.glob(str(DATA_ROOT / "seg_*/atmosphere")))
ds0 = xr.open_dataset(f"{seg_dirs[0]}/6h_surface_TS_predictions.nc",
                       decode_timedelta=False)
lat = ds0["lat"].values
lon = ds0["lon"].values
nlat, nlon = len(lat), len(lon)
ds0.close()

cos_w = np.cos(np.deg2rad(lat))
cos_w_norm = cos_w / cos_w.sum()

# Region masks
tropics = (lat >= -30) & (lat <= 30)
midlat_nh = (lat >= 30) & (lat <= 60)
midlat_sh = (lat >= -60) & (lat <= -30)
midlat = midlat_nh | midlat_sh

# ── Compute or load annual extremes ───────────────────────────────────────
if CACHE.exists():
    _ts(f"Loading cached extremes from {CACHE}")
    c = np.load(CACHE)
    ts_global_max = c["ts_global_max"]
    ts_global_min = c["ts_global_min"]
    pr_rx1day_global = c["pr_rx1day_global"]
    pr_rx5day_global = c["pr_rx5day_global"]
    pr_rx1day_tropics = c["pr_rx1day_tropics"]
    pr_rx5day_tropics = c["pr_rx5day_tropics"]
    pr_rx1day_midlat = c["pr_rx1day_midlat"]
    pr_rx5day_midlat = c["pr_rx5day_midlat"]
    yr_labels = c["yr_labels"]
else:
    _ts(f"Computing annual extremes from {len(seg_dirs)} segments …")
    ts_global_max, ts_global_min = [], []
    pr_rx1day_global, pr_rx5day_global = [], []
    pr_rx1day_tropics, pr_rx5day_tropics = [], []
    pr_rx1day_midlat, pr_rx5day_midlat = [], []
    yr_labels = []

    def _regional_rx(daily, rx5, mask):
        """Area-weighted max of daily and 5-day precip over a region."""
        w_r = cos_w[mask]
        w_r = w_r / w_r.sum()
        rx1_map = np.nanmax(daily[:, mask, :], axis=0)
        rx1 = np.average(rx1_map, weights=np.broadcast_to(
            w_r[:, None], rx1_map.shape))
        rx5_map = np.nanmax(rx5[:, mask, :], axis=0)
        rx5_val = np.average(rx5_map, weights=np.broadcast_to(
            w_r[:, None], rx5_map.shape))
        return rx1, rx5_val

    for si, sd in enumerate(seg_dirs):
        t0 = time.time()
        yr_start = START_YEAR + si * 80

        # Open lazily — don't load into memory
        ds_ts = xr.open_dataset(f"{sd}/6h_surface_TS_predictions.nc",
                                decode_timedelta=False)
        ds_pr = xr.open_dataset(
            f"{sd}/6h_surface_surface_precipitation_rate_predictions.nc",
            decode_timedelta=False)

        ntime = ds_ts.dims["time"]
        n_years = ntime // STEPS_PER_YEAR

        for y in range(n_years):
            t_s = y * STEPS_PER_YEAR
            t_e = t_s + STEPS_PER_YEAR

            # Load one year at a time (~ 1.5 GB each, not 30 GB)
            ts_yr = ds_ts["TS"].values[0, t_s:t_e]    # (1460, lat, lon)
            pr_yr = ds_pr["surface_precipitation_rate"].values[0, t_s:t_e]

            # ── TS extremes ──
            ann_max = np.nanmax(ts_yr, axis=0)
            ann_min = np.nanmin(ts_yr, axis=0)
            w2d = np.broadcast_to(cos_w_norm[:, None], ann_max.shape)
            ts_global_max.append(np.average(ann_max, weights=w2d))
            ts_global_min.append(np.average(ann_min, weights=w2d))

            # ── Precip extremes ──
            n_days = ts_yr.shape[0] // STEPS_PER_DAY
            pr_daily = (pr_yr[:n_days * STEPS_PER_DAY]
                        .reshape(n_days, STEPS_PER_DAY, nlat, nlon)
                        .mean(axis=1) * KG_M2S_TO_MM_DAY)

            # RX5day: 5-day running sum
            pr_5day = np.apply_along_axis(
                lambda x: np.convolve(x, np.ones(5), mode="valid"), 0, pr_daily)

            rx1_g, rx5_g = _regional_rx(pr_daily, pr_5day,
                                         np.ones(nlat, dtype=bool))
            rx1_t, rx5_t = _regional_rx(pr_daily, pr_5day, tropics)
            rx1_m, rx5_m = _regional_rx(pr_daily, pr_5day, midlat)

            pr_rx1day_global.append(rx1_g)
            pr_rx5day_global.append(rx5_g)
            pr_rx1day_tropics.append(rx1_t)
            pr_rx5day_tropics.append(rx5_t)
            pr_rx1day_midlat.append(rx1_m)
            pr_rx5day_midlat.append(rx5_m)

            yr_labels.append(yr_start + y)

        ds_ts.close()
        ds_pr.close()
        _ts(f"seg_{si:04d} (yr {yr_start}–{yr_start+n_years}): "
            f"{n_years} yr, {time.time()-t0:.0f}s")

    # Convert to arrays and cache
    arrs = {}
    for name in ["ts_global_max", "ts_global_min",
                  "pr_rx1day_global", "pr_rx5day_global",
                  "pr_rx1day_tropics", "pr_rx5day_tropics",
                  "pr_rx1day_midlat", "pr_rx5day_midlat",
                  "yr_labels"]:
        arrs[name] = np.array(locals()[name])
        locals()[name] = arrs[name]

    ts_global_max = arrs["ts_global_max"]
    ts_global_min = arrs["ts_global_min"]
    pr_rx1day_global = arrs["pr_rx1day_global"]
    pr_rx5day_global = arrs["pr_rx5day_global"]
    pr_rx1day_tropics = arrs["pr_rx1day_tropics"]
    pr_rx5day_tropics = arrs["pr_rx5day_tropics"]
    pr_rx1day_midlat = arrs["pr_rx1day_midlat"]
    pr_rx5day_midlat = arrs["pr_rx5day_midlat"]
    yr_labels = arrs["yr_labels"]

    np.savez(CACHE, **arrs)
    _ts(f"Cached to {CACHE}")

n = len(ts_global_max)
_ts(f"{n} years of annual extremes")

# ── GEV fit ───────────────────────────────────────────────────────────────
try:
    from scipy.stats import genextreme
    HAS_GEV = True
except ImportError:
    HAS_GEV = False


def plot_return_level(ax, vals, title, color, is_max=True):
    """Plot empirical return levels + GEV fit on a semilog axis."""
    n_pts = len(vals)
    sorted_v = np.sort(vals)[::-1] if is_max else np.sort(vals)
    rp = (n_pts + 1) / np.arange(1, n_pts + 1)

    ax.semilogx(rp, sorted_v, "x", color=color, ms=4, alpha=0.5,
                label="Empirical")

    if HAS_GEV:
        c_gev, loc, scale = genextreme.fit(vals)
        rp_fit = np.logspace(np.log10(1.01), np.log10(n_pts * 2), 200)
        if is_max:
            rl_fit = genextreme.isf(1.0 / rp_fit, c_gev, loc=loc, scale=scale)
        else:
            rl_fit = genextreme.ppf(1.0 / rp_fit, c_gev, loc=loc, scale=scale)
        ax.semilogx(rp_fit, rl_fit, "-", color=color, lw=2, alpha=0.8,
                    label=f"GEV (ξ={c_gev:.3f})")

    ax.set_xlabel("Return Period (years)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ── Figure 1: Temperature return levels ──────────────────────────────────
_ts("Plotting temperature return levels …")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_return_level(axes[0], ts_global_max - 273.15,
                  f"Annual Max TS ({n} yr)", "firebrick", is_max=True)
axes[0].set_ylabel("TS (°C)", fontsize=11)

plot_return_level(axes[1], ts_global_min - 273.15,
                  f"Annual Min TS ({n} yr)", "steelblue", is_max=False)
axes[1].set_ylabel("TS (°C)", fontsize=11)

fig.suptitle("Temperature Extremes — Return Levels (Global Weighted Mean)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "05a_ts_return_levels.png", dpi=200, bbox_inches="tight")
plt.close(fig)
_ts(f"  → 05a_ts_return_levels.png")

# ── Figure 2: Precipitation return levels (global) ───────────────────────
_ts("Plotting precipitation return levels (global) …")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_return_level(axes[0], pr_rx1day_global,
                  f"RX1day — Global ({n} yr)", "darkorange", is_max=True)
axes[0].set_ylabel("Precipitation (mm/day)", fontsize=11)

plot_return_level(axes[1], pr_rx5day_global,
                  f"RX5day — Global ({n} yr)", "darkgreen", is_max=True)
axes[1].set_ylabel("Precipitation (mm/5day)", fontsize=11)

fig.suptitle("Precipitation Extremes — Return Levels (Global Weighted Mean)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "05b_precip_return_levels_global.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)
_ts(f"  → 05b_precip_return_levels_global.png")

# ── Figure 3: Precipitation return levels (regional) ─────────────────────
_ts("Plotting precipitation return levels (regional) …")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

plot_return_level(axes[0, 0], pr_rx1day_tropics,
                  f"RX1day — Tropics 30°S–30°N ({n} yr)", "darkorange")
axes[0, 0].set_ylabel("Precip (mm/day)", fontsize=10)

plot_return_level(axes[0, 1], pr_rx5day_tropics,
                  f"RX5day — Tropics 30°S–30°N ({n} yr)", "darkgreen")
axes[0, 1].set_ylabel("Precip (mm/5day)", fontsize=10)

plot_return_level(axes[1, 0], pr_rx1day_midlat,
                  f"RX1day — Midlatitudes 30°–60° ({n} yr)", "darkorange")
axes[1, 0].set_ylabel("Precip (mm/day)", fontsize=10)

plot_return_level(axes[1, 1], pr_rx5day_midlat,
                  f"RX5day — Midlatitudes 30°–60° ({n} yr)", "darkgreen")
axes[1, 1].set_ylabel("Precip (mm/5day)", fontsize=10)

fig.suptitle("Precipitation Extremes — Return Levels by Region",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUTDIR / "05c_precip_return_levels_regional.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)
_ts(f"  → 05c_precip_return_levels_regional.png")

print(f"\nDone ({time.time()-t_start:.0f}s)")
