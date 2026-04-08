"""
Standalone: Global-mean surface temperature climate drift.

Reads monthly atmosphere predictions, computes area-weighted global-mean TS,
and plots the full timeseries with linear trends + detrended anomaly.

Output: figs_standalone/01_climate_drift.png
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

DATA_ROOT = Path("/lcrc/globalscratch/ac.ngmahfouz/picontrol_run")
START_YEAR = 401

t0 = time.time()

# ── Load monthly data ─────────────────────────────────────────────────────
files = sorted(glob.glob(str(DATA_ROOT / "seg_*/atmosphere/monthly_mean_predictions.nc")))
print(f"Loading {len(files)} segments …")

ds = xr.open_mfdataset(files, combine="nested", concat_dim="time",
                        chunks={}, drop_variables=["init_time"]).isel(sample=0)
ds["time"] = ds.valid_time
ds = ds.drop_vars("valid_time")

lat = ds.lat.values
cos_w = np.cos(np.deg2rad(lat))

# ── Compute yearly global-mean TS ─────────────────────────────────────────
ts_weighted = ds.TS.weighted(xr.DataArray(cos_w, dims=["lat"]))
ts_global = ts_weighted.mean(("lon", "lat")).groupby("time.year").mean("time").compute()

years = ts_global.year.values
ts_vals = ts_global.values
n_years = len(years)

print(f"  {n_years} years: {years[0]}–{years[-1]}  ({time.time()-t0:.0f}s)")

# ── Trends ────────────────────────────────────────────────────────────────
trend_full = np.polyfit(np.arange(n_years), ts_vals, 1)
trend_full_century = trend_full[0] * 100

detrended = ts_vals - np.polyval(trend_full, np.arange(n_years))

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(years, ts_vals, alpha=0.6, lw=0.8, color="steelblue")
axes[0].plot(years, np.polyval(trend_full, np.arange(n_years)), "r--", lw=2,
             label=f"Trend: {trend_full_century:+.3f} K/century")
axes[0].set_ylabel("TS (K)", fontsize=12)
axes[0].set_title(f"Global-Mean Surface Temperature — Full Simulation ({n_years} years)",
                  fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(years, detrended, alpha=0.6, lw=0.8, color="steelblue")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_xlabel("Model Year", fontsize=12)
axes[1].set_ylabel("TS Anomaly (K)", fontsize=12)
axes[1].set_title("Detrended Global-Mean TS", fontsize=13)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
outpath = OUTDIR / "01_climate_drift.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  → {outpath}  ({time.time()-t0:.0f}s)")
