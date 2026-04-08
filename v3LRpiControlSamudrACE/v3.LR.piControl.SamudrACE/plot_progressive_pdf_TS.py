"""
Standalone: Progressive PDF of surface temperature (TS).

Memory-safe: loads one year at a time (~1.5 GB) instead of a full
30 GB segment. Produces 4 figures:
  - Global PDF (all grid cells)
  - PDF difference vs first segment
  - Arctic-only (>60N)
  - Global-mean TS (area-weighted scalar per timestep)

Output: figs_standalone/00_progressive_pdf_TS_{global,diff,arctic,gmean}.png
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

seg_dirs = sorted(glob.glob(
    "/lcrc/globalscratch/ac.ngmahfouz/picontrol_run/seg_*/atmosphere"))
START_YEAR = 401
STEPS_PER_YEAR = 1460  # 365 * 4

t_start = time.time()


def _ts(msg):
    elapsed = time.time() - t_start
    m, s = int(elapsed // 60), int(elapsed % 60)
    print(f"  [{m:02d}:{s:02d}] {msg}", flush=True)


# ── Grid info ─────────────────────────────────────────────────────────────
ds0 = xr.open_dataset(f"{seg_dirs[0]}/6h_surface_TS_predictions.nc",
                       decode_timedelta=False)
lat = ds0["lat"].values
ds0.close()

cos_lat = np.cos(np.deg2rad(lat))
weights = cos_lat / cos_lat.sum()
arctic_mask = lat > 60

# ── Bin definitions ───────────────────────────────────────────────────────
bins_g = np.linspace(190, 330, 501)
bins_a = np.linspace(220, 290, 301)
bins_m = np.linspace(285, 292, 301)
centres_g = 0.5 * (bins_g[:-1] + bins_g[1:])
centres_a = 0.5 * (bins_a[:-1] + bins_a[1:])
centres_m = 0.5 * (bins_m[:-1] + bins_m[1:])
widths_g = np.diff(bins_g)
widths_a = np.diff(bins_a)
widths_m = np.diff(bins_m)

# ── Accumulate histograms per segment (year by year to save memory) ───────
CACHE = Path("cache_progressive_pdf_TS.npz")

hist_global, hist_arctic, hist_gmean, labels = [], [], [], []
cmap = plt.get_cmap("coolwarm", len(seg_dirs))

if CACHE.exists():
    _ts(f"Loading cached histograms from {CACHE}")
    c = np.load(CACHE, allow_pickle=True)
    hist_global = list(c["hist_global"])
    hist_arctic = list(c["hist_arctic"])
    hist_gmean = list(c["hist_gmean"])
    labels = list(c["labels"])
    _ts(f"  {len(labels)} segments loaded from cache")
else:
    print(f"Processing {len(seg_dirs)} segments …")
    for i, sd in enumerate(seg_dirs):
        t0 = time.time()
        yr = START_YEAR + i * 80

        ds = xr.open_dataset(f"{sd}/6h_surface_TS_predictions.nc",
                             decode_timedelta=False)
        ntime = ds.dims["time"]
        n_years = ntime // STEPS_PER_YEAR

        # Running histograms for this segment
        h_g = np.zeros(len(bins_g) - 1, dtype=np.int64)
        h_a = np.zeros(len(bins_a) - 1, dtype=np.int64)
        gmean_all = []

        for y in range(n_years):
            t_s = y * STEPS_PER_YEAR
            t_e = t_s + STEPS_PER_YEAR
            data = ds["TS"].values[0, t_s:t_e]  # (1460, lat, lon) ~1.5 GB

            # Global histogram
            flat = data.ravel()
            flat = flat[np.isfinite(flat)]
            counts, _ = np.histogram(flat, bins=bins_g)
            h_g += counts

            # Arctic histogram
            ad = data[:, arctic_mask, :].ravel()
            ad = ad[np.isfinite(ad)]
            counts, _ = np.histogram(ad, bins=bins_a)
            h_a += counts

            # Global mean per timestep (weighted)
            gm = np.nansum(data * weights[None, :, None], axis=(1, 2)) / weights.sum()
            gmean_all.append(gm)

        ds.close()

        if h_g.sum() == 0:
            _ts(f"seg_{i:04d}: no full years, skipping")
            continue

        hist_global.append(h_g)
        hist_arctic.append(h_a)

        if gmean_all:
            gmean_cat = np.concatenate(gmean_all)
            h_m, _ = np.histogram(gmean_cat[np.isfinite(gmean_cat)], bins=bins_m)
        else:
            h_m = np.zeros(len(bins_m) - 1, dtype=np.int64)
        hist_gmean.append(h_m)

        labels.append(f"yr {yr}–{yr+80}")
        _ts(f"seg_{i:04d} ({labels[-1]}): {time.time()-t0:.0f}s")

    # Save cache
    np.savez(CACHE,
             hist_global=np.array(hist_global),
             hist_arctic=np.array(hist_arctic),
             hist_gmean=np.array(hist_gmean),
             labels=np.array(labels))
    _ts(f"Cached to {CACHE}")

# ── Plot helper ───────────────────────────────────────────────────────────


def save(fig, name, dpi=150):
    p = OUTDIR / f"00_progressive_pdf_TS_{name}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _ts(f"→ {p}")


n_segs = len(hist_global)

# ── 1. Ridgeline plot — Global PDF ───────────────────────────────────────
_ts("Plotting ridgeline (global) …")
fig, ax = plt.subplots(figsize=(12, 10))
offset_scale = 0.008  # vertical offset between curves
for i, h in enumerate(hist_global):
    density = h / (h.sum() * widths_g)
    offset = (n_segs - 1 - i) * offset_scale
    ax.fill_between(centres_g, offset, density + offset,
                    color=cmap(i), alpha=0.35, lw=0)
    ax.plot(centres_g, density + offset, color=cmap(i), lw=0.8)
    ax.text(centres_g[-1] + 1, offset + 0.0005, labels[i],
            fontsize=7, va="bottom", color=cmap(i))
ax.set_xlabel("TS (K)", fontsize=12)
ax.set_ylabel("Probability density (offset)", fontsize=12)
ax.set_title("Progressive PDF of Surface Temperature — Ridgeline", fontsize=13)
ax.set_yticks([])
ax.grid(True, alpha=0.2, axis="x")
fig.tight_layout()
save(fig, "global_ridgeline")

# ── 2. Small multiples — Global PDF with reference ───────────────────────
_ts("Plotting small multiples (global) …")
ncols = 4
nrows = (n_segs + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), sharex=True,
                          sharey=True)
axes_flat = axes.ravel()
ref_density = hist_global[0] / (hist_global[0].sum() * widths_g)

for i in range(len(axes_flat)):
    ax = axes_flat[i]
    if i < n_segs:
        density = hist_global[i] / (hist_global[i].sum() * widths_g)
        ax.fill_between(centres_g, ref_density, alpha=0.15, color="gray",
                        label="First segment" if i == 0 else None)
        ax.plot(centres_g, ref_density, color="gray", lw=0.5, alpha=0.5)
        ax.plot(centres_g, density, color=cmap(i), lw=1.2)
        ax.set_title(labels[i], fontsize=9, color=cmap(i))
        ax.grid(True, alpha=0.2)
    else:
        ax.set_visible(False)

fig.suptitle("Progressive PDF of TS — Each Segment vs. First (gray)",
             fontsize=13, y=1.01)
fig.supxlabel("TS (K)", fontsize=11)
fig.supylabel("Probability density", fontsize=11)
fig.tight_layout()
save(fig, "global_small_multiples")

# ── 3. PDF difference (kept — this is the most quantitative view) ────────
_ts("Plotting PDF difference …")
fig, ax = plt.subplots(figsize=(12, 5))
for i, h in enumerate(hist_global[1:], 1):
    density = h / (h.sum() * widths_g)
    ax.plot(centres_g, density - ref_density, color=cmap(i), lw=1.0,
            label=labels[i])
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set(xlabel="TS (K)", ylabel="Density difference",
       title=f"PDF Difference vs. First Segment ({labels[0]})")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "diff")

# ── 4. Ridgeline — Arctic ────────────────────────────────────────────────
_ts("Plotting ridgeline (Arctic) …")
fig, ax = plt.subplots(figsize=(12, 10))
offset_scale_a = 0.012
for i, h in enumerate(hist_arctic):
    density = h / (h.sum() * widths_a)
    offset = (n_segs - 1 - i) * offset_scale_a
    ax.fill_between(centres_a, offset, density + offset,
                    color=cmap(i), alpha=0.35, lw=0)
    ax.plot(centres_a, density + offset, color=cmap(i), lw=0.8)
    ax.text(centres_a[-1] + 0.5, offset + 0.0005, labels[i],
            fontsize=7, va="bottom", color=cmap(i))
ax.set_xlabel("TS (K)", fontsize=12)
ax.set_ylabel("Probability density (offset)", fontsize=12)
ax.set_title("Progressive PDF of TS — Arctic (>60°N) — Ridgeline", fontsize=13)
ax.set_yticks([])
ax.grid(True, alpha=0.2, axis="x")
fig.tight_layout()
save(fig, "arctic_ridgeline")

# ── 5. Ridgeline — Global-mean TS ────────────────────────────────────────
_ts("Plotting ridgeline (global-mean) …")
fig, ax = plt.subplots(figsize=(10, 10))
offset_scale_m = 0.4
for i, h in enumerate(hist_gmean):
    if h.sum() == 0:
        continue
    density = h / (h.sum() * widths_m)
    offset = (n_segs - 1 - i) * offset_scale_m
    ax.fill_between(centres_m, offset, density + offset,
                    color=cmap(i), alpha=0.35, lw=0)
    ax.plot(centres_m, density + offset, color=cmap(i), lw=0.8)
    ax.text(centres_m[-1] + 0.05, offset + 0.02, labels[i],
            fontsize=7, va="bottom", color=cmap(i))
ax.set_xlabel("Global-mean TS (K)", fontsize=12)
ax.set_ylabel("Probability density (offset)", fontsize=12)
ax.set_title("Progressive PDF of Global-Mean TS — Ridgeline", fontsize=13)
ax.set_yticks([])
ax.grid(True, alpha=0.2, axis="x")
fig.tight_layout()
save(fig, "gmean_ridgeline")

print(f"\nDone — 4 figures in {OUTDIR}/  ({time.time()-t_start:.0f}s)")
