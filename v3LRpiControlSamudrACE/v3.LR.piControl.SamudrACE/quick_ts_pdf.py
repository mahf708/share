"""Progressive PDF of TS — segment-by-segment, with zoomed views."""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, time, os

seg_dirs = sorted(glob.glob("/lcrc/globalscratch/ac.ngmahfouz/picontrol_run/seg_*/atmosphere"))
outdir = "figs_progressive_pdfs"
os.makedirs(outdir, exist_ok=True)

# ── Precompute lat weights ────────────────────────────────────────────────
ds0 = xr.open_dataset(f"{seg_dirs[0]}/6h_surface_TS_predictions.nc", decode_timedelta=False)
lat = ds0["lat"].values
cos_lat = np.cos(np.deg2rad(lat))
weights = cos_lat / cos_lat.sum()  # (180,)
ds0.close()

# ── Collect per-segment data ──────────────────────────────────────────────
bins_global = np.linspace(190, 330, 501)
bins_arctic = np.linspace(220, 290, 301)
bins_gmean = np.linspace(285, 292, 301)

centres_g = 0.5 * (bins_global[:-1] + bins_global[1:])
centres_a = 0.5 * (bins_arctic[:-1] + bins_arctic[1:])
centres_m = 0.5 * (bins_gmean[:-1] + bins_gmean[1:])
widths_g = np.diff(bins_global)
widths_a = np.diff(bins_arctic)
widths_m = np.diff(bins_gmean)

arctic_mask = lat > 60  # Arctic: lat > 60N

hist_global = []
hist_arctic = []
hist_gmean = []
labels = []

cmap = plt.get_cmap("coolwarm", len(seg_dirs))

for i, sd in enumerate(seg_dirs):
    t0 = time.time()
    yr_start = 401 + i * 80
    f = f"{sd}/6h_surface_TS_predictions.nc"
    ds = xr.open_dataset(f, decode_timedelta=False)
    data = ds["TS"].values[0]  # (time, lat, lon)
    ds.close()

    # 1. Global PDF (all grid cells)
    flat = data.ravel()
    flat = flat[np.isfinite(flat)]
    h, _ = np.histogram(flat, bins=bins_global)
    hist_global.append(h)

    # 2. Arctic-only PDF
    arctic_data = data[:, arctic_mask, :].ravel()
    arctic_data = arctic_data[np.isfinite(arctic_data)]
    h, _ = np.histogram(arctic_data, bins=bins_arctic)
    hist_arctic.append(h)

    # 3. Global-mean TS per timestep (area-weighted)
    gmean = np.nansum(data * weights[None, :, None], axis=(1, 2)) / np.nansum(
        np.isfinite(data) * weights[None, :, None], axis=(1, 2)
    )
    # More precise: proper weighted mean
    w3d = np.broadcast_to(weights[None, :, None], data.shape)
    mask = np.isfinite(data)
    gmean = np.where(mask, data * w3d, 0).sum(axis=(1, 2)) / np.where(mask, w3d, 0).sum(axis=(1, 2))
    h, _ = np.histogram(gmean[np.isfinite(gmean)], bins=bins_gmean)
    hist_gmean.append(h)

    labels.append(f"yr {yr_start}–{yr_start+80}")
    print(f"  seg_{i:04d} ({labels[-1]}): {time.time()-t0:.0f}s", flush=True)

# ── Plot 1: Global PDF (full range) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for i, h in enumerate(hist_global):
    density = h / (h.sum() * widths_g)
    ax.plot(centres_g, density, color=cmap(i), lw=1.2, label=labels[i])
ax.set_xlabel("TS (K)", fontsize=12)
ax.set_ylabel("Probability density", fontsize=12)
ax.set_title("Progressive PDF of Surface Temperature — All Grid Cells", fontsize=13)
ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{outdir}/progressive_pdf_TS_global.png", dpi=150)
plt.close(fig)
print(f"  → {outdir}/progressive_pdf_TS_global.png")

# ── Plot 2: Global PDF difference vs first segment ───────────────────────
ref_density = hist_global[0] / (hist_global[0].sum() * widths_g)
fig, ax = plt.subplots(figsize=(12, 5))
for i, h in enumerate(hist_global[1:], 1):
    density = h / (h.sum() * widths_g)
    ax.plot(centres_g, density - ref_density, color=cmap(i), lw=1.0,
            label=labels[i])
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("TS (K)", fontsize=12)
ax.set_ylabel("Density difference", fontsize=12)
ax.set_title("PDF Difference vs. First Segment (yr 401–481)", fontsize=13)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{outdir}/progressive_pdf_TS_diff.png", dpi=150)
plt.close(fig)
print(f"  → {outdir}/progressive_pdf_TS_diff.png")

# ── Plot 3: Arctic-only PDF ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for i, h in enumerate(hist_arctic):
    density = h / (h.sum() * widths_a)
    ax.plot(centres_a, density, color=cmap(i), lw=1.2, label=labels[i])
ax.set_xlabel("TS (K)", fontsize=12)
ax.set_ylabel("Probability density", fontsize=12)
ax.set_title("Progressive PDF of Surface Temperature — Arctic (>60°N)", fontsize=13)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{outdir}/progressive_pdf_TS_arctic.png", dpi=150)
plt.close(fig)
print(f"  → {outdir}/progressive_pdf_TS_arctic.png")

# ── Plot 4: Global-mean TS PDF ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for i, h in enumerate(hist_gmean):
    density = h / (h.sum() * widths_m)
    ax.plot(centres_m, density, color=cmap(i), lw=1.5, label=labels[i])
ax.set_xlabel("Global-mean TS (K)", fontsize=12)
ax.set_ylabel("Probability density", fontsize=12)
ax.set_title("Progressive PDF of Global-Mean Surface Temperature", fontsize=13)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{outdir}/progressive_pdf_TS_gmean.png", dpi=150)
plt.close(fig)
print(f"  → {outdir}/progressive_pdf_TS_gmean.png")

print(f"\nDone — 4 figures in {outdir}/")
