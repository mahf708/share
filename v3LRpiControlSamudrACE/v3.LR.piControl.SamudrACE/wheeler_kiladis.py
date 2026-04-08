"""
Wheeler–Kiladis space-time spectral analysis of tropical precipitation.

Computes the zonal wavenumber–frequency power spectrum of 6-hourly
precipitation averaged over 15°S–15°N, following Wheeler and Kiladis (1999).

Steps:
  1. Extract equatorial belt (15°S–15°N), cos(lat)-weight, compute
     symmetric and antisymmetric components about the equator.
  2. Taper and segment into 96-day windows with 60-day overlap.
  3. 2-D FFT (longitude → wavenumber, time → frequency) per segment.
  4. Average power over all segments from all requested data segments.
  5. Compute a smoothed red-noise background (1-2-1 filter, 40 passes).
  6. Plot raw power and signal/background ratio with theoretical
     dispersion curves for Kelvin, equatorial Rossby, and MJO.

Usage:
    python wheeler_kiladis.py                       # use 1 segment (fast test)
    python wheeler_kiladis.py --n_segments 3        # use 3 data segments
    python wheeler_kiladis.py --n_segments all       # use all segments
    python wheeler_kiladis.py --overwrite           # recompute cache

Output:
    figs_wheeler_kiladis/wheeler_kiladis_symmetric.png
    figs_wheeler_kiladis/wheeler_kiladis_antisymmetric.png
"""

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ── Configuration ──────────────────────────────────────────────────────────

OUTDIR = Path("figs_wheeler_kiladis")
DATA_PATTERN = "/lcrc/globalscratch/ac.ngmahfouz/picontrol_run/seg_*/atmosphere"

# Spectral parameters (following Wheeler & Kiladis 1999)
SEGMENT_DAYS = 96        # window length in days
OVERLAP_DAYS = 60        # overlap between windows
SAMPLES_PER_DAY = 4      # 6-hourly data
LAT_BOUND = 15.0         # equatorial belt half-width

# Dispersion curve parameters
G = 9.81                 # m/s²
BETA = 2.28e-11          # df/dy at equator (1/m/s)
R_EARTH_M = 6.371e6      # m
EQUIV_DEPTHS = [12, 25, 50]  # meters, for dispersion curves


# ── Helpers ────────────────────────────────────────────────────────────────

def _ts(msg):
    elapsed = time.time() - _ts.t0
    m, s = int(elapsed // 60), int(elapsed % 60)
    print(f"  [{m:02d}:{s:02d}] {msg}", flush=True)

_ts.t0 = time.time()


def load_tropical_precip(seg_dir):
    """Load 6h precip for 15°S–15°N, return (time, lat, lon) in mm/day."""
    f = os.path.join(seg_dir, "6h_surface_surface_precipitation_rate_predictions.nc")
    ds = xr.open_dataset(f, decode_timedelta=False)
    lat = ds["lat"].values
    mask = (lat >= -LAT_BOUND) & (lat <= LAT_BOUND)
    lat_sel = lat[mask]
    # (time, lat_tropical, lon), convert kg/m²/s → mm/day
    data = ds["surface_precipitation_rate"].values[0][:, mask, :] * 86400.0
    ds.close()
    return data, lat_sel


def decompose_symmetric(data, lat):
    """Split into symmetric and antisymmetric components about equator.

    Returns sym, asym each shaped (time, n_lat_half, lon).
    Symmetric  = (f(φ) + f(-φ)) / 2
    Antisymmetric = (f(φ) - f(-φ)) / 2
    """
    nh_mask = lat >= 0
    sh_mask = lat < 0
    nh = data[:, nh_mask, :]             # (time, n_nh, lon)
    sh = data[:, sh_mask, :][:, ::-1, :] # flip SH so lats align with NH

    # Ensure same number of lats
    n = min(nh.shape[1], sh.shape[1])
    nh = nh[:, :n, :]
    sh = sh[:, :n, :]

    sym = (nh + sh) / 2.0
    asym = (nh - sh) / 2.0
    return sym, asym


def compute_wk_spectrum(data, lat):
    """Compute wavenumber-frequency power spectrum.

    Args:
        data: (time, lat, lon) array
        lat: latitude values for weighting

    Returns:
        power: (n_freq, n_wavenum) averaged power spectrum
        freqs: frequency axis in cycles/day
        wavenums: wavenumber axis
    """
    ntime, nlat, nlon = data.shape
    seg_len = SEGMENT_DAYS * SAMPLES_PER_DAY
    overlap = OVERLAP_DAYS * SAMPLES_PER_DAY
    step = seg_len - overlap

    # Cosine-latitude weights
    cos_w = np.cos(np.deg2rad(lat))
    cos_w = cos_w / cos_w.sum()

    # Hann window for tapering
    hann_t = np.hanning(seg_len)

    # Accumulate power
    power_sum = None
    n_windows = 0

    for t0 in range(0, ntime - seg_len + 1, step):
        chunk = data[t0:t0 + seg_len, :, :]  # (seg_len, nlat, nlon)

        # Latitude-weighted average (reduces to (seg_len, nlon))
        weighted = np.sum(chunk * cos_w[None, :, None], axis=1)

        # Remove time mean
        weighted = weighted - weighted.mean(axis=0, keepdims=True)

        # Apply Hann taper in time
        weighted = weighted * hann_t[:, None]

        # 2D FFT: axis 0 = time → frequency, axis 1 = lon → wavenumber
        fft2d = np.fft.fft2(weighted) / (seg_len * nlon)
        power = np.abs(fft2d) ** 2

        if power_sum is None:
            power_sum = power
        else:
            power_sum += power
        n_windows += 1

    power_avg = power_sum / n_windows

    # Frequency axis (cycles per day)
    freqs = np.fft.fftfreq(seg_len, d=1.0 / SAMPLES_PER_DAY)
    # Wavenumber axis — negate so positive k = eastward propagation
    # (np.fft.fft uses exp(-ikx), but with lon increasing eastward,
    # positive k from FFT is westward; WK convention is opposite)
    wavenums = -np.fft.fftfreq(nlon, d=1.0 / nlon)

    return power_avg, freqs, wavenums, n_windows


def smooth_background(power, n_passes=40):
    """Compute smoothed background spectrum using 1-2-1 filter.

    Apply repeated 1-2-1 smoothing in both frequency and wavenumber
    directions to get a red-noise-like background.
    """
    bg = power.copy()
    for _ in range(n_passes):
        # Smooth along frequency (axis 0)
        bg = uniform_filter1d(bg, size=3, axis=0, mode='wrap')
        # Smooth along wavenumber (axis 1)
        bg = uniform_filter1d(bg, size=3, axis=1, mode='wrap')
    return bg


def kelvin_dispersion(k, he):
    """Kelvin wave dispersion: omega = k * sqrt(g*he)."""
    c = np.sqrt(G * he)
    # Convert to cycles/day: k is in wavenumber (cycles/circumference)
    # omega in rad/s = k * 2*pi / (2*pi*R) * c = k * c / R
    # freq in cycles/day = omega / (2*pi) * 86400
    omega = k * c / R_EARTH_M  # rad/s
    return omega / (2 * np.pi) * 86400  # cycles/day


def er_dispersion(k, he, n=1):
    """Equatorial Rossby wave dispersion (n=1 meridional mode).

    omega = -beta * k / (k^2 + (2n+1)*beta/sqrt(g*he))
    where k is in 1/m.
    """
    c = np.sqrt(G * he)
    k_m = k * 2 * np.pi / (2 * np.pi * R_EARTH_M)  # convert to 1/m
    # Avoid division by zero
    denom = k_m**2 + (2 * n + 1) * BETA / c
    with np.errstate(divide='ignore', invalid='ignore'):
        omega = -BETA * k_m / denom  # rad/s
    freq = np.abs(omega) / (2 * np.pi) * 86400  # cycles/day
    return freq


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_wk_diagram(power, freqs, wavenums, component, n_windows, outdir):
    """Plot the Wheeler-Kiladis diagram for one component."""

    # Rearrange to have negative wavenumbers (westward) on left,
    # positive (eastward) on right, and only positive frequencies
    freq_pos = freqs[:len(freqs) // 2]  # positive frequencies
    power_pos = power[:len(freqs) // 2, :]  # corresponding power

    # Shift wavenumber axis so 0 is in the center
    wn_shift = np.fft.fftshift(wavenums)
    power_shift = np.fft.fftshift(power_pos, axes=1)

    # Limit to reasonable range
    wn_max = 15
    freq_max = 0.8  # cycles/day (period > 1.25 days)
    freq_min = 1.0 / SEGMENT_DAYS  # Nyquist-limited by segment length

    wn_mask = (wn_shift >= -wn_max) & (wn_shift <= wn_max)
    freq_mask = (freq_pos >= freq_min) & (freq_pos <= freq_max)

    wn_sel = wn_shift[wn_mask]
    freq_sel = freq_pos[freq_mask]
    power_sel = power_shift[np.ix_(freq_mask, wn_mask)]

    # Compute background and ratio
    bg = smooth_background(power_shift)
    bg_sel = bg[np.ix_(freq_mask, wn_mask)]
    ratio = power_sel / np.where(bg_sel > 0, bg_sel, 1e-30)

    # ── Figure: signal / background ratio ──
    fig, ax = plt.subplots(figsize=(10, 8))

    levels = np.arange(1.1, 3.1, 0.2)
    cf = ax.contourf(wn_sel, freq_sel, ratio, levels=levels,
                     cmap="YlOrRd", extend="max")
    ax.contour(wn_sel, freq_sel, ratio, levels=levels,
               colors="0.5", linewidths=0.3)
    plt.colorbar(cf, ax=ax, label="Signal / Background", shrink=0.8)

    # Dispersion curves
    k_arr = np.linspace(0.5, wn_max, 200)
    for he in EQUIV_DEPTHS:
        # Kelvin (eastward, positive k)
        freq_kelvin = kelvin_dispersion(k_arr, he)
        ax.plot(k_arr, freq_kelvin, "k--", lw=0.8, alpha=0.7)

        # ER n=1 (westward, negative k)
        freq_er = er_dispersion(-k_arr, he, n=1)
        ax.plot(-k_arr, freq_er, "k--", lw=0.8, alpha=0.7)

    # Label equivalent depths
    for he in EQUIV_DEPTHS:
        f_k = kelvin_dispersion(wn_max - 1, he)
        if f_k < freq_max:
            ax.text(wn_max - 0.5, f_k, f"{he}m", fontsize=7, va="bottom",
                    ha="right", color="0.3")

    # Reference lines
    ax.axvline(0, color="k", lw=0.5)

    # Period labels on right y-axis
    ax2 = ax.twinx()
    period_ticks = [3, 6, 10, 20, 30, 60, 90]
    period_freqs = [1.0 / p for p in period_ticks]
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(period_freqs)
    ax2.set_yticklabels([f"{p}d" for p in period_ticks])
    ax2.set_ylabel("Period")

    # MJO box
    from matplotlib.patches import Rectangle
    mjo_rect = Rectangle((1, 1/90), 4, 1/30 - 1/90,
                          linewidth=1.5, edgecolor="blue",
                          facecolor="none", linestyle="-", label="MJO")
    ax.add_patch(mjo_rect)

    ax.set_xlabel("Zonal Wavenumber (negative=westward, positive=eastward)",
                  fontsize=11)
    ax.set_ylabel("Frequency (cycles/day)", fontsize=11)
    ax.set_title(f"Wheeler–Kiladis Diagram — {component}\n"
                 f"(6h precip, 15°S–15°N, {n_windows} windows)",
                 fontsize=13)
    ax.set_xlim(-wn_max, wn_max)
    ax.set_ylim(freq_min, freq_max)

    # Westward / Eastward labels
    ax.text(-wn_max / 2, freq_max * 0.95, "← Westward", fontsize=10,
            ha="center", va="top", color="0.4")
    ax.text(wn_max / 2, freq_max * 0.95, "Eastward →", fontsize=10,
            ha="center", va="top", color="0.4")

    fig.tight_layout()
    outpath = outdir / f"wheeler_kiladis_{component.lower()}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    _ts(f"  → {outpath}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wheeler-Kiladis diagram from 6h precipitation")
    parser.add_argument("--n_segments", default="3",
                        help="Number of data segments to use, or 'all'")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    cache_path = Path("cache_wheeler_kiladis.npz")

    seg_dirs = sorted(glob.glob(DATA_PATTERN))
    if args.n_segments == "all":
        n_use = len(seg_dirs)
    else:
        n_use = min(int(args.n_segments), len(seg_dirs))
    seg_dirs = seg_dirs[:n_use]

    print(f"Wheeler-Kiladis analysis")
    print(f"  Data segments: {n_use} (~{n_use * 80} years)")
    print(f"  Window: {SEGMENT_DAYS} days, overlap: {OVERLAP_DAYS} days")
    print()

    if cache_path.exists() and not args.overwrite:
        _ts(f"Loading cached spectra from {cache_path}")
        cached = np.load(cache_path)
        sym_power = cached["sym_power"]
        asym_power = cached["asym_power"]
        freqs = cached["freqs"]
        wavenums = cached["wavenums"]
        total_windows = int(cached["total_windows"])
    else:
        sym_power_sum = None
        asym_power_sum = None
        total_windows = 0

        for i, sd in enumerate(seg_dirs):
            _ts(f"Segment {i}: {sd}")
            data, lat = load_tropical_precip(sd)
            _ts(f"  Loaded: {data.shape} ({data.shape[0]/4:.0f} days)")

            sym, asym = decompose_symmetric(data, lat)
            del data

            _ts(f"  Computing symmetric spectrum …")
            sp, freqs, wavenums, nw = compute_wk_spectrum(sym, lat[lat >= 0][:sym.shape[1]])
            if sym_power_sum is None:
                sym_power_sum = sp * nw
            else:
                sym_power_sum += sp * nw
            del sym

            _ts(f"  Computing antisymmetric spectrum …")
            ap, _, _, nw2 = compute_wk_spectrum(asym, lat[lat >= 0][:asym.shape[1]])
            if asym_power_sum is None:
                asym_power_sum = ap * nw2
            else:
                asym_power_sum += ap * nw2
            del asym

            total_windows += nw
            _ts(f"  {nw} windows this segment, {total_windows} total")

        sym_power = sym_power_sum / total_windows
        asym_power = asym_power_sum / total_windows

        np.savez(cache_path, sym_power=sym_power, asym_power=asym_power,
                 freqs=freqs, wavenums=wavenums, total_windows=total_windows)
        _ts(f"Cached to {cache_path}")

    print()
    _ts("Generating figures …")
    plot_wk_diagram(sym_power, freqs, wavenums, "Symmetric", total_windows, OUTDIR)
    plot_wk_diagram(asym_power, freqs, wavenums, "Antisymmetric", total_windows, OUTDIR)

    print(f"\nDone — 2 figures in {OUTDIR}/")


if __name__ == "__main__":
    main()
