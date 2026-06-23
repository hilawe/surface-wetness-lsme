"""Per-cell anomaly-map quality progression for the 37 GHz horizontal channel.

Plots the 37 GHz horizontal-polarization per-cell temporal anomaly correlation
(LSME vs TELSEM) three ways. At the native 0.25 degree grid the F-13
descending-orbit sampling leaves arc striping. Coarsened to 0.5 degree, block
averaging across the stripes removes the arcs and raises the correlation. With a
TELSEM seasonal-amplitude floor added, the no-signal desert cores are masked. The
arcs are orbital sampling rather than footprint or scan handling, since the
pipeline reads the CSU FCDR-GRID pre-gridded daily field and never raw scans.

Usage:
    python -m scripts.swath_compare --emis-dir ../scratch/lsme_monthly \\
        --telsem ../data/telsem2 --out ../scratch/anomaly
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swi import anomaly, io_lsme_monthly, telsem, validate
from swi.channels import CHANNEL_NAMES


def opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def load_stacks(emis_dir, telsem_dir):
    paths = sorted(glob.glob(os.path.join(emis_dir, "LSME_emis_F13_??????.nc")))
    lat = lon = None
    lsme, tel = [], []
    for p in paths:
        mo = int(re.search(r"_(\d{4})(\d{2})\.nc$", p).group(2))
        lat, lon, emis, _ = io_lsme_monthly.read_monthly_emis(p)
        lsme.append(emis)
        _, _, atlas = telsem.load_atlas(telsem_dir, mo, lat=lat, lon=lon)
        tel.append(atlas)
    return lat, lon, np.stack(lsme, 0), np.stack(tel, 0)


def rmap_37h(lsme_stack, telsem_stack, lat, lon, coarsen, min_signal=0.0):
    """37H per-cell temporal anomaly correlation, optionally coarsened and masked.

    coarsen blocks the grid; min_signal drops cells where the TELSEM 37H seasonal
    amplitude is below the floor (the no-signal deserts).
    """
    land = telsem.land_mask(lat, lon)
    if coarsen > 1:
        for ax in (1, 2):
            lsme_stack = anomaly.block_mean(lsme_stack, coarsen, ax)
            telsem_stack = anomaly.block_mean(telsem_stack, coarsen, ax)
        lat = anomaly.block_mean(lat, coarsen, 0)
        lon = anomaly.block_mean(lon, coarsen, 0)
        land = (anomaly.block_mean(anomaly.block_mean(land.astype(float), coarsen, 0),
                                   coarsen, 1) >= 0.5)
    c = CHANNEL_NAMES.index("37H")
    a, b = lsme_stack[..., c], telsem_stack[..., c]
    if min_signal > 0:
        with np.errstate(invalid="ignore"):
            low = np.nanstd(b, axis=0) < min_signal
        a = np.where(low[None], np.nan, a)
        b = np.where(low[None], np.nan, b)
    rmap, _ = validate.temporal_anomaly_correlation(a, b, min_n=max(2, a.shape[0] // 2))
    rmap = np.where(land, rmap, np.nan)
    return lat, lon, rmap


def main(argv):
    emis_dir = opt(argv, "--emis-dir", "../scratch/lsme_monthly")
    telsem_dir = opt(argv, "--telsem", "../data/telsem2")
    out = opt(argv, "--out", "../scratch/anomaly")
    os.makedirs(out, exist_ok=True)

    lat0, lon0, lsme_stack, telsem_stack = load_stacks(emis_dir, telsem_dir)

    configs = [(1, 0.0, "native 0.25 deg (orbital arcs)"),
               (2, 0.0, "coarsened to 0.5 deg (arcs averaged out)"),
               (2, 0.006, "0.5 deg plus signal floor 0.006 (no-signal deserts masked)")]
    fig, axes = plt.subplots(3, 1, figsize=(13, 13))
    fig.suptitle("Per-cell temporal anomaly correlation, 37H (LSME vs TELSEM): "
                 "swath sampling, coarsening, and the no-signal mask", fontsize=13)
    for ax, (coarsen, ms, tag) in zip(axes, configs):
        lat, lon, rmap = rmap_37h(lsme_stack, telsem_stack, lat0, lon0, coarsen, ms)
        med = float(np.nanmedian(rmap))
        n = int(np.isfinite(rmap).sum())
        rolled = np.roll(rmap, lon.size // 2, axis=1)
        im = ax.imshow(rolled, origin="lower", extent=(-180, 180, lat[0], lat[-1]),
                       cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"{0.25 * coarsen:g} deg: {tag}  -  median r {med:.3f}, "
                     f"{n:,} land cells", fontsize=10)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out, "swath_compare.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
