"""Deep dive on the LSME emissivity validation against TELSEM (July 1998 F-13).

Recomputes the full 31-day July composite TWICE per day with only the skin
temperature source swapped (Ken's GridSat-B1 IR clr versus ERA5 reanalysis
skin temperature), over an IDENTICAL clear-sky-and-land domain and the same
ERA5 atmospheric correction, so the GridSat-Ts contribution is cleanly isolated
per channel at monthly scale. Then validates both against the real TELSEM2 July
climatology and writes a set of diagnostic figures plus a stats JSON.

Usage (in the JupyterHub env):
    python -m scripts.validation_deepdive \
        --gridsat-root /path/to/swi_data/data/gridsat_cld --out ../scratch/deepdive

Figures (PNG, in --out):
    scatter_grid.png   per-channel density of derived vs TELSEM emissivity
    residual_maps.png  spatial (derived - TELSEM) for 37H and 85V
    ts_effect.png      GridSat-Ts vs ERA5-Ts: Pearson and RMSE per channel
    pol_structure.png  V vs H emissivity distributions + the 19 GHz V-H map
"""

import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swi import io_csu_grid, lsme, telsem
from swi.channels import CHANNEL_NAMES
from scripts.run_lsme import gridsat_ts, era5_field

F13DIR_DEFAULT = "../data/f13_1998"
GRIDSAT_DEFAULT = "../data/gridsat_july"
F13_GLOB = "CSU_SSMI_FCDR-GRID_V02R00_F13_D199807{dd}.nc"
TELSEM_LABEL = [f"{f:g}{p}" for f, p in telsem.BASIST_TO_TELSEM]


def opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def discover_days(f13dir):
    paths = glob.glob(os.path.join(f13dir, F13_GLOB.format(dd="??")))
    return sorted(re.search(r"D199807(\d\d)\.nc$", p).group(1) for p in paths)


def composite(f13dir, gridsat_root, days, pass_="dsc"):
    """Return (lat, lon, e_gs, e_e5, land): monthly-mean emissivity with GridSat Ts
    and with ERA5 Ts over the identical GridSat-clear+land domain, and the land mask."""
    sg = cg = se = ce = lat = lon = land = None
    used = []
    for dd in days:
        f13 = os.path.join(f13dir, F13_GLOB.format(dd=dd))
        if not os.path.exists(f13):
            continue
        lat, lon, tb, _ = io_csu_grid.read_channels(f13, pass_=pass_)
        g = gridsat_ts(gridsat_root, lat, lon, csu_path=f13)
        if g is None:
            continue
        ts_gs, clear_gs = g
        land = telsem.land_mask(lat, lon)
        clear = clear_gs & land
        tcwv = era5_field(f13, "tcwv", lat, lon)
        t_atm = era5_field(f13, "t2m", lat, lon)
        ts_e5 = era5_field(f13, "skt", lat, lon)
        r_gs = lsme.derive_emissivity(tb, ts_gs, clear=clear, tcwv_mm=tcwv, t_atm=t_atm)
        e_gs = r_gs["emissivity"]
        if sg is None:
            sg = np.zeros_like(e_gs); cg = np.zeros_like(e_gs)
            se = np.zeros_like(e_gs); ce = np.zeros_like(e_gs)
        vg = np.isfinite(e_gs); sg[vg] += e_gs[vg]; cg[vg] += 1.0
        if ts_e5 is not None:
            r_e5 = lsme.derive_emissivity(tb, ts_e5, clear=clear, tcwv_mm=tcwv, t_atm=t_atm)
            e_e5 = r_e5["emissivity"]
            ve = np.isfinite(e_e5); se[ve] += e_e5[ve]; ce[ve] += 1.0
        used.append(dd)
        print(f"07-{dd}: {int(r_gs['n_clear']):,} clear-land px")
    e_gs = np.where(cg > 0, sg / np.where(cg > 0, cg, 1.0), np.nan)
    e_e5 = np.where(ce > 0, se / np.where(ce > 0, ce, 1.0), np.nan)
    print(f"composite of {len(used)} days ({used[0]}..{used[-1]})")
    return lat, lon, e_gs, e_e5, land


def per_channel_stats(e, lat, lon, atlas):
    """List of dicts per channel: pearson, spearman, bias, rmse, n (vs TELSEM, land)."""
    return telsem.compare(e, lat, lon, atlas, lat, lon)


def zonal_stats(e, atlas, lat, lon, land, ch):
    """Pearson and bias of channel ch by latitude band."""
    lat2d = np.repeat(np.asarray(lat)[:, None], lon.size, axis=1)
    bands = {"tropics |lat|<23.5": np.abs(lat2d) < 23.5,
             "midlat 23.5-60": (np.abs(lat2d) >= 23.5) & (np.abs(lat2d) < 60),
             "high-lat >60": np.abs(lat2d) >= 60}
    a = e[:, :, ch]; b = atlas[:, :, ch]
    out = {}
    for name, m in bands.items():
        sel = m & land & np.isfinite(a) & np.isfinite(b)
        if sel.sum() < 50:
            out[name] = {"n": int(sel.sum()), "pearson": None, "bias": None}
            continue
        x = a[sel]; y = b[sel]
        r = float(np.corrcoef(x, y)[0, 1])
        out[name] = {"n": int(sel.sum()), "pearson": r, "bias": float(np.mean(x - y))}
    return out


def roll180(arr, lon):
    """Roll a 0..360 field to -180..180 for display; return (rolled, extent_x)."""
    return np.roll(arr, lon.size // 2, axis=1), (-180.0, 180.0)


# --- figures -------------------------------------------------------------------

def fig_scatter(e, atlas, land, out):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.suptitle("Derived LSME emissivity vs TELSEM2 July climatology "
                 "(F-13 1998, 31-day composite, land)", fontsize=13)
    for c in range(7):
        ax = axes.flat[c]
        a = e[:, :, c]; b = atlas[:, :, c]
        m = land & np.isfinite(a) & np.isfinite(b)
        x = b[m]; y = a[m]
        ax.hexbin(x, y, gridsize=60, bins="log", cmap="viridis", extent=(0.3, 1.0, 0.3, 1.0))
        ax.plot([0.3, 1.0], [0.3, 1.0], "w--", lw=1)
        r = np.corrcoef(x, y)[0, 1]
        bias = np.mean(y - x)
        ax.set_title(f"{CHANNEL_NAMES[c]}  ({TELSEM_LABEL[c]})", fontsize=10)
        ax.text(0.05, 0.93, f"r={r:.3f}\nbias={bias:+.3f}\nn={x.size:,}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", alpha=0.7, ec="none"))
        ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("TELSEM"); ax.set_ylabel("derived")
    axes.flat[7].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, "scatter_grid.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


def fig_residual_maps(e, atlas, land, lat, lon, out):
    chans = [(4, "37H"), (5, "85V")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle("Derived minus TELSEM emissivity (31-day July composite, land)",
                 fontsize=13)
    for ax, (c, name) in zip(axes, chans):
        d = e[:, :, c] - atlas[:, :, c]
        d = np.where(land & np.isfinite(d), d, np.nan)
        dr, ex = roll180(d, lon)
        im = ax.imshow(dr, origin="lower", extent=(ex[0], ex[1], lat[0], lat[-1]),
                       cmap="RdBu_r", vmin=-0.1, vmax=0.1, aspect="auto")
        ax.set_title(f"{CHANNEL_NAMES[c]} ({TELSEM_LABEL[c]})", fontsize=11)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        fig.colorbar(im, ax=ax, shrink=0.85, label="derived - TELSEM")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, "residual_maps.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


def fig_ts_effect(stats_gs, stats_e5, out):
    chs = [s["channel"] for s in stats_gs]
    r_gs = [s["pearson_r"] for s in stats_gs]
    r_e5 = [s["pearson_r"] for s in stats_e5]
    rmse_gs = [s["rmse"] for s in stats_gs]
    rmse_e5 = [s["rmse"] for s in stats_e5]
    x = np.arange(len(chs)); w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Effect of the skin-temperature source: GridSat-B1 IR vs ERA5 "
                 "(same clear-sky+land domain, same atmosphere)", fontsize=12)
    a1.bar(x - w/2, r_e5, w, label="ERA5 Ts", color="#bbbbbb")
    a1.bar(x + w/2, r_gs, w, label="GridSat-B1 Ts", color="#2c7fb8")
    a1.set_xticks(x); a1.set_xticklabels(chs); a1.set_ylabel("Pearson r vs TELSEM")
    a1.set_ylim(0.5, 1.0); a1.set_title("Pattern correlation"); a1.legend()
    a1.grid(axis="y", alpha=0.3)
    a2.bar(x - w/2, rmse_e5, w, label="ERA5 Ts", color="#bbbbbb")
    a2.bar(x + w/2, rmse_gs, w, label="GridSat-B1 Ts", color="#2c7fb8")
    a2.set_xticks(x); a2.set_xticklabels(chs); a2.set_ylabel("RMSE (emissivity)")
    a2.set_title("Root-mean-square difference"); a2.legend(); a2.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "ts_effect.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


def fig_pol_structure(e, land, lat, lon, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Polarization structure of the derived emissivity (31-day July, land)",
                 fontsize=12)
    # V vs H histograms for the three dual-pol window pairs
    pairs = [(0, 1, "19 GHz"), (3, 4, "37 GHz"), (5, 6, "85 GHz")]
    colorsV = ["#1b9e77", "#7570b3", "#d95f02"]
    for (iv, ih, name), col in zip(pairs, colorsV):
        v = e[:, :, iv][land & np.isfinite(e[:, :, iv])]
        h = e[:, :, ih][land & np.isfinite(e[:, :, ih])]
        a1.hist(v, bins=80, range=(0.3, 1.0), histtype="step", lw=1.8,
                color=col, label=f"{name} V")
        a1.hist(h, bins=80, range=(0.3, 1.0), histtype="step", lw=1.2, ls="--",
                color=col, label=f"{name} H")
    a1.set_xlabel("emissivity"); a1.set_ylabel("land cells")
    a1.set_title("V (solid) sits above H (dashed); H low tail = wet/open water")
    a1.legend(fontsize=8)
    # 19 GHz V-H polarization difference map (wetness-sensitive)
    pd = e[:, :, 0] - e[:, :, 1]
    pd = np.where(land & np.isfinite(pd), pd, np.nan)
    pr, ex = roll180(pd, lon)
    im = a2.imshow(pr, origin="lower", extent=(ex[0], ex[1], lat[0], lat[-1]),
                   cmap="YlOrBr", vmin=0.0, vmax=0.25, aspect="auto")
    a2.set_title("19 GHz V - H: bright = bare/smooth (deserts), dark = vegetated/rough; "
                 "soil moisture modulates it upward at fixed roughness", fontsize=8)
    a2.set_xlabel("longitude"); a2.set_ylabel("latitude")
    fig.colorbar(im, ax=a2, shrink=0.85, label="V - H emissivity")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "pol_structure.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


def main(argv):
    f13dir = opt(argv, "--f13-dir", F13DIR_DEFAULT)
    gridsat_root = opt(argv, "--gridsat-root", GRIDSAT_DEFAULT)
    out = opt(argv, "--out", "../scratch/deepdive")
    days_arg = opt(argv, "--days", "all")
    days = discover_days(f13dir) if days_arg == "all" else days_arg.split(",")
    os.makedirs(out, exist_ok=True)

    print(f"f13-dir={f13dir}  gridsat-root={gridsat_root}  days={len(days)}  out={out}\n")
    lat, lon, e_gs, e_e5, land = composite(f13dir, gridsat_root, days)

    alat, alon, atlas = telsem.load_atlas(None, 7, lat=lat, lon=lon)
    stats_gs = per_channel_stats(e_gs, lat, lon, atlas)
    stats_e5 = per_channel_stats(e_e5, lat, lon, atlas)

    print(f"\n{'ch':>4} {'telsem':>7} {'n':>8} | "
          f"{'r(GS)':>6} {'r(E5)':>6} {'dr':>6} | "
          f"{'rmse(GS)':>8} {'rmse(E5)':>8} | {'bias(GS)':>8}")
    for sg, se in zip(stats_gs, stats_e5):
        print(f"{sg['channel']:>4} {sg['telsem']:>7} {sg['n']:>8,} | "
              f"{sg['pearson_r']:>6.3f} {se['pearson_r']:>6.3f} "
              f"{sg['pearson_r']-se['pearson_r']:>+6.3f} | "
              f"{sg['rmse']:>8.3f} {se['rmse']:>8.3f} | {sg['bias']:>+8.3f}")

    zon = {CHANNEL_NAMES[c]: zonal_stats(e_gs, atlas, lat, lon, land, c)
           for c in (1, 4, 5)}            # 19H, 37H, 85V
    print("\nZonal (GridSat Ts) Pearson by band:")
    for ch, z in zon.items():
        print(f"  {ch}: " + ", ".join(
            f"{b}={d['pearson']:.3f}(n={d['n']:,})" if d['pearson'] is not None
            else f"{b}=NA" for b, d in z.items()))

    # figures
    fig_scatter(e_gs, atlas, land, out)
    fig_residual_maps(e_gs, atlas, land, lat, lon, out)
    fig_ts_effect(stats_gs, stats_e5, out)
    fig_pol_structure(e_gs, land, lat, lon, out)

    summary = {"days": len(days), "stats_gridsat": stats_gs, "stats_era5ts": stats_e5,
               "zonal_gridsat": zon}
    jp = os.path.join(out, "deepdive_stats.json")
    with open(jp, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\nwrote", jp)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
