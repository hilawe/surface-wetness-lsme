"""Two-estimator cross-check: GridSat thermal inertia versus microwave emissivity.

Builds the monthly composite of BOTH independent wetness estimators on the same
clear-sky, land domain and asks whether they agree:

  thermal     dT = Tmax - Tmin from the GridSat-B1 diurnal cycle (swi.thermal);
              a SMALL dT means a wetter, higher-thermal-inertia surface.
  emissivity  the LSME microwave emissivity (swi.lsme) with the GridSat-B1 skin
              temperature and the ERA5 atmosphere; LOWER emissivity means wetter.

Dry land should show a large diurnal range and high emissivity; wet land a small
range and low emissivity, so the two should be POSITIVELY correlated. That
agreement, from two physically independent signals that share only the GridSat-B1
anchor, is the point of the two-estimator product.

Usage (in-env):
    python -m scripts.run_thermal --gridsat-root /path/to/swi_data/data/gridsat_cld \
        --year 1998 --month 7 --out ../scratch/thermal
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

from swi import io_csu_grid, lsme, telsem, thermal
from scripts.run_lsme import build_inputs

F13DIR_DEFAULT = "../data/f13_1998"
CROSS_CH = {"19H": 1, "37H": 4, "85V": 5}        # emissivity channels to cross-check


def opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def discover_days(f13dir, year, month):
    pat = f"CSU_SSMI_FCDR-GRID_V02R00_F13_D{year}{month:02d}??.nc"
    paths = glob.glob(os.path.join(f13dir, pat))
    return sorted(re.search(r"D\d{6}(\d\d)\.nc$", p).group(1) for p in paths)


def rankcorr(x, y):
    """Spearman rank correlation without scipy."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main(argv):
    f13dir = opt(argv, "--f13-dir", F13DIR_DEFAULT)
    gridsat_root = opt(argv, "--gridsat-root", "../data/gridsat_july")
    year = int(opt(argv, "--year", "1998"))
    month = int(opt(argv, "--month", "7"))
    out = opt(argv, "--out", "../scratch/thermal")
    days = discover_days(f13dir, year, month)
    os.makedirs(out, exist_ok=True)
    isccp, ml = thermal.split_root(gridsat_root)

    print(f"f13-dir={f13dir} gridsat={gridsat_root} {year}-{month:02d} "
          f"days={len(days)} out={out}\n")

    sdT = cdT = se = ce = lat = lon = None
    used = []
    for dd in days:
        f13 = os.path.join(f13dir,
                           f"CSU_SSMI_FCDR-GRID_V02R00_F13_D{year}{month:02d}{dd}.nc")
        if not os.path.exists(f13):
            continue
        lat, lon, tb, _ = io_csu_grid.read_channels(f13, pass_="dsc")
        # microwave emissivity (GridSat Ts + ERA5 atmosphere, land + clear)
        ts, _, t_atm, _, tcwv, _, clear, clbl = build_inputs(
            f13, lat, lon, [argv[0], "--gridsat", gridsat_root])
        if "GridSat" not in clbl:
            continue
        e = lsme.derive_emissivity(tb, ts, clear=clear, tcwv_mm=tcwv,
                                   t_atm=t_atm)["emissivity"]
        # thermal-inertia diurnal range from the same day's GridSat slots
        try:
            th = thermal.diurnal_range_on_grid(isccp, year, month, int(dd),
                                               lat, lon, ml_root=ml)
        except FileNotFoundError:
            continue
        dT = th["dT"]
        if sdT is None:
            sdT = np.zeros_like(dT); cdT = np.zeros_like(dT)
            se = np.zeros_like(e); ce = np.zeros_like(e)
        vd = np.isfinite(dT); sdT[vd] += dT[vd]; cdT[vd] += 1.0
        ve = np.isfinite(e); se[ve] += e[ve]; ce[ve] += 1.0
        used.append(dd)
        print(f"{month:02d}-{dd}: dT clear cells {int(vd.sum()):,}, "
              f"median dT {np.nanmedian(dT):.1f} K")

    dT_m = np.where(cdT > 0, sdT / np.where(cdT > 0, cdT, 1.0), np.nan)
    e_m = np.where(ce > 0, se / np.where(ce > 0, ce, 1.0), np.nan)
    land = telsem.land_mask(lat, lon)
    print(f"\ncomposite of {len(used)} days")

    # --- two-estimator cross-check ---
    # Both estimators are dominated by surface type. The microwave field that shares
    # dT's bare-versus-vegetated axis is the 19 GHz V-H polarization difference (high
    # over bare/dry/smooth, low over vegetated/wet), so it is the natural partner; the
    # H-pol emissivity runs the OTHER way (vegetation raises it while lowering dT).
    pd = e_m[:, :, 0] - e_m[:, :, 1]                  # 19 GHz V-H polarization difference
    targets = {"19 V-H PD": pd, "37H emis": e_m[:, :, 4], "85V emis": e_m[:, :, 5]}
    summary = {"year": year, "month": month, "days": len(used), "cross": {}}
    lat2d = np.repeat(np.asarray(lat)[:, None], lon.size, axis=1)
    zones = {"global": land,
             "tropics": (np.abs(lat2d) < 23.5) & land,
             "midlat": (np.abs(lat2d) >= 23.5) & (np.abs(lat2d) < 60) & land,
             "high-lat": (np.abs(lat2d) >= 60) & land}
    print("\nThermal dT vs microwave fields over land "
          "(dT and V-H PD should agree positively):")
    for name, fld in targets.items():
        row = {}
        for zn, zm in zones.items():
            m = zm & np.isfinite(dT_m) & np.isfinite(fld)
            if m.sum() < 50:
                row[zn] = {"n": int(m.sum())}; continue
            x = dT_m[m]; y = fld[m]
            row[zn] = {"n": int(m.sum()), "pearson": float(np.corrcoef(x, y)[0, 1]),
                       "spearman": rankcorr(x, y)}
        summary["cross"][name] = row
        g = row["global"]
        print(f"  dT vs {name:10s}: global r={g['pearson']:+.3f} rho={g['spearman']:+.3f} "
              f"(n={g['n']:,})")

    fig_maps(dT_m, pd, land, lat, lon, out)
    fig_scatter(dT_m, pd, land, out)

    jp = os.path.join(out, "thermal_stats.json")
    with open(jp, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\nwrote", jp)
    return 0


def roll180(arr, lon):
    return np.roll(arr, lon.size // 2, axis=1)


def fig_maps(dT, pd, land, lat, lon, out):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle("Two GridSat-anchored surface signals, monthly composite (land): "
                 "thermal diurnal range and microwave V-H", fontsize=12)
    d = np.where(land & np.isfinite(dT), dT, np.nan)
    im1 = a1.imshow(roll180(d, lon), origin="lower", extent=(-180, 180, lat[0], lat[-1]),
                    cmap="YlOrRd", vmin=0, vmax=35, aspect="auto")
    a1.set_title("GridSat diurnal range dT (K): LARGE = bare/dry, small = vegetated/wet",
                 fontsize=10)
    fig.colorbar(im1, ax=a1, shrink=0.85, label="dT (K)")
    p = np.where(land & np.isfinite(pd), pd, np.nan)
    im2 = a2.imshow(roll180(p, lon), origin="lower", extent=(-180, 180, lat[0], lat[-1]),
                    cmap="YlOrBr", vmin=0, vmax=0.25, aspect="auto")
    a2.set_title("19 GHz V-H polarization difference: HIGH = bare/dry/smooth, "
                 "low = vegetated/wet", fontsize=10)
    fig.colorbar(im2, ax=a2, shrink=0.85, label="V - H emissivity")
    for a in (a1, a2):
        a.set_xlabel("longitude"); a.set_ylabel("latitude")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p_out = os.path.join(out, "thermal_vs_pd_maps.png")
    fig.savefig(p_out, dpi=110); plt.close(fig); print("wrote", p_out)


def fig_scatter(dT, pd, land, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    m = land & np.isfinite(dT) & np.isfinite(pd)
    x = dT[m]; y = pd[m]
    ax.hexbin(x, y, gridsize=55, bins="log", cmap="viridis", extent=(0, 40, 0, 0.25))
    r = np.corrcoef(x, y)[0, 1]
    ax.set_xlabel("thermal diurnal range dT (K)")
    ax.set_ylabel("19 GHz V-H polarization difference")
    ax.set_title(f"Two GridSat-anchored estimators over land (r={r:+.3f}, n={x.size:,})\n"
                 "up-right = bare/dry (large swing, large V-H); "
                 "down-left = vegetated/wet", fontsize=10)
    fig.tight_layout()
    p_out = os.path.join(out, "thermal_vs_pd_scatter.png")
    fig.savefig(p_out, dpi=110); plt.close(fig); print("wrote", p_out)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
