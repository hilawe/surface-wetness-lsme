"""Build the monthly two-estimator relative-wetness product (thermal + microwave).

Composites both GridSat-B1-anchored estimators over a month, turns each into a
0-to-1 relative wetness index (1 = wet), averages them, reports their agreement,
and writes a CF-1.8 NetCDF product plus a quick-look map.

  thermal     dT = Tmax - Tmin diurnal infrared range (swi.thermal); small = wet.
  microwave   19 GHz V - H polarization difference from the LSME emissivity; small
              = wet (vegetated, rough, or moist). Chosen because it agrees with dT,
              unlike the H-pol emissivity which runs the other way under vegetation.

Usage (in-env):
    python -m scripts.make_two_estimator --year 1998 --month 7 \
        --gridsat-root /path/to/swi_data/data/gridsat_cld --out ../scratch/two_estimator
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swi import io_csu_grid, lsme, telsem, thermal, two_estimator
from scripts.run_lsme import build_inputs

F13DIR_DEFAULT = "../data/f13_1998"


def opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def discover_days(f13dir, ym):
    paths = glob.glob(os.path.join(f13dir, f"CSU_SSMI_FCDR-GRID_V02R00_F13_D{ym}??.nc"))
    return sorted(re.search(r"D\d{6}(\d\d)\.nc$", p).group(1) for p in paths)


def composite(f13dir, gridsat_root, year, month, days, pass_="dsc"):
    """Monthly-mean emissivity (for the V-H PD) and thermal dT on the F-13 grid."""
    isccp, ml = thermal.split_root(gridsat_root)
    ym = f"{year}{month:02d}"
    se = ce = sdt = cdt = lat = lon = None
    used = []
    for dd in days:
        f13 = os.path.join(f13dir, f"CSU_SSMI_FCDR-GRID_V02R00_F13_D{ym}{dd}.nc")
        if not os.path.exists(f13):
            continue
        lat, lon, tb, _ = io_csu_grid.read_channels(f13, pass_=pass_)
        ts, _, t_atm, _, tcwv, _, clear, clbl = build_inputs(
            f13, lat, lon, [sys.argv[0], "--gridsat", gridsat_root])
        if "GridSat" not in clbl:
            continue
        e = lsme.derive_emissivity(tb, ts, clear=clear, tcwv_mm=tcwv, t_atm=t_atm)["emissivity"]
        try:
            dT = thermal.diurnal_range_on_grid(isccp, year, month, int(dd), lat, lon,
                                               ml_root=ml)["dT"]
        except FileNotFoundError:
            continue
        if se is None:
            se = np.zeros_like(e); ce = np.zeros_like(e)
            sdt = np.zeros_like(dT); cdt = np.zeros_like(dT)
        ve = np.isfinite(e); se[ve] += e[ve]; ce[ve] += 1.0
        vd = np.isfinite(dT); sdt[vd] += dT[vd]; cdt[vd] += 1.0
        used.append(dd)
        print(f"{month:02d}-{dd}: ok")
    e_m = np.where(ce > 0, se / np.where(ce > 0, ce, 1.0), np.nan)
    dT_m = np.where(cdt > 0, sdt / np.where(cdt > 0, cdt, 1.0), np.nan)
    print(f"composite of {len(used)} days")
    # cdt is 2-D (dT is 2-D); ce is per-channel, take channel 0 for the count.
    return lat, lon, e_m, dT_m, cdt, ce[:, :, 0]


def main(argv):
    f13dir = opt(argv, "--f13-dir", F13DIR_DEFAULT)
    gridsat_root = opt(argv, "--gridsat-root", "../data/gridsat_july")
    year = int(opt(argv, "--year", "1998"))
    month = int(opt(argv, "--month", "7"))
    out = opt(argv, "--out", "../scratch/two_estimator")
    os.makedirs(out, exist_ok=True)
    ym = f"{year}{month:02d}"
    days = discover_days(f13dir, ym)

    lat, lon, e_m, dT_m, n_dt, n_e = composite(f13dir, gridsat_root, year, month, days)
    land = telsem.land_mask(lat, lon)
    pd = e_m[:, :, 0] - e_m[:, :, 1]                       # 19 GHz V - H

    thermal_wet = two_estimator.wetness_index(dT_m, land, wet_is_low=True)
    micro_wet = two_estimator.wetness_index(pd, land, wet_is_low=True)
    combined, agreement = two_estimator.combine(thermal_wet, micro_wet)

    both = np.isfinite(combined)
    r = np.corrcoef(thermal_wet[both], micro_wet[both])[0, 1] if both.sum() > 50 else float("nan")
    print(f"land cells with both estimators: {int(both.sum()):,}; "
          f"thermal-vs-microwave wetness r = {r:+.3f}; "
          f"mean agreement = {np.nanmean(agreement):.3f}")

    nc = os.path.join(out, f"SWI_two_estimator_F13_{ym}.nc")
    two_estimator.write_product(
        nc, lat, lon,
        {"thermal_dT": (dT_m, "K", "GridSat-B1 clear-sky diurnal infrared range Tmax-Tmin"),
         "vh_pd_19": (pd, "1", "19 GHz V minus H emissivity polarization difference"),
         "thermal_wetness": (thermal_wet, "1", "thermal-inertia relative wetness index, 1=wet"),
         "microwave_wetness": (micro_wet, "1", "polarization-difference relative wetness index, 1=wet"),
         "combined_wetness": (combined, "1", "mean of the thermal and microwave wetness indices"),
         "agreement": (agreement, "1", "1 minus the absolute difference of the two indices"),
         "n_obs_thermal": (n_dt, "1", "clear days contributing to thermal_dT"),
         "n_obs_microwave": (n_e, "1", "clear days contributing to the emissivity")},
        {"title": "Two-estimator relative land surface wetness (thermal + microwave)",
         "summary": ("Monthly relative wetness from two independent GridSat-B1-anchored "
                     "estimators: the clear-sky diurnal infrared range and the 19 GHz "
                     "V-H microwave emissivity polarization difference. Relative, not "
                     "absolute soil moisture; most meaningful over low vegetation; "
                     "time-varying signal is cleanest as an anomaly once a climatology "
                     "exists."),
         "institution": "NOAA National Centers for Environmental Information",
         "source": "CSU SSM/I FCDR (F-13) and GridSat-B1 (Knapp), F-13 1998",
         "creator_name": "Hilawe Semunegus",
         "contributor_name": "Kenneth R. Knapp (GridSat-B1)",
         "Conventions": "CF-1.8, ACDD-1.3",
         "calibration": "85-to-91 GHz not applicable (F-13 is true 85.5 GHz)"})
    print("wrote", nc)

    fig_quicklook(thermal_wet, micro_wet, combined, agreement, land, lat, lon, out, ym)
    return 0


def roll180(a, lon):
    return np.roll(a, lon.size // 2, axis=1)


def fig_quicklook(thermal_wet, micro_wet, combined, agreement, land, lat, lon, out, ym):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(f"Two-estimator relative wetness, F-13 {ym} (land; 1 = wet)", fontsize=13)
    panels = [(thermal_wet, "thermal-inertia wetness (small dT)", "YlGnBu"),
              (micro_wet, "microwave V-H wetness (small PD)", "YlGnBu"),
              (combined, "combined wetness", "YlGnBu"),
              (agreement, "agreement (1 = same wet/dry place)", "RdYlGn")]
    ext = (-180, 180, lat[0], lat[-1])
    for ax, (fld, title, cmap) in zip(axes.flat, panels):
        d = np.where(land & np.isfinite(fld), fld, np.nan)
        im = ax.imshow(roll180(d, lon), origin="lower", extent=ext, cmap=cmap,
                       vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out, f"two_estimator_{ym}.png")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
