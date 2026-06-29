"""Monthly composite of the LSME emissivity, validated against TELSEM.

Averages the per-day clear-sky emissivity over the available July 1998 days
(mean per cell, per channel, where clear), then compares the monthly-mean
emissivity to the TELSEM July climatology. This is the apples-to-apples
comparison (a monthly mean against a monthly climatology) and aggregates clear
matchups across days.

Each day uses GridSat Ts and the ML clear mask via build_inputs. A day is only
folded into the composite when GridSat Ts was actually used for it, so the mean
is never silently contaminated by the placeholder skin temperature.

Usage:
    python -m scripts.monthly_lsme [asc|dsc]
        [--f13-dir DIR] [--gridsat-root DIR] [--days all|DD,DD,...]

Defaults reproduce the original 5-day Mac sample when pointed at the 5-day
GridSat subset; pass --gridsat-root /path/to/swi_data/data/gridsat_cld in the
JupyterHub env to run the full month straight off the GridSat archive. With
--days omitted (or "all") the day list is auto-discovered from the F-13 files
present in --f13-dir, so the composite uses every July day on disk.
"""

import glob
import os
import re
import sys

import numpy as np

from swi import io_csu_grid, lsme, telsem
from scripts.run_lsme import build_inputs

F13DIR_DEFAULT = "../data/f13_1998"
GRIDSAT_DEFAULT = "../data/gridsat_july"


def csu_fmt(sat):
    """CSU FCDR-GRID filename pattern for a satellite, with {ym} and {dd} slots.

    F08 to F15 are SSM/I (CSU_SSMI_...), F16 and up are SSMIS (CSU_SSMIS_...),
    the same convention as scripts/fetch_csu.py.
    """
    sensor = "SSMI" if int(sat[1:]) <= 15 else "SSMIS"
    return f"CSU_{sensor}_FCDR-GRID_V02R00_{sat}_D{{ym}}{{dd}}.nc"


def opt(argv, name, default):
    """Return the value following --name in argv, or default if absent."""
    return argv[argv.index(name) + 1] if name in argv else default


def discover_days(f13dir, ym, fmt):
    """All days (DD) for which a CSU file matching fmt exists in f13dir for ym."""
    paths = glob.glob(os.path.join(f13dir, fmt.format(ym=ym, dd="??")))
    days = sorted(re.search(r"D\d{6}(\d\d)\.nc$", p).group(1) for p in paths)
    return days


def main(argv):
    pass_ = "dsc"
    for a in argv[1:]:
        if a in ("asc", "dsc"):
            pass_ = a

    f13dir = opt(argv, "--f13-dir", F13DIR_DEFAULT)
    gridsat_root = opt(argv, "--gridsat-root", GRIDSAT_DEFAULT)
    sat = opt(argv, "--sat", "F13").upper()
    sat_disp = f"{sat[:1]}-{sat[1:]}"   # F13 -> F-13 for human-readable labels
    fmt = csu_fmt(sat)
    year = int(opt(argv, "--year", "1998"))
    month = int(opt(argv, "--month", "7"))
    ym = f"{year}{month:02d}"
    days_arg = opt(argv, "--days", "all")
    days = discover_days(f13dir, ym, fmt) if days_arg == "all" else days_arg.split(",")
    # Optional 85 GHz scattering screen. --scatter-screen with no value uses the
    # calibrated default; --scatter-screen K uses threshold K (Kelvin).
    scatter_k = None
    if "--scatter-screen" in argv:
        i = argv.index("--scatter-screen")
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        try:
            scatter_k = float(nxt)
        except (TypeError, ValueError):
            scatter_k = lsme.SI37_DEFAULT_K

    print(f"f13-dir      : {f13dir}")
    print(f"satellite    : {sat}")
    print(f"gridsat-root : {gridsat_root}")
    print(f"period       : {year}-{month:02d}")
    print(f"pass         : {pass_}")
    print(f"scatter-screen: {'off' if scatter_k is None else f'{scatter_k:g} K (37V-85V)'}")
    print(f"days ({len(days)}) : {' '.join(days)}\n")

    build_argv = [argv[0], "--gridsat", gridsat_root]

    sum_e = cnt = lat = lon = None
    used, skipped = [], []
    for dd in days:
        f13 = os.path.join(f13dir, fmt.format(ym=ym, dd=dd))
        if not os.path.exists(f13):
            skipped.append((dd, f"no {sat_disp} file"))
            continue
        lat, lon, tb, _ = io_csu_grid.read_channels(f13, pass_=pass_)
        ts, _, t_atm, _, tcwv, _, clear, clear_label = build_inputs(
            f13, lat, lon, build_argv, pass_=pass_)
        if "GridSat" not in clear_label:
            # GridSat Ts/mask unavailable for this day; do not fold a
            # placeholder-Ts emissivity into the composite.
            skipped.append((dd, "no GridSat Ts"))
            continue
        r = lsme.derive_emissivity(tb, ts, clear=clear, tcwv_mm=tcwv, t_atm=t_atm,
                                   scatter_screen_k=scatter_k)
        e = r["emissivity"]
        if sum_e is None:
            sum_e = np.zeros_like(e)
            cnt = np.zeros_like(e)
        v = np.isfinite(e)
        sum_e[v] += e[v]
        cnt[v] += 1.0
        used.append(dd)
        print(f"07-{dd}: {int(r['n_clear']):,} clear-land pixels")

    if not used:
        print("\nNo days produced a GridSat-Ts emissivity; nothing to composite.")
        for dd, why in skipped:
            print(f"  skipped 07-{dd}: {why}")
        return 1

    monthly = np.where(cnt > 0, sum_e / np.where(cnt > 0, cnt, 1.0), np.nan)
    cov = np.isfinite(monthly).all(axis=2)
    print(f"\ncomposite of {len(used)} days "
          f"({used[0]}..{used[-1]}): {int(cov.sum()):,} cells with >=1 obs")
    if skipped:
        print("skipped: " + ", ".join(f"07-{dd}({why})" for dd, why in skipped))

    save = opt(argv, "--save-emis", None)
    if save:
        from swi import io_lsme_monthly
        os.makedirs(save, exist_ok=True)
        outp = os.path.join(save, f"LSME_emis_{sat}_{ym}.nc")
        io_lsme_monthly.write_monthly_emis(
            outp, lat, lon, monthly, cnt,
            attrs={"title": f"Monthly-mean LSME emissivity, {sat_disp} {ym}",
                   "source": f"CSU {'SSM/I' if int(sat[1:]) <= 15 else 'SSMIS'} "
                             f"FCDR ({sat_disp}) and GridSat-B1 (Knapp)",
                   "institution": "NOAA National Centers for Environmental Information",
                   "creator_name": "Hilawe Semunegus",
                   "days_used": " ".join(used)})
        print("wrote", outp)

    alat, alon, atlas = telsem.load_atlas(None, month, lat=lat, lon=lon)
    rows = telsem.compare(monthly, lat, lon, atlas, alat, alon)
    print(f"\n{'ch':>4} {'telsem':>7} {'n':>9} {'spearman':>9} {'pearson':>8} "
          f"{'bias':>7} {'rmse':>6}")
    for s in rows:
        print(f"{s['channel']:>4} {s['telsem']:>7} {s['n']:>9,} "
              f"{s['spearman_r']:>9.3f} {s['pearson_r']:>8.3f} "
              f"{s['bias']:>7.3f} {s['rmse']:>6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
