"""Two-estimator wetness anomaly product: the time-varying wetness signal.

Reads the monthly two-estimator products (SWI_two_estimator_F13_YYYYMM.nc, from
make_two_estimator), stacks the combined wetness index and its two components,
removes each cell's mean over the available months, and writes the anomaly stack
as a CF-1.8 NetCDF with a time axis plus the climatology (the per-cell mean over
the months). This converts the relative monthly wetness index into a time-varying
signal: where and when a cell is wetter or drier than its own annual mean.

Honest scope is unchanged from the static product. It is a relative
wetness-and-vegetation signal, not absolute soil moisture, and is most meaningful
over low vegetation. With a single year the anomaly is the seasonal cycle; across
years the same operation gives the interannual wetness anomaly that a monitoring
record reports.

Usage:
    python -m scripts.make_anomaly_product --in-dir ../scratch/two_estimator \\
        --out ../scratch/anomaly
"""

import calendar
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swi import anomaly, telsem

VARS = ("combined_wetness", "thermal_wetness", "microwave_wetness")


def opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def read_wetness(path):
    """Read the three 0-to-1 wetness fields and the grid from a monthly product."""
    import netCDF4

    ds = netCDF4.Dataset(path)
    try:
        ds.set_auto_mask(False)
        lat = np.asarray(ds.variables["lat"][:], np.float64)
        lon = np.asarray(ds.variables["lon"][:], np.float64)
        fields = {k: np.asarray(ds.variables[k][:], np.float64) for k in VARS}
    finally:
        ds.close()
    return lat, lon, fields


def mid_month_days_since(year, month, ref_year=1998):
    """Days from ref_year-01-01 to the 15th of the given month (CF time axis)."""
    days = 0
    for y in range(ref_year, year):
        days += 366 if calendar.isleap(y) else 365
    days += sum(calendar.monthrange(year, m)[1] for m in range(1, month)) + 14
    return days


def write_anomaly_product(path, lat, lon, times, ref_year, anomalies, clim, labels):
    """Write the time-resolved wetness anomaly product as a CF-1.8 NetCDF."""
    import netCDF4

    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        ds.createDimension("time", len(times))
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        vt = ds.createVariable("time", "f8", ("time",))
        vt.units = f"days since {ref_year}-01-01 00:00:00"
        vt.calendar = "standard"; vt.standard_name = "time"; vt[:] = times
        vlat = ds.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"; vlat.standard_name = "latitude"; vlat[:] = lat
        vlon = ds.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"; vlon.standard_name = "longitude"; vlon[:] = lon
        for k in VARS:
            va = ds.createVariable(k + "_anomaly", "f4", ("time", "lat", "lon"),
                                   fill_value=np.float32(np.nan), zlib=True)
            va.units = "1"
            va.long_name = (f"{k.replace('_', ' ')} anomaly "
                            "(month minus per-cell mean over the available months)")
            va[:] = anomalies[k].astype(np.float32)
            vc = ds.createVariable(k + "_climatology", "f4", ("lat", "lon"),
                                   fill_value=np.float32(np.nan), zlib=True)
            vc.units = "1"
            vc.long_name = f"{k.replace('_', ' ')} per-cell mean over the months"
            vc[:] = clim[k].astype(np.float32)
        ds.title = ("Two-estimator relative land surface wetness anomaly "
                    "(thermal + microwave)")
        ds.summary = (
            "Monthly anomaly of a relative wetness index from two independent "
            "GridSat-B1-anchored estimators (clear-sky diurnal infrared range and "
            "the 19 GHz V-H microwave emissivity polarization difference), each "
            "minus its per-cell mean over the available months. Relative wetness "
            "and vegetation, not absolute soil moisture; most meaningful over low "
            "vegetation. With one year the anomaly is the seasonal cycle.")
        ds.institution = "NOAA National Centers for Environmental Information"
        ds.source = "CSU SSM/I FCDR (F-13) and GridSat-B1 (Knapp), F-13 1998"
        ds.creator_name = "Hilawe Semunegus"
        ds.contributor_name = "Kenneth R. Knapp (GridSat-B1)"
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.month_labels = " ".join(labels)
    finally:
        ds.close()
    return path


def roll180(a, lon):
    return np.roll(a, lon.size // 2, axis=-1)


def fig_quicklook(anom, land, lat, lon, labels, out):
    """Combined-wetness anomaly maps for up to six months."""
    n = min(len(labels), 6)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.2 * rows), squeeze=False)
    fig.suptitle("Combined relative-wetness anomaly, F-13 1998 "
                 "(blue wetter, red drier than the cell's annual mean)", fontsize=13)
    masked = np.where(land[None, :, :], anom, np.nan)
    lim = np.nanpercentile(np.abs(masked), 98)
    ext = (-180, 180, lat[0], lat[-1])
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n:
            ax.axis("off"); continue
        im = ax.imshow(roll180(masked[i], lon), origin="lower", extent=ext,
                       cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(labels[i], fontsize=10)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, "two_estimator_anomaly.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p)


def main(argv):
    in_dir = opt(argv, "--in-dir", "../scratch/two_estimator")
    out = opt(argv, "--out", "../scratch/anomaly")
    os.makedirs(out, exist_ok=True)

    found = []
    for p in sorted(glob.glob(os.path.join(in_dir, "SWI_two_estimator_F13_??????.nc"))):
        m = re.search(r"_(\d{4})(\d{2})\.nc$", p)
        found.append((int(m.group(1)), int(m.group(2)), p))
    if len(found) < 2:
        print(f"need >= 2 monthly two-estimator products in {in_dir}; found {len(found)}")
        return 1
    labels = [f"{y}-{m:02d}" for y, m, _ in found]
    print(f"months ({len(found)}): " + ", ".join(labels))

    lat = lon = None
    stacks = {k: [] for k in VARS}
    for _, _, p in found:
        lat, lon, fields = read_wetness(p)
        for k in VARS:
            stacks[k].append(fields[k])

    anomalies, clim = {}, {}
    for k in VARS:
        s = np.stack(stacks[k], axis=0)
        clim[k] = anomaly.temporal_mean(s)
        anomalies[k] = anomaly.temporal_anomaly(s)

    ref_year = found[0][0]
    times = [mid_month_days_since(y, m, ref_year) for y, m, _ in found]
    ym0 = f"{found[0][0]}{found[0][1]:02d}"
    ym1 = f"{found[-1][0]}{found[-1][1]:02d}"
    nc = os.path.join(out, f"SWI_two_estimator_anomaly_F13_{ym0}_{ym1}.nc")
    write_anomaly_product(nc, lat, lon, times, ref_year, anomalies, clim, labels)
    print("wrote", nc)

    land = telsem.land_mask(lat, lon)
    a = anomalies["combined_wetness"]
    valid = np.isfinite(a) & land[None, :, :]
    print(f"combined-wetness anomaly: {int(valid.sum()):,} valid (month, land cell) "
          f"values; range {np.nanmin(a[valid]):+.3f}..{np.nanmax(a[valid]):+.3f}")
    fig_quicklook(a, land, lat, lon, labels, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
