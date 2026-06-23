"""Read and write the monthly-mean LSME emissivity composite.

The monthly composite emissivity (per channel, on the F-13 0.25 degree grid) is
the intermediate the anomaly form builds on: stacking the monthly composites for
a year and removing each cell's annual mean gives the emissivity anomaly. The
monthly_lsme driver writes one file per month with --save-emis, and
anomaly_validation reads the stack back. The file is a small CF-flavored NetCDF
holding an (lat, lon, channel) emissivity array, the per-channel valid-day count,
and the channel names, so the anomaly stage never has to recompute the composite.
"""

import numpy as np

from .channels import CHANNEL_NAMES, N_CHANNELS


def write_monthly_emis(path, lat, lon, emis, n_obs, attrs=None):
    """Write a monthly emissivity composite to a NetCDF file.

    emis is (nlat, nlon, 7), n_obs is the matching per-channel valid-day count.
    Emissivity is stored as float32 with NaN fill; n_obs as int32.
    """
    import netCDF4

    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    emis = np.asarray(emis, np.float64)
    n_obs = np.asarray(n_obs, np.float64)
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        ds.createDimension("channel", N_CHANNELS)
        vlat = ds.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"; vlat.standard_name = "latitude"; vlat[:] = lat
        vlon = ds.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"; vlon.standard_name = "longitude"; vlon[:] = lon
        ve = ds.createVariable("emissivity", "f4", ("lat", "lon", "channel"),
                               fill_value=np.float32(np.nan), zlib=True)
        ve.long_name = "monthly-mean clear-sky land surface emissivity"
        ve.units = "1"
        ve[:] = emis.astype(np.float32)
        vn = ds.createVariable("n_obs", "i4", ("lat", "lon", "channel"), zlib=True)
        vn.long_name = "clear days contributing to the monthly mean"
        vn[:] = np.nan_to_num(n_obs).astype(np.int32)
        ds.channel_names = " ".join(CHANNEL_NAMES)
        for key, val in (attrs or {}).items():
            setattr(ds, key, val)
    finally:
        ds.close()
    return path


def read_monthly_emis(path):
    """Read a monthly emissivity composite. Returns (lat, lon, emis, n_obs)."""
    import netCDF4

    ds = netCDF4.Dataset(path)
    try:
        ds.set_auto_mask(False)
        lat = np.asarray(ds.variables["lat"][:], np.float64)
        lon = np.asarray(ds.variables["lon"][:], np.float64)
        emis = np.asarray(ds.variables["emissivity"][:], np.float64)
        n_obs = np.asarray(ds.variables["n_obs"][:], np.float64)
    finally:
        ds.close()
    return lat, lon, emis, n_obs
