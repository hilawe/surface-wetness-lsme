"""Approximate geostationary viewing zenith angle (VZA) for the GridSat-B1 mosaic.

Ken Knapp's GridSat-B1 cloud product is stitched from the geostationary
constellation, and he suggested testing the tropical 85 GHz emissivity error
against geostationary VZA: the GridSat cloud detector over-flags at high VZA
(disk edges), so if the 85 GHz error falls at high VZA the mask is catching more
cirrus there, which would mean thin cirrus slips through near nadir and
contaminates 85 GHz. His delivered GRIDSAT-CLOUD files carry only irwin, cld, and
clr (no satid or VZA), so the VZA is derived here from the constellation geometry:
each pixel is assigned the smallest VZA across the sub-satellite longitudes (the
best-viewing satellite, which is roughly how the mosaic chooses). This is an
approximation; the exact per-pixel satellite id lives in the original GridSat-B1
files and could replace this later.
"""

import numpy as np

R_EARTH_KM = 6378.137
R_GEO_KM = 42164.0                                  # geostationary orbit radius

# Geostationary sub-satellite longitudes (degrees East) for the GridSat-B1 era
# around 1998: GOES-W, GOES-E, Meteosat-prime, Meteosat-IODC, GMS.
GEO_SUBLON_1998 = (-135.0, -75.0, 0.0, 63.0, 140.0)


def vza_from_sublon(lat, lon, sublon, r_earth=R_EARTH_KM, r_geo=R_GEO_KM):
    """Viewing zenith angle (degrees) at (lat, lon) from a geostationary satellite
    over the equator at sublon. NaN beyond that satellite's visible horizon.

    Uses the Earth-center / surface-point / satellite triangle: with central angle
    psi between the point and the sub-satellite point,
        sin(theta_z) = r_geo * sin(psi) / slant_range,
    valid until psi reaches arccos(r_earth / r_geo), the geometric horizon.
    """
    lat = np.deg2rad(np.asarray(lat, dtype=np.float64))
    dlon = np.deg2rad(np.asarray(lon, dtype=np.float64) - sublon)
    cos_psi = np.cos(lat) * np.cos(dlon)
    psi = np.arccos(np.clip(cos_psi, -1.0, 1.0))
    slant = np.sqrt(r_earth**2 + r_geo**2 - 2.0 * r_earth * r_geo * np.cos(psi))
    sin_tz = np.clip(r_geo * np.sin(psi) / slant, -1.0, 1.0)
    tz = np.degrees(np.arcsin(sin_tz))
    horizon = np.arccos(r_earth / r_geo)
    return np.where(psi <= horizon, tz, np.nan)


def min_geo_vza(lat, lon, sublons=GEO_SUBLON_1998):
    """Per-cell best (minimum) geostationary VZA over a constellation, on a grid.

    lat, lon are 1-D; returns an (nlat, nlon) array of the smallest VZA across the
    sub-satellite longitudes, NaN where no satellite in the set sees the cell.
    """
    lon2d, lat2d = np.meshgrid(np.asarray(lon, dtype=np.float64),
                               np.asarray(lat, dtype=np.float64))
    stack = np.stack([vza_from_sublon(lat2d, lon2d, s) for s in sublons], axis=0)
    allnan = np.all(np.isnan(stack), axis=0)
    out = np.full(lat2d.shape, np.nan)
    good = ~allnan
    out[good] = np.nanmin(stack[:, good], axis=0)
    return out
