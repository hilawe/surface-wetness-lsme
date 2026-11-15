"""Boolean land mask for a latitude and longitude grid.

This is a module of its own, rather than part of the grid utilities, because
it is shared by published subsets of this repository that ship different sets
of modules and only some of them carry the grid engine. While the function
lived among the grid utilities, a subset that excludes those could not import
it at all, and that subset failed for anyone who cloned it even though every
test in this repository passed. The function reads nothing but the two
coordinate axes and the `global_land_mask` package, so it was never
grid-engine code.

`swi.grids.land_mask` re-exports it, so existing call sites are unaffected.
"""

import numpy as np


def land_mask(lat, lon):
    """Boolean (nlat, nlon) land mask for a grid given 1-D lat and lon.

    Accepts lon in 0..360 or -180..180. Uses global_land_mask if available and
    falls back to all-True (every cell treated as land) if it is not installed,
    so callers that require a real mask should check for an all-True result.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    lon2d, lat2d = np.meshgrid(lon180, lat)
    try:
        from global_land_mask import globe
        return globe.is_land(lat2d, lon2d)
    except Exception:
        return np.ones((lat.size, lon.size), dtype=bool)
