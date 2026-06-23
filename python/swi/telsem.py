"""TELSEM microwave land-surface emissivity climatology: Step 3 validation.

TELSEM (Aires, Prigent et al. 2011, QJRMS) and TELSEM2 are monthly-mean SSM/I
land surface emissivity atlases (1993 to 2004), the primary reference for the
LSME emissivity derived in swi.lsme. July 1998 sits inside the TELSEM window, so
the derived July emissivity can be checked against the TELSEM July climatology
per channel.

This module provides the comparison machinery (Basist channel to TELSEM
frequency and polarization mapping, a land mask, and a per-channel skill table
built on swi.validate), plus a synthetic atlas so the machinery runs and is
tested today. The one remaining real-data step is load_atlas(), which reads the
actual NWP-SAF TELSEM2 atlas; its source and format are documented there.

TELSEM provides emissivity at the SSM/I frequencies natively, so all seven
Basist channels map directly (no 85-to-91 GHz issue for F-13, which is true
85.5 GHz).
"""

import os

import numpy as np

from .channels import N_CHANNELS, CHANNEL_NAMES
from . import validate

# Basist order P1..P7 = 19V 19H 22V 37V 37H 85V 85H mapped to TELSEM
# (frequency GHz, polarization). SSM/I samples 22 GHz at V only, which matches.
BASIST_TO_TELSEM = [
    (19.35, "V"),
    (19.35, "H"),
    (22.235, "V"),
    (37.0, "V"),
    (37.0, "H"),
    (85.5, "V"),
    (85.5, "H"),
]


def land_mask(lat, lon):
    """Boolean (nlat, nlon) land mask for a grid given 1-D lat and lon.

    Accepts lon in 0..360 or -180..180. Uses global_land_mask if available and
    falls back to all-True (every cell treated as land) if it is not installed.
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


def synthetic_atlas(lat, lon, month=7, seed=0):
    """A TELSEM-like emissivity atlas (nlat, nlon, 7) for testing the machinery.

    NOT real TELSEM. Land emissivity is high (V > H, gently falling with
    frequency) with smooth spatial structure; ocean is low. Use only to exercise
    and test compare(); replace with load_atlas() for real validation.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    rng = np.random.default_rng(seed + month)
    land = land_mask(lat, lon)
    nlat, nlon = lat.size, lon.size

    # smooth spatial term in 0..1 (drier toward subtropics)
    lat2d = np.repeat(lat[:, None], nlon, axis=1)
    smooth = 0.5 * (1.0 + np.cos(np.deg2rad(2.0 * lat2d)))

    emis = np.empty((nlat, nlon, N_CHANNELS), dtype=np.float64)
    for c, (freq, pol) in enumerate(BASIST_TO_TELSEM):
        fdrop = 0.01 * (freq - 19.35) / 66.15        # gentle fall with frequency
        if pol == "V":
            land_e = 0.955 - fdrop + 0.02 * smooth
            ocean_e = 0.50
        else:
            land_e = 0.915 - fdrop + 0.03 * smooth
            ocean_e = 0.32
        field = np.where(land, land_e, ocean_e)
        field = field + rng.normal(0.0, 0.004, size=field.shape)
        emis[:, :, c] = np.clip(field, 0.2, 1.0)
    return emis


# --- Real NWP-SAF TELSEM2 atlas -------------------------------------------------
#
# The 12 monthly files `ssmi_mean_emis_climato_<MM>_cov_interpol_M2` (ASCII) plus
# the Fortran module are obtained by extracting `telsem2_mw_atlas.tar.bz2`, a free
# direct download (no login) from the EUMETSAT NWP-SAF:
#   https://nwp-saf.eumetsat.int/downloads/emis_data/telsem2_mw_atlas.tar.bz2
# Place the extracted files under data/telsem2/. Each cell's position is encoded
# in its cell number on the TELSEM2 equal-area grid; read_telsem_ascii ports the
# equare() / get_coordinates() mapping from mod_mwatlas_m2.F90. The SSM/I atlas
# carries exactly the seven Basist channels (19V 19H 22V 37V 37H 85V 85H).


def resample_cells_to_grid(cell_lat, cell_lon, cell_emis, dst_lat, dst_lon):
    """Rasterize atlas cells (each a center lat/lon + 7-channel emissivity) onto a
    regular (dst_lat, dst_lon) grid by nearest grid index.

    Exact when the atlas and target are both about 0.25 degree. Cells landing in
    one target cell are averaged; empty target cells are NaN. dst_lat/dst_lon must
    be regular ascending; lon in 0..360.
    """
    cell_lat = np.asarray(cell_lat, dtype=np.float64)
    cell_lon = np.asarray(cell_lon, dtype=np.float64) % 360.0
    cell_emis = np.asarray(cell_emis, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64)
    nlat, nlon, nch = dst_lat.size, dst_lon.size, cell_emis.shape[1]
    dlat = (dst_lat[-1] - dst_lat[0]) / (nlat - 1)
    dlon = (dst_lon[-1] - dst_lon[0]) / (nlon - 1)

    ilat = np.clip(np.rint((cell_lat - dst_lat[0]) / dlat).astype(int), 0, nlat - 1)
    ilon = np.clip(np.rint((cell_lon - dst_lon[0]) / dlon).astype(int), 0, nlon - 1)
    flat = ilat * nlon + ilon

    summ = np.zeros((nlat * nlon, nch), dtype=np.float64)
    cnt = np.zeros(nlat * nlon, dtype=np.float64)
    np.add.at(summ, flat, cell_emis)
    np.add.at(cnt, flat, 1.0)
    out = np.full((nlat * nlon, nch), np.nan, dtype=np.float64)
    nz = cnt > 0
    out[nz] = summ[nz] / cnt[nz, None]
    return out.reshape(nlat, nlon, nch)


def inspect_atlas_header(path, n=3):
    """Return the first n lines of an atlas file, to confirm the column layout."""
    with open(path) as fh:
        return [next(fh).rstrip("\n") for _ in range(n)]


def telsem2_grid(dlat=0.25):
    """Cells-per-band and 1-based first-cell index for the TELSEM2 equal-area grid.

    Ported from equare() in mod_mwatlas_m2.F90: each 0.25 degree latitude band
    holds a number of longitude cells proportional to the band area, so cells are
    roughly equal-area. Returns (ncells, firstcell), each length 720.
    """
    maxlat = int(180.0 / dlat)
    rearth, pi = 6371.2, np.pi
    aecell = (2 * pi * rearth * (rearth * np.sin(dlat * pi / 180.0))) * dlat / 360.0
    half = maxlat // 2
    lat = np.arange(1, half + 1)
    htb = rearth * np.sin(2 * pi * ((lat - 1) * dlat) / 360.0)
    hte = rearth * np.sin(2 * pi * (lat * dlat) / 360.0)
    icellr = np.floor((2 * pi * rearth * (hte - htb)) / aecell + 0.5).astype(np.int64)
    ncells = np.zeros(maxlat, dtype=np.int64)
    ncells[(lat + half) - 1] = icellr            # northern bands
    ncells[(half + 1 - lat) - 1] = icellr         # southern bands (mirror)
    firstcell = np.empty(maxlat, dtype=np.int64)
    firstcell[0] = 1
    firstcell[1:] = 1 + np.cumsum(ncells)[:-1]
    return ncells, firstcell


def cellnum_to_latlon(cellnum, ncells, firstcell, dlat=0.25):
    """Map TELSEM2 cell numbers to (lat, lon) centers (ported from get_coordinates)."""
    cellnum = np.asarray(cellnum, dtype=np.int64)
    band = np.clip(np.searchsorted(firstcell, cellnum, side="right"), 1, ncells.size)
    lat = (band - 0.5) * dlat - 90.0
    ilon = cellnum - firstcell[band - 1] + 1
    lon = (ilon - 0.5) * (360.0 / ncells[band - 1])
    return lat, lon


def read_telsem_ascii(path, dlat=0.25):
    """Parse a TELSEM2 ASCII atlas file to (cell_lat, cell_lon, cell_emis[N,7]).

    Format (per mod_mwatlas_m2.F90): line 1 is the cell count; each data line is
    ``cellnum e1..e7 std1..std7 class1 class2`` with the position encoded in the
    cell number on the equal-area grid (there is no explicit lat or lon). Channels
    are the seven SSM/I emissivities in Basist order 19V 19H 22V 37V 37H 85V 85H.
    """
    ncells, firstcell = telsem2_grid(dlat)
    nums, emis = [], []
    with open(path) as fh:
        next(fh)                                  # header line: number of cells
        for line in fh:
            tok = line.split()
            if len(tok) < 1 + N_CHANNELS:
                continue
            nums.append(int(tok[0]))
            emis.append([float(t) for t in tok[1:1 + N_CHANNELS]])
    if not emis:
        raise ValueError(f"no data rows parsed from {path}")
    lat, lon = cellnum_to_latlon(np.asarray(nums, dtype=np.int64),
                                 ncells, firstcell, dlat)
    return lat, lon, np.asarray(emis, dtype=np.float64)


def load_atlas(path, month, lat=None, lon=None):
    """Load the real NWP-SAF TELSEM2 monthly atlas onto the (lat, lon) grid.

    path is a directory holding `ssmi_mean_emis_climato_<MM>_cov_interpol_M2`
    (or the file itself). Returns (lat, lon, emis[nlat, nlon, 7]) in Basist order.
    Raises FileNotFoundError with acquisition guidance if the file is absent.
    """
    import glob

    if lat is None or lon is None:
        raise ValueError("load_atlas needs the target lat and lon grid")

    fpath = path
    if path is None or os.path.isdir(str(path)):
        base = path or os.path.join("..", "data", "telsem2")
        hits = glob.glob(os.path.join(base, f"ssmi_mean_emis_climato_{month:02d}_*"))
        fpath = hits[0] if hits else None
    if not fpath or not os.path.exists(fpath):
        raise FileNotFoundError(
            f"TELSEM2 atlas month {month:02d} not found under data/telsem2/. "
            "Extract telsem2_mw_atlas.tar.bz2 (free direct download from "
            "https://nwp-saf.eumetsat.int/downloads/emis_data/telsem2_mw_atlas.tar.bz2) "
            "into data/telsem2/.")
    cell_lat, cell_lon, cell_emis = read_telsem_ascii(fpath)
    emis = resample_cells_to_grid(cell_lat, cell_lon, cell_emis, lat, lon)
    return np.asarray(lat), np.asarray(lon), emis


def compare(ours, ours_lat, ours_lon, atlas, atlas_lat, atlas_lon, land=None):
    """Per-channel skill of the derived emissivity against a TELSEM atlas.

    ours  : (nlat, nlon, 7) derived emissivity on (ours_lat, ours_lon).
    atlas : (alat, alon, 7) TELSEM emissivity on (atlas_lat, atlas_lon); regridded
            to the ours grid per channel by nearest neighbor.
    land  : optional (nlat, nlon) bool mask; defaults to the land mask of the
            ours grid. Comparison runs over finite, land cells common to both.

    Returns a list of per-channel dicts: channel, telsem freq/pol, and the
    validate.skill_scores output (n, pearson_r, spearman_r, bias, rmse), where
    bias is (ours - telsem).
    """
    ours = np.asarray(ours, dtype=np.float64)
    atlas = np.asarray(atlas, dtype=np.float64)
    if ours.shape[-1] != N_CHANNELS or atlas.shape[-1] != N_CHANNELS:
        raise ValueError(f"both fields need {N_CHANNELS} channels")
    if land is None:
        land = land_mask(ours_lat, ours_lon)

    rows = []
    for c, (freq, pol) in enumerate(BASIST_TO_TELSEM):
        ref = validate.regrid_nearest(
            np.asarray(atlas_lat, float), np.asarray(atlas_lon, float),
            atlas[:, :, c], np.asarray(ours_lat, float),
            np.asarray(ours_lon, float))
        m = validate.common_valid(ours[:, :, c], ref, land=land)
        sc = validate.skill_scores(ours[:, :, c][m], ref[m])
        rows.append({"channel": CHANNEL_NAMES[c],
                     "telsem": f"{freq:g}{pol}", **sc})
    return rows
