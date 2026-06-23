"""Tests for the TELSEM Step 3 validation machinery."""

import os

import numpy as np
import pytest

from swi import telsem
from swi.channels import N_CHANNELS


LAT = np.linspace(-58.0, 58.0, 40)
LON = np.linspace(0.0, 357.0, 80)
REAL_ATLAS = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                          "telsem2", "ssmi_mean_emis_climato_07_cov_interpol_M2")


def test_channel_map_covers_all_seven_at_ssmi_freqs():
    assert len(telsem.BASIST_TO_TELSEM) == N_CHANNELS
    freqs = [f for f, _ in telsem.BASIST_TO_TELSEM]
    assert set(freqs) == {19.35, 22.235, 37.0, 85.5}
    # 22 GHz is V-only in SSM/I, matching our single 22V channel
    assert telsem.BASIST_TO_TELSEM[2] == (22.235, "V")


def test_compare_perfect_when_ours_equals_atlas():
    atlas = telsem.synthetic_atlas(LAT, LON, month=7, seed=1)
    rows = telsem.compare(atlas, LAT, LON, atlas, LAT, LON)
    assert len(rows) == N_CHANNELS
    for s in rows:
        assert s["n"] > 50                       # land cells present
        assert s["spearman_r"] > 0.99
        assert s["pearson_r"] > 0.99
        assert abs(s["bias"]) < 1e-9
        assert s["rmse"] < 1e-9


def test_compare_recovers_a_known_bias():
    atlas = telsem.synthetic_atlas(LAT, LON, month=7, seed=2)
    ours = atlas + 0.05                          # uniform +0.05 offset
    rows = telsem.compare(ours, LAT, LON, atlas, LAT, LON)
    for s in rows:
        assert s["bias"] == pytest.approx(0.05, abs=1e-9)
        assert s["pearson_r"] > 0.99             # offset does not change pattern


def test_compare_degrades_with_noise():
    rng = np.random.default_rng(3)
    atlas = telsem.synthetic_atlas(LAT, LON, month=7, seed=3)
    ours = atlas + rng.normal(0.0, 0.05, size=atlas.shape)
    rows = telsem.compare(ours, LAT, LON, atlas, LAT, LON)
    for s in rows:
        assert 0.0 < s["pearson_r"] < 0.99       # correlated but not perfect
        assert s["rmse"] > 0.01


def test_resample_cells_to_grid_places_cells_correctly():
    dst_lat = np.array([-1.0, 0.0, 1.0])      # dlat = 1
    dst_lon = np.array([0.0, 1.0, 2.0])       # dlon = 1
    cell_lat = np.array([0.0, -1.0])
    cell_lon = np.array([1.0, 2.0])
    cell_emis = np.tile(np.array([0.5]), (2, telsem.N_CHANNELS))
    cell_emis[1, :] = 0.8
    grid = telsem.resample_cells_to_grid(cell_lat, cell_lon, cell_emis,
                                         dst_lat, dst_lon)
    assert grid.shape == (3, 3, telsem.N_CHANNELS)
    assert np.allclose(grid[1, 1, :], 0.5)    # cell 0 -> (lat 0, lon 1)
    assert np.allclose(grid[0, 2, :], 0.8)    # cell 1 -> (lat -1, lon 2)
    assert np.isnan(grid[2, 0, 0])            # untouched cell


def test_telsem2_equal_area_grid():
    ncells, firstcell = telsem.telsem2_grid(0.25)
    assert ncells.size == 720 and firstcell.size == 720
    assert firstcell[0] == 1
    assert np.array_equal(ncells, ncells[::-1])        # symmetric about equator
    assert ncells[0] < ncells[360]                     # pole band << equatorial
    assert 650000 < int(ncells.sum()) < 665000         # ~660066 total cells


def test_cellnum_to_latlon_band1():
    ncells, firstcell = telsem.telsem2_grid(0.25)
    lat, lon = telsem.cellnum_to_latlon(np.array([1]), ncells, firstcell)
    assert np.isclose(lat[0], -89.875)                 # cell 1 is the south-pole row
    assert 0.0 <= lon[0] < 360.0


def test_read_telsem_ascii_real_format(tmp_path):
    f = tmp_path / "ssmi_mean_emis_climato_07_cov_interpol_M2"
    # header (cell count), then `cellnum e1..e7 std1..std7 class1 class2`
    f.write_text(
        "2\n"
        "52 0.95 0.91 0.93 0.94 0.90 0.88 0.86 "
        "0.001 0.001 0.001 0.001 0.001 0.001 0.001 6 21\n"
        "53 0.96 0.92 0.94 0.95 0.91 0.89 0.87 "
        "0.001 0.001 0.001 0.001 0.001 0.001 0.001 6 21\n")
    lat, lon, emis = telsem.read_telsem_ascii(str(f))
    assert emis.shape == (2, telsem.N_CHANNELS)
    assert np.isclose(emis[0, 0], 0.95) and np.isclose(emis[1, -1], 0.87)
    assert np.all((lat >= -90) & (lat <= 90))
    assert np.all((lon >= 0) & (lon < 360))


@pytest.mark.skipif(not os.path.exists(REAL_ATLAS), reason="TELSEM2 atlas not staged")
def test_real_telsem2_july_atlas():
    lat, lon, emis = telsem.read_telsem_ascii(REAL_ATLAS)
    assert lat.size > 200000                           # ~233945 land cells
    assert np.all((lat >= -90) & (lat <= 90))
    assert np.all((lon >= 0) & (lon < 360))
    med = np.median(emis, axis=0)
    assert 0.85 < med[0] < 0.98                        # 19V land emissivity
    assert med[0] > med[1]                             # V > H at 19 GHz


def test_load_atlas_missing_file_raises_with_guidance():
    lat = np.linspace(-89, 89, 10)
    lon = np.linspace(0, 357, 10)
    with pytest.raises(FileNotFoundError):
        telsem.load_atlas("/no/such/telsem2/dir", 7, lat=lat, lon=lon)
    with pytest.raises(ValueError):
        telsem.load_atlas("/whatever", 7)          # no grid given
