"""Round-trip test for the monthly emissivity composite I/O (swi.io_lsme_monthly)."""

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from swi import io_lsme_monthly
from swi.channels import N_CHANNELS


def test_write_read_round_trip(tmp_path):
    lat = np.linspace(-89.875, 89.875, 8)
    lon = np.linspace(0.125, 359.875, 12)
    rng = np.random.default_rng(0)
    emis = rng.uniform(0.7, 0.98, size=(lat.size, lon.size, N_CHANNELS))
    emis[0, 0, :] = np.nan                       # a fully-missing cell
    n_obs = rng.integers(0, 31, size=emis.shape).astype(float)

    path = tmp_path / "LSME_emis_F13_199807.nc"
    io_lsme_monthly.write_monthly_emis(str(path), lat, lon, emis, n_obs,
                                       attrs={"title": "test"})

    rlat, rlon, remis, rn = io_lsme_monthly.read_monthly_emis(str(path))
    assert np.allclose(rlat, lat)
    assert np.allclose(rlon, lon)
    # float32 round-trip tolerance; NaN cell preserved
    finite = np.isfinite(emis)
    assert np.allclose(remis[finite], emis[finite], atol=1e-6)
    assert np.all(np.isnan(remis[0, 0, :]))
    assert np.array_equal(rn, np.nan_to_num(n_obs).astype(np.int32).astype(float))
