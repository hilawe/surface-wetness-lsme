"""CSU ICDR-GRID reader test.

Runs only when a sample file is available (large binary, not in the repo). Point
the SWI_CSU_SAMPLE environment variable at a CSU ICDR-GRID .nc file to enable it:

    SWI_CSU_SAMPLE=/path/to/CSU_..._ICDR-GRID_..._D20260101.nc pytest
"""

import os

import numpy as np
import pytest

SAMPLE = os.environ.get("SWI_CSU_SAMPLE")
pytestmark = pytest.mark.skipif(
    not (SAMPLE and os.path.exists(SAMPLE)),
    reason="set SWI_CSU_SAMPLE to a CSU ICDR-GRID file to run",
)


def test_evaluate_file_shapes_and_physical_ranges():
    netCDF4 = pytest.importorskip("netCDF4")  # noqa: F841
    from swi import io_csu_grid as io

    with np.errstate(divide="ignore", invalid="ignore"):
        r = io.evaluate_file(SAMPLE, pass_="dsc")

    nlat, nlon = r["lat"].size, r["lon"].size
    assert r["wet"].shape == (nlat, nlon)
    assert r["temp"].shape == (nlat, nlon)
    # some valid retrievals exist
    assert r["valid"].sum() > 0
    # wetness index stays within the expected band where defined
    w = r["wet"][np.isfinite(r["wet"]) & (r["wet"] >= 0)]
    assert w.size > 0
    assert w.max() <= 100.5
    # valid land skin temperatures are physical
    t = r["temp"][r["temp"] > -90]
    assert t.size > 0
    assert 200.0 < t.min() and t.max() < 360.0
