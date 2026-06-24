# Land Surface Microwave Emissivity (LSME) from SSM/I and GridSat-B1

[![tests](https://github.com/hilawe/surface-wetness-lsme/actions/workflows/ci.yml/badge.svg)](https://github.com/hilawe/surface-wetness-lsme/actions/workflows/ci.yml)

Forward research toward a long-term land surface microwave emissivity record
derived from two NOAA-stewarded climate data record inputs. The microwave side
is the Colorado State University Special Sensor Microwave/Imager (SSM/I) and
Special Sensor Microwave Imager/Sounder (SSMIS) Brightness Temperature
Fundamental Climate Data Record. The infrared side is the NOAA GridSat-B1
cloud-cleared infrared product (Knapp 2011), which provides the independent
land surface skin temperature and the clear-sky mask. The two are combined
through the microwave + infrared emissivity inversion of Prigent, Rossow,
Aires, and colleagues, with a first-order ERA5 single-levels atmospheric
correction, validated against the TELSEM2 climatology (Aires et al. 2011).

The work also implements a second, independent estimator of surface wetness
from the same GridSat-B1 record: the thermal-inertia diurnal infrared range,
which damps over wet or vegetated ground. The two estimators are combined
into a relative wetness product, and an anomaly form removes the static
surface-type pattern to leave the time-varying signal that a monitoring
climate record reports.

A sustained, observational, multi-decadal LSME record released as a
monitored climate variable does not exist (Hu et al. 2026 acknowledges the
gap in the nearest competitor's own words; see also Aires et al. 2011 and
Prigent et al. 2003 for the method and atlas). The defensible wedge is
sustained NOAA stewardship and the use of two NOAA-controlled CDR inputs.

## Status

Forward research, paper in preparation. The pipeline runs end to end on the
full 1998 F-13 annual cycle. Monthly clear-sky emissivity composites
validate against the TELSEM2 monthly climatology with window and
horizontal-polarization Pearson correlations of 0.83 to 0.94 across all
four seasons; the anomaly form, with a TELSEM seasonal-amplitude floor that
masks hyper-arid no-signal regions, reaches a 37 GHz horizontal-polarization
per-cell temporal anomaly correlation median of 0.818. A two-estimator
wetness product is written as a CF-compliant NetCDF and shows the expected
seasonal agreement between the microwave polarization difference and the
GridSat-B1 thermal-inertia diurnal range.

## Principal Investigator and collaboration

Hilawe Semunegus (NOAA National Centers for Environmental Information) is
the Principal Investigator. The GridSat-B1 infrared cloud product is
provided and maintained by Ken Knapp (formerly NOAA NCEI), and the 
GridSat-B1 infrared time series. The machine learning cloud mask was 
developed Ken Knapp, Hilawe Semunegus, Ken Knapp and Xuepeng (Tom) Zhao 
in an Earth Science Data Science course at the North Carolina 
Institute for Climate Studies (NCICS).

The multi-decadal extension awaits additional GridSat-B1 years and the
inter-satellite drift handling that a multi-satellite anomaly requires.

## What is here

- **Single isothermal-layer clear-sky emissivity inversion**
  (`python/swi/lsme.py`): `e = (Tb - A - tau A) / (tau (Ts - A))` with
  `A = Ta (1 - tau)` and three plug-in points: skin temperature, clear-sky
  mask, and column water vapor.
- **TELSEM2 atlas reader** (`python/swi/telsem.py`): ports `equare` and
  `get_coordinates` from `mod_mwatlas_m2.F90` to rasterize the equal-area
  ASCII atlas onto a regular grid, plus a per-channel comparison harness
  built on `swi.validate`.
- **Thermal-inertia estimator** (`python/swi/thermal.py`): per-cell
  clear-sky diurnal infrared range from the eight GridSat-B1 3-hourly slots,
  with a diurnal-bracketing quality filter.
- **GridSat-B1 reader with the ML cloud mask** (`python/swi/io_gridsat.py`):
  flat `GRIDSAT-CLOUD` and `ML-CLOUD` layouts; reads the cloud-cleared
  infrared skin temperature, an approximate per-pixel geostationary view
  angle when needed, and the overpass-time composite for a given date.
- **Anomaly form** (`python/swi/anomaly.py`): per-cell temporal anomaly
  (month minus the cell's annual mean, shape-agnostic over channels);
  spatial pattern skill per time step and pooled; block coarsening; a
  TELSEM seasonal-amplitude no-signal mask.
- **Two-estimator relative wetness product**
  (`python/swi/two_estimator.py`, `python/scripts/make_two_estimator.py`):
  rank-normalized combination of the thermal-inertia range and the 19 GHz
  V minus H polarization difference, with an agreement field, written as a
  CF NetCDF file.
- **First-order ERA5 atmospheric correction**
  (`python/swi/io_era5_atmos.py`): per-longitude pick of the synoptic time
  nearest the morning overpass, regridded to the F-13 grid for column
  water vapor, two-meter temperature, and skin temperature.
- **Minimal shared core** (`python/swi/channels.py`,
  `python/swi/io_csu_grid.py`, `python/swi/validate.py`): the SSM/I channel
  conventions, the CSU FCDR-GRID reader, and the skill metrics, copied from
  the sibling Basist revival repository for self-containment.

## Running the tests

```bash
pip install -r requirements-test.txt
cd python && python -m pytest tests
```

Tests that need third-party reference data skip cleanly when those files
are not present, so the suite runs on any host with the scientific Python
stack.

## Sibling project

The original Basist Surface Wetness Index, the empirical decision-tree
predecessor to this work, is at
[github.com/hilawe/surface-wetness-index](https://github.com/hilawe/surface-wetness-index).
The two repositories share a minimal core of channel and grid conventions
but otherwise stand alone.

## References

- Aires, F., C. Prigent, F. Bernardo, C. Jimenez, R. Saunders, and P. Brunel
  (2011). A Tool to Estimate Land-Surface Emissivities at Microwave
  frequencies (TELSEM) for use in numerical weather prediction. Quarterly
  Journal of the Royal Meteorological Society 137(656), 690 to 699.
- Berg, W. and C. Kummerow (2018). Multisatellite intercalibrated CDR of
  brightness temperatures. Remote Sensing 10(8), 1306.
- Hu, J., et al. (2026). A harmonized multi-sensor land surface emissivity
  framework from GMI, AMSR2, and FY-3 MWRI with geostationary cloud masks.
  Remote Sensing of Environment 334, 115169.
- Knapp, K. R. (2011). Globally Gridded Satellite (GridSat) Observations
  for Climate Studies. Journal of Climate 24(20), 5275 to 5290.
- Prigent, C., F. Aires, and W. B. Rossow (2003). Land surface microwave
  emissivities over the globe for a decade. Bulletin of the American
  Meteorological Society 84(11), 1573 to 1584.

## License

This work was prepared by a U.S. Government employee as part of official
duties and is in the public domain in the United States (17 U.S.C. 105). To
the extent any rights exist, they are dedicated to the public domain under
Creative Commons Zero 1.0 (see `LICENSE`).
