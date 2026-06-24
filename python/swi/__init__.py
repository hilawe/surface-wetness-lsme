"""LSME (land surface microwave emissivity) from microwave imagers and geostationary IR data.

A microwave + infrared land surface emissivity inversion (Prigent and Rossow
lineage) driven by two NOAA-stewarded climate data record inputs: the
Colorado State University SSM/I and SSMIS Brightness Temperature FCDR for the
microwave channels, and the NOAA GridSat-B1 cloud-cleared infrared product
(Knapp 2011) for the surface skin temperature and clear-sky mask.
"""

from . import channels  # noqa: F401

__all__ = ["channels"]
