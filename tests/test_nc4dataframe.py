import math
from pathlib import Path

import pytest

import numpy as np
np.random.seed(5446)
import pandas as pd

from netCDF4 import Dataset
from hyncu import nc4io, nc4dataframe

import warnings

FHERE = Path(__file__).resolve().parent


def test_dataframe(allclose):
    fnc = FHERE / "test_dataframe.nc"
    with Dataset(fnc, "w") as nc:
        sites = ["a", "b", "c"]
        variables = ["v1", "v2", "v3", "v4"]
        times = pd.date_range("1990-01-01", "2000-12-31", freq="D")
        dataname = "truc"
        nc4dataframe.set_netcdf_file(nc, siteids=sites, \
                    variables=variables, times=times, dataname=dataname)

        size = (len(sites), len(times), len(variables))
        nc[dataname][:] = np.random.uniform(-1, 1, size=size)

    with Dataset(fnc, "r") as nc:
        df = nc4dataframe.get_dataframe(nc, "a", dataname)
        assert df.shape == (len(times), len(variables))

    fnc.unlink()

