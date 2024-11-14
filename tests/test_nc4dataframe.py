import math
from pathlib import Path

import pytest
from string import ascii_letters as letters

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
        stations = ["a", "b", "c"]
        variables = ["v1", "v2", "v3", "v4"]
        times = pd.date_range("1990-01-01", "2000-12-31", freq="D")
        dataname = "truc"
        units = ["mm.day-1", "m3.s-1", "degC", "kg"]
        nc4dataframe.configure_netcdf_file(nc, stationids=stations, \
                    variables=variables, \
                    units=units, \
                    times=times, \
                    dataname=dataname)

        assert "time" in nc.dimensions
        assert "stationid" in nc.dimensions
        assert f"{dataname}_variable" in nc.dimensions
        assert f"{dataname}_variable_units" in nc.variables
        assert dataname in nc.variables

        size = (len(stations), len(times), len(variables))
        nc[dataname][:] = np.random.uniform(-1, 1, size=size)

    with Dataset(fnc, "r") as nc:
        df = nc4dataframe.get_dataframe(nc, "a", dataname)
        assert df.shape == (len(times), len(variables))

    fnc.unlink()


def test_stations(allclose):
    fnc = FHERE / "test_dataframe_stations.nc"
    with Dataset(fnc, "w") as nc:
        stations = ["a", "b", "c"]
        variables = ["v1", "v2[m/s]", "v3[-]", "v4"]

        # Random station info
        d = np.random.uniform(0, 1, (len(stations), len(variables)))
        df = pd.DataFrame(d, index=stations, columns=variables)
        df.loc[:, "NAME"] = ["".join([letters[i] for i in np.random.randint(0, 26, 10)]) for s in stations]
        df.loc[:, "TRUC[-]"] = ["".join([letters[i] for i in np.random.randint(0, 26, 10)]) for s in stations]

        nc4dataframe.set_station_info(nc, df)

    with Dataset(fnc, "r") as nc:
        df2 = nc4dataframe.get_station_info(nc)
        assert df2.shape == df.shape
        assert allclose(df2.iloc[:, :4].values, df.iloc[:, :4].values, atol=1e-6)

    fnc.unlink()

