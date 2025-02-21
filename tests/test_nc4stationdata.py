import math
import re
from pathlib import Path

import random
import string

import pytest

import numpy as np
np.random.seed(5446)
import pandas as pd

from netCDF4 import Dataset
from hyncu import nc4io, nc4stationdata as nc4sd

from test_nc4io import generate_random_strings

import warnings

FHERE = Path(__file__).resolve().parent

SEP = nc4sd.LABEL_SEPARATOR

random.seed(5446)

@pytest.mark.parametrize("index_type",
                         ["time", "integer", "real",
                          "str-ascii", "str-unicode"])
def test_dataframe(index_type):
    fnc = FHERE / f"test_dataframe_{index_type}_index.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        stationids = ["a", "b", "c", "d"]
        variable_names = ["v1", "v2", "v3", "v4"]

        if index_type == "time":
            index = pd.date_range("1990-01-01", "2000-12-31", freq="D")
        elif index_type == "integer":
            index = np.arange(100)
        elif index_type == "real":
            index = np.linspace(0, 1, 100)
        elif index_type == "str-ascii":
            index = generate_random_strings(100, False)
        elif index_type == "str-unicode":
            index = generate_random_strings(100, True)

        dataname = "truc"
        units = ["mm.day-1", "m3.s-1", "degC", "kg"]
        attrs = {"author": "me"}

        if index_type == "str-unicode":
            msg = "Values in dimension"
            with pytest.raises(ValueError, match=msg):
                svar = nc4sd.StationVariable(nc, dataname,
                                     stationids=stationids,
                                     index=index,
                                     column_names=variable_names,
                                     column_units=units,
                                     attrs=attrs)
            return

        svar = nc4sd.StationVariable(nc, dataname,
                                     stationids=stationids,
                                     index=index,
                                     column_names=variable_names,
                                     column_units=units,
                                     attrs=attrs)

        assert nc4sd.STATION_DATA_INDEX_DIMENSION_NAME in nc.dimensions
        assert nc4sd.STATIONID_DIMENSION_NAME in nc.dimensions

        dlabel = nc4sd.NUMERICAL_DATA_TYPE_LABEL
        colvarname = nc4sd.column_variable_ncname(dataname, dlabel)
        assert colvarname in nc.dimensions
        assert colvarname in nc.variables

        unitname = nc4sd.unit_variable_ncname(dataname, dlabel)
        assert unitname in nc.variables
        assert all([u1 == u2 for u1, u2 in zip(nc[unitname][:], units)])

        varname = nc4sd.station_variable_ncname(dataname, dlabel)
        assert varname in nc.variables

        assert nc[varname].author == "me"

        size = (len(index), len(variable_names))
        cols = [f"{v}[{u}]" for v, u in zip(variable_names, units)]
        df = pd.DataFrame(np.random.uniform(-1, 1, size=size), \
                                index=index, columns=cols)

        svar.write_data_for_single_station("a", df)

        # Set partial dataset
        df2 = df.iloc[:len(df)//2, :2]
        svar.write_data_for_single_station("b", df2)

        # Set numpy array
        v = df.values.copy()
        svar.write_data_for_single_station("c", v)

        with pytest.raises(ValueError, match="Expected data of size"):
            v = df2.values.copy()
            svar.write_data_for_single_station("c", v)


    with Dataset(fnc, "r") as nc:
        svar2 = nc4sd.StationVariable(nc, dataname)

        df, attrs = svar2.read_data_from_single_station("a")
        assert df.shape == (len(index), len(variable_names))

        df3, attrs = svar2.read_data_from_single_station("b", clip=False)
        assert df3.shape == (len(index), len(variable_names))

        coln = svar2.get_column_names()
        for icn, cn in enumerate(coln):
            v = df3.loc[df2.index[-1]:, cn].iloc[1:]
            assert v.isnull().all()

            if icn>=2:
                assert df3.loc[:, cn].isnull().all()

        df4, attrs = svar2.read_data_from_single_station("b")
        assert df4.shape == df2.shape

        with pytest.raises(ValueError, match="No station data"):
            svar2.read_data_from_single_station("d")

    fnc.unlink()


@pytest.mark.parametrize("unicode", [False, True])
def test_stations(unicode, allclose):
    fnc = FHERE / "test_dataframe_stations.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        stations = generate_random_strings(10, unicode)
        units = ["", "[-]", "[m.s-1]", "[kg]"]
        v = generate_random_strings(20, unicode)
        variables = [f"{v[i]}{random.choice(units)}" for i in range(20)]
        # Random station info
        d = np.random.uniform(0, 1, (len(stations), len(variables)))
        df = pd.DataFrame(d, index=stations, columns=variables)

        nsites = len(stations)
        df.loc[:, "NAME"] = generate_random_strings(nsites, unicode)
        df.loc[:, "TRUC[-]"] = generate_random_strings(nsites, unicode)

        with pytest.raises(ValueError, match="LONGITUDE was expected"):
            ncsta = nc4sd.StationMetaData(nc, df)

        df.loc[:, "LONGITUDE"] = 1.
        df.loc[:, "LATITUDE"] = 2.
        ncsta = nc4sd.StationMetaData(nc, df)

        # Add a second dataset
        df_bis = df.iloc[:, 1:]
        ncsta = nc4sd.StationMetaData(nc, df_bis, name="station_bis")

    with Dataset(fnc, "r") as nc:
        ncsta2 = nc4sd.StationMetaData(nc)
        df2, attrs = ncsta2.read_metadata_from_dataset()

        assert df2.shape == df.shape
        # Both have same columns
        df2 = df2.loc[:, df.columns]
        assert allclose(df2.iloc[:, :4].values, df.iloc[:, :4].values, atol=1e-6)

        ncsta3 = nc4sd.StationMetaData(nc, name="station_bis")
        df3, attrs = ncsta3.read_metadata_from_dataset()
        assert df3.shape[0] == df.shape[0]
        assert df3.shape[1] == df.shape[1]-1

    fnc.unlink()

