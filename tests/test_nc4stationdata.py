import math
from pathlib import Path

import pytest
from string import ascii_letters as letters

import numpy as np
np.random.seed(5446)
import pandas as pd

from netCDF4 import Dataset
from hyncu import nc4io, nc4stationdata as nc4sd

import warnings

FHERE = Path(__file__).resolve().parent

SEP = nc4sd.LABEL_SEPARATOR

@pytest.mark.parametrize("index_type",
                         ["time", "integer", "real", "str"])
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
        elif index_type == "str":
            index = ["".join([letters[i]
                              for i in np.random.randint(0, 26, 10)])
                                 for k in range(100)]

        dataname = "truc"
        units = ["mm.day-1", "m3.s-1", "degC", "kg"]
        attrs = {"author": "me"}
        svar = nc4sd.StationVariable(nc, dataname,
                                     stationids=stationids,
                                     index=index,
                                     column_names=variable_names,
                                     column_units=units,
                                     attrs=attrs)

        assert nc4sd.STATION_DATA_INDEX_DIMENSION_NAME in nc.dimensions
        assert nc4sd.STATIONID_DIMENSION_NAME in nc.dimensions

        dlabel = nc4sd.NUMERICAL_DATA_LABEL
        colvarname = nc4sd.column_variable_ncname(dataname, dlabel)
        assert colvarname in nc.dimensions
        assert colvarname in nc.variables

        unitname = nc4sd.unit_variable_ncname(dataname, dlabel)
        assert unitname in nc.variables

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
        nc4sd.write_data_for_single_station(nc, dataname, "b", df2)

        # Set numpy array
        v = df.values.copy()
        nc4sd.write_data_for_single_station(nc, dataname, "c", v)

        with pytest.raises(ValueError, match="Expected data of size"):
            v = df2.values.copy()
            nc4sd.write_data_for_single_station(nc, dataname, "c", v)


    with Dataset(fnc, "r") as nc:
        df, attrs = nc4sd.read_data_from_single_station(nc, dataname, "a")
        assert df.shape == (len(index), len(variable_names))

        df3, attrs = nc4sd.read_data_from_single_station(nc, dataname, "b", clip=False)
        assert df3.shape == (len(index), len(variable_names))

        coln = nc4sd.get_dataset_variable_names(nc, dataname)
        for icn, cn in enumerate(coln):
            v = df3.loc[df2.index[-1]:, cn].iloc[1:]
            assert v.isnull().all()

            if icn>=2:
                assert df3.loc[:, cn].isnull().all()

        df4, attrs = nc4sd.read_data_from_single_station(nc, dataname, "b")
        assert df4.shape == df2.shape

        with pytest.raises(ValueError, match="No station data"):
            nc4sd.read_data_from_single_station(nc, dataname, "d")

    fnc.unlink()


def test_stations(allclose):
    fnc = FHERE / "test_dataframe_stations.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        stations = ["a", "b", "c"]
        variables = ["v1", "v2[m/s]", "v3[-]", "v4"]

        # Random station info
        d = np.random.uniform(0, 1, (len(stations), len(variables)))
        df = pd.DataFrame(d, index=stations, columns=variables)
        df.loc[:, "NAME"] = ["".join([letters[i] \
                                    for i in np.random.randint(0, 26, 10)]) \
                                        for s in stations]
        df.loc[:, "TRUC[-]"] = ["".join([letters[i] \
                                    for i in np.random.randint(0, 26, 10)]) \
                                        for s in stations]
        with pytest.raises(ValueError, match="LONGITUDE was expected"):
            nc4sd.write_station_info(nc, df)

        df.loc[:, "LONGITUDE"] = 1.
        df.loc[:, "LATITUDE"] = 2.
        nc4sd.write_station_info(nc, df)

        # Add a second dataset
        df_bis = df.copy().drop(["NAME", "LONGITUDE"], axis=1)
        nc4sd.write_station_info(nc, df_bis, "stations_bis")

    with Dataset(fnc, "r") as nc:
        df2, attrs = nc4sd.read_station_info(nc)
        assert df2.shape == df.shape
        assert allclose(df2.iloc[:, :4].values, df.iloc[:, :4].values, atol=1e-6)

        # Test unit is not added to text columns
        assert "NAME" in df2.columns

        df3, attrs = nc4sd.read_station_info(nc, "stations_bis")
        assert df3.shape[0] == df.shape[0]
        assert df3.shape[1] == df.shape[1]-2

    fnc.unlink()

