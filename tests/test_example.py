import math
from pathlib import Path
from string import ascii_letters as letters
import pandas as pd

import pytest

import numpy as np
np.random.seed(5446)

from netCDF4 import Dataset
from hyncu import nc4io
from hyncu import nc4stationdata as nc4sd

import warnings

FHERE = Path(__file__).resolve().parent


def test_example(allclose):
    fnc = FHERE / "test_example.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        # --- working with gridded data
        grp = nc.createGroup("gridded")

        # define dimensions
        lons = np.linspace(110, 140, 10)
        lats = np.linspace(-40, -10, 10)
        nc4io.add_dimensions(grp, longitude=lons, latitude=lats)

        # Setup variable
        dataset_name = "important_gridded_stuff"
        attrs = {"description": "Some description"}
        nvar = nc4io.Variable(dataset_name, ["latitude", "longitude"], \
                    units="mm.month-1", attrs=attrs)
        nvar.to_dataset(grp)

        # Store data
        data = np.random.uniform(size=(len(lats), len(lons)))
        grp[dataset_name][:] = data


        # --- Working with data frames ----
        grp = nc.createGroup("dataframe")

        # Define list of stations with meta data (random here)
        stationids = ["sta1", "sta2", "sta3"]
        names = ["".join([letters[i] \
                                    for i in np.random.randint(0, 26, 10)]) \
                                        for s in stationids]
        lons = np.random.uniform(110, 150, len(stationids))
        lats = np.random.uniform(-50, -10, len(stationids))
        df = pd.DataFrame({"NAME": names, "LONGITUDE": lons, "LATITUDE": lats}, \
                            index=stationids)

        # Store stations meta data
        nc4sd.set_data_stations(grp, df)

        # Define data variables and their units
        variables = ["v1", "v2", "v3", "v4"]
        units = ["mm.day-1", "m3.s-1", "degC", "kg"]

        # Define data index
        times = pd.date_range("1990-01-01", "2000-12-31", freq="D")

        # Configure netcdf file
        dataset_name = "important_dataframe_stuff"
        nc4sd.add_variables(grp, \
                    stationids=stationids, \
                    variables=variables, \
                    units=units, \
                    index=times, \
                    dataset_name=dataset_name)

        # store data for each station
        for stationid in stationids:
            data  = np.random.uniform(0, 1, (len(times), len(variables)))
            nc4sd.set_data_single_site(grp, dataset_name, stationid, data)

        # Retrieve station meta data
        df, attrs = nc4sd.get_data_stations(grp)

        # Retrieve station data
        for stationid in stationids:
            d = nc4sd.get_data_single_site(grp, dataset_name, stationid)

    fnc.unlink()

