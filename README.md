# hyncu

[![Tests](https://github.com/jlerat/hyncu/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/jlerat/hyncu/actions/workflows/python-package-conda.yml)
![Coverage](https://gist.githubusercontent.com/jlerat/7ec5d1842cdd8ac006db3a2393113299/raw/coverage_badge.svg)

Minimalistic wrapper around netCDF4 written in pure python.

# What is hyncu
This package provides a simple interface to add data to a netcdf4 file with a
minimalistic interface.

# Installation
- Create a suitable python environment. We recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) combined with the environment specification provided in the [env\_hyncu.yml] (env_hyncu.yml) file in this repository.
- Git clone this repository and run `pip install .`

# Basic use

```python
from string import ascii_letters as letters
from netCDF4 import Dataset
import numpy as np
import pandas as pd

from hyncu import nc4io
from hyncu import nc4stationdata as nc4sd

# Meta data
attrs = {
    "author": "me",
    "description": "test data",
    "source_file": "script_file.py",
    "data_provider": "This organisation",
    "version": "v999.999"
    }

with Dataset(fnc, "w") as nc:
    # ---------------------------------
    # working with gridded data

    grp = nc.createGroup("gridded")

    # 0. Generate data to store
    # .. dimensions
    lons = np.linspace(110, 140, 10)
    lats = np.linspace(-40, -10, 10)
    # .. gridded data
    data = np.random.uniform(size=(len(lats), len(lons)))

    # I. define grid dimensions
    nc4io.add_dimensions(grp, longitude=lons, latitude=lats)

    # II. Setup gridded data variable
    dataset_name = "data_gridded"
    nvar = nc4io.Variable(dataset_name, ["latitude", "longitude"], \
                units="mm.month-1", attrs=attrs)
    nvar.to_dataset(grp)

    # III. Store data. This is a netCDF4 command.
    grp[dataset_name][:] = data

    # ---------------------------------
    # Working with data frames

    grp = nc.createGroup("dataframe")

    # 0. Generate data to store
    # .. Random station informations
    stationids = ["sta1", "sta2", "sta3"]
    names = ["".join([letters[i] for i in np.random.randint(0, 26, 10)])
             for s in stationids]

    lons = np.random.uniform(110, 150, len(stationids))
    lats = np.random.uniform(-50, -10, len(stationids))
    info = pd.DataFrame({
        "NAME": names,
        "LONGITUDE": lons,
        "LATITUDE": lats
        }, index=stationids)

    # .. Define station data variables and their units
    variables = ["v1", "v2", "v3", "v4"]
    units = ["mm.day-1", "m3.s-1", "degC", "kg"]

    # .. Define station data index
    times = pd.date_range("1990-01-01", "2000-12-31", freq="D")

    # .. random station data
    data = {}
    for stationid in stationids:
        data[stationid]  = np.random.uniform(0, 1, (len(times), len(variables)))

    # I. Store stations informations
    nc4sd.write_station_info(grp, info, attrs=attrs)

    # II. Configure netcdf file
    dataset_name = "station_data"
    nc4sd.add_variables(grp,
                        stationids=stationids,
                        variables=variables,
                        units=units,
                        index=times,
                        dataset_name=dataset_name,
                        attrs=attrs)

    # III. store data for each station (random data)
    for stationid in stationids:
        dt = data[stationid]
        nc4sd.write_data_single_station(grp, dataset_name, stationid, dt)

    # IV. Retrieve station info
    info, _ = nc4sd.read_station_info(grp)

    # V. Retrieve data for each station
    for stationid in stationids:
        d = nc4sd.read_data_single_station(grp, dataset_name, stationid)

```

## Attribution
This project is licensed under the [MIT License](LICENSE), which allows for free use, modification, and distribution of the code under the terms of the license.


