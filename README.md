# hyncu

Minimalistic wrapper around netCDF4 written in pure python.

# What is hyncu
This package provides a simple interface to add data to a netcdf4 including file with a
minimalistic interface. implements the Quadratic Solution of the Approximate Reservoir 

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

fnc = "test_dataframe.nc"
with Dataset(fnc, "w") as nc:

    # --- working with gridded data
    grp = nc.createGroup("gridded")

    # define dimensions
    lons = np.linspace(110, 140, 10)
    lats = np.linspace(-40, -10, 10)
    nc4io.add_dimensions(grp, longitude=lons, latitude=lats)

    # Setup variable
    dataset_name = "important_gridded_stuff"
    nvar = nc4io.Variable(dataset_name, ["latitude", "longitude"], \
                units="mm.month-1")
    nvar.to_dataset(grp)

    # Store data
    data = np.random.uniform(size=(len(lats), len(lons)))
    grp[dataset_name][:] = data


    # --- Working with pandas data frames ----
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
    nc4sd.set_station_data(grp, df)

    # Define data variables and their units
    variables = ["v1", "v2", "v3", "v4"]
    units = ["mm.day-1", "m3.s-1", "degC", "kg"]

    # Define index
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
        nc4sd.set_data(grp, dataset_name, stationid, data)

    # Retrieve station meta data
    df = nc4sd.get_station_data(grp)

    # Retrieve station data
    for stationid in stationids:
        d = nc4sd.get_data(grp, dataset_name, stationid)
```

## Attribution
This project is licensed under the [MIT License](LICENSE), which allows for free use, modification, and distribution of the code under the terms of the license.


