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

fnc = MYPATH / "test_example.nc"
with Dataset(fnc, "w") as nc:
    # -----------------------------------------
    # working with gridded data

    # 0. Generate random gridded data to play with
    # .. dimensions
    lons = np.linspace(110, 140, 10)
    lats = np.linspace(-40, -10, 10)
    # .. gridded data
    data = np.random.uniform(size=(len(lats), len(lons)))

    # I. Setup gridded data variable
    grp = nc.createGroup("bunch_of_spatial_stuff")
    dataset_name = "my_gridded_data"
    nvar = nc4io.SpatialVariable(grp, dataset_name,
                longitudes=lons,
                latitudes=lats,
                units="mm.month-1", attrs=attrs)

    # II. Store data.
    nvar.write_data_to_dataset(data)

    # -----------------------------------------
    # Working with data frames

    # 0. Generate random station data to play with
    # .. Random station metadata
    stationids = ["sta1", "sta2", "sta3"]
    names = ["".join([letters[i] for i in np.random.randint(0, 26, 10)])
             for s in stationids]
    lons = np.random.uniform(110, 150, len(stationids))
    lats = np.random.uniform(-50, -10, len(stationids))
    metadata = pd.DataFrame({
        "NAME": names,
        "LONGITUDE": lons,
        "LATITUDE": lats
        }, index=stationids)
    # .. Define variables and their units
    variable_names = ["v1", "v2", "v3", "v4"]
    units = ["mm.day-1", "m3.s-1", "degC", "kg"]
    # .. Define station data index
    times = pd.date_range("1990-01-01", "2000-12-31", freq="D")
    # .. Generate random station data
    data = {}
    for stationid in stationids:
        data[stationid]  = np.random.uniform(0, 1,
                                             (len(times),
                                             len(variable_names)))

    # I. Store stations metadata
    grp = nc.createGroup("bunch_of_station_stuff")
    nc4sd.StationMetaData(grp, metadata, attrs=attrs)

    # II. Configure station info variables
    dataset_name = "station_data"
    svar = nc4sd.StationVariable(grp, dataset_name,
                          stationids=stationids,
                          index=times,
                          column_names=variable_names,
                          column_units=units,
                          attrs=attrs)

    # III. store data for each station (random data)
    for stationid in stationids:
        dt = data[stationid]
        svar.write_data_for_single_station(stationid, dt)

    # IV. Retrieve station info
    smeta = nc4sd.StationMetaData(grp)
    metadata, _ = smeta.read_metadata_from_dataset()

    # V. Retrieve data for each station
    svar = nc4sd.StationVariable(grp, dataset_name)
    for stationid in stationids:
        d, _ = svar.read_data_from_single_station(stationid)
        assert d.shape == (len(times), len(variable_names))

    # Show netcdf file content
    print("NC file content:")
    for varnc in grp.variables:
        print(f"{varnc} : {grp[varnc].shape}")
```

## Attribution
This project is licensed under the [MIT License](LICENSE), which allows for free use, modification, and distribution of the code under the terms of the license.


