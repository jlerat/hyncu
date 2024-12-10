"""Utility functions to export netcdf files """
from __future__ import annotations
import re
from typing import Optional
from getpass import getuser
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import cf_units

import netCDF4
from netCDF4 import Dataset

TIME_UNITS = "minutes since 1900-01-01 00:00:00"
TIME_DTYPE = np.int64
DEFAULT_MISSING_VALUE = -99999.0
DEFAULT_SIGNIFICANT_DIGIT = 5
DEFAULT_DTYPE = np.float32

DIM_LONGITUDE_NAME = "longitude"
DIM_LATITUDE_NAME = "latitude"
DIM_TIME_NAME = "time"


def num2date(nums: np.ndarray,
             units: Optional[str] = TIME_UNITS) -> pd.DatetimeIndex:
    """ Function to convert numerical time indexes to pandas
    DatetimeIndex. The function ensures that cftime datetime
    type are not used.
    """
    try:
        nums = netCDF4.num2date(nums, units,
                                only_use_cftime_datetimes=False)
    except TypeError:
        nums = netCDF4.num2date(nums, units)

    return pd.to_datetime(nums)


def date2num(times: pd.DatetimeIndex) -> np.ndarray:
    """ Function to convert pandas datetime to numerical
    values.
    """
    times = times.to_pydatetime()
    nums = netCDF4.date2num(times, TIME_UNITS)
    return nums


class Dimension():
    def __init__(self, name: str, values: np.ndarray, dtype: str, units: str):
        self.name = str(name)
        self.values = np.array(values)
        self.dtype = np.dtype(dtype)
        self.dimension_type = "generic"

        # Check unit
        cf_units.Unit(units)
        self.units = units

    def __str__(self):
        txt = f"Dimension {self.name} [self.dimension_type]"\
              + f", {self.dtype}, {len(self.values)} values"
        return txt

    def to_dataset(self, ncdset: Dataset) -> None:
        nval = len(self.values)

        # Check if dimension does not already exist
        if self.name in ncdset.dimensions:
            check = ncdset[self.name][:]

            if not np.all(check == self.values):
                errmess = "Existing dimensions does not match."
                raise ValueError(errmess)

            return

        ncdset.createDimension(self.name, nval)
        var = ncdset.createVariable(self.name, self.dtype,
                                    dimensions=[self.name])
        var[:] = self.values
        var.dimension_type = self.dimension_type
        var.units = self.units
        var.long_name = self.name
        var.standard_name = self.name

    @classmethod
    def from_dataset(cls, ncdset: Dataset, name: str) -> Dimension:
        dvar = ncdset[name]
        dtype = dvar.dtype
        units = dvar.units
        values = dvar[:]
        return Dimension(name, values, dtype, units)


class TimeDimension(Dimension):
    def __init__(self, name: str, values: np.ndarray):
        times = netCDF4.date2num(pd.to_datetime(values).to_pydatetime(),
                                 units=TIME_UNITS)
        dtype = TIME_DTYPE
        units = TIME_UNITS
        super(TimeDimension, self).__init__(name, times, dtype, units)
        self.dimension_type = "time"

    @classmethod
    def from_dataset(cls, ncdset: Dataset) -> Dimension:
        dim = Dimension.from_dataset(ncdset, DIM_TIME_NAME)
        dim.values = num2date(dim.values, dim.units)
        return dim


class CoordinateDimension(Dimension):
    def __init__(self, values: np.ndarray, name: str):
        assert name in [DIM_LONGITUDE_NAME, DIM_LATITUDE_NAME]
        values = np.array(values).astype(np.float64)
        dtype = np.float64
        units = "degrees_east" if name == DIM_LONGITUDE_NAME\
                else "degrees_north"
        super(CoordinateDimension, self).__init__(name, values, dtype, units)
        self.dimension_type = "coordinate"

    @classmethod
    def from_dataset(cls, ncdset: Dataset, name: str) -> Dimension:
        assert name in [DIM_LONGITUDE_NAME, DIM_LATITUDE_NAME]
        return Dimension.from_dataset(ncdset, name)


def add_dimensions(ncdset: Dataset, **kwargs) -> None:
    dims = {}
    for dname, values in kwargs.items():
        # Test if the dimension is time
        try:
            np.datetime_data(values.dtype)
            istime = True
        except Exception:
            istime = False

        if istime:
            tdim = TimeDimension(dname, values)
            tdim.to_dataset(ncdset)
            dims[dname] = tdim
        else:
            if dname in [DIM_LONGITUDE_NAME, DIM_LATITUDE_NAME]:
                ldim = CoordinateDimension(values, dname)
                ldim.to_dataset(ncdset)
                dims[dname] = ldim
            else:
                values = np.array(values)
                dim = Dimension(dname, values, values.dtype, "-")
                dim.to_dataset(ncdset)
                dims[dname] = dim

    return dims


def minimal_metadata(attrs: Optional[dict] = None):
    if attrs is None:
        attrs = {}

    attrs["date_created"] = attrs.get("date_created",
                                      str(datetime.now()))
    attrs["version"] = re.sub("^v", "", str(attrs.get("version", "1.0")))
    attrs["data_provider"] = attrs.get("data_provider", "unknown")
    attrs["source_file"] = attrs.get("source_file",
                                     str(Path(__file__).resolve()))
    try:
        user = getuser()
    except Exception:
        user = "unknown"

    attrs["author"] = attrs.get("author", user)
    return attrs


class Variable():
    def __init__(self, name: str, dimensions: list,
                 units: Optional[str] = "-",
                 dtype: Optional[str] = DEFAULT_DTYPE,
                 compression: Optional[str] = "zlib",
                 chunksizes: Optional[tuple] = None,
                 fill_value: Optional[float] = DEFAULT_MISSING_VALUE,
                 significant_digit: Optional[int] = DEFAULT_SIGNIFICANT_DIGIT,
                 attrs: Optional[dict] = None):
        self.name = str(name)
        self.dtype = np.dtype(dtype)
        cf_units.Unit(units)
        self.units = units
        self.dimensions = dimensions
        if significant_digit is not None:
            self.significant_digit = int(significant_digit)
        else:
            self.significant_digit = None
        self.compression = compression
        self.fill_value = self.dtype.type(fill_value)
        self.chunksizes = chunksizes
        self.attrs = minimal_metadata(attrs)

    def to_dataset(self, ncdset: Dataset):
        sdigit = self.significant_digit
        var = ncdset.createVariable(varname=self.name,
                                    dimensions=self.dimensions,
                                    datatype=self.dtype,
                                    chunksizes=self.chunksizes,
                                    least_significant_digit=sdigit,
                                    compression=self.compression,
                                    fill_value=self.fill_value)
        var.units = self.units
        var.long_name = self.name
        var.standard_name = self.name
        for key, val in self.attrs.items():
            setattr(var, str(key), val)
