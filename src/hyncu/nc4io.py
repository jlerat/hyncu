"""Utility functions to export netcdf files """
from __future__ import annotations
import re, sys
from getpass import getuser
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


def num2date(nums: np.ndarray, units:str) -> pd.DatetimeIndex:
    """ Function to convert numerical time indexes to pandas
    DatetimeIndex. The function ensures that cftime datetime
    type are not used.
    """
    try:
        nums = netCDF4.num2date(nums, units, \
                            only_use_cftime_datetimes=False)
    except TypeError:
        nums = netCDF4.num2date(nums, units)

    return pd.to_datetime(nums)


class Dimension():
    def __init__(self, name: str, values: np.ndarray, dtype: str, units: str):
        self.name = str(name)
        self.values = np.array(values)
        self.dtype = np.dtype(dtype)

        # Check unit
        cf_units.Unit(units)
        self.units = units

    def to_dataset(self, nc4dset:Dataset) -> None:
        nval = len(self.values)

        # Check if dimension does not already exist
        if self.name in nc4dset.dimensions:
            check = nc4dset[self.name][:]
            if not np.all(check==self.values):
                errmess = f"Existing dimension does not match."
                raise ValueError(errmess)
            return

        nc4dset.createDimension(self.name, nval)
        var = nc4dset.createVariable(self.name, self.dtype, \
                         dimensions=[self.name])
        var[:] = self.values
        var.units = self.units
        var.long_name = self.name
        var.standard_name = self.name

    @classmethod
    def from_dataset(cls, nc4dset: Dataset, name: str) -> Dimension:
        dvar = nc4dset[name]
        dtype = dvar.dtype
        units = dvar.units
        values = dvar[:]
        return Dimension(name, values, dtype, units)


class TimeDimension(Dimension):
    def __init__(self, values: np.ndarray):
        name = DIM_TIME_NAME
        times = netCDF4.date2num(pd.to_datetime(values).to_pydatetime(), \
                            units=TIME_UNITS)
        dtype = TIME_DTYPE
        units = TIME_UNITS
        super(TimeDimension, self).__init__(name, times, dtype, units)

    @classmethod
    def from_dataset(cls, nc4dset: Dataset) -> Dimension:
        dim = Dimension.from_dataset(nc4dset, DIM_TIME_NAME)
        dim.values = num2date(dim.values, dim.units)
        return dim


class CoordinateDimension(Dimension):
    def __init__(self, values: np.ndarray, name: str):
        assert name in [DIM_LONGITUDE_NAME, DIM_LATITUDE_NAME]
        values = np.array(values).astype(np.float64)
        dtype = np.float64
        units = "degrees_east" if name==DIM_LONGITUDE_NAME else "degrees_north"
        super(CoordinateDimension, self).__init__(name, values, dtype, units)

    @classmethod
    def from_dataset(cls, nc4dset: Dataset, name: str) -> Dimension:
        assert name in [DIM_LONGITUDE_NAME, DIM_LATITUDE_NAME]
        return Dimension.from_dataset(nc4dset, name)


def add_dimensions(ncdset: Dataset, **kwargs) -> None:
    dims = {}
    for dname, values in kwargs.items():
        if dname=="time":
            tdim = TimeDimension(values)
            tdim.to_dataset(ncdset)
            dims[dname] = tdim

        elif dname in [DIM_LONGITUDE_NAME, DIM_LATITUDE_NAME]:
            ldim = CoordinateDimension(values, dname)
            ldim.to_dataset(ncdset)
            dims[dname] = ldim

        else:
            values = np.array(values)
            dim = Dimension(dname, values, values.dtype, "-")
            dim.to_dataset(ncdset)
            dims[dname] = dim

    return dims


class Variable():
    def __init__(self, name: str, dimensions: list,
                    units: Optional[str] = "-", \
                    dtype: Optional[str] = DEFAULT_DTYPE, \
                    compression: Optional[str] = "zlib", \
                    chunksizes: Optional[tuple] = None, \
                    fill_value: Optional[float] = DEFAULT_MISSING_VALUE, \
                    significant_digit: Optional[int] = DEFAULT_SIGNIFICANT_DIGIT, \
                    attrs: Optional[dict] = None):
        self.name = str(name)
        self.dtype = np.dtype(dtype)
        cf_units.Unit(units)
        self.units = units
        self.dimensions = dimensions
        self.significant_digit = int(significant_digit)\
                            if not significant_digit is None else None
        self.compression = compression
        self.fill_value = self.dtype.type(fill_value)
        self.chunksizes = chunksizes
        self.attrs = {} if attrs is None else attrs


    def to_dataset(self, nc4dset: Dataset):
        var = nc4dset.createVariable(varname=self.name, \
                            dimensions=self.dimensions, \
                            datatype=self.dtype, \
                            chunksizes=self.chunksizes, \
                            least_significant_digit=self.significant_digit, \
                            compression=self.compression, \
                            fill_value=self.fill_value)
        var.units = self.units
        var.long_name = self.name
        var.standard_name = self.name
        for key, val in self.attrs.items():
            setattr(var, str(key), val)



def add_meta(nc4dset: Dataset, **kwargs: str):
    # Set minimal information
    kwargs["date_created"] = kwargs.get("date_created", \
                                            str(datetime.now()))
    kwargs["version"] = kwargs.get("version", \
                                            str(datetime.now()))
    try:
        user = getuser()
    except Exception:
        user = "unknown"
    kwargs["author"] = kwargs.get("author", user)

    # Set meta data
    for key, value in kwargs.items():
        setattr(nc4dset, key, value)


