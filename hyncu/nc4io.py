"""Utility functions to export netcdf files """
import re, sys
import numpy as np
import pandas as pd
from datetime import datetime
import cf_units
import netCDF4

TIME_UNITS = "hours since 1900-01-01 00:00:00"
TIME_DTYPE = np.int64
DEFAULT_MISSING_VALUE = -99999.0
DEFAULT_SIGNIFICANT_DIGIT = 5
DEFAULT_DTYPE = np.float32

DIM_LON_NAME = "longitude"
DIM_LAT_NAME = "latitude"
DIM_TIME_NAME = "time"


def num2date(nums, units):
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
    def __init__(self, name, values, dtype, units):
        self.name = str(name)
        self.values = np.array(values)
        self.dtype = np.dtype(dtype)

        # Check unit
        cf_units.Unit(units)
        self.units = units

    def to_dataset(self, nc4dset):
        nval = len(self.values)
        nc4dset.createDimension(self.name, nval)
        var = nc4dset.createVariable(self.name, self.dtype, \
                         dimensions=[self.name])
        var[:] = self.values
        var.units = self.units
        var.long_name = self.name
        var.standard_name = self.name

    @classmethod
    def from_dataset(cls, nc4dset, name):
        dvar = nc4dset[name]
        dtype = dvar.dtype
        units = dvar.units
        values = dvar[:]
        return Dimension(name, values, dtype, units)


class TimeDimension(Dimension):
    def __init__(self, values):
        name = DIM_TIME_NAME
        times = netCDF4.date2num(pd.to_datetime(values).to_pydatetime(), \
                            units=TIME_UNITS)
        dtype = TIME_DTYPE
        units = TIME_UNITS
        super(TimeDimension, self).__init__(name, times, dtype, units)

    @classmethod
    def from_dataset(cls, nc4dset):
        dim = Dimension.from_dataset(nc4dset, DIM_TIME_NAME)
        dim.values = num2date(dim.values, dim.units)
        return dim


class CoordinateDimension(Dimension):
    def __init__(self, values, name):
        assert name in [DIM_LON_NAME, DIM_LAT_NAME]
        values = np.array(values).astype(np.float64)
        dtype = np.float64
        units = "degrees_east" if name==DIM_LON_NAME else "degrees_north"
        super(CoordinateDimension, self).__init__(name, values, dtype, units)

    @classmethod
    def from_dataset(cls, nc4dset, name):
        assert name in [DIM_LON_NAME, DIM_LAT_NAME]
        return Dimension.from_dataset(nc4dset, name)


def add_dimensions(ncdset, **kwargs):
    for dname, values in kwargs.items():
        if dname=="time":
            tdim = TimeDimension(values)
            tdim.to_dataset(ncdset)

        elif dname in [DIM_LON_NAME, DIM_LAT_NAME]:
            ldim = CoordinateDimension(values, dname)
            ldim.to_dataset(ncdset)

        else:
            values = np.array(values)
            dim = Dimension(dname, values, values.dtype, "-")
            dim.to_dataset(ncdset)


class Variable():
    def __init__(self, name, dimensions,
                    units="-", \
                    dtype=DEFAULT_DTYPE, \
                    compression="zlib", \
                    chunksizes=None, \
                    fill_value=DEFAULT_MISSING_VALUE, \
                    significant_digit=DEFAULT_SIGNIFICANT_DIGIT, \
                    attrs=None):
        self.name = str(name)
        self.dtype = np.dtype(dtype)
        cf_units.Unit(units)
        self.units = units
        self.dimensions = dimensions
        self.significant_digit = int(significant_digit)
        self.compression = compression
        self.fill_value = fill_value
        self.chunksizes = chunksizes
        self.attrs = {} if attrs is None else attrs


    def to_dataset(self, nc4dset):
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



def add_meta(nc4dset, **kwargs):
    if not "summary" in kwargs:
        kwargs["summary"] = "Data file created by CSIRO Environment"

    if not "institution" in kwargs:
        kwargs["institution"] = "CSIRO Environment"

    if not "date_created" in kwargs:
        kwargs["date_created"] = str(datetime.now())

    for key, value in kwargs.items():
        setattr(nc4dset, key, value)


