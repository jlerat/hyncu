"""Utility functions to export netcdf files """
import re, sys
from collections import namedtuple
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


class SiteidDimension(Dimension):
    def __init__(self, values):
        name = "siteid"
        values = np.array(values)
        dtype = values.dtype
        units = "-"
        super(SiteidDimension, self).__init__(name, values, dtype, units)

    @classmethod
    def from_dataset(cls, nc4dset):
        return Dimension.from_dataset(nc4dset, "siteid")


class LatLongDimension(Dimension):
    def __init__(self, values, islon=True):
        name = DIM_LON_NAME if islon else DIM_LAT_NAME
        values = np.array(values).astype(np.float64)
        dtype = np.float64
        units = "degrees_east" if islon else "degrees_north"
        super(LatLongDimension, self).__init__(name, values, dtype, units)

    @classmethod
    def from_dataset(cls, nc4dset, islon=True):
        name = DIM_LON_NAME if islon else DIM_LAT_NAME
        return Dimension.from_dataset(nc4dset, name)


def add_dimensions(ncdset, times=None, lons=None, lats=None):
    if not times is None:
        tdim = TimeDimension(times)
        tdim.to_dataset(ncdset)

    if not lons is None:
        ldim = LatLongDimension(lons)
        ldim.to_dataset(ncdset)

    if not lats is None:
        ldim = LatLongDimension(lats, False)
        ldim.to_dataset(ncdset)



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



def add_meta(nc4dset, summary=None, institution=None):
    if summary is None:
        ncdset.summary = "Data file created by CSIRO Environment"

    if institution is None:
        ncdset.institution = "CSIRO Environment"

    ncdset.date_created = str(datetime.now())

