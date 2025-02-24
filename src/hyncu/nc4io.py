"""Utility functions to export netcdf files """
from __future__ import annotations
import re
from collections import OrderedDict
from typing import Optional, Union
from getpass import getuser
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import cf_units

import netCDF4
from netCDF4 import Dataset
from netCDF4 import stringtochar

TIME_DIMENSION_TYPE_LABEL = "time"
SPATIAL_DIMENSION_TYPE_LABEL = "spatial"
GENERIC_DIMENSION_TYPE_LABEL = "generic"

TIME_ORIGIN = "1900"
TIME_UNITS = f"minutes since {TIME_ORIGIN}-01-01 00:00:00"
TIME_NUMPY_DTYPE = "i8"
DEFAULT_MISSING_VALUE = -99999.0
DEFAULT_SIGNIFICANT_DIGIT = 5
DEFAULT_NUMPY_DTYPE = "f4"

DIMENSION_LONGITUDE_NAME = "longitude"
DIMENSION_LATITUDE_NAME = "latitude"
DIMENSION_TIME_NAME = "time"
DIMENSION_NCHARS_NAME = "nchars"

STRING_MAX_LENGTH = 50


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
    times_py = times.to_pydatetime()
    nums = netCDF4.date2num(times_py, TIME_UNITS)
    return nums


def remove_non_ascii_element(s):
    return "".join(filter(lambda x: ord(x) < 128, s))

remove_non_ascii_vectorized = np.vectorize(remove_non_ascii_element)

def is_ascii_element(s):
    return all(ord(c) < 128 for c in s)

is_ascii_vectorised = np.vectorize(is_ascii_element)


class Dimension():
    def __init__(self, name: str, values: np.ndarray,
                 numpy_dtype: Union[str, numpy.dtype],
                 units: str):
        self.name = str(name)
        self.numpy_dtype = np.dtype(numpy_dtype)

        values = np.array(values).astype(self.numpy_dtype)
        if re.search("S|U", str(numpy_dtype)):
            all_ascii = np.all(is_ascii_vectorised(values))
            if not all_ascii:
                errmess = "Values in dimension cannot be "\
                          + "non-ascii characters."
                raise ValueError(errmess)

        self.values = values
        self.dimension_type = GENERIC_DIMENSION_TYPE_LABEL

        # Check unit
        cf_units.Unit(units)
        self.units = units

    def __str__(self):
        txt = f"Dimension {self.name} [self.dimension_type]"\
              + f", {self.numpy_dtype}, {len(self.values)} values"
        return txt

    def write_dimension_to_dataset(self, ncdset: Dataset) -> None:
        if self.name in ncdset.dimensions:
            return

        nval = len(self.values)
        ncdset.createDimension(self.name, nval)

        # Do not create variable if the dimension is nchars
        if self.name == DIMENSION_NCHARS_NAME:
            return

        var = ncdset.createVariable(self.name,
                                    datatype=self.numpy_dtype,
                                    dimensions=[self.name])
        try:
            var[:] = self.values
        except UnicodeEncodeError:
            errmess = "Only ascii character accepted in dimension"
            raise ValueError(errmess)

        var.dimension_type = self.dimension_type
        var.units = self.units
        var.long_name = self.name
        var.standard_name = self.name

    @classmethod
    def from_dataset(cls, ncdset: Dataset, name: str) -> Dimension:
        dvar = ncdset[name]
        numpy_dtype = dvar.dtype
        units = dvar.units
        values = dvar[:]
        return Dimension(name, values, numpy_dtype, units)


class TimeDimension(Dimension):
    def __init__(self, name: str, times: np.ndarray):
        numpy_dtype = TIME_NUMPY_DTYPE
        units = TIME_UNITS
        values = date2num(pd.to_datetime(times))
        super(TimeDimension, self).__init__(name, values, numpy_dtype, units)
        self.dimension_type = TIME_DIMENSION_TYPE_LABEL

    @classmethod
    def from_dataset(cls, ncdset: Dataset) -> Dimension:
        dim = Dimension.from_dataset(ncdset, DIMENSION_TIME_NAME)
        dim.values = num2date(dim.values, dim.units)
        return dim


class SpatialDimension(Dimension):
    def __init__(self, values: np.ndarray, name: str):
        if name not in [DIMENSION_LONGITUDE_NAME,
                        DIMENSION_LATITUDE_NAME]:
            errmess = "Wrong dimension name."
            raise ValueError(errmess)

        values = np.array(values).astype(np.float64)
        numpy_dtype = np.float64
        units = "degrees_east" if name == DIMENSION_LONGITUDE_NAME\
                else "degrees_north"
        super(SpatialDimension, self).__init__(name, values, numpy_dtype, units)
        self.dimension_type = SPATIAL_DIMENSION_TYPE_LABEL

    @classmethod
    def from_dataset(cls, ncdset: Dataset, name: str) -> Dimension:
        assert name in [DIMENSION_LONGITUDE_NAME, DIMENSION_LATITUDE_NAME]
        return Dimension.from_dataset(ncdset, name)


def add_dimension(ncdset: Dataset, dname: str, values: np.ndarray) -> None:
    try:
        np.datetime_data(values.dtype)
        dim = TimeDimension(dname, values)
        dim.write_dimension_to_dataset(ncdset)
        return dim
    except Exception:
        pass

    if dname == DIMENSION_NCHARS_NAME:
        values = np.array(values).astype(np.int32)
        dim = Dimension(dname, values, values.dtype, "-")
        dim.write_dimension_to_dataset(ncdset)
    elif dname in [DIMENSION_LONGITUDE_NAME, DIMENSION_LATITUDE_NAME]:
        dim = SpatialDimension(values, dname)
        dim.write_dimension_to_dataset(ncdset)
    else:
        values = np.array(values)
        dim = Dimension(dname, values, values.dtype, "-")
        dim.write_dimension_to_dataset(ncdset)

    return dim


def add_spatial_dimensions(ncdset: Dataset,
                           longitudes: np.ndarray,
                           latitudes: np.ndarray) -> None:
    add_dimension(ncdset, DIMENSION_LATITUDE_NAME, latitudes)
    add_dimension(ncdset, DIMENSION_LONGITUDE_NAME, longitudes)


def minimal_metadata(attrs: Optional[dict] = None):
    if attrs is None:
        attrs = {}

    attrs["title"] = attrs.get("title", "Unknown")
    attrs["description"] = attrs.get("description", "Unknown")
    attrs["comment"] = attrs.get("comment", "Unknown")
    attrs["date_created"] = attrs.get("date_created",
                                      str(datetime.now()))
    attrs["version"] = re.sub("^v", "", str(attrs.get("version", "0.1")))
    attrs["institution"] = attrs.get("institution", "Unknown")
    attrs["data_provider"] = attrs.get("data_provider", "Unknown")
    attrs["source_file"] = attrs.get("source_file",
                                     str(Path(__file__).resolve()))
    try:
        user = getuser()
    except Exception:
        user = "Unknown"

    attrs["author"] = attrs.get("author", user)
    return attrs


class Variable():
    def __init__(self, ncdset: Dataset,
                 name: str,
                 dimensions: Optional[Union[list, OrderedDict]] = None,
                 units: Optional[str] = "-",
                 numpy_dtype: Optional[str] = DEFAULT_NUMPY_DTYPE,
                 compression: Optional[str] = "zlib",
                 chunksizes: Optional[tuple] = None,
                 fill_value: Optional[float] = DEFAULT_MISSING_VALUE,
                 significant_digit: Optional[int] = DEFAULT_SIGNIFICANT_DIGIT,
                 attrs: Optional[dict] = None):

        # Check dataset is reasonble
        if not hasattr(ncdset, "variables") or \
                not hasattr(ncdset, "dimensions"):
            errmess = "Dataset does not have variables or dimensions"
            raise ValueError(errmess)

        if ncdset.file_format not in ["NETCDF4", "NETCDF4_CLASSIC"]:
            errmess = "Need NETCDF4 or NETCDF4_CLASSIC Dataset"\
                      + f" file format, got {ncdset.file_format}."
            raise ValueError(errmess)

        self.ncdset = ncdset
        self.name = str(name)
        if re.search("S|U", numpy_dtype):
            numpy_nchar = int(re.sub("^.*(U|S)", "", numpy_dtype))
            if numpy_nchar < 2:
                errmess = "Number of characters should be at least 2."
                raise ValueError(errmess)

            numpy_dtype = "S1"
            # Remove compression for string data
            compression = None
        else:
            numpy_nchar = 0

        self.numpy_dtype = np.dtype(numpy_dtype)
        self.numpy_nchar = numpy_nchar

        # Check units
        cf_units.Unit(units)
        self.units = units

        # Build dimensions
        if dimensions is None and self.name in self.ncdset.variables:
            dimensions = self.ncdset[self.name].dimensions
        self.dimensions = self.build_dimensions(dimensions)

        if significant_digit is not None and numpy_nchar == 0:
            self.significant_digit = int(significant_digit)
        else:
            self.significant_digit = None

        self.compression = compression
        try:
            self.fill_value = self.numpy_dtype.type(fill_value)
        except AttributeError:
            self.fill_value = "-"

        # Get chunksizes
        self.chunksizes = self.build_chunksizes(chunksizes)

        self.attrs = minimal_metadata(attrs)

        # Create dimensions and variables if needed
        self.write_dimensions_to_dataset()

        if self.name not in self.ncdset.variables:
            self.write_variable_to_dataset()

    def build_dimensions(self, dimensions):
        dims = OrderedDict()
        for dname in dimensions:
            # dimensions is present in netcdf Dataset
            if dname in self.ncdset.variables:
                if dname == DIMENSION_TIME_NAME:
                    times = self.ncdset[dname]
                    dims[dname] = num2date(times[:], times.units)
                else:
                    dim_values = self.ncdset[dname][:]
                    if str(dim_values.dtype) == "object":
                        dtype = f"U{STRING_MAX_LENGTH}"
                        dim_values = dim_values.astype(dtype)

                    dims[dname] = dim_values

                continue

            try:
                # dimensions is dict-like, hence we
                # can get values
                dims[dname] = dimensions[dname]
            except:
                # dimesions is list-like
                dims[dname] = None

        if self.numpy_nchar > 0:
            nchars = np.arange(self.numpy_nchar)
            dims[DIMENSION_NCHARS_NAME] = nchars

        return dims

    def build_chunksizes(self, chunksizes):
        dims = self.dimensions
        if dims is None:
            return chunksizes

        # Case default chunksizes
        if chunksizes is None:
            try:
                return [len(d) for _, d in self.dimensions.items()]
            except Exception:
                return None

        # Check dimensions
        # Case of string array
        if np.isscalar(chunksizes):
            chunksizes = [chunksizes]
        else:
            chunksizes = list(chunksizes)

        if len(chunksizes) == len(dims) - 1\
                and "nchars" in dims:
            n = len(dims["nchars"])
            chunksizes = list(chunksizes) + [n]

        # Check chunks
        for idim, dname in enumerate(dims):
            n = len(dims[dname])
            ck = chunksizes[idim]
            if ck > n:
                errmess = f"{idim} chunk ({ck}) bigger than data size ({n})."
                raise ValueError(errmess)

        return chunksizes

    def write_dimensions_to_dataset(self):
        for dname, dvalue in self.dimensions.items():
            if dvalue is None:
                continue
            add_dimension(self.ncdset, dname, dvalue)

    def write_variable_to_dataset(self):
        # Case where self.dimensions is a dict
        dims = [n for n in self.dimensions]

        sdigit = self.significant_digit

        var = self.ncdset.createVariable(varname=self.name,
                                dimensions=dims,
                                datatype=self.numpy_dtype,
                                chunksizes=self.chunksizes,
                                least_significant_digit=sdigit,
                                compression=self.compression,
                                fill_value=self.fill_value)

        # Set char attributes if we are dealing with strings
        # See https://unidata.github.io/netcdf4-python
        if self.numpy_nchar > 0:
            var._Encoding = "ascii"

        # Set basic attributes
        if self.units is not None:
            var.units = self.units

        var.long_name = self.name
        # Set other attributes
        for key, val in self.attrs.items():
            setattr(var, str(key), val)

    def write_data_to_dataset(self, data: np.ndarray,
                              subset_index: Optional[Union[list, tuple, np.ndarray]] = None):
        ncvar = self.ncdset[self.name]
        data = np.array(data)
        if self.numpy_nchar > 0:
            try:
                data = data.astype(f"S{self.numpy_nchar}")
            except UnicodeEncodeError:
                # Convert all non-ascii to ascii
                data = remove_non_ascii_vectorized(data)
                data = data.astype(f"S{self.numpy_nchar}")

        if subset_index is None:
            ncvar[:] = data
        else:
            ncvar[subset_index] = data


class SpatialVariable(Variable):
    def __init__(self, ncdset: Dataset, name: str,
                 longitudes: Optional[np.ndarray] = None,
                 latitudes: Optional[np.ndarray] = None,
                 units: Optional[str] = "-",
                 numpy_dtype: Optional[str] = DEFAULT_NUMPY_DTYPE,
                 compression: Optional[str] = "zlib",
                 chunksizes: Optional[tuple] = None,
                 fill_value: Optional[float] = DEFAULT_MISSING_VALUE,
                 significant_digit: Optional[int] = DEFAULT_SIGNIFICANT_DIGIT,
                 attrs: Optional[dict] = None):

        sdigit = significant_digit
        if longitudes is not None and latitudes is not None:
            dims = OrderedDict()
            dims[DIMENSION_LATITUDE_NAME] = latitudes
            dims[DIMENSION_LONGITUDE_NAME] = longitudes
        else:
            dims = [DIMENSION_LATITUDE_NAME,
                    DIMENSION_LONGITUDE_NAME]

        super(SpatialVariable, self).__init__(ncdset, name,
                                              dimensions=dims,
                                              units=units,
                                              numpy_dtype=numpy_dtype,
                                              compression=compression,
                                              chunksizes=chunksizes,
                                              fill_value=fill_value,
                                              significant_digit=sdigit,
                                              attrs=attrs)


class SpatialTimeVariable(Variable):
    def __init__(self, ncdset: Dataset, name: str,
                 longitudes: np.ndarray,
                 latitudes: np.ndarray,
                 times: np.ndarray,
                 units: Optional[str] = "-",
                 time_first=True,
                 numpy_dtype: Optional[str] = DEFAULT_NUMPY_DTYPE,
                 compression: Optional[str] = "zlib",
                 chunksizes: Optional[tuple] = None,
                 fill_value: Optional[float] = DEFAULT_MISSING_VALUE,
                 significant_digit: Optional[int] = DEFAULT_SIGNIFICANT_DIGIT,
                 attrs: Optional[dict] = None):

        sdigit = significant_digit
        if longitudes is not None and latitudes is not None \
                and times is not None:
            dims = OrderedDict()
            if time_first:
                dims[DIMENSION_TIME_NAME] = times
                dims[DIMENSION_LATITUDE_NAME] = latitudes
                dims[DIMENSION_LONGITUDE_NAME] = longitudes
            else:
                dims[DIMENSION_LATITUDE_NAME] = latitudes
                dims[DIMENSION_LONGITUDE_NAME] = longitudes
                dims[DIMENSION_TIME_NAME] = times
        else:
            if time_first:
                dims = [DIMENSION_TIME_NAME,
                        DIMENSION_LATITUDE_NAME,
                        DIMENSION_LONGITUDE_NAME]
            else:
                dims = [DIMENSION_LATITUDE_NAME,
                        DIMENSION_LONGITUDE_NAME,
                        DIMENSION_TIME_NAME]

        super(SpatialTimeVariable, self).__init__(ncdset, name,
                                                  dimensions=dims,
                                                  units=units,
                                                  numpy_dtype=numpy_dtype,
                                                  compression=compression,
                                                  chunksizes=chunksizes,
                                                  fill_value=fill_value,
                                                  significant_digit=sdigit,
                                                  attrs=attrs)
