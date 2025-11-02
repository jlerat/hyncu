"""Utility functions to export netcdf files """
from __future__ import annotations
from typing import Optional, Union
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

from hyncu import HAS_CF_UNITS
if HAS_CF_UNITS:
    import cf_units

from netCDF4 import Dataset

from hyncu.nc4io import STRING_MAX_LENGTH
from hyncu.nc4io import DEFAULT_NUMPY_DTYPE
from hyncu.nc4io import DEFAULT_MISSING_VALUE
from hyncu.nc4io import DEFAULT_SIGNIFICANT_DIGIT
from hyncu.nc4io import DIMENSION_TIME_NAME
from hyncu.nc4io import date2num
from hyncu.nc4io import num2date
from hyncu.nc4io import minimal_metadata
from hyncu.nc4io import Variable
from hyncu.nc4io import remove_non_ascii_vectorized

TEXT_DATA_TYPE_LABEL = "text"
NUMERICAL_DATA_TYPE_LABEL = "numerical"
TIME_DATA_TYPE_LABEL = "time"

DEFAULT_STATION_DATASET_NAME = "station_metadata"
EXPECTED_STATIONS_COLUMNS = ["NAME", "LONGITUDE", "LATITUDE"]

STATIONID_DIMENSION_NAME = "station_id"
STATION_DATA_INDEX_DIMENSION_NAME = "station_data_index"

LABEL_SEPARATOR = "."


def validate_units(units: list) -> None:
    if not HAS_CF_UNITS:
        return

    for unit in units:
        try:
            cf_units.Unit(unit)
        except Exception:
            errmess = f"Unit {unit} is not valid."
            raise ValueError(errmess)


def read_attributes(ncdset: Dataset, dataset_name: str) -> dict[str]:
    attrs = {}
    ndt = ncdset[dataset_name]
    for key in ndt.ncattrs():
        attrs[key] = getattr(ndt, key)

    return attrs


def get_item_index_from_dimension(ncdset: Dataset,
                                  dimension_name: str,
                                  items: np.ndarray) -> np.ndarray:
    if dimension_name not in ncdset.dimensions:
        errmess = f"{dimension_name} is not a dimension."
        raise ValueError(errmess)

    dim = ncdset[dimension_name][:]
    _, idx_nc, idx_data = np.intersect1d(dim, items, return_indices=True)

    if len(idx_nc) == 0 or len(idx_data) == 0:
        errmess = f"No items found in dimension {dimension_name}."
        raise ValueError(errmess)

    return idx_nc, idx_data


def column_variable_ncname(dataset_name, dtype):
    return f"{dataset_name}{LABEL_SEPARATOR}column_name"\
           + f"{LABEL_SEPARATOR}{dtype}"


def station_variable_ncname(dataset_name, dtype):
    return f"{dataset_name}{LABEL_SEPARATOR}{dtype}"


def unit_variable_ncname(dataset_name, dtype):
    return f"{dataset_name}{LABEL_SEPARATOR}column_units"\
           + f"{LABEL_SEPARATOR}{dtype}"


def read_from_nc(ncdset: Dataset, variable_name):
    if variable_name not in ncdset.variables:
        errmess = f"{variable_name} was not found in variables."
        raise ValueError(errmess)
    return ncdset[variable_name][:]


def select_types(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    if data_type == NUMERICAL_DATA_TYPE_LABEL:
        return df.select_dtypes(include="number").astype(np.float32)
    elif data_type == TEXT_DATA_TYPE_LABEL:
        dtype = f"U{STRING_MAX_LENGTH}"
        return df.select_dtypes(include=["object", "datetime"]).astype(dtype)
    else:
        errmess = f"Data type '{data_type}' not recognised."\
                  + f" Expected [{TEXT_DATA_TYPE_LABEL}"\
                  + f"/{NUMERICAL_DATA_TYPE_LABEL}]"
        raise ValueError(errmess)


def dataframe_items_to_dimensions(df):
    dtype = f"U{STRING_MAX_LENGTH}"
    index = pd.Series(df.index).values.astype(dtype)
    index = remove_non_ascii_vectorized(index)

    columns = df.columns.str\
        .replace("\\[.*", "", regex=True)
    columns = columns.values.astype(dtype)
    columns = remove_non_ascii_vectorized(columns)
    return index, columns


class StationMetaData():
    def __init__(self, ncdset: Dataset,
                 metadata: Optional[pd.DataFrame] = None,
                 name: Optional[str] = DEFAULT_STATION_DATASET_NAME,
                 metadata_column_units: Optional[list] = None,
                 attrs: Optional[dict] = None):
        self.name = name
        self.ncdset = ncdset
        self.metadata = metadata
        self.attrs = minimal_metadata(attrs)

        if metadata is None:
            # Data retrieval mode
            return

        # Check columns in metadata
        self.check_expected_metadata_columns()

        # Create variables and write metadata to dataset
        for data_type in [TEXT_DATA_TYPE_LABEL,
                          NUMERICAL_DATA_TYPE_LABEL]:
            self.write_metadata_data_to_dataset(data_type)
            self.write_metadata_units_to_dataset(data_type)

    def check_expected_metadata_columns(self):
        metadata_column_names = [re.sub("\\[.*", "", cn)
                                 for cn in self.metadata.columns]
        for n in EXPECTED_STATIONS_COLUMNS:
            if n not in metadata_column_names:
                errmess = f"{n} was expected in meta data column names."
                raise ValueError(errmess)

    def write_metadata_data_to_dataset(self, data_type):
        ncdset = self.ncdset
        name = self.name
        metadata_typed = select_types(self.metadata, data_type)

        # Create data dimensions
        dims = OrderedDict()

        index, columns = dataframe_items_to_dimensions(metadata_typed)
        dims[STATIONID_DIMENSION_NAME] = index

        column_dimname = column_variable_ncname(name, data_type)
        dims[column_dimname] = columns

        # Create data variable
        if data_type == TEXT_DATA_TYPE_LABEL:
            compression = None
            fill_value = "-"
            numpy_dtype = f"U{STRING_MAX_LENGTH}"
            sdigit = None
        elif data_type == NUMERICAL_DATA_TYPE_LABEL:
            compression = None
            fill_value = np.nan
            numpy_dtype = DEFAULT_NUMPY_DTYPE
            sdigit = 7

        metadata_varname = station_variable_ncname(name, data_type)
        nvar = Variable(ncdset, metadata_varname,
                        dimensions=dims,
                        units=None,
                        numpy_dtype=numpy_dtype,
                        compression=compression,
                        fill_value=fill_value,
                        significant_digit=sdigit,
                        attrs=self.attrs)
        nvar.write_data_to_dataset(metadata_typed)

    def write_metadata_units_to_dataset(self, data_type):
        ncdset = self.ncdset
        name = self.name
        metadata_typed = select_types(self.metadata, data_type)

        # Extract units from columns
        metadata_units = [re.sub(".*\\[|\\]$", "", cn)
                          if re.search("\\[", cn) else ""
                          for cn in metadata_typed.columns]
        metadata_units = np.array(metadata_units)

        # Create unit variable
        unit_varname = unit_variable_ncname(name, data_type)
        column_dimname = column_variable_ncname(name, data_type)
        numpy_dtype = f"U{STRING_MAX_LENGTH}"
        nvar = Variable(ncdset, unit_varname,
                        dimensions=[column_dimname],
                        units=None,
                        numpy_dtype=numpy_dtype,
                        compression=None,
                        fill_value=None,
                        significant_digit=None)
        nvar.write_data_to_dataset(metadata_units)

    def read_metadata_from_dataset(self):
        name = self.name
        ncdset = self.ncdset
        metadata = []
        for data_type in [TEXT_DATA_TYPE_LABEL,
                          NUMERICAL_DATA_TYPE_LABEL]:
            metadata_varname = station_variable_ncname(name, data_type)
            metadata_typed = ncdset[metadata_varname][:]

            stationids = read_from_nc(ncdset, STATIONID_DIMENSION_NAME)

            column_dimname = column_variable_ncname(name, data_type)
            metadata_columns = read_from_nc(ncdset, column_dimname)

            unit_varname = unit_variable_ncname(name, data_type)
            metadata_units = read_from_nc(ncdset, unit_varname)
            metadata_columns = [f"{n}[{u}]" if u != "" else n for n, u in
                                zip(metadata_columns, metadata_units)]

            # Generate dataframe
            metadata_typed = pd.DataFrame(metadata_typed, index=stationids,
                                          columns=metadata_columns)
            metadata.append(metadata_typed)

            attrs = read_attributes(ncdset, metadata_varname)

        metadata = pd.concat(metadata, axis=1)
        return metadata, attrs


class StationVariable(Variable):
    def __init__(self, ncdset: Dataset,
                 name: str,
                 stationids: Optional[Union[list, pd.Index]] = None,
                 index: Optional[pd.Index] = None,
                 column_names: Optional[list] = None,
                 column_units: Optional[list] = None,
                 numpy_dtype: Optional[str] = DEFAULT_NUMPY_DTYPE,
                 compression: Optional[str] = "zlib",
                 chunksizes: Optional[tuple] = None,
                 fill_value: Optional[float] = DEFAULT_MISSING_VALUE,
                 significant_digit: Optional[int] = DEFAULT_SIGNIFICANT_DIGIT,
                 attrs: Optional[dict] = None):

        # Storing only numerical data
        self.data_type = NUMERICAL_DATA_TYPE_LABEL

        # Set default chunksizes
        if chunksizes is None and stationids is not None\
                and index is not None and column_names is not None:
            chunksizes = (1, min(5000, len(index)),
                          len(column_names))

        # Data dimensions : sites x index x variable
        dtype = self.data_type
        colvar_ncname = column_variable_ncname(name, dtype)
        if stationids is None or index is None \
                or column_names is None:
            dims = [STATIONID_DIMENSION_NAME,
                    STATION_DATA_INDEX_DIMENSION_NAME,
                    colvar_ncname]
        else:
            dims = OrderedDict()
            dims[STATIONID_DIMENSION_NAME] = stationids
            dims[STATION_DATA_INDEX_DIMENSION_NAME] = index
            dims[colvar_ncname] = column_names

        # Initialise station variable
        var_ncname = station_variable_ncname(name, dtype)
        sdigit = significant_digit
        super(StationVariable, self).__init__(ncdset, var_ncname,
                                              dimensions=dims,
                                              units="-",
                                              numpy_dtype=numpy_dtype,
                                              compression=compression,
                                              chunksizes=chunksizes,
                                              fill_value=fill_value,
                                              significant_digit=sdigit,
                                              attrs=attrs)

        # Careful here, self.name is initialise to var_ncname
        self.name = name
        self.variable_ncname = var_ncname

        # Dealing with units
        if column_units is not None:
            unit_var_ncname = unit_variable_ncname(name, dtype)
            if unit_var_ncname not in ncdset.variables:
                self.write_column_units_to_dataset(column_units)

    def get_column_names(self, add_units: Optional[bool] = True):
        ncdset = self.ncdset
        name = self.name
        dtype = self.data_type

        colvar_ncname = column_variable_ncname(name, dtype)
        colnames = read_from_nc(ncdset, colvar_ncname)
        if add_units:
            unit_var_ncname = unit_variable_ncname(name, dtype)
            try:
                units = read_from_nc(ncdset, unit_var_ncname)
                colnames = [v if u in ["", "-"] else f"{v}[{u}]"
                            for v, u in zip(colnames, units)]
            except ValueError:
                pass

        return colnames

    def write_column_units_to_dataset(self, column_units):
        name = self.name
        ncdset = self.ncdset
        dtype = self.data_type

        # Check number of units
        dtype = NUMERICAL_DATA_TYPE_LABEL
        colvar_ncname = column_variable_ncname(name, dtype)
        ncolumns = ncdset[colvar_ncname].shape[0]
        if len(column_units) != ncolumns:
            errmess = f"Expected {ncolumns} units, "\
                      + f"got {len(column_units)}."
            raise ValueError(errmess)

        # Set variable
        unit_var_ncname = unit_variable_ncname(name, dtype)
        numpy_dtype = f"U{STRING_MAX_LENGTH}"
        column_units = np.array(column_units).astype(numpy_dtype)
        unit_var = Variable(ncdset, unit_var_ncname,
                            dimensions=[colvar_ncname],
                            compression=None,
                            numpy_dtype=numpy_dtype)
        unit_var.write_data_to_dataset(column_units)

    def write_data_for_single_station(self, stationid: str,
                                      data: pd.DataFrame) -> None:
        name = self.name
        dtype = self.data_type
        ncdset = self.ncdset
        istation, _ = get_item_index_from_dimension(ncdset,
                                                    STATIONID_DIMENSION_NAME,
                                                    [stationid])
        istation = istation[0]
        variable_ncname = self.variable_ncname
        obj = ncdset[variable_ncname]
        nsites, nindex, nvars = obj.shape

        if isinstance(data, pd.DataFrame):
            try:
                index = date2num(data.index)
            except Exception:
                index = data.index.values

            dname = STATION_DATA_INDEX_DIMENSION_NAME
            iindex_nc, iindex_data = \
                get_item_index_from_dimension(ncdset, dname, index)

            column_names = [re.sub("\\[.*", "", cn)
                            for cn in data.columns.values]
            dname = column_variable_ncname(name, dtype)
            ivars_nc, ivars_data = \
                get_item_index_from_dimension(ncdset, dname, column_names)
        else:
            if data.shape != (nindex, nvars):
                errmess = f"Expected data of size ({nindex}, {nvars}), "\
                          + f"got {data.shape}."
                raise ValueError(errmess)

            iindex_nc = np.arange(nindex)
            iindex_data = iindex_nc
            ivars_nc = np.arange(nvars)
            ivars_data = ivars_nc

        # Write data to netcdf only for selected part of the dataset
        tostore = np.array(data)[iindex_data[:, None], ivars_data[None, :]]
        obj[istation, iindex_nc, ivars_nc] = tostore

    def read_data_from_single_station(self, stationid: str,
                                      clip: Optional[bool] = True):
        # Get attributes
        ncdset = self.ncdset
        variable_ncname = self.variable_ncname
        attrs = read_attributes(ncdset, variable_ncname)

        # Find station index
        dname = STATIONID_DIMENSION_NAME
        istation, _ = get_item_index_from_dimension(ncdset,
                                                    dname, [stationid])
        if len(istation) == 0:
            errmess = f"Cannot find stationid {stationid}."
            raise ValueError(errmess)

        istation = istation[0]

        # Build dataframe
        colnames = self.get_column_names()

        dname = STATION_DATA_INDEX_DIMENSION_NAME
        index = read_from_nc(ncdset, dname)
        if ncdset[dname].dimension_type == DIMENSION_TIME_NAME:
            index = num2date(index)

        data = ncdset[variable_ncname][istation, :, :]

        df = pd.DataFrame(data, index=index,
                          columns=colnames)
        if clip:
            index_ok = df.notnull().any(axis=1)
            var_ok = df.notnull().any(axis=0)
            df = df.loc[index_ok, var_ok]

            if 0 in df.shape:
                raise ValueError("No station data.")

        return df, attrs
