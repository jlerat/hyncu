"""Utility functions to export netcdf files """
from __future__ import annotations
from typing import Optional
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

import cf_units
from netCDF4 import Dataset

from hyncu import nc4io

DEFAULT_STATION_DATASET_NAME = "station_metadata"
EXPECTED_STATIONS_COLUMNS = ["NAME", "LONGITUDE", "LATITUDE"]

STATIONID_DIMENSION_NAME = "stationid"
STATION_DATA_INDEX_DIMENSION_NAME = "index"

TEXT_DATA_LABEL = "text"
NUMERICAL_DATA_LABEL = "numerical"
STRING_MAX_LENGTH = 50

LABEL_SEPARATOR = "."


def validate_units(units: list) -> None:
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


def column_variable_ncname(dataset_name, dlabel):
    return f"{dataset_name}{LABEL_SEPARATOR}column_name"\
           + f"{LABEL_SEPARATOR}{dlabel}"


def station_variable_ncname(dataset_name, dlabel):
    return f"{dataset_name}{LABEL_SEPARATOR}{dlabel}"


def unit_variable_ncname(dataset_name, dlabel):
    return f"{dataset_name}{LABEL_SEPARATOR}column_units"\
           + f"{LABEL_SEPARATOR}{dlabel}"


def read_from_nc(ncdset: Dataset, variable_name):
    if variable_name not in ncdset.variables:
        errmess = f"{variable_name} was not found in variables."
        raise ValueError(errmess)
    return ncdset[variable_name][:]


def get_dataset_variable_names(ncdset: Dataset, dataset_name: str,
                             add_units: Optional[bool] = True) -> list:
    varnames = read_from_nc(ncdset, ncvariable_name(dataset_name))
    units = read_from_nc(ncdset, unit_ncname(dataset_name))
    colnames = []
    for v, u in zip(varnames, units):
        cn = f"{v}[{u}]" if add_units else v
        colnames.append(cn)
    return colnames


def select_types(df: pd.DataFrame, dlabel: str) -> pd.DataFrame:
    if dlabel == NUMERICAL_DATA_LABEL:
        return df.select_dtypes(include="number").astype(np.float32)
    elif dlabel == TEXT_DATA_LABEL:
        return df.select_dtypes(include=["object", "datetime"]).astype(str)
    else:
        errmess = f"dlabel {dlabel} not recognised."\
                    f" Expected [{TEXT_DATA_LABEL}/{NUMERICAL_DATA_LABEL}]"
        raise ValueError(errmess)


class StationMetaData():
    def __init__(self, ncdset: Dataset,
                 metadata: Optional[pd.DataFrame] = None,
                 name: Optional[str] = DEFAULT_STATION_DATASET_NAME,
                 attrs: Optional[dict] = None):
        self.name = name
        self.ncdset = ncdset
        self.metadata = metadata
        self.attrs = nc4io.minimal_metadata(attrs)

        if metadata is not None and units is not None:
            if len(units) != metadata.shape[1]:
                errmess = "Expected same number of columns in metadata"\
                          + " than units."
                raise ValueError(errmess)

        self.units = units

        if metadata is not None:
            self.check_expected_metadata_columns()
            self.select_metadata_types()
            self.write_stationid_dimension_to_dataset()
            self.write_metadata_column_dimension_to_dataset()
            self.write_metadata_to_dataset()

    def check_expected_metadata_columns(self):
        metadata_column_names = [re.sub("\\[.*", "", cn)
                                 for cn in self.metadata.columns]
        for n in EXPECTED_STATIONS_COLUMNS:
            if n not in metadata_column_names:
                errmess = f"{n} was expected in meta data column names."
                raise ValueError(errmess)

    def select_metadata_types(self):
        self._metadata_typed = {}
        for dlabel in [TEXT_DATA_LABEL, NUMERICAL_DATA_LABEL]:
            self._metadata_typed[dlabel] = select_types(self.metadata, dlabel)

    def write_stationid_dimension_to_dataset(self):
        stationids = np.array(self.metadata.index).astype(str)
        nc4io.add_dimension(self.ncdset, STATIONID_DIMENSION_NAME,
                            stationids)

    def write_metadata_column_dimension_to_dataset(self, dlabel):
        column_names = self._metadata_typed[dlabel].columns.str\
                     .replace("\\[.*", "", regex=True).values.astype(str)
        column_dimname = column_variable_ncname(self.name, dlabel)
        nc4io.add_dimension(self.ncdset, column_dimname, column_names)
        desc = "Variable storing the list of meta data"\
               + f" columns for type {dlabel}."
        ncdset[column_dimname].description = desc

    def write_metadata_column_variable_to_dataset(self, dlabel):
        metadata_varname = station_variable_ncname(self.name, dlabel)
        column_dimname = column_variable_ncname(self.name, dlabel)
        dims = [STATIONID_DIMENSION_NAME, column_dimname]

        if dlabel == NUMERICAL_DATA_LABEL:
            dtype = np.float32
            fill_value = np.nan
            sdt = 7
        else:
            dtype = f"S{STRING_MAX_LENGTH}"
            fill_value = "NA"
            sdt = None

        nc4io.Variable(self.ncdset, metadata_varname,
                       dimensions=dims,
                       units=units,
                       dtype=dtype,
                       compression=None,
                       fill_value=fill_value,
                       significant_digit=sdt,
                       attrs=attrs)

    def write_metadata_to_dataset(self, dlabel):
        metadata_varname = station_variable_ncname(self.name, dlabel)
        self.ncdset[metadata_varname][:] = self._metadata_typed[dlabel].values



class StationVariable(nc4io.Variable):
    def __init__(self, ncdset: Dataset,
                 name: str,
                 stationids: Union[list, pd.Index],
                 index: pd.Index,
                 column_names: list,
                 column_units: Optional[list] = None,
                 dtype: Optional[str] = nc4io.DEFAULT_DTYPE,
                 compression: Optional[str] = "zlib",
                 chunksizes: Optional[tuple] = None,
                 fill_value: Optional[float] = nc4io.DEFAULT_MISSING_VALUE,
                 significant_digit: Optional[int] = nc4io.DEFAULT_SIGNIFICANT_DIGIT,
                 attrs: Optional[dict] = None):

        # Storing only numerical data
        self.dlabel = NUMERICAL_DATA_LABEL

        # Data dimensions : sites x index x variable
        dims = OrderedDict()
        dims[STATIONID_DIMENSION_NAME] = stationids
        dims[STATION_DATA_INDEX_DIMENSION_NAME] = index
        dlabel = NUMERICAL_DATA_LABEL
        colvar_ncname = column_variable_ncname(name, dlabel)
        dims[colvar_ncname] = column_names

        sdt = significant_digit

        var_ncname = station_variable_ncname(name, dlabel)
        super(StationVariable, self).__init__(ncdset, var_ncname,
                                              dimensions=dims,
                                              units="-",
                                              dtype=dtype,
                                              compression=compression,
                                              chunksizes=chunksizes,
                                              fill_value=fill_value,
                                              significant_digit=sdt,
                                              attrs=attrs)
        # dodgy.. have to overwrite self.name to avoid using varname
        self.name = name
        self.variable_ncname = var_ncname

        unit_var_ncname = unit_variable_ncname(self.name, self.dlabel)
        self.column_units = np.array(column_units)
        if unit_var_ncname not in ncdset.variables \
                and column_units is not None:
            self.write_units_to_dataset()

    def write_units_to_dataset(self):
        ncdset = self.ncdset
        units = self.column_units

        # Check number of units
        dlabel = NUMERICAL_DATA_LABEL
        colvar_ncname = column_variable_ncname(self.name, self.dlabel)
        ncolumns = ncdset[colvar_ncname].shape[0]
        if len(units) != ncolumns:
            errmess = f"Expected {ncolumns} units, got {len(units)}."
            raise ValueError(errmess)

        # Set variable
        unit_var_ncname = unit_variable_ncname(self.name, dlabel)
        dtype = f"S{STRING_MAX_LENGTH}"
        unit_var = nc4io.Variable(ncdset, unit_var_ncname,
                                  dimensions=[colvar_ncname],
                                  compression=None,
                                  dtype=dtype)
        # Bugging
        #ncdset[unit_var_ncname][:] = self.column_units


    def write_data_for_single_station(self, stationid: str,
                                  data: pd.DataFrame) -> None:
        istation, _ = get_item_index_from_dimension(self.ncdset,
                                                    STATIONID_DIMENSION_NAME,
                                                    [stationid])
        istation = istation[0]
        nsites, nindex, nvars = ncdset[self.name].shape

        if hasattr(data, STATION_DATA_INDEX_DIMENSION_NAME):
            # Data is assumed to be a dataframe
            try:
                index = nc4io.date2num(data.index)
            except Exception:
                index = data.index.values
            iindex_nc, iindex_data = get_item_index_from_dimension(ncdset,
                                                                   "index",
                                                                   index)
            variable_names = [re.sub("\\[.*", "", cn) for cn in data.columns.values]
            dname = ncvariable_name(dataset_name)
            ivars_nc, ivars_data = get_item_index_from_dimension(ncdset,
                                                                 dname,
        else:
            if data.shape != (nindex, nvars):
                errmess = f"Expected data of size ({nindex}, {nvars}), "\
                          + f"got {data.shape}."
                raise ValueError(errmess)

            iindex_nc = np.arange(nindex)
            iindex_data = iindex_nc
            ivars_nc = np.arange(nvars)
            ivars_data = ivars_nc

        # Write data to netcdf only for the common portion
        tostore = np.array(data)[iindex_data[:, None], ivars_data[None, :]]
        ncdset[self.variable_ncname][istation, iindex_nc, ivars_nc] = tostore

    def read_data_from_single_station(stationid: str,
                                 clip: Optional[bool] = True):
        # Get attributes
        attrs = read_attributes(self.ncdset, self.name)

        # Find station index
        istation, _ = get_item_index_from_dimension(ncdset,
                                                    "stationid",
                                                    [stationid])
        # Build dataframe
        istation = istation[0]
        colnames = get_dataset_variable_names(ncdset, dataset_name)
        index = read_from_nc(ncdset, "index")
        istime = ncdset["index"].dimension_type == "time"
        if istime:
            index = nc4io.num2date(index)

        df = pd.DataFrame(ncdset[dataset_name][istation, :, :],
                          index=index,
                          columns=colnames)
        if clip:
            index_ok = df.notnull().any(axis=1)
            var_ok = df.notnull().any(axis=0)
            df = df.loc[index_ok, var_ok]

            if 0 in df.shape:
                raise ValueError("No station data.")

        return df, attrs
