"""Utility functions to export netcdf files """
from __future__ import annotations
from typing import Optional
import re, sys

import numpy as np
import pandas as pd

import cf_units
from netCDF4 import Dataset

from hyncu import nc4io

DEFAULT_DATASET_NAME = "data"
STATION_DATASET_NAME = "stations"
EXPECTED_STATIONS_COLUMNS = ["NAME", "LONGITUDE", "LATITUDE"]

TEXT_DATA_LABEL = "text"
NUMERICAL_DATA_LABEL = "numerical"

NAME_SEPARATOR = "#"

def validate_units(units: list) -> None:
    for unit in units:
        try:
            cf_units.Unit(unit)
        except:
            errmess = f"Unit {unit} is not valid."
            raise ValueError(errmess)


def get_item_index_from_dimension(nc4dset: Dataset, \
                                            dimension_name: str, \
                                            item: str) -> int:
    if not dimension_name in nc4dset.dimensions:
        errmess = f"{dimension_name} is not a dimension."
        raise ValueError(errmess)

    dim = nc4dset[dimension_name][:]
    if not item in dim:
        errmess = f"{item} is not in dimension {dimension_name}."
        raise ValueError(errmess)

    return np.where(item==dim)[0][0]


def data_variable_name(dataset_name):
    return f"{dataset_name}{NAME_SEPARATOR}variable"


def station_data_name(dlabel):
    return f"{STATION_DATASET_NAME}{NAME_SEPARATOR}{dlabel}_data"


def station_data_variable_name(dlabel):
    return data_variable_name(station_data_name(dlabel))


def data_unit_name(dataset_name):
    return f"{dataset_name}{NAME_SEPARATOR}variable"\
            +f"{NAME_SEPARATOR}units"


def station_data_unit_name(dlabel):
    return data_unit_name(station_data_name(dlabel))


def get_dataset_column_names(nc4dset:Dataset, dataset_name: str):
    varnames = nc4dset[data_variable_name(dataset_name)][:]
    units = nc4dset[data_unit_name(dataset_name)][:]
    colnames = []
    for v, u in zip(varnames, units):
        cn = f"{v}[{u}]"
        colnames.append(cn)
    return colnames


def get_units_from_columns(columns: list) -> list:
    return [re.sub(".*\\[|\\]$", "", cn) for cn in columns]


def add_variables(nc4dset: Dataset, \
                    stationids: list, \
                    variables: list, \
                    times: list,\
                    units: Optional[str] = None, \
                    dataset_name: Optional[str] = DEFAULT_DATASET_NAME, \
                    attrs: Optional[dict] = None):
    # Create dimensions
    kw = {data_variable_name(dataset_name): variables}
    nc4io.add_dimensions(nc4dset, time=times, stationid=stationids, **kw)

    # Create data variable
    var = nc4io.Variable(dataset_name, ["stationid", "time",
                                            data_variable_name(dataset_name)], \
                                units="-", \
                                chunksizes=(1, min(10000, len(times)), \
                                                min(10, len(variables))), \
                                attrs=attrs)
    var.to_dataset(nc4dset)

    # Units
    units_var = nc4io.Variable(data_unit_name(dataset_name), \
                        [data_variable_name(dataset_name)], units="-", \
                        dtype=str, fill_value="NA", \
                        significant_digit=None, \
                        compression=None, \
                        attrs={"description": "Units for each data variable"})
    units_var.to_dataset(nc4dset)
    if units is None:
        units = ["-"]*len(variables)

    units = np.array(units)
    validate_units(units)

    if len(units) != len(variables):
        errmess = f"Expected same number of variables ({len(variables)})"\
                    f" and units ({len(units)})"
        raise ValueError(errmess)
    nc4dset[data_unit_name(dataset_name)][:] = units

    # Meta data
    nc4io.add_meta(nc4dset)


def select_types(df: pd.DataFrame, dlabel: str) -> pd.DataFrame:
    if dlabel == NUMERICAL_DATA_LABEL:
        return df.select_dtypes(include="number")
    elif dlabel == TEXT_DATA_LABEL:
        return df.select_dtypes(include=["object", "datetime"]).astype(str)
    else:
        errmess = f"dlabel {dlabel} not recognised."\
                    f" Expected [{TEXT_DATA_LABEL}/{NUMERICAL_DATA_LABEL}]"
        raise ValueError(errmess)


def check_expected_stations_columns(columns: list):
    variables = [re.sub("\\[.*", "", cn) for cn in columns]
    for n in EXPECTED_STATIONS_COLUMNS:
        if not n in variables:
            errmess = f"{n} was expected in station variables"
            raise ValueError(errmess)


def configure_stations_variables(nc4dset: Dataset, info: pd.DataFrame, dlabel: str) -> None:
    variable_names = info.columns.str.replace("\\[.*", "", regex=True).values.astype(str)

    vdim_name = station_data_variable_name(dlabel)
    vdim = nc4io.Dimension(vdim_name, variable_names, variable_names.dtype, "-")
    vdim.to_dataset(nc4dset)


def configure_stations_units(nc4dset: Dataset, info: pd.DataFrame, dlabel: str) -> None:
    if dlabel == TEXT_DATA_LABEL:
        units = np.array(["-"]*info.shape[1])
    else:
        cols = info.columns.str
        units = cols.replace(".*\\[|\\]$", "", regex=True).values.astype(str)
        variables = cols.replace("\\[.*", "", regex=True).values.astype(str)
        units[units==variables] = "-"

    validate_units(units)

    uname = station_data_unit_name(dlabel)
    vdim_name = station_data_variable_name(dlabel)
    un = nc4io.Variable(uname, [vdim_name], units="-", \
                        dtype=units.dtype, fill_value="NA", \
                        significant_digit=None, \
                        compression=None, \
                        attrs={"description": "Units for each station info column"})
    un.to_dataset(nc4dset)
    units = np.array(units)
    nc4dset[uname][:] = units


def store_stations_data(nc4dset: Dataset, info: pd.DataFrame, dlabel: str) -> None:
    if dlabel == NUMERICAL_DATA_LABEL:
        values = info.values
        fill_value = np.nan
        digit = 7
    else:
        values = info.values.astype(str)
        fill_value = "NA"
        digit = None

    dtype = values.dtype
    attrs = dict(description=f"Stations / {dlabel} data")
    dset = station_data_name(dlabel)
    vdim_name = station_data_variable_name(dlabel)
    var = nc4io.Variable(dset, ["stationid", vdim_name], \
                                units="-", dtype=dtype, \
                                fill_value=fill_value, \
                                significant_digit=digit, \
                                chunksizes=values.shape, \
                                compression=None, \
                                attrs=attrs)
    var.to_dataset(nc4dset)
    nc4dset[dset][:] = values


def set_station_ids(nc4dset: Dataset, stations: pd.DataFrame) -> None:
    stationids = np.array(stations.index.values).astype(str)
    if not "stationid" in nc4dset.dimensions:
        dim = nc4io.Dimension("stationid", stationids, stationids.dtype, "-")
        dim.to_dataset(nc4dset)
    else:
        if not stationids == nc4dset["stationid"][:]:
            errmess = f"Station IDs should be identical to the one "\
                        "stored in the 'stationid' dimension"
            raise ValueError(errmess)


def set_station_data(nc4dset: Dataset, stations: pd.DataFrame) -> None:
    check_expected_stations_columns(stations.columns)
    set_station_ids(nc4dset, stations)

    for dlabel in [TEXT_DATA_LABEL, NUMERICAL_DATA_LABEL]:
        info = select_types(stations, dlabel)
        configure_stations_variables(nc4dset, info, dlabel)
        configure_stations_units(nc4dset, info, dlabel)
        store_stations_data(nc4dset, info, dlabel)


def get_station_data(nc4dset: Dataset):
    df = []
    stationids = nc4dset["stationid"][:]
    for dlabel in [NUMERICAL_DATA_LABEL, TEXT_DATA_LABEL]:
        dset = station_data_name(dlabel)
        info = nc4dset[dset][:]
        colnames = get_dataset_column_names(nc4dset, dset)
        df.append(pd.DataFrame(info, index=stationids, columns=colnames))
    return pd.concat(df, axis=1)


def set_data(nc4dset, stationid, data, dataset_name=DEFAULT_DATASET_NAME):
    istation = get_item_index_from_dimension(nc4dset, "stationid", stationid)
    data = np.array(data)

    _, ntimes, nvar = nc4dset[dataset_name].shape
    if data.shape != (ntimes, nvar):
        errmess = f"Expected data of size [{ntimes}x{nvar}], got {data.shape}"
        raise ValueError(errmess)

    nc4dset[dataset_name][istation, :, :] = data


def get_data(nc4dset, stationid, dataset_name=DEFAULT_DATASET_NAME):
    istation = get_item_index_from_dimension(nc4dset, "stationid", stationid)
    colnames = get_dataset_column_names(nc4dset, dataset_name)
    tdim = nc4io.TimeDimension.from_dataset(nc4dset)
    times = tdim.values
    df = pd.DataFrame(nc4dset[dataset_name][istation, :, :], index=times, \
                            columns=colnames)
    return df
