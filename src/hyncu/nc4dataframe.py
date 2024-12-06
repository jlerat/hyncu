"""Utility functions to export netcdf files """
from __future__ import annotations
from typing import Optional
import re, sys

import numpy as np
import pandas as pd

import cf_units
from netCDF4 import Dataset

from hyncu import nc4io

STATION_DATASET_NAME = "stations"
EXPECTED_STATIONS_COLUMNS = ["NAME", "LONGITUDE", "LATITUDE"]

TEXT_DATA_LABEL = "text"
NUMERICAL_DATA_LABEL = "numerical"

LABEL_SEPARATOR = "#"

def validate_units(units: list) -> None:
    for unit in units:
        try:
            cf_units.Unit(unit)
        except:
            errmess = f"Unit {unit} is not valid."
            raise ValueError(errmess)


def get_item_index_from_dimension(nc4dset: Dataset, \
                                            dimension_name: str, \
                                            items: np.ndarray) -> np.ndarray:
    if not dimension_name in nc4dset.dimensions:
        errmess = f"{dimension_name} is not a dimension."
        raise ValueError(errmess)

    dim = nc4dset[dimension_name][:]
    _, idx_nc, idx_data = np.intersect1d(dim, items, return_indices=True)

    if len(idx_nc)==0 or len(idx_data)==0:
        errmess = f"No items found in dimension {dimension_name}."
        raise ValueError(errmess)

    return idx_nc, idx_data


def data_variable_name(dataset_name):
    return f"{dataset_name}{LABEL_SEPARATOR}variable"


def station_data_name(dlabel):
    return f"{STATION_DATASET_NAME}{LABEL_SEPARATOR}{dlabel}_data"


def station_data_variable_name(dlabel):
    return data_variable_name(station_data_name(dlabel))


def data_unit_name(dataset_name):
    return f"{dataset_name}{LABEL_SEPARATOR}variable"\
            +f"{LABEL_SEPARATOR}units"


def station_data_unit_name(dlabel):
    return data_unit_name(station_data_name(dlabel))


def read_from_nc(nc4dset: Dataset, variable_name):
    if not variable_name in nc4dset.variables:
        errmess = f"{variable_name} was not found in variables."\
                    +" Make sure the function nc4dataframe.add_variables"\
                    +" has been run."
        raise ValueError(errmess)
    return nc4dset[variable_name][:]


def get_dataset_column_names(nc4dset:Dataset, dataset_name: str):
    varnames = read_from_nc(nc4dset, data_variable_name(dataset_name))
    units = read_from_nc(nc4dset, data_unit_name(dataset_name))
    colnames = []
    for v, u in zip(varnames, units):
        cn = f"{v}[{u}]"
        colnames.append(cn)
    return colnames


def add_variables(nc4dset: Dataset, \
                    dataset_name: str, \
                    stationids: list, \
                    variables: list, \
                    times: list,\
                    units: Optional[str] = None, \
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
        return df.select_dtypes(include="number").astype(np.float32)
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
        if not np.all(stationids == nc4dset["stationid"][:]):
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
        info = read_from_nc(nc4dset, dset)
        colnames = get_dataset_column_names(nc4dset, dset)
        df.append(pd.DataFrame(info, index=stationids, columns=colnames))
    return pd.concat(df, axis=1)


def set_data(nc4dset: Dataset, \
                dataset_name: str, \
                stationid: str, \
                data: pd.DataFrame):
    istation, _ = get_item_index_from_dimension(nc4dset, "stationid",\
                                                    [stationid])
    istation = istation[0]

    nsites, ntimes, nvars = nc4dset[dataset_name].shape

    if hasattr(data, "index"):
        # Data is assumed to be a dataframe
        times = nc4io.date2num(data.index)
        itimes_nc, itimes_data = get_item_index_from_dimension(nc4dset, "time",\
                                                                times)
        variables = data.columns.values
        vname = data_variable_name(dataset_name)
        ivars_nc, ivars_data = get_item_index_from_dimension(nc4dset, vname,\
                                                                variables)
    else:
        if data.shape != (ntimes, nvars):
            errmess = f"Expected data of size ({ntimes}, {nvars}), "\
                        +f"got {data.shape}."
            raise ValueError(errmess)

        itimes_nc = np.arange(ntimes)
        itimes_data = itimes_nc
        ivars_nc = np.arange(nvars)
        ivars_data = ivars_nc

    tostore = np.array(data)[itimes_data[:, None], ivars_data[None, :]]
    nc4dset[dataset_name][istation, itimes_nc, ivars_nc] = tostore


def get_data(nc4dset: Dataset, \
                dataset_name: str, \
                stationid: str, \
                clip: Optional[bool] = True):
    istation, _ = get_item_index_from_dimension(nc4dset, "stationid", [stationid])
    istation = istation[0]
    colnames = get_dataset_column_names(nc4dset, dataset_name)
    times = nc4io.num2date(read_from_nc(nc4dset, "time"))

    df = pd.DataFrame(nc4dset[dataset_name][istation, :, :], index=times, \
                            columns=colnames)

    if clip:
        time_ok = df.notnull().any(axis=1)
        var_ok = df.notnull().any(axis=0)
        df = df.loc[time_ok, var_ok]

    return df

