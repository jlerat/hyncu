"""Utility functions to export netcdf files """
from __future__ import annotations
from typing import Optional
import re

import numpy as np
import pandas as pd

import cf_units
from netCDF4 import Dataset

from hyncu import nc4io

DEFAULT_STATION_DATASET_NAME = "stations"
EXPECTED_STATIONS_COLUMNS = ["NAME", "LONGITUDE", "LATITUDE"]

TEXT_DATA_LABEL = "text"
NUMERICAL_DATA_LABEL = "numerical"

LABEL_SEPARATOR = "."


def validate_units(units: list) -> None:
    for unit in units:
        try:
            cf_units.Unit(unit)
        except Exception:
            errmess = f"Unit {unit} is not valid."
            raise ValueError(errmess)


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


def check_datasetname(dataset_name):
    if LABEL_SEPARATOR in dataset_name:
        errmess = f"Dataset name '{dataset_name}' contains the"\
                   + f" separation character '{LABEL_SEPARATOR}'."
        raise ValueError(errmess)


def data_variable_name(dataset_name):
    return f"{dataset_name}{LABEL_SEPARATOR}variable_name"


def station_data_name(station_dataset_name, dlabel):
    return f"{station_dataset_name}{LABEL_SEPARATOR}{dlabel}"


def station_data_variable_name(station_dataset_name, dlabel):
    sdn = station_data_name(station_dataset_name, dlabel)
    return data_variable_name(sdn)


def data_unit_name(dataset_name):
    return f"{dataset_name}{LABEL_SEPARATOR}variable_unit"


def station_data_unit_name(station_dataset_name, dlabel):
    sdn = station_data_name(station_dataset_name, dlabel)
    return data_unit_name(sdn)


def read_from_nc(ncdset: Dataset, variable_name):
    if variable_name not in ncdset.variables:
        errmess = f"{variable_name} was not found in variables."\
                    + " Make sure the function nc4dataframe.add_variables"\
                    + " has been run."
        raise ValueError(errmess)
    return ncdset[variable_name][:]


def get_dataset_column_names(ncdset: Dataset, dataset_name: str) -> list:
    varnames = read_from_nc(ncdset, data_variable_name(dataset_name))
    units = read_from_nc(ncdset, data_unit_name(dataset_name))
    colnames = []
    for v, u in zip(varnames, units):
        cn = f"{v}[{u}]"
        colnames.append(cn)
    return colnames


def add_variables(ncdset: Dataset,
                  dataset_name: str,
                  stationids: np.ndarray,
                  variables: np.ndarray,
                  index: np.ndarray,
                  units: Optional[str] = None,
                  attrs: Optional[dict] = None):

    check_datasetname(dataset_name)

    # Create dimensions
    kw = {data_variable_name(dataset_name): variables}
    nc4io.add_dimensions(ncdset, index=index, stationid=stationids, **kw)

    # Create data variable
    varnames = data_variable_name(dataset_name)
    var = nc4io.Variable(dataset_name,
                         ["stationid", "index", varnames],
                         units="-",
                         chunksizes=(1, min(10000, len(index)),
                                     min(10, len(variables))),
                         attrs=attrs)
    var.to_dataset(ncdset)

    # Units
    units_var = nc4io.Variable(data_unit_name(dataset_name),
                               [varnames],
                               units="-",
                               dtype=str, fill_value="NA",
                               significant_digit=None,
                               compression=None,
                               attrs={"description":
                                      "Units for each data variable"})
    units_var.to_dataset(ncdset)
    if units is None:
        units = ["-"]*len(variables)

    units = np.array(units)
    validate_units(units)

    if len(units) != len(variables):
        errmess = f"Expected same number of variables ({len(variables)})"\
                    f" and units ({len(units)})"
        raise ValueError(errmess)
    ncdset[data_unit_name(dataset_name)][:] = units

    # Meta data
    nc4io.add_meta(ncdset)


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
        if n not in variables:
            errmess = f"{n} was expected in station variables"
            raise ValueError(errmess)


def configure_stations_variables(ncdset: Dataset,
                                 station_dataset_name: str,
                                 info: pd.DataFrame, dlabel: str) -> None:
    variable_names = info.columns.str\
                     .replace("\\[.*", "", regex=True).values.astype(str)

    vdim_name = station_data_variable_name(station_dataset_name, dlabel)
    vdim = nc4io.Dimension(vdim_name, variable_names,
                           variable_names.dtype, "-")
    vdim.to_dataset(ncdset)

    desc = "Variable storing the list of station meta data"\
           + f" for dataset {station_dataset_name}."
    ncdset[vdim_name].description = desc


def configure_stations_units(ncdset: Dataset,
                             station_dataset_name: str,
                             info: pd.DataFrame, dlabel: str) -> None:
    if dlabel == TEXT_DATA_LABEL:
        units = np.array(["-"]*info.shape[1])
    else:
        cols = info.columns.str
        units = cols.replace(".*\\[|\\]$", "", regex=True).values.astype(str)
        variables = cols.replace("\\[.*", "", regex=True).values.astype(str)
        units[units == variables] = "-"

    validate_units(units)

    uname = station_data_unit_name(station_dataset_name, dlabel)
    vdim_name = station_data_variable_name(station_dataset_name, dlabel)
    descr = "Variable storing station meta data units"\
            + f" for dataset {station_dataset_name}."
    un = nc4io.Variable(uname, [vdim_name], units="-",
                        dtype=units.dtype, fill_value="NA",
                        significant_digit=None,
                        compression=None,
                        attrs={"description": descr})
    un.to_dataset(ncdset)
    units = np.array(units)
    ncdset[uname][:] = units


def store_stations_data_by_type(ncdset: Dataset,
                        station_dataset_name: str,
                        info: pd.DataFrame,
                        dlabel: str) -> None:
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
    dset = station_data_name(station_dataset_name, dlabel)

    vdim_name = station_data_variable_name(station_dataset_name,
                                           dlabel)
    var = nc4io.Variable(dset,
                         ["stationid", vdim_name],
                         units="-", dtype=dtype,
                         fill_value=fill_value,
                         significant_digit=digit,
                         chunksizes=values.shape,
                         compression=None,
                         attrs=attrs)
    var.to_dataset(ncdset)
    ncdset[dset][:] = values


def set_stationid_dimension(ncdset: Dataset, stations: pd.DataFrame) -> None:
    stationids = np.array(stations.index.values).astype(str)
    if "stationid" not in ncdset.dimensions:
        dim = nc4io.Dimension("stationid", stationids, stationids.dtype, "-")
        dim.to_dataset(ncdset)
    else:
        if not np.all(stationids == ncdset["stationid"][:]):
            errmess = "Station IDs should be identical to the one "\
                      "stored in the 'stationid' dimension"
            raise ValueError(errmess)


def set_data_stations(ncdset: Dataset,
                     stations: pd.DataFrame,
                     station_dataset_name:
                     Optional[str] = DEFAULT_STATION_DATASET_NAME,
                     attrs: Optional[dict] = None) -> None:
    if station_dataset_name == DEFAULT_STATION_DATASET_NAME:
        check_expected_stations_columns(stations.columns)

    check_datasetname(station_dataset_name)

    set_stationid_dimension(ncdset, stations)

    attrs = nc4io.minimal_metadata(attrs)

    for dlabel in [TEXT_DATA_LABEL, NUMERICAL_DATA_LABEL]:
        info = select_types(stations, dlabel)
        configure_stations_variables(ncdset, station_dataset_name,
                                     info, dlabel)
        configure_stations_units(ncdset, station_dataset_name,
                                 info, dlabel)
        store_stations_data_by_type(ncdset, station_dataset_name,
                                    info, dlabel)
        # Set metadata
        attrs["data_type"] = dlabel
        dset = station_data_name(station_dataset_name, dlabel)
        for key, value in attrs.items():
            setattr(ncdset[dset], key, value)


def get_attributes(ncdset: Dataset, dataset_name: str) -> dict[str]:
    attrs = {}
    ndt = ncdset[dataset_name]
    for key in ndt.ncattrs():
        attrs[key] = getattr(ndt, key)

    return attrs


def get_data_stations(ncdset: Dataset,
                     station_dataset_name:
                     Optional[str] = DEFAULT_STATION_DATASET_NAME) -> pd.DataFrame:
    # Get attributes
    dset = station_data_name(station_dataset_name, NUMERICAL_DATA_LABEL)
    attrs = get_attributes(ncdset, dset)

    # Get data
    df = []
    stationids = ncdset["stationid"][:]
    for dlabel in [NUMERICAL_DATA_LABEL, TEXT_DATA_LABEL]:
        dset = station_data_name(station_dataset_name, dlabel)
        info = read_from_nc(ncdset, dset)
        colnames = get_dataset_column_names(ncdset, dset)
        df.append(pd.DataFrame(info, index=stationids, columns=colnames))

    return pd.concat(df, axis=1), attrs


def set_data_single_site(ncdset: Dataset,
             dataset_name: str,
             stationid: str,
             data: pd.DataFrame) -> None:

    check_datasetname(dataset_name)
    istation, _ = get_item_index_from_dimension(ncdset, "stationid",
                                                [stationid])
    istation = istation[0]

    nsites, nindex, nvars = ncdset[dataset_name].shape

    if hasattr(data, "index"):
        # Data is assumed to be a dataframe
        try:
            index = nc4io.date2num(data.index)
        except Exception:
            index = data.index.values
        iindex_nc, iindex_data = get_item_index_from_dimension(ncdset,
                                                               "index",
                                                               index)
        variables = [re.sub("\\[.*", "", cn) for cn in data.columns.values]
        vname = data_variable_name(dataset_name)
        ivars_nc, ivars_data = get_item_index_from_dimension(ncdset,
                                                             vname,
                                                             variables)
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
    ncdset[dataset_name][istation, iindex_nc, ivars_nc] = tostore


def get_data_single_site(ncdset: Dataset,
             dataset_name: str,
             stationid: str,
             clip: Optional[bool] = True):
    # Get attributes
    attrs = get_attributes(ncdset, dataset_name)

    # Find station index
    istation, _ = get_item_index_from_dimension(ncdset,
                                                "stationid",
                                                [stationid])
    # Build dataframe
    istation = istation[0]
    colnames = get_dataset_column_names(ncdset, dataset_name)
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

    return df, attrs
