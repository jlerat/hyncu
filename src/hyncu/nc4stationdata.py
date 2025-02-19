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


def ncvariable_name(dataset_name):
    return f"{dataset_name}{LABEL_SEPARATOR}variable_name"


def station_data_ncname(station_dataset_name, dlabel):
    return f"{station_dataset_name}{LABEL_SEPARATOR}{dlabel}"


def station_variable_ncname(station_dataset_name, dlabel):
    sdn = station_data_ncname(station_dataset_name, dlabel)
    return ncvariable_name(sdn)


def unit_ncname(dataset_name):
    return f"{dataset_name}{LABEL_SEPARATOR}variable_unit"


def station_unit_ncname(station_dataset_name, dlabel):
    sdn = station_data_ncname(station_dataset_name, dlabel)
    return unit_ncname(sdn)


def read_from_nc(ncdset: Dataset, variable_name):
    if variable_name not in ncdset.variables:
        errmess = f"{variable_name} was not found in variables."\
                  + " Make sure the function"\
                  + "nc4dataframe.add_station_variables"\
                  + " has been run."
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


def add_station_info_variables(ncdset: Dataset,
                          dataset_name: str,
                          stationids: np.ndarray,
                          info_variable_names: np.ndarray,
                          index: np.ndarray,
                          units: Optional[str] = None,
                          attrs: Optional[dict] = None):

    check_datasetname(dataset_name)

    # Create dimensions
    kw = {ncvariable_name(dataset_name): info_variable_names}
    nc4io.add_dimensions(ncdset, index=index, stationid=stationids, **kw)

    # Create nc variable for station data
    variable_name_nc = ncvariable_name(dataset_name)
    data_var = nc4io.Variable(ncdset, dataset_name,
                              dimensions=["stationid", "index", variable_name_nc],
                              units="-",
                              chunksizes=(1, min(10000, len(index)),
                                          min(10, len(info_variable_names))),
                              attrs=attrs)
    data_var.write_variable_to_dataset()

    # Create nc variable for Units
    units_var = nc4io.Variable(ncdset, unit_ncname(dataset_name),
                               dimensions=[variable_name_nc],
                               units="-",
                               dtype=str, fill_value="NA",
                               significant_digit=None,
                               compression=None,
                               attrs={"description":
                                      "Units for each data variable"})
    units_var.write_variable_to_dataset()
    if units is None:
        units = ["-"]*len(variable_names)

    units = np.array(units)
    validate_units(units)

    if len(units) != len(info_variable_names):
        errmess = "Expected same number of variables "\
                    f"({len(info_variable_names)}) and units ({len(units)})"
        raise ValueError(errmess)
    units_var.write_data_to_dataset(units)


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
    variable_names = [re.sub("\\[.*", "", cn) for cn in columns]
    for n in EXPECTED_STATIONS_COLUMNS:
        if n not in variable_names:
            errmess = f"{n} was expected in station variable_names"
            raise ValueError(errmess)


def configure_stations_variable_names(ncdset: Dataset,
                                 station_dataset_name: str,
                                 info: pd.DataFrame, dlabel: str) -> None:
    variable_names = info.columns.str\
                     .replace("\\[.*", "", regex=True).values.astype(str)

    vdim_name = station_variable_ncname(station_dataset_name, dlabel)
    vdim = nc4io.Dimension(vdim_name, variable_names,
                           variable_names.dtype, "-")
    vdim.write_dimension_to_dataset(ncdset)

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
        variable_names = cols.replace("\\[.*", "", regex=True).values.astype(str)
        units[units == variable_names] = "-"

    validate_units(units)

    uname = station_unit_ncname(station_dataset_name, dlabel)
    vdim_name = station_variable_ncname(station_dataset_name, dlabel)
    descr = "Variable storing station meta data units"\
            + f" for dataset {station_dataset_name}."
    un = nc4io.Variable(ncdset, uname, [vdim_name], units="-",
                        dtype=units.dtype, fill_value="NA",
                        significant_digit=None,
                        compression=None,
                        attrs={"description": descr})
    un.write_variable_to_dataset()
    un.write_data_to_dataset(np.array(units))


def write_stations_data_by_type(ncdset: Dataset,
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
    var_ncname = station_data_ncname(station_dataset_name, dlabel)
    vdim_name = station_variable_ncname(station_dataset_name,
                                        dlabel)
    var = nc4io.Variable(ncdset, var_ncname,
                         dimensions=["stationid", vdim_name],
                         units="-", dtype=dtype,
                         fill_value=fill_value,
                         significant_digit=digit,
                         chunksizes=values.shape,
                         compression=None,
                         attrs=attrs)
    var.write_data_to_dataset(values)


def add_stationid_dimension(ncdset: Dataset,
                            stationids: Union[list, pd.Index]) -> None:
    stationids = np.array(stationids).astype(str)
    if "stationid" not in ncdset.dimensions:
        dim = nc4io.Dimension("stationid", stationids, stationids.dtype, "-")
        dim.write_dimension_to_dataset(ncdset)
    else:
        if not np.all(stationids == ncdset["stationid"][:]):
            errmess = "Station IDs should be identical to the one "\
                      "stored in the 'stationid' dimension"
            raise ValueError(errmess)


def write_station_info(ncdset: Dataset,
                       info: pd.DataFrame,
                       station_dataset_name:
                       Optional[str] = DEFAULT_STATION_DATASET_NAME,
                       attrs: Optional[dict] = None) -> None:
    if station_dataset_name == DEFAULT_STATION_DATASET_NAME:
        check_expected_stations_columns(info.columns)

    check_datasetname(station_dataset_name)

    add_stationid_dimension(ncdset, info.index)

    attrs = nc4io.minimal_metadata(attrs)

    for dlabel in [TEXT_DATA_LABEL, NUMERICAL_DATA_LABEL]:
        info_type = select_types(info, dlabel)

        configure_stations_variable_names(ncdset, station_dataset_name,
                                     info_type, dlabel)

        configure_stations_units(ncdset, station_dataset_name,
                                 info_type, dlabel)

        write_stations_data_by_type(ncdset, station_dataset_name,
                                    info_type, dlabel)
        # Set metadata
        attrs["data_type"] = dlabel
        dset = station_data_ncname(station_dataset_name, dlabel)
        for key, value in attrs.items():
            setattr(ncdset[dset], key, value)


def read_attributes(ncdset: Dataset, dataset_name: str) -> dict[str]:
    attrs = {}
    ndt = ncdset[dataset_name]
    for key in ndt.ncattrs():
        attrs[key] = getattr(ndt, key)

    return attrs


def read_station_info(ncdset: Dataset,
                      station_dataset_name:
                      Optional[str] =
                      DEFAULT_STATION_DATASET_NAME) -> pd.DataFrame:
    # Get attributes
    dset = station_data_ncname(station_dataset_name, NUMERICAL_DATA_LABEL)
    attrs = read_attributes(ncdset, dset)
    for key in ["long_name", "data_type",
                "units", "_FillValue", "least_significant_digit"]:
        attrs.pop(key)

    # Get data
    info = []
    stationids = ncdset["stationid"][:]
    for dlabel in [NUMERICAL_DATA_LABEL, TEXT_DATA_LABEL]:
        dset = station_data_ncname(station_dataset_name, dlabel)
        array = read_from_nc(ncdset, dset)
        colnames = get_dataset_variable_names(ncdset, dset,
                                            dlabel == NUMERICAL_DATA_LABEL)
        info.append(pd.DataFrame(array, index=stationids, columns=colnames))

    return pd.concat(info, axis=1), attrs


def write_data_for_single_station(ncdset: Dataset,
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
        variable_names = [re.sub("\\[.*", "", cn) for cn in data.columns.values]
        dname = ncvariable_name(dataset_name)
        ivars_nc, ivars_data = get_item_index_from_dimension(ncdset,
                                                             dname,
                                                             variable_names)
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


def read_data_from_single_station(ncdset: Dataset,
                             dataset_name: str,
                             stationid: str,
                             clip: Optional[bool] = True):
    check_datasetname(dataset_name)

    # Get attributes
    attrs = read_attributes(ncdset, dataset_name)

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
