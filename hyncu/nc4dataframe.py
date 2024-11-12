"""Utility functions to export netcdf files """
import re, sys
import numpy as np
import pandas as pd
from datetime import datetime
import cf_units
import netCDF4

from hyncu import nc4io

DEFAULT_DATANAME = "site_timeseries_data"


def configure_netcdf_file(nc4dset, siteids, variables, times,\
                    units=None, dataname=DEFAULT_DATANAME, attrs=None):
    # Create dimensions
    nc4io.add_dimensions(nc4dset, time=times, siteid=siteids, variable=variables)

    # Variable to store data
    var = nc4io.Variable(dataname, ["siteid", "time", "variable"], \
                                units="-", \
                                chunksizes=(1, len(times),len(variables)), \
                                attrs=attrs)
    var.to_dataset(nc4dset)

    # Units
    un = nc4io.Variable("variable_units", ["variable"], units="-", \
                        dtype=str, fill_value="NA", \
                        significant_digit=None, \
                        compression=None, \
                        attrs={"description": "Units for each variable"})
    un.to_dataset(nc4dset)
    if units is None:
        units = ["-"]*len(variables)

    units = np.array(units)
    assert len(units) == len(variables)

    for u in units:
        try:
            cf_units.Unit(u)
        except:
            errmess = f"Unit {u} is not valid."
            raise ValueError(errmess)

    nc4dset["variable_units"][:] = units

    # Meta data
    nc4io.add_meta(nc4dset)


def get_item_index(nc4dset, item_type, item_name):
    assert item_type in ["siteid", "variable"]
    dim = nc4dset[item_type][:]
    return np.where(item_name==dim)[0][0]


def get_colnames(nc4dset):
    varnames = nc4dset["variable"][:]
    units = nc4dset["variable_units"][:]
    return [f"{v}[{u}]" for v, u in zip(varnames, units)]


def set_dataframe(nc4dset, siteid, df, dataname=DEFAULT_DATANAME):
    isite = get_item_index(nc4dset, "siteid", siteid)
    colnames = get_colnames(nc4dset)
    # TODO
    nc4dset[dataname][isite, :, :] = data


def get_dataframe(nc4dset, siteid, dataname=DEFAULT_DATANAME):
    isite = get_item_index(nc4dset, "siteid", siteid)
    colnames = get_colnames(nc4dset)
    tdim = nc4io.TimeDimension.from_dataset(nc4dset)
    times = tdim.values
    df = pd.DataFrame(nc4dset[dataname][isite, :, :], index=times, \
                            columns=colnames)
    return df
