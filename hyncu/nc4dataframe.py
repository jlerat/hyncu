"""Utility functions to export netcdf files """
import re, sys
import numpy as np
import pandas as pd
from datetime import datetime
import cf_units
import netCDF4

from hyncu import nc4io

DEFAULT_DATANAME = "site_dataframe"


def set_netcdf_file(nc4dset, siteids, variables, times,\
                    dataname=DEFAULT_DATANAME, attrs=None):
    nc4io.add_dimensions(nc4dset, time=times, siteid=siteids, variable=variables)
    var = nc4io.Variable(dataname, ["siteid", "time", "variable"], units="-", \
                                        attrs=attrs)
    var.to_dataset(nc4dset)
    nc4io.add_meta(nc4dset)


def get_item_index(nc4dset, item_type, item_name):
    assert item_type in ["siteid", "variable"]
    dim = nc4dset[item_type][:]
    return np.where(item_name==dim)[0][0]


def get_dataframe(nc4dset, siteid, dataname=DEFAULT_DATANAME):
    isite = get_item_index(nc4dset, "siteid", siteid)

    datanames = nc4dset["variable"][:]

    tdim = nc4io.TimeDimension.from_dataset(nc4dset)
    times = tdim.values

    return pd.DataFrame(nc4dset[dataname][isite, :, :], index=times, \
                            columns=datanames)
