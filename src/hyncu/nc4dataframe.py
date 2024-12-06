"""Utility functions to export netcdf files """
import re, sys
import numpy as np
import pandas as pd
from datetime import datetime
import cf_units
import netCDF4

from hyncu import nc4io

DEFAULT_DATANAME = "data"

def validate_units(units):
    for u in units:
        try:
            cf_units.Unit(u)
        except:
            errmess = f"Unit {u} is not valid."
            raise ValueError(errmess)


def get_item_index(nc4dset, item_type, item_name):
    assert item_type in ["stationid", "variable"]
    dim = nc4dset[item_type][:]
    return np.where(item_name==dim)[0][0]


def get_colnames(nc4dset, vname):
    varnames = nc4dset[vname][:]
    units = nc4dset[f"{vname}_units"][:]
    vtxtdim = "station_info_txt_variable"
    if vtxtdim in nc4dset.dimensions:
        vtxtnames = nc4dset[vtxtdim][:]
    else:
        vtxtnames = []

    colnames = []
    for v, u in zip(varnames, units):
        cn = f"{v}[{u}]" if not v in vtxtnames else v
        colnames.append(cn)

    return colnames



def configure_netcdf_file(nc4dset, stationids, variables, times,\
                    units=None, dataname=DEFAULT_DATANAME, attrs=None):
    # Create dimensions
    kw = {f"{dataname}_variable": variables}
    nc4io.add_dimensions(nc4dset, time=times, stationid=stationids, **kw)

    # Variable to store data
    var = nc4io.Variable(dataname, ["stationid", "time", f"{dataname}_variable"], \
                                units="-", \
                                chunksizes=(1, len(times),len(variables)), \
                                attrs=attrs)
    var.to_dataset(nc4dset)

    # Units
    un = nc4io.Variable(f"{dataname}_variable_units", [f"{dataname}_variable"], units="-", \
                        dtype=str, fill_value="NA", \
                        significant_digit=None, \
                        compression=None, \
                        attrs={"description": "Units for each variable"})
    un.to_dataset(nc4dset)
    if units is None:
        units = ["-"]*len(variables)

    units = np.array(units)
    validate_units(units)
    assert len(units) == len(variables)
    nc4dset[f"{dataname}_variable_units"][:] = units

    # Meta data
    nc4io.add_meta(nc4dset)


def set_station_info(nc4dset, stations):
    # Split stations into num and txt
    stations_num = stations.select_dtypes(include="number")
    stations_txt = stations.select_dtypes(include=["object", "datetime"]).astype(str)

    assert "NAME" in stations_txt.columns

    # Set station id dim if not existing
    stationids = np.array(stations.index.values).astype(str)
    if not "stationid" in nc4dset.dimensions:
        dim = nc4io.Dimension("stationid", stationids, stationids.dtype, "-")
        dim.to_dataset(nc4dset)

    else:
        assert np.all(stationids == nc4dset["stationid"][:])

    for itype, info in dict(num=stations_num, txt=stations_txt).items():
        # Get station info variables and units
        variables = info.columns.str.replace("\\[.*", "", regex=True).values.astype(str)
        if itype == "num":
            units = info.columns.str.replace(".*\\[|\\]$", "", regex=True).values.astype(str)
            units[units==variables] = "-"
        else:
            units = np.array(["-"]*len(variables))

        # Create dimensions
        vname = f"station_info_{itype}_variable"
        vdim = nc4io.Dimension(vname, variables, variables.dtype, "-")
        vdim.to_dataset(nc4dset)

        # Variable to store stations info
        dname = f"station_info_{itype}"
        if itype == "num":
            values = info.values
            fill_value = np.nan
            digit = 7
        else:
            values = info.values.astype(str)
            fill_value = "NA"
            digit = None

        dtype = values.dtype
        attrs = dict(description="Station info / {itype}")
        var = nc4io.Variable(dname, ["stationid", vname], \
                                    units="-", dtype=dtype, \
                                    fill_value=fill_value, \
                                    significant_digit=digit, \
                                    chunksizes=values.shape, \
                                    compression=None, \
                                    attrs=attrs)
        var.to_dataset(nc4dset)
        nc4dset[dname][:] = values

        # Units
        uname = f"station_info_{itype}_variable_units"
        un = nc4io.Variable(uname, [vname], units="-", \
                            dtype=units.dtype, fill_value="NA", \
                            significant_digit=None, \
                            compression=None, \
                            attrs={"description": "Units for each station info variable"})
        un.to_dataset(nc4dset)
        units = np.array(units)
        validate_units(units)
        assert len(units) == len(variables)
        nc4dset[uname][:] = units


def get_station_info(nc4dset):
    df = []
    stationids = nc4dset["stationid"][:]
    for itype in ["num", "txt"]:
        info = nc4dset[f"station_info_{itype}"][:]
        colnames = get_colnames(nc4dset, f"station_info_{itype}_variable")
        df.append(pd.DataFrame(info, index=stationids, columns=colnames))
    return pd.concat(df, axis=1)


def set_dataframe(nc4dset, stationid, df, dataname=DEFAULT_DATANAME):
    istation = get_item_index(nc4dset, "stationid", stationid)
    colnames = get_colnames(nc4dset, f"{dataname}_variable")
    # TODO
    nc4dset[dataname][istation, :, :] = data


def get_dataframe(nc4dset, stationid, dataname=DEFAULT_DATANAME):
    istation = get_item_index(nc4dset, "stationid", stationid)
    colnames = get_colnames(nc4dset, f"{dataname}_variable")
    tdim = nc4io.TimeDimension.from_dataset(nc4dset)
    times = tdim.values
    df = pd.DataFrame(nc4dset[dataname][istation, :, :], index=times, \
                            columns=colnames)
    return df
