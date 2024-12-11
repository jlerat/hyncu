import math
from pathlib import Path
import pandas as pd

import pytest

import numpy as np
np.random.seed(5446)

from netCDF4 import Dataset
from hyncu import nc4io

import warnings

FHERE = Path(__file__).resolve().parent

def test_num2date(allclose):
    t = pd.date_range("1850-01-01", "2100-12-31", freq="MS")
    n = nc4io.date2num(t)
    t2 = nc4io.num2date(n)
    assert allclose(t.astype(np.int64), t2.astype(np.int64))

    n2 = nc4io.date2num(t2)
    assert allclose(n, n2)


def test_dimension(allclose):
    fnc = FHERE / "test_dimension.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        values = np.arange(10)
        dim = nc4io.Dimension("bidule", values, np.float64, "m/s")
        dim.to_dataset(nc)
        assert "bidule" in nc.dimensions
        d = nc.dimensions["bidule"]
        assert allclose(values.shape, d.size)

        times = pd.date_range("2001-01-01", "2010-12-01", freq="MS")
        tdim = nc4io.TimeDimension("time", times)
        tdim.to_dataset(nc)
        assert "time" in nc.dimensions

        lons = np.linspace(110, 140, 10)
        ldim = nc4io.CoordinateDimension(lons, "longitude")
        ldim.to_dataset(nc)
        assert "longitude" in nc.dimensions

        lats = np.linspace(-40, -10, 10)
        ldim = nc4io.CoordinateDimension(lats, "latitude")
        ldim.to_dataset(nc)
        assert "latitude" in nc.dimensions


    with Dataset(fnc, "r") as nc:
        dim = nc4io.Dimension.from_dataset(nc, "bidule")
        assert allclose(dim.values, values)

        tdim = nc4io.TimeDimension.from_dataset(nc)
        diff = (tdim.values-times).seconds
        assert np.all(diff==0)

        ldim = nc4io.CoordinateDimension.from_dataset(nc, "longitude")
        assert allclose(ldim.values, lons)

        ldim = nc4io.CoordinateDimension.from_dataset(nc, "latitude")
        assert allclose(ldim.values, lats)

    fnc.unlink()


def test_add_known_dimensions(allclose):
    fnc = FHERE / "test_add_known_dimension.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        times = pd.date_range("2001-01-01", "2010-12-01", freq="MS")
        lons = np.linspace(110, 140, 10)
        lats = np.linspace(-40, -10, 10)
        nc4io.add_dimensions(nc, time=times, longitude=lons, latitude=lats)

    with Dataset(fnc, "r") as nc:
        tdim = nc4io.TimeDimension.from_dataset(nc)
        diff = (tdim.values-times).seconds
        assert np.all(diff==0)

        ldim = nc4io.CoordinateDimension.from_dataset(nc, "longitude")
        assert allclose(ldim.values, lons)

        ldim = nc4io.CoordinateDimension.from_dataset(nc, "latitude")
        assert allclose(ldim.values, lats)

    fnc.unlink()


def test_add_generic_dimensions(allclose):
    fnc = FHERE / "test_add_generic_dimension.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        siteids = ["a", "b", "c", "d"]
        variables = ["v1", "v2", "v3"]
        nc4io.add_dimensions(nc, siteid=siteids, variable=variables)

    with Dataset(fnc, "r") as nc:
        dim = nc4io.Dimension.from_dataset(nc, "siteid")
        assert np.all(dim.values==siteids)

        dim = nc4io.Dimension.from_dataset(nc, "variable")
        assert np.all(dim.values==variables)

    fnc.unlink()


def test_variable(allclose):
    fnc = FHERE / "test_variable.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        lons = np.linspace(110, 140, 10)
        lats = np.linspace(-40, -10, 10)
        nc4io.add_dimensions(nc, longitude=lons, latitude=lats)

        attrs = {"comment": "bidule", "data_provider": "yo"}
        nvar = nc4io.Variable("bidule", ["latitude", "longitude"], \
                    units="mm.month-1", \
                    attrs=attrs)
        nvar.to_dataset(nc)
        data = np.random.uniform(size=(len(lats), len(lons)))
        data.flat[:5] = np.nan
        nc["bidule"][:] = data

    with Dataset(fnc, "r") as nc:
        v = nc["bidule"]
        assert v.comment == "bidule"
        assert v.data_provider == "yo"
        assert len(v.author)>0
        assert len(v.version)>0
        assert len(v.source_file)>0
        assert v.units == "mm.month-1"
        d = v[:].filled()
        assert allclose(d, data, atol=1e-5, equal_nan=True)

    fnc.unlink()

