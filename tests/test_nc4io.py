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


def test_dimension(allclose):
    fnc = FHERE / "test_dimension.nc"
    with Dataset(fnc, "w") as nc:
        values = np.arange(10)
        dim = nc4io.Dimension("bidule", values, np.float64, "m/s")
        dim.to_dataset(nc)
        assert "bidule" in nc.dimensions
        d = nc.dimensions["bidule"]
        assert allclose(values.shape, d.size)

        times = pd.date_range("2001-01-01", "2010-12-01", freq="MS")
        tdim = nc4io.TimeDimension(times)
        tdim.to_dataset(nc)
        assert "time" in nc.dimensions

        lons = np.linspace(110, 140, 10)
        ldim = nc4io.LatLongDimension(lons)
        ldim.to_dataset(nc)
        assert "longitude" in nc.dimensions

        lats = np.linspace(-40, -10, 10)
        ldim = nc4io.LatLongDimension(lats, False)
        ldim.to_dataset(nc)
        assert "latitude" in nc.dimensions

        sites = ["203013", "422310A"]
        sdim = nc4io.SiteidDimension(sites)
        sdim.to_dataset(nc)
        assert "siteid" in nc.dimensions


    with Dataset(fnc, "r") as nc:
        dim = nc4io.Dimension.from_dataset(nc, "bidule")
        assert allclose(dim.values, values)

        tdim = nc4io.TimeDimension.from_dataset(nc)
        diff = (tdim.values-times).seconds
        assert np.all(diff==0)

        ldim = nc4io.LatLongDimension.from_dataset(nc)
        assert allclose(ldim.values, lons)

        ldim = nc4io.LatLongDimension.from_dataset(nc, False)
        assert allclose(ldim.values, lats)

        sdim = nc4io.SiteidDimension.from_dataset(nc)
        assert np.all(sdim.values==sites)

    fnc.unlink()


def test_add_dimensions(allclose):
    fnc = FHERE / "test_add_dimension.nc"
    with Dataset(fnc, "w") as nc:
        times = pd.date_range("2001-01-01", "2010-12-01", freq="MS")
        lons = np.linspace(110, 140, 10)
        lats = np.linspace(-40, -10, 10)
        nc4io.add_dimensions(nc, times, lons, lats)

    with Dataset(fnc, "r") as nc:
        tdim = nc4io.TimeDimension.from_dataset(nc)
        diff = (tdim.values-times).seconds
        assert np.all(diff==0)

        ldim = nc4io.LatLongDimension.from_dataset(nc)
        assert allclose(ldim.values, lons)

        ldim = nc4io.LatLongDimension.from_dataset(nc, False)
        assert allclose(ldim.values, lats)

    fnc.unlink()


def test_variable(allclose):
    fnc = FHERE / "test_variable.nc"
    with Dataset(fnc, "w") as nc:
        lons = np.linspace(110, 140, 10)
        lats = np.linspace(-40, -10, 10)
        nc4io.add_dimensions(nc, lons=lons, lats=lats)

        attrs = {"comment": "bidule"}
        nvar = nc4io.Variable("bidule", ["latitude", "longitude"], \
                    units="mm.month-1", \
                    attrs=attrs)
        nvar.to_dataset(nc)
        data = np.random.uniform(size=(len(lats), len(lons)))
        data.flat[:5] = np.nan
        nc["bidule"][:] = data

    with Dataset(fnc, "r") as nc:
        v = nc["bidule"]
        assert v.comment=="bidule"
        assert v.units=="mm.month-1"
        d = v[:].filled()
        assert allclose(d, data, atol=1e-5, equal_nan=True)

    fnc.unlink()
