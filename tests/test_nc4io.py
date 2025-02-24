import math
from pathlib import Path
import pandas as pd
from collections import OrderedDict

import re
import random
import string

import pytest

import numpy as np
np.random.seed(5446)

from netCDF4 import Dataset
from hyncu import nc4io

import warnings

FHERE = Path(__file__).resolve().parent


def generate_random_strings(nval, unicode=0,
                            minlength=5, maxlength=40):
    strings = []
    for i in range(nval):
        length = random.randint(minlength, maxlength)
        if unicode == 0:
            use_unicode = False
        elif unicode == 1:
            use_unicode = True
        else:
            use_unicode = random.choice([True, False])

        if use_unicode:
            s = "".join(chr(random.randint(0, 0x10FFFF)) for _ in range(length))
        else:
            punctuation = re.sub("\\[|\\]", "", string.punctuation)
            ascii_chars = string.ascii_letters + string.digits + punctuation
            s = ''.join(random.choice(ascii_chars) for _ in range(length))
        strings.append(s)

    return np.array(strings)


def test_num2date(allclose):
    t = pd.date_range("1850-01-01", "2100-12-31", freq="MS")
    n = nc4io.date2num(t)
    t2 = nc4io.num2date(n)
    assert allclose(t.astype(np.int64), t2.astype(np.int64))

    n2 = nc4io.date2num(t2)
    assert allclose(n, n2)


def test_remove_non_ascii_element():
    s = "ひらがな" + generate_random_strings(1)[0]
    sna = nc4io.remove_non_ascii_element(s)
    assert sna == s[4:]

    s = generate_random_strings(1)[0]
    sna = nc4io.remove_non_ascii_element(s)
    assert s == sna


@pytest.mark.parametrize("dimension_type",
                         ["generic", "time", "longitude", "latitude",
                          "txt-ascii", "txt-unicode"])
def test_dimension(dimension_type, allclose):
    fnc = FHERE / "test_dimension.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        if dimension_type == "generic":
            values = np.arange(10)
            dim = nc4io.Dimension("bidule", values, np.float64, "m/s")
            dim.write_dimension_to_dataset(nc)
            assert "bidule" in nc.dimensions
            d = nc.dimensions["bidule"]
            assert allclose(values.shape, d.size)

        elif dimension_type == "time":
            times = pd.date_range("2001-01-01", "2010-12-01", freq="MS")
            tdim = nc4io.TimeDimension("time", times)
            tdim.write_dimension_to_dataset(nc)
            assert "time" in nc.dimensions

        elif dimension_type == "longitude":
            lons = np.linspace(110, 140, 10)
            ldim = nc4io.SpatialDimension(lons, "longitude")
            ldim.write_dimension_to_dataset(nc)
            assert "longitude" in nc.dimensions

        elif dimension_type == "latitude":
            lats = np.linspace(-40, -10, 10)
            ldim = nc4io.SpatialDimension(lats, "latitude")
            ldim.write_dimension_to_dataset(nc)
            assert "latitude" in nc.dimensions

        elif dimension_type == "txt-ascii":
            values = generate_random_strings(10, 0)
            dim = nc4io.Dimension("bidule-ascii", values, values.dtype, "m/s")
            dim.write_dimension_to_dataset(nc)
            assert "bidule-ascii" in nc.dimensions

        elif dimension_type == "txt-unicode":
            values = generate_random_strings(10, 2)
            msg = "Values in dimension"
            with pytest.raises(ValueError, match=msg):
                dim = nc4io.Dimension("bidule-unicode", values,
                                      values.dtype, "m/s")

    with Dataset(fnc, "r") as nc:
        if dimension_type == "generic":
            dim = nc4io.Dimension.from_dataset(nc, "bidule")
            assert allclose(dim.values, values)

        elif dimension_type == "time":
            tdim = nc4io.TimeDimension.from_dataset(nc)
            diff = (tdim.values-times).seconds
            assert np.all(diff==0)

        elif dimension_type == "longitude":
            ldim = nc4io.SpatialDimension.from_dataset(nc, "longitude")
            assert allclose(ldim.values, lons)

        elif dimension_type == "latitude":
            ldim = nc4io.SpatialDimension.from_dataset(nc, "latitude")
            assert allclose(ldim.values, lats)

        elif dimension_type == "txt-ascii":
            dim = nc4io.Dimension.from_dataset(nc, "bidule-ascii")
            assert all([v1 == v2 for v1, v2 in zip(dim.values, values)])

    fnc.unlink()


def test_add_known_dimensions(allclose):
    fnc = FHERE / "test_add_known_dimension.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        times = pd.date_range("2001-01-01", "2010-12-01", freq="MS")
        lons = np.linspace(110, 140, 10)
        lats = np.linspace(-40, -10, 10)
        nc4io.add_dimension(nc, "time", times)
        nc4io.add_dimension(nc, "longitude", lons)
        nc4io.add_dimension(nc, "latitude", lats)

    with Dataset(fnc, "r") as nc:
        tdim = nc4io.TimeDimension.from_dataset(nc)
        diff = (tdim.values-times).seconds
        assert np.all(diff==0)

        ldim = nc4io.SpatialDimension.from_dataset(nc, "longitude")
        assert allclose(ldim.values, lons)

        ldim = nc4io.SpatialDimension.from_dataset(nc, "latitude")
        assert allclose(ldim.values, lats)

    fnc.unlink()


def test_add_spatial_dimensions(allclose):
    fnc = FHERE / "test_add_spatial_dimension.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        times = pd.date_range("2001-01-01", "2010-12-01", freq="MS")
        lons = np.linspace(110, 140, 10)
        lats = np.linspace(-40, -10, 10)
        nc4io.add_spatial_dimensions(nc, lons, lats)

    with Dataset(fnc, "r") as nc:
        ldim = nc4io.SpatialDimension.from_dataset(nc, "longitude")
        assert allclose(ldim.values, lons)

        ldim = nc4io.SpatialDimension.from_dataset(nc, "latitude")
        assert allclose(ldim.values, lats)

    fnc.unlink()


def test_add_generic_dimensions(allclose):
    fnc = FHERE / "test_add_generic_dimension.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        siteids = ["a", "b", "c", "d"]
        variables = ["v1", "v2", "v3"]
        nc4io.add_dimension(nc, "siteid", siteids)
        nc4io.add_dimension(nc, "variable", variables)

    with Dataset(fnc, "r") as nc:
        dim = nc4io.Dimension.from_dataset(nc, "siteid")
        assert np.all(dim.values==siteids)

        dim = nc4io.Dimension.from_dataset(nc, "variable")
        assert np.all(dim.values==variables)

    fnc.unlink()


@pytest.mark.parametrize("compression", [None, "zlib"])
@pytest.mark.parametrize("fill_value", [None, -999.])
@pytest.mark.parametrize("chunksizes", [None, (2, 2)])
def test_variable(compression, fill_value, chunksizes, allclose):
    fnc = FHERE / "test_variable.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        dims = OrderedDict()
        dims["latitude"] = np.linspace(-40, -10, 10)
        dims["longitude"] = np.linspace(110, 140, 10)
        attrs = {"comment": "bidule", "data_provider": "yo"}
        nvar = nc4io.Variable(nc, "bidule",
                              dimensions=dims,
                              units="mm.month-1",
                              compression=compression,
                              fill_value=fill_value,
                              chunksizes=chunksizes,
                              attrs=attrs)

        nlats = len(dims["latitude"])
        nlons = len(dims["longitude"])
        data = np.random.uniform(size=(nlats, nlons))
        data.flat[:5] = np.nan
        nvar.write_data_to_dataset(data)

        # Write data subset
        idx = np.arange(3), np.arange(3)
        nvar.write_data_to_dataset(-999, idx)
        data[idx[0][:, None], idx[1][None, :]] = -999

    with Dataset(fnc, "r") as nc:
        v = nc["bidule"]
        if fill_value is None:
            assert np.isnan(v._FillValue)
        else:
            assert v._FillValue == fill_value
        if chunksizes is None:
            assert v.chunking() in ["contiguous", [10, 10]]
        else:
            assert allclose(v.chunking(), chunksizes)
        assert v.comment == "bidule"
        assert v.data_provider == "yo"
        assert len(v.author)>0
        assert len(v.version)>0
        assert len(v.source_file)>0
        assert v.units == "mm.month-1"
        d = v[:].filled()
        assert allclose(d, data, atol=1e-5, equal_nan=True)

        for n in ["latitude", "longitude"]:
            assert allclose(nvar.dimensions[n], dims[n])

        nvar = nc4io.Variable(nc, "bidule")
        for dname in nvar.dimensions:
            assert allclose(nvar.dimensions[dname],
                            dims[dname])
    fnc.unlink()


def test_spatialvariable(allclose):
    fnc = FHERE / "test_spatialvariable.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        lons = np.linspace(110, 140, 10)
        lats = np.linspace(-40, -10, 10)
        nvar = nc4io.SpatialVariable(nc, "bidule", lons, lats)
        data = np.random.uniform(size=(len(lats), len(lons)))
        nvar.write_data_to_dataset(data)

    with Dataset(fnc, "r") as nc:
        v = nc["bidule"]
        assert len(v.version) > 0
        assert len(v.source_file) > 0
        d = v[:].filled()
        assert allclose(d, data, atol=1e-5, equal_nan=True)

    fnc.unlink()


def test_spatialtimevariable(allclose):
    fnc = FHERE / "test_spatialtimevariable.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        lons = np.linspace(110, 140, 10)
        lats = np.linspace(-40, -10, 10)
        times = pd.date_range("2001-01-01", "2001-04-30")
        nvar = nc4io.SpatialTimeVariable(nc, "bidule", lons, lats, times)
        data = np.random.uniform(size=(len(lats), len(lons), len(times)))
        nvar.write_data_to_dataset(data)

    with Dataset(fnc, "r") as nc:
        v = nc["bidule"]
        assert len(v.version) > 0
        assert len(v.source_file) > 0
        d = v[:].filled()
        assert allclose(d, data, atol=1e-5, equal_nan=True)

    fnc.unlink()


@pytest.mark.parametrize("dtype", ["S", "U", "<S", "<U", "unicode"])
def test_textvariable(dtype,  allclose):
    fnc = FHERE / "test_textvariable.nc"
    if fnc.exists():
        fnc.unlink()

    with Dataset(fnc, "w") as nc:
        dims = OrderedDict()
        nlons, nlats = 100, 50
        dims["latitude"] = np.linspace(-40, -10, nlats)
        dims["longitude"] = np.linspace(110, 140, nlons)
        nchar = 20
        is_unicode = dtype == "unicode"
        np_dtype = f"U{nchar}" if is_unicode else f"{dtype}{nchar}"
        nvar = nc4io.Variable(nc, "bidule",
                              dimensions=dims,
                              numpy_dtype=np_dtype,
                              units="mm.month-1")

        nlats = len(dims["latitude"])
        nlons = len(dims["longitude"])
        unicode = 2 if is_unicode else 0
        data = [generate_random_strings(nlons, unicode,
                                        maxlength=nchar)
                for la in range(nlats)]
        data = np.array(data)
        if dtype != "unicode":
            data = data.astype(dtype)

        nvar.write_data_to_dataset(data)

        # Write data subset
        idx = np.arange(3), np.arange(3)
        st = "xxx"
        nvar.write_data_to_dataset(st, idx)
        data[idx[0][:, None], idx[1][None, :]] = st

    with Dataset(fnc, "r") as nc:
        d = nc["bidule"][:]
        if dtype != "unicode":
            d = d.astype(dtype)
            assert np.all(d == data)
        else:
            is_ascii = nc4io.is_ascii_vectorised(d)
            assert np.all(is_ascii)

    fnc.unlink()


