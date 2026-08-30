"""Tests for TypeDesc.for_type() constructor."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from typetrace import TypeDesc


class TestForType:
    """Test TypeDesc.for_type() derives kind from concrete_type."""

    def test_xarray_dataarray(self) -> None:
        td = TypeDesc.for_type(xr.DataArray, dtype="float64", dims=(("x", 10, None),))
        assert td.kind == "xarray.DataArray"
        assert td.dtype == "float64"
        assert td.dims == (("x", 10, None),)

    def test_xarray_dataset(self) -> None:
        td = TypeDesc.for_type(
            xr.Dataset, fields=(("a", TypeDesc(kind="numpy.ndarray", dims=(("x", 5, None),))),)
        )
        assert td.kind == "xarray.Dataset"
        assert td.fields is not None
        assert "a" in {key for key, _ in td.fields}

    def test_numpy_ndarray(self) -> None:
        td = TypeDesc.for_type(np.ndarray, dtype="int32", shape=(10, 20))
        assert td.kind == "numpy.ndarray"
        assert td.dtype == "int32"
        assert td.shape == (10, 20)

    @pytest.mark.parametrize(
        "columns,expected_columns",
        [
            (("a", "b", "c"), ("a", "b", "c")),
            (("a", "b", ...), ("a", "b", ...)),
        ],
    )
    def test_pandas_dataframe(self, columns: tuple, expected_columns: tuple) -> None:
        td = TypeDesc.for_type(
            pd.DataFrame,
            columns=columns,
        )
        assert td.kind == "pandas.DataFrame"
        assert td.columns == expected_columns

    def test_pandas_series(self) -> None:
        td = TypeDesc.for_type(pd.Series, dtype="float64")
        assert td.kind == "pandas.Series"
        assert td.dtype == "float64"

    def test_unknown_type_falls_back_to_class(self) -> None:
        class CustomClass:
            pass

        td = TypeDesc.for_type(CustomClass)
        assert td.kind == "opaque"

    @pytest.mark.parametrize(
        "concrete_type,expected_kind",
        [
            (type("DataFrame", (), {"__module__": "polars"}), "polars.DataFrame"),
            (type("Series", (), {"__module__": "polars"}), "polars.Series"),
            (type("Table", (), {"__module__": "pyarrow"}), "pyarrow.Table"),
            (type("FloatArray", (), {"__module__": "drjit"}), "drjit.Array"),
        ],
    )
    def test_kind_for_type_optional_module_roots(
        self, concrete_type: type, expected_kind: str
    ) -> None:
        td = TypeDesc.for_type(concrete_type)
        assert td.kind == expected_kind


class TestDatasetSample:
    """Test make_sample for Dataset kind."""

    def test_dataset_from_fields(self) -> None:
        td = TypeDesc(
            kind="xarray.Dataset",
            fields=(
                (
                    "temp",
                    TypeDesc(
                        kind="xarray.DataArray",
                        dtype="float64",
                        dims=(("x", 3, None), ("y", 2, None)),
                    ),
                ),
                (
                    "pressure",
                    TypeDesc(kind="xarray.DataArray", dtype="float64", dims=(("x", 3, None),)),
                ),
            ),
        )
        sample = td.make_sample()
        assert isinstance(sample, xr.Dataset)
        assert "temp" in sample.data_vars
        assert "pressure" in sample.data_vars
        assert sample["temp"].dims == ("x", "y")

    def test_dataset_from_dims(self) -> None:
        td = TypeDesc(
            kind="xarray.Dataset",
            fields=(
                (
                    "data",
                    TypeDesc(
                        kind="xarray.DataArray",
                        dims=(("time", 5, None), ("space", 3, None)),
                        dtype="float32",
                    ),
                ),
            ),
        )
        sample = td.make_sample()
        assert isinstance(sample, xr.Dataset)
        assert "data" in sample.data_vars
        assert sample["data"].dims == ("time", "space")


class TestRoundTrip:
    """Test from_value → for_type equivalence."""

    def test_dataarray_roundtrip(self) -> None:
        arr = xr.DataArray(np.zeros((3, 4)), dims=["x", "y"])
        td_from_value = TypeDesc.from_value(arr)

        td_for_type = TypeDesc.for_type(
            xr.DataArray,
            dtype="float64",
            dims=(
                ("x", 3, None),
                ("y", 4, None),
            ),
        )

        assert td_from_value.kind == td_for_type.kind == "xarray.DataArray"

    def test_dataset_roundtrip(self) -> None:
        ds = xr.Dataset({"a": xr.DataArray(np.zeros((2, 3)), dims=["x", "y"])})
        td = TypeDesc.from_value(ds)

        assert td.kind == "xarray.Dataset"
        assert td.fields is not None
        assert "a" in {key for key, _ in td.fields}
