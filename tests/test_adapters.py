"""Tests for typetrace adapters."""

from importlib.util import find_spec

import pytest


# Skip markers for optional dependencies
def skip_if_no_pandas():
    return find_spec("pandas") is None


def skip_if_no_xarray():
    return find_spec("xarray") is None


def skip_if_no_polars():
    return find_spec("polars") is None


def skip_if_no_pyarrow():
    return find_spec("pyarrow") is None


def skip_if_no_drjit():
    return find_spec("drjit") is None


pandas_required = pytest.mark.skipif(skip_if_no_pandas(), reason="pandas not installed")
xarray_required = pytest.mark.skipif(skip_if_no_xarray(), reason="xarray not installed")
polars_required = pytest.mark.skipif(skip_if_no_polars(), reason="polars not installed")
pyarrow_required = pytest.mark.skipif(skip_if_no_pyarrow(), reason="pyarrow not installed")
drjit_required = pytest.mark.skipif(skip_if_no_drjit(), reason="drjit not installed")


@pandas_required
class TestPandasAdapter:
    """Tests for pandas adapter."""

    def test_from_pandas_dataframe(self) -> None:
        """from_pandas extracts TypeDesc from DataFrame."""
        import pandas as pd

        from typetrace.adapters.pandas import from_pandas

        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        result = from_pandas(df)

        assert result.kind == "dataframe"
        assert result.columns == ["a", "b"]
        assert result.dtypes == {"a": "int64", "b": "float64"}

    def test_from_pandas_dataframe_with_named_index(self) -> None:
        """from_pandas captures named index."""
        import pandas as pd

        from typetrace.adapters.pandas import from_pandas

        df = pd.DataFrame({"a": [1, 2, 3]})
        df.index.name = "row_id"
        result = from_pandas(df)

        assert result.index == {"row_id": 3}

    def test_from_pandas_dataframe_with_multiindex(self) -> None:
        """from_pandas captures MultiIndex."""
        import pandas as pd

        from typetrace.adapters.pandas import from_pandas

        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        df.index = pd.MultiIndex.from_tuples(
            [("x", 1), ("x", 2), ("y", 1), ("y", 2)], names=["group", "sub"]
        )
        result = from_pandas(df)

        assert result.index == {"group": 2, "sub": 2}

    def test_from_pandas_series(self) -> None:
        """from_pandas extracts TypeDesc from Series."""
        import pandas as pd

        from typetrace.adapters.pandas import from_pandas

        s = pd.Series([1.0, 2.0, 3.0], name="values")
        result = from_pandas(s)

        assert result.kind == "series"
        assert result.dtype == "float64"

    def test_from_pandas_series_with_named_index(self) -> None:
        """from_pandas captures Series named index."""
        import pandas as pd

        from typetrace.adapters.pandas import from_pandas

        s = pd.Series([1, 2, 3])
        s.index.name = "idx"
        result = from_pandas(s)

        assert result.index == {"idx": 3}

    def test_from_pandas_invalid_type(self) -> None:
        """from_pandas raises TypeError for non-pandas types."""
        from typetrace.adapters.pandas import from_pandas

        with pytest.raises(TypeError, match="Expected pandas type"):
            from_pandas([1, 2, 3])

    def test_make_dataframe_sample(self) -> None:
        """make_dataframe_sample creates empty DataFrame with schema."""
        import pandas as pd

        from typetrace.adapters.pandas import make_dataframe_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(
            kind="dataframe",
            columns=["a", "b", "c"],
            dtypes={"a": "float64", "b": "int32", "c": "bool"},
        )
        result = make_dataframe_sample(t)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b", "c"]
        assert len(result) == 4
        assert result["a"].iloc[0] == 0.0
        assert result["b"].iloc[1] == 1
        assert bool(result["c"].iloc[0]) is True

    def test_make_dataframe_sample_with_index(self) -> None:
        """make_dataframe_sample sets index name."""

        from typetrace.adapters.pandas import make_dataframe_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(
            kind="dataframe",
            columns=["a"],
            dtypes={"a": "float64"},
            index={"row_id": 10},
        )
        result = make_dataframe_sample(t)

        assert result.index.name == "row_id"
        assert len(result) == 10
        assert result["a"].iloc[3] == 0.3333333333333333

    def test_make_dataframe_sample_no_columns(self) -> None:
        """make_dataframe_sample raises ValueError without columns."""
        from typetrace.adapters.pandas import make_dataframe_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="dataframe")
        with pytest.raises(ValueError, match="Cannot make DataFrame sample"):
            make_dataframe_sample(t)

    def test_make_series_sample(self) -> None:
        """make_series_sample creates empty Series with dtype."""
        import pandas as pd

        from typetrace.adapters.pandas import make_series_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="series", dtype="int64")
        result = make_series_sample(t)

        assert isinstance(result, pd.Series)
        assert result.dtype == "int64"
        assert len(result) == 4
        assert result.iloc[-1] == 3

    def test_make_series_sample_with_index(self) -> None:
        """make_series_sample sets index name."""

        from typetrace.adapters.pandas import make_series_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="series", dtype="float64", index={"time": 100})
        result = make_series_sample(t)

        assert result.index.name == "time"
        assert len(result) == 100
        assert result.iloc[50] == 0.5050505050505051

    def test_make_series_sample_default_dtype(self) -> None:
        """make_series_sample uses float64 as default dtype."""

        from typetrace.adapters.pandas import make_series_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="series")
        result = make_series_sample(t)

        assert result.dtype == "float64"


@xarray_required
class TestXarrayAdapter:
    """Tests for xarray adapter."""

    def test_from_xarray_dataarray(self) -> None:
        """from_xarray extracts TypeDesc from DataArray."""
        import numpy as np
        import xarray as xr

        from typetrace.adapters.xarray import from_xarray

        da = xr.DataArray(np.zeros((10, 20)), dims=["x", "y"], attrs={"units": "meters"})
        result = from_xarray(da)

        assert result.kind == "ndarray"
        assert result.dims == {"x": 10, "y": 20}
        assert result.dtype == "float64"

    def test_from_xarray_dataarray_with_object_dtype(self) -> None:
        """from_xarray handles object dtype."""
        import numpy as np
        import xarray as xr

        from typetrace.adapters.xarray import from_xarray

        # Create actual object dtype array with mixed types
        da = xr.DataArray(np.array([1, "a", None], dtype=object), dims=["x"])
        result = from_xarray(da)

        assert result.dtype == "object"

    def test_from_xarray_dataset(self) -> None:
        """from_xarray extracts TypeDesc from Dataset."""
        import numpy as np
        import xarray as xr

        from typetrace.adapters.xarray import from_xarray

        ds = xr.Dataset(
            {
                "temp": xr.DataArray(np.zeros((10, 20)), dims=["x", "y"]),
                "pressure": xr.DataArray(np.zeros((10,)), dims=["x"]),
            }
        )
        result = from_xarray(ds)

        assert result.kind == "dataset"
        assert "temp" in result.fields
        assert "pressure" in result.fields
        assert result.fields["temp"].dims == {"x": 10, "y": 20}
        assert result.fields["pressure"].dims == {"x": 10}

    def test_from_xarray_invalid_type(self) -> None:
        """from_xarray raises TypeError for non-xarray types."""
        from typetrace.adapters.xarray import from_xarray

        with pytest.raises(TypeError, match="Expected xarray type"):
            from_xarray([1, 2, 3])

    def test_make_xarray_sample(self) -> None:
        """make_xarray_sample creates DataArray with correct dims."""
        import xarray as xr

        from typetrace.adapters.xarray import make_xarray_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="ndarray", dims={"x": 10, "y": 20}, dtype="float32")
        result = make_xarray_sample(t)

        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {"x", "y"}
        assert result.dtype == "float32"
        assert result.shape == (10, 20)
        assert result.coords["x"].values[0] == 0

    def test_make_xarray_sample_with_symbol(self) -> None:
        """make_xarray_sample handles symbolic dims."""
        import xarray as xr

        from typetrace.adapters.xarray import make_xarray_sample
        from typetrace.core import Symbol, TypeDesc

        t = TypeDesc(kind="ndarray", dims={"x": Symbol("N"), "y": 20}, dtype="float64")
        result = make_xarray_sample(t)

        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {"x", "y"}
        assert result.shape == (4, 20)

    def test_make_xarray_sample_no_dims(self) -> None:
        """make_xarray_sample raises ValueError without dims."""
        from typetrace.adapters.xarray import make_xarray_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="ndarray")
        with pytest.raises(ValueError, match="Cannot make xarray sample"):
            make_xarray_sample(t)

    def test_make_xarray_sample_default_dtype(self) -> None:
        """make_xarray_sample uses float64 as default dtype."""

        from typetrace.adapters.xarray import make_xarray_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="ndarray", dims={"x": 10})
        result = make_xarray_sample(t)

        assert result.dtype == "float64"


@polars_required
class TestPolarsAdapter:
    """Tests for Polars adapter."""

    def test_from_polars_dataframe(self) -> None:
        """from_polars extracts TypeDesc from DataFrame."""
        import polars as pl

        from typetrace.adapters.polars import from_polars

        df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        result = from_polars(df)

        assert result.kind == "dataframe"
        assert result.columns == ["a", "b"]
        assert "a" in result.dtypes
        assert "b" in result.dtypes

    def test_from_polars_series(self) -> None:
        """from_polars extracts TypeDesc from Series."""
        import polars as pl

        from typetrace.adapters.polars import from_polars

        s = pl.Series("values", [1.0, 2.0, 3.0])
        result = from_polars(s)

        assert result.kind == "series"
        assert "float" in result.dtype.lower() or "f64" in result.dtype.lower()

    def test_from_polars_invalid_type(self) -> None:
        """from_polars raises TypeError for non-Polars types."""
        from typetrace.adapters.polars import from_polars

        with pytest.raises(TypeError, match="Expected Polars type"):
            from_polars([1, 2, 3])

    @pytest.mark.parametrize(
        "columns,dtypes,expected_dtypes",
        [
            (["a", "b"], {"a": "Float64", "b": "Int64"}, ["Float64", "Int64"]),
            (["x"], {"x": "Boolean"}, ["Boolean"]),
            (["c", "d"], {}, ["Float64", "Float64"]),  # Default dtype
        ],
    )
    def test_make_polars_dataframe_sample(
        self, columns: list, dtypes: dict, expected_dtypes: list
    ) -> None:
        """make_polars_dataframe_sample creates empty DataFrame with schema."""
        import polars as pl

        from typetrace.adapters.polars import make_polars_dataframe_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="dataframe", columns=columns, dtypes=dtypes)
        result = make_polars_dataframe_sample(t)

        assert isinstance(result, pl.DataFrame)
        assert result.columns == columns
        assert len(result) == 0
        for col, expected in zip(columns, expected_dtypes):
            assert str(result[col].dtype) == expected

    def test_make_polars_dataframe_sample_no_columns(self) -> None:
        """make_polars_dataframe_sample raises ValueError without columns."""
        from typetrace.adapters.polars import make_polars_dataframe_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="dataframe")
        with pytest.raises(ValueError, match="Cannot make Polars DataFrame sample"):
            make_polars_dataframe_sample(t)

    @pytest.mark.parametrize(
        "dtype,expected_dtypes",
        [
            ("Float64", ["Float64"]),
            ("Int32", ["Int32"]),
            ("Boolean", ["Boolean"]),
            ("Utf8", ["Utf8", "String"]),  # Polars renamed Utf8 -> String
            (None, ["Float64"]),  # Default dtype
        ],
    )
    def test_make_polars_series_sample(self, dtype: str | None, expected_dtypes: list) -> None:
        """make_polars_series_sample creates empty Series with dtype."""
        import polars as pl

        from typetrace.adapters.polars import make_polars_series_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="series", dtype=dtype)
        result = make_polars_series_sample(t)

        assert isinstance(result, pl.Series)
        assert len(result) == 0
        assert str(result.dtype) in expected_dtypes

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            ("Float32", "Float32"),
            ("Int16", "Int16"),
            ("Int8", "Int8"),
            ("UInt64", "UInt64"),
            ("UInt32", "UInt32"),
            ("UInt16", "UInt16"),
            ("UInt8", "UInt8"),
            ("unknown", "Float64"),
        ],
    )
    def test_get_polars_dtype_additional_branches(self, dtype: str, expected: str) -> None:
        """_get_polars_dtype maps integer/unsigned and fallback dtypes."""
        from typetrace.adapters.polars import _get_polars_dtype

        assert _get_polars_dtype(dtype).__name__ == expected


@pyarrow_required
class TestArrowAdapter:
    """Tests for Arrow adapter."""

    def test_from_arrow_table(self) -> None:
        """from_arrow extracts TypeDesc from Table."""
        import pyarrow as pa

        from typetrace.adapters.arrow import from_arrow

        table = pa.table({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        result = from_arrow(table)

        assert result.kind == "columnar"
        assert result.columns == ["a", "b"]
        assert "a" in result.dtypes
        assert "b" in result.dtypes

    def test_from_arrow_array(self) -> None:
        """from_arrow extracts TypeDesc from Array."""
        import pyarrow as pa

        from typetrace.adapters.arrow import from_arrow

        arr = pa.array([1.0, 2.0, 3.0])
        result = from_arrow(arr)

        assert result.kind == "series"
        assert "double" in result.dtype or "float" in result.dtype

    def test_from_arrow_invalid_type(self) -> None:
        """from_arrow raises TypeError for non-Arrow types."""
        from typetrace.adapters.arrow import from_arrow

        with pytest.raises(TypeError, match="Expected Arrow type"):
            from_arrow([1, 2, 3])

    @pytest.mark.parametrize(
        "columns,dtypes,expected_types",
        [
            (["a", "b"], {"a": "float64", "b": "int64"}, ["double", "int64"]),
            (["x"], {"x": "bool"}, ["bool"]),
            (["c", "d"], {}, ["double", "double"]),  # Default dtype
            (["s"], {"s": "string"}, ["string"]),
        ],
    )
    def test_make_arrow_table_sample(
        self, columns: list, dtypes: dict, expected_types: list
    ) -> None:
        """make_arrow_table_sample creates empty Table with schema."""
        import pyarrow as pa

        from typetrace.adapters.arrow import make_arrow_table_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="columnar", columns=columns, dtypes=dtypes)
        result = make_arrow_table_sample(t)

        assert isinstance(result, pa.Table)
        assert result.column_names == columns
        assert result.num_rows == 0
        for col, expected in zip(columns, expected_types):
            assert str(result.schema.field(col).type) == expected

    def test_make_arrow_table_sample_no_columns(self) -> None:
        """make_arrow_table_sample raises ValueError without columns."""
        from typetrace.adapters.arrow import make_arrow_table_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="columnar")
        with pytest.raises(ValueError, match="Cannot make Arrow Table sample"):
            make_arrow_table_sample(t)

    @pytest.mark.parametrize(
        "dtype,expected_type",
        [
            ("float64", "double"),
            ("int32", "int32"),
            ("bool", "bool"),
            ("string", "string"),
            (None, "double"),  # Default dtype
        ],
    )
    def test_make_arrow_array_sample(self, dtype: str | None, expected_type: str) -> None:
        """make_arrow_array_sample creates empty Array with type."""
        import pyarrow as pa

        from typetrace.adapters.arrow import make_arrow_array_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="series", dtype=dtype)
        result = make_arrow_array_sample(t)

        assert isinstance(result, pa.Array)
        assert len(result) == 0
        assert str(result.type) == expected_type

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            ("float32", "float"),
            ("int16", "int16"),
            ("int8", "int8"),
            ("uint64", "uint64"),
            ("uint32", "uint32"),
            ("uint16", "uint16"),
            ("uint8", "uint8"),
            ("mystery", "double"),
        ],
    )
    def test_get_arrow_type_additional_branches(self, dtype: str, expected: str) -> None:
        """_get_arrow_type maps additional integer/unsigned and fallback dtypes."""
        from typetrace.adapters.arrow import _get_arrow_type

        assert str(_get_arrow_type(dtype)) == expected

    def test_make_sample_columnar_via_core(self) -> None:
        """TypeDesc.make_sample() works for columnar kind."""
        import pyarrow as pa

        from typetrace.core import TypeDesc

        t = TypeDesc(
            kind="columnar",
            columns=["x", "y"],
            dtypes={"x": "int64", "y": "float64"},
        )
        result = t.make_sample()

        assert isinstance(result, pa.Table)
        assert result.column_names == ["x", "y"]
        assert result.num_rows == 0


@drjit_required
class TestDrJitAdapter:
    """Tests for DrJit adapter."""

    def test_from_drjit_float_array(self) -> None:
        """from_drjit extracts TypeDesc from float array."""
        from drjit import llvm

        from typetrace.adapters.drjit import from_drjit

        arr = llvm.Float64([1.0, 2.0, 3.0])
        result = from_drjit(arr)

        assert result.kind == "drjit"
        assert result.dtype == "float64"
        assert result.drjit_type is type(arr)

    def test_from_drjit_int_array(self) -> None:
        """from_drjit extracts TypeDesc from int array."""
        from drjit import llvm

        from typetrace.adapters.drjit import from_drjit

        arr = llvm.Int([1, 2, 3])
        result = from_drjit(arr)

        assert result.kind == "drjit"
        assert "int" in result.dtype

    @pytest.mark.parametrize(
        "type_name,expected_dtype",
        [
            ("Float64Array", "float64"),
            ("Float32Array", "float32"),
            ("DoubleArray", "float64"),
            ("FloatArray", "float32"),
            ("Int64Array", "int64"),
            ("Int32Array", "int32"),
            ("IntArray", "int32"),
            ("UInt64Array", "uint64"),
            ("UInt32Array", "uint32"),
            ("UIntArray", "uint32"),
            ("BoolArray", "bool"),
            ("UnknownArray", "unknown"),
        ],
    )
    def test_drjit_dtype_inference(self, type_name: str, expected_dtype: str) -> None:
        """_drjit_dtype correctly infers dtype from type name."""
        from typetrace.adapters.drjit import _drjit_dtype

        class MockArray:
            pass

        MockArray.__name__ = type_name
        result = _drjit_dtype(MockArray())
        assert result == expected_dtype

    def test_make_drjit_sample_with_type(self) -> None:
        """make_drjit_sample creates array with correct type."""
        import drjit as dr
        from drjit import llvm

        from typetrace.adapters.drjit import make_drjit_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="drjit", drjit_type=llvm.Float64, dtype="float64")
        result = make_drjit_sample(t)

        assert type(result) is llvm.Float64
        assert dr.width(result) == 0

    def test_make_drjit_sample_infer_type(self) -> None:
        """make_drjit_sample infers type from dtype."""
        from drjit import llvm

        from typetrace.adapters.drjit import make_drjit_sample
        from typetrace.core import TypeDesc

        t = TypeDesc(kind="drjit", dtype="int64")
        result = make_drjit_sample(t)

        assert type(result) is llvm.Int64
