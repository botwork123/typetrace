"""Tests for concrete_transform static type propagation."""

from importlib.util import find_spec

import pytest
from typetrace.concrete import concrete_transform

# Conditional imports with markers
pandas_available = find_spec("pandas") is not None
dask_available = find_spec("dask") is not None
xarray_available = find_spec("xarray") is not None

pandas_required = pytest.mark.skipif(not pandas_available, reason="pandas not installed")
dask_required = pytest.mark.skipif(not dask_available, reason="dask not installed")
xarray_required = pytest.mark.skipif(not xarray_available, reason="xarray not installed")


class TestMethodTransforms:
    """Tests for method call type transforms."""

    @pandas_required
    @pytest.mark.parametrize(
        "input_type,method,expected_name",
        [
            ("DataFrame", "sum", "Series"),
            ("DataFrame", "mean", "Series"),
            ("DataFrame", "std", "Series"),
            ("DataFrame", "var", "Series"),
            ("DataFrame", "min", "Series"),
            ("DataFrame", "max", "Series"),
            ("DataFrame", "count", "Series"),
            ("DataFrame", "median", "Series"),
            ("DataFrame", "prod", "Series"),
        ],
    )
    def test_pandas_dataframe_aggregations(
        self, input_type: str, method: str, expected_name: str
    ) -> None:
        """pandas DataFrame aggregations return Series."""
        import pandas as pd

        input_cls = getattr(pd, input_type)
        expected_cls = getattr(pd, expected_name)

        result = concrete_transform(input_cls, method)

        assert result == expected_cls

    @pandas_required
    @pytest.mark.parametrize(
        "method",
        ["sum", "mean", "std", "var", "min", "max", "count", "median", "prod"],
    )
    def test_pandas_series_aggregations_return_scalar(self, method: str) -> None:
        """pandas Series aggregations return None (scalar)."""
        import pandas as pd

        result = concrete_transform(pd.Series, method)

        assert result is None

    @dask_required
    @pytest.mark.parametrize(
        "input_type,method",
        [
            ("DataFrame", "head"),
            ("DataFrame", "compute"),
            ("DataFrame", "tail"),
        ],
    )
    def test_dask_dataframe_to_pandas(self, input_type: str, method: str) -> None:
        """dask DataFrame methods return pandas DataFrame."""
        import dask.dataframe as dd
        import pandas as pd

        input_cls = getattr(dd, input_type)

        result = concrete_transform(input_cls, method)

        assert result == pd.DataFrame

    @dask_required
    @pytest.mark.parametrize(
        "method",
        ["head", "compute"],
    )
    def test_dask_series_to_pandas(self, method: str) -> None:
        """dask Series methods return pandas Series."""
        import dask.dataframe as dd
        import pandas as pd

        result = concrete_transform(dd.Series, method)

        assert result == pd.Series

    @xarray_required
    @pytest.mark.parametrize(
        "input_type,method",
        [
            ("DataArray", "mean"),
            ("DataArray", "sum"),
            ("DataArray", "std"),
            ("DataArray", "min"),
            ("DataArray", "max"),
            ("Dataset", "mean"),
            ("Dataset", "sum"),
        ],
    )
    def test_xarray_aggregations_same_type(self, input_type: str, method: str) -> None:
        """xarray aggregations return same type."""
        import xarray as xr

        input_cls = getattr(xr, input_type)

        result = concrete_transform(input_cls, method)

        assert result == input_cls


class TestBinaryOps:
    """Tests for binary operation type transforms."""

    @pytest.mark.parametrize(
        "left,right,op,expected",
        [
            (int, int, "truediv", float),
            (int, float, "truediv", float),
            (float, int, "truediv", float),
            (float, float, "truediv", float),
        ],
    )
    def test_truediv_returns_float(self, left: type, right: type, op: str, expected: type) -> None:
        """True division always returns float."""
        result = concrete_transform((left, right), op)

        assert result == expected

    @pytest.mark.parametrize(
        "left,right,op,expected",
        [
            (int, int, "floordiv", int),
            (int, float, "floordiv", float),
            (float, int, "floordiv", float),
            (float, float, "floordiv", float),
        ],
    )
    def test_floordiv_types(self, left: type, right: type, op: str, expected: type) -> None:
        """Floor division preserves int for int//int, else float."""
        result = concrete_transform((left, right), op)

        assert result == expected

    @pytest.mark.parametrize(
        "left,right,op",
        [
            (int, int, "lt"),
            (int, int, "le"),
            (int, int, "gt"),
            (int, int, "ge"),
            (int, int, "eq"),
            (int, int, "ne"),
            (float, float, "lt"),
            (float, float, "gt"),
            (float, float, "eq"),
            (int, float, "lt"),
            (int, float, "gt"),
            (float, int, "lt"),
        ],
    )
    def test_comparison_returns_bool(self, left: type, right: type, op: str) -> None:
        """Comparison operations return bool."""
        result = concrete_transform((left, right), op)

        assert result is bool

    @pytest.mark.parametrize(
        "left,right,op,expected",
        [
            (int, float, "add", float),
            (float, int, "add", float),
            (int, float, "sub", float),
            (float, int, "sub", float),
            (int, float, "mul", float),
            (float, int, "mul", float),
        ],
    )
    def test_type_promotion_int_float(
        self, left: type, right: type, op: str, expected: type
    ) -> None:
        """int + float operations promote to float."""
        result = concrete_transform((left, right), op)

        assert result == expected


class TestUnaryOps:
    """Tests for unary operation type transforms."""

    @pytest.mark.parametrize(
        "input_type,op,expected",
        [
            (int, "neg", int),
            (int, "pos", int),
            (int, "abs", int),
            (int, "invert", int),
            (float, "neg", float),
            (float, "pos", float),
            (float, "abs", float),
            (bool, "not", bool),
            (bool, "invert", int),  # ~True == -2
        ],
    )
    def test_unary_ops(self, input_type: type, op: str, expected: type) -> None:
        """Unary operations preserve or transform type correctly."""
        result = concrete_transform(input_type, op)

        assert result == expected


class TestPassthrough:
    """Tests for passthrough behavior on unknown operations."""

    def test_unknown_method_returns_input_type(self) -> None:
        """Unknown method returns input type (passthrough)."""
        result = concrete_transform(str, "unknown_method")

        assert result is str

    def test_unknown_binary_returns_left_type(self) -> None:
        """Unknown binary op returns left type (passthrough)."""
        result = concrete_transform((str, int), "unknown_op")

        assert result is str

    @pandas_required
    def test_unknown_pandas_method_passthrough(self) -> None:
        """Unknown pandas method returns input type."""
        import pandas as pd

        result = concrete_transform(pd.DataFrame, "nonexistent_method")

        assert result == pd.DataFrame
