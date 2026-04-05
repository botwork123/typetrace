"""Sample-path inference contracts for wrapped package methods/functions."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import pytest
from typetrace.core import TypeDesc
from typetrace.inference import infer_by_execution

pandas_required = pytest.mark.skipif(find_spec("pandas") is None, reason="pandas not installed")
xarray_required = pytest.mark.skipif(find_spec("xarray") is None, reason="xarray not installed")
polars_required = pytest.mark.skipif(find_spec("polars") is None, reason="polars not installed")
dask_required = pytest.mark.skipif(find_spec("dask") is None, reason="dask not installed")


def _invoke_method(method: str):
    def _call(obj: Any, **kwargs: Any) -> Any:
        return getattr(obj, method)(**kwargs)

    return _call


PANDAS_METHOD_CASES = [
    {
        "method": "sum",
        "type_desc": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "float64", "b": "int64"},
        ),
        "kwargs": {},
        "expected": TypeDesc(kind="series", dtype="float64", index=None),
    },
    {
        "method": "head",
        "type_desc": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "float64", "b": "int64"},
        ),
        "kwargs": {"n": 2},
        "expected": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "float64", "b": "int64"},
        ),
    },
]

XARRAY_METHOD_CASES = [
    {
        "method": "mean",
        "type_desc": TypeDesc(kind="ndarray", dims={"time": 5, "asset": 3}, dtype="float64"),
        "kwargs": {"dim": "time"},
        "expected": TypeDesc(kind="ndarray", dims={"asset": 3}, dtype="float64"),
    },
    {
        "method": "isel",
        "type_desc": TypeDesc(kind="ndarray", dims={"time": 5, "asset": 3}, dtype="float64"),
        "kwargs": {"time": 0},
        "expected": TypeDesc(kind="ndarray", dims={"asset": 3}, dtype="float64"),
    },
]

POLARS_METHOD_CASES = [
    {
        "method": "head",
        "type_desc": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "Float64", "b": "Int64"},
        ),
        "kwargs": {"n": 2},
        "expected": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "Float64", "b": "Int64"},
        ),
    },
    {
        "method": "tail",
        "type_desc": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "Float64", "b": "Int64"},
        ),
        "kwargs": {"n": 1},
        "expected": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "Float64", "b": "Int64"},
        ),
    },
]

DASK_FUNCTION_CASES = [
    {
        "method": "head",
        "type_desc": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "float64", "b": "int64"},
        ),
        "kwargs": {"n": 2},
        "expected": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "float64", "b": "int64"},
        ),
    },
    {
        "method": "compute",
        "type_desc": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "float64", "b": "int64"},
        ),
        "kwargs": {},
        "expected": TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "float64", "b": "int64"},
        ),
    },
]

WRAPPED_METHOD_INVENTORY = {
    "pandas": {case["method"] for case in PANDAS_METHOD_CASES},
    "xarray": {case["method"] for case in XARRAY_METHOD_CASES},
    "polars": {case["method"] for case in POLARS_METHOD_CASES},
    "dask": {case["method"] for case in DASK_FUNCTION_CASES},
}


@pandas_required
@pytest.mark.parametrize(
    "case", PANDAS_METHOD_CASES, ids=[c["method"] for c in PANDAS_METHOD_CASES]
)
def test_pandas_method_sample_contracts(case: dict[str, Any]) -> None:
    result = infer_by_execution(_invoke_method(case["method"]), case["type_desc"], **case["kwargs"])

    assert result.kind == case["expected"].kind
    assert result.columns == case["expected"].columns
    assert result.dtypes == case["expected"].dtypes
    assert result.dims == case["expected"].dims
    assert result.dtype == case["expected"].dtype
    assert result.index == case["expected"].index


@xarray_required
@pytest.mark.parametrize(
    "case", XARRAY_METHOD_CASES, ids=[c["method"] for c in XARRAY_METHOD_CASES]
)
def test_xarray_method_sample_contracts(case: dict[str, Any]) -> None:
    result = infer_by_execution(_invoke_method(case["method"]), case["type_desc"], **case["kwargs"])

    assert result.kind == case["expected"].kind
    assert result.dims == case["expected"].dims
    assert result.dtype == case["expected"].dtype


@polars_required
@pytest.mark.parametrize(
    "case", POLARS_METHOD_CASES, ids=[c["method"] for c in POLARS_METHOD_CASES]
)
def test_polars_method_sample_contracts(case: dict[str, Any]) -> None:
    import polars as pl

    def _polars_method(method: str):
        def _call(df: Any, **kwargs: Any) -> Any:
            pl_df = pl.from_pandas(df)
            return getattr(pl_df, method)(**kwargs)

        return _call

    result = infer_by_execution(_polars_method(case["method"]), case["type_desc"], **case["kwargs"])

    assert result.kind == case["expected"].kind
    assert result.columns == case["expected"].columns
    assert result.dtypes == case["expected"].dtypes


@dask_required
@pytest.mark.parametrize(
    "case", DASK_FUNCTION_CASES, ids=[c["method"] for c in DASK_FUNCTION_CASES]
)
def test_dask_dataframe_method_sample_contracts(case: dict[str, Any]) -> None:
    import dask.dataframe as dd

    def _dask_method(method: str):
        def _call(df: Any, **kwargs: Any) -> Any:
            ddf = dd.from_pandas(df, npartitions=2)
            return getattr(ddf, method)(**kwargs)

        return _call

    result = infer_by_execution(_dask_method(case["method"]), case["type_desc"], **case["kwargs"])

    assert result.kind == case["expected"].kind
    assert result.columns == case["expected"].columns
    assert result.dtypes == case["expected"].dtypes


@pytest.mark.parametrize(
    "backend,cases",
    [
        ("pandas", PANDAS_METHOD_CASES),
        ("xarray", XARRAY_METHOD_CASES),
        ("polars", POLARS_METHOD_CASES),
        ("dask", DASK_FUNCTION_CASES),
    ],
)
def test_wrapped_method_inventory_covered(backend: str, cases: list[dict[str, Any]]) -> None:
    tested = {case["method"] for case in cases}
    assert tested == WRAPPED_METHOD_INVENTORY[backend]


def test_infer_by_execution_fail_fast_sample_build_error_context() -> None:
    with pytest.raises(ValueError, match="sample-build failed") as excinfo:
        infer_by_execution(_invoke_method("head"), TypeDesc(kind="dataframe"), n=1)

    message = str(excinfo.value)
    assert "input index 0" in message
    assert "Cannot make DataFrame sample without columns" in message


def test_infer_by_execution_fail_fast_execution_error_context() -> None:
    def _boom(_: Any) -> Any:
        raise RuntimeError("explode")

    with pytest.raises(ValueError, match="execution failed") as excinfo:
        infer_by_execution(
            _boom,
            TypeDesc(kind="dataframe", columns=["a"], dtypes={"a": "float64"}),
        )

    assert "explode" in str(excinfo.value)
