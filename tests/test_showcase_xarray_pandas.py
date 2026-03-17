"""Paired xarray/pandas showcase coverage for specs and execution inference."""

from importlib.util import find_spec

import pytest
import yaml

from typetrace.core import TypeDesc
from typetrace.inference import infer_by_execution

XR = pytest.mark.skipif(find_spec("xarray") is None, reason="xarray not installed")
PD = pytest.mark.skipif(find_spec("pandas") is None, reason="pandas not installed")


@pytest.mark.parametrize(
    "concept,expected_ids",
    [
        (
            "weighted_cross_sectional_return",
            [
                "weighted_cross_sectional_return_xarray",
                "weighted_cross_sectional_return_pandas_multiindex",
            ],
        ),
        ("rolling_beta", ["rolling_beta_xarray", "rolling_beta_pandas_multiindex"]),
        ("ridge_exposures", ["ridge_exposures_xarray", "ridge_exposures_pandas_multiindex"]),
        (
            "selection_concat_flow",
            ["selection_concat_flow_xarray", "selection_concat_flow_pandas_multiindex"],
        ),
    ],
)
def test_paired_recipes_present(concept: str, expected_ids: list[str]) -> None:
    with open("specs/recipes.yaml", encoding="utf-8") as handle:
        recipe_ids = {r["id"] for r in yaml.safe_load(handle)["recipes"]}
    assert all(recipe_id in recipe_ids for recipe_id in expected_ids), concept


@XR
@pytest.mark.parametrize(
    "fn,input_type,expected_dims",
    [
        (
            lambda da: (da * (da.mean("time") / da.mean("time").sum())).sum("asset"),
            TypeDesc(kind="ndarray", dims={"time": 6, "asset": 3}, dtype="float64"),
            {"time": 6},
        ),
        (
            lambda da: __import__("xarray").cov(
                da.rolling(time=3).construct("window"),
                da.rolling(time=3).construct("window"),
                dim="window",
            ),
            TypeDesc(kind="ndarray", dims={"time": 6}, dtype="float64"),
            {"time": 6},
        ),
        (
            lambda da: da.isel(asset=[0, 1])
            .rename(asset="scenario")
            .assign_coords(scenario=["base", "alt"]),
            TypeDesc(kind="ndarray", dims={"time": 5, "asset": 3}, dtype="float64"),
            {"time": 5, "scenario": 2},
        ),
    ],
)
def test_infer_by_execution_xarray_showcase(
    fn, input_type: TypeDesc, expected_dims: dict[str, int]
) -> None:
    result = infer_by_execution(fn, input_type)
    assert result.kind == "ndarray"
    assert result.dims == expected_dims


@PD
@pytest.mark.parametrize(
    "fn,input_type,expected_kind,expected_index",
    [
        (
            lambda df: (
                df["ret"]
                * df.index.get_level_values("asset").map({"A000": 0.6, "A001": 0.4, "A002": 0.0})
            )
            .groupby(level="time")
            .sum(),
            TypeDesc(
                kind="dataframe",
                columns=["ret"],
                dtypes={"ret": "float64"},
                index={"time": 4, "asset": 3},
            ),
            "series",
            {"time": 4},
        ),
        (
            lambda df: df.groupby(level="asset")
            .apply(lambda g: g["ret"].rolling(3).cov(g["mkt"]))
            .droplevel(0)
            .to_frame("beta"),
            TypeDesc(
                kind="dataframe",
                columns=["ret", "mkt"],
                dtypes={"ret": "float64", "mkt": "float64"},
                index={"time": 5, "asset": 3},
            ),
            "dataframe",
            {"time": 5, "asset": 3},
        ),
    ],
)
def test_infer_by_execution_pandas_showcase(
    fn, input_type: TypeDesc, expected_kind: str, expected_index: dict[str, int]
) -> None:
    result = infer_by_execution(fn, input_type)
    assert result.kind == expected_kind
    assert result.index == expected_index


@XR
@PD
@pytest.mark.parametrize("alpha", [0.1, 1.0])
def test_infer_by_execution_ridge_exposures(alpha: float) -> None:
    import numpy as np
    import xarray as xr

    def ridge_xarray(factors: xr.DataArray, returns: xr.DataArray) -> xr.DataArray:
        x = factors.values
        y = returns.values
        beta = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
        return xr.DataArray(beta, dims=("factor", "asset"))

    factors_t = TypeDesc(kind="ndarray", dims={"time": 6, "factor": 2}, dtype="float64")
    returns_t = TypeDesc(kind="ndarray", dims={"time": 6, "asset": 3}, dtype="float64")
    result = infer_by_execution(ridge_xarray, factors_t, returns_t)
    assert result.kind == "ndarray"
    assert result.dims == {"factor": 2, "asset": 3}
