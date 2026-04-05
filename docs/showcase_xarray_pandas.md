# Xarray + Pandas MultiIndex Showcase

This showcase demonstrates paired workflows represented both in execution inference and explicit specs.

## Covered paired concepts

- weighted cross-sectional return
- rolling beta
- ridge exposures
- selection + concat flow

## Execution inference (generic route)

```python
from typetrace.core import TypeDesc
from typetrace.inference import infer_by_execution

# xarray-like panel
xarray_input = TypeDesc(kind="ndarray", dims={"time": 6, "asset": 3}, dtype="float64")

def weighted_xarray(da):
    w = da.mean("time")
    w = w / w.sum("asset")
    return (da * w).sum("asset")

print(infer_by_execution(weighted_xarray, xarray_input))
# => TypeDesc(kind='ndarray', dims={'time': 6}, dtype='float64')
```

```python
# pandas MultiIndex panel
panel_input = TypeDesc(
    kind="dataframe",
    columns=["ret", "mkt", ...],  # known columns + unknown extras
    dtypes={"ret": "float64", "mkt": "float64"},
    index={"time": 4, "asset": 3},
)

def rolling_beta_panel(df):
    grp = df.groupby(level="asset")
    return grp["ret"].rolling(3).cov(grp["mkt"].rolling(3).mean()).rename("beta")

print(infer_by_execution(rolling_beta_panel, panel_input))
```

## Explicit spec route

The paired recipes live in `specs/recipes.yaml` with ids:

- `weighted_cross_sectional_return_xarray`
- `weighted_cross_sectional_return_pandas_multiindex`
- `rolling_beta_xarray`
- `rolling_beta_pandas_multiindex`
- `ridge_exposures_xarray`
- `ridge_exposures_pandas_multiindex`
- `selection_concat_flow_xarray`
- `selection_concat_flow_pandas_multiindex`

Tests validate these ids exist and exercise equivalent xarray/pandas transformations.
