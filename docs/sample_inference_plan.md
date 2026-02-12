# Sample Inference Plan

## Goals

1. Improve generic `make_sample()` realism so execution-based inference exercises common shape/index paths.
2. Add explicit paired recipe showcases for xarray and pandas MultiIndex workflows.

## Architecture

- Keep `TypeDesc.make_sample()` dispatch unchanged for backward compatibility.
- Upgrade only adapter builders:
  - `xarray.make_xarray_sample()` now returns small non-empty arrays with deterministic coords.
  - `pandas.make_dataframe_sample()` / `make_series_sample()` now return deterministic non-empty samples with realistic indexes, including MultiIndex.
- Preserve extraction APIs (`from_xarray`, `from_pandas`) so inferred contracts remain represented in `TypeDesc`.

## Scope

- xarray sample enhancements:
  - meaningful coord labels (`time`, `asset`, generic integer labels)
  - symbolic dims resolved to configurable sample size
  - deterministic numeric payloads
- pandas sample enhancements:
  - deterministic typed column values
  - robust MultiIndex generation from `TypeDesc.index`
  - named defaults and time/asset-aware labels
- recipe pack enhancements:
  - paired xarray/pandas-MultiIndex recipes for weighted return, rolling beta, ridge exposures, and selection+concat
- tests:
  - explicit-spec validation (paired recipes present)
  - execution inference validation (paired xarray/pandas flows)

## Tradeoffs

- Behavior changes from empty samples to small non-empty samples improve inference coverage but may alter assumptions in downstream tests.
- To keep control, sample size is configurable via `TYPETRACE_SAMPLE_SIZE` (default `4`).
- MultiIndex metadata now stores per-level cardinality (nunique) for better reconstruction fidelity.
