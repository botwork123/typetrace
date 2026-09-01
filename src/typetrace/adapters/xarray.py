"""
xarray adapter for typetrace.

Handles xarray DataArray and Dataset types.
"""

import os
from typing import Any

from typetrace.core import Dims, Symbol, TypeDesc

ADAPTER_KINDS = ("xarray.DataArray", "xarray.Dataset")
OPERATIONS: dict[tuple[str, str], Any] = {}


def supports(value: object) -> bool:
    """Return whether value is an xarray DataArray or Dataset."""
    import xarray as xr

    return isinstance(value, (xr.DataArray, xr.Dataset))


def from_xarray(value: Any) -> TypeDesc:
    """
    Extract TypeDesc from xarray DataArray or Dataset.

    Args:
        value: xarray.DataArray or xarray.Dataset

    Returns:
        TypeDesc with dims and dtype
    """
    import numpy as np
    import xarray as xr

    if isinstance(value, xr.DataArray):
        dims: Dims = tuple((str(name), size, None) for name, size in zip(value.dims, value.shape))
        dtype = str(value.dtype) if value.dtype != np.dtype("O") else "object"
        return TypeDesc(kind="xarray.DataArray", dims=dims, dtype=dtype)
    elif isinstance(value, xr.Dataset):
        fields = tuple((name, from_xarray(da)) for name, da in value.data_vars.items())
        return TypeDesc(kind="xarray.Dataset", fields=fields)
    else:
        raise TypeError(f"Expected xarray type, got {type(value)}")


def _default_sample_size() -> int:
    value = os.getenv("TYPETRACE_SAMPLE_SIZE", "4")
    return max(int(value), 1)


def _sample_dim_size(size: int | Symbol) -> int:
    if isinstance(size, Symbol):
        return _default_sample_size()
    return size


def _coord_values(dim_name: str, size: int) -> Any:
    import numpy as np

    key = dim_name.lower()
    if "time" in key or key in {"date", "datetime"}:
        return np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + size)
    if "asset" in key or key in {"symbol", "ticker"}:
        return np.array([f"A{i:03d}" for i in range(size)], dtype=object)
    return np.arange(size)


def make_xarray_sample(type_desc: TypeDesc) -> Any:
    """
    Create xarray DataArray from TypeDesc.

    Builds a small, non-empty array with meaningful coordinates so execution-based
    inference can exercise selection, alignment, and concat behaviors.
    """
    import numpy as np
    import xarray as xr

    if type_desc.dims is None:
        raise ValueError("Cannot make xarray sample without dims")

    entries = list(type_desc.dims)
    dim_names = [name for name, _, _ in entries]
    shape = tuple(_sample_dim_size(size) for _, size, _ in entries)
    coords = {name: _coord_values(name, size) for name, size in zip(dim_names, shape)}
    dtype = type_desc.dtype or "float64"
    data = np.arange(int(np.prod(shape)), dtype="float64").reshape(shape).astype(dtype)
    return xr.DataArray(data, dims=dim_names, coords=coords)


def make_dataset_sample(type_desc: TypeDesc) -> Any:
    """
    Create xarray Dataset from TypeDesc.

    If type_desc has fields (nested TypeDescs), creates a Dataset with
    one DataArray per field. Otherwise creates a Dataset with a single
    'data' variable using the dims/dtype from type_desc.
    """
    import numpy as np
    import xarray as xr

    if type_desc.fields:
        # Create Dataset from nested TypeDescs
        data_vars = {}
        for name, field_td in type_desc.fields:
            if field_td.kind == "xarray.DataArray" and field_td.dims:
                data_vars[name] = make_xarray_sample(field_td)
        return xr.Dataset(data_vars)

    # No fields - create single-variable Dataset from dims/dtype
    if type_desc.dims is None:
        raise ValueError("Cannot make Dataset sample without dims or fields")

    entries = list(type_desc.dims)
    dim_names = [name for name, _, _ in entries]
    shape = tuple(_sample_dim_size(size) for _, size, _ in entries)
    coords = {name: _coord_values(name, size) for name, size in zip(dim_names, shape)}
    dtype = type_desc.dtype or "float64"
    data = np.arange(int(np.prod(shape)), dtype="float64").reshape(shape).astype(dtype)
    da = xr.DataArray(data, dims=dim_names, coords=coords)
    return xr.Dataset({"data": da})


def infer(value: object) -> TypeDesc:
    """Protocol entry point for xarray inference."""
    return from_xarray(value)


def make_sample(desc: TypeDesc) -> object:
    """Protocol entry point for xarray sample creation."""
    return (
        make_xarray_sample(desc) if desc.kind == "xarray.DataArray" else make_dataset_sample(desc)
    )


def validate(desc: TypeDesc, value: object) -> None:
    """Validate an xarray sample against its nominal kind."""
    import xarray as xr

    expected = xr.DataArray if desc.kind == "xarray.DataArray" else xr.Dataset
    if desc.kind not in ADAPTER_KINDS or not isinstance(value, expected):
        raise TypeError(f"expected {desc.kind} sample, got {type(value)!r}")
