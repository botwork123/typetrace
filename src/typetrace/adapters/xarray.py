"""
xarray adapter for typetrace.

Handles xarray DataArray and Dataset types.
"""

import os
from typing import Any

from typetrace.core import Dims, Symbol, TypeDesc


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
        dims: Dims = {str(name): size for name, size in zip(value.dims, value.shape)}
        dtype = str(value.dtype) if value.dtype != np.dtype("O") else "object"
        return TypeDesc(kind="ndarray", dims=dims, dtype=dtype)
    elif isinstance(value, xr.Dataset):
        fields = {name: from_xarray(da) for name, da in value.data_vars.items()}
        return TypeDesc(kind="dataset", fields=fields)
    else:
        raise TypeError(f"Expected xarray type, got {type(value)}")


def _default_sample_size() -> int:
    value = os.getenv("TYPETRACE_SAMPLE_SIZE", "4")
    return max(int(value), 1)


def _sample_dim_size(size: int | Symbol) -> int:
    if isinstance(size, Symbol):
        return _default_sample_size()
    return size if size > 0 else _default_sample_size()


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

    dim_names = list(type_desc.dims.keys())
    shape = tuple(_sample_dim_size(size) for size in type_desc.dims.values())
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
        for name, field_td in type_desc.fields.items():
            if field_td.kind == "ndarray" and field_td.dims:
                data_vars[name] = make_xarray_sample(field_td)
        return xr.Dataset(data_vars)

    # No fields - create single-variable Dataset from dims/dtype
    if type_desc.dims is None:
        raise ValueError("Cannot make Dataset sample without dims or fields")

    dim_names = list(type_desc.dims.keys())
    shape = tuple(_sample_dim_size(size) for size in type_desc.dims.values())
    coords = {name: _coord_values(name, size) for name, size in zip(dim_names, shape)}
    dtype = type_desc.dtype or "float64"
    data = np.arange(int(np.prod(shape)), dtype="float64").reshape(shape).astype(dtype)
    da = xr.DataArray(data, dims=dim_names, coords=coords)
    return xr.Dataset({"data": da})
