"""
xarray adapter for typetrace.

Handles xarray DataArray and Dataset types.
"""

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
        # For Dataset, return a class-like TypeDesc with fields
        fields = {name: from_xarray(da) for name, da in value.data_vars.items()}
        return TypeDesc(kind="class", fields=fields)
    else:
        raise TypeError(f"Expected xarray type, got {type(value)}")


def make_xarray_sample(type_desc: TypeDesc) -> Any:
    """
    Create minimal xarray DataArray from TypeDesc.

    Creates a zero-sized array with correct dims and dtype.

    Args:
        type_desc: TypeDesc with kind='ndarray'

    Returns:
        xarray.DataArray with correct structure
    """
    import numpy as np
    import xarray as xr

    if type_desc.dims is None:
        raise ValueError("Cannot make xarray sample without dims")

    # Use size 0 for each dimension to minimize memory
    shape = []
    dim_names = []
    for name, size in type_desc.dims.items():
        dim_names.append(name)
        if isinstance(size, Symbol):
            shape.append(0)  # Symbolic dims get size 0 in sample
        else:
            shape.append(0)  # All dims get size 0 for minimal sample

    dtype = type_desc.dtype or "float64"
    data = np.empty(shape, dtype=dtype)

    return xr.DataArray(data, dims=dim_names)
