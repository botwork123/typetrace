"""
numpy adapter for typetrace.

Handles numpy ndarray types.
"""

import os
from typing import Any

from typetrace.core import Dims, Symbol, TypeDesc


def from_numpy(value: Any) -> TypeDesc:
    """
    Extract TypeDesc from numpy ndarray.

    Args:
        value: numpy.ndarray

    Returns:
        TypeDesc with dims and dtype
    """
    import numpy as np

    if not isinstance(value, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(value)}")

    # Create named dimensions (dim_0, dim_1, etc.) with their sizes
    dims: Dims = {f"dim_{i}": size for i, size in enumerate(value.shape)}
    dtype = str(value.dtype)
    return TypeDesc(kind="ndarray", dims=dims, dtype=dtype)


def _default_sample_size() -> int:
    value = os.getenv("TYPETRACE_SAMPLE_SIZE", "4")
    return max(int(value), 1)


def _sample_dim_size(size: int | Symbol) -> int:
    if isinstance(size, Symbol):
        return _default_sample_size()
    return size if size > 0 else _default_sample_size()


def make_numpy_sample(type_desc: TypeDesc) -> Any:
    """
    Create numpy ndarray from TypeDesc.

    Builds a small, non-empty array with meaningful data so execution-based
    inference can exercise operations on it.
    """
    import numpy as np

    if type_desc.dims is None:
        raise ValueError("Cannot make numpy sample without dims")

    # Extract shape from dims
    shape = tuple(_sample_dim_size(size) for size in type_desc.dims.values())
    dtype = type_desc.dtype or "float64"
    
    # Create array with sequential values to make debugging easier
    total_size = int(np.prod(shape))
    data = np.arange(total_size, dtype="float64").reshape(shape)
    
    # Cast to target dtype
    return data.astype(dtype)
