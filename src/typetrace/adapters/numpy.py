"""Adapter for numpy arrays and scalars."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typetrace.core import TypeDesc


def from_numpy(value: Any) -> "TypeDesc":
    """Extract TypeDesc from numpy array or scalar.

    Args:
        value: A numpy ndarray or numpy scalar (np.float64, np.int32, etc.)

    Returns:
        TypeDesc with kind="ndarray" for arrays, kind="scalar" for numpy scalars
    """
    import numpy as np

    from typetrace.core import TypeDesc

    # Handle numpy scalars (np.float64, np.int32, etc.)
    if isinstance(value, np.generic):
        return TypeDesc(kind="scalar", dtype=str(value.dtype))

    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        # Build dims dict from shape
        # Use positional dim names: dim0, dim1, dim2, ...
        dims = {f"dim{i}": size for i, size in enumerate(value.shape)}
        return TypeDesc(kind="ndarray", dtype=str(value.dtype), dims=dims)

    raise TypeError(f"Expected numpy array or scalar, got {type(value)}")
