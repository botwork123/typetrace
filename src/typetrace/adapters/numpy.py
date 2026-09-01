"""Adapter for numpy arrays and scalars."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typetrace.core import TypeDesc

ADAPTER_KINDS = ("numpy.ndarray",)
OPERATIONS: dict[tuple[str, str], Any] = {}


def supports(value: object) -> bool:
    """Return whether value is a NumPy ndarray."""
    import numpy as np

    return isinstance(value, np.ndarray)


def from_numpy(value: Any) -> "TypeDesc":
    """Extract TypeDesc from numpy array or scalar.

    Args:
        value: A numpy ndarray or numpy scalar (np.float64, np.int32, etc.)

    Returns:
        TypeDesc with kind="numpy.ndarray" for arrays, kind="scalar" for numpy scalars
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
        dims = tuple((f"dim{i}", size, None) for i, size in enumerate(value.shape))
        return TypeDesc(kind="numpy.ndarray", dtype=str(value.dtype), dims=dims)

    raise TypeError(f"Expected numpy array or scalar, got {type(value)}")


def make_numpy_sample(type_desc: "TypeDesc") -> Any:
    """Create a minimal NumPy array matching a descriptor's shape contract."""
    import numpy as np

    if type_desc.shape is not None:
        shape = tuple(size if isinstance(size, int) else 0 for size in type_desc.shape)
    elif type_desc.dims is not None:
        shape = tuple(size if isinstance(size, int) else 0 for _, size, _ in type_desc.dims)
    else:
        shape = ()
    return np.empty(shape, dtype=type_desc.dtype or "float64")


def infer(value: object) -> "TypeDesc":
    """Protocol entry point for NumPy inference."""
    return from_numpy(value)


def make_sample(desc: "TypeDesc") -> object:
    """Protocol entry point for NumPy sample creation."""
    return make_numpy_sample(desc)


def validate(desc: "TypeDesc", value: object) -> None:
    """Validate a NumPy sample against its nominal kind and dtype."""
    import numpy as np

    if desc.kind != "numpy.ndarray" or not isinstance(value, np.ndarray):
        raise TypeError(f"expected numpy.ndarray sample, got {type(value)!r}")
    if desc.dtype is not None and str(value.dtype) != desc.dtype:
        raise TypeError(f"numpy sample dtype {value.dtype!s} does not match {desc.dtype!r}")
