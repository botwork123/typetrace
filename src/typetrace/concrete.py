"""
Static concrete type propagation for typetrace.

Provides `concrete_transform()` to determine output types from operations
without runtime execution. Uses general rules instead of lookup tables:
- Comparison ops → bool
- Division ops → float
- Aggregation ops → reduce dimension
- Type promotion for binary ops
"""

from __future__ import annotations

from typing import cast, overload

# Operation categories (frozenset for O(1) lookup)
_COMPARISON_OPS = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})
_DIVISION_OPS = frozenset({"truediv", "div"})
_FLOOR_DIV_OPS = frozenset({"floordiv"})
_AGGREGATION_OPS = frozenset(
    {
        "sum",
        "mean",
        "std",
        "var",
        "min",
        "max",
        "count",
        "median",
        "prod",
        "sem",
        "skew",
        "kurt",
        "quantile",
        "nunique",
        "idxmin",
        "idxmax",
        "all",
        "any",
    }
)
_DASK_TO_PANDAS_OPS = frozenset({"head", "tail", "compute"})
_PRESERVE_TYPE_UNARY_OPS = frozenset({"neg", "pos", "abs", "invert"})
_BOOL_UNARY_OPS = frozenset({"not"})


def _is_dask_type(t: type) -> bool:
    """Check if type is from dask."""
    return t.__module__.startswith("dask")


def _is_xarray_type(t: type) -> bool:
    """Check if type is from xarray."""
    return t.__module__.startswith("xarray")


def _is_pandas_type(t: type) -> bool:
    """Check if type is from pandas."""
    return t.__module__.startswith("pandas")


def _dask_to_pandas(t: type) -> type | None:
    """Convert dask type to pandas equivalent."""
    try:
        import pandas as pd
    except ImportError:
        return None

    name = t.__name__
    if name == "DataFrame":
        return cast(type, pd.DataFrame)
    if name == "Series":
        return cast(type, pd.Series)
    return None


def _reduce_type(t: type) -> type | None:
    """
    Aggregation reduces dimension: DataFrame→Series, Series→scalar.
    xarray stays xarray (fewer dims).
    """
    # Handle optional pandas
    try:
        import pandas as pd

        if t is pd.DataFrame:
            return cast(type, pd.Series)
        if t is pd.Series:
            return None  # scalar
    except ImportError:
        pass

    # xarray aggregations stay xarray
    if _is_xarray_type(t):
        return t

    # Unknown types: passthrough
    return t


def _promote_types(left: type, right: type) -> type:
    """
    Binary op result: highest precedence type wins.
    Array types dominate, then float dominates int.
    """
    # Try to get array types (optional deps)
    array_types: list[type] = []

    try:
        import xarray as xr

        array_types.append(xr.DataArray)
    except ImportError:
        pass

    try:
        import numpy as np

        array_types.append(np.ndarray)
    except ImportError:
        pass

    try:
        import pandas as pd

        array_types.extend([pd.DataFrame, pd.Series])
    except ImportError:
        pass

    # Array types dominate
    for arr_type in array_types:
        if left is arr_type or right is arr_type:
            return arr_type

    # float dominates int
    if left is float or right is float:
        return float

    return left


def _handle_binary_op(left: type, right: type, operation: str) -> type | None:
    """Handle binary operations with general rules."""
    # Comparisons → bool
    if operation in _COMPARISON_OPS:
        return bool

    # True division → float
    if operation in _DIVISION_OPS:
        return float

    # Floor division: int//int → int, otherwise float
    if operation in _FLOOR_DIV_OPS:
        if left is int and right is int:
            return int
        if left in (int, float) and right in (int, float):
            return float
        return _promote_types(left, right)

    # Type promotion for other ops (add, sub, mul, etc.)
    return _promote_types(left, right)


def _handle_unary_or_method(input_type: type, operation: str) -> type | None:
    """Handle unary operations and method calls."""
    # Unary ops that preserve type (neg, pos, abs, invert)
    if operation in _PRESERVE_TYPE_UNARY_OPS:
        # Special case: bool invert returns int (~True == -2)
        if input_type is bool and operation == "invert":
            return int
        return input_type

    # Bool unary ops (not)
    if operation in _BOOL_UNARY_OPS:
        return bool

    # Dask → pandas conversions
    if operation in _DASK_TO_PANDAS_OPS and _is_dask_type(input_type):
        result = _dask_to_pandas(input_type)
        if result is not None:
            return result

    # Aggregations reduce dimension
    if operation in _AGGREGATION_OPS:
        return _reduce_type(input_type)

    # Default: passthrough (same type out)
    return input_type


@overload
def concrete_transform(input_type: type, operation: str) -> type | None: ...


@overload
def concrete_transform(input_type: tuple[type, type], operation: str) -> type | None: ...


def concrete_transform(input_type: type | tuple[type, type], operation: str) -> type | None:
    """
    Determine output type from an operation without runtime execution.

    Args:
        input_type: Single type for method/unary ops, tuple for binary ops
        operation: Method name or operator name (e.g., "sum", "truediv", "neg")

    Returns:
        Output type, or None if result is scalar/unknown.
        Returns input_type if no rule found (passthrough).

    Examples:
        >>> concrete_transform(pd.DataFrame, "sum")
        <class 'pandas.core.series.Series'>

        >>> concrete_transform((int, int), "truediv")
        <class 'float'>

        >>> concrete_transform(int, "neg")
        <class 'int'>
    """
    if isinstance(input_type, tuple):
        left, right = input_type
        return _handle_binary_op(left, right, operation)

    return _handle_unary_or_method(input_type, operation)
