"""
Common dimension transform patterns.

These are reusable building blocks for type_transform implementations.
Most calcs use one of these patterns, so we avoid duplicating logic.
"""

from typetrace.core import Dims, DimValue, Symbol


class DimMismatch(Exception):
    """Raised when dimensions don't match as required."""

    pass


def unify(d1: Dims | None, d2: Dims | None) -> Dims:
    """
    Unify two dimension sets - dims must match where they overlap.

    Used for element-wise operations (spread, add, sub).
    Both inputs must have compatible dims.

    Args:
        d1: First dimension set
        d2: Second dimension set

    Returns:
        Merged dimensions (intersection semantics - must match)

    Raises:
        DimMismatch: If same-named dims have different sizes
    """
    if d1 is None:
        return d2 or {}
    if d2 is None:
        return d1

    result = dict(d1)
    for name, size in d2.items():
        if name in result:
            if result[name] != size:
                raise DimMismatch(f"Dimension {name!r}: {result[name]} vs {size}")
        else:
            result[name] = size
    return result


def broadcast(d1: Dims | None, d2: Dims | None) -> Dims:
    """
    Broadcast two dimension sets - xarray-style union.

    Used for operations that expand dims (outer product).
    Result has all dims from both inputs.

    Args:
        d1: First dimension set
        d2: Second dimension set

    Returns:
        Union of dimensions (all dims from both)
    """
    if d1 is None:
        return d2 or {}
    if d2 is None:
        return d1

    return {**d1, **d2}


def add_dim(d: Dims | None, name: str, size: DimValue) -> Dims:
    """
    Add a new dimension.

    Used for operations that introduce a new axis (rolling windows, expand_dims).

    Args:
        d: Existing dimensions
        name: Name of new dimension
        size: Size of new dimension (int or Symbol)

    Returns:
        Dimensions with new dim added
    """
    result = dict(d) if d else {}
    result[name] = size
    return result


def reduce_dim(d: Dims | None, name: str) -> Dims:
    """
    Remove a dimension.

    Used for aggregation operations (sum, mean over a dim).

    Args:
        d: Existing dimensions
        name: Name of dimension to remove

    Returns:
        Dimensions with specified dim removed
    """
    if d is None:
        return {}
    return {k: v for k, v in d.items() if k != name}


def promote_dtype(dtype1: str | None, dtype2: str | None) -> str | None:
    """
    Promote two dtypes to their common supertype.

    Simple promotion rules (can be extended):
    - float64 wins over float32
    - float wins over int
    - Same type returns same

    Args:
        dtype1: First dtype
        dtype2: Second dtype

    Returns:
        Promoted dtype
    """
    if dtype1 is None:
        return dtype2
    if dtype2 is None:
        return dtype1
    if dtype1 == dtype2:
        return dtype1

    # Simple promotion hierarchy
    hierarchy = ["bool", "int32", "int64", "float32", "float64"]
    try:
        idx1 = hierarchy.index(dtype1)
        idx2 = hierarchy.index(dtype2)
        return hierarchy[max(idx1, idx2)]
    except ValueError:
        # Unknown dtype, return first
        return dtype1


def bind_symbols(d: Dims | None, bindings: dict[str, int]) -> Dims:
    """
    Bind symbolic dimensions to concrete values.

    Args:
        d: Dimensions with potential Symbol values
        bindings: Map from symbol name to concrete int

    Returns:
        Dimensions with symbols replaced by concrete values where bound
    """
    if d is None:
        return {}

    result: Dims = {}
    for name, size in d.items():
        if isinstance(size, Symbol) and size.name in bindings:
            result[name] = bindings[size.name]
        else:
            result[name] = size
    return result


# =============================================================================
# Unary operation type transforms
# =============================================================================

# Operations that always return bool
_BOOL_RESULT_OPS = frozenset({"not", "isnan", "isinf", "isfinite"})

# Operations that convert complex to real
_COMPLEX_TO_REAL_OPS = frozenset({"abs", "real", "imag"})

# Operations that return int (sign, comparison results in some contexts)
_INT_RESULT_OPS = frozenset({"sign"})


# =============================================================================
# Binary operation type transforms
# =============================================================================

# Comparison operations always return bool
_COMPARISON_OPS = frozenset({
    "eq", "ne", "lt", "le", "gt", "ge",
    "==", "!=", "<", "<=", ">", ">=",
})

# True division always returns float
_FLOAT_RESULT_OPS = frozenset({"truediv", "/"})

# Floor division returns int
_INT_RESULT_OPS_BINARY = frozenset({"floordiv", "//"})


def binary_result_dtype(
    left_dtype: str | None,
    right_dtype: str | None,
    operation: str,
) -> str | None:
    """
    Compute result dtype for a binary operation.

    Type rules:
    - Comparisons (eq, ne, lt, le, gt, ge) → bool
    - True division (/) → float64
    - Floor division (//) → int64
    - Other ops → promote_dtype(left, right)

    Args:
        left_dtype: Left operand dtype
        right_dtype: Right operand dtype
        operation: Operation name (e.g., "add", "eq", "truediv")

    Returns:
        Result dtype
    """
    if operation in _COMPARISON_OPS:
        return "bool"

    if operation in _FLOAT_RESULT_OPS:
        # Division always produces float
        return "float64"

    if operation in _INT_RESULT_OPS_BINARY:
        return "int64"

    # Default: promote both operand types
    return promote_dtype(left_dtype, right_dtype)


def unary_result_dtype(input_dtype: str | None, operation: str) -> str | None:
    """
    Compute result dtype for a unary operation.

    Type rules:
    - not, isnan, isinf, isfinite → bool
    - abs on complex → float64
    - sign → int64
    - neg, pos, exp, log, sqrt, etc. → preserve dtype

    Args:
        input_dtype: Input dtype (e.g., "float64", "complex128")
        operation: Operation name (e.g., "neg", "abs", "not")

    Returns:
        Result dtype
    """
    if operation in _BOOL_RESULT_OPS:
        return "bool"

    if operation in _INT_RESULT_OPS:
        return "int64"

    if operation in _COMPLEX_TO_REAL_OPS:
        if input_dtype and "complex" in input_dtype:
            # complex64 → float32, complex128 → float64
            return "float32" if input_dtype == "complex64" else "float64"

    # Default: preserve dtype
    return input_dtype
