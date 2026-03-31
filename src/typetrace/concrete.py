"""
Static concrete type propagation for typetrace.

Provides `concrete_transform()` to determine output types from operations
without runtime execution. Registry-based approach for:
- Method calls: (pd.DataFrame, "sum") → pd.Series
- Binary ops: ((int, int), "truediv") → float
- Unary ops: (int, "neg") → int
"""

from __future__ import annotations

from typing import overload

# Method call rules: {(module_root, class_name, method): result_class_name}
# None means scalar (no concrete type), "same" means same as input
_METHOD_RULES: dict[tuple[str, str, str], str | None] = {
    # pandas DataFrame aggregations → Series
    ("pandas", "DataFrame", "sum"): "Series",
    ("pandas", "DataFrame", "mean"): "Series",
    ("pandas", "DataFrame", "std"): "Series",
    ("pandas", "DataFrame", "var"): "Series",
    ("pandas", "DataFrame", "min"): "Series",
    ("pandas", "DataFrame", "max"): "Series",
    ("pandas", "DataFrame", "count"): "Series",
    ("pandas", "DataFrame", "median"): "Series",
    ("pandas", "DataFrame", "prod"): "Series",
    # pandas Series aggregations → scalar
    ("pandas", "Series", "sum"): None,
    ("pandas", "Series", "mean"): None,
    ("pandas", "Series", "std"): None,
    ("pandas", "Series", "var"): None,
    ("pandas", "Series", "min"): None,
    ("pandas", "Series", "max"): None,
    ("pandas", "Series", "count"): None,
    ("pandas", "Series", "median"): None,
    ("pandas", "Series", "prod"): None,
    # dask DataFrame → pandas DataFrame
    ("dask", "DataFrame", "head"): "pandas.DataFrame",
    ("dask", "DataFrame", "compute"): "pandas.DataFrame",
    ("dask", "DataFrame", "tail"): "pandas.DataFrame",
    # dask Series → pandas Series
    ("dask", "Series", "head"): "pandas.Series",
    ("dask", "Series", "compute"): "pandas.Series",
    # xarray aggregations (stay xarray)
    ("xarray", "DataArray", "mean"): "same",
    ("xarray", "DataArray", "sum"): "same",
    ("xarray", "DataArray", "std"): "same",
    ("xarray", "DataArray", "min"): "same",
    ("xarray", "DataArray", "max"): "same",
    ("xarray", "Dataset", "mean"): "same",
    ("xarray", "Dataset", "sum"): "same",
}

# Binary operation rules: {(left_type, right_type, op): result_type}
# Using type names as strings for portability
_BINARY_RULES: dict[tuple[str, str, str], type | None] = {
    # Division always returns float
    ("int", "int", "truediv"): float,
    ("int", "float", "truediv"): float,
    ("float", "int", "truediv"): float,
    ("float", "float", "truediv"): float,
    # Floor division preserves int
    ("int", "int", "floordiv"): int,
    ("int", "float", "floordiv"): float,
    ("float", "int", "floordiv"): float,
    ("float", "float", "floordiv"): float,
    # Comparison returns bool
    ("int", "int", "lt"): bool,
    ("int", "int", "le"): bool,
    ("int", "int", "gt"): bool,
    ("int", "int", "ge"): bool,
    ("int", "int", "eq"): bool,
    ("int", "int", "ne"): bool,
    ("float", "float", "lt"): bool,
    ("float", "float", "gt"): bool,
    ("float", "float", "eq"): bool,
    ("int", "float", "lt"): bool,
    ("int", "float", "gt"): bool,
    ("int", "float", "eq"): bool,
    ("float", "int", "lt"): bool,
    ("float", "int", "gt"): bool,
    ("float", "int", "eq"): bool,
    # Type promotion: int + float → float
    ("int", "float", "add"): float,
    ("float", "int", "add"): float,
    ("int", "float", "sub"): float,
    ("float", "int", "sub"): float,
    ("int", "float", "mul"): float,
    ("float", "int", "mul"): float,
}

# Unary operation rules: {(type, op): result_type}
_UNARY_RULES: dict[tuple[str, str], type | None] = {
    ("int", "neg"): int,
    ("int", "pos"): int,
    ("int", "abs"): int,
    ("int", "invert"): int,
    ("float", "neg"): float,
    ("float", "pos"): float,
    ("float", "abs"): float,
    ("bool", "not"): bool,
    ("bool", "invert"): int,  # ~True == -2
}


def _get_type_name(t: type) -> str:
    """Get normalized type name for lookup."""
    return t.__name__


def _resolve_method_result(input_type: type, result: str | None) -> type | None:
    """Resolve method result string to actual type."""
    if result is None:
        return None
    if result == "same":
        return input_type
    # Handle cross-module results like "pandas.DataFrame"
    if "." in result:
        module, cls_name = result.rsplit(".", 1)
        try:
            mod = __import__(module)
            cls: type = getattr(mod, cls_name)
            return cls
        except (ImportError, AttributeError):
            return None
    # Same-module result
    module = input_type.__module__.split(".")[0]
    try:
        mod = __import__(module)
        cls2: type = getattr(mod, result)
        return cls2
    except (ImportError, AttributeError):
        return None


@overload
def concrete_transform(input_type: type, operation: str) -> type | None:
    ...


@overload
def concrete_transform(input_type: tuple[type, type], operation: str) -> type | None:
    ...


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
        return _transform_binary(input_type, operation)
    return _transform_method_or_unary(input_type, operation)


def _transform_binary(types: tuple[type, type], operation: str) -> type | None:
    """Handle binary operations."""
    left, right = types
    left_name = _get_type_name(left)
    right_name = _get_type_name(right)
    key = (left_name, right_name, operation)
    if key in _BINARY_RULES:
        return _BINARY_RULES[key]
    # Passthrough: return left type for unknown ops
    return left


def _transform_method_or_unary(input_type: type, operation: str) -> type | None:
    """Handle method calls and unary operations."""
    # Try unary first (for primitives)
    type_name = _get_type_name(input_type)
    unary_key = (type_name, operation)
    if unary_key in _UNARY_RULES:
        return _UNARY_RULES[unary_key]

    # Try method rules
    module = input_type.__module__.split(".")[0]
    class_name = input_type.__name__
    method_key = (module, class_name, operation)
    if method_key in _METHOD_RULES:
        return _resolve_method_result(input_type, _METHOD_RULES[method_key])

    # Passthrough: return input type for unknown methods
    return input_type
