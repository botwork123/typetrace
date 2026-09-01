"""
Common dimension transform patterns.

These are reusable building blocks for type_transform implementations.
Most calcs use one of these patterns, so we avoid duplicating logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from typetrace.core import (
    Dims,
    DimValue,
    OperationBindingError,
    Symbol,
    TypeDesc,
    TypeDescConflictError,
    TypeDescUnknownError,
    TypeDescValidationError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from typetrace.core import TypeDesc


class DimMismatch(Exception):
    """Raised when dimensions don't match as required."""

    pass


def _dim_entries(
    dims: Dims | None,
) -> list[tuple[str, DimValue, tuple[Any, ...] | None]]:
    if dims is None:
        return []
    result = []
    for entry in dims:
        if len(entry) == 2:
            result.append((entry[0], entry[1], None))
        else:
            result.append((entry[0], entry[1], entry[2]))
    return result


def _dims_tuple(entries: list[tuple[str, DimValue, tuple[Any, ...] | None]]) -> Dims:
    return tuple(entries)


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
    result = _dim_entries(d1)
    positions = {name: index for index, (name, _, _) in enumerate(result)}
    for name, size, labels in _dim_entries(d2):
        if name in positions:
            index = positions[name]
            old_name, old_size, old_labels = result[index]
            if old_size != size or (
                old_labels is not None and labels is not None and old_labels != labels
            ):
                raise DimMismatch(f"Dimension {name!r}: {old_size} vs {size}")
            result[index] = (old_name, old_size, old_labels if old_labels is not None else labels)
        else:
            positions[name] = len(result)
            result.append((name, size, labels))
    return _dims_tuple(result)


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
    return unify(d1, d2)


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
    result = _dim_entries(d)
    if any(existing == name for existing, _, _ in result):
        raise DimMismatch(f"Dimension {name!r} already exists")
    result.append((name, size, None))
    return _dims_tuple(result)


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
    return _dims_tuple(
        [(key, size, labels) for key, size, labels in _dim_entries(d) if key != name]
    )


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
        return ()

    result: list[tuple[str, DimValue, tuple[Any, ...] | None]] = []
    for name, size, labels in _dim_entries(d):
        if isinstance(size, Symbol) and size.name in bindings:
            result.append((name, bindings[size.name], labels))
        else:
            result.append((name, size, labels))
    return _dims_tuple(result)


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
_COMPARISON_OPS = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
    }
)

# True division always returns float
_FLOAT_RESULT_OPS = frozenset({"truediv", "div", "/"})

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


# =============================================================================
# Full TypeDesc transforms (convenience API)
# =============================================================================


def apply_unary(td: TypeDesc, operation: str) -> TypeDesc:
    """
    Apply unary operation type transform to a TypeDesc.

    Preserves kind and dims, transforms dtype according to operation rules.

    Args:
        td: Input TypeDesc
        operation: Operation name (e.g., "neg", "abs", "not")

    Returns:
        New TypeDesc with transformed dtype
    """
    new_dtype = unary_result_dtype(td.dtype, operation)
    if new_dtype is None or new_dtype == td.dtype:
        return td
    return td.with_dtype(new_dtype)


def apply_binary(left: TypeDesc, right: TypeDesc, operation: str) -> TypeDesc:
    """
    Apply binary operation type transform to two TypeDescs.

    Broadcasts dims (xarray-style union), transforms dtype according to
    operation rules. Result kind follows left operand.

    Args:
        left: Left operand TypeDesc
        right: Right operand TypeDesc
        operation: Operation name (e.g., "add", "eq", "truediv")

    Returns:
        New TypeDesc with broadcasted dims and transformed dtype
    """
    new_dims = broadcast(left.dims, right.dims)
    new_dtype = binary_result_dtype(left.dtype, right.dtype, operation)

    return replace(left, dims=new_dims, dtype=new_dtype)


# =============================================================================
# TypeDesc v2 pure structural algebra
# =============================================================================

_BINARY_OPERATIONS = frozenset({"add", "sub", "mul", "div", "eq", "ne", "lt", "le", "gt", "ge"})
_UNARY_OPERATIONS = frozenset({"neg", "pos", "invert", "abs"})
_REDUCE_OPERATIONS = frozenset({"sum", "mean", "min", "max", "count"})
_METHOD_OPERATIONS = _REDUCE_OPERATIONS | {"astype"}


def _merge_labels(
    left: tuple[Any, ...] | None,
    right: tuple[Any, ...] | None,
    path: tuple[str | int, ...],
) -> tuple[Any, ...] | None:
    if left is None and right is None:
        return None
    if left is None or right is None:
        raise TypeDescUnknownError("one descriptor has unknown labels", path=path)
    if left != right:
        raise TypeDescConflictError("labels differ", path=path)
    return left


def _merge_dims(left: Dims | None, right: Dims | None, *, append_disjoint: bool) -> Dims | None:
    if left is None and right is None:
        return None
    left_entries = list(left or ())
    right_entries = list(right or ())
    right_by_name = {name: (size, labels) for name, size, labels in right_entries}
    result: list[tuple[str, DimValue, tuple[Any, ...] | None]] = []
    for name, size, labels in left_entries:
        if name not in right_by_name:
            if not append_disjoint and right_entries:
                raise TypeDescConflictError(f"dimension {name!r} is absent from right descriptor")
            result.append((name, size, labels))
            continue
        other_size, other_labels = right_by_name[name]
        if size != other_size:
            raise TypeDescConflictError(f"dimension {name!r} sizes differ")
        result.append((name, size, _merge_labels(labels, other_labels, ("dims", name, "labels"))))
    if append_disjoint:
        left_names = {name for name, _, _ in left_entries}
        result.extend(entry for entry in right_entries if entry[0] not in left_names)
    elif any(name not in {item[0] for item in left_entries} for name, _, _ in right_entries):
        raise TypeDescConflictError("descriptors have disjoint dimensions")
    return tuple(result)


def _merge_equal_payload(left: TypeDesc, right: TypeDesc, field: str) -> object:
    left_value = getattr(left, field)
    right_value = getattr(right, field)
    if left_value != right_value:
        raise TypeDescConflictError(f"{field} differs", path=(field,))
    return left_value


def _combine_type_desc(left: TypeDesc, right: TypeDesc, *, append_disjoint: bool) -> TypeDesc:
    if left.kind != right.kind:
        raise TypeDescConflictError("nominal kinds differ", path=("kind",))
    if left.metadata != right.metadata:
        raise TypeDescConflictError("metadata differs", path=("metadata",))
    dims = _merge_dims(left.dims, right.dims, append_disjoint=append_disjoint)
    index = _merge_dims(left.index, right.index, append_disjoint=append_disjoint)
    for field in ("shape", "columns", "dtypes", "fields", "drjit_type", "static_dims"):
        _merge_equal_payload(left, right, field)
    return replace(left, dims=dims, index=index)


def _binary_impl(left: TypeDesc, right: TypeDesc, operation: str) -> TypeDesc:
    if left.dtype is None and right.dtype is None:
        raise UnsupportedOperationError(f"binary {operation!r} requires an element dtype")
    combined = _combine_type_desc(left, right, append_disjoint=False)
    return replace(combined, dtype=binary_result_dtype(left.dtype, right.dtype, operation))


def _binary_type_desc(left: TypeDesc, right: TypeDesc, operation: str) -> TypeDesc:
    if not isinstance(right, TypeDesc):
        raise TypeDescConflictError("binary operand must be a TypeDesc", path=("other",))
    try:
        callback = STRUCTURAL_OPERATIONS[(left.kind, operation)]
    except KeyError as exc:
        raise UnsupportedOperationError(
            f"unsupported binary operation {operation!r} for kind {left.kind!r}"
        ) from exc
    return cast(TypeDesc, callback(left, (right,), {}))


def _unary_type_desc(td: TypeDesc, operation: str) -> TypeDesc:
    try:
        callback = STRUCTURAL_OPERATIONS[(td.kind, operation)]
    except KeyError as exc:
        raise UnsupportedOperationError(
            f"unsupported unary operation {operation!r} for kind {td.kind!r}"
        ) from exc
    return cast(TypeDesc, callback(td, (), {}))


def _unary_impl(td: TypeDesc, operation: str) -> TypeDesc:
    return replace(td, dtype=unary_result_dtype(td.dtype, operation))


def _reduced(td: TypeDesc, dimensions: tuple[str, ...], operation: str) -> TypeDesc:
    if operation == "count" and td.dtype is None:
        raise UnsupportedOperationError(f"count is unsupported for kind {td.kind!r}")
    if td.dims is None:
        if dimensions:
            raise TypeDescUnknownError("descriptor has no named dimensions", path=("dims",))
        return replace(td, dtype="int64" if operation == "count" else td.dtype)
    dims = list(td.dims or ())
    names = {name for name, _, _ in dims}
    missing = [name for name in dimensions if name not in names]
    if missing:
        raise TypeDescUnknownError(f"unknown dimensions: {missing}", path=("dims",))
    removed = set(dimensions)
    dtype = "int64" if operation == "count" else td.dtype
    return replace(td, dims=tuple(entry for entry in dims if entry[0] not in removed), dtype=dtype)


def _method_operand(td: TypeDesc, name: str, args: tuple[object, ...]) -> TypeDesc:
    if len(args) != 1 or not isinstance(args[0], TypeDesc):
        raise OperationBindingError(f"{name} requires exactly one TypeDesc operand")
    other = args[0]
    if other.kind != td.kind:
        raise TypeDescConflictError(f"{name}: nominal kinds differ", path=("kind",))
    return other


def _method_shape(td: TypeDesc, name: str) -> tuple[DimValue, ...]:
    if td.shape is not None:
        return td.shape
    if td.dims is not None:
        return tuple(size for _, size, _ in td.dims)
    raise TypeDescUnknownError(f"{name}: shape is unknown", path=("shape",))


def _method_axes(
    td: TypeDesc, name: str
) -> tuple[tuple[str, DimValue, tuple[Any, ...] | None], ...] | None:
    return td.dims if td.dims is not None else None


def _method_dtype(left: TypeDesc, right: TypeDesc, name: str) -> str | None:
    if left.dtype is None or right.dtype is None:
        raise TypeDescUnknownError(f"{name}: dtype is unknown", path=("dtype",))
    return (
        TypeDesc(kind="scalar", dtype=left.dtype)
        .binary(TypeDesc(kind="scalar", dtype=right.dtype), "mul")
        .dtype
    )


def _method_result(td: TypeDesc, shape: tuple[DimValue, ...], dtype: str | None) -> TypeDesc:
    if td.dims is not None:
        dims = tuple((f"dim{index}", size, None) for index, size in enumerate(shape))
        return replace(td, dims=dims, shape=None, dtype=dtype)
    return replace(td, shape=shape, dtype=dtype)


def _method_result_axes(
    td: TypeDesc,
    axes: tuple[tuple[str, DimValue, tuple[Any, ...] | None], ...],
    dtype: str | None,
) -> TypeDesc:
    return replace(td, dims=axes, shape=None, dtype=dtype)


def _matmul_method(
    td: TypeDesc, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> TypeDesc:
    if kwargs:
        raise OperationBindingError("matmul does not accept keyword arguments")
    other = _method_operand(td, "matmul", args)
    left = _method_shape(td, "matmul")
    right = _method_shape(other, "matmul")
    if not 1 <= len(left) <= 2 or not 1 <= len(right) <= 2:
        raise TypeDescValidationError("matmul: only rank-one and rank-two operands are supported")
    left_inner = left[-1]
    right_inner = right[0] if len(right) == 1 else right[-2]
    if left_inner != right_inner:
        raise TypeDescValidationError(
            f"matmul: contracted dimensions differ: {left_inner!r} != {right_inner!r}"
        )
    if len(left) == 1 and len(right) == 1:
        result_shape: tuple[DimValue, ...] = ()
    elif len(left) == 1:
        result_shape = (right[-1],)
    elif len(right) == 1:
        result_shape = (left[-2],)
    else:
        result_shape = (left[-2], right[-1])
    dtype = _method_dtype(td, other, "matmul")
    left_axes = _method_axes(td, "matmul")
    right_axes = _method_axes(other, "matmul")
    if left_axes is not None and right_axes is not None:
        if len(left) == 1 and len(right) == 1:
            return _method_result_axes(td, (), dtype)
        if len(left) == 1:
            return _method_result_axes(td, (right_axes[-1],), dtype)
        if len(right) == 1:
            return _method_result_axes(td, (left_axes[-2],), dtype)
        return _method_result_axes(td, (left_axes[-2], right_axes[-1]), dtype)
    return _method_result(td, result_shape, dtype)


def _outer_method(td: TypeDesc, args: tuple[object, ...], kwargs: Mapping[str, object]) -> TypeDesc:
    if kwargs:
        raise OperationBindingError("outer does not accept keyword arguments")
    other = _method_operand(td, "outer", args)
    left = _method_shape(td, "outer")
    right = _method_shape(other, "outer")
    if len(left) != 1 or len(right) != 1:
        raise TypeDescValidationError("outer: operands must be rank-one")
    dtype = _method_dtype(td, other, "outer")
    left_axes = _method_axes(td, "outer")
    right_axes = _method_axes(other, "outer")
    if left_axes is not None and right_axes is not None:
        if {axis[0] for axis in left_axes} & {axis[0] for axis in right_axes}:
            raise TypeDescConflictError("outer: axis names must be unique", path=("dims",))
        return _method_result_axes(td, left_axes + right_axes, dtype)
    return _method_result(
        td,
        left + right,
        dtype,
    )


def _stack_method(td: TypeDesc, args: tuple[object, ...], kwargs: Mapping[str, object]) -> TypeDesc:
    if (
        len(args) != 1
        or not isinstance(args[0], tuple)
        or not all(isinstance(item, TypeDesc) for item in args[0])
    ):
        raise OperationBindingError("stack requires one tuple of TypeDesc operands")
    others = args[0]
    axis = kwargs.get("axis", 0)
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeDescValidationError("stack: invalid axis")
    base = _method_shape(td, "stack")
    if any(item.kind != td.kind or _method_shape(item, "stack") != base for item in others):
        raise TypeDescConflictError("stack: operands must have matching nominal kind and shape")
    if axis < 0:
        axis += len(base) + 1
    if axis < 0 or axis > len(base):
        raise TypeDescValidationError("stack: invalid axis")
    shape = base[:axis] + (len(others) + 1,) + base[axis:]
    axes = _method_axes(td, "stack")
    if axes is not None:
        stacked_axes = axes[:axis] + ((f"stack{axis}", len(others) + 1, None),) + axes[axis:]
        return _method_result_axes(td, stacked_axes, td.dtype)
    return _method_result(td, shape, td.dtype)


def _method_impl(
    td: TypeDesc, name: str, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> TypeDesc:
    if name == "matmul":
        return _matmul_method(td, args, kwargs)
    if name == "outer":
        return _outer_method(td, args, kwargs)
    if name == "stack":
        return _stack_method(td, args, kwargs)
    if name == "astype":
        if td.dtype is None:
            raise UnsupportedOperationError(f"astype is unsupported for kind {td.kind!r}")
        dtype = kwargs.get("dtype", args[0] if len(args) == 1 else None)
        if not isinstance(dtype, str) or len(args) > 1 or (args and "dtype" in kwargs):
            raise OperationBindingError("astype requires exactly one string dtype")
        return replace(td, dtype=dtype)
    if (
        len(args) > 1
        or (args and ("dim" in kwargs or "axis" in kwargs))
        or ("dim" in kwargs and "axis" in kwargs)
    ):
        raise OperationBindingError("reduction accepts at most one dimension argument")
    dimensions = kwargs.get("dim", kwargs.get("axis", args[0] if args else None))
    if dimensions is None:
        selected: tuple[str, ...] = tuple(name for name, _, _ in (td.dims or ()))
    elif isinstance(dimensions, str):
        selected = (dimensions,)
    elif isinstance(dimensions, (tuple, list)) and all(
        isinstance(item, str) for item in dimensions
    ):
        selected = tuple(dimensions)
    else:
        raise OperationBindingError("reduction dimension must be a string or tuple of strings")
    return _reduced(td, selected, name)


def _method_type_desc(
    td: TypeDesc, name: str, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> TypeDesc:
    try:
        callback = STRUCTURAL_OPERATIONS[(td.kind, name)]
    except KeyError as exc:
        raise UnsupportedOperationError(
            f"unsupported method {name!r} for kind {td.kind!r}"
        ) from exc
    return cast(TypeDesc, callback(td, args, kwargs))


def _project_type_desc(td: TypeDesc, field: Any) -> TypeDesc:
    if td.fields is None:
        raise TypeDescUnknownError("fields are unknown", path=("fields",))
    for name, descriptor in td.fields:
        if name == field:
            return descriptor
    raise KeyError(field)


def _select_type_desc(td: TypeDesc, fields: tuple[Any, ...]) -> TypeDesc:
    if not fields or len(set(fields)) != len(fields):
        raise OperationBindingError("select requires unique non-empty fields")
    if td.kind == "record" and td.fields is not None:
        mapping = dict(td.fields)
        if any(field not in mapping for field in fields):
            missing = next(field for field in fields if field not in mapping)
            raise KeyError(missing)
        return replace(td, fields=tuple((field, mapping[field]) for field in fields))
    if td.kind not in {"pandas.DataFrame", "polars.DataFrame", "pyarrow.Table"}:
        raise UnsupportedOperationError(f"select is unsupported for kind {td.kind!r}")
    if td.columns is None:
        raise TypeDescUnknownError("columns are unknown", path=("columns",))
    known = set(td.columns)
    if any(field not in known for field in fields):
        missing = next(field for field in fields if field not in known)
        raise KeyError(missing)
    dtypes = (
        None
        if td.dtypes is None
        else tuple((field, dtype) for field, dtype in td.dtypes if field in fields)
    )
    return replace(td, columns=fields, dtypes=dtypes)


def _reduce_type_desc(td: TypeDesc, dimensions: tuple[str, ...], operation: str) -> TypeDesc:
    if operation not in _REDUCE_OPERATIONS:
        raise UnsupportedOperationError(f"unsupported reduction {operation!r}")
    return _reduced(td, dimensions, operation)


def _reshape_type_desc(td: TypeDesc, value: object) -> TypeDesc:
    if isinstance(value, (tuple, list)) and all(isinstance(item, int) for item in value):
        new_shape = tuple(value)
        if any(item < 0 for item in new_shape):
            raise OperationBindingError("reshape dimensions must be non-negative")
        old_shape = td.shape or tuple(size for _, size, _ in (td.dims or ()))
        if old_shape and all(isinstance(item, int) for item in old_shape):
            old_product = 1
            for item in old_shape:
                old_product *= item if isinstance(item, int) else 1
            new_product = 1
            for item in new_shape:
                new_product *= item if isinstance(item, int) else 1
            if old_product != new_product:
                raise TypeDescConflictError("reshape changes element count", path=("shape",))
        return replace(td, shape=new_shape, dims=None)
    if isinstance(value, (tuple, list)):
        new_dims = tuple(tuple(entry) if isinstance(entry, list) else entry for entry in value)
        old_shape = td.shape or tuple(size for _, size, _ in (td.dims or ()))
        new_shape = tuple(entry[1] for entry in new_dims)
        if (
            old_shape
            and all(isinstance(item, int) for item in old_shape)
            and all(
                isinstance(item, (tuple, list)) and len(item) >= 2 and isinstance(item[1], int)
                for item in new_dims
            )
        ):
            old_product = 1
            for item in old_shape:
                old_product *= item if isinstance(item, int) else 1
            new_product = 1
            for item in new_shape:
                new_product *= item if isinstance(item, int) else 1
            if old_product != new_product:
                raise TypeDescConflictError("reshape changes element count", path=("dims",))
        return replace(td, dims=new_dims, shape=None)
    raise OperationBindingError("reshape requires a shape or dimensions sequence")


def _add_dim_type_desc(td: TypeDesc, name: str, size: DimValue, position: int | None) -> TypeDesc:
    dims = list(td.dims or ())
    if any(existing == name for existing, _, _ in dims):
        raise TypeDescConflictError(f"dimension {name!r} already exists", path=("dims", name))
    entry = (name, size, None)
    index = len(dims) if position is None else position
    if index < 0 or index > len(dims):
        raise OperationBindingError("dimension position is out of range")
    dims.insert(index, entry)
    return replace(td, dims=tuple(dims))


def _remove_dim_type_desc(td: TypeDesc, name: str) -> TypeDesc:
    dims = list(td.dims or ())
    if not any(existing == name for existing, _, _ in dims):
        raise TypeDescUnknownError(f"unknown dimension {name!r}", path=("dims", name))
    return replace(td, dims=tuple(entry for entry in dims if entry[0] != name))


def _rename_axis_type_desc(td: TypeDesc, old: str, new: str) -> TypeDesc:
    dims = list(td.dims or ())
    if not any(name == old for name, _, _ in dims):
        raise TypeDescUnknownError(f"unknown dimension {old!r}", path=("dims", old))
    if any(name == new for name, _, _ in dims):
        raise TypeDescConflictError(f"dimension {new!r} already exists", path=("dims", new))
    return replace(
        td, dims=tuple((new if name == old else name, size, labels) for name, size, labels in dims)
    )


def _build_structural_operations() -> dict[tuple[str, str], Any]:
    kinds = (
        "scalar",
        "numpy.ndarray",
        "xarray.DataArray",
        "xarray.Dataset",
        "pandas.Series",
        "pandas.DataFrame",
        "polars.Series",
        "polars.DataFrame",
        "pyarrow.Array",
        "pyarrow.Table",
        "drjit.Array",
        "record",
        "opaque",
    )
    operations: dict[tuple[str, str], Any] = {}
    for kind in kinds:
        operations.update(
            {
                (kind, name): (lambda td, args, kwargs, op=name: _binary_impl(td, args[0], op))
                for name in _BINARY_OPERATIONS
            }
        )
        operations.update(
            {
                (kind, name): (lambda td, args, kwargs, op=name: _unary_impl(td, op))
                for name in _UNARY_OPERATIONS
            }
        )
        operations.update(
            {
                (kind, name): (lambda td, args, kwargs, op=name: _method_impl(td, op, args, kwargs))
                for name in _METHOD_OPERATIONS
            }
        )
        if kind in {"numpy.ndarray", "drjit.Array"}:
            operations.update(
                {
                    (kind, name): (
                        lambda td, args, kwargs, op=name: _method_impl(td, op, args, kwargs)
                    )
                    for name in ("matmul", "outer", "stack")
                }
            )
    return operations


STRUCTURAL_OPERATIONS: Mapping[tuple[str, str], Any] = _build_structural_operations()
