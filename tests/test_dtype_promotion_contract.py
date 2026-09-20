"""Red contract tests for issue #29 numeric dtype promotion."""

from __future__ import annotations

import pytest

from typetrace import TypeDesc
from typetrace.core import TypeDescUnknownError, UnsupportedOperationError
from typetrace.patterns import binary_result_dtype, promote_dtype, unary_result_dtype

SIGNED_UNSIGNED_MATRIX = [
    ("int8", "uint8", "int16"),
    ("int8", "uint16", "int32"),
    ("int8", "uint32", "int64"),
    ("int8", "uint64", "float64"),
    ("int16", "uint8", "int16"),
    ("int16", "uint16", "int32"),
    ("int16", "uint32", "int64"),
    ("int16", "uint64", "float64"),
    ("int32", "uint8", "int32"),
    ("int32", "uint16", "int32"),
    ("int32", "uint32", "int64"),
    ("int32", "uint64", "float64"),
    ("int64", "uint8", "int64"),
    ("int64", "uint16", "int64"),
    ("int64", "uint32", "int64"),
    ("int64", "uint64", "float64"),
]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("int64", "float32", "float64"),
        ("float32", "int64", "float64"),
        *SIGNED_UNSIGNED_MATRIX,
        ("float64", "complex64", "complex128"),
    ],
)
def test_promote_dtype_complete_lattice(left: str, right: str, expected: str) -> None:
    assert promote_dtype(left, right) == expected
    assert promote_dtype(right, left) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("float16", "complex64", "complex64"),
        ("float32", "complex64", "complex64"),
        ("float64", "complex64", "complex128"),
        ("complex64", "float32", "complex64"),
        ("complex64", "complex128", "complex128"),
        ("complex128", "float64", "complex128"),
    ],
)
def test_promote_dtype_cross_family_matrix(left: str, right: str, expected: str) -> None:
    assert promote_dtype(left, right) == expected
    assert promote_dtype(right, left) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("bool", "int8", "int8"),
        ("bool", "int16", "int16"),
        ("bool", "int32", "int32"),
        ("bool", "int64", "int64"),
        ("bool", "uint8", "uint8"),
        ("bool", "uint16", "uint16"),
        ("bool", "uint32", "uint32"),
        ("bool", "uint64", "uint64"),
        ("bool", "float16", "float16"),
        ("bool", "float32", "float32"),
        ("bool", "float64", "float64"),
        ("bool", "complex64", "complex64"),
        ("bool", "complex128", "complex128"),
    ],
)
def test_promote_dtype_bool_with_every_numeric_family(left: str, right: str, expected: str) -> None:
    assert promote_dtype(left, right) == expected
    assert promote_dtype(right, left) == expected


@pytest.mark.parametrize(
    ("integer", "complex", "expected"),
    [
        ("int8", "complex64", "complex64"),
        ("int16", "complex64", "complex64"),
        ("int32", "complex64", "complex64"),
        ("int64", "complex64", "complex128"),
        ("uint8", "complex64", "complex64"),
        ("uint16", "complex64", "complex64"),
        ("uint32", "complex64", "complex64"),
        ("uint64", "complex64", "complex128"),
        ("int8", "complex128", "complex128"),
        ("int16", "complex128", "complex128"),
        ("int32", "complex128", "complex128"),
        ("int64", "complex128", "complex128"),
        ("uint8", "complex128", "complex128"),
        ("uint16", "complex128", "complex128"),
        ("uint32", "complex128", "complex128"),
        ("uint64", "complex128", "complex128"),
    ],
)
def test_promote_dtype_integer_complex_matrix(integer: str, complex: str, expected: str) -> None:
    assert promote_dtype(integer, complex) == expected
    assert promote_dtype(complex, integer) == expected


@pytest.mark.parametrize(
    ("integer", "floating"),
    [
        (integer, floating)
        for integer in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
        for floating in ("float16", "float32", "float64")
    ],
)
def test_promote_dtype_every_integer_float_pair(integer: str, floating: str) -> None:
    assert promote_dtype(integer, floating) == "float64"
    assert promote_dtype(floating, integer) == "float64"


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("int8", "int16", "int16"),
        ("int8", "int32", "int32"),
        ("int8", "int64", "int64"),
        ("int16", "int32", "int32"),
        ("int16", "int64", "int64"),
        ("int32", "int64", "int64"),
        ("uint8", "uint16", "uint16"),
        ("uint8", "uint32", "uint32"),
        ("uint8", "uint64", "uint64"),
        ("uint16", "uint32", "uint32"),
        ("uint16", "uint64", "uint64"),
        ("uint32", "uint64", "uint64"),
        ("float16", "float32", "float32"),
        ("float16", "float64", "float64"),
        ("float32", "float64", "float64"),
        ("complex64", "complex128", "complex128"),
    ],
)
def test_promote_dtype_same_family_widening(left: str, right: str, expected: str) -> None:
    assert promote_dtype(left, right) == expected
    assert promote_dtype(right, left) == expected


@pytest.mark.parametrize("floating", ["float16", "float32", "float64"])
def test_promote_dtype_every_float_complex128_pair(floating: str) -> None:
    assert promote_dtype(floating, "complex128") == "complex128"
    assert promote_dtype("complex128", floating) == "complex128"


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_promote_dtype_diagonal_is_canonical(dtype: str) -> None:
    assert promote_dtype(dtype, dtype) == dtype


@pytest.mark.parametrize(
    ("left", "right"),
    [("int64", "float32"), ("int8", "uint16"), ("float64", "complex64")],
)
def test_promote_dtype_is_symmetric(left: str, right: str) -> None:
    assert promote_dtype(left, right) == promote_dtype(right, left)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("boolean", "Int8", "int8"),
        ("Boolean", "Int64", "int64"),
        ("UInt16", "Int32", "int32"),
        ("Float16", "Int8", "float64"),
        ("Float32", "Int64", "float64"),
        ("double", "Complex64", "complex128"),
        ("Complex128", "Float64", "complex128"),
    ],
)
def test_promote_dtype_normalizes_aliases(left: str, right: str, expected: str) -> None:
    assert promote_dtype(left, right) == expected


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("boolean", "bool"),
        ("Boolean", "bool"),
        ("Int8", "int8"),
        ("Int16", "int16"),
        ("Int32", "int32"),
        ("Int64", "int64"),
        ("UInt8", "uint8"),
        ("UInt16", "uint16"),
        ("UInt32", "uint32"),
        ("UInt64", "uint64"),
        ("Float16", "float16"),
        ("Float32", "float32"),
        ("Float64", "float64"),
        ("double", "float64"),
        ("Complex64", "complex64"),
        ("Complex128", "complex128"),
    ],
)
def test_promote_dtype_normalizes_every_declared_alias(alias: str, canonical: str) -> None:
    assert promote_dtype(alias, None) == canonical


def test_declared_alias_matrix_has_unique_rows() -> None:
    aliases = (
        "boolean",
        "Boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float16",
        "Float32",
        "Float64",
        "double",
        "Complex64",
        "Complex128",
    )
    assert len(aliases) == len(set(aliases)) == 16


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(None, None, None), (None, "Float64", "float64"), ("Int64", None, "int64")],
)
def test_promote_dtype_handles_none(
    left: str | None, right: str | None, expected: str | None
) -> None:
    assert promote_dtype(left, right) == expected


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_promote_dtype_none_matrix_covers_every_canonical_dtype(dtype: str) -> None:
    assert promote_dtype(None, dtype) == dtype
    assert promote_dtype(dtype, None) == dtype


@pytest.mark.parametrize("dtype", ["unknown_dtype", ""])
def test_promote_dtype_rejects_unknown_and_empty(dtype: str) -> None:
    with pytest.raises(TypeDescUnknownError) as left_error:
        promote_dtype(dtype, "float64")
    assert left_error.value.path == ("dtype1",)
    with pytest.raises(TypeDescUnknownError) as right_error:
        promote_dtype("float64", dtype)
    assert right_error.value.path == ("dtype2",)


@pytest.mark.parametrize("operation", ["add", "eq", "div", "floordiv"])
def test_binary_result_dtype_unknown_paths_are_stable(operation: str) -> None:
    with pytest.raises(TypeDescUnknownError) as left_error:
        binary_result_dtype("unknown_dtype", "float32", operation)
    assert left_error.value.path == ("dtype1",)
    with pytest.raises(TypeDescUnknownError) as right_error:
        binary_result_dtype("float32", "unknown_dtype", operation)
    assert right_error.value.path == ("dtype2",)


@pytest.mark.parametrize("operation", ["neg", "abs", "not", "sign"])
def test_unary_result_dtype_unknown_path_is_stable(operation: str) -> None:
    with pytest.raises(TypeDescUnknownError) as error:
        unary_result_dtype("unknown_dtype", operation)
    assert error.value.path == ("input_dtype",)


@pytest.mark.parametrize(
    ("left", "right", "operation", "expected"),
    [("Float32", "Int64", "eq", "bool"), ("Int64", "Int32", "truediv", "float64")],
)
def test_binary_result_dtype_normalizes_before_comparison_and_division(
    left: str, right: str, operation: str, expected: str
) -> None:
    assert binary_result_dtype(left, right, operation) == expected


@pytest.mark.parametrize(
    ("left", "right", "operation", "expected"),
    [
        (None, None, "add", None),
        (None, "Float64", "add", "float64"),
        ("Int64", None, "mod", "int64"),
        (None, None, "eq", "bool"),
        ("unknown_dtype", "float64", "eq", "raises"),
        ("Float32", "unknown_dtype", "div", "raises"),
        (None, None, "div", "float64"),
        (None, "Float32", "truediv", "float64"),
        ("Int64", None, "/", "float64"),
        (None, None, "floordiv", "int64"),
        ("Int64", None, "//", "int64"),
    ],
)
def test_binary_result_dtype_has_total_none_and_alias_matrix(
    left: str | None,
    right: str | None,
    operation: str,
    expected: str | None,
) -> None:
    if expected == "raises":
        with pytest.raises(TypeDescUnknownError):
            binary_result_dtype(left, right, operation)
    else:
        assert binary_result_dtype(left, right, operation) == expected


CONSTANT_BINARY_OPERATIONS = [
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
    "div",
    "truediv",
    "/",
    "floordiv",
    "//",
]


@pytest.mark.parametrize(
    ("operation", "bad_left", "bad_right"),
    [
        (operation, bad_left, bad_right)
        for operation in CONSTANT_BINARY_OPERATIONS
        for bad_left, bad_right in [("unknown_dtype", "float32"), ("float32", "")]
    ],
)
def test_binary_result_dtype_constant_operations_validate_both_operands(
    operation: str, bad_left: str, bad_right: str
) -> None:
    for bad in ("unknown_dtype", ""):
        with pytest.raises(TypeDescUnknownError) as left_error:
            binary_result_dtype(bad, "float32", operation)
        assert left_error.value.path == ("dtype1",)
        with pytest.raises(TypeDescUnknownError) as right_error:
            binary_result_dtype("float32", bad, operation)
        assert right_error.value.path == ("dtype2",)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("add", "float64"),
        ("sub", "float64"),
        ("mul", "float64"),
        ("mod", "float64"),
        ("pow", "float64"),
        ("eq", "bool"),
        ("ne", "bool"),
        ("lt", "bool"),
        ("le", "bool"),
        ("gt", "bool"),
        ("ge", "bool"),
        ("==", "bool"),
        ("!=", "bool"),
        ("<", "bool"),
        ("<=", "bool"),
        (">", "bool"),
        (">=", "bool"),
        ("div", "float64"),
        ("truediv", "float64"),
        ("/", "float64"),
        ("floordiv", "int64"),
        ("//", "int64"),
    ],
)
def test_binary_result_dtype_operation_matrix(operation: str, expected: str) -> None:
    assert binary_result_dtype("Int64", "Float32", operation) == expected


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "mod", "pow"])
def test_binary_result_dtype_none_generic_operations(operation: str) -> None:
    assert binary_result_dtype(None, None, operation) is None
    assert binary_result_dtype(None, "Float32", operation) == "float32"
    assert binary_result_dtype("Int32", None, operation) == "int32"


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "mod", "pow"])
@pytest.mark.parametrize("bad_left, bad_right", [("unknown_dtype", "float32"), ("float32", "")])
def test_binary_result_dtype_generic_operations_validate_both_operands(
    operation: str, bad_left: str, bad_right: str
) -> None:
    for bad in ("unknown_dtype", ""):
        with pytest.raises(TypeDescUnknownError) as left_error:
            binary_result_dtype(bad, "float32", operation)
        assert left_error.value.path == ("dtype1",)
        with pytest.raises(TypeDescUnknownError) as right_error:
            binary_result_dtype("float32", bad, operation)
        assert right_error.value.path == ("dtype2",)


@pytest.mark.parametrize(
    "operation",
    [
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
    ],
)
def test_binary_result_dtype_none_comparisons(operation: str) -> None:
    assert binary_result_dtype(None, None, operation) == "bool"
    assert binary_result_dtype(None, "Float32", operation) == "bool"
    assert binary_result_dtype("Float32", None, operation) == "bool"


@pytest.mark.parametrize("operation", ["div", "truediv", "/"])
def test_binary_result_dtype_none_true_division(operation: str) -> None:
    assert binary_result_dtype(None, None, operation) == "float64"
    assert binary_result_dtype(None, "Float32", operation) == "float64"
    assert binary_result_dtype("Float32", None, operation) == "float64"


@pytest.mark.parametrize("operation", ["floordiv", "//"])
def test_binary_result_dtype_none_floor_division(operation: str) -> None:
    assert binary_result_dtype(None, None, operation) == "int64"
    assert binary_result_dtype(None, "Float32", operation) == "int64"
    assert binary_result_dtype("Float32", None, operation) == "int64"


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("add", "float64"), ("eq", "bool"), ("div", "float64"), ("mod", "float64")],
)
def test_binary_result_dtype_aliases_normalize_before_every_result_path(
    operation: str, expected: str
) -> None:
    assert binary_result_dtype("Float32", "Int64", operation) == expected


@pytest.mark.parametrize(
    "operation",
    [
        "add",
        "sub",
        "mul",
        "div",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "mod",
        "pow",
        "floordiv",
        "//",
        "truediv",
        "/",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
    ],
)
def test_binary_result_dtype_accepts_existing_helper_operations(operation: str) -> None:
    assert binary_result_dtype("Int32", "Float32", operation) is not None


@pytest.mark.parametrize("operation", ["invalid", "", "__add__"])
def test_binary_result_dtype_rejects_invalid_operations(operation: str) -> None:
    with pytest.raises(UnsupportedOperationError):
        binary_result_dtype("int32", "int32", operation)


@pytest.mark.parametrize(
    ("dtype", "operation", "expected"),
    [
        (None, "neg", None),
        (None, "abs", None),
        ("Complex64", "abs", "float32"),
        ("Complex128", "real", "float64"),
        ("Float32", "neg", "float32"),
        ("Int64", "not", "bool"),
        ("Float64", "sign", "int64"),
    ],
)
def test_unary_result_dtype_normalizes_before_conversion(
    dtype: str | None, operation: str, expected: str | None
) -> None:
    assert unary_result_dtype(dtype, operation) == expected


@pytest.mark.parametrize("operation", ["invalid", ""])
def test_unary_result_dtype_rejects_invalid_operations(operation: str) -> None:
    with pytest.raises(UnsupportedOperationError):
        unary_result_dtype("float32", operation)


@pytest.mark.parametrize("operation", ["neg", "pos", "invert", "abs"])
def test_unary_result_dtype_none_passthrough(operation: str) -> None:
    assert unary_result_dtype(None, operation) is None


@pytest.mark.parametrize("operation", ["real", "imag", "exp", "log", "sqrt"])
def test_unary_result_dtype_none_preserves_unknown_value(operation: str) -> None:
    assert unary_result_dtype(None, operation) is None


@pytest.mark.parametrize("operation", ["not", "isnan", "isinf", "isfinite"])
def test_unary_result_dtype_none_predicates(operation: str) -> None:
    assert unary_result_dtype(None, operation) == "bool"


def test_unary_result_dtype_none_sign() -> None:
    assert unary_result_dtype(None, "sign") == "int64"


@pytest.mark.parametrize(
    "operation",
    [
        "neg",
        "pos",
        "invert",
        "abs",
        "real",
        "imag",
        "exp",
        "log",
        "sqrt",
    ],
)
def test_unary_result_dtype_aliases_normalize_before_preserving(operation: str) -> None:
    assert unary_result_dtype("Float32", operation) == "float32"


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("not", "bool"),
        ("isnan", "bool"),
        ("isinf", "bool"),
        ("isfinite", "bool"),
        ("sign", "int64"),
    ],
)
def test_unary_result_dtype_aliases_normalize_before_constant_results(
    operation: str, expected: str
) -> None:
    assert unary_result_dtype("Float32", operation) == expected


@pytest.mark.parametrize("dtype", ["unknown_dtype", ""])
def test_unary_result_dtype_rejects_unknown_and_empty(dtype: str) -> None:
    for operation in (
        "neg",
        "pos",
        "invert",
        "abs",
        "real",
        "imag",
        "exp",
        "log",
        "sqrt",
        "not",
        "isnan",
        "isinf",
        "isfinite",
        "sign",
    ):
        with pytest.raises(TypeDescUnknownError) as error:
            unary_result_dtype(dtype, operation)
        assert error.value.path == ("input_dtype",)


@pytest.mark.parametrize(
    "operation",
    [
        "neg",
        "pos",
        "invert",
        "abs",
        "real",
        "imag",
        "not",
        "isnan",
        "isinf",
        "isfinite",
        "sign",
        "exp",
        "log",
        "sqrt",
    ],
)
def test_unary_result_dtype_accepts_existing_helper_operations(operation: str) -> None:
    assert unary_result_dtype("Float32", operation) is not None


def test_typedesc_binary_uses_canonical_promotion() -> None:
    left = TypeDesc(kind="numpy.ndarray", dtype="int64")
    right = TypeDesc(kind="numpy.ndarray", dtype="float32")

    assert left.binary(right, "add").dtype == "float64"


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("add", "float64"),
        ("sub", "float64"),
        ("mul", "float64"),
        ("div", "float64"),
        ("eq", "bool"),
        ("ne", "bool"),
        ("lt", "bool"),
        ("le", "bool"),
        ("gt", "bool"),
        ("ge", "bool"),
    ],
)
def test_typedesc_binary_operation_parity(operation: str, expected: str) -> None:
    left = TypeDesc(kind="numpy.ndarray", dtype="int64")
    right = TypeDesc(kind="numpy.ndarray", dtype="float32")
    assert left.binary(right, operation).dtype == expected


@pytest.mark.parametrize(
    "operation",
    [
        "truediv",
        "/",
        "//",
        "floordiv",
        "mod",
        "pow",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
    ],
)
def test_typedesc_binary_rejects_helper_only_aliases(operation: str) -> None:
    left = TypeDesc(kind="numpy.ndarray", dtype="int64")
    right = TypeDesc(kind="numpy.ndarray", dtype="float32")
    with pytest.raises(UnsupportedOperationError):
        left.binary(right, operation)


@pytest.mark.parametrize(
    ("dtype", "operation", "expected"),
    [
        ("bool", "neg", "bool"),
        ("int64", "neg", "int64"),
        ("float32", "pos", "float32"),
        ("int32", "invert", "int32"),
        ("float64", "abs", "float64"),
        ("complex64", "abs", "float32"),
        ("complex128", "abs", "float64"),
    ],
)
def test_typedesc_unary_canonical_operations_preserve_contract(
    dtype: str, operation: str, expected: str
) -> None:
    value = TypeDesc(kind="numpy.ndarray", dtype=dtype)
    assert value.unary(operation).dtype == expected


@pytest.mark.parametrize(
    "operation",
    [
        "real",
        "imag",
        "not",
        "isnan",
        "isinf",
        "isfinite",
        "sign",
        "exp",
        "log",
        "sqrt",
    ],
)
def test_typedesc_unary_rejects_helper_only_operations(operation: str) -> None:
    value = TypeDesc(kind="numpy.ndarray", dtype="float32")
    with pytest.raises(UnsupportedOperationError):
        value.unary(operation)
