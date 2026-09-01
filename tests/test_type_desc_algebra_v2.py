"""Contract tests for the TypeDesc v2 pure structural algebra."""

from __future__ import annotations

import pytest

from typetrace import Symbol, TypeDesc
from typetrace.errors import (
    OperationBindingError,
    TypeDescConflictError,
    TypeDescUnknownError,
    TypeDescValidationError,
    UnsupportedOperationError,
)


def array(**overrides: object) -> TypeDesc:
    values: dict[str, object] = {
        "kind": "numpy.ndarray",
        "dims": (("x", 2, ("a", "b")),),
        "dtype": "float32",
        "metadata": (("unit", "USD"),),
    }
    values.update(overrides)
    return TypeDesc(**values)


@pytest.mark.parametrize(
    "operation", ["add", "sub", "mul", "div", "eq", "ne", "lt", "le", "gt", "ge"]
)
def test_binary_matrix_preserves_structure_and_sets_result_dtype(operation: str) -> None:
    result = array().binary(array(), operation)
    assert result.kind == "numpy.ndarray"
    assert result.dims == (("x", 2, ("a", "b")),)
    assert result.metadata == (("unit", "USD"),)
    assert result.dtype == (
        "bool"
        if operation in {"eq", "ne", "lt", "le", "gt", "ge"}
        else "float64"
        if operation == "div"
        else "float32"
    )


@pytest.mark.parametrize("operation", ["neg", "pos", "invert", "abs"])
def test_unary_matrix_preserves_all_unrelated_payloads(operation: str) -> None:
    result = array().unary(operation)
    assert result.kind == "numpy.ndarray"
    assert result.dims == array().dims
    assert result.metadata == array().metadata


@pytest.mark.parametrize(
    "kind",
    [
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
    ],
)
def test_method_matrix_has_complete_results(kind: str) -> None:
    if kind == "scalar":
        td = TypeDesc(kind=kind, dtype="float64")
    elif kind == "opaque":
        td = TypeDesc(kind=kind)
    elif kind == "record":
        td = TypeDesc(kind=kind, fields=(("price", TypeDesc(kind="scalar", dtype="float64")),))
    elif kind == "xarray.Dataset":
        td = TypeDesc(
            kind=kind,
            fields=(
                (
                    "price",
                    TypeDesc(kind="xarray.DataArray", dims=(("x", 2, None),), dtype="float64"),
                ),
            ),
        )
    elif kind.endswith("DataFrame") or kind.endswith("Table"):
        td = TypeDesc(kind=kind, columns=("price",), dtypes=(("price", "float64"),))
    elif kind.endswith("Series") or kind.endswith("Array"):
        td = TypeDesc(
            kind=kind,
            dtype="float64",
            dims=(("x", 2, None),) if kind == "xarray.DataArray" else None,
        )
    else:
        td = TypeDesc(kind=kind, dtype="float64")
    assert td.method("sum").kind == kind
    if td.dtype is not None:
        assert td.method("astype", args=("float64",)).dtype == "float64"


def test_broadcast_and_unify_have_distinct_dimension_rules() -> None:
    left = array(dims=(("x", 2, ("a", "b")),))
    right = array(dims=(("y", 3, None),))
    assert left.broadcast(right).dims == (("x", 2, ("a", "b")), ("y", 3, None))
    with pytest.raises(TypeDescConflictError):
        left.unify(right)


def test_known_and_unknown_labels_are_not_silently_merged() -> None:
    with pytest.raises(TypeDescUnknownError):
        array().binary(array(dims=(("x", 2, None),)), "add")


def test_conflicting_metadata_and_nominal_kind_raise() -> None:
    with pytest.raises(TypeDescConflictError):
        array().binary(array(metadata=(("unit", "EUR"),)), "add")
    with pytest.raises(TypeDescConflictError):
        array().binary(
            TypeDesc(
                kind="xarray.DataArray",
                dims=array().dims,
                dtype="float32",
                metadata=array().metadata,
            ),
            "add",
        )


def test_projection_selection_reduction_and_axis_verbs() -> None:
    record = TypeDesc(
        kind="record",
        fields=(
            ("price", TypeDesc(kind="scalar", dtype="float64")),
            ("volume", TypeDesc(kind="scalar", dtype="int64")),
        ),
    )
    assert record.project("price").dtype == "float64"
    assert tuple(name for name, _ in record.select(("volume",)).fields or ()) == ("volume",)
    td = array(dims=(("x", 2, ("a", "b")), ("y", 3, None)))
    assert td.reduce(("x",), "sum").dims == (("y", 3, None),)
    assert td.rename_axis("x", "time").dims[0][0] == "time"
    assert td.remove_dim("x").dims == (("y", 3, None),)
    assert td.add_dim("z", 4).dims[-1] == ("z", 4, None)


def test_symbolic_bind_and_invalid_operations() -> None:
    td = TypeDesc(kind="numpy.ndarray", dims=(("x", Symbol("N"), None),), dtype="float64")
    assert td.bind({"N": 4}).dims == (("x", 4, None),)
    with pytest.raises(UnsupportedOperationError):
        td.unary("unknown")
    with pytest.raises(TypeDescConflictError):
        array().reshape((3,))
    with pytest.raises(OperationBindingError):
        td.method("sum", args=("x", "y"))
    dataset = TypeDesc(
        kind="xarray.Dataset",
        fields=(("price", TypeDesc(kind="scalar", dtype="float64")),),
    )
    with pytest.raises(UnsupportedOperationError):
        dataset.select(("price",))
    with pytest.raises(UnsupportedOperationError):
        td.reduce(("x",), "astype")


@pytest.mark.parametrize(
    "td",
    [
        TypeDesc(kind="opaque"),
        TypeDesc(
            kind="xarray.Dataset",
            fields=(
                (
                    "price",
                    TypeDesc(kind="xarray.DataArray", dims=(("x", 2, None),), dtype="float64"),
                ),
            ),
        ),
    ],
)
def test_full_operation_matrix_covers_dataset_and_opaque(td: TypeDesc) -> None:
    for operation in ["add", "sub", "mul", "div", "eq", "ne", "lt", "le", "gt", "ge"]:
        if td.dtype is None:
            with pytest.raises(UnsupportedOperationError):
                td.binary(td, operation)
        else:
            assert td.binary(td, operation).kind == td.kind
    for operation in ["neg", "pos", "invert", "abs"]:
        assert td.unary(operation).kind == td.kind
    for operation in ["sum", "mean", "min", "max", "count"]:
        if operation == "count" and td.dtype is None:
            with pytest.raises(UnsupportedOperationError):
                td.method(operation)
        else:
            assert td.method(operation).kind == td.kind
    with pytest.raises(UnsupportedOperationError):
        td.method("astype", args=("float64",))


def test_matrix_methods_preserve_nominal_kind_dtype_and_metadata() -> None:
    left = TypeDesc("numpy.ndarray", shape=(2, 3), dtype="float32", metadata=(("unit", "USD"),))
    right = TypeDesc("numpy.ndarray", shape=(3, 4), dtype="float64")
    assert left.method("matmul", args=(right,), kwargs={}) == TypeDesc(
        "numpy.ndarray", shape=(2, 4), dtype="float64", metadata=(("unit", "USD"),)
    )


def test_outer_and_stack_methods_have_complete_shapes() -> None:
    vector = TypeDesc("numpy.ndarray", shape=(2,), dtype="float64")
    other = TypeDesc("numpy.ndarray", shape=(3,), dtype="float64")
    assert vector.method("outer", args=(other,), kwargs={}).shape == (2, 3)
    matrix = TypeDesc("numpy.ndarray", shape=(2, 3), dtype="float64")
    assert matrix.method("stack", args=((matrix,),), kwargs={"axis": 0}).shape == (2, 2, 3)


def test_named_axes_survive_matrix_methods() -> None:
    left = TypeDesc(
        "numpy.ndarray", dims=(("row", 2, ("a", "b")), ("inner", 3, None)), dtype="float64"
    )
    right = TypeDesc(
        "numpy.ndarray", dims=(("inner", 3, None), ("column", 4, None)), dtype="float64"
    )
    assert left.method("matmul", args=(right,), kwargs={}).dims == (
        ("row", 2, ("a", "b")),
        ("column", 4, None),
    )
    first = TypeDesc("numpy.ndarray", dims=(("left", 2, None),), dtype="float64")
    second = TypeDesc("numpy.ndarray", dims=(("right", 3, None),), dtype="float64")
    assert first.method("outer", args=(second,), kwargs={}).dims == (
        ("left", 2, None),
        ("right", 3, None),
    )


@pytest.mark.parametrize(
    ("name", "args", "kwargs", "message"),
    [
        ("matmul", (TypeDesc("numpy.ndarray", shape=(4, 5), dtype="float64"),), {}, "matmul:"),
        ("outer", (TypeDesc("numpy.ndarray", shape=(2, 3), dtype="float64"),), {}, "outer:"),
        (
            "stack",
            ((TypeDesc("numpy.ndarray", shape=(2, 3), dtype="float64"),),),
            {"axis": 4},
            "stack:",
        ),
    ],
)
def test_matrix_methods_reject_invalid_shapes_and_axes(
    name: str, args: tuple[object, ...], kwargs: dict[str, object], message: str
) -> None:
    value = TypeDesc("numpy.ndarray", shape=(2, 3), dtype="float64")
    with pytest.raises(TypeDescValidationError, match=message):
        value.method(name, args=args, kwargs=kwargs)


def test_matrix_methods_are_nominal_and_unregistered_merge_is_rejected() -> None:
    value = TypeDesc("numpy.ndarray", shape=(2,), dtype="float64")
    other = TypeDesc("xarray.DataArray", shape=(2,), dtype="float64")
    with pytest.raises(TypeDescConflictError):
        value.method("outer", args=(other,), kwargs={})
    with pytest.raises(UnsupportedOperationError, match="merge"):
        value.method("merge", args=(value,), kwargs={"how": "inner"})
