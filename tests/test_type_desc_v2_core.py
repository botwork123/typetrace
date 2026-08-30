"""Red tests for the immutable TypeDesc v2 core contract."""

from dataclasses import FrozenInstanceError, fields

import pytest

from typetrace.core import (
    Symbol,
    TypeDesc,
    TypeDescConflictError,
    TypeDescValidationError,
)


def test_constructor_has_exact_frozen_schema() -> None:
    assert [field.name for field in fields(TypeDesc)] == [
        "kind",
        "dims",
        "shape",
        "dtype",
        "dtypes",
        "index",
        "columns",
        "fields",
        "drjit_type",
        "static_dims",
        "metadata",
    ]
    assert TypeDesc("scalar", dtype="float64").metadata == ()


@pytest.mark.parametrize(
    "descriptor",
    [
        TypeDesc("numpy.ndarray", dims=(("x", 2, ("a", "b")),), dtype="float64"),
        TypeDesc(
            "pandas.DataFrame", index=(("row", 2, None),), columns=("a",), dtypes=(("a", "int64"),)
        ),
        TypeDesc("record", fields=(("value", TypeDesc("scalar", dtype="int64")),)),
        TypeDesc("opaque", metadata=(("source", ("test", 1)),)),
    ],
)
def test_nested_values_are_immutable_and_hashable(descriptor: TypeDesc) -> None:
    assert hash(descriptor)
    with pytest.raises((TypeError, FrozenInstanceError)):
        descriptor.metadata += (("new", 1),)


def test_mutable_inputs_are_frozen_recursively() -> None:
    dims = [["x", 2, ["a", "b"]]]
    metadata = {"payload": [1, {"nested": 2}]}
    descriptor = TypeDesc("numpy.ndarray", dims=dims, metadata=metadata)
    dims[0][2].append("c")
    metadata["payload"].append(3)
    assert descriptor.dims == (("x", 2, ("a", "b")),)
    assert descriptor.metadata == (("payload", (1, (("nested", 2),))),)


def test_kind_is_nominal_and_canonical_identity_is_tagged() -> None:
    numpy = TypeDesc("numpy.ndarray", shape=(2,), dtype="float64")
    xarray = TypeDesc("xarray.DataArray", shape=(2,), dtype="float64")
    assert numpy != xarray
    assert hash(numpy) != hash(xarray)
    assert numpy.fingerprint() != xarray.fingerprint()
    assert numpy == TypeDesc("numpy.ndarray", shape=(2,), dtype="float64")


def test_metadata_order_is_canonical_but_structural_order_is_semantic() -> None:
    left = TypeDesc(
        "record", fields=(("a", TypeDesc("scalar", dtype="int64")),), metadata=(("z", 1), ("a", 2))
    )
    right = TypeDesc(
        "record", fields=(("a", TypeDesc("scalar", dtype="int64")),), metadata=(("a", 2), ("z", 1))
    )
    assert left == right
    assert left.fingerprint() == right.fingerprint()
    assert TypeDesc(
        "record", fields=(("b", TypeDesc("scalar")), ("a", TypeDesc("scalar")))
    ) != TypeDesc("record", fields=(("a", TypeDesc("scalar")), ("b", TypeDesc("scalar"))))


def test_validation_rejects_bad_sizes_labels_duplicates_cycles_and_payloads() -> None:
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", shape=(-1,))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", dims=(("x", 2, ("a",)),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", dims=(("x", 1, None), ("x", 2, None)))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("scalar", shape=(1,))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("opaque", metadata=(("bad", {"unhashable"}),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("scalar", fields=(("value", TypeDesc("scalar")),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("opaque", dtype="float64")
    with pytest.raises(TypeDescValidationError):
        TypeDesc("record", dtype="float64")


def test_validation_rejects_cyclic_metadata() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(TypeDescValidationError):
        TypeDesc("opaque", metadata=(("cycle", cyclic),))


def test_bind_is_complete_and_rejects_unknown_or_conflicting_bindings() -> None:
    descriptor = TypeDesc(
        "example.record",
        dims=(("x", Symbol("N"), (Symbol("label"),)),),
        shape=(Symbol("N"),),
        index=(("row", Symbol("N"), None),),
        fields=(("nested", TypeDesc("numpy.ndarray", shape=(Symbol("N"),))),),
        metadata=(("symbol", Symbol("N")),),
    )
    bound = descriptor.bind({"N": 1, "label": "L"})
    assert bound.dims == (("x", 1, ("L",)),)
    assert bound.shape == (1,)
    assert bound.index == (("row", 1, None),)
    assert bound.fields[0][1].shape == (1,)
    assert bound.metadata == (("symbol", 1),)
    with pytest.raises(TypeDescValidationError):
        descriptor.bind({"UNKNOWN": 1})
    with pytest.raises(TypeDescConflictError):
        descriptor.bind({"N": -1})


def test_symbol_names_are_validated() -> None:
    with pytest.raises(TypeDescValidationError):
        Symbol("")
    with pytest.raises(TypeDescValidationError):
        Symbol("not valid")
