"""Red tests for the immutable TypeDesc v2 core contract."""

from dataclasses import FrozenInstanceError, fields

import pytest

from typetrace.core import (
    AdapterAmbiguityError,
    AdapterRegistrationError,
    AdapterUnavailableError,
    OperationBindingError,
    OperationExecutionError,
    ResultInferenceError,
    SampleMaterializationError,
    Symbol,
    TypeDesc,
    TypeDescConflictError,
    TypeDescValidationError,
)


@pytest.mark.parametrize(
    "error_type",
    [
        AdapterRegistrationError,
        AdapterUnavailableError,
        AdapterAmbiguityError,
        SampleMaterializationError,
        OperationBindingError,
        OperationExecutionError,
        ResultInferenceError,
    ],
)
def test_public_boundary_errors_carry_operation_and_path(error_type: type[Exception]) -> None:
    error = error_type("bad", operation="sample", path=("args", 0))
    assert error.operation == "sample"
    assert error.path == ("args", 0)
    assert "at args.0" in str(error)


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


def test_bind_discovers_symbols_only_in_nested_descriptors() -> None:
    descriptor = TypeDesc(
        "record",
        fields=(("nested", TypeDesc("numpy.ndarray", shape=(Symbol("N"),))),),
    )
    bound = descriptor.bind({"N": 4})
    assert bound.fields[0][1].shape == (4,)


def test_symbol_names_are_validated() -> None:
    with pytest.raises(TypeDescValidationError):
        Symbol("")
    with pytest.raises(TypeDescValidationError):
        Symbol("not valid")


def test_validation_and_identity_edges() -> None:
    """Exercise the complete constructor boundary, including negative paths."""
    assert TypeDesc(
        "scalar", dtype="float64", metadata=(("none", None), ("bytes", b"x"))
    ).fingerprint()
    assert TypeDesc("numpy.ndarray", shape=[1, Symbol("N")]).shape == (1, Symbol("N"))
    assert TypeDesc("numpy.ndarray", dims=(("x", 2),)).dims == (("x", 2, None),)
    assert TypeDesc("opaque", metadata={"b": 2, "a": [1, 2]}).metadata == (
        ("a", (1, 2)),
        ("b", 2),
    )
    for descriptor in (
        TypeDesc("numpy.ndarray"),
        TypeDesc("pandas.DataFrame"),
        TypeDesc("record", fields=(("x", TypeDesc("scalar")),)),
    ):
        assert descriptor.bind({}) == descriptor
    assert TypeDesc("pandas.DataFrame", columns=("a", ...)).known_columns() == ["a"]
    assert TypeDesc("pandas.DataFrame", columns=(1, ...)).known_columns() == [1]
    with pytest.raises(TypeDescValidationError):
        TypeDesc("")
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", shape="bad")
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", dims=(("x", 1, ("a", "b")),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", dims=(("x", True, None),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", dims=(("x", 1, None), ("x", 1, None)))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", columns=("x",))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("pandas.DataFrame", dtypes=(("a", 1),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("pandas.DataFrame", dtypes=(([], "int64"),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("pandas.DataFrame", dtypes=(("a", "int64"), ("a", "float64")))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("record", fields=(("x", TypeDesc("scalar")), ("x", TypeDesc("scalar"))))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("scalar", dtype=1)
    with pytest.raises(TypeDescValidationError):
        TypeDesc("drjit.Array", static_dims=(True,))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("drjit.Array", drjit_type=1)
    with pytest.raises(TypeDescValidationError):
        TypeDesc("opaque", metadata=((1, 2),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("opaque", metadata=(("x", {"unhashable"}),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("record", fields=(([], TypeDesc("scalar")),))
    assert TypeDesc("record", fields=(("x", TypeDesc("scalar")),)).field("x").kind == "scalar"
    with pytest.raises(ValueError):
        TypeDesc("scalar").field("x")
    with pytest.raises(KeyError):
        TypeDesc("record", fields=(("x", TypeDesc("scalar")),)).field("y")


def test_canonical_and_binding_edge_paths() -> None:
    from typetrace.core import _canonical

    descriptor = TypeDesc(
        "custom",
        metadata=(
            ("bool", True),
            ("nan", float("nan")),
            ("inf", float("inf")),
            ("mapping", {"x": 1}),
            ("type", int),
        ),
    )
    assert _canonical(descriptor)
    nested = TypeDesc("custom", metadata=(("nested", {"N": Symbol("N")}),))
    assert nested.bind({"N": 3}).metadata == (("nested", (("N", 3),)),)
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(TypeDescValidationError):
        TypeDesc("custom", metadata=(("cycle", cyclic),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", dims=1)
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", dims=(("x", 1, None, 2),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("numpy.ndarray", dims=(("x", 1, ([],)),))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("custom", dtypes=(("x", "a"), ("x", "b")))
    with pytest.raises(TypeDescValidationError):
        TypeDesc("custom", metadata=(("x", 1), ("x", 2)))
    assert TypeDesc("custom", metadata=None).metadata == ()
    assert TypeDesc("drjit.Array", static_dims=(1, 2)).static_dims == (1, 2)
    with pytest.raises(TypeDescValidationError):
        TypeDesc("drjit.Array", static_dims=1)
    with pytest.raises(TypeDescValidationError):
        TypeDesc("opaque", metadata=1)
    with pytest.raises(TypeDescValidationError):
        TypeDesc("opaque", metadata=[1])


def test_arbitrary_hashable_labels_are_structural() -> None:
    class Label:
        def __repr__(self) -> str:
            return "Label()"

        def __hash__(self) -> int:
            return 1

    label = Label()
    descriptor = TypeDesc("pandas.DataFrame", columns=(label,), dtypes=((label, "float64"),))
    assert descriptor.known_columns() == [label]
    assert descriptor.fingerprint() == descriptor.fingerprint()
    other_label = Label()
    other = TypeDesc("pandas.DataFrame", columns=(other_label,), dtypes=((other_label, "float64"),))
    assert descriptor != other
    assert descriptor.fingerprint() != other.fingerprint()
