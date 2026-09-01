"""Bounded cross-backend conformance matrix for TypeDesc v2."""

from __future__ import annotations

import numpy as np
import pytest

from typetrace import Symbol, TypeDesc, infer_by_execution, make_samples
from typetrace.errors import (
    OperationBindingError,
    TypeDescConflictError,
    TypeDescUnknownError,
    UnsupportedOperationError,
)


def pandas_pair():
    import pandas as pd

    return pd.DataFrame({"id": [1, 2], "price": [10.0, 20.0]}), pd.DataFrame(
        {"id": [2, 3], "qty": [7, 8]}
    )


def pandas_axis_pair():
    import pandas as pd

    return pd.DataFrame({"left": [1, 2]}), pd.DataFrame({"right": [3, 4]})


def xarray_pair():
    import xarray as xr

    return (
        xr.DataArray(np.array([10.0, 20.0]), dims=("id",), coords={"id": [1, 2]}),
        xr.DataArray(np.array([7.0, 8.0]), dims=("id",), coords={"id": [2, 3]}),
    )


def dataset_nonoverlap():
    import xarray as xr

    return (
        xr.Dataset({"price": ("id", [10.0, 20.0])}, coords={"id": [1, 2]}),
        xr.Dataset({"qty": ("id", [7, 8])}, coords={"id": [1, 2]}),
    )


def dataset_identical():
    import xarray as xr

    return (
        xr.Dataset({"price": ("id", [10.0, 20.0])}, coords={"id": [1, 2]}),
        xr.Dataset({"price": ("id", [10.0, 20.0])}, coords={"id": [1, 2]}),
    )


def dataset_conflict():
    import xarray as xr

    return (
        xr.Dataset({"price": ("id", [10.0, 20.0])}, coords={"id": [1, 2]}),
        xr.Dataset({"price": ("id", [11.0, 21.0])}, coords={"id": [1, 2]}),
    )


def dataset_equals():
    left, right = dataset_identical()
    right["price"].attrs["unit"] = "USD"
    return left, right


def polars_pair():
    import polars as pl

    return pl.DataFrame({"id": [1, 2], "price": [10.0, 20.0]}), pl.DataFrame(
        {"id": [2, 3], "qty": [7, 8]}
    )


def arrow_equal_pair():
    import pyarrow as pa

    return (
        pa.table({"id": pa.array([1, 2]), "value": pa.array([10.0, 20.0])}),
        pa.table({"id": pa.array([3]), "value": pa.array([30.0])}),
    )


def arrow_mismatch_pair():
    import pyarrow as pa

    return (
        pa.table({"id": pa.array([1], type=pa.int64())}),
        pa.table({"id": pa.array(["x"], type=pa.string())}),
    )


def xarray_empty():
    import xarray as xr

    return xr.DataArray(
        np.empty((0,), dtype="float64"), dims=("id",), coords={"id": np.array([], dtype="int64")}
    )


def xarray_dataset_empty():
    import xarray as xr

    return xr.Dataset(
        {"price": ("id", np.array([], dtype="float64"))}, coords={"id": np.array([], dtype="int64")}
    )


def arrow_empty():
    import pyarrow as pa

    return pa.table({"id": pa.array([], type=pa.int64()), "price": pa.array([], type=pa.float64())})


def numpy_empty():
    return np.empty((0, 3), dtype="float64")


def pandas_empty():
    import pandas as pd

    return pd.DataFrame(
        {"id": pd.Series([], dtype="int64"), "price": pd.Series([], dtype="float64")}
    )


def polars_empty():
    import polars as pl

    return pl.DataFrame(
        {"id": pl.Series([], dtype=pl.Int64), "price": pl.Series([], dtype=pl.Float64)}
    )


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        pytest.param(3, "scalar", id="scalar"),
        pytest.param(np.ones((2, 3), dtype="float64"), "numpy.ndarray", id="ndarray"),
    ],
)
def test_identity_and_round_trip(value, kind: str) -> None:
    desc = TypeDesc.from_value(value)
    assert desc.kind == kind
    assert TypeDesc.from_value(value) == desc
    assert hash(desc) == hash(TypeDesc.from_value(value))


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("pandas.Series", id="pandas-series"),
        pytest.param("polars.Series", id="polars-series"),
        pytest.param("pyarrow.Array", id="pyarrow-array"),
    ],
)
def test_series_and_array_forms(kind: str) -> None:
    if kind == "pandas.Series":
        import pandas as pd

        value = pd.Series([1, 2], dtype="int64")
    elif kind == "polars.Series":
        import polars as pl

        value = pl.Series("value", [1, 2], dtype=pl.Int64)
    else:
        import pyarrow as pa

        value = pa.array([1, 2], type=pa.int64())
    desc = TypeDesc.from_value(value)
    assert desc.kind == kind
    assert desc.dtype == ("Int64" if kind == "polars.Series" else "int64")


def test_drjit_form_when_llvm_is_available() -> None:
    import drjit

    assert drjit.has_backend(drjit.JitBackend.LLVM), (
        "DrJit LLVM is required by the test environment"
    )
    from drjit import llvm

    desc = TypeDesc.from_value(llvm.Float64([1.0, 2.0]))
    assert desc.kind == "drjit.Array"
    assert desc.dtype == "float64"


def test_drjit_shape_bind_when_llvm_is_available() -> None:
    import drjit

    assert drjit.has_backend(drjit.JitBackend.LLVM), (
        "DrJit LLVM is required by the test environment"
    )
    expected = TypeDesc("drjit.Array", shape=(2, 3), dtype="float64")
    assert (
        TypeDesc("drjit.Array", shape=(Symbol("N"), 3), dtype="float64").bind({"N": 2}) == expected
    )


def test_named_and_positional_structures_preserve_metadata() -> None:
    td = TypeDesc("numpy.ndarray", dims=(("row", 2, ("a", "b")),), dtype="float64")
    assert td.unary("neg") == td
    assert td.rename_axis("row", "item").dims == (("item", 2, ("a", "b")),)
    assert td.reshape((2,)).shape == (2,)


def test_scalar_binary_and_reduction() -> None:
    assert TypeDesc("scalar", dtype="int64").binary(
        TypeDesc("scalar", dtype="float64"), "add"
    ) == TypeDesc("scalar", dtype="float64")
    td = TypeDesc(
        "numpy.ndarray", dims=(("id", 2, (1, 2)), ("x", 3, ("a", "b", "c"))), dtype="float64"
    )
    assert td.reduce(("id",), "sum").dims == (("x", 3, ("a", "b", "c")),)


def test_broadcast_bind_and_conflict() -> None:
    left = TypeDesc("numpy.ndarray", dims=(("row", 2, None), ("col", 3, None)), dtype="float64")
    right = TypeDesc("numpy.ndarray", dims=(("row", 2, None), ("col", 3, None)), dtype="float64")
    assert left.broadcast(right).dims == left.dims
    assert TypeDesc("numpy.ndarray", shape=(Symbol("N"), 3), dtype="float64").bind(
        {"N": 2}
    ).shape == (2, 3)
    with pytest.raises(TypeDescConflictError):
        left.broadcast(
            TypeDesc("numpy.ndarray", dims=(("row", 4, None), ("col", 3, None)), dtype="float64")
        )


def test_projection_and_selection() -> None:
    price, volume = TypeDesc("scalar", dtype="float64"), TypeDesc("scalar", dtype="int64")
    record = TypeDesc("record", fields=(("price", price), ("volume", volume)))
    assert record.project("price") == price
    assert record.select(("price", "volume")).fields == record.fields


def test_negative_fields_missing() -> None:
    record = TypeDesc("record", fields=(("price", TypeDesc("scalar", dtype="float64")),))
    with pytest.raises(KeyError):
        record.select(("missing",))


def test_negative_fields_duplicate() -> None:
    record = TypeDesc("record", fields=(("price", TypeDesc("scalar", dtype="float64")),))
    with pytest.raises(OperationBindingError):
        record.select(("price", "price"))


def test_nested_samples_and_literals() -> None:
    assert make_samples(
        (TypeDesc("scalar", dtype="int64"), [TypeDesc("scalar", dtype="float64")]), {"literal": 3}
    ) == ((0, [0.0]), {"literal": 3})


def test_round_trip_infer_by_execution() -> None:
    result = infer_by_execution(lambda value: value + 1, TypeDesc("scalar", dtype="int64"))
    assert result == TypeDesc("scalar", dtype="int64")


@pytest.mark.parametrize(
    ("how", "rows"),
    [
        pytest.param("inner", 1, id="pandas-merge-inner"),
        pytest.param("left", 2, id="pandas-merge-left"),
        pytest.param("right", 2, id="pandas-merge-right"),
        pytest.param("outer", 3, id="pandas-merge-outer"),
    ],
)
def test_pandas_merge_matrix(how: str, rows: int) -> None:
    import pandas as pd

    result = pd.merge(*pandas_pair(), on="id", how=how, sort=how == "outer")
    qty_dtype = "float64" if how in {"left", "outer"} else "int64"
    assert TypeDesc.from_value(result) == TypeDesc(
        "pandas.DataFrame",
        columns=("id", "price", "qty"),
        dtypes=(("id", "int64"), ("price", "float64"), ("qty", qty_dtype)),
    )
    assert len(result) == rows


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(
            TypeDesc(
                "pandas.DataFrame",
                columns=("id_x", "price", "id_y", "qty"),
                dtypes=(
                    ("id_x", "int64"),
                    ("price", "float64"),
                    ("id_y", "int64"),
                    ("qty", "int64"),
                ),
            ),
            id="pandas-merge-cross",
        )
    ],
)
def test_pandas_cross(expected: TypeDesc) -> None:
    import pandas as pd

    cross = pd.merge(*pandas_pair(), how="cross")
    assert TypeDesc.from_value(cross) == expected
    assert len(cross) == 4


@pytest.mark.parametrize(
    ("values", "axis", "expected", "rows"),
    [
        pytest.param(
            pandas_pair,
            0,
            TypeDesc(
                "pandas.DataFrame",
                columns=("id", "price", "qty"),
                dtypes=(("id", "int64"), ("price", "float64"), ("qty", "float64")),
            ),
            4,
            id="pandas-concat-axis0",
        ),
        pytest.param(
            pandas_axis_pair,
            1,
            TypeDesc(
                "pandas.DataFrame",
                columns=("left", "right"),
                dtypes=(("left", "int64"), ("right", "int64")),
            ),
            2,
            id="pandas-concat-axis1",
        ),
    ],
)
def test_pandas_concat(values, axis: int, expected: TypeDesc, rows: int) -> None:
    import pandas as pd

    result = pd.concat(values(), axis=axis, ignore_index=axis == 0, join="outer")
    assert TypeDesc.from_value(result) == expected
    assert len(result) == rows


@pytest.mark.parametrize(
    "join",
    [
        pytest.param("inner", id="xarray-align-inner"),
        pytest.param("outer", id="xarray-align-outer"),
        pytest.param("left", id="xarray-align-left"),
        pytest.param("right", id="xarray-align-right"),
    ],
)
def test_xarray_align_matrix(join: str) -> None:
    import xarray as xr

    first, second = xr.align(*xarray_pair(), join=join)
    expected = {"inner": (2,), "outer": (1, 2, 3), "left": (1, 2), "right": (2, 3)}[join]
    assert tuple(first.coords["id"].values) == expected
    expected_desc = TypeDesc(
        "xarray.DataArray", dims=(("id", len(expected), None),), dtype="float64"
    )
    assert TypeDesc.from_value(first) == expected_desc
    assert TypeDesc.from_value(second) == expected_desc


def test_xarray_align_override_preserves_first_coordinates() -> None:
    import xarray as xr

    first, _ = xarray_pair()
    second = xr.DataArray(np.array([7.0, 8.0]), dims=("id",), coords={"id": [8, 9]})
    aligned_first, aligned_second = xr.align(first, second, join="override")
    assert tuple(aligned_first.coords["id"].values) == (1, 2)
    assert tuple(aligned_second.coords["id"].values) == (1, 2)
    expected = TypeDesc("xarray.DataArray", dims=(("id", 2, None),), dtype="float64")
    assert TypeDesc.from_value(aligned_first) == expected
    assert TypeDesc.from_value(aligned_second) == expected


@pytest.mark.parametrize(
    "join",
    [
        pytest.param("exact", id="xarray-align-exact"),
        pytest.param("override", id="xarray-align-override-error"),
    ],
)
def test_xarray_align_backend_errors(join: str) -> None:
    import xarray as xr

    if join == "exact":
        args = xarray_pair()
    else:
        first, _ = xarray_pair()
        args = (
            first,
            xr.DataArray(np.array([7.0, 8.0, 9.0]), dims=("id",), coords={"id": [2, 3, 4]}),
        )
    with pytest.raises(xr.AlignmentError, match=f"cannot align objects with join='{join}'"):
        xr.align(*args, join=join)


@pytest.mark.parametrize("compat", [pytest.param("identical", id="xarray-merge-conflict")])
def test_xarray_merge_conflict_backend_error(compat: str) -> None:
    import xarray as xr

    with pytest.raises(xr.MergeError, match="conflicting values"):
        xr.merge(dataset_conflict(), compat=compat)


@pytest.mark.parametrize(
    ("mode", "factory"),
    [
        pytest.param("no_conflicts", dataset_nonoverlap, id="xarray-merge-no-conflicts"),
        pytest.param("identical", dataset_identical, id="xarray-merge-identical"),
        pytest.param("equals", dataset_equals, id="xarray-merge-equal"),
        pytest.param("override", dataset_conflict, id="xarray-merge-override"),
    ],
)
def test_xarray_merge_modes(mode: str, factory) -> None:
    import xarray as xr

    args = factory()
    expected = TypeDesc(
        "xarray.Dataset",
        fields=(("price", TypeDesc("xarray.DataArray", dims=(("id", 2, None),), dtype="float64")),)
        if mode != "no_conflicts"
        else (
            ("price", TypeDesc("xarray.DataArray", dims=(("id", 2, None),), dtype="float64")),
            ("qty", TypeDesc("xarray.DataArray", dims=(("id", 2, None),), dtype="int64")),
        ),
    )
    assert TypeDesc.from_value(xr.merge(args, compat=mode)) == expected


def test_dataset_project_exact_field() -> None:
    dataset, _ = dataset_nonoverlap()
    projected = TypeDesc.from_value(dataset).project("price")
    assert projected == TypeDesc("xarray.DataArray", dims=(("id", 2, None),), dtype="float64")


@pytest.mark.parametrize(
    ("how", "columns", "rows"),
    [
        pytest.param("inner", ("id", "price", "qty"), 1, id="polars-join-inner"),
        pytest.param("left", ("id", "price", "qty"), 2, id="polars-join-left"),
        pytest.param("right", ("price", "id", "qty"), 2, id="polars-join-right"),
        pytest.param("full", ("id", "price", "id_right", "qty"), 3, id="polars-join-full"),
        pytest.param("semi", ("id", "price"), 1, id="polars-join-semi"),
        pytest.param("anti", ("id", "price"), 1, id="polars-join-anti"),
        pytest.param("cross", ("id", "price", "id_right", "qty"), 4, id="polars-join-cross"),
    ],
)
def test_polars_join_matrix(how: str, columns: tuple[str, ...], rows: int) -> None:
    left, right = polars_pair()
    result = left.join(right, on="id", how=how) if how != "cross" else left.join(right, how="cross")
    expected_dtypes = tuple((name, "Float64" if name == "price" else "Int64") for name in columns)
    assert TypeDesc.from_value(result) == TypeDesc(
        "polars.DataFrame", columns=columns, dtypes=expected_dtypes
    )
    assert len(result) == rows


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("equal", id="arrow-concat-equal"),
        pytest.param("mismatch", id="arrow-concat-mismatch"),
    ],
)
def test_arrow_concat_modes(mode: str) -> None:
    import pyarrow as pa

    if mode == "equal":
        desc = TypeDesc.from_value(pa.concat_tables(arrow_equal_pair()))
        assert desc == TypeDesc(
            "pyarrow.Table", columns=("id", "value"), dtypes=(("id", "int64"), ("value", "double"))
        )
    else:
        with pytest.raises(pa.lib.ArrowInvalid, match="Schema at index 1 was different"):
            pa.concat_tables(arrow_mismatch_pair())


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        pytest.param(
            numpy_empty,
            TypeDesc("numpy.ndarray", dims=(("dim0", 0, None), ("dim1", 3, None)), dtype="float64"),
            id="empty-numpy",
        ),
        pytest.param(
            xarray_empty,
            TypeDesc("xarray.DataArray", dims=(("id", 0, None),), dtype="float64"),
            id="empty-xarray",
        ),
        pytest.param(
            xarray_dataset_empty,
            TypeDesc(
                "xarray.Dataset",
                fields=(
                    (
                        "price",
                        TypeDesc("xarray.DataArray", dims=(("id", 0, None),), dtype="float64"),
                    ),
                ),
            ),
            id="empty-xarray-dataset",
        ),
        pytest.param(
            arrow_empty,
            TypeDesc(
                "pyarrow.Table",
                columns=("id", "price"),
                dtypes=(("id", "int64"), ("price", "double")),
            ),
            id="empty-arrow",
        ),
    ],
)
def test_empty_backend_descriptors(factory, expected: TypeDesc) -> None:
    assert TypeDesc.from_value(factory()) == expected


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        pytest.param(
            pandas_empty,
            TypeDesc(
                "pandas.DataFrame",
                columns=("id", "price"),
                dtypes=(("id", "int64"), ("price", "float64")),
            ),
            id="empty-pandas",
        ),
        pytest.param(
            polars_empty,
            TypeDesc(
                "polars.DataFrame",
                columns=("id", "price"),
                dtypes=(("id", "Int64"), ("price", "Float64")),
            ),
            id="empty-polars",
        ),
    ],
)
def test_empty_tabular_descriptors(factory, expected: TypeDesc) -> None:
    assert TypeDesc.from_value(factory()) == expected


def test_unsupported_operation() -> None:
    with pytest.raises(UnsupportedOperationError):
        TypeDesc("opaque").method("not_registered")


def test_unknown_labels_both_unknown() -> None:
    unknown = TypeDesc("xarray.DataArray", dims=(("id", 2, None),), dtype="float64")
    assert unknown.unify(unknown).dims == unknown.dims


def test_unknown_labels_known_unknown() -> None:
    unknown = TypeDesc("xarray.DataArray", dims=(("id", 2, None),), dtype="float64")
    with pytest.raises(TypeDescUnknownError):
        TypeDesc("xarray.DataArray", dims=(("id", 2, (1, 2)),), dtype="float64").unify(unknown)
