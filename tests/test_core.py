"""Tests for typetrace.core module."""

# Skip markers for optional dependencies
from importlib.util import find_spec

import pytest

from typetrace.core import Symbol, TypeDesc


def skip_if_no_pandas():
    return find_spec("pandas") is None


def skip_if_no_xarray():
    return find_spec("xarray") is None


def skip_if_no_polars():
    return find_spec("polars") is None


def skip_if_no_pyarrow():
    return find_spec("pyarrow") is None


def skip_if_no_drjit():
    return find_spec("drjit") is None


pandas_required = pytest.mark.skipif(skip_if_no_pandas(), reason="pandas not installed")
xarray_required = pytest.mark.skipif(skip_if_no_xarray(), reason="xarray not installed")
polars_required = pytest.mark.skipif(skip_if_no_polars(), reason="polars not installed")
pyarrow_required = pytest.mark.skipif(skip_if_no_pyarrow(), reason="pyarrow not installed")
drjit_required = pytest.mark.skipif(skip_if_no_drjit(), reason="drjit not installed")


class TestSymbol:
    """Tests for Symbol class."""

    @pytest.mark.parametrize(
        "name,expected_repr",
        [
            ("N", "Symbol('N')"),
            ("universe", "Symbol('universe')"),
            ("T", "Symbol('T')"),
        ],
    )
    def test_symbol_repr(self, name: str, expected_repr: str) -> None:
        """Symbol repr shows name."""
        sym = Symbol(name)
        assert repr(sym) == expected_repr

    def test_symbol_equality(self) -> None:
        """Symbols with same name are equal."""
        assert Symbol("N") == Symbol("N")
        assert Symbol("N") != Symbol("M")

    def test_symbol_hashable(self) -> None:
        """Symbols can be used as dict keys."""
        d = {Symbol("N"): 100}
        assert d[Symbol("N")] == 100


class TestTypeDesc:
    """Tests for TypeDesc class."""

    @pytest.mark.parametrize(
        "kind,dims,dtype",
        [
            ("numpy.ndarray", (("x", 10, None), ("y", 20, None)), "float64"),
            ("numpy.ndarray", (("symbol", Symbol("N"), None),), "float32"),
            ("pandas.DataFrame", None, None),
            ("pandas.Series", None, "int64"),
        ],
    )
    def test_typedesc_creation(self, kind: str, dims: tuple | None, dtype: str | None) -> None:
        """TypeDesc can be created with various configurations."""
        t = TypeDesc(kind=kind, dims=dims, dtype=dtype)
        assert t.kind == kind
        assert t.dims == dims
        assert t.dtype == dtype

    def test_with_dims(self) -> None:
        """with_dims returns new TypeDesc with updated dims."""
        original = TypeDesc(kind="numpy.ndarray", dims=(("x", 10, None),), dtype="float64")
        updated = original.with_dims((("x", 20, None), ("y", 30, None)))

        assert original.dims == (("x", 10, None),)  # Original unchanged
        assert updated.dims == (("x", 20, None), ("y", 30, None))
        assert updated.dtype == "float64"  # Other fields preserved

    def test_with_dtype(self) -> None:
        """with_dtype returns new TypeDesc with updated dtype."""
        original = TypeDesc(kind="numpy.ndarray", dims=(("x", 10, None),), dtype="float64")
        updated = original.with_dtype("float32")

        assert original.dtype == "float64"  # Original unchanged
        assert updated.dtype == "float32"
        assert updated.dims == (("x", 10, None),)  # Other fields preserved

    def test_field_access(self) -> None:
        """field() returns nested TypeDesc for classes."""
        inner = TypeDesc(kind="numpy.ndarray", dims=(("x", 10, None),), dtype="float64")
        outer = TypeDesc(kind="record", fields=(("value", inner),))

        assert outer.field("value") == inner

    def test_field_access_missing_fields(self) -> None:
        """field() raises ValueError when no fields."""
        t = TypeDesc(kind="numpy.ndarray", dims=(("x", 10, None),))
        with pytest.raises(ValueError, match="no fields"):
            t.field("value")

    def test_field_access_missing_key(self) -> None:
        """field() raises KeyError for missing field."""
        inner = TypeDesc(kind="numpy.ndarray", dims=(("x", 10, None),), dtype="float64")
        outer = TypeDesc(kind="record", fields=(("value", inner),))

        with pytest.raises(KeyError, match="missing"):
            outer.field("missing")

    def test_typedesc_is_deeply_immutable_and_hashable(self) -> None:
        dims = [("instrument", Symbol("N"), None)]
        fields = {"price": TypeDesc(kind="numpy.ndarray", dims=(("instrument", 2, None),))}
        columns = ["price", ...]
        descriptor = TypeDesc(
            kind="custom.record",
            dims=dims,
            fields=tuple(fields.items()),
            columns=columns,
            dtypes=(("price", "float64"),),
            index=(("row", 2, None),),
        )

        dims.append(("other", 3, None))
        fields["volume"] = TypeDesc(kind="scalar", dtype="float64")
        columns.append("volume")

        assert descriptor.dims == (("instrument", Symbol("N"), None),)
        assert descriptor.columns == ("price", ...)
        assert hash(descriptor) == hash(descriptor)
        with pytest.raises(TypeError):
            descriptor.dims[0] = ("x", 1, None)  # type: ignore[index]
        with pytest.raises(TypeError):
            descriptor.fields[0] = ("x", descriptor)  # type: ignore[index]
        with pytest.raises(TypeError):
            descriptor.columns[0] = "x"  # type: ignore[index]

    def test_equal_typedescs_have_stable_fingerprint(self) -> None:
        left = TypeDesc(kind="numpy.ndarray", dims=(("x", 2, None),), dtype="float64")
        right = TypeDesc(kind="numpy.ndarray", dims=(("x", 2, None),), dtype="float64")

        assert left == right
        assert hash(left) == hash(right)
        assert left.fingerprint() == right.fingerprint()

    def test_frozen(self) -> None:
        """TypeDesc is immutable."""
        t = TypeDesc(kind="numpy.ndarray", dims=(("x", 10, None),))
        with pytest.raises(Exception):  # FrozenInstanceError
            t.dims = (("y", 20, None),)  # type: ignore

    def test_dataframe_exact_schema_default(self) -> None:
        """Exact schema remains default without trailing ellipsis."""
        t = TypeDesc(kind="pandas.DataFrame", columns=("a",), dtypes=(("a", "int64"),))
        assert t.columns == ("a",)

    @pytest.mark.parametrize(
        "columns,expected_known",
        [
            (("a", ...), ["a"]),
            ((...,), []),
            (("a",), ["a"]),
        ],
    )
    def test_dataframe_partial_schema_ellipsis_only(
        self, columns: tuple, expected_known: list
    ) -> None:
        """Trailing ellipsis is the only partial-schema signal."""
        t = TypeDesc(kind="pandas.DataFrame", columns=columns, dtypes=(("a", "int64"),))
        assert t.known_columns() == expected_known

    @pytest.mark.parametrize(
        "columns",
        [
            ("a", ..., "b"),
            ("a", ..., ...),
        ],
    )
    def test_dataframe_partial_schema_ellipsis_must_be_trailing(self, columns: tuple) -> None:
        """Ellipsis marker must only appear as final columns entry."""
        with pytest.raises(ValueError, match="ellipsis must be trailing"):
            TypeDesc(kind="pandas.DataFrame", columns=columns)


class TestTypeDescFromValue:
    """Tests for TypeDesc.from_value class method."""

    @xarray_required
    def test_from_value_xarray_dataarray(self) -> None:
        """from_value dispatches to xarray adapter for DataArray."""
        import numpy as np
        import xarray as xr

        da = xr.DataArray(np.zeros((5, 10)), dims=["x", "y"])
        result = TypeDesc.from_value(da)

        assert result.kind == "xarray.DataArray"
        assert result.dims == (("x", 5, None), ("y", 10, None))
        assert result.dtype == "float64"

    @pandas_required
    def test_from_value_pandas_dataframe(self) -> None:
        """from_value dispatches to pandas adapter for DataFrame."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = TypeDesc.from_value(df)

        assert result.kind == "pandas.DataFrame"
        assert result.columns == ("a", "b")

    @pandas_required
    def test_from_value_pandas_series(self) -> None:
        """from_value dispatches to pandas adapter for Series."""
        import pandas as pd

        s = pd.Series([1.0, 2.0, 3.0])
        result = TypeDesc.from_value(s)

        assert result.kind == "pandas.Series"
        assert result.dtype == "float64"

    @polars_required
    def test_from_value_polars_dataframe(self) -> None:
        """from_value dispatches to polars adapter for DataFrame."""
        import polars as pl

        df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = TypeDesc.from_value(df)

        assert result.kind == "polars.DataFrame"
        assert result.columns == ("a", "b")

    @pyarrow_required
    def test_from_value_arrow_table(self) -> None:
        """from_value dispatches to arrow adapter for Table."""
        import pyarrow as pa

        table = pa.table({"a": [1, 2], "b": [3.0, 4.0]})
        result = TypeDesc.from_value(table)

        assert result.kind == "pyarrow.Table"
        assert result.columns == ("a", "b")

    @drjit_required
    def test_from_value_drjit_array(self) -> None:
        """from_value dispatches to drjit adapter."""
        from drjit import llvm

        arr = llvm.Float64([1.0, 2.0, 3.0])
        result = TypeDesc.from_value(arr)

        assert result.kind == "drjit.Array"
        assert result.dtype == "float64"

    def test_from_value_opaque_object(self) -> None:
        """from_value falls back to _from_object for unknown types."""

        class CustomClass:
            def __init__(self):
                self.value = 42
                self.name = "test"

        obj = CustomClass()
        result = TypeDesc.from_value(obj)

        assert result.kind == "record"
        assert result.fields is not None
        assert {key for key, _ in result.fields} >= {"value", "name"}

    def test_from_value_opaque_object_no_public_attrs(self) -> None:
        """from_value handles objects with no public non-callable attrs."""

        class EmptyClass:
            def __init__(self):
                self._private = 123

            def method(self):
                pass

        obj = EmptyClass()
        result = TypeDesc.from_value(obj)

        assert result.kind == "record"
        assert result.fields == ()

    @pytest.mark.parametrize("error_cls", [AttributeError, RuntimeError, ValueError])
    def test_from_value_opaque_object_with_property_error_types(
        self, error_cls: type[Exception]
    ) -> None:
        """from_value skips attributes whose property getters raise known runtime errors."""

        class ProblematicClass:
            def __init__(self):
                self.good_attr = 1

            @property
            def bad_attr(self):
                raise error_cls("Cannot access this!")

        result = TypeDesc.from_value(ProblematicClass())

        assert result.kind == "record"
        assert result.fields is not None
        assert [key for key, _ in result.fields] == ["good_attr"]

    def test_from_value_module_substring_does_not_misdispatch(self) -> None:
        """from_value does not dispatch adapters when module merely contains adapter name."""

        class FakeXarrayLike:
            __module__ = "notxarray.fake"

            def __init__(self):
                self.value = 7

        result = TypeDesc.from_value(FakeXarrayLike())

        assert result.kind == "record"
        assert result.fields is not None
        assert [key for key, _ in result.fields] == ["value"]
        assert result.field("value") == TypeDesc(kind="scalar", dtype="int64")

    def test_from_value_self_referencing_object(self) -> None:
        """from_value handles objects that reference themselves without infinite recursion."""

        class SelfRef:
            def __init__(self):
                self.me = self
                self.value = 42

        obj = SelfRef()
        result = TypeDesc.from_value(obj)

        assert result.kind == "record"
        assert result.fields is not None
        assert "value" in {key for key, _ in result.fields}
        # Self-reference should be detected and marked as recursive
        assert "me" in {key for key, _ in result.fields}
        assert result.field("me").kind == "recursive"

    def test_from_value_circular_reference(self) -> None:
        """from_value handles circular references between objects."""

        class A:
            pass

        class B:
            pass

        a = A()
        b = B()
        a.other = b
        b.other = a

        result = TypeDesc.from_value(a)

        assert result.kind == "record"
        assert result.fields is not None
        assert "other" in {key for key, _ in result.fields}
        # B should be extracted, with its reference back to A marked as recursive
        b_desc = result.field("other")
        assert b_desc.kind == "record"
        assert b_desc.fields is not None
        assert "other" in {key for key, _ in b_desc.fields}
        assert b_desc.field("other").kind == "recursive"

    def test_from_value_deep_nesting_no_cycle(self) -> None:
        """from_value handles deep nesting without cycles (no false positive recursion)."""

        class Node:
            def __init__(self, val, child=None):
                self.val = val
                self.child = child

        # Deep but no cycle
        deep = Node(1, Node(2, Node(3, Node(4, Node(5)))))
        result = TypeDesc.from_value(deep)

        assert result.kind == "record"
        # Should traverse all levels without marking anything as recursive
        current = result
        for _ in range(5):
            assert current.kind == "record"
            assert current.fields is not None
            assert "val" in {key for key, _ in current.fields}
            child = current.field("child")
            if child.fields:
                current = child


class TestTypeDescMakeSample:
    """Tests for TypeDesc.make_sample method."""

    @xarray_required
    def test_make_sample_ndarray(self) -> None:
        """make_sample creates xarray DataArray for ndarray kind."""
        import xarray as xr

        t = TypeDesc(
            kind="xarray.DataArray",
            dims=(
                ("x", 10, None),
                ("y", 20, None),
            ),
            dtype="float32",
        )
        result = t.make_sample()

        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {"x", "y"}
        assert result.dtype == "float32"

    @pandas_required
    def test_make_sample_dataframe(self) -> None:
        """make_sample creates pandas DataFrame for dataframe kind."""
        import pandas as pd

        t = TypeDesc(
            kind="pandas.DataFrame",
            columns=(
                "a",
                "b",
            ),
            dtypes=(
                ("a", "float64"),
                ("b", "int32"),
            ),
        )
        result = t.make_sample()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

    @pandas_required
    def test_make_sample_series(self) -> None:
        """make_sample creates pandas Series for series kind."""
        import pandas as pd

        t = TypeDesc(kind="pandas.Series", dtype="int64")
        result = t.make_sample()

        assert isinstance(result, pd.Series)
        assert result.dtype == "int64"

    @drjit_required
    def test_make_sample_drjit(self) -> None:
        """make_sample creates DrJit array for drjit kind."""
        from drjit import llvm

        t = TypeDesc(kind="drjit.Array", drjit_type=llvm.Float64, dtype="float64")
        result = t.make_sample()

        assert type(result) is llvm.Float64

    @pytest.mark.skipif(find_spec("pyarrow") is None, reason="pyarrow not installed")
    def test_make_sample_columnar(self) -> None:
        """make_sample works for columnar kind (Arrow tables)."""
        import pyarrow as pa

        t = TypeDesc(
            kind="pyarrow.Table",
            columns=(
                "a",
                "b",
            ),
            dtypes=(
                ("a", "int64"),
                ("b", "float64"),
            ),
        )
        result = t.make_sample()

        assert isinstance(result, pa.Table)
        assert result.column_names == ["a", "b"]
        assert result.num_rows == 0

    def test_make_sample_class_not_implemented(self) -> None:
        """make_sample raises NotImplementedError for class kind."""
        t = TypeDesc(
            kind="record", fields=(("x", TypeDesc(kind="numpy.ndarray", dims=(("a", 1, None),))),)
        )

        with pytest.raises(NotImplementedError, match="make_sample not implemented"):
            t.make_sample()
