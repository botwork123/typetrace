"""Tests for typetrace.core module."""

import pytest

from typetrace.core import TypeDesc, Symbol


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
            ("ndarray", {"x": 10, "y": 20}, "float64"),
            ("ndarray", {"symbol": Symbol("N")}, "float32"),
            ("dataframe", None, None),
            ("series", None, "int64"),
        ],
    )
    def test_typedesc_creation(
        self, kind: str, dims: dict | None, dtype: str | None
    ) -> None:
        """TypeDesc can be created with various configurations."""
        t = TypeDesc(kind=kind, dims=dims, dtype=dtype)
        assert t.kind == kind
        assert t.dims == dims
        assert t.dtype == dtype

    def test_with_dims(self) -> None:
        """with_dims returns new TypeDesc with updated dims."""
        original = TypeDesc(kind="ndarray", dims={"x": 10}, dtype="float64")
        updated = original.with_dims({"x": 20, "y": 30})

        assert original.dims == {"x": 10}  # Original unchanged
        assert updated.dims == {"x": 20, "y": 30}
        assert updated.dtype == "float64"  # Other fields preserved

    def test_with_dtype(self) -> None:
        """with_dtype returns new TypeDesc with updated dtype."""
        original = TypeDesc(kind="ndarray", dims={"x": 10}, dtype="float64")
        updated = original.with_dtype("float32")

        assert original.dtype == "float64"  # Original unchanged
        assert updated.dtype == "float32"
        assert updated.dims == {"x": 10}  # Other fields preserved

    def test_field_access(self) -> None:
        """field() returns nested TypeDesc for classes."""
        inner = TypeDesc(kind="ndarray", dims={"x": 10}, dtype="float64")
        outer = TypeDesc(kind="class", fields={"value": inner})

        assert outer.field("value") == inner

    def test_field_access_missing_fields(self) -> None:
        """field() raises ValueError when no fields."""
        t = TypeDesc(kind="ndarray", dims={"x": 10})
        with pytest.raises(ValueError, match="no fields"):
            t.field("value")

    def test_field_access_missing_key(self) -> None:
        """field() raises KeyError for missing field."""
        inner = TypeDesc(kind="ndarray", dims={"x": 10}, dtype="float64")
        outer = TypeDesc(kind="class", fields={"value": inner})

        with pytest.raises(KeyError, match="missing"):
            outer.field("missing")

    def test_frozen(self) -> None:
        """TypeDesc is immutable."""
        t = TypeDesc(kind="ndarray", dims={"x": 10})
        with pytest.raises(Exception):  # FrozenInstanceError
            t.dims = {"y": 20}  # type: ignore
