"""Tests for typetrace.patterns module."""

import pytest

from typetrace.core import Symbol
from typetrace.patterns import (
    DimMismatch,
    add_dim,
    binary_result_dtype,
    bind_symbols,
    broadcast,
    promote_dtype,
    reduce_dim,
    unary_result_dtype,
    unify,
)


class TestUnify:
    """Tests for unify function."""

    @pytest.mark.parametrize(
        "d1,d2,expected",
        [
            # Same dims
            ({"x": 10}, {"x": 10}, {"x": 10}),
            # Disjoint dims - both included
            ({"x": 10}, {"y": 20}, {"x": 10, "y": 20}),
            # Empty cases
            ({}, {"x": 10}, {"x": 10}),
            ({"x": 10}, {}, {"x": 10}),
            ({}, {}, {}),
            # None cases
            (None, {"x": 10}, {"x": 10}),
            ({"x": 10}, None, {"x": 10}),
            (None, None, {}),
            # Multiple dims
            ({"x": 10, "y": 20}, {"y": 20, "z": 30}, {"x": 10, "y": 20, "z": 30}),
            # Symbolic dims
            ({"x": Symbol("N")}, {"x": Symbol("N")}, {"x": Symbol("N")}),
        ],
    )
    def test_unify_success(self, d1, d2, expected):
        """unify merges compatible dims."""
        assert unify(d1, d2) == expected

    @pytest.mark.parametrize(
        "d1,d2",
        [
            ({"x": 10}, {"x": 20}),  # Same name, different size
            ({"x": Symbol("N")}, {"x": 10}),  # Symbol vs int
        ],
    )
    def test_unify_mismatch(self, d1, d2):
        """unify raises DimMismatch for incompatible dims."""
        with pytest.raises(DimMismatch):
            unify(d1, d2)


class TestBroadcast:
    """Tests for broadcast function."""

    @pytest.mark.parametrize(
        "d1,d2,expected",
        [
            ({"x": 10}, {"y": 20}, {"x": 10, "y": 20}),
            ({"x": 10}, {"x": 10}, {"x": 10}),  # Same dim, second wins
            ({}, {"x": 10}, {"x": 10}),
            (None, {"x": 10}, {"x": 10}),
            ({"x": 10}, None, {"x": 10}),
            (None, None, {}),
        ],
    )
    def test_broadcast(self, d1, d2, expected):
        """broadcast unions dims."""
        assert broadcast(d1, d2) == expected


class TestAddDim:
    """Tests for add_dim function."""

    @pytest.mark.parametrize(
        "d,name,size,expected",
        [
            ({"x": 10}, "y", 20, {"x": 10, "y": 20}),
            ({}, "x", 10, {"x": 10}),
            (None, "x", 10, {"x": 10}),
            ({"x": 10}, "y", Symbol("M"), {"x": 10, "y": Symbol("M")}),
        ],
    )
    def test_add_dim(self, d, name, size, expected):
        """add_dim adds new dimension."""
        assert add_dim(d, name, size) == expected


class TestReduceDim:
    """Tests for reduce_dim function."""

    @pytest.mark.parametrize(
        "d,name,expected",
        [
            ({"x": 10, "y": 20}, "x", {"y": 20}),
            ({"x": 10}, "x", {}),
            ({"x": 10}, "y", {"x": 10}),  # Non-existent dim
            ({}, "x", {}),
            (None, "x", {}),
        ],
    )
    def test_reduce_dim(self, d, name, expected):
        """reduce_dim removes dimension."""
        assert reduce_dim(d, name) == expected


class TestPromoteDtype:
    """Tests for promote_dtype function."""

    @pytest.mark.parametrize(
        "dt1,dt2,expected",
        [
            ("float64", "float64", "float64"),
            ("float32", "float64", "float64"),
            ("int32", "float32", "float32"),
            ("int32", "int64", "int64"),
            ("bool", "int32", "int32"),
            (None, "float64", "float64"),
            ("float64", None, "float64"),
            (None, None, None),
            ("complex64", "float32", "complex64"),
        ],
    )
    def test_promote_dtype(self, dt1, dt2, expected):
        """promote_dtype returns common supertype."""
        assert promote_dtype(dt1, dt2) == expected


class TestBindSymbols:
    """Tests for bind_symbols function."""

    @pytest.mark.parametrize(
        "dims,bindings,expected",
        [
            # Bind symbolic to concrete
            ({"x": Symbol("N")}, {"N": 100}, {"x": 100}),
            # Mixed symbolic and concrete
            ({"x": Symbol("N"), "y": 20}, {"N": 100}, {"x": 100, "y": 20}),
            # Unbound symbol stays
            ({"x": Symbol("N")}, {"M": 100}, {"x": Symbol("N")}),
            # Empty cases
            ({}, {"N": 100}, {}),
            (None, {"N": 100}, {}),
        ],
    )
    def test_bind_symbols(self, dims, bindings, expected):
        """bind_symbols replaces symbols with bound values."""
        assert bind_symbols(dims, bindings) == expected


class TestBinaryResultDtype:
    """Tests for binary_result_dtype function."""

    @pytest.mark.parametrize(
        "left_dtype,right_dtype,operation,expected",
        [
            # Comparison ops → bool
            ("float64", "float64", "eq", "bool"),
            ("int32", "float64", "eq", "bool"),
            ("float64", "int32", "ne", "bool"),
            ("float64", "float64", "lt", "bool"),
            ("float64", "float64", "le", "bool"),
            ("float64", "float64", "gt", "bool"),
            ("float64", "float64", "ge", "bool"),
            # Symbolic comparison ops
            ("float64", "float64", "==", "bool"),
            ("float64", "float64", "!=", "bool"),
            ("float64", "float64", "<", "bool"),
            ("float64", "float64", "<=", "bool"),
            ("float64", "float64", ">", "bool"),
            ("float64", "float64", ">=", "bool"),
            # True division → float64
            ("int32", "int32", "truediv", "float64"),
            ("float32", "int32", "truediv", "float64"),
            ("int64", "int64", "/", "float64"),
            # Floor division → int64
            ("float64", "float64", "floordiv", "int64"),
            ("int32", "int32", "//", "int64"),
            # Arithmetic ops → promoted dtype
            ("float64", "float64", "add", "float64"),
            ("float32", "float64", "add", "float64"),
            ("int32", "float32", "mul", "float32"),
            ("int32", "int64", "sub", "int64"),
            ("float64", "float64", "pow", "float64"),
            ("float64", "float64", "mod", "float64"),
            # None handling
            (None, "float64", "add", "float64"),
            ("float64", None, "add", "float64"),
            (None, None, "add", None),
            (None, None, "eq", "bool"),  # Comparison still returns bool
        ],
    )
    def test_binary_result_dtype(self, left_dtype, right_dtype, operation, expected):
        """binary_result_dtype computes correct output dtype."""
        assert binary_result_dtype(left_dtype, right_dtype, operation) == expected


class TestUnaryResultDtype:
    """Tests for unary_result_dtype function."""

    @pytest.mark.parametrize(
        "input_dtype,operation,expected",
        [
            # Bool result ops
            ("float64", "not", "bool"),
            ("int32", "not", "bool"),
            ("float64", "isnan", "bool"),
            ("float64", "isinf", "bool"),
            ("float64", "isfinite", "bool"),
            # Complex to real
            ("complex128", "abs", "float64"),
            ("complex64", "abs", "float32"),
            ("float64", "abs", "float64"),  # Non-complex preserved
            ("complex128", "real", "float64"),
            ("complex128", "imag", "float64"),
            # Sign returns int
            ("float64", "sign", "int64"),
            ("int32", "sign", "int64"),
            # Preserve dtype (neg, pos, exp, log, etc.)
            ("float64", "neg", "float64"),
            ("float32", "neg", "float32"),
            ("int64", "neg", "int64"),
            ("float64", "pos", "float64"),
            ("float64", "exp", "float64"),
            ("float64", "log", "float64"),
            ("float64", "sqrt", "float64"),
            # None passthrough
            (None, "neg", None),
            (None, "not", "bool"),  # Still returns bool
        ],
    )
    def test_unary_result_dtype(self, input_dtype, operation, expected):
        """unary_result_dtype computes correct output dtype."""
        assert unary_result_dtype(input_dtype, operation) == expected
