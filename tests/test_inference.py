"""Tests for typetrace.inference module."""

from dataclasses import dataclass

import pytest
from typetrace.core import Symbol, TypeDesc
from typetrace.inference import TypeContext, infer_by_execution, infer_types


class TestTypeContext:
    """Tests for TypeContext class."""

    def test_default_construction(self) -> None:
        """TypeContext can be created with defaults."""
        ctx = TypeContext()
        assert ctx.bindings == {}
        assert ctx.sources == {}
        assert ctx._cache == {}

    def test_construction_with_values(self) -> None:
        """TypeContext can be created with initial values."""
        bindings = {"N": 100, "T": 50}
        sources = {"data": TypeDesc(kind="ndarray", dims={"x": 10})}
        ctx = TypeContext(bindings=bindings, sources=sources)

        assert ctx.bindings == {"N": 100, "T": 50}
        assert ctx.sources == {"data": TypeDesc(kind="ndarray", dims={"x": 10})}

    def test_bind_returns_new_context(self) -> None:
        """bind() returns new context with additional binding."""
        ctx = TypeContext(bindings={"N": 100})
        new_ctx = ctx.bind("T", 50)

        assert ctx.bindings == {"N": 100}  # Original unchanged
        assert new_ctx.bindings == {"N": 100, "T": 50}  # New has both
        assert new_ctx._cache == {}  # Cache cleared

    def test_bind_preserves_sources(self) -> None:
        """bind() preserves sources from original context."""
        source = TypeDesc(kind="ndarray", dims={"x": 10})
        ctx = TypeContext(sources={"data": source})
        new_ctx = ctx.bind("N", 100)

        assert new_ctx.sources == {"data": source}

    def test_with_source_returns_new_context(self) -> None:
        """with_source() returns new context with additional source."""
        ctx = TypeContext(sources={"a": TypeDesc(kind="ndarray", dims={"x": 10})})
        new_type = TypeDesc(kind="ndarray", dims={"y": 20})
        new_ctx = ctx.with_source("b", new_type)

        assert "a" in ctx.sources
        assert "b" not in ctx.sources  # Original unchanged
        assert "a" in new_ctx.sources
        assert "b" in new_ctx.sources
        assert new_ctx.sources["b"] == new_type

    def test_with_source_preserves_bindings(self) -> None:
        """with_source() preserves bindings from original context."""
        ctx = TypeContext(bindings={"N": 100})
        new_ctx = ctx.with_source("data", TypeDesc(kind="ndarray", dims={"x": 10}))

        assert new_ctx.bindings == {"N": 100}

    @pytest.mark.parametrize(
        "dims,bindings,expected_dims",
        [
            ({"x": Symbol("N")}, {"N": 100}, {"x": 100}),
            ({"x": Symbol("N"), "y": 50}, {"N": 100}, {"x": 100, "y": 50}),
            ({"x": Symbol("N")}, {"T": 50}, {"x": Symbol("N")}),
            ({"x": 10, "y": 20}, {"N": 100}, {"x": 10, "y": 20}),
        ],
    )
    def test_resolve_dims(self, dims: dict, bindings: dict, expected_dims: dict) -> None:
        """resolve_dims() replaces Symbols with bound values."""
        ctx = TypeContext(bindings=bindings)
        t = TypeDesc(kind="ndarray", dims=dims, dtype="float64")
        result = ctx.resolve_dims(t)

        assert result.dims == expected_dims
        assert result.dtype == "float64"  # Other fields preserved

    def test_resolve_dims_with_none(self) -> None:
        """resolve_dims() returns unchanged TypeDesc when dims is None."""
        ctx = TypeContext(bindings={"N": 100})
        t = TypeDesc(kind="series", dtype="float64")
        result = ctx.resolve_dims(t)

        assert result == t


class TestInferTypes:
    """Tests for infer_types function."""

    def test_infer_simple_node(self) -> None:
        """infer_types works on a single node with no upstream."""

        @dataclass
        class SimpleNode:
            output_type: TypeDesc

            def upstream(self) -> tuple:
                return ()

            def type_transform(self) -> TypeDesc:
                return self.output_type

        expected = TypeDesc(kind="ndarray", dims={"x": 10}, dtype="float64")
        node = SimpleNode(output_type=expected)
        ctx = TypeContext()

        result = infer_types(node, ctx)
        assert result == expected

    def test_infer_with_upstream(self) -> None:
        """infer_types recursively processes upstream nodes."""

        @dataclass
        class SourceNode:
            output_type: TypeDesc

            def upstream(self) -> tuple:
                return ()

            def type_transform(self) -> TypeDesc:
                return self.output_type

        @dataclass
        class TransformNode:
            source: SourceNode

            def upstream(self) -> tuple:
                return (self.source,)

            def type_transform(self, input_type: TypeDesc) -> TypeDesc:
                # Double the x dimension
                new_dims = dict(input_type.dims) if input_type.dims else {}
                if "x" in new_dims:
                    new_dims["x"] = new_dims["x"] * 2  # type: ignore
                return input_type.with_dims(new_dims)

        source = SourceNode(output_type=TypeDesc(kind="ndarray", dims={"x": 10}, dtype="float64"))
        transform = TransformNode(source=source)
        ctx = TypeContext()

        result = infer_types(transform, ctx)
        assert result.dims == {"x": 20}
        assert result.dtype == "float64"

    def test_infer_caches_results(self) -> None:
        """infer_types caches results and reuses them."""
        call_count = 0

        @dataclass
        class CountingNode:
            def upstream(self) -> tuple:
                return ()

            def type_transform(self) -> TypeDesc:
                nonlocal call_count
                call_count += 1
                return TypeDesc(kind="ndarray", dims={"x": 10}, dtype="float64")

        node = CountingNode()
        ctx = TypeContext()

        # First call
        result1 = infer_types(node, ctx)
        assert call_count == 1

        # Second call on same context - should use cache
        result2 = infer_types(node, ctx)
        assert call_count == 1  # Not called again
        assert result1 == result2

    def test_infer_with_custom_get_upstream(self) -> None:
        """infer_types works with custom get_upstream function."""

        @dataclass
        class NodeWithDeps:
            deps: list
            output: TypeDesc

            def type_transform(self, *inputs: TypeDesc) -> TypeDesc:
                return self.output

        node = NodeWithDeps(deps=[], output=TypeDesc(kind="ndarray", dims={"x": 5}))
        ctx = TypeContext()

        result = infer_types(node, ctx, get_upstream=lambda n: tuple(n.deps))
        assert result.dims == {"x": 5}

    def test_infer_with_custom_get_transform(self) -> None:
        """infer_types works with custom get_transform function."""

        @dataclass
        class OpaqueNode:
            def upstream(self) -> tuple:
                return ()

        class Transformer:
            def type_transform(self) -> TypeDesc:
                return TypeDesc(kind="ndarray", dims={"y": 15}, dtype="float32")

        node = OpaqueNode()
        ctx = TypeContext()
        transformer = Transformer()

        result = infer_types(node, ctx, get_transform=lambda _: transformer)
        assert result.dims == {"y": 15}
        assert result.dtype == "float32"

    def test_infer_resolves_symbols(self) -> None:
        """infer_types resolves symbolic dims after type_transform."""

        @dataclass
        class SymbolicNode:
            def upstream(self) -> tuple:
                return ()

            def type_transform(self) -> TypeDesc:
                return TypeDesc(
                    kind="ndarray",
                    dims={"x": Symbol("N"), "y": 10},
                    dtype="float64",
                )

        node = SymbolicNode()
        ctx = TypeContext(bindings={"N": 100})

        result = infer_types(node, ctx)
        assert result.dims == {"x": 100, "y": 10}

    def test_infer_with_upstream_nodes_attr(self) -> None:
        """infer_types handles nodes with upstream_nodes() method."""

        @dataclass
        class AltNode:
            output_type: TypeDesc

            def upstream_nodes(self) -> tuple:
                return ()

            def type_transform(self) -> TypeDesc:
                return self.output_type

        expected = TypeDesc(kind="ndarray", dims={"x": 30}, dtype="float64")
        node = AltNode(output_type=expected)
        ctx = TypeContext()

        result = infer_types(node, ctx)
        assert result == expected

    def test_infer_node_without_upstream_interface(self) -> None:
        """infer_types treats nodes without upstream accessors as source nodes."""

        class LeafNode:
            def type_transform(self) -> TypeDesc:
                return TypeDesc(kind="ndarray", dims={"z": 9}, dtype="float64")

        result = infer_types(LeafNode(), TypeContext())

        assert result == TypeDesc(kind="ndarray", dims={"z": 9}, dtype="float64")

    @pytest.mark.parametrize("cycle_kind", ["self", "two_node"])
    def test_infer_detects_cycles(self, cycle_kind: str) -> None:
        """infer_types raises ValueError when upstream graph contains a cycle."""

        @dataclass
        class CyclicNode:
            deps: list["CyclicNode"]

            def upstream(self) -> tuple["CyclicNode", ...]:
                return tuple(self.deps)

            def type_transform(self, *inputs: TypeDesc) -> TypeDesc:
                return TypeDesc(kind="ndarray", dims={"x": 1}, dtype="float64")

        if cycle_kind == "self":
            node = CyclicNode(deps=[])
            node.deps.append(node)
            target = node
        else:
            left = CyclicNode(deps=[])
            right = CyclicNode(deps=[left])
            left.deps.append(right)
            target = left

        with pytest.raises(ValueError, match="Cycle detected"):
            infer_types(target, TypeContext())


class TestInferByExecution:
    """Tests for infer_by_execution function."""

    @pytest.fixture
    def xarray_available(self) -> bool:
        """Check if xarray is available."""
        from importlib.util import find_spec

        return find_spec("xarray") is not None

    @pytest.fixture
    def pandas_available(self) -> bool:
        """Check if pandas is available."""
        from importlib.util import find_spec

        return find_spec("pandas") is not None

    def test_infer_by_execution_xarray(self, xarray_available: bool) -> None:
        """infer_by_execution runs function and extracts type."""
        if not xarray_available:
            pytest.skip("xarray not installed")

        import xarray as xr

        def double_values(arr: xr.DataArray) -> xr.DataArray:
            return arr * 2

        input_type = TypeDesc(kind="ndarray", dims={"x": 10, "y": 20}, dtype="float64")
        result = infer_by_execution(double_values, input_type)

        assert result.kind == "ndarray"
        assert result.dtype == "float64"
        # Dims should be present (sizes may be 0 from sample)
        assert "x" in result.dims
        assert "y" in result.dims

    def test_infer_by_execution_with_kwargs(self, xarray_available: bool) -> None:
        """infer_by_execution passes kwargs to function."""
        if not xarray_available:
            pytest.skip("xarray not installed")

        import xarray as xr

        def scale_values(arr: xr.DataArray, factor: float = 1.0) -> xr.DataArray:
            return arr * factor

        input_type = TypeDesc(kind="ndarray", dims={"x": 10}, dtype="float64")
        result = infer_by_execution(scale_values, input_type, factor=2.0)

        assert result.kind == "ndarray"
        assert result.dtype == "float64"

    def test_infer_by_execution_pandas(self, pandas_available: bool) -> None:
        """infer_by_execution works with pandas DataFrame."""
        if not pandas_available:
            pytest.skip("pandas not installed")

        import pandas as pd

        def add_column(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["new_col"] = 0.0
            return df

        input_type = TypeDesc(
            kind="dataframe",
            columns=["a", "b"],
            dtypes={"a": "float64", "b": "int64"},
        )
        result = infer_by_execution(add_column, input_type)

        assert result.kind == "dataframe"
        assert "new_col" in result.columns

    def test_infer_by_execution_partial_schema_known_column_op_succeeds(
        self, pandas_available: bool
    ) -> None:
        """Partial-schema descriptors still support known-column operations."""
        if not pandas_available:
            pytest.skip("pandas not installed")

        import pandas as pd

        def select_known(df: pd.DataFrame) -> pd.DataFrame:
            return df[["a"]]

        input_type = TypeDesc(
            kind="dataframe",
            columns=["a"],
            dtypes={"a": "int64"},
            allow_extra_columns=True,
        )
        result = infer_by_execution(select_known, input_type, operation_name="select_known")

        assert result.kind == "dataframe"
        assert result.columns == ["a"]

    def test_infer_by_execution_partial_schema_unknown_column_op_fails_with_context(
        self, pandas_available: bool
    ) -> None:
        """Unknown-column access fails with infer_by_execution context."""
        if not pandas_available:
            pytest.skip("pandas not installed")

        import pandas as pd

        def select_unknown(df: pd.DataFrame) -> pd.DataFrame:
            return df[["b"]]

        input_type = TypeDesc(
            kind="dataframe",
            columns=["a"],
            dtypes={"a": "int64"},
            allow_extra_columns=True,
        )

        with pytest.raises(
            ValueError, match=r"infer_by_execution\(select_unknown\) execution failed"
        ):
            infer_by_execution(select_unknown, input_type, operation_name="select_unknown")

    @pytest.mark.parametrize("operation_name", ["select_all", "drop_complement"])
    def test_infer_by_execution_partial_schema_full_schema_required_fails_fast(
        self,
        pandas_available: bool,
        operation_name: str,
    ) -> None:
        """Full-schema-required mode fails before executing function."""
        if not pandas_available:
            pytest.skip("pandas not installed")

        called = {"value": False}

        def should_not_run(df):
            called["value"] = True
            return df

        input_type = TypeDesc(
            kind="dataframe",
            columns=["a"],
            dtypes={"a": "int64"},
            allow_extra_columns=True,
        )

        with pytest.raises(
            ValueError,
            match=r"partial dataframe schema \(allow_extra_columns=True\); operation requires exact full column set",
        ):
            infer_by_execution(
                should_not_run,
                input_type,
                require_exact_dataframe_schema=True,
                operation_name=operation_name,
            )

        assert called["value"] is False

    def test_infer_by_execution_exact_schema_path_not_blocked(self, pandas_available: bool) -> None:
        """Exact-schema descriptors are not blocked by full-schema-required guard."""
        if not pandas_available:
            pytest.skip("pandas not installed")

        import pandas as pd

        def identity(df: pd.DataFrame) -> pd.DataFrame:
            return df

        input_type = TypeDesc(
            kind="dataframe",
            columns=["a"],
            dtypes={"a": "int64"},
            allow_extra_columns=False,
        )
        result = infer_by_execution(
            identity,
            input_type,
            require_exact_dataframe_schema=True,
            operation_name="select_all",
        )

        assert result.kind == "dataframe"
        assert result.columns == ["a"]
