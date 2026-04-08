"""
Type inference engine for typetrace.

Provides the inference pass that walks a DAG-like structure and computes
output types using type_transform methods.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, cast

from typetrace.core import TypeDesc
from typetrace.execution_traits import ExecutionTraits, infer_execution_traits
from typetrace.layout_ops import check_handoff_compatibility
from typetrace.patterns import bind_symbols


class HasTypeTransform(Protocol):
    """Protocol for objects that can transform types."""

    def type_transform(self, *input_types: TypeDesc) -> TypeDesc:
        """Compute output type from input types."""
        ...


class HasUpstream(Protocol):
    """Protocol for objects that have upstream dependencies."""

    def upstream(self) -> tuple[Any, ...]:
        """Return upstream dependencies."""
        ...


@dataclass
class TypeContext:
    """
    Context for type inference - holds bindings and source types.

    Attributes:
        bindings: Symbol name → concrete int value
        sources: Named data source → TypeDesc
        cache: Memoization cache for inferred types
    """

    bindings: dict[str, int] = field(default_factory=dict)
    sources: dict[str, TypeDesc] = field(default_factory=dict)
    _cache: dict[int, TypeDesc] = field(default_factory=dict, repr=False)

    def bind(self, name: str, value: int) -> "TypeContext":
        """Return new context with additional binding."""
        return TypeContext(
            bindings={**self.bindings, name: value},
            sources=self.sources,
            _cache={},  # Clear cache on new bindings
        )

    def with_source(self, name: str, type_desc: TypeDesc) -> "TypeContext":
        """Return new context with additional source type."""
        return TypeContext(
            bindings=self.bindings,
            sources={**self.sources, name: type_desc},
            _cache={},
        )

    def resolve_dims(self, type_desc: TypeDesc) -> TypeDesc:
        """Resolve symbolic dims using current bindings."""
        if type_desc.dims is None:
            return type_desc
        resolved = bind_symbols(type_desc.dims, self.bindings)
        return type_desc.with_dims(resolved)


def _get_upstream_nodes(
    node: Any,
    get_upstream: Callable[[Any], tuple[Any, ...]] | None,
) -> tuple[Any, ...]:
    """Resolve upstream nodes from custom accessor or node methods."""
    if get_upstream is not None:
        return get_upstream(node)
    if hasattr(node, "upstream"):
        return cast(tuple[Any, ...], node.upstream())
    if hasattr(node, "upstream_nodes"):
        return cast(tuple[Any, ...], node.upstream_nodes())
    return ()


def _get_transformer(
    node: Any,
    get_transform: Callable[[Any], HasTypeTransform] | None,
) -> HasTypeTransform:
    """Resolve a node transformer."""
    if get_transform is not None:
        return get_transform(node)
    return cast(HasTypeTransform, node)


def infer_types(
    node: Any,
    context: TypeContext,
    get_transform: Callable[[Any], HasTypeTransform] | None = None,
    get_upstream: Callable[[Any], tuple[Any, ...]] | None = None,
    _visiting: set[int] | None = None,
) -> TypeDesc:
    """
    Infer output type for a node by walking its dependencies.

    This is the main inference function. It recursively computes types
    for all upstream nodes, then applies the node's type_transform.

    Args:
        node: The node to infer type for
        context: Type context with bindings and source types
        get_transform: Optional function to get type transformer from node
                      (defaults to node itself if it has type_transform)
        get_upstream: Optional function to get upstream nodes
                     (defaults to node.upstream() if it exists)

    Returns:
        TypeDesc for the node's output
    """
    node_id = id(node)
    if node_id in context._cache:
        return context._cache[node_id]

    visiting = _visiting if _visiting is not None else set()
    if node_id in visiting:
        raise ValueError(f"Cycle detected while inferring node {node!r}")

    visiting.add(node_id)
    try:
        upstream = _get_upstream_nodes(node, get_upstream)
        input_types = tuple(
            infer_types(up, context, get_transform, get_upstream, visiting) for up in upstream
        )
        transformer = _get_transformer(node, get_transform)
        output_type = transformer.type_transform(*input_types)
        resolved = context.resolve_dims(output_type)
    finally:
        visiting.remove(node_id)

    context._cache[node_id] = resolved
    return resolved


def _callable_label(fn: Callable[..., Any]) -> str:
    """Return readable callable label for error messages."""
    if hasattr(fn, "__qualname__") and hasattr(fn, "__module__"):
        return f"{fn.__module__}.{fn.__qualname__}"
    if hasattr(fn, "__name__"):
        return str(fn.__name__)
    return repr(fn)


def infer_by_execution(
    fn: Callable,
    *input_types: TypeDesc,
    expected_output_traits: ExecutionTraits | None = None,
    allow_device_copy: bool = False,
    require_exact_dataframe_schema: bool = False,
    operation_name: str | None = None,
    **kwargs: Any,
) -> TypeDesc:
    """
    Infer output type by executing function on sample data.

    For complex operations (like pd.merge) where encoding the type
    transform logic is harder than just running it.

    Args:
        fn: Function to execute
        *input_types: TypeDescs for inputs
        expected_output_traits: Optional runtime execution contract
        allow_device_copy: Allow cross-device handoff if transfer copy is required
        require_exact_dataframe_schema: Fail fast if any dataframe input uses
            partial schema semantics (trailing ellipsis in `columns`).
        operation_name: Optional operation context for error messages.
        **kwargs: Additional keyword arguments for fn

    Returns:
        TypeDesc extracted from function's output
    """
    operation = operation_name or _callable_label(fn)

    if require_exact_dataframe_schema:
        for index, type_desc in enumerate(input_types):
            has_trailing_ellipsis = (
                type_desc.columns is not None
                and len(type_desc.columns) > 0
                and type_desc.columns[-1] is ...
            )
            if type_desc.kind == "dataframe" and has_trailing_ellipsis:
                raise ValueError(
                    f"infer_by_execution({operation}): input[{index}] has partial "
                    "dataframe schema (columns end with ...); operation "
                    "requires exact full column set."
                )

    samples: list[Any] = []
    for index, type_desc in enumerate(input_types):
        try:
            samples.append(type_desc.make_sample())
        except Exception as exc:
            raise ValueError(
                f"infer_by_execution({operation}) sample-build failed for "
                f"input index {index} (input[{index}], kind={type_desc.kind}): {exc}"
            ) from exc

    try:
        result = fn(*samples, **kwargs)
    except Exception as exc:
        raise ValueError(f"infer_by_execution({operation}) execution failed: {exc}") from exc

    try:
        _validate_execution_handoff(result, expected_output_traits, allow_device_copy)
    except Exception as exc:
        raise ValueError(f"infer_by_execution({operation}) handoff-check failed: {exc}") from exc

    try:
        return TypeDesc.from_value(result)
    except Exception as exc:
        raise ValueError(f"infer_by_execution({operation}) output-extract failed: {exc}") from exc


def _validate_execution_handoff(
    result: Any,
    expected_output_traits: ExecutionTraits | None,
    allow_device_copy: bool,
) -> None:
    if expected_output_traits is None:
        return
    observed = infer_execution_traits(result)
    compatibility = check_handoff_compatibility(
        observed,
        expected_output_traits,
        allow_device_copy=allow_device_copy,
    )
    if compatibility.compatible:
        return
    reasons = "; ".join(compatibility.reasons)
    raise ValueError(f"Execution traits handoff check failed: {reasons}")
