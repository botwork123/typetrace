"""
Type inference engine for typetrace.

Provides the inference pass that walks a DAG-like structure and computes
output types using type_transform methods.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

from typetrace.core import TypeDesc
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


T = TypeVar("T", bound=HasUpstream)


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


def infer_types(
    node: T,
    context: TypeContext,
    get_transform: Callable[[T], HasTypeTransform] | None = None,
    get_upstream: Callable[[T], tuple[T, ...]] | None = None,
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
    # Check cache
    node_id = id(node)
    if node_id in context._cache:
        return context._cache[node_id]

    # Get upstream nodes
    if get_upstream is not None:
        upstream = get_upstream(node)
    elif hasattr(node, "upstream"):
        upstream = node.upstream()
    elif hasattr(node, "upstream_nodes"):
        upstream = node.upstream_nodes()
    else:
        upstream = ()

    # Recursively infer upstream types
    input_types = tuple(infer_types(up, context, get_transform, get_upstream) for up in upstream)

    # Get type transformer
    if get_transform is not None:
        transformer = get_transform(node)
    else:
        transformer = node

    # Compute output type
    output_type = transformer.type_transform(*input_types)

    # Resolve any symbolic dims that are now bound
    output_type = context.resolve_dims(output_type)

    # Cache and return
    context._cache[node_id] = output_type
    return output_type


def infer_by_execution(
    fn: Callable,
    *input_types: TypeDesc,
    **kwargs: Any,
) -> TypeDesc:
    """
    Infer output type by executing function on sample data.

    For complex operations (like pd.merge) where encoding the type
    transform logic is harder than just running it.

    Args:
        fn: Function to execute
        *input_types: TypeDescs for inputs
        **kwargs: Additional keyword arguments for fn

    Returns:
        TypeDesc extracted from function's output
    """
    # Create sample data for each input
    samples = [t.make_sample() for t in input_types]

    # Execute function
    result = fn(*samples, **kwargs)

    # Extract type from result
    return TypeDesc.from_value(result)
