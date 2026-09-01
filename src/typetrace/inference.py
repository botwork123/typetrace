"""
Type inference engine for typetrace.

Provides the inference pass that walks a DAG-like structure and computes
output types using type_transform methods.
"""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, cast

from typetrace.core import TypeDesc
from typetrace.errors import (
    OperationBindingError,
    OperationExecutionError,
    ResultInferenceError,
    SampleMaterializationError,
)
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


def _materialize(value: object, path: tuple[str | int, ...]) -> object:
    """Recursively replace descriptors with validated adapter samples."""
    from typetrace.adapters import adapter_for_value, get_adapter

    if isinstance(value, TypeDesc):
        try:
            adapter = get_adapter(value.kind)
            sample = adapter.make_sample(value)
            adapter.validate(value, sample)
            return sample
        except Exception as exc:
            raise SampleMaterializationError(
                f"could not materialize TypeDesc(kind={value.kind!r}): {exc}", path=path
            ) from exc
    if isinstance(value, tuple):
        return tuple(_materialize(item, path + (index,)) for index, item in enumerate(value))
    if isinstance(value, list):
        return [_materialize(item, path + (index,)) for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SampleMaterializationError("mapping keys must be strings", path=path)
            result[key] = _materialize(item, path + (key,))
        return result
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if hasattr(value, "upstream") or hasattr(value, "type_transform"):
        raise SampleMaterializationError("Nodes cannot be materialized", path=path)
    try:
        adapter_for_value(value)
    except Exception:
        pass
    else:
        raise SampleMaterializationError("live backend values cannot be materialized", path=path)
    raise SampleMaterializationError(f"unsupported sample value {type(value)!r}", path=path)


def make_samples(
    args: tuple[object, ...], kwargs: Mapping[str, object]
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Materialize descriptor values in supported argument containers."""
    if not isinstance(args, tuple):
        raise SampleMaterializationError("args must be a tuple", path=("args",))
    if not isinstance(kwargs, Mapping):
        raise SampleMaterializationError("kwargs must be a mapping", path=("kwargs",))
    return (
        tuple(_materialize(value, ("args", index)) for index, value in enumerate(args)),
        cast(dict[str, object], _materialize(kwargs, ("kwargs",))),
    )


def infer_by_execution(
    fn: Callable,
    *input_types: TypeDesc,
    call_args: tuple[object, ...] = (),
    call_kwargs: Mapping[str, object] | None = None,
    expected_output_traits: ExecutionTraits | None = None,
    allow_device_copy: bool = False,
    require_exact_dataframe_schema: bool = False,
    operation_name: str | None = None,
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
            if type_desc.kind in {"pandas.DataFrame", "polars.DataFrame"} and has_trailing_ellipsis:
                raise ValueError(
                    f"infer_by_execution({operation}): input[{index}] has partial "
                    "dataframe schema (columns end with ...); operation "
                    "requires exact full column set."
                )

    kwargs = dict(call_kwargs or {})
    positional: tuple[object, ...] = tuple(input_types) + tuple(call_args)
    signature = inspect.signature(fn)
    placeholders = tuple(object() if isinstance(item, TypeDesc) else item for item in positional)
    try:
        bound = signature.bind(*placeholders, **kwargs)
        bound.apply_defaults()
    except TypeError as exc:
        raise OperationBindingError(str(exc), operation=operation) from exc

    try:
        samples, sample_kwargs = make_samples(positional, kwargs)
    except SampleMaterializationError as exc:
        if exc.path and exc.path[0] == "args":
            index = exc.path[1] if len(exc.path) > 1 and isinstance(exc.path[1], int) else -1
            if 0 <= index < len(input_types):
                error_path = ("input_types",) + exc.path[1:]
            else:
                error_path = ("call_args", max(index - len(input_types), 0)) + exc.path[2:]
        else:
            error_path = exc.path
        raise SampleMaterializationError(
            f"infer_by_execution({operation}) sample-build failed for input index "
            f"{error_path[1] if len(error_path) > 1 else '?'}: {exc}",
            operation=operation,
            path=error_path,
        ) from exc

    try:
        result = fn(*samples, **sample_kwargs)
    except Exception as exc:
        raise OperationExecutionError(
            f"infer_by_execution({operation}) execution failed: {exc}", operation=operation
        ) from exc

    try:
        _validate_execution_handoff(result, expected_output_traits, allow_device_copy)
    except Exception as exc:
        raise OperationExecutionError(
            f"infer_by_execution({operation}) handoff-check failed: {exc}", operation=operation
        ) from exc

    try:
        return TypeDesc.from_value(result)
    except Exception as exc:
        raise ResultInferenceError(
            f"infer_by_execution({operation}) output inference failed: {exc}", operation=operation
        ) from exc


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
