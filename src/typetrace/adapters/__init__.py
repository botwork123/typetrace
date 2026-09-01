"""Dynamic adapter protocol, discovery, and exact lookup."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Mapping
from functools import lru_cache
from importlib.util import find_spec
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Protocol

from typetrace.core import (
    AdapterAmbiguityError,
    AdapterRegistrationError,
    AdapterUnavailableError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from typetrace.core import TypeDesc


class Adapter(Protocol):
    """Protocol implemented by a backend adapter module."""

    ADAPTER_KINDS: tuple[str, ...]
    OPERATIONS: Mapping[tuple[str, str], Callable[..., object]]

    def supports(self, value: object) -> bool: ...

    def infer(self, value: object) -> "TypeDesc": ...

    def make_sample(self, desc: "TypeDesc") -> object: ...

    def validate(self, desc: "TypeDesc", value: object) -> None: ...


_BUILTIN_MODULES = (
    "typetrace.adapters.arrow",
    "typetrace.adapters.core",
    "typetrace.adapters.drjit",
    "typetrace.adapters.numpy",
    "typetrace.adapters.pandas",
    "typetrace.adapters.polars",
    "typetrace.adapters.xarray",
)
_OPTIONAL_DEPENDENCIES = {
    "typetrace.adapters.arrow": "pyarrow",
    "typetrace.adapters.drjit": "drjit",
    "typetrace.adapters.numpy": "numpy",
    "typetrace.adapters.pandas": "pandas",
    "typetrace.adapters.polars": "polars",
    "typetrace.adapters.xarray": "xarray",
}


def _load_builtin(name: str) -> ModuleType | None:
    dependency = _OPTIONAL_DEPENDENCIES.get(name)
    if dependency is not None:
        try:
            if find_spec(dependency) is None:
                return None
        except (ImportError, ModuleNotFoundError, ValueError):
            return None
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError):
        if dependency is not None:
            return None
        raise


def _entry_point_modules() -> tuple[ModuleType, ...]:
    entries = importlib.metadata.entry_points()
    selected = entries.select(group="typetrace.adapters") if hasattr(entries, "select") else ()
    if not selected and isinstance(entries, dict):
        selected = entries.get("typetrace.adapters", ())
    modules: list[tuple[str, str, ModuleType]] = []
    for entry in selected:
        try:
            loaded = entry.load()
        except (ImportError, ModuleNotFoundError) as exc:
            raise AdapterUnavailableError(
                f"adapter entry point {entry.name!r} is unavailable"
            ) from exc
        modules.append((getattr(loaded, "__name__", ""), entry.name, loaded))
    return tuple(module for _, _, module in sorted(modules, key=lambda item: item[:2]))


def _validate_adapter(adapter: object) -> Adapter:
    required = ("ADAPTER_KINDS", "supports", "infer", "make_sample", "validate", "OPERATIONS")
    missing = [name for name in required if not hasattr(adapter, name)]
    if missing:
        raise AdapterRegistrationError(f"adapter is missing exports: {', '.join(missing)}")
    kinds = getattr(adapter, "ADAPTER_KINDS")
    if (
        not isinstance(kinds, tuple)
        or not kinds
        or any(not isinstance(kind, str) or not kind for kind in kinds)
    ):
        raise AdapterRegistrationError("ADAPTER_KINDS must be a non-empty tuple of strings")
    if len(set(kinds)) != len(kinds):
        raise AdapterRegistrationError("ADAPTER_KINDS contains duplicate kinds")
    for name in required[1:5]:
        if not callable(getattr(adapter, name)):
            raise AdapterRegistrationError(f"adapter export {name!r} must be callable")
    operations = getattr(adapter, "OPERATIONS")
    if not isinstance(operations, Mapping):
        raise AdapterRegistrationError("OPERATIONS must be a mapping")
    for key, operation in operations.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(item, str) and item for item in key)
            or key[0] not in kinds
            or not callable(operation)
        ):
            raise AdapterRegistrationError(f"invalid adapter operation registration: {key!r}")
    return adapter  # type: ignore[return-value]


def _build_registry(modules: tuple[object, ...]) -> tuple[Adapter, ...]:
    """Validate modules and reject duplicate kind or operation ownership."""
    registered: list[Adapter] = []
    kinds: dict[str, str] = {}
    operations: dict[tuple[str, str], str] = {}
    for raw in modules:
        adapter = _validate_adapter(raw)
        module_name = getattr(raw, "__name__", repr(raw))
        for kind in adapter.ADAPTER_KINDS:
            if kind in kinds:
                raise AdapterRegistrationError(
                    f"duplicate kind {kind!r}: {module_name} and {kinds[kind]}"
                )
            kinds[kind] = module_name
        for key in adapter.OPERATIONS:
            if key in operations:
                raise AdapterRegistrationError(
                    f"duplicate operation {key!r}: {module_name} and {operations[key]}"
                )
            operations[key] = module_name
        registered.append(adapter)
    return tuple(registered)


@lru_cache(maxsize=1)
def discover_adapters() -> tuple[Adapter, ...]:
    """Return the deterministic cached built-in and entry-point registry."""
    builtins = tuple(module for name in _BUILTIN_MODULES if (module := _load_builtin(name)))
    entry_points = _entry_point_modules()
    return _build_registry(
        tuple(sorted(builtins + entry_points, key=lambda item: getattr(item, "__name__", "")))
    )


def get_adapter(kind: str) -> Adapter:
    """Look up an adapter by its exact nominal kind."""
    for adapter in discover_adapters():
        if kind in adapter.ADAPTER_KINDS:
            return adapter
    raise AdapterUnavailableError(f"no adapter is available for kind {kind!r}")


def get_operation(kind: str, name: str) -> Callable[..., object]:
    """Look up a runtime operation by its exact ``(kind, name)`` key."""
    adapter = get_adapter(kind)
    try:
        return adapter.OPERATIONS[(kind, name)]
    except KeyError as exc:
        raise UnsupportedOperationError(
            f"unsupported operation {name!r} for kind {kind!r}"
        ) from exc


def adapter_for_value(value: object, *, adapters: tuple[Adapter, ...] | None = None) -> Adapter:
    """Find the unique adapter whose ``supports`` predicate matches a value."""
    matches = [adapter for adapter in adapters or discover_adapters() if adapter.supports(value)]
    if len(matches) > 1:
        matches = [
            adapter
            for adapter in matches
            if getattr(adapter, "__name__", "") != "typetrace.adapters.core"
        ] or matches
    if len(matches) > 1:
        names = ", ".join(getattr(adapter, "__name__", repr(adapter)) for adapter in matches)
        raise AdapterAmbiguityError(f"multiple adapters support value: {names}")
    if not matches:
        raise AdapterUnavailableError(f"no adapter supports value of type {type(value)!r}")
    return matches[0]


__all__ = [
    "Adapter",
    "adapter_for_value",
    "discover_adapters",
    "get_adapter",
    "get_operation",
]
