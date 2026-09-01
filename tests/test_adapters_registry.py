"""Contract tests for dynamic adapter discovery and exact lookup."""

from __future__ import annotations

from types import ModuleType

import pytest

from typetrace import TypeDesc
from typetrace.adapters import (
    discover_adapters,
    get_adapter,
    get_operation,
)
from typetrace.errors import (
    AdapterAmbiguityError,
    AdapterRegistrationError,
    AdapterUnavailableError,
    UnsupportedOperationError,
)


def test_builtin_registry_has_seven_modules_and_thirteen_kinds() -> None:
    adapters = discover_adapters()

    assert len(adapters) == 7
    assert [module.__name__.rsplit(".", 1)[-1] for module in adapters] == [
        "arrow",
        "core",
        "drjit",
        "numpy",
        "pandas",
        "polars",
        "xarray",
    ]
    assert {kind for adapter in adapters for kind in adapter.ADAPTER_KINDS} == {
        "scalar",
        "record",
        "opaque",
        "numpy.ndarray",
        "xarray.DataArray",
        "xarray.Dataset",
        "pandas.Series",
        "pandas.DataFrame",
        "polars.Series",
        "polars.DataFrame",
        "pyarrow.Array",
        "pyarrow.Table",
        "drjit.Array",
    }


@pytest.mark.parametrize("kind", ["scalar", "numpy.ndarray", "pandas.DataFrame"])
def test_get_adapter_is_exact(kind: str) -> None:
    assert kind in get_adapter(kind).ADAPTER_KINDS


def test_missing_kind_and_operation_are_named_errors() -> None:
    with pytest.raises(AdapterUnavailableError, match="missing.kind"):
        get_adapter("missing.kind")

    with pytest.raises(UnsupportedOperationError, match="missing-operation"):
        get_operation("scalar", "missing-operation")


def _adapter_module(name: str, kinds: tuple[str, ...]) -> ModuleType:
    module = ModuleType(name)
    module.ADAPTER_KINDS = kinds
    module.supports = lambda value: False
    module.infer = lambda value: TypeDesc(kind=kinds[0])
    module.make_sample = lambda desc: 0
    module.validate = lambda desc, value: None
    module.OPERATIONS = {}
    return module


def test_registration_rejects_duplicate_kind_and_malformed_module(monkeypatch) -> None:
    import typetrace.adapters as registry

    duplicate = _adapter_module("synthetic.duplicate", ("scalar",))
    malformed = ModuleType("synthetic.malformed")

    with pytest.raises(AdapterRegistrationError, match="duplicate"):
        registry._build_registry((duplicate, duplicate))
    with pytest.raises(AdapterRegistrationError, match="ADAPTER_KINDS"):
        registry._validate_adapter(malformed)


def test_supports_ambiguity_is_rejected() -> None:
    import typetrace.adapters as registry

    first = _adapter_module("synthetic.first", ("first",))
    second = _adapter_module("synthetic.second", ("second",))
    first.supports = lambda value: True
    second.supports = lambda value: True
    adapters = registry._build_registry((first, second))

    with pytest.raises(AdapterAmbiguityError, match="first.*second"):
        registry.adapter_for_value(object(), adapters=adapters)


def test_dynamic_adapter_takes_precedence_over_core_fallback() -> None:
    import typetrace.adapters as registry

    core = _adapter_module("typetrace.adapters.core", ("opaque",))
    custom = _adapter_module("synthetic.custom", ("custom.Value",))
    custom.supports = lambda value: True

    assert registry.adapter_for_value(object(), adapters=(core, custom)) is custom


def test_entry_point_adapter_dispatches_through_type_desc(monkeypatch) -> None:
    import typetrace.adapters as registry

    custom = _adapter_module("synthetic.entry_point", ("custom.Value",))
    custom.supports = lambda value: type(value).__name__ == "CustomValue"

    monkeypatch.setattr(registry, "_load_builtin", lambda name: None)
    monkeypatch.setattr(registry, "_entry_point_modules", lambda: (custom,))
    registry.discover_adapters.cache_clear()
    try:

        class CustomValue:
            pass

        assert TypeDesc.from_value(CustomValue()) == TypeDesc(kind="custom.Value")
    finally:
        registry.discover_adapters.cache_clear()
