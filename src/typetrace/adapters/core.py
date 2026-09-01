"""Core adapter for Python scalars, records, and opaque values."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from typetrace.core import TypeDesc
from typetrace.runtime_utils import module_root

ADAPTER_KINDS = ("scalar", "record", "opaque")
OPERATIONS: dict[tuple[str, str], Any] = {}
_BACKEND_ROOTS = {"numpy", "xarray", "pandas", "polars", "pyarrow", "drjit"}


def supports(value: object) -> bool:
    """Return whether the value belongs to the core Python namespace."""
    if module_root(value) in _BACKEND_ROOTS:
        if module_root(value) == "numpy":
            try:
                import numpy as np

                return isinstance(value, np.generic)
            except ImportError:
                return False
        return False
    if isinstance(value, (bool, int, float, str, bytes, type(None), Mapping)):
        return True
    if module_root(value) == "numpy":
        try:
            import numpy as np

            return isinstance(value, np.generic)
        except ImportError:
            return False
    return module_root(value) not in _BACKEND_ROOTS


def infer(value: object) -> TypeDesc:
    """Infer the scalar, record, or opaque descriptor for a Python value."""
    if isinstance(value, bool):
        return TypeDesc(kind="scalar", dtype="bool")
    if isinstance(value, int):
        return TypeDesc(kind="scalar", dtype="int64")
    if isinstance(value, float):
        return TypeDesc(kind="scalar", dtype="float64")
    if isinstance(value, str):
        return TypeDesc(kind="scalar", dtype="str")
    if isinstance(value, bytes) or value is None:
        return TypeDesc(kind="opaque", metadata=(("value", None),))
    if module_root(value) == "numpy":
        import numpy as np

        if isinstance(value, np.generic):
            return TypeDesc(kind="scalar", dtype=str(value.dtype))
    return TypeDesc._from_object(value)


def make_sample(desc: TypeDesc) -> object:
    """Create a deterministic Python sample for a core descriptor."""
    if desc.kind == "scalar":
        return {"bool": False, "int64": 0, "float64": 0.0, "str": ""}.get(desc.dtype or "", 0)
    if desc.kind == "opaque":
        return None
    if desc.kind == "record":
        values = {str(name): desc_value.make_sample() for name, desc_value in (desc.fields or ())}
        return SimpleNamespace(**values)
    raise TypeError(f"core adapter cannot make sample for kind={desc.kind!r}")


def validate(desc: TypeDesc, value: object) -> None:
    """Validate that a produced sample belongs to the descriptor namespace."""
    if desc.kind == "scalar" and not supports(value):
        raise TypeError(f"expected scalar sample, got {type(value)!r}")
    if desc.kind == "record" and not isinstance(value, (Mapping, SimpleNamespace)):
        raise TypeError(f"expected record sample, got {type(value)!r}")
    if desc.kind not in ADAPTER_KINDS:
        raise TypeError(f"core adapter cannot validate kind={desc.kind!r}")
