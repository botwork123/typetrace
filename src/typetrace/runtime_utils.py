"""Shared runtime helpers for backend dispatch and dtype normalization."""

from typing import Any


def module_root(value: Any) -> str:
    """Return the top-level module name for a runtime value."""
    return type(value).__module__.split(".", 1)[0]


def infer_drjit_dtype(value: Any) -> str:
    """Infer normalized dtype string from a DrJit type name."""
    type_name = type(value).__name__.lower()
    checks = (
        (("float64", "double"), "float64"),
        (("float",), "float32"),
        (("uint64",), "uint64"),
        (("uint",), "uint32"),
        (("int64",), "int64"),
        (("int",), "int32"),
        (("bool",), "bool"),
    )
    for tokens, dtype in checks:
        if any(token in type_name for token in tokens):
            return dtype
    return "unknown"
