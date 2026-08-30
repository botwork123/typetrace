"""Stable exception taxonomy for TypeDesc v2 boundaries."""

from typetrace.core import (
    TypeDescConflictError,
    TypeDescError,
    TypeDescUnknownError,
    TypeDescValidationError,
    UnsupportedOperationError,
)

__all__ = [
    "TypeDescError",
    "TypeDescValidationError",
    "TypeDescConflictError",
    "TypeDescUnknownError",
    "UnsupportedOperationError",
]
