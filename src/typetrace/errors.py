"""Stable exception taxonomy for TypeDesc v2 boundaries."""

from typetrace.core import (
    AdapterAmbiguityError,
    AdapterRegistrationError,
    AdapterUnavailableError,
    OperationBindingError,
    OperationExecutionError,
    ResultInferenceError,
    SampleMaterializationError,
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
    "AdapterRegistrationError",
    "AdapterUnavailableError",
    "AdapterAmbiguityError",
    "SampleMaterializationError",
    "OperationBindingError",
    "OperationExecutionError",
    "ResultInferenceError",
]
