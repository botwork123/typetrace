"""
DrJit adapter for typetrace.

Handles DrJit arrays and tensors.
"""

from typing import Any

from typetrace.core import TypeDesc
from typetrace.runtime_utils import infer_drjit_dtype

ADAPTER_KINDS = ("drjit.Array",)
OPERATIONS: dict[tuple[str, str], Any] = {}


def supports(value: object) -> bool:
    """Return whether value is a DrJit array."""
    import drjit as dr

    try:
        return bool(dr.is_array_v(value))
    except (AttributeError, TypeError):
        return type(value).__module__.split(".")[0] == "drjit"


def from_drjit(value: Any) -> TypeDesc:
    """
    Extract TypeDesc from DrJit array or tensor.

    Args:
        value: DrJit array (Float64, TensorXf, etc.)

    Returns:
        TypeDesc with shape, dtype, and drjit_type
    """
    import drjit as dr

    shape = dr.shape(value)
    dtype = _drjit_dtype(value)
    drjit_type = type(value)

    return TypeDesc(
        kind="drjit.Array",
        shape=shape,
        dtype=dtype,
        drjit_type=drjit_type,
    )


def _drjit_dtype(value: Any) -> str:
    """Extract normalized dtype string from DrJit array type name."""
    return infer_drjit_dtype(value)


def make_drjit_sample(type_desc: TypeDesc) -> Any:
    """
    Create minimal DrJit array from TypeDesc.

    Args:
        type_desc: TypeDesc with kind='drjit.Array'

    Returns:
        DrJit array with correct type (size 0)
    """
    import drjit as dr

    if type_desc.drjit_type is not None:
        # Use the exact DrJit type
        return dr.zeros(type_desc.drjit_type, 0)
    else:
        # Infer from dtype
        # Default to LLVM backend
        from drjit import llvm

        dtype_map = {
            "float64": llvm.Float64,
            "float32": llvm.Float,
            "int64": llvm.Int64,
            "int32": llvm.Int,
            "uint64": llvm.UInt64,
            "uint32": llvm.UInt,
            "bool": llvm.Bool,
        }
        drjit_type = dtype_map.get(type_desc.dtype or "float64", llvm.Float64)
        return dr.zeros(drjit_type, 0)


def infer(value: object) -> TypeDesc:
    """Protocol entry point for DrJit inference."""
    return from_drjit(value)


def make_sample(desc: TypeDesc) -> object:
    """Protocol entry point for DrJit sample creation."""
    return make_drjit_sample(desc)


def validate(desc: TypeDesc, value: object) -> None:
    """Validate a DrJit sample against its nominal kind."""
    if desc.kind != "drjit.Array" or not supports(value):
        raise TypeError(f"expected drjit.Array sample, got {type(value)!r}")
