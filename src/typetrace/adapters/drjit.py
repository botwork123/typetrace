"""
DrJit adapter for typetrace.

Handles DrJit arrays and tensors.
"""

from typing import Any

from typetrace.core import TypeDesc, Symbol


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
        kind="drjit",
        shape=shape,
        dtype=dtype,
        drjit_type=drjit_type,
    )


def _drjit_dtype(value: Any) -> str:
    """Extract dtype string from DrJit array type name."""
    type_name = type(value).__name__

    if "Float64" in type_name or "Double" in type_name:
        return "float64"
    elif "Float32" in type_name or "Float" in type_name:
        return "float32"
    elif "Int64" in type_name:
        return "int64"
    elif "Int32" in type_name or "Int" in type_name:
        return "int32"
    elif "UInt64" in type_name:
        return "uint64"
    elif "UInt32" in type_name or "UInt" in type_name:
        return "uint32"
    elif "Bool" in type_name:
        return "bool"
    else:
        return "unknown"


def make_drjit_sample(type_desc: TypeDesc) -> Any:
    """
    Create minimal DrJit array from TypeDesc.

    Args:
        type_desc: TypeDesc with kind='drjit'

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
