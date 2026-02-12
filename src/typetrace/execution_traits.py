"""Runtime execution contract for array layout/device compatibility."""

import importlib
from dataclasses import dataclass
from typing import Any, Literal, cast

Device = Literal["cpu", "cuda"]
LayoutOrder = Literal["C", "F", "strided"]


@dataclass(frozen=True)
class ExecutionTraits:
    """Runtime-only data contract for concrete array/tensor execution."""

    dtype: str
    shape: tuple[int, ...]
    device: Device = "cpu"
    layout_order: LayoutOrder = "strided"
    contiguous_c: bool = False
    contiguous_f: bool = False
    readonly: bool = False
    owner: str = "unknown"

    def __post_init__(self) -> None:
        _validate_shape(self.shape)
        _validate_device(self.device)
        _validate_layout(self.layout_order, self.contiguous_c, self.contiguous_f)


@dataclass(frozen=True)
class CompatibilityResult:
    """Result of checking whether two execution contracts are compatible."""

    compatible: bool
    requires_copy: bool
    reasons: tuple[str, ...]


def _validate_shape(shape: tuple[int, ...]) -> None:
    if any(dim < 0 for dim in shape):
        raise ValueError(f"Shape dimensions must be >= 0, got {shape}")


def _validate_device(device: Device) -> None:
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device {device!r}; expected 'cpu' or 'cuda'")


def _validate_layout(layout_order: str, contiguous_c: bool, contiguous_f: bool) -> None:
    if layout_order == "C" and not contiguous_c:
        raise ValueError("layout_order='C' requires contiguous_c=True")
    if layout_order == "F" and not contiguous_f:
        raise ValueError("layout_order='F' requires contiguous_f=True")


def infer_execution_traits(value: Any) -> ExecutionTraits:
    """Infer runtime traits from a concrete value (numpy first)."""
    module_root = type(value).__module__.split(".", 1)[0]
    if module_root == "numpy":
        return _infer_numpy_traits(value)
    if module_root == "xarray":
        return _infer_numpy_traits(value.values)
    if module_root == "pandas":
        return _infer_numpy_traits(value.to_numpy())
    if module_root == "torch":
        return _infer_torch_traits(value)
    if module_root == "drjit":
        return _infer_drjit_traits(value)
    raise TypeError(f"Cannot infer execution traits from type {type(value)!r}")


def _infer_numpy_traits(value: Any) -> ExecutionTraits:
    flags = value.flags
    return ExecutionTraits(
        dtype=str(value.dtype),
        shape=tuple(int(dim) for dim in value.shape),
        device="cpu",
        layout_order=_numpy_layout(flags.c_contiguous, flags.f_contiguous),
        contiguous_c=bool(flags.c_contiguous),
        contiguous_f=bool(flags.f_contiguous),
        readonly=not bool(flags.writeable),
        owner="numpy",
    )


def _numpy_layout(c_contiguous: bool, f_contiguous: bool) -> LayoutOrder:
    if c_contiguous:
        return "C"
    if f_contiguous:
        return "F"
    return "strided"


def _infer_torch_traits(value: Any) -> ExecutionTraits:
    is_cuda = bool(getattr(value, "is_cuda", False))
    return ExecutionTraits(
        dtype=str(value.dtype).replace("torch.", ""),
        shape=tuple(int(dim) for dim in value.shape),
        device="cuda" if is_cuda else "cpu",
        layout_order="C" if value.is_contiguous() else "strided",
        contiguous_c=bool(value.is_contiguous()),
        contiguous_f=False,
        readonly=False,
        owner="torch",
    )


def _infer_drjit_traits(value: Any) -> ExecutionTraits:
    dr = importlib.import_module("drjit")
    shape = tuple(int(dim) for dim in dr.shape(value))
    backend = str(dr.backend_v(value)).lower()
    device = cast(Device, "cuda" if "cuda" in backend else "cpu")
    return ExecutionTraits(
        dtype=_drjit_dtype(value),
        shape=shape,
        device=device,
        layout_order="strided",
        contiguous_c=False,
        contiguous_f=False,
        readonly=False,
        owner="drjit",
    )


def _drjit_dtype(value: Any) -> str:
    type_name = type(value).__name__.lower()
    if "float64" in type_name or "double" in type_name:
        return "float64"
    if "float" in type_name:
        return "float32"
    if "uint64" in type_name:
        return "uint64"
    if "uint" in type_name:
        return "uint32"
    if "int64" in type_name:
        return "int64"
    if "int" in type_name:
        return "int32"
    if "bool" in type_name:
        return "bool"
    return "unknown"
