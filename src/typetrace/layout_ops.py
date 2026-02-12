"""Layout compatibility and transition helpers for runtime handoff boundaries."""

from typing import Literal

from typetrace.execution_traits import CompatibilityResult, ExecutionTraits


def check_handoff_compatibility(
    source: ExecutionTraits,
    target: ExecutionTraits,
    allow_device_copy: bool = False,
) -> CompatibilityResult:
    """Check runtime compatibility for evaluator/backend handoffs."""
    reasons: list[str] = []
    requires_copy = False
    _check_dtype(source, target, reasons)
    _check_rank_shape(source, target, reasons)
    requires_copy = _check_layout(source, target, reasons) or requires_copy
    requires_copy = _check_contiguous(source, target, reasons) or requires_copy
    device_check = _check_device(source, target, allow_device_copy, reasons)
    requires_copy = device_check or requires_copy
    return CompatibilityResult(not reasons, requires_copy, tuple(reasons))


def _check_dtype(source: ExecutionTraits, target: ExecutionTraits, reasons: list[str]) -> None:
    if source.dtype != target.dtype:
        reasons.append(f"dtype mismatch: source={source.dtype}, target={target.dtype}")


def _check_rank_shape(source: ExecutionTraits, target: ExecutionTraits, reasons: list[str]) -> None:
    if len(source.shape) != len(target.shape):
        reasons.append(f"rank mismatch: source={len(source.shape)}, target={len(target.shape)}")
        return
    if source.shape != target.shape:
        reasons.append(f"shape mismatch: source={source.shape}, target={target.shape}")


def _check_layout(source: ExecutionTraits, target: ExecutionTraits, reasons: list[str]) -> bool:
    if target.layout_order == "strided" or source.layout_order == target.layout_order:
        return False
    reasons.append(
        f"layout order mismatch: source={source.layout_order}, target={target.layout_order}"
    )
    return True


def _check_contiguous(source: ExecutionTraits, target: ExecutionTraits, reasons: list[str]) -> bool:
    if target.contiguous_c and not source.contiguous_c:
        reasons.append("C-contiguous buffer required by target")
        return True
    if target.contiguous_f and not source.contiguous_f:
        reasons.append("F-contiguous buffer required by target")
        return True
    return False


def _check_device(
    source: ExecutionTraits,
    target: ExecutionTraits,
    allow_device_copy: bool,
    reasons: list[str],
) -> bool:
    if source.device == target.device:
        return False
    if allow_device_copy:
        reasons.append(f"device transfer required: {source.device}->{target.device}")
        return True
    reasons.append(f"device mismatch not allowed: {source.device}!={target.device}")
    return False


def slice_view_traits(source: ExecutionTraits) -> ExecutionTraits:
    """Selection/slice keeps shape-rank and usually returns a strided view."""
    return ExecutionTraits(
        dtype=source.dtype,
        shape=source.shape,
        device=source.device,
        layout_order="strided",
        contiguous_c=False,
        contiguous_f=False,
        readonly=source.readonly,
        owner=source.owner,
    )


def transpose_traits(source: ExecutionTraits, axes: tuple[int, ...]) -> ExecutionTraits:
    """Transpose returns a non-contiguous strided layout by default."""
    shape = tuple(source.shape[axis] for axis in axes)
    return ExecutionTraits(
        dtype=source.dtype,
        shape=shape,
        device=source.device,
        layout_order="strided",
        contiguous_c=False,
        contiguous_f=False,
        readonly=source.readonly,
        owner=source.owner,
    )


def reshape_restack_traits(
    source: ExecutionTraits,
    new_shape: tuple[int, ...],
    order: Literal["C", "F"] = "C",
) -> tuple[ExecutionTraits, bool]:
    """Reshape/restack preserving semantics; flag copy requirement when needed."""
    same_elements = _element_count(source.shape) == _element_count(new_shape)
    needs_copy = not same_elements or source.layout_order not in (order, "strided")
    traits = ExecutionTraits(
        dtype=source.dtype,
        shape=new_shape,
        device=source.device,
        layout_order=order if not needs_copy else order,
        contiguous_c=order == "C",
        contiguous_f=order == "F",
        readonly=source.readonly,
        owner=source.owner,
    )
    return traits, needs_copy


def concat_traits(sources: tuple[ExecutionTraits, ...], axis: int) -> ExecutionTraits:
    """Concat yields contiguous output; requires homogeneous dtype/device."""
    if not sources:
        raise ValueError("concat_traits requires at least one source")
    first = sources[0]
    if any(source.dtype != first.dtype for source in sources):
        raise ValueError("concat dtype mismatch across sources")
    if any(source.device != first.device for source in sources):
        raise ValueError("concat device mismatch across sources")
    out_shape = list(first.shape)
    out_shape[axis] = sum(source.shape[axis] for source in sources)
    return ExecutionTraits(
        dtype=first.dtype,
        shape=tuple(out_shape),
        device=first.device,
        layout_order="C",
        contiguous_c=True,
        contiguous_f=False,
        readonly=False,
        owner=first.owner,
    )


def normalize_handoff_traits(target: ExecutionTraits, owner: str) -> ExecutionTraits:
    """Normalize traits for backend handoff materialization."""
    return ExecutionTraits(
        dtype=target.dtype,
        shape=target.shape,
        device=target.device,
        layout_order="C",
        contiguous_c=True,
        contiguous_f=False,
        readonly=False,
        owner=owner,
    )


def _element_count(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
