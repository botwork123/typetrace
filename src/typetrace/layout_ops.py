"""Layout compatibility and transition helpers for runtime handoff boundaries."""

from typing import Literal

from typetrace.execution_traits import CompatibilityResult, ExecutionTraits


def check_handoff_compatibility(
    source: ExecutionTraits,
    target: ExecutionTraits,
    allow_device_copy: bool = False,
) -> CompatibilityResult:
    """Check runtime compatibility for evaluator/backend handoffs."""
    hard_reasons: list[str] = []
    copy_reasons: list[str] = []
    requires_copy = False
    _check_dtype(source, target, hard_reasons)
    _check_rank_shape(source, target, hard_reasons)
    requires_copy = _check_layout(source, target, hard_reasons, copy_reasons) or requires_copy
    requires_copy = _check_contiguous(source, target, hard_reasons, copy_reasons) or requires_copy
    requires_copy = (
        _check_device(
            source,
            target,
            allow_device_copy,
            hard_reasons,
            copy_reasons,
        )
        or requires_copy
    )
    return CompatibilityResult(
        compatible=not hard_reasons,
        requires_copy=requires_copy,
        reasons=tuple([*hard_reasons, *copy_reasons]),
    )


def _check_dtype(source: ExecutionTraits, target: ExecutionTraits, reasons: list[str]) -> None:
    if source.dtype != target.dtype:
        reasons.append(f"dtype mismatch: source={source.dtype}, target={target.dtype}")


def _check_rank_shape(source: ExecutionTraits, target: ExecutionTraits, reasons: list[str]) -> None:
    if len(source.shape) != len(target.shape):
        reasons.append(f"rank mismatch: source={len(source.shape)}, target={len(target.shape)}")
        return
    if source.shape != target.shape:
        reasons.append(f"shape mismatch: source={source.shape}, target={target.shape}")


def _check_layout(
    source: ExecutionTraits,
    target: ExecutionTraits,
    hard_reasons: list[str],
    copy_reasons: list[str],
) -> bool:
    if target.layout_order == "strided" or source.layout_order == target.layout_order:
        return False
    message = f"layout order mismatch: source={source.layout_order}, target={target.layout_order}"
    _record_copy_or_hard_requirement(
        source.device == target.device, message, hard_reasons, copy_reasons
    )
    return source.device == target.device


def _check_contiguous(
    source: ExecutionTraits,
    target: ExecutionTraits,
    hard_reasons: list[str],
    copy_reasons: list[str],
) -> bool:
    if target.contiguous_c and not source.contiguous_c:
        _record_copy_or_hard_requirement(
            source.device == target.device,
            "C-contiguous buffer required by target",
            hard_reasons,
            copy_reasons,
        )
        return source.device == target.device
    if target.contiguous_f and not source.contiguous_f:
        _record_copy_or_hard_requirement(
            source.device == target.device,
            "F-contiguous buffer required by target",
            hard_reasons,
            copy_reasons,
        )
        return source.device == target.device
    return False


def _check_device(
    source: ExecutionTraits,
    target: ExecutionTraits,
    allow_device_copy: bool,
    hard_reasons: list[str],
    copy_reasons: list[str],
) -> bool:
    if source.device == target.device:
        return False
    if allow_device_copy:
        copy_reasons.append(f"device transfer required: {source.device}->{target.device}")
        return True
    hard_reasons.append(f"device mismatch not allowed: {source.device}!={target.device}")
    return False


def _record_copy_or_hard_requirement(
    same_device: bool,
    message: str,
    hard_reasons: list[str],
    copy_reasons: list[str],
) -> None:
    if same_device:
        copy_reasons.append(message)
    else:
        hard_reasons.append(message)


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
    _validate_transpose_axes(source.shape, axes)
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
    """Reshape/restack target traits with a conservative copy-vs-view heuristic.

    Returns `(target_traits, needs_copy)` where `needs_copy=False` means the
    request may be representable as a view (not guaranteed for every backend).
    """
    same_elements = _element_count(source.shape) == _element_count(new_shape)
    can_view = same_elements and _reshape_view_possible(source, order)
    traits = ExecutionTraits(
        dtype=source.dtype,
        shape=new_shape,
        device=source.device,
        layout_order=order,
        contiguous_c=order == "C",
        contiguous_f=order == "F",
        readonly=source.readonly,
        owner=source.owner,
    )
    return traits, not can_view


def concat_traits(sources: tuple[ExecutionTraits, ...], axis: int) -> ExecutionTraits:
    """Concat yields contiguous output; requires homogeneous dtype/device."""
    if not sources:
        raise ValueError("concat_traits requires at least one source")
    first = sources[0]
    if any(source.dtype != first.dtype for source in sources):
        raise ValueError("concat dtype mismatch across sources")
    if any(source.device != first.device for source in sources):
        raise ValueError("concat device mismatch across sources")
    normalized_axis = _normalize_axis(axis, len(first.shape))
    _validate_concat_shapes(sources, normalized_axis)
    out_shape = list(first.shape)
    out_shape[normalized_axis] = sum(source.shape[normalized_axis] for source in sources)
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


def _validate_transpose_axes(shape: tuple[int, ...], axes: tuple[int, ...]) -> None:
    rank = len(shape)
    if len(axes) != rank:
        raise ValueError(f"transpose axes must have length {rank}, got {len(axes)}")
    normalized = tuple(_normalize_axis(axis, rank) for axis in axes)
    expected = tuple(range(rank))
    if tuple(sorted(normalized)) != expected:
        raise ValueError(f"transpose axes must be a permutation of {expected}, got {axes}")


def _reshape_view_possible(source: ExecutionTraits, order: Literal["C", "F"]) -> bool:
    if source.layout_order == "strided":
        return False
    if order == "C":
        return source.contiguous_c
    return source.contiguous_f


def _normalize_axis(axis: int, rank: int) -> int:
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"axis {axis} out of bounds for rank {rank}")
    return normalized


def _validate_concat_shapes(sources: tuple[ExecutionTraits, ...], axis: int) -> None:
    rank = len(sources[0].shape)
    if any(len(source.shape) != rank for source in sources):
        raise ValueError("concat rank mismatch across sources")
    for dim in range(rank):
        if dim == axis:
            continue
        expected = sources[0].shape[dim]
        if any(source.shape[dim] != expected for source in sources):
            raise ValueError(f"concat non-axis dimension mismatch at dim {dim}")


def _element_count(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
