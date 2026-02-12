# Memory Layout and Runtime Contract Specification

This document defines the split between semantic typing and runtime execution
requirements in typetrace.

## Contract split

- `TypeDesc` remains the semantic/math contract (kind, dims/shape, dtype, schema).
- `ExecutionTraits` is runtime-only (dtype, shape, device, layout, contiguity, owner).

`TypeDesc` must not include backend/runtime memory fields.

## `ExecutionTraits`

Implemented in `typetrace.execution_traits`:

```python
ExecutionTraits(
    dtype: str,
    shape: tuple[int, ...],
    device: Literal["cpu", "cuda"] = "cpu",
    layout_order: Literal["C", "F", "strided"] = "strided",
    contiguous_c: bool = False,
    contiguous_f: bool = False,
    readonly: bool = False,
    owner: str = "unknown",
)
```

Validation rules:

- All shape dimensions are `>= 0`
- `layout_order="C"` requires `contiguous_c=True`
- `layout_order="F"` requires `contiguous_f=True`

## Compatibility checks

Use `check_handoff_compatibility(source, target, allow_device_copy=False)`.

Checks with explicit failure reasons:

- dtype exact match
- rank + shape match
- layout-order expectations
- contiguity expectations
- device policy (`cpu`/`cuda`)

When `allow_device_copy=True`, cross-device handoff is marked as
`requires_copy=True` with an explicit transfer reason.

## Transition helpers

Implemented in `typetrace.layout_ops`:

- `slice_view_traits` → strided view traits
- `transpose_traits` → reordered shape + strided traits
- `reshape_restack_traits` → target shape + copy-required flag
- `concat_traits` → validated homogeneous concat output traits
- `normalize_handoff_traits` → explicit contiguous C materialized traits

## Adapter extraction

`infer_execution_traits(value)` supports:

- NumPy arrays
- xarray objects (via `.values`)
- pandas objects (via `.to_numpy()`)
- torch tensors (best-effort runtime traits)
- drjit arrays/tensors (best-effort runtime traits)

## Inference boundary enforcement

`infer_by_execution(..., expected_output_traits=..., allow_device_copy=...)`
performs an optional runtime handoff check after execution and raises:

- `ValueError("Execution traits handoff check failed: ...")`

when compatibility fails.

This keeps inference-by-execution semantic behavior unchanged while adding
strict runtime boundary validation when requested.
