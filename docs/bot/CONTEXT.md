# Bot Context Guide — typetrace

**Purpose:** Fast orientation for bots making changes in typetrace.

---

## What typetrace is

`typetrace` is a **runtime type-descriptor and transform** library.
It describes concrete values as `TypeDesc`, and supports both:
- static transform rules (`type_transform(...)`)
- execution-based inference (`infer_by_execution(...)`) when rules are harder

---

## Core concepts

| Concept | Meaning |
|---|---|
| `TypeDesc` | Canonical runtime type descriptor |
| `kind` | Structural category (`ndarray`, `dataframe`, `series`, `dataset`, `scalar`, etc.) |
| `dtype` | Primitive value type (`float64`, `int64`, `bool`, …) |
| `dims` | Dimension map; named dims for xarray-like data and positional `dim0/dim1/...` for positional arrays |
| `columns` | DataFrame schema (`["a", "b"]` or partial form `["a", ...]`) |

---

## Key files (current)

| Area | Files |
|---|---|
| Core model + dispatch | `src/typetrace/core.py` |
| Inference engine | `src/typetrace/inference.py` |
| Transform patterns | `src/typetrace/patterns.py` |
| Adapters | `src/typetrace/adapters/*.py` |
| Runtime/layout traits | `src/typetrace/execution_traits.py`, `src/typetrace/layout_ops.py` |
| Tests | `tests/` |

---

## Working rules for bots

1. **Prefer narrow changes.** Don’t broaden API surface unless requested.
2. **No silent fallback.** Fail fast with contextual errors.
3. **Keep error messages contract-stable** when tests assert wording.
4. **Update tests with behavior changes** (especially adapter + inference contracts).
5. **Use existing conventions** (dim naming, schema semantics, kind names).

---

## Quick examples

```python
from typetrace import TypeDesc

# ndarray descriptor
x = TypeDesc(kind="ndarray", dtype="float64", dims={"instrument": 5, "time": 100})

# partial dataframe schema (known columns + trailing ellipsis)
df = TypeDesc(kind="dataframe", columns=["price", ...], dtypes={"price": "float64"})
```

---

## Common gotchas

- `TypeDesc.from_value(...)` has recursion protection (`kind="recursive"` on cycles).
- `infer_by_execution(...)` is for hard-to-encode transforms; prefer explicit rules when simple.
- Partial DataFrame schemas are represented by **trailing ellipsis in `columns`**.

---

*Last updated: 2026-04-08*
