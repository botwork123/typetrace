# Bot Context Guide — typetrace

**Purpose:** Help bots understand typetrace before implementing changes.

---

## What is typetrace?

Runtime type description library. Like TypeScript types but at runtime.

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **TypeDesc** | Type descriptor with kind, dtype, dims |
| **kind** | "ndarray", "class", "dataarray", etc. |
| **dtype** | "float64", "int64", "bool", etc. |
| **dims** | Named dimensions with sizes |

---

## Key Files

| Area | Files |
|------|-------|
| TypeDesc | `src/typetrace/type_desc.py` |
| Inference | `src/typetrace/inference.py` |
| Tests | `tests/` |

---

## Usage

```python
from typetrace import TypeDesc

# Create type descriptor
td = TypeDesc(kind="ndarray", dtype="float64", dims={"instrument": 5})

# Check properties
td.dims  # {"instrument": 5}
td.dtype  # "float64"
```

---

## Gotchas

- `from_value()` has cycle detection — returns `TypeDesc(kind="recursive")` for cycles
- dims are **named** (like xarray), not positional (like numpy)

---

*Last updated: 2026-03-18*
