# AGENTS.md - typetrace

## Project Overview

typetrace is a runtime type inference library for array/dataframe libraries.

**Core concepts:**
- `TypeDesc`: Universal type descriptor (xarray, pandas, DrJit, etc.)
- `Symbol`: Symbolic dimension bound at runtime
- `infer_types`: Walk DAG and compute types via type_transform
- `infer_by_execution`: Run function on samples to discover output type

## Repository Structure

```
typetrace/
├── src/typetrace/
│   ├── __init__.py      # Public API exports
│   ├── core.py          # TypeDesc, Symbol, Dims
│   ├── inference.py     # infer_types, TypeContext
│   ├── patterns.py      # unify, broadcast, add_dim, reduce_dim
│   └── adapters/        # Library-specific adapters
│       ├── xarray.py
│       ├── pandas.py
│       ├── drjit.py
│       ├── polars.py
│       └── arrow.py
└── tests/
```

## Coding Standards

See CODING_STANDARDS.md for:
- Test patterns (parametrize, assert values)
- Function size limits (<30 lines)
- Type hints required
- Pre-commit hooks

## Key Design Decisions

1. **TypeDesc is frozen/immutable** - use with_* methods for modifications
2. **Adapters are lazy-loaded** - no hard dependencies on xarray, pandas, etc.
3. **Symbols vs ints** - dims can be symbolic until bound
4. **make_sample creates minimal data** - zero-sized arrays for inference

## Testing Requirements

- 95% code coverage minimum
- Use `@pytest.mark.parametrize` for multiple cases
- Assert actual expected values, not just types
- Integration tests for each adapter (with optional deps)

## Related Projects

- dag-graph: Uses typetrace for DAG type inference
- drjit-feature-pipeline: Codegen target

## Common Tasks

### Adding a new adapter

1. Create `src/typetrace/adapters/newlib.py`
2. Implement `from_newlib(value) -> TypeDesc`
3. Implement `make_newlib_sample(type_desc) -> value`
4. Update `TypeDesc.from_value()` dispatch in core.py
5. Add tests in `tests/test_adapters_newlib.py`
6. Add optional dependency in pyproject.toml

### Adding a new pattern

1. Add function to `patterns.py`
2. Export in `__init__.py`
3. Add tests in `tests/test_patterns.py`
