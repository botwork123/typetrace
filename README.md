# typetrace

Runtime type inference by execution for array libraries.

## Overview

typetrace provides a universal type descriptor (`TypeDesc`) that can represent types across different array and dataframe libraries:

- **xarray**: DataArray with named dimensions
- **pandas**: DataFrame and Series
- **DrJit**: JIT-compiled arrays and tensors
- **Polars**: DataFrames
- **Arrow**: Columnar tables

The key feature is **inference by execution**: for complex operations where encoding type transform logic is difficult, you can run the operation on sample data and observe the output type.

## Installation

```bash
pip install typetrace

# With specific adapters
pip install typetrace[xarray]
pip install typetrace[pandas]
pip install typetrace[all]
```

## Quick Start

```python
from typetrace import TypeDesc, Symbol, infer_types

# Create a type descriptor
t = TypeDesc(
    kind='ndarray',
    dims={'symbol': Symbol('N'), 'time': Symbol('T')},
    dtype='float64'
)

# Extract type from a value
import xarray as xr
import numpy as np

data = xr.DataArray(np.zeros((100, 50)), dims=['symbol', 'time'])
t = TypeDesc.from_value(data)
print(t.dims)  # {'symbol': 100, 'time': 50}

# Create sample for inference
sample = t.make_sample()  # Zero-sized array with same structure
```

## Dimension Patterns

Common patterns for type transforms:

```python
from typetrace import unify, broadcast, add_dim, reduce_dim

# Element-wise ops: dims must match
result_dims = unify(a.dims, b.dims)

# Broadcasting: union of dims
result_dims = broadcast(a.dims, b.dims)

# Add dimension (e.g., rolling window)
result_dims = add_dim(a.dims, 'window', 10)

# Remove dimension (e.g., aggregation)
result_dims = reduce_dim(a.dims, 'time')
```

## Inference by Execution

For complex operations like `pd.merge`:

```python
from typetrace import TypeDesc
from typetrace.inference import infer_by_execution
import pandas as pd

left_t = TypeDesc(
    kind='dataframe',
    columns=['id', 'value', ...],  # ellipsis => known columns + unknown extras
    dtypes={'id': 'int64', 'value': 'float64'},
)
right_t = TypeDesc.from_value(right_df)

# Let pandas figure out the output type
result_t = infer_by_execution(pd.merge, left_t, right_t, how='left', on='id')
```

## Integration with DAG Systems

typetrace is designed to work with DAG-based computation systems. Nodes can implement `type_transform`:

```python
class MyNode:
    def type_transform(self, *input_types: TypeDesc) -> TypeDesc:
        # Return output type based on inputs
        return unify(input_types[0], input_types[1])
```

Then use `infer_types` to walk the DAG:

```python
from typetrace import infer_types, TypeContext

context = TypeContext(bindings={'N': 1000})
output_type = infer_types(root_node, context)
```

### Shape-contract requirement helper

Downstream systems like dag-core can delegate shape/schema contract checks to typetrace:

```python
from typetrace import TypeDesc, requires_shape_contract

def node_requires_contract(output_type: TypeDesc) -> bool:
    return requires_shape_contract(output_type)
```

`requires_shape_contract` returns `True` for shape/schema-bearing kinds
(`ndarray`, `dataset`, `dataframe`, `series`, `columnar`, `drjit`) and `False`
for non-shape kinds (`scalar`, `class`, `recursive`).

## Agent contract

`AGENTS.md` is the canonical implementation contract for architecture, testing, and quality gates.
All model-specific instruction files (`CODEX.md`, `CLAUDE.md`, `GEMINI.md`, `AI.md`, `COPILOT.md`, `GROK.md`) defer to it via symlink to prevent drift.

## Development checks

```bash
# Local parity with configured hooks (lint, format, type, tests, coverage)
pre-commit run --all-files

# CI-style explicit commands (same scope as workflows)
ruff check .
ruff format --check .
mypy src/typetrace
pytest --cov=src/typetrace --cov-report=term-missing --cov-fail-under=95 tests/
```

`pre-commit` is the recommended single command locally; direct commands are
kept for explicit CI parity and targeted debugging.

## License

MIT
