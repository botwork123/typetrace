# Coding Standards

## Testing

### Use parametrize for multiple cases

```python
# ✅ Good
@pytest.mark.parametrize("a,b,expected", [
    ({'x': 1}, {'x': 1}, {'x': 1}),
    ({'x': 1}, {'y': 2}, {'x': 1, 'y': 2}),
    ({}, {'x': 1}, {'x': 1}),
])
def test_unify(a, b, expected):
    assert unify(a, b) == expected

# ❌ Bad - separate test functions
def test_unify_same():
    assert unify({'x': 1}, {'x': 1}) == {'x': 1}

def test_unify_different():
    assert unify({'x': 1}, {'y': 2}) == {'x': 1, 'y': 2}
```

### Assert actual values

```python
# ✅ Good
assert result.dims == {'symbol': 100, 'time': 50}
assert result.dtype == 'float64'

# ❌ Bad
assert result.dims is not None
assert isinstance(result, TypeDesc)
```

### Test edge cases via parametrize

```python
@pytest.mark.parametrize("dims,name,expected", [
    ({'x': 1}, 'x', {}),           # Remove existing
    ({'x': 1, 'y': 2}, 'x', {'y': 2}),  # Remove one of many
    ({}, 'x', {}),                  # Remove from empty
    ({'x': 1}, 'y', {'x': 1}),     # Remove non-existent
])
def test_reduce_dim(dims, name, expected):
    assert reduce_dim(dims, name) == expected
```

## Code Style

### Function size
- Maximum 30 lines per function
- Extract helpers for complex logic

### Type hints
- Required on all public functions
- Use `Any` sparingly

### Imports
- Group: stdlib, third-party, local
- Lazy import optional dependencies in adapters

### Naming
- Descriptive names, no abbreviations
- `type_desc` not `td`
- `symbol_name` not `sym`

## Pre-commit

Run before every commit:
```bash
pre-commit run --all-files
```

Hooks:
- ruff (lint + format)
- trailing whitespace
- pytest
- coverage check (95% minimum)
