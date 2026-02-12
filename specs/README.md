# Recipe Spec Pack

This directory contains machine-readable recipe specs for matrix-cookbook-style coverage.

## File

- `recipes.yaml`: canonical recipe registry for test generation.

## Schema

Top-level fields:

- `version` (int): schema version.
- `primitive_vocabulary` (list[str]): allowed DAG op names; bots should treat this as the reusable primitive set.
- `recipes` (list[object]): recipe entries.

Each recipe entry must include:

- `id` (snake_case, stable): unique recipe identifier.
- `category`: one of broad domains such as `matrix_ops`, `fitting`, `statistics`, `quant_finance`.
- `summary`: short natural-language intent.
- `inputs`: ordered input contracts (`name`, symbolic `shape`, `dtype`).
- `preconditions`: list of symbolic constraints that define valid domains.
- `dag`: sequence of primitive operation steps (`op`, `inputs`, `output`, optional `params`).
- `expected_outputs`: output contracts (`name`, symbolic `shape`, `dtype`).
- `invariants`: algebraic or numerical properties to assert in tests.
- `oracle`: reference mapping to NumPy/SciPy expressions.

## Bot/Test Consumption Contract

1. Parse YAML.
2. Validate every `dag[].op` appears in `primitive_vocabulary`.
3. Materialize symbolic shapes with randomized positive integers satisfying each recipe `preconditions`.
4. Generate typed random inputs matching `inputs`.
5. Execute DAG against the backend under test.
6. Evaluate `invariants` and compare with `oracle` outputs (NumPy/SciPy baseline) within tolerance.
7. Report per-recipe pass/fail with failing invariant/operator details.

## Notes

- Keep `id` values stable to preserve historical test tracking.
- Additive extensions should be backward-compatible: append new recipes instead of renaming existing ids.
