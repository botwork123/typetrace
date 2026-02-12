# AGENTS.md — typetrace Agent Contract (Canonical)

This file is the **single source of truth for agent behavior in this repo**.

`CODEX.md`, `CLAUDE.md`, `GEMINI.md`, `AI.md`, `COPILOT.md`, and `GROK.md` should mirror or defer to this file, not diverge from it.

---

## 1) Project Mission

`typetrace` provides shape/type inference for DAG-style compute systems across heterogeneous backends.

Core objective:
- Keep **math semantics stable** while allowing **runtime evaluator swap** (Numba, DrJit, Torch, etc.).

---

## 2) Core Architecture Principles (Non-Negotiable)

1. **Type vs Runtime split is mandatory**
   - `TypeDesc` = semantic/math contract (dims, shape, dtype, meaning)
   - `ExecutionTraits` = runtime/memory contract (device, contiguity, layout, strides, ownership)

2. **No hidden copies in hot paths**
   - Any materialization/re-layout must be explicit and measurable.

3. **DAG math is evaluator-agnostic**
   - Changing evaluator must not require rewriting semantic graph logic.

4. **Defensive checks only at boundaries**
   - Validate hard at adapter/handoff boundaries.
   - Keep internals clean and composable.

5. **Prefer compositional FP-style transforms**
   - Pure helpers, small functions, explicit inputs/outputs.
   - Avoid long branching ladders where a registry/table works.

---

## 3) Coding Rules for Agents

Use these rules in every implementation:

- Functions should generally be **<30 lines**.
- Use explicit type hints on public functions.
- Prefer `dataclasses.replace` over field-by-field copy constructors when updating immutable dataclasses.
- Avoid duplicate dtype/dispatch logic across modules; centralize in shared helpers/registries.
- Keep adapters thin; avoid pushing backend policy into `core.py`.
- If behavior is heuristic (not exact), name/document it clearly.

When adding runtime policy:
- Separate **hard incompatibility** from **copy-required-but-allowed** conditions.
- Keep failure reasons explicit and testable.

---

## 4) Test Rules for Agents

Required:
- `@pytest.mark.parametrize` for multi-case behavior.
- Assert concrete values/messages (not only type existence).
- Add both positive and negative path tests for compatibility checks.
- Keep tests deterministic.

Avoid:
- Copy-paste test scaffolding and repeated import/setup blocks.
- Large `if/elif` ladders in generated tests when dict/table dispatch can be used.

Coverage target:
- Maintain repo coverage gate and avoid lowering thresholds as a shortcut.

---

## 5) Quality Gates (Must Pass Before “Done”)

Run and report:

```bash
pre-commit run --all-files
pytest -q
```

CI-parity commands (for debugging and explicit reporting):

```bash
ruff check .
ruff format --check .
mypy src/typetrace
pytest --cov=src/typetrace --cov-report=term-missing --cov-fail-under=95 tests/
```

If there is a scope mismatch (e.g., mypy on tests), explicitly call it out in report.

---

## 6) Review Workflow (Mandatory)

For non-trivial changes:
1. Implementation pass
2. Independent review pass (separate bot/session)
3. Fix pass for review findings

No “production-ready” claim without an independent review summary.

---

## 7) Repository Map (Working Areas)

- `src/typetrace/core.py` — semantic type model
- `src/typetrace/inference.py` — DAG type inference and execution inference
- `src/typetrace/execution_traits.py` — runtime/memory contract model
- `src/typetrace/layout_ops.py` — handoff compatibility and layout transitions
- `src/typetrace/adapters/` — backend extraction/sample generation
- `specs/recipes.yaml` — machine-readable recipe specs
- `tools/generate_recipe_tests.py` — recipe-to-test generator
- `tests/` — contracts, adapters, inference, generated recipe tests
- `docs/memory_layout_spec.md` — runtime layout policy reference

---

## 8) Priority Direction (Current)

Primary roadmap emphasis:
1. Numba + DrJit core interoperability
2. Memory layout correctness and explicit handoff semantics
3. Deduplicate adapter/test logic and increase composability
4. Preserve evaluator-swappable DAG semantics

---

## 9) When Updating This File

If you change this contract, include in your PR/commit summary:
- What principle changed
- Why it changed
- Which files/processes must now follow the new rule
