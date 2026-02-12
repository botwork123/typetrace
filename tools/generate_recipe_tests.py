#!/usr/bin/env python3
"""Generate pytest files from specs/recipes.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

MATRIX_CASES: dict[str, list[dict[str, Any]]] = {
    "row_select": [
        {"X": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "i": 1},
        {"X": [[2.0, -1.0], [7.0, 3.5], [0.0, 4.0]], "i": 0},
    ],
    "col_select": [
        {"X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "j": 1},
        {"X": [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], "j": 0},
    ],
    "transpose_matrix": [
        {"X": [[1.0, 2.0], [3.0, 4.0]]},
        {"X": [[5.0, -2.0, 1.0]]},
    ],
    "flatten_row_major": [
        {"X": [[1.0, 2.0], [3.0, 4.0]]},
        {"X": [[0.0, -1.0, 2.0], [3.0, 4.0, 5.0]]},
    ],
    "flatten_col_major": [
        {"X": [[1.0, 2.0], [3.0, 4.0]]},
        {"X": [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]},
    ],
    "reshape_vector_to_matrix": [
        {"v": [1.0, 2.0, 3.0, 4.0], "m": 2, "n": 2},
        {"v": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], "m": 3, "n": 2},
    ],
    "restack_rows_to_columns": [
        {"X": [[1.0, 2.0], [3.0, 4.0]]},
        {"X": [[2.0, 0.0, -2.0], [1.0, 3.0, 5.0]]},
    ],
}

SMOKE_CASES: dict[str, dict[str, Any]] = {
    "ols_normal_equations": {
        "X": [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
        "y": [1.0, 2.0, 2.0, 4.0],
    },
    "covariance_matrix": {"X": [[1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [5.0, 10.0]]},
    "simple_returns": {"p": [100.0, 102.0, 101.0, 103.0]},
}


def load_recipes(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return sorted(data["recipes"], key=lambda recipe: recipe["id"])


def as_numpy_env(case: dict[str, Any]) -> dict[str, Any]:
    env: dict[str, Any] = {"np": np}
    for key, value in case.items():
        env[key] = np.array(value) if isinstance(value, list) else value
    matrix = env.get("X")
    if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
        env.setdefault("m", matrix.shape[0])
        env.setdefault("n", matrix.shape[1])
    return env


def evaluate_oracle(expr: str, case: dict[str, Any]) -> Any:
    env = as_numpy_env(case)
    lines = [line.strip() for line in expr.split(";") if line.strip()]
    for line in lines[:-1]:
        exec(line, {}, env)
    return eval(lines[-1], {}, env)


def normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_rows(
    recipes: dict[str, dict[str, Any]],
    cases_by_recipe: dict[str, Any],
    expand_lists: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for recipe_id in sorted(cases_by_recipe):
        oracle = recipes[recipe_id]["oracle"]["numpy"]
        cases = cases_by_recipe[recipe_id]
        if not expand_lists:
            cases = [cases]
        for case in cases:
            rows.append(
                {
                    "recipe_id": recipe_id,
                    "case": case,
                    "expected": normalize(evaluate_oracle(oracle, case)),
                }
            )
    return rows


def render_case_block(rows: list[dict[str, Any]], const_name: str) -> str:
    lines = [f"{const_name} = ["]
    for row in rows:
        recipe_id = row["recipe_id"]
        payload = repr(row)
        lines.append(f"    # recipe_id: {recipe_id}")
        lines.append(f"    {payload},")
    lines.append("]")
    return "\n".join(lines)


def render_matrix_test(rows: list[dict[str, Any]]) -> str:
    cases = render_case_block(rows, "MATRIX_OP_CASES")
    return (
        '"""Generated tests for matrix_ops recipes from specs/recipes.yaml."""\n\n'
        "import numpy as np\nimport pytest\n\n"
        f"{cases}\n\n"
        '@pytest.mark.parametrize("row", MATRIX_OP_CASES, ids=[r["recipe_id"] for r in MATRIX_OP_CASES])\n'
        "def test_matrix_ops_oracles(row):\n"
        '    case = row["case"]\n'
        '    recipe_id = row["recipe_id"]\n'
        '    if recipe_id == "row_select":\n'
        '        actual = np.array(case["X"])[case["i"], :]\n'
        '    elif recipe_id == "col_select":\n'
        '        actual = np.array(case["X"])[:, case["j"]]\n'
        '    elif recipe_id == "transpose_matrix":\n'
        '        actual = np.array(case["X"]).T\n'
        '    elif recipe_id == "flatten_row_major":\n'
        '        actual = np.array(case["X"]).reshape(-1, order="C")\n'
        '    elif recipe_id == "flatten_col_major":\n'
        '        actual = np.array(case["X"]).reshape(-1, order="F")\n'
        '    elif recipe_id == "reshape_vector_to_matrix":\n'
        '        actual = np.array(case["v"]).reshape(case["m"], case["n"])\n'
        '    elif recipe_id == "restack_rows_to_columns":\n'
        '        matrix = np.array(case["X"])\n'
        "        actual = matrix.T.reshape(matrix.shape[1], matrix.shape[0])\n"
        "    else:\n"
        '        raise ValueError(f"Unknown recipe id: {recipe_id}")\n'
        '    np.testing.assert_allclose(actual, np.array(row["expected"]))\n'
    )


def render_smoke_test(rows: list[dict[str, Any]]) -> str:
    cases = render_case_block(rows, "SMOKE_CASES")
    return (
        '"""Generated smoke tests from specs/recipes.yaml."""\n\n'
        "import numpy as np\nimport pytest\n\n"
        f"{cases}\n\n"
        '@pytest.mark.parametrize("row", SMOKE_CASES, ids=[r["recipe_id"] for r in SMOKE_CASES])\n'
        "def test_recipe_smoke_oracles(row):\n"
        '    case = row["case"]\n'
        '    recipe_id = row["recipe_id"]\n'
        '    if recipe_id == "ols_normal_equations":\n'
        '        x = np.array(case["X"])\n'
        '        y = np.array(case["y"])\n'
        "        actual = np.linalg.solve(x.T @ x, x.T @ y)\n"
        '    elif recipe_id == "covariance_matrix":\n'
        '        actual = np.cov(np.array(case["X"]), rowvar=False, ddof=1)\n'
        '    elif recipe_id == "simple_returns":\n'
        '        prices = np.array(case["p"])\n'
        "        actual = prices[1:] / prices[:-1] - 1\n"
        "    else:\n"
        '        raise ValueError(f"Unknown recipe id: {recipe_id}")\n'
        '    np.testing.assert_allclose(actual, np.array(row["expected"]))\n'
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    recipes = {
        recipe["id"]: recipe for recipe in load_recipes(repo_root / "specs" / "recipes.yaml")
    }
    matrix_rows = build_rows(recipes, MATRIX_CASES, expand_lists=True)
    smoke_rows = build_rows(recipes, SMOKE_CASES, expand_lists=False)
    tests_dir = repo_root / "tests" / "generated"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_recipe_matrix_ops_generated.py").write_text(
        render_matrix_test(matrix_rows), encoding="utf-8"
    )
    (tests_dir / "test_recipe_smoke_generated.py").write_text(
        render_smoke_test(smoke_rows), encoding="utf-8"
    )
    print(f"Generated matrix_ops cases: {len(matrix_rows)}")
    print(f"Generated cross-category smoke cases: {len(smoke_rows)}")


if __name__ == "__main__":
    main()
