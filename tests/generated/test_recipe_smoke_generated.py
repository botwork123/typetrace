"""Generated smoke tests from specs/recipes.yaml."""

import numpy as np
import pytest

SMOKE_CASES = [
    # recipe_id: covariance_matrix
    {
        "recipe_id": "covariance_matrix",
        "case": {"X": [[1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [5.0, 10.0]]},
        "expected": [
            [3.333333333333333, 6.666666666666666],
            [6.666666666666666, 13.333333333333332],
        ],
    },
    # recipe_id: ols_normal_equations
    {
        "recipe_id": "ols_normal_equations",
        "case": {"X": [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], "y": [1.0, 2.0, 2.0, 4.0]},
        "expected": [0.8999999999999991, 0.9000000000000004],
    },
    # recipe_id: simple_returns
    {
        "recipe_id": "simple_returns",
        "case": {"p": [100.0, 102.0, 101.0, 103.0]},
        "expected": [0.020000000000000018, -0.009803921568627416, 0.01980198019801982],
    },
]


@pytest.mark.parametrize("row", SMOKE_CASES, ids=[r["recipe_id"] for r in SMOKE_CASES])
def test_recipe_smoke_oracles(row):
    case = row["case"]
    recipe_id = row["recipe_id"]
    if recipe_id == "ols_normal_equations":
        x = np.array(case["X"])
        y = np.array(case["y"])
        actual = np.linalg.solve(x.T @ x, x.T @ y)
    elif recipe_id == "covariance_matrix":
        actual = np.cov(np.array(case["X"]), rowvar=False, ddof=1)
    elif recipe_id == "simple_returns":
        prices = np.array(case["p"])
        actual = prices[1:] / prices[:-1] - 1
    else:
        raise ValueError(f"Unknown recipe id: {recipe_id}")
    np.testing.assert_allclose(actual, np.array(row["expected"]))
