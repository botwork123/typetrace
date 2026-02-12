"""Generated tests for matrix_ops recipes from specs/recipes.yaml."""

import numpy as np
import pytest

MATRIX_OP_CASES = [
    # recipe_id: col_select
    {
        "recipe_id": "col_select",
        "case": {"X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "j": 1},
        "expected": [2.0, 4.0, 6.0],
    },
    # recipe_id: col_select
    {
        "recipe_id": "col_select",
        "case": {"X": [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], "j": 0},
        "expected": [9.0, 6.0],
    },
    # recipe_id: flatten_col_major
    {
        "recipe_id": "flatten_col_major",
        "case": {"X": [[1.0, 2.0], [3.0, 4.0]]},
        "expected": [1.0, 3.0, 2.0, 4.0],
    },
    # recipe_id: flatten_col_major
    {
        "recipe_id": "flatten_col_major",
        "case": {"X": [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]},
        "expected": [7.0, 1.0, 8.0, 2.0, 9.0, 3.0],
    },
    # recipe_id: flatten_row_major
    {
        "recipe_id": "flatten_row_major",
        "case": {"X": [[1.0, 2.0], [3.0, 4.0]]},
        "expected": [1.0, 2.0, 3.0, 4.0],
    },
    # recipe_id: flatten_row_major
    {
        "recipe_id": "flatten_row_major",
        "case": {"X": [[0.0, -1.0, 2.0], [3.0, 4.0, 5.0]]},
        "expected": [0.0, -1.0, 2.0, 3.0, 4.0, 5.0],
    },
    # recipe_id: reshape_vector_to_matrix
    {
        "recipe_id": "reshape_vector_to_matrix",
        "case": {"v": [1.0, 2.0, 3.0, 4.0], "m": 2, "n": 2},
        "expected": [[1.0, 2.0], [3.0, 4.0]],
    },
    # recipe_id: reshape_vector_to_matrix
    {
        "recipe_id": "reshape_vector_to_matrix",
        "case": {"v": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], "m": 3, "n": 2},
        "expected": [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    },
    # recipe_id: restack_rows_to_columns
    {
        "recipe_id": "restack_rows_to_columns",
        "case": {"X": [[1.0, 2.0], [3.0, 4.0]]},
        "expected": [[1.0, 3.0], [2.0, 4.0]],
    },
    # recipe_id: restack_rows_to_columns
    {
        "recipe_id": "restack_rows_to_columns",
        "case": {"X": [[2.0, 0.0, -2.0], [1.0, 3.0, 5.0]]},
        "expected": [[2.0, 1.0], [0.0, 3.0], [-2.0, 5.0]],
    },
    # recipe_id: row_select
    {
        "recipe_id": "row_select",
        "case": {"X": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "i": 1},
        "expected": [4.0, 5.0, 6.0],
    },
    # recipe_id: row_select
    {
        "recipe_id": "row_select",
        "case": {"X": [[2.0, -1.0], [7.0, 3.5], [0.0, 4.0]], "i": 0},
        "expected": [2.0, -1.0],
    },
    # recipe_id: transpose_matrix
    {
        "recipe_id": "transpose_matrix",
        "case": {"X": [[1.0, 2.0], [3.0, 4.0]]},
        "expected": [[1.0, 3.0], [2.0, 4.0]],
    },
    # recipe_id: transpose_matrix
    {
        "recipe_id": "transpose_matrix",
        "case": {"X": [[5.0, -2.0, 1.0]]},
        "expected": [[5.0], [-2.0], [1.0]],
    },
]


@pytest.mark.parametrize("row", MATRIX_OP_CASES, ids=[r["recipe_id"] for r in MATRIX_OP_CASES])
def test_matrix_ops_oracles(row):
    case = row["case"]
    recipe_id = row["recipe_id"]
    if recipe_id == "row_select":
        actual = np.array(case["X"])[case["i"], :]
    elif recipe_id == "col_select":
        actual = np.array(case["X"])[:, case["j"]]
    elif recipe_id == "transpose_matrix":
        actual = np.array(case["X"]).T
    elif recipe_id == "flatten_row_major":
        actual = np.array(case["X"]).reshape(-1, order="C")
    elif recipe_id == "flatten_col_major":
        actual = np.array(case["X"]).reshape(-1, order="F")
    elif recipe_id == "reshape_vector_to_matrix":
        actual = np.array(case["v"]).reshape(case["m"], case["n"])
    elif recipe_id == "restack_rows_to_columns":
        matrix = np.array(case["X"])
        actual = matrix.T.reshape(matrix.shape[1], matrix.shape[0])
    else:
        raise ValueError(f"Unknown recipe id: {recipe_id}")
    np.testing.assert_allclose(actual, np.array(row["expected"]))
