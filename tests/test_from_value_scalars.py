"""Tests for TypeDesc.from_value handling of Python scalars and numpy arrays."""

from importlib.util import find_spec

import pytest

from typetrace.core import TypeDesc


class TestFromValuePythonScalars:
    """Tests for Python scalar handling in from_value."""

    @pytest.mark.parametrize(
        "value,expected_dtype",
        [
            (42, "int64"),
            (-1, "int64"),
            (0, "int64"),
            (2**60, "int64"),  # Large int
        ],
    )
    def test_from_value_int(self, value: int, expected_dtype: str) -> None:
        """from_value handles Python int → scalar with dtype int64."""
        result = TypeDesc.from_value(value)

        assert result.kind == "scalar"
        assert result.dtype == expected_dtype
        assert result.dims is None

    @pytest.mark.parametrize(
        "value,expected_dtype",
        [
            (3.14, "float64"),
            (-0.001, "float64"),
            (0.0, "float64"),
            (float("inf"), "float64"),
            (float("-inf"), "float64"),
        ],
    )
    def test_from_value_float(self, value: float, expected_dtype: str) -> None:
        """from_value handles Python float → scalar with dtype float64."""
        result = TypeDesc.from_value(value)

        assert result.kind == "scalar"
        assert result.dtype == expected_dtype
        assert result.dims is None

    @pytest.mark.parametrize(
        "value,expected_dtype",
        [
            ("hello", "str"),
            ("", "str"),
            ("日本語", "str"),  # Unicode
        ],
    )
    def test_from_value_str(self, value: str, expected_dtype: str) -> None:
        """from_value handles Python str → scalar with dtype str."""
        result = TypeDesc.from_value(value)

        assert result.kind == "scalar"
        assert result.dtype == expected_dtype
        assert result.dims is None

    @pytest.mark.parametrize(
        "value,expected_dtype",
        [
            (True, "bool"),
            (False, "bool"),
        ],
    )
    def test_from_value_bool(self, value: bool, expected_dtype: str) -> None:
        """from_value handles Python bool → scalar with dtype bool."""
        result = TypeDesc.from_value(value)

        assert result.kind == "scalar"
        assert result.dtype == expected_dtype
        assert result.dims is None

    def test_bool_not_treated_as_int(self) -> None:
        """bool is handled as bool, not int (despite being subclass)."""
        result_true = TypeDesc.from_value(True)
        result_false = TypeDesc.from_value(False)
        result_int = TypeDesc.from_value(1)

        assert result_true.dtype == "bool"
        assert result_false.dtype == "bool"
        assert result_int.dtype == "int64"


numpy_required = pytest.mark.skipif(find_spec("numpy") is None, reason="numpy not installed")


@numpy_required
class TestFromValueNumpyArrays:
    """Tests for numpy array handling in from_value."""

    @pytest.mark.parametrize(
        "shape,dtype_str,expected_dims",
        [
            ((10,), "float64", {"dim0": 10}),
            ((5, 3), "float32", {"dim0": 5, "dim1": 3}),
            ((2, 3, 4), "int64", {"dim0": 2, "dim1": 3, "dim2": 4}),
        ],
    )
    def test_from_value_ndarray_shapes(
        self, shape: tuple, dtype_str: str, expected_dims: dict
    ) -> None:
        """from_value handles numpy arrays with various shapes."""
        import numpy as np

        arr = np.zeros(shape, dtype=dtype_str)
        result = TypeDesc.from_value(arr)

        assert result.kind == "ndarray"
        assert result.dtype == dtype_str
        assert result.dims == expected_dims

    @pytest.mark.parametrize(
        "dtype_str",
        [
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "bool",
            "complex64",
            "complex128",
        ],
    )
    def test_from_value_ndarray_dtypes(self, dtype_str: str) -> None:
        """from_value preserves numpy dtype as string."""
        import numpy as np

        arr = np.zeros((3,), dtype=dtype_str)
        result = TypeDesc.from_value(arr)

        assert result.kind == "ndarray"
        assert result.dtype == dtype_str

    def test_from_value_0d_array(self) -> None:
        """from_value handles 0-dimensional numpy arrays (scalars wrapped as array)."""
        import numpy as np

        arr = np.array(42.0)
        assert arr.ndim == 0
        result = TypeDesc.from_value(arr)

        assert result.kind == "ndarray"
        assert result.dtype == "float64"
        assert result.dims == {}  # No dimensions for 0-d array

    def test_from_value_empty_array(self) -> None:
        """from_value handles empty numpy arrays."""
        import numpy as np

        arr = np.array([], dtype="float64")
        result = TypeDesc.from_value(arr)

        assert result.kind == "ndarray"
        assert result.dtype == "float64"
        assert result.dims == {"dim0": 0}

    def test_from_value_empty_2d_array(self) -> None:
        """from_value handles empty 2D numpy arrays."""
        import numpy as np

        arr = np.zeros((0, 5), dtype="int32")
        result = TypeDesc.from_value(arr)

        assert result.kind == "ndarray"
        assert result.dtype == "int32"
        assert result.dims == {"dim0": 0, "dim1": 5}


@numpy_required
class TestFromValueNumpyScalars:
    """Tests for numpy scalar handling in from_value."""

    @pytest.mark.parametrize(
        "np_type,expected_dtype",
        [
            ("float16", "float16"),
            ("float32", "float32"),
            ("float64", "float64"),
            ("int8", "int8"),
            ("int16", "int16"),
            ("int32", "int32"),
            ("int64", "int64"),
            ("uint8", "uint8"),
            ("uint16", "uint16"),
            ("uint32", "uint32"),
            ("uint64", "uint64"),
        ],
    )
    def test_from_value_numpy_scalar_types(self, np_type: str, expected_dtype: str) -> None:
        """from_value handles numpy scalar types (np.float64, np.int32, etc.)."""
        import numpy as np

        np_dtype = getattr(np, np_type)
        scalar = np_dtype(42)
        result = TypeDesc.from_value(scalar)

        assert result.kind == "scalar"
        assert result.dtype == expected_dtype
        assert result.dims is None

    def test_from_value_numpy_bool(self) -> None:
        """from_value handles numpy bool_ scalar."""
        import numpy as np

        scalar = np.bool_(True)
        result = TypeDesc.from_value(scalar)

        assert result.kind == "scalar"
        assert result.dtype == "bool"

    def test_from_value_array_element_is_scalar(self) -> None:
        """Array element access returns numpy scalar, properly handled."""
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0], dtype="float32")
        element = arr[0]  # This is np.float32, not Python float
        result = TypeDesc.from_value(element)

        assert result.kind == "scalar"
        assert result.dtype == "float32"
