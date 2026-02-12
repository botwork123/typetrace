"""Tests for runtime execution traits and layout transitions."""

import pytest
from typetrace.core import TypeDesc
from typetrace.execution_traits import ExecutionTraits, _drjit_dtype, infer_execution_traits
from typetrace.inference import infer_by_execution
from typetrace.layout_ops import (
    check_handoff_compatibility,
    concat_traits,
    normalize_handoff_traits,
    reshape_restack_traits,
    slice_view_traits,
    transpose_traits,
)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (
            {
                "dtype": "float32",
                "shape": (2, 3),
                "layout_order": "C",
                "contiguous_c": False,
            },
            "layout_order='C' requires contiguous_c=True",
        ),
        (
            {
                "dtype": "float32",
                "shape": (2, 3),
                "layout_order": "F",
                "contiguous_f": False,
            },
            "layout_order='F' requires contiguous_f=True",
        ),
    ],
)
def test_execution_traits_validation_errors(kwargs: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        ExecutionTraits(**kwargs)


@pytest.mark.parametrize(
    "source,target,allow_device_copy,expected",
    [
        (
            ExecutionTraits("float32", (2, 3), "cpu", "C", True, False),
            ExecutionTraits("float32", (2, 3), "cpu", "C", True, False),
            False,
            (True, False, ()),
        ),
        (
            ExecutionTraits("float64", (2, 3), "cpu", "C", True, False),
            ExecutionTraits("float32", (2, 3), "cpu", "C", True, False),
            False,
            (False, False, ("dtype mismatch: source=float64, target=float32",)),
        ),
        (
            ExecutionTraits("float32", (2, 3), "cpu", "strided", False, False),
            ExecutionTraits("float32", (2, 3), "cpu", "C", True, False),
            False,
            (
                False,
                True,
                (
                    "layout order mismatch: source=strided, target=C",
                    "C-contiguous buffer required by target",
                ),
            ),
        ),
        (
            ExecutionTraits("float32", (2, 3), "cpu", "C", True, False),
            ExecutionTraits("float32", (2, 3), "cuda", "C", True, False),
            False,
            (False, False, ("device mismatch not allowed: cpu!=cuda",)),
        ),
        (
            ExecutionTraits("float32", (2, 3), "cpu", "C", True, False),
            ExecutionTraits("float32", (2, 3), "cuda", "C", True, False),
            True,
            (False, True, ("device transfer required: cpu->cuda",)),
        ),
    ],
)
def test_handoff_compatibility(
    source: ExecutionTraits,
    target: ExecutionTraits,
    allow_device_copy: bool,
    expected: tuple[bool, bool, tuple[str, ...]],
) -> None:
    result = check_handoff_compatibility(source, target, allow_device_copy)
    assert (result.compatible, result.requires_copy, result.reasons) == expected


@pytest.mark.parametrize(
    "operation,expected_shape,expected_layout,expected_copy",
    [
        ("slice", (4, 5), "strided", None),
        ("transpose", (5, 4), "strided", None),
        ("reshape", (2, 10), "C", False),
    ],
)
def test_layout_transition_utilities(
    operation: str,
    expected_shape: tuple[int, ...],
    expected_layout: str,
    expected_copy: bool | None,
) -> None:
    source = ExecutionTraits("float32", (4, 5), "cpu", "C", True, False)
    if operation == "slice":
        result = slice_view_traits(source)
        assert result.shape == expected_shape
        assert result.layout_order == expected_layout
        return
    if operation == "transpose":
        result = transpose_traits(source, (1, 0))
        assert result.shape == expected_shape
        assert result.layout_order == expected_layout
        return
    result, needs_copy = reshape_restack_traits(source, expected_shape, "C")
    assert result.shape == expected_shape
    assert result.layout_order == expected_layout
    assert needs_copy == expected_copy


@pytest.mark.parametrize(
    "shapes,axis,expected_shape",
    [
        (((2, 3), (4, 3)), 0, (6, 3)),
        (((2, 3), (2, 7)), 1, (2, 10)),
    ],
)
def test_concat_and_handoff_normalization(
    shapes: tuple[tuple[int, ...], tuple[int, ...]],
    axis: int,
    expected_shape: tuple[int, ...],
) -> None:
    first = ExecutionTraits("float32", shapes[0], "cpu", "C", True, False, owner="numpy")
    second = ExecutionTraits("float32", shapes[1], "cpu", "C", True, False, owner="numpy")
    concat_result = concat_traits((first, second), axis)
    handoff_result = normalize_handoff_traits(concat_result, owner="torch")
    assert concat_result.shape == expected_shape
    assert concat_result.layout_order == "C"
    assert handoff_result.owner == "torch"
    assert handoff_result.contiguous_c is True


@pytest.mark.parametrize(
    "case",
    ["pass", "fail"],
)
def test_infer_by_execution_enforces_traits(case: str) -> None:
    pytest.importorskip("xarray")

    def identity(arr):
        return arr

    input_type = TypeDesc(kind="ndarray", dims={"x": 4, "y": 5}, dtype="float64")
    expected = ExecutionTraits("float64", (4, 5), "cpu", "C", True, False)
    if case == "pass":
        result = infer_by_execution(identity, input_type, expected_output_traits=expected)
        assert result.kind == "ndarray"
        assert result.dtype == "float64"
        return
    bad_target = ExecutionTraits("float32", (4, 5), "cpu", "C", True, False)
    with pytest.raises(ValueError, match="dtype mismatch"):
        infer_by_execution(identity, input_type, expected_output_traits=bad_target)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cpu_gpu_policy_for_infer_by_execution(device: str) -> None:
    pytest.importorskip("xarray")

    def identity(arr):
        return arr

    input_type = TypeDesc(kind="ndarray", dims={"x": 4}, dtype="float64")
    target = ExecutionTraits("float64", (4,), device, "C", True, False)
    if device == "cpu":
        result = infer_by_execution(identity, input_type, expected_output_traits=target)
        assert result.dims == {"x": 4}
        return
    with pytest.raises(ValueError, match="device mismatch not allowed"):
        infer_by_execution(identity, input_type, expected_output_traits=target)


@pytest.mark.parametrize("stride_slice", [(slice(None), slice(None, None, 2))])
def test_infer_execution_traits_numpy_hidden_copy_condition(
    stride_slice: tuple[slice, slice],
) -> None:
    np = pytest.importorskip("numpy")

    array = np.zeros((6, 6), dtype=np.float32)
    view = array[stride_slice]
    source = infer_execution_traits(view)
    target = ExecutionTraits("float32", view.shape, "cpu", "C", True, False)
    compatibility = check_handoff_compatibility(source, target)
    assert source.layout_order == "strided"
    assert compatibility.requires_copy is True
    assert compatibility.reasons[-1] == "C-contiguous buffer required by target"


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"dtype": "float32", "shape": (-1,)}, "Shape dimensions must be >= 0"),
        ({"dtype": "float32", "shape": (1,), "device": "metal"}, "Unsupported device"),
    ],
)
def test_execution_traits_additional_validation(kwargs: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        ExecutionTraits(**kwargs)


@pytest.mark.parametrize(
    "dtype_name,expected",
    [
        ("Float64Array", "float64"),
        ("FloatArray", "float32"),
        ("UInt64Array", "uint64"),
        ("UIntArray", "uint32"),
        ("Int64Array", "int64"),
        ("IntArray", "int32"),
        ("BoolArray", "bool"),
        ("MysteryArray", "unknown"),
    ],
)
def test_drjit_dtype_mapping(dtype_name: str, expected: str) -> None:
    fake_type = type(dtype_name, (), {})
    fake_value = fake_type()
    assert _drjit_dtype(fake_value) == expected


@pytest.mark.parametrize(
    "case,expected",
    [
        ("torch_cpu", ("torch", "cpu", "C")),
        ("torch_strided", ("torch", "cpu", "strided")),
        ("drjit_cuda", ("drjit", "cuda", "strided")),
    ],
)
def test_infer_execution_traits_optional_backends(
    case: str, expected: tuple[str, str, str], monkeypatch
) -> None:
    if case.startswith("torch"):
        cls = type("FakeTorchTensor", (), {"__module__": "torch"})
        obj = cls()
        obj.dtype = "torch.float32"
        obj.shape = (2, 3)
        obj.is_cuda = False
        obj.is_contiguous = lambda: case == "torch_cpu"
        result = infer_execution_traits(obj)
        assert (result.owner, result.device, result.layout_order) == expected
        return

    cls = type("FakeDrJitTensor", (), {"__module__": "drjit", "__name__": "FloatArray"})
    obj = cls()

    class FakeModule:
        @staticmethod
        def shape(_value):
            return (4, 5)

        @staticmethod
        def backend_v(_value):
            return "cuda_ad"

    monkeypatch.setattr("importlib.import_module", lambda _name: FakeModule)
    result = infer_execution_traits(obj)
    assert (result.owner, result.device, result.layout_order) == expected


@pytest.mark.parametrize("module_name", ["xarray", "pandas"])
def test_infer_execution_traits_tabular_types(module_name: str) -> None:
    np = pytest.importorskip("numpy")
    if module_name == "xarray":
        xr = pytest.importorskip("xarray")
        value = xr.DataArray(np.zeros((2, 2), dtype=np.float64), dims=("x", "y"))
    else:
        pd = pytest.importorskip("pandas")
        value = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = infer_execution_traits(value)
    assert result.owner == "numpy"
    assert result.shape == (2, 2)


@pytest.mark.parametrize(
    "source_shape,target_shape,expected_reason",
    [
        ((2, 3), (2, 3, 1), "rank mismatch"),
        ((2, 3), (4, 3), "shape mismatch"),
    ],
)
def test_handoff_rank_shape_errors(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    expected_reason: str,
) -> None:
    source = ExecutionTraits("float32", source_shape, "cpu", "C", True, False)
    target = ExecutionTraits("float32", target_shape, "cpu", "C", True, False)
    result = check_handoff_compatibility(source, target)
    assert result.compatible is False
    assert expected_reason in result.reasons[0]


@pytest.mark.parametrize(
    "source,axis,expected_reason",
    [
        (
            (),
            0,
            "requires at least one source",
        ),
        (
            (
                ExecutionTraits("float32", (2, 2), "cpu", "C", True, False),
                ExecutionTraits("float64", (2, 2), "cpu", "C", True, False),
            ),
            0,
            "dtype mismatch",
        ),
        (
            (
                ExecutionTraits("float32", (2, 2), "cpu", "C", True, False),
                ExecutionTraits("float32", (2, 2), "cuda", "C", True, False),
            ),
            0,
            "device mismatch",
        ),
    ],
)
def test_concat_error_paths(
    source: tuple[ExecutionTraits, ...],
    axis: int,
    expected_reason: str,
) -> None:
    with pytest.raises(ValueError, match=expected_reason):
        concat_traits(source, axis)
