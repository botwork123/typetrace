"""Contract tests for recursive sample materialization."""

from __future__ import annotations

import pytest

from typetrace import TypeDesc
from typetrace.errors import SampleMaterializationError
from typetrace.inference import make_samples


def test_make_samples_recurses_through_supported_containers() -> None:
    args, kwargs = make_samples(
        (
            TypeDesc(kind="scalar", dtype="int64"),
            [TypeDesc(kind="scalar", dtype="float64")],
        ),
        {"nested": {"value": TypeDesc(kind="scalar", dtype="bool")}, "literal": 3},
    )

    assert args == (0, [0.0])
    assert kwargs == {"nested": {"value": False}, "literal": 3}


@pytest.mark.parametrize(
    "args,kwargs,path",
    [
        (({1: TypeDesc(kind="scalar", dtype="int64")},), {"ok": 1}, "args.0"),
        ((object(),), {}, "args.0"),
    ],
)
def test_make_samples_reports_stable_paths(args, kwargs, path: str) -> None:
    with pytest.raises(SampleMaterializationError, match=path):
        make_samples(args, kwargs)


def test_make_samples_rejects_live_backend_values() -> None:
    np = pytest.importorskip("numpy")

    with pytest.raises(SampleMaterializationError, match="args.0"):
        make_samples((np.zeros(2),), {})
