"""Tests for shape-contract capability helpers."""

import pytest

from typetrace.core import TypeDesc, requires_shape_contract


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("scalar", False),
        ("opaque", False),
        ("recursive", False),
        ("numpy.ndarray", True),
        ("xarray.Dataset", True),
        ("pandas.DataFrame", True),
        ("pandas.Series", True),
        ("pyarrow.Table", True),
        ("drjit.Array", True),
    ],
)
def test_requires_shape_contract_by_kind(kind: str, expected: bool) -> None:
    """requires_shape_contract follows typetrace kind semantics."""
    type_desc = TypeDesc(kind=kind)
    assert requires_shape_contract(type_desc) is expected
