"""
Polars adapter for typetrace.

Handles Polars DataFrame and Series types.
"""

from typing import Any

from typetrace.core import TypeDesc


def from_polars(value: Any) -> TypeDesc:
    """
    Extract TypeDesc from Polars DataFrame or Series.

    Args:
        value: polars.DataFrame or polars.Series

    Returns:
        TypeDesc with columns and dtypes
    """
    import polars as pl

    if isinstance(value, pl.DataFrame):
        columns = value.columns
        dtypes = {col: str(value[col].dtype) for col in columns}
        return TypeDesc(kind="dataframe", columns=columns, dtypes=dtypes)
    elif isinstance(value, pl.Series):
        return TypeDesc(kind="series", dtype=str(value.dtype))
    else:
        raise TypeError(f"Expected Polars type, got {type(value)}")
