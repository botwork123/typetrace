"""
Apache Arrow adapter for typetrace.

Handles Arrow tables and arrays.
"""

from typing import Any

from typetrace.core import TypeDesc


def from_arrow(value: Any) -> TypeDesc:
    """
    Extract TypeDesc from Arrow Table or Array.

    Args:
        value: pyarrow.Table or pyarrow.Array

    Returns:
        TypeDesc with schema info
    """
    import pyarrow as pa

    if isinstance(value, pa.Table):
        columns = value.column_names
        dtypes = {name: str(value.schema.field(name).type) for name in columns}
        return TypeDesc(kind="columnar", columns=columns, dtypes=dtypes)
    elif isinstance(value, pa.Array):
        return TypeDesc(kind="series", dtype=str(value.type))
    else:
        raise TypeError(f"Expected Arrow type, got {type(value)}")
