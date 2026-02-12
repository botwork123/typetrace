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
    if isinstance(value, pa.Array):
        return TypeDesc(kind="series", dtype=str(value.type))
    raise TypeError(f"Expected Arrow type, got {type(value)}")


def _get_arrow_type(dtype_str: str) -> Any:
    """Convert dtype string to Arrow type."""
    import pyarrow as pa

    dtype_lower = dtype_str.lower()
    mapping: list[tuple[tuple[str, ...], Any]] = [
        (("float64", "double"), pa.float64()),
        (("float32",), pa.float32()),
        (("uint64",), pa.uint64()),
        (("uint32",), pa.uint32()),
        (("uint16",), pa.uint16()),
        (("uint8",), pa.uint8()),
        (("int64",), pa.int64()),
        (("int32",), pa.int32()),
        (("int16",), pa.int16()),
        (("int8",), pa.int8()),
        (("bool",), pa.bool_()),
        (("string", "utf8"), pa.string()),
    ]
    for tokens, arrow_type in mapping:
        if any(token in dtype_lower for token in tokens):
            return arrow_type
    return pa.float64()


def make_arrow_table_sample(type_desc: TypeDesc) -> Any:
    """
    Create minimal Arrow Table from TypeDesc.

    Args:
        type_desc: TypeDesc with kind='columnar'

    Returns:
        pyarrow.Table with correct schema
    """
    import pyarrow as pa

    if type_desc.columns is None:
        raise ValueError("Cannot make Arrow Table sample without columns")

    dtypes = type_desc.dtypes or {}
    fields = []
    arrays = []

    for col in type_desc.columns:
        dtype_str = dtypes.get(col, "float64")
        arrow_type = _get_arrow_type(dtype_str)
        fields.append(pa.field(col, arrow_type))
        arrays.append(pa.array([], type=arrow_type))

    schema = pa.schema(fields)
    return pa.table(dict(zip(type_desc.columns, arrays)), schema=schema)


def make_arrow_array_sample(type_desc: TypeDesc) -> Any:
    """
    Create minimal Arrow Array from TypeDesc.

    Args:
        type_desc: TypeDesc with kind='series'

    Returns:
        pyarrow.Array with correct type
    """
    import pyarrow as pa

    dtype_str = type_desc.dtype or "float64"
    arrow_type = _get_arrow_type(dtype_str)

    return pa.array([], type=arrow_type)
