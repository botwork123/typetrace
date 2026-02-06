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


def _get_arrow_type(dtype_str: str) -> Any:
    """Convert dtype string to Arrow type."""
    import pyarrow as pa

    dtype_lower = dtype_str.lower()

    if "float64" in dtype_lower or "double" in dtype_lower:
        return pa.float64()
    elif "float32" in dtype_lower or dtype_lower == "float":
        return pa.float32()
    elif "int64" in dtype_lower:
        return pa.int64()
    elif "int32" in dtype_lower:
        return pa.int32()
    elif "int16" in dtype_lower:
        return pa.int16()
    elif "int8" in dtype_lower:
        return pa.int8()
    elif "uint64" in dtype_lower:
        return pa.uint64()
    elif "uint32" in dtype_lower:
        return pa.uint32()
    elif "uint16" in dtype_lower:
        return pa.uint16()
    elif "uint8" in dtype_lower:
        return pa.uint8()
    elif "bool" in dtype_lower:
        return pa.bool_()
    elif "string" in dtype_lower or "utf8" in dtype_lower:
        return pa.string()
    else:
        return pa.float64()  # Default


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
