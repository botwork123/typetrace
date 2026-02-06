"""
Polars adapter for typetrace.

Handles Polars DataFrame and Series types.
"""

from typing import Any

from typetrace.core import TypeDesc

# Polars dtype mapping (polars string -> polars dtype)
POLARS_DTYPE_MAP = {
    "Float64": "Float64",
    "Float32": "Float32",
    "Int64": "Int64",
    "Int32": "Int32",
    "Int16": "Int16",
    "Int8": "Int8",
    "UInt64": "UInt64",
    "UInt32": "UInt32",
    "UInt16": "UInt16",
    "UInt8": "UInt8",
    "Boolean": "Boolean",
    "Utf8": "Utf8",
    "String": "String",
    "Date": "Date",
    "Datetime": "Datetime",
}


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


def _get_polars_dtype(dtype_str: str) -> Any:
    """Convert dtype string to polars dtype."""
    import polars as pl

    # Normalize common dtype names
    dtype_lower = dtype_str.lower()

    if "float64" in dtype_lower or dtype_str == "Float64":
        return pl.Float64
    elif "float32" in dtype_lower or dtype_str == "Float32":
        return pl.Float32
    elif "int64" in dtype_lower or dtype_str == "Int64":
        return pl.Int64
    elif "int32" in dtype_lower or dtype_str == "Int32":
        return pl.Int32
    elif "int16" in dtype_lower or dtype_str == "Int16":
        return pl.Int16
    elif "int8" in dtype_lower or dtype_str == "Int8":
        return pl.Int8
    elif "uint64" in dtype_lower or dtype_str == "UInt64":
        return pl.UInt64
    elif "uint32" in dtype_lower or dtype_str == "UInt32":
        return pl.UInt32
    elif "uint16" in dtype_lower or dtype_str == "UInt16":
        return pl.UInt16
    elif "uint8" in dtype_lower or dtype_str == "UInt8":
        return pl.UInt8
    elif "bool" in dtype_lower or dtype_str == "Boolean":
        return pl.Boolean
    elif "utf8" in dtype_lower or "string" in dtype_lower or dtype_str in ("Utf8", "String"):
        return pl.Utf8
    else:
        return pl.Float64  # Default


def make_polars_dataframe_sample(type_desc: TypeDesc) -> Any:
    """
    Create minimal Polars DataFrame from TypeDesc.

    Args:
        type_desc: TypeDesc with kind='dataframe'

    Returns:
        polars.DataFrame with correct structure
    """
    import polars as pl

    if type_desc.columns is None:
        raise ValueError("Cannot make Polars DataFrame sample without columns")

    dtypes = type_desc.dtypes or {}
    schema = {}
    for col in type_desc.columns:
        dtype_str = dtypes.get(col, "Float64")
        schema[col] = _get_polars_dtype(dtype_str)

    return pl.DataFrame(schema=schema)


def make_polars_series_sample(type_desc: TypeDesc) -> Any:
    """
    Create minimal Polars Series from TypeDesc.

    Args:
        type_desc: TypeDesc with kind='series'

    Returns:
        polars.Series with correct dtype
    """
    import polars as pl

    dtype_str = type_desc.dtype or "Float64"
    dtype = _get_polars_dtype(dtype_str)

    return pl.Series(name="", values=[], dtype=dtype)
