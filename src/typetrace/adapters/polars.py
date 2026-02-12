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
    if isinstance(value, pl.Series):
        return TypeDesc(kind="series", dtype=str(value.dtype))
    raise TypeError(f"Expected Polars type, got {type(value)}")


def _get_polars_dtype(dtype_str: str) -> Any:
    """Convert dtype string to polars dtype."""
    import polars as pl

    dtype_lower = dtype_str.lower()
    mapping: list[tuple[tuple[str, ...], Any]] = [
        (("float64",), pl.Float64),
        (("float32",), pl.Float32),
        (("uint64",), pl.UInt64),
        (("uint32",), pl.UInt32),
        (("uint16",), pl.UInt16),
        (("uint8",), pl.UInt8),
        (("int64",), pl.Int64),
        (("int32",), pl.Int32),
        (("int16",), pl.Int16),
        (("int8",), pl.Int8),
        (("bool",), pl.Boolean),
        (("utf8", "string"), pl.Utf8),
    ]
    for tokens, polars_dtype in mapping:
        if any(token in dtype_lower for token in tokens):
            return polars_dtype

    named_map = {
        "Float64": pl.Float64,
        "Float32": pl.Float32,
        "UInt64": pl.UInt64,
        "UInt32": pl.UInt32,
        "UInt16": pl.UInt16,
        "UInt8": pl.UInt8,
        "Int64": pl.Int64,
        "Int32": pl.Int32,
        "Int16": pl.Int16,
        "Int8": pl.Int8,
        "Boolean": pl.Boolean,
        "Utf8": pl.Utf8,
        "String": pl.Utf8,
    }
    return named_map.get(dtype_str, pl.Float64)


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
