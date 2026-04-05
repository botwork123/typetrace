"""
pandas adapter for typetrace.

Handles pandas DataFrame and Series types.
"""

import os
from typing import Any

from typetrace.core import Dims, TypeDesc


def from_pandas(value: Any) -> TypeDesc:
    """
    Extract TypeDesc from pandas DataFrame or Series.

    Args:
        value: pandas.DataFrame or pandas.Series

    Returns:
        TypeDesc with appropriate kind, columns, and dtypes
    """
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        columns = list(value.columns)
        dtypes = {col: str(dtype) for col, dtype in value.dtypes.items()}

        index: Dims | None = None
        if value.index.name is not None or isinstance(value.index, pd.MultiIndex):
            if isinstance(value.index, pd.MultiIndex):
                index = {
                    str(name or f"level_{i}"): int(value.index.get_level_values(i).nunique())
                    for i, name in enumerate(value.index.names)
                }
            else:
                index = {value.index.name or "index": len(value.index)}

        return TypeDesc(kind="dataframe", columns=columns, dtypes=dtypes, index=index)
    elif isinstance(value, pd.Series):
        dtype = str(value.dtype)
        index = None
        if value.index.name is not None:
            index = {value.index.name: len(value.index)}

        return TypeDesc(kind="series", dtype=dtype, index=index)
    else:
        raise TypeError(f"Expected pandas type, got {type(value)}")


def _default_sample_size() -> int:
    value = os.getenv("TYPETRACE_SAMPLE_SIZE", "4")
    return max(int(value), 1)


def _series_values(dtype: str, size: int) -> Any:
    import numpy as np

    if "int" in dtype and "uint" not in dtype:
        return np.arange(size, dtype=dtype)
    if "uint" in dtype:
        return np.arange(size, dtype=dtype)
    if "bool" in dtype:
        return np.array([(i % 2) == 0 for i in range(size)], dtype="bool")
    if "datetime" in dtype:
        return np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + size)
    if "str" in dtype or "object" in dtype:
        return np.array([f"v_{i}" for i in range(size)], dtype=object)
    return np.linspace(0.0, 1.0, num=size, dtype="float64").astype(dtype)


def _index_values(name: str, size: int) -> list[Any]:
    import pandas as pd

    key = name.lower()
    if "time" in key or key == "date":
        return list(pd.date_range("2024-01-01", periods=size, freq="D"))
    if "asset" in key or key in {"symbol", "ticker"}:
        return [f"A{i:03d}" for i in range(size)]
    return list(range(size))


def _build_index(index_desc: Dims | None, rows: int) -> Any:
    import pandas as pd

    if not index_desc:
        return pd.RangeIndex(rows)
    names = list(index_desc.keys())
    sizes = [
        max(int(v) if isinstance(v, int) else _default_sample_size(), 1)
        for v in index_desc.values()
    ]
    levels = [_index_values(name, size) for name, size in zip(names, sizes)]
    return (
        pd.MultiIndex.from_product(levels, names=names)
        if len(names) > 1
        else pd.Index(levels[0], name=names[0])
    )


def make_dataframe_sample(type_desc: TypeDesc) -> Any:
    """Create deterministic pandas DataFrame sample from TypeDesc."""
    import pandas as pd

    known_columns = type_desc.known_columns()
    if known_columns is None:
        raise ValueError("Cannot make DataFrame sample without columns")

    index = _build_index(type_desc.index, _default_sample_size())
    row_count = len(index)
    dtypes = type_desc.dtypes or {}
    data = {col: _series_values(dtypes.get(col, "float64"), row_count) for col in known_columns}
    return pd.DataFrame(data, index=index)


def make_series_sample(type_desc: TypeDesc) -> Any:
    """Create deterministic pandas Series sample from TypeDesc."""
    import pandas as pd

    dtype = type_desc.dtype or "float64"
    index = _build_index(type_desc.index, _default_sample_size())
    data = _series_values(dtype, len(index))
    return pd.Series(data, index=index, dtype=dtype)
