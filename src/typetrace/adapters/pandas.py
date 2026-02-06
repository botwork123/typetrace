"""
pandas adapter for typetrace.

Handles pandas DataFrame and Series types.
"""

from typing import Any

from typetrace.core import TypeDesc


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

        # Handle index
        index = None
        if value.index.name is not None or isinstance(value.index, pd.MultiIndex):
            if isinstance(value.index, pd.MultiIndex):
                index = {
                    name: len(value.index.get_level_values(i))
                    for i, name in enumerate(value.index.names)
                }
            else:
                index = {value.index.name or "index": len(value.index)}

        return TypeDesc(
            kind="dataframe",
            columns=columns,
            dtypes=dtypes,
            index=index,
        )
    elif isinstance(value, pd.Series):
        dtype = str(value.dtype)
        index = None
        if value.index.name is not None:
            index = {value.index.name: len(value.index)}

        return TypeDesc(
            kind="series",
            dtype=dtype,
            index=index,
        )
    else:
        raise TypeError(f"Expected pandas type, got {type(value)}")


def make_dataframe_sample(type_desc: TypeDesc) -> Any:
    """
    Create minimal pandas DataFrame from TypeDesc.

    Args:
        type_desc: TypeDesc with kind='dataframe'

    Returns:
        pandas.DataFrame with correct structure
    """
    import pandas as pd

    if type_desc.columns is None:
        raise ValueError("Cannot make DataFrame sample without columns")

    # Create empty columns with correct dtypes
    data = {}
    dtypes = type_desc.dtypes or {}
    for col in type_desc.columns:
        dtype = dtypes.get(col, "float64")
        data[col] = pd.array([], dtype=dtype)

    df = pd.DataFrame(data)

    # Set index if specified
    if type_desc.index:
        index_names = list(type_desc.index.keys())
        if len(index_names) == 1:
            df.index.name = index_names[0]
        # MultiIndex would need more work

    return df


def make_series_sample(type_desc: TypeDesc) -> Any:
    """
    Create minimal pandas Series from TypeDesc.

    Args:
        type_desc: TypeDesc with kind='series'

    Returns:
        pandas.Series with correct structure
    """
    import pandas as pd

    dtype = type_desc.dtype or "float64"
    series = pd.Series([], dtype=dtype)

    if type_desc.index:
        index_name = list(type_desc.index.keys())[0]
        series.index.name = index_name

    return series
