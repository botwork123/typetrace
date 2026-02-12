"""
Core type descriptors for typetrace.

TypeDesc is the universal type descriptor that can represent:
- ndarray (xarray, numpy)
- dataframe (pandas, polars)
- series (pandas)
- columnar (arrow)
- class (opaque custom classes)
- drjit (DrJit arrays/tensors)
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class Symbol:
    """
    Symbolic dimension - bound at runtime.

    Examples:
        Symbol('N')  - universe size, bound when data loads
        Symbol('T')  - time dimension, bound per batch
        Symbol('M')  - output dimension from a calc
    """

    name: str

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


# Type alias for dimension values
DimValue = int | Symbol

# Type alias for named dimensions
Dims = dict[str, DimValue]


@dataclass(frozen=True)
class TypeDesc:
    """
    Universal type descriptor for heterogeneous data structures.

    Supports:
    - ndarray: xarray DataArray, numpy ndarray (named dims)
    - dataframe: pandas/polars DataFrame (index + columns)
    - series: pandas Series (index + single dtype)
    - columnar: Arrow tables (schema)
    - class: opaque custom classes (fields)
    - drjit: DrJit arrays/tensors (positional shape)

    Attributes:
        kind: The category of data structure
        dims: Named dimensions for ndarrays (xarray-style)
        shape: Positional dimensions for DrJit/numpy
        dtype: Element type (single or default)
        dtypes: Per-column types for dataframes
        index: Index dimensions for pandas
        columns: Column names for dataframes
        fields: Nested TypeDescs for opaque classes
        drjit_type: Actual DrJit type for codegen
        static_dims: DrJit static dimensions baked into type
    """

    kind: Literal["ndarray", "dataframe", "series", "columnar", "class", "drjit"]

    # For ndarrays (xarray) - named dimensions
    dims: Dims | None = None

    # For DrJit/numpy - positional dimensions
    shape: tuple[DimValue, ...] | None = None

    # Element type
    dtype: str | None = None

    # Per-column types (dataframe)
    dtypes: dict[str, str] | None = None

    # Index dimensions (pandas)
    index: Dims | None = None

    # Column names (dataframe)
    columns: list[str] | None = None

    # Nested type descriptors (opaque classes)
    fields: dict[str, "TypeDesc"] | None = None

    # DrJit specific
    drjit_type: type | None = None
    static_dims: tuple[int, ...] | None = None

    def with_dims(self, dims: Dims) -> "TypeDesc":
        """Return copy with updated dims."""
        return TypeDesc(
            kind=self.kind,
            dims=dims,
            shape=self.shape,
            dtype=self.dtype,
            dtypes=self.dtypes,
            index=self.index,
            columns=self.columns,
            fields=self.fields,
            drjit_type=self.drjit_type,
            static_dims=self.static_dims,
        )

    def with_dtype(self, dtype: str) -> "TypeDesc":
        """Return copy with updated dtype."""
        return TypeDesc(
            kind=self.kind,
            dims=self.dims,
            shape=self.shape,
            dtype=dtype,
            dtypes=self.dtypes,
            index=self.index,
            columns=self.columns,
            fields=self.fields,
            drjit_type=self.drjit_type,
            static_dims=self.static_dims,
        )

    @classmethod
    def from_value(cls, value: Any) -> "TypeDesc":
        """
        Extract TypeDesc from a runtime value.

        Dispatches to appropriate adapter based on value type.
        """
        module_root = type(value).__module__.split(".", 1)[0]
        dispatch = cls._dispatch_table()
        if module_root in dispatch:
            return dispatch[module_root](value)
        if isinstance(value, (int, float, str, bool, bytes, type(None))):
            return cls(kind="class", fields=None)
        return cls._from_object(value)

    @staticmethod
    def _dispatch_table() -> dict[str, Callable[[Any], "TypeDesc"]]:
        """Build adapter dispatch table lazily to avoid hard dependencies."""

        from typetrace.adapters.arrow import from_arrow
        from typetrace.adapters.drjit import from_drjit
        from typetrace.adapters.pandas import from_pandas
        from typetrace.adapters.polars import from_polars
        from typetrace.adapters.xarray import from_xarray

        return {
            "xarray": from_xarray,
            "pandas": from_pandas,
            "drjit": from_drjit,
            "polars": from_polars,
            "pyarrow": from_arrow,
        }

    @classmethod
    def _from_object(cls, value: Any) -> "TypeDesc":
        """Extract TypeDesc from arbitrary Python object."""
        fields: dict[str, TypeDesc] = {}
        for name in dir(value):
            if name.startswith("_"):
                continue
            try:
                attr = getattr(value, name)
            except (AttributeError, RuntimeError, ValueError):
                continue
            if callable(attr):
                continue
            fields[name] = cls.from_value(attr)
        return cls(kind="class", fields=fields or None)

    def make_sample(self) -> Any:
        """
        Create minimal instance with correct schema for inference-by-execution.

        Returns a zero-sized or minimal array/dataframe that has the right
        structure (dims, columns, dtype) but no actual data.

        Note: For kind='dataframe' and 'series', this returns pandas objects
        by default. Use make_polars_dataframe_sample/make_polars_series_sample
        from the polars adapter for Polars objects.
        """
        if self.kind == "ndarray":
            from typetrace.adapters.xarray import make_xarray_sample

            return make_xarray_sample(self)
        elif self.kind == "dataframe":
            from typetrace.adapters.pandas import make_dataframe_sample

            return make_dataframe_sample(self)
        elif self.kind == "series":
            from typetrace.adapters.pandas import make_series_sample

            return make_series_sample(self)
        elif self.kind == "columnar":
            from typetrace.adapters.arrow import make_arrow_table_sample

            return make_arrow_table_sample(self)
        elif self.kind == "drjit":
            from typetrace.adapters.drjit import make_drjit_sample

            return make_drjit_sample(self)
        else:
            raise NotImplementedError(f"make_sample not implemented for kind={self.kind}")

    def field(self, name: str) -> "TypeDesc":
        """Get type descriptor for a field (opaque classes)."""
        if self.fields is None:
            raise ValueError(f"TypeDesc has no fields (kind={self.kind})")
        if name not in self.fields:
            raise KeyError(f"Field {name!r} not found in {list(self.fields.keys())}")
        return self.fields[name]
