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

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import EllipsisType
from typing import Any, Callable, Hashable, Literal

from typetrace.runtime_utils import module_root


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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name) is None:
            raise TypeDescValidationError(f"invalid symbol name: {self.name!r}")

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


# Type alias for dimension values
DimValue = int | Symbol

Dims = tuple[tuple[str, DimValue, tuple[Hashable, ...] | None], ...]


class TypeDescError(ValueError):
    """Base class for structural TypeDesc failures."""

    def __init__(self, message: str, *, operation: str | None = None, path: tuple[str | int, ...] = ()) -> None:
        self.operation = operation
        self.path = path
        context = f" at {'.'.join(map(str, path))}" if path else ""
        super().__init__(f"{message}{context}")


class TypeDescValidationError(TypeDescError):
    """A TypeDesc violates the v2 structural schema."""


class TypeDescConflictError(TypeDescError):
    """Two concrete structural facts conflict."""


class TypeDescUnknownError(TypeDescError):
    """An operation requires a fact that is unknown."""


class UnsupportedOperationError(TypeDescError):
    """A structural operation is not registered for a nominal kind."""


class _FrozenDict(dict[Any, Any]):
    """Hashable mapping retaining the normal dict comparison/access API."""

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items(), key=lambda item: repr(item[0]))))

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("TypeDesc mappings are immutable")

    __delitem__ = __setitem__ = clear = pop = popitem = setdefault = update = _immutable


class _FrozenList(list[Any]):
    """Hashable list retaining list equality for compatibility."""

    def __hash__(self) -> int:
        return hash(tuple(self))

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("TypeDesc sequences are immutable")

    __delitem__ = __setitem__ = __iadd__ = __imul__ = append = extend = insert = _immutable
    pop = remove = reverse = sort = clear = _immutable


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_metadata(value: Any) -> Hashable:
    """Convert metadata containers to deterministic immutable tuples."""
    if isinstance(value, Mapping):
        items = tuple(sorted((key, _freeze_metadata(item)) for key, item in value.items()))
        return items
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    try:
        hash(value)
    except TypeError as exc:
        raise TypeDescValidationError("metadata contains an unhashable value") from exc
    return value


def _canonical(value: Any) -> Any:
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Symbol):
        return ("symbol", value.name)
    if isinstance(value, TypeDesc):
        return ("typedesc", tuple(_canonical(getattr(value, field)) for field in value.__dataclass_fields__))
    if isinstance(value, Mapping):
        items = ((_canonical(key), _canonical(item)) for key, item in value.items())
        return ("mapping", tuple(sorted(items, key=repr)))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_canonical(item) for item in value))
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, Hashable):
        return ("hashable", type(value).__module__, type(value).__qualname__, repr(value))
    raise TypeDescValidationError(f"unsupported canonical value: {type(value)!r}")


_SHAPE_CONTRACT_KINDS = frozenset(
    {
        "ndarray",
        "dataset",
        "dataframe",
        "series",
        "columnar",
        "drjit",
    }
)


def requires_shape_contract(type_desc: "TypeDesc") -> bool:
    """Return whether the type requires an explicit shape/schema contract."""
    return type_desc.kind in _SHAPE_CONTRACT_KINDS


@dataclass(frozen=True, eq=False)
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

    kind: str

    # For ndarrays (xarray) - named dimensions
    dims: Dims | None = None

    # For DrJit/numpy - positional dimensions
    shape: tuple[DimValue, ...] | None = None

    # Element type
    dtype: str | None = None

    # Per-column types (dataframe)
    dtypes: tuple[tuple[Hashable, str], ...] | None = None

    # Index dimensions (pandas)
    index: Dims | None = None

    # Column names (dataframe/columnar). Use trailing literal ellipsis to denote
    # partial schema semantics: ["known_a", "known_b", ...].
    columns: tuple[Hashable, ...] | None = None

    # Nested type descriptors (opaque classes)
    fields: tuple[tuple[Hashable, "TypeDesc"], ...] | None = None

    # DrJit specific
    drjit_type: type | None = None
    static_dims: tuple[int, ...] | None = None

    # Canonical structural metadata
    metadata: tuple[tuple[str, Hashable], ...] = ()

    def __post_init__(self) -> None:
        """Normalize and validate the complete immutable v2 schema."""
        if not isinstance(self.kind, str) or not self.kind:
            raise TypeDescValidationError("kind must be a non-empty string")
        dims = self._normalize_dims(self.dims, "dims")
        index = self._normalize_dims(self.index, "index")
        shape = self._normalize_shape(self.shape, "shape")
        dtypes = self._normalize_pairs(self.dtypes, "dtypes")
        columns = self._normalize_hashables(self.columns, "columns")
        fields = self._normalize_fields(self.fields)
        metadata = self._normalize_metadata(self.metadata)
        for name, value in {
            "dims": dims,
            "index": index,
            "shape": shape,
            "dtypes": dtypes,
            "columns": columns,
            "fields": fields,
            "metadata": metadata,
        }.items():
            object.__setattr__(self, name, value)
        if self.dtype is not None and not isinstance(self.dtype, str):
            raise TypeDescValidationError("dtype must be a string or None")
        if self.static_dims is not None:
            if any(not isinstance(size, int) or size < 0 for size in self.static_dims):
                raise TypeDescValidationError("static_dims must contain non-negative integers")
            object.__setattr__(self, "static_dims", tuple(self.static_dims))
        if self.drjit_type is not None and not isinstance(self.drjit_type, type):
            raise TypeDescValidationError("drjit_type must be a type or None")
        if self.kind == "scalar" and (self.shape is not None or self.dims is not None):
            raise TypeDescValidationError("scalar cannot declare dimensions or shape")

    @staticmethod
    def _normalize_shape(value: Any, path: str) -> tuple[DimValue, ...] | None:
        if value is None:
            return None
        if not isinstance(value, (tuple, list)):
            raise TypeDescValidationError(f"{path} must be a sequence")
        result: list[DimValue] = []
        for index, size in enumerate(value):
            if isinstance(size, Symbol):
                result.append(size)
            elif isinstance(size, int) and size >= 0:
                result.append(size)
            else:
                raise TypeDescValidationError(f"{path}[{index}] must be a non-negative integer or Symbol")
        return tuple(result)

    @classmethod
    def _normalize_dims(cls, value: Any, path: str) -> Dims | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            entries = tuple((name, size, None) for name, size in value.items())
        else:
            entries = tuple(tuple(entry) for entry in value)
        result: list[tuple[str, DimValue, tuple[Hashable, ...] | None]] = []
        names: set[str] = set()
        for index, entry in enumerate(entries):
            if len(entry) == 2:
                name, size = entry
                labels = None
            elif len(entry) == 3:
                name, size, labels = entry
            else:
                raise TypeDescValidationError(f"{path}[{index}] must have 2 or 3 values")
            if not isinstance(name, str) or not name or name in names:
                raise TypeDescValidationError(f"{path}[{index}] has an invalid or duplicate name")
            names.add(name)
            cls._validate_size(size, f"{path}[{index}]")
            normalized_labels = cls._normalize_hashables(labels, f"{path}[{index}].labels")
            if normalized_labels is not None and isinstance(size, int) and len(normalized_labels) != size:
                raise TypeDescValidationError(f"{path}[{index}].labels length disagrees with size")
            result.append((name, size, normalized_labels))
        return tuple(result)

    @staticmethod
    def _validate_size(size: Any, path: str) -> None:
        if not isinstance(size, Symbol) and (not isinstance(size, int) or size < 0):
            raise TypeDescValidationError(f"{path} size must be non-negative or Symbol")

    @staticmethod
    def _normalize_hashables(value: Any, path: str) -> tuple[Hashable, ...] | None:
        if value is None:
            return None
        if not isinstance(value, (tuple, list)):
            raise TypeDescValidationError(f"{path} must be a sequence")
        result = tuple(value)
        try:
            for item in result:
                hash(item)
        except TypeError as exc:
            raise TypeDescValidationError(f"{path} contains an unhashable value") from exc
        if ... in result and result[-1] is not ...:
            raise TypeDescValidationError(f"{path} ellipsis must be trailing")
        return result

    @staticmethod
    def _normalize_pairs(value: Any, path: str) -> tuple[tuple[Hashable, str], ...] | None:
        if value is None:
            return None
        result: list[tuple[Hashable, str]] = []
        keys: set[Hashable] = set()
        for index, pair in enumerate(value):
            if len(pair) != 2 or not isinstance(pair[1], str):
                raise TypeDescValidationError(f"{path}[{index}] must be (hashable, str)")
            key, dtype = pair
            try:
                hash(key)
            except TypeError as exc:
                raise TypeDescValidationError(f"{path}[{index}] key is unhashable") from exc
            if key in keys:
                raise TypeDescValidationError(f"{path} contains duplicate key {key!r}")
            keys.add(key)
            result.append((key, dtype))
        return tuple(result)

    @classmethod
    def _normalize_fields(cls, value: Any) -> tuple[tuple[Hashable, "TypeDesc"], ...] | None:
        if value is None:
            return None
        result: list[tuple[Hashable, TypeDesc]] = []
        keys: set[Hashable] = set()
        for index, pair in enumerate(value):
            if len(pair) != 2 or not isinstance(pair[1], TypeDesc):
                raise TypeDescValidationError(f"fields[{index}] must be (hashable, TypeDesc)")
            key, descriptor = pair
            try:
                hash(key)
            except TypeError as exc:
                raise TypeDescValidationError(f"fields[{index}] key is unhashable") from exc
            if key in keys:
                raise TypeDescValidationError(f"fields contains duplicate key {key!r}")
            keys.add(key)
            result.append((key, descriptor))
        return tuple(result)

    @classmethod
    def _normalize_metadata(cls, value: Any) -> tuple[tuple[str, Hashable], ...]:
        items = value.items() if isinstance(value, Mapping) else value
        if items is None:
            return ()
        result: dict[str, Hashable] = {}
        for index, pair in enumerate(items):
            if len(pair) != 2 or not isinstance(pair[0], str):
                raise TypeDescValidationError(f"metadata[{index}] must be (str, hashable)")
            key, raw = pair
            frozen = _freeze_metadata(raw)
            result[key] = frozen
        return tuple(sorted(result.items()))

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeDesc) and _canonical(self) == _canonical(other)

    def fingerprint(self) -> str:
        """Return a deterministic identity for this structural descriptor."""
        payload = repr(_canonical(self)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def bind(self, bindings: Mapping[str, Any]) -> "TypeDesc":
        """Return a descriptor with matching symbolic dimensions replaced."""
        symbols = self._symbols()
        unknown = set(bindings) - symbols
        if unknown:
            raise TypeDescValidationError(f"bindings contain unknown symbols: {sorted(unknown)!r}")

        def bind_value(value: Any) -> Any:
            if isinstance(value, Symbol):
                bound = bindings.get(value.name, value)
                if isinstance(bound, int) and bound < 0:
                    raise TypeDescConflictError(f"binding for {value.name!r} is negative")
                return bound
            if isinstance(value, TypeDesc):
                nested_bindings = {name: item for name, item in bindings.items() if name in value._symbols()}
                return value.bind(nested_bindings)
            if isinstance(value, Mapping):
                return {key: bind_value(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(bind_value(item) for item in value)
            if isinstance(value, list):
                return [bind_value(item) for item in value]
            return value

        values = {field: bind_value(getattr(self, field)) for field in self.__dataclass_fields__}
        return replace(self, **values)

    def _symbols(self) -> set[str]:
        found: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, Symbol):
                found.add(value.name)
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    visit(key)
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        for field in self.__dataclass_fields__:
            visit(getattr(self, field))
        return found

    def known_columns(self) -> list[str] | None:
        """Return concrete known columns, excluding optional trailing ellipsis marker."""
        if self.columns is None:
            return None
        if self.columns and self.columns[-1] is ...:
            return [col for col in self.columns[:-1] if isinstance(col, str)]
        return [col for col in self.columns if isinstance(col, str)]

    def with_dims(self, dims: Dims) -> "TypeDesc":
        """Return copy with updated dims."""
        return replace(self, dims=dims)

    def with_dtype(self, dtype: str) -> "TypeDesc":
        """Return copy with updated dtype."""
        return replace(self, dtype=dtype)

    @classmethod
    def from_value(cls, value: Any, *, _seen: set[int] | None = None) -> "TypeDesc":
        """
        Extract TypeDesc from a runtime value.

        Dispatches to appropriate adapter based on value type.
        Tracks visited objects to prevent infinite recursion on cycles.

        Handles:
        - Python scalars (int, float, str, bool) → TypeDesc(kind="scalar", dtype=...)
        - numpy arrays → TypeDesc(kind="ndarray", dtype=..., dims={...})
        - numpy scalars → TypeDesc(kind="scalar", dtype=...)
        - xarray/pandas/polars/arrow/drjit → dispatched to adapters
        - Other objects → introspected as class
        """
        # Handle Python scalars first (before module dispatch)
        if isinstance(value, bool):  # Must check before int since bool is subclass of int
            return cls(kind="scalar", dtype="bool")
        if isinstance(value, int):
            return cls(kind="scalar", dtype="int64")
        if isinstance(value, float):
            return cls(kind="scalar", dtype="float64")
        if isinstance(value, str):
            return cls(kind="scalar", dtype="str")
        if isinstance(value, (bytes, type(None))):
            return cls(kind="class", fields=None)

        root = module_root(value)
        dispatch = cls._dispatch_table()
        if root in dispatch:
            return dispatch[root](value)
        return cls._from_object(value, _seen=_seen)

    @staticmethod
    def _dispatch_table() -> dict[str, Callable[[Any], "TypeDesc"]]:
        """Build adapter dispatch table lazily to avoid hard dependencies."""

        from typetrace.adapters.arrow import from_arrow
        from typetrace.adapters.drjit import from_drjit
        from typetrace.adapters.numpy import from_numpy
        from typetrace.adapters.pandas import from_pandas
        from typetrace.adapters.polars import from_polars
        from typetrace.adapters.xarray import from_xarray

        return {
            "xarray": from_xarray,
            "pandas": from_pandas,
            "drjit": from_drjit,
            "polars": from_polars,
            "pyarrow": from_arrow,
            "numpy": from_numpy,
        }

    @classmethod
    def _from_object(cls, value: Any, *, _seen: set[int] | None = None) -> "TypeDesc":
        """Extract TypeDesc from arbitrary Python object.

        Tracks visited object ids to detect cycles and prevent infinite recursion.
        """
        if _seen is None:
            _seen = set()

        obj_id = id(value)
        if obj_id in _seen:
            return cls(kind="recursive")
        _seen.add(obj_id)

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
            fields[name] = cls.from_value(attr, _seen=_seen)
        return cls(kind="class", fields=fields or None)

    def make_sample(self) -> Any:
        """Create minimal runtime sample preserving this descriptor schema."""
        samples = self._sample_dispatch_table()
        if self.kind not in samples:
            raise NotImplementedError(f"make_sample not implemented for kind={self.kind}")
        return samples[self.kind](self)

    @staticmethod
    def _sample_dispatch_table() -> dict[str, Callable[["TypeDesc"], Any]]:
        """Build sample-materialization dispatch table lazily."""
        from typetrace.adapters.arrow import make_arrow_table_sample
        from typetrace.adapters.drjit import make_drjit_sample
        from typetrace.adapters.pandas import make_dataframe_sample, make_series_sample
        from typetrace.adapters.xarray import make_dataset_sample, make_xarray_sample

        return {
            "ndarray": make_xarray_sample,
            "dataset": make_dataset_sample,
            "dataframe": make_dataframe_sample,
            "series": make_series_sample,
            "columnar": make_arrow_table_sample,
            "drjit": make_drjit_sample,
        }

    def field(self, name: str) -> "TypeDesc":
        """Get type descriptor for a field (opaque classes)."""
        if self.fields is None:
            raise ValueError(f"TypeDesc has no fields (kind={self.kind})")
        for field_name, descriptor in self.fields:
            if field_name == name:
                return descriptor
        raise KeyError(f"Field {name!r} not found in {[key for key, _ in self.fields]}")

    @classmethod
    def for_type(
        cls,
        concrete_type: type,
        *,
        dtype: str | None = None,
        dims: Dims | None = None,
        shape: tuple[DimValue, ...] | None = None,
        columns: list[str | EllipsisType] | None = None,
        dtypes: dict[str, str] | None = None,
        index: Dims | None = None,
        fields: dict[str, "TypeDesc"] | None = None,
        drjit_type: type | None = None,
        static_dims: tuple[int, ...] | None = None,
    ) -> "TypeDesc":
        """Create TypeDesc by inferring kind from concrete_type.

        This is the preferred way to create TypeDesc when you know the
        Python type. The kind is automatically derived from the type.

        Args:
            concrete_type: Python type (xr.DataArray, pd.DataFrame, etc.)
            dtype: Element dtype (e.g., "float64")
            dims: Named dimensions for ndarrays
            shape: Positional dimensions for DrJit/numpy
            columns: Column names for dataframes. Use trailing literal
                ellipsis for partial schema, e.g. ["a", "b", ...].
            dtypes: Per-column dtypes for dataframes
            index: Index dimensions for pandas
            fields: Nested TypeDescs for opaque classes
            drjit_type: Actual DrJit type for codegen
            static_dims: DrJit static dimensions

        Returns:
            TypeDesc with kind inferred from concrete_type

        Examples:
            >>> TypeDesc.for_type(xr.DataArray, dtype="float64", dims={"x": 10})
            TypeDesc(kind='ndarray', dtype='float64', dims={'x': 10}, ...)

            >>> TypeDesc.for_type(pd.DataFrame, columns=["a", "b"])
            TypeDesc(kind='dataframe', columns=['a', 'b'], ...)
        """
        kind = cls._kind_for_type(concrete_type)
        return cls(
            kind=kind,
            dtype=dtype,
            dims=dims,
            shape=shape,
            columns=columns,
            dtypes=dtypes,
            index=index,
            fields=fields,
            drjit_type=drjit_type,
            static_dims=static_dims,
        )

    @staticmethod
    def _kind_for_type(
        concrete_type: type,
    ) -> Literal[
        "ndarray",
        "dataset",
        "dataframe",
        "series",
        "columnar",
        "class",
        "drjit",
        "recursive",
        "scalar",
    ]:
        """Map Python type to TypeDesc kind.

        Supports:
        - xarray: DataArray, Dataset → "ndarray"
        - numpy: ndarray → "ndarray"
        - pandas: DataFrame → "dataframe", Series → "series"
        - polars: DataFrame → "dataframe", Series → "series"
        - pyarrow: Table → "columnar"
        - drjit: any dr.* array type → "drjit"
        """
        module_root = concrete_type.__module__.split(".")[0]

        # xarray types
        if module_root == "xarray":
            if concrete_type.__name__ == "DataArray":
                return "ndarray"
            if concrete_type.__name__ == "Dataset":
                return "dataset"

        # numpy
        if module_root == "numpy" and concrete_type.__name__ == "ndarray":
            return "ndarray"

        # pandas
        if module_root == "pandas":
            if concrete_type.__name__ == "DataFrame":
                return "dataframe"
            if concrete_type.__name__ == "Series":
                return "series"

        # polars
        if module_root == "polars":
            if concrete_type.__name__ == "DataFrame":
                return "dataframe"
            if concrete_type.__name__ == "Series":
                return "series"

        # pyarrow
        if module_root == "pyarrow" and concrete_type.__name__ == "Table":
            return "columnar"

        # drjit - any type from drjit module
        if module_root == "drjit":
            return "drjit"

        # Fallback to class for unknown types
        return "class"
