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
from typing import Any, Hashable, cast


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
        if (
            not isinstance(self.name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name) is None
        ):
            raise TypeDescValidationError(f"invalid symbol name: {self.name!r}")

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


# Type alias for dimension values
DimValue = int | Symbol

Dims = tuple[tuple[str, DimValue, tuple[Hashable, ...] | None], ...]


class TypeDescError(ValueError):
    """Base class for structural TypeDesc failures."""

    def __init__(
        self, message: str, *, operation: str | None = None, path: tuple[str | int, ...] = ()
    ) -> None:
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


class AdapterRegistrationError(TypeDescError):
    """An adapter registration is invalid or collides with an existing adapter."""


class AdapterUnavailableError(TypeDescError):
    """A requested adapter backend is not installed or available."""


class AdapterAmbiguityError(TypeDescError):
    """More than one adapter matches a value or nominal type."""


class SampleMaterializationError(TypeDescError):
    """A TypeDesc could not be materialized into an execution sample."""


class OperationBindingError(TypeDescError):
    """Operation arguments could not be bound to a callable contract."""


class OperationExecutionError(TypeDescError):
    """Execution of a structural inference operation failed."""


class ResultInferenceError(TypeDescError):
    """An executed result could not be converted into a TypeDesc."""


def _operation_error(exc: TypeDescError, operation: str) -> TypeDescError:
    """Attach the public verb name while preserving the named error/path."""
    return type(exc)(str(exc), operation=operation, path=exc.path)


def _freeze_metadata(value: Any, _seen: set[int] | None = None) -> Hashable:
    """Convert metadata containers to deterministic immutable tuples."""
    seen = _seen if _seen is not None else set()
    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in seen:
            raise TypeDescValidationError("metadata contains a cycle")
        seen.add(object_id)
        try:
            items = tuple(
                sorted(
                    (
                        (_validate_metadata_key(key), _freeze_metadata(item, seen))
                        for key, item in value.items()
                    ),
                    key=repr,
                )
            )
        finally:
            seen.remove(object_id)
        return items
    if isinstance(value, (list, tuple)):
        object_id = id(value)
        if object_id in seen:
            raise TypeDescValidationError("metadata contains a cycle")
        seen.add(object_id)
        try:
            return tuple(_freeze_metadata(item, seen) for item in value)
        finally:
            seen.remove(object_id)
    _canonical(value)
    try:
        hash(value)
    except TypeError as exc:
        raise TypeDescValidationError("metadata contains an unhashable value") from exc
    return cast(Hashable, value)


def _validate_metadata_key(key: Any) -> Hashable:
    """Validate a mapping key while retaining its semantic value."""
    _canonical(key)
    try:
        hash(key)
    except TypeError as exc:
        raise TypeDescValidationError("metadata contains an unhashable key") from exc
    return cast(Hashable, key)


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
        return (
            "typedesc",
            tuple(_canonical(getattr(value, field)) for field in value.__dataclass_fields__),
        )
    if isinstance(value, Mapping):
        items = ((_canonical(key), _canonical(item)) for key, item in value.items())
        return ("mapping", tuple(sorted(items, key=repr)))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_canonical(item) for item in value))
    if value is Ellipsis:
        return ("ellipsis",)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    try:
        hash(value)
    except TypeError as exc:
        raise TypeDescValidationError(f"unsupported canonical value: {type(value)!r}") from exc
    # Unsupported custom hashables have no portable value encoding. Preserve
    # only their object identity: retaining the raw object would make a
    # descriptor's hash change if that object's implementation mutates later.
    return ("hashable", type(value).__module__, type(value).__qualname__, id(value))


_SHAPE_CONTRACT_KINDS = frozenset(
    {
        "numpy.ndarray",
        "xarray.DataArray",
        "xarray.Dataset",
        "pandas.DataFrame",
        "pandas.Series",
        "polars.DataFrame",
        "polars.Series",
        "pyarrow.Array",
        "pyarrow.Table",
        "drjit.Array",
    }
)


def requires_shape_contract(type_desc: "TypeDesc") -> bool:
    """Return whether the type requires an explicit shape/schema contract."""
    return type_desc.kind in _SHAPE_CONTRACT_KINDS


@dataclass(frozen=True, eq=False)
class TypeDesc:
    """
    Universal type descriptor for heterogeneous data structures.

    Supports nominal backend/container kinds while keeping their structural
    payloads in one hash-stable schema.

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
        if self.dtypes is not None and self.columns is not None:
            declared = {column for column in self.columns if column is not ...}
            observed = {column for column, _ in self.dtypes}
            if not observed <= declared:
                raise TypeDescValidationError("dtypes contains a column not present in columns")
            if ... not in self.columns and observed != declared:
                raise TypeDescValidationError("dtypes must cover every declared column")
        if self.dtype is not None and not isinstance(self.dtype, str):
            raise TypeDescValidationError("dtype must be a string or None")
        if self.static_dims is not None:
            try:
                static_dims = tuple(self.static_dims)
            except TypeError as exc:
                raise TypeDescValidationError("static_dims must be a sequence") from exc
            if any(
                not isinstance(size, int) or isinstance(size, bool) or size < 0
                for size in static_dims
            ):
                raise TypeDescValidationError("static_dims must contain non-negative integers")
            object.__setattr__(self, "static_dims", static_dims)
        if self.drjit_type is not None and not isinstance(self.drjit_type, type):
            raise TypeDescValidationError("drjit_type must be a type or None")
        if self.kind == "scalar" and (self.shape is not None or self.dims is not None):
            raise TypeDescValidationError("scalar cannot declare dimensions or shape")
        payloads = {
            "scalar": {"dtype", "metadata"},
            "record": {"fields", "metadata"},
            "opaque": {"metadata"},
            "numpy.ndarray": {"dims", "shape", "dtype", "metadata"},
            "xarray.DataArray": {"dims", "shape", "dtype", "metadata"},
            "xarray.Dataset": {"dims", "fields", "metadata"},
            "pandas.Series": {"shape", "dtype", "index", "metadata"},
            "pandas.DataFrame": {"shape", "index", "columns", "dtypes", "metadata"},
            "polars.Series": {"shape", "dtype", "index", "metadata"},
            "polars.DataFrame": {"shape", "index", "columns", "dtypes", "metadata"},
            "pyarrow.Array": {"shape", "dtype", "metadata"},
            "pyarrow.Table": {"shape", "columns", "dtypes", "metadata"},
            "drjit.Array": {"shape", "dtype", "drjit_type", "static_dims", "metadata"},
        }
        allowed = payloads.get(self.kind)
        if allowed is not None:
            populated = {
                name
                for name in self.__dataclass_fields__
                if name not in {"kind", "metadata"} and getattr(self, name) is not None
            }
            illegal = populated - (allowed - {"metadata"})
            if illegal:
                raise TypeDescValidationError(
                    f"kind {self.kind!r} cannot declare {sorted(illegal)!r}"
                )

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
            elif isinstance(size, int) and not isinstance(size, bool) and size >= 0:
                result.append(size)
            else:
                raise TypeDescValidationError(
                    f"{path}[{index}] must be a non-negative integer or Symbol"
                )
        return tuple(result)

    @classmethod
    def _normalize_dims(cls, value: Any, path: str) -> Dims | None:
        if value is None:
            return None
        try:
            entries = tuple(tuple(entry) for entry in value)
        except TypeError as exc:
            raise TypeDescValidationError(f"{path} must be a sequence") from exc
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
            if (
                normalized_labels is not None
                and isinstance(size, int)
                and len(normalized_labels) != size
            ):
                raise TypeDescValidationError(f"{path}[{index}].labels length disagrees with size")
            result.append((name, size, normalized_labels))
        return tuple(result)

    @staticmethod
    def _validate_size(size: Any, path: str) -> None:
        if not isinstance(size, Symbol) and (
            not isinstance(size, int) or isinstance(size, bool) or size < 0
        ):
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
        if ... in result and (result[-1] is not ... or result.count(...) != 1):
            raise TypeDescValidationError(f"{path} ellipsis must be trailing")
        return result

    @staticmethod
    def _normalize_pairs(value: Any, path: str) -> tuple[tuple[Hashable, str], ...] | None:
        if value is None:
            return None
        result: list[tuple[Hashable, str]] = []
        keys: set[Hashable] = set()
        try:
            pairs = enumerate(value)
            for index, pair in pairs:
                if len(pair) != 2 or not isinstance(pair[1], str):
                    raise TypeDescValidationError(f"{path}[{index}] must be (hashable, str)")
                key, dtype = pair
                _canonical(key)
                try:
                    hash(key)
                except TypeError as exc:
                    raise TypeDescValidationError(f"{path}[{index}] key is unhashable") from exc
                if key in keys:
                    raise TypeDescValidationError(f"{path} contains duplicate key {key!r}")
                keys.add(key)
                result.append((key, dtype))
        except TypeDescValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise TypeDescValidationError(f"{path} must be a sequence of pairs") from exc
        return tuple(result)

    @classmethod
    def _normalize_fields(cls, value: Any) -> tuple[tuple[Hashable, "TypeDesc"], ...] | None:
        if value is None:
            return None
        result: list[tuple[Hashable, TypeDesc]] = []
        keys: set[Hashable] = set()
        try:
            pairs = enumerate(value)
            for index, pair in pairs:
                if len(pair) != 2 or not isinstance(pair[1], TypeDesc):
                    raise TypeDescValidationError(f"fields[{index}] must be (hashable, TypeDesc)")
                key, descriptor = pair
                _canonical(key)
                try:
                    hash(key)
                except TypeError as exc:
                    raise TypeDescValidationError(f"fields[{index}] key is unhashable") from exc
                if key in keys:
                    raise TypeDescValidationError(f"fields contains duplicate key {key!r}")
                keys.add(key)
                result.append((key, descriptor))
        except TypeDescValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise TypeDescValidationError("fields must be a sequence of pairs") from exc
        return tuple(result)

    @classmethod
    def _normalize_metadata(cls, value: Any) -> tuple[tuple[str, Hashable], ...]:
        items = value.items() if isinstance(value, Mapping) else value
        if items is None:
            return ()
        result: dict[str, Hashable] = {}
        try:
            pairs = enumerate(items)
            for index, pair in pairs:
                if len(pair) != 2 or not isinstance(pair[0], str):
                    raise TypeDescValidationError(f"metadata[{index}] must be (str, hashable)")
                key, raw = pair
                frozen = _freeze_metadata(raw)
                if key in result:
                    raise TypeDescValidationError(f"metadata contains duplicate key {key!r}")
                result[key] = frozen
        except TypeDescValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise TypeDescValidationError("metadata must be a sequence of pairs") from exc
        return tuple(sorted(result.items()))

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeDesc) and _canonical(self) == _canonical(other)

    def fingerprint(self) -> str:
        """Return a structural identity.

        Built-in scalar/container values have deterministic fingerprints. An
        arbitrary user-defined hashable label has no portable encoding, so it
        is intentionally identity-scoped to the Python process and descriptor
        lifetime; this preserves the stronger invariant that unequal labels
        cannot collapse in equality, hashing, or fingerprints.
        """
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
                nested_bindings = {
                    name: item for name, item in bindings.items() if name in value._symbols()
                }
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
            elif isinstance(value, TypeDesc):
                for field in value.__dataclass_fields__:
                    visit(getattr(value, field))
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

    def known_columns(self) -> list[Hashable] | None:
        """Return known columns, excluding the optional trailing ellipsis marker."""
        if self.columns is None:
            return None
        if self.columns and self.columns[-1] is ...:
            return list(self.columns[:-1])
        return list(self.columns)

    def with_dims(self, dims: Dims) -> "TypeDesc":
        """Return copy with updated dims."""
        return replace(self, dims=dims)

    def with_dtype(self, dtype: str) -> "TypeDesc":
        """Return copy with updated dtype."""
        return replace(self, dtype=dtype)

    def binary(self, other: "TypeDesc", operation: str) -> "TypeDesc":
        """Apply a pure binary structural operation."""
        from typetrace.patterns import _binary_type_desc

        try:
            return _binary_type_desc(self, other, operation)
        except TypeDescError as exc:
            raise _operation_error(exc, operation) from exc

    def unary(self, operation: str) -> "TypeDesc":
        """Apply a pure unary structural operation."""
        from typetrace.patterns import _unary_type_desc

        try:
            return _unary_type_desc(self, operation)
        except TypeDescError as exc:
            raise _operation_error(exc, operation) from exc

    def method(
        self,
        name: str,
        args: tuple[object, ...] = (),
        kwargs: Mapping[str, object] | None = None,
    ) -> "TypeDesc":
        """Apply a registered pure method operation."""
        from typetrace.patterns import _method_type_desc

        try:
            return _method_type_desc(self, name, args, kwargs or {})
        except TypeDescError as exc:
            raise _operation_error(exc, name) from exc

    def project(self, field: Hashable) -> "TypeDesc":
        """Project one record or Dataset field."""
        from typetrace.patterns import _project_type_desc

        try:
            return _project_type_desc(self, field)
        except TypeDescError as exc:
            raise _operation_error(exc, "project") from exc

    def select(self, fields: tuple[Hashable, ...]) -> "TypeDesc":
        """Select an ordered subset of record fields or table columns."""
        from typetrace.patterns import _select_type_desc

        try:
            return _select_type_desc(self, fields)
        except TypeDescError as exc:
            raise _operation_error(exc, "select") from exc

    def reduce(self, dimensions: tuple[str, ...], operation: str) -> "TypeDesc":
        """Reduce exactly the named dimensions with a finite operation."""
        from typetrace.patterns import _reduce_type_desc

        try:
            return _reduce_type_desc(self, dimensions, operation)
        except TypeDescError as exc:
            raise _operation_error(exc, operation) from exc

    def broadcast(self, other: "TypeDesc") -> "TypeDesc":
        """Combine descriptors using named-dimension broadcast rules."""
        from typetrace.patterns import _combine_type_desc

        try:
            return _combine_type_desc(self, other, append_disjoint=True)
        except TypeDescError as exc:
            raise _operation_error(exc, "broadcast") from exc

    def unify(self, other: "TypeDesc") -> "TypeDesc":
        """Unify compatible descriptors without adding dimensions."""
        from typetrace.patterns import _combine_type_desc

        try:
            return _combine_type_desc(self, other, append_disjoint=False)
        except TypeDescError as exc:
            raise _operation_error(exc, "unify") from exc

    def reshape(self, shape_or_dimensions: object) -> "TypeDesc":
        """Replace positional shape or named dimensions after validation."""
        from typetrace.patterns import _reshape_type_desc

        try:
            return _reshape_type_desc(self, shape_or_dimensions)
        except TypeDescError as exc:
            raise _operation_error(exc, "reshape") from exc

    def add_dim(self, name: str, size: DimValue, position: int | None = None) -> "TypeDesc":
        """Insert one unlabeled named dimension."""
        from typetrace.patterns import _add_dim_type_desc

        try:
            return _add_dim_type_desc(self, name, size, position)
        except TypeDescError as exc:
            raise _operation_error(exc, "add_dim") from exc

    def remove_dim(self, name: str) -> "TypeDesc":
        """Remove one named dimension."""
        from typetrace.patterns import _remove_dim_type_desc

        try:
            return _remove_dim_type_desc(self, name)
        except TypeDescError as exc:
            raise _operation_error(exc, "remove_dim") from exc

    def rename_axis(self, old: str, new: str) -> "TypeDesc":
        """Rename one named dimension."""
        from typetrace.patterns import _rename_axis_type_desc

        try:
            return _rename_axis_type_desc(self, old, new)
        except TypeDescError as exc:
            raise _operation_error(exc, "rename_axis") from exc

    @classmethod
    def from_value(cls, value: Any, *, _seen: set[int] | None = None) -> "TypeDesc":
        """
        Extract TypeDesc from a runtime value.

        Dispatches to appropriate adapter based on value type.
        Tracks visited objects to prevent infinite recursion on cycles.

        Handles:
        - Python scalars (int, float, str, bool) → TypeDesc(kind="scalar", dtype=...)
        - numpy arrays → TypeDesc(kind="numpy.ndarray", dtype=..., dims=...)
        - numpy scalars → TypeDesc(kind="scalar", dtype=...)
        - xarray/pandas/polars/arrow/drjit → dispatched to adapters
        - Other objects → introspected as class
        """
        from typetrace.adapters import adapter_for_value

        if _seen is not None and id(value) in _seen:
            return cls(kind="recursive")

        try:
            adapter = adapter_for_value(value)
            if getattr(adapter, "__name__", "") == "typetrace.adapters.core":
                from typetrace.runtime_utils import module_root

                is_builtin = isinstance(value, (bool, int, float, str, bytes, type(None), Mapping))
                is_numpy_scalar = module_root(value) == "numpy"
                if _seen is not None and not is_builtin and not is_numpy_scalar:
                    return cls._from_object(value, _seen=_seen)
            return adapter.infer(value)
        except AdapterUnavailableError:
            return cls._from_object(value, _seen=_seen)

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

        fields: list[tuple[str, TypeDesc]] = []
        for name in dir(value):
            if name.startswith("_"):
                continue
            try:
                attr = getattr(value, name)
            except (AttributeError, RuntimeError, ValueError):
                continue
            if callable(attr):
                continue
            fields.append((name, cls.from_value(attr, _seen=_seen)))
        return cls(kind="record", fields=tuple(fields))

    def make_sample(self) -> Any:
        """Create minimal runtime sample preserving this descriptor schema."""
        from typetrace.adapters import get_adapter

        return get_adapter(self.kind).make_sample(self)

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
        columns: tuple[Hashable, ...] | None = None,
        dtypes: tuple[tuple[Hashable, str], ...] | None = None,
        index: Dims | None = None,
        fields: tuple[tuple[Hashable, "TypeDesc"], ...] | None = None,
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
            >>> TypeDesc.for_type(xr.DataArray, dtype="float64", dims=(("x", 10, None),))
            TypeDesc(kind='xarray.DataArray', dtype='float64', ...)

            >>> TypeDesc.for_type(pd.DataFrame, columns=("a", "b"))
            TypeDesc(kind='pandas.DataFrame', columns=('a', 'b'), ...)
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
    ) -> str:
        """Map Python type to TypeDesc kind.

        Supports:
        - xarray: DataArray, Dataset → nominal xarray kind
        - numpy: ndarray → ``numpy.ndarray``
        - pandas/polars: DataFrame and Series → nominal backend kind
        - pyarrow: Table → ``pyarrow.Table``
        - drjit: any dr.* array type → ``drjit.Array``
        """
        module_root = concrete_type.__module__.split(".")[0]

        # xarray types
        if module_root == "xarray":
            if concrete_type.__name__ == "DataArray":
                return "xarray.DataArray"
            if concrete_type.__name__ == "Dataset":
                return "xarray.Dataset"

        # numpy
        if module_root == "numpy" and concrete_type.__name__ == "ndarray":
            return "numpy.ndarray"

        # pandas
        if module_root == "pandas":
            if concrete_type.__name__ == "DataFrame":
                return "pandas.DataFrame"
            if concrete_type.__name__ == "Series":
                return "pandas.Series"

        # polars
        if module_root == "polars":
            if concrete_type.__name__ == "DataFrame":
                return "polars.DataFrame"
            if concrete_type.__name__ == "Series":
                return "polars.Series"

        # pyarrow
        if module_root == "pyarrow":
            if concrete_type.__name__ == "Array":
                return "pyarrow.Array"
            if concrete_type.__name__ == "Table":
                return "pyarrow.Table"

        # drjit - any type from drjit module
        if module_root == "drjit":
            return "drjit.Array"

        # Fallback to the generic opaque noun for unknown types.
        return "opaque"
