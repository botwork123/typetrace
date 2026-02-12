"""
typetrace - Runtime type inference by execution for array libraries.

Core concepts:
- TypeDesc: Universal type descriptor for arrays, dataframes, and custom classes
- Symbol: Symbolic dimension bound at runtime
- infer: Run function on samples to discover output type
"""

from typetrace.core import Dims, Symbol, TypeDesc
from typetrace.execution_traits import ExecutionTraits, infer_execution_traits
from typetrace.inference import TypeContext, infer_types
from typetrace.layout_ops import (
    check_handoff_compatibility,
    concat_traits,
    normalize_handoff_traits,
    reshape_restack_traits,
    slice_view_traits,
    transpose_traits,
)
from typetrace.patterns import add_dim, broadcast, reduce_dim, unify

__all__ = [
    "TypeDesc",
    "Symbol",
    "Dims",
    "infer_types",
    "TypeContext",
    "ExecutionTraits",
    "infer_execution_traits",
    "check_handoff_compatibility",
    "slice_view_traits",
    "transpose_traits",
    "reshape_restack_traits",
    "concat_traits",
    "normalize_handoff_traits",
    "unify",
    "broadcast",
    "add_dim",
    "reduce_dim",
]

__version__ = "0.1.0"
