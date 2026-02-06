"""
typetrace - Runtime type inference by execution for array libraries.

Core concepts:
- TypeDesc: Universal type descriptor for arrays, dataframes, and custom classes
- Symbol: Symbolic dimension bound at runtime
- infer: Run function on samples to discover output type
"""

from typetrace.core import Dims, Symbol, TypeDesc
from typetrace.inference import TypeContext, infer_types
from typetrace.patterns import add_dim, broadcast, reduce_dim, unify

__all__ = [
    "TypeDesc",
    "Symbol",
    "Dims",
    "infer_types",
    "TypeContext",
    "unify",
    "broadcast",
    "add_dim",
    "reduce_dim",
]

__version__ = "0.1.0"
