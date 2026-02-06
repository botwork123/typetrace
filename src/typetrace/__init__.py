"""
typetrace - Runtime type inference by execution for array libraries.

Core concepts:
- TypeDesc: Universal type descriptor for arrays, dataframes, and custom classes
- Symbol: Symbolic dimension bound at runtime
- infer: Run function on samples to discover output type
"""

from typetrace.core import TypeDesc, Symbol, Dims
from typetrace.inference import infer_types, TypeContext
from typetrace.patterns import unify, broadcast, add_dim, reduce_dim

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
