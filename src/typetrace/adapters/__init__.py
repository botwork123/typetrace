"""
Adapters for various array/dataframe libraries.

Each adapter provides:
- from_X(value) -> TypeDesc: Extract type from value
- make_X_sample(type_desc) -> X: Create minimal sample for inference
"""
