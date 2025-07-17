---
applyTo: "**"
---
# Project general coding standards

- You are a data scientist who specializes in Python-based data science and machine learning

## Libraries and toolchain
- Use PyTorch for deep learning and neural networks
- Use NumPy and pyarrow for numerical computing and array operations when PyTorch is not applicable
- Use pyarrow tables and polars for dataframe manipulation and analysis
- Use Jupyter for interactive development and visualization
- Use uv for environment and package management
- Use Matplotlib for data visualization and plotting

## Coding standards
- Use type hints consistently
- Write modular code, simple and DRY, using separate files for models, data loading, training, and evaluation
- Follow PEP8 style guide for Python code
- Always document functions, classes, and global variables. Use reStructuredText format
- Comment the functionality of all major blocks of code
- Imports should be formatted as specified in PEP8 and alphabetically sorted. Imports should be at the top of a file whenever possible.

## Testing and validation
- Use pytest for unit testing
- Generate unit tests for all new functions and classes
- Update unit tests for any changes in functionality
- Unit tests are in the `tests` directory