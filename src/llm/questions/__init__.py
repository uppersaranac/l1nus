"""Question generation package.

Contains generic, domain-agnostic utilities for generating question/answer pairs
from raw data and declarative YAML configuration files.

At runtime this package is **not** aware of chemistry or any other domain; all
information about how to form a question is supplied via the YAML config passed
to `cli_generate.py`.
"""
