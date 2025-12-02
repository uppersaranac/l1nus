# l1nus: LLM fine tuning utilities

l1nus is a toolkit for fine tuning LLMs

<!-- toc -->
## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Install the environment](#install-the-environment)
  - [Verify the installation](#verify-the-installation)
- [Development Guide](#development-guide)
  - [Testing](#testing)
  - [Linting & formatting](#linting--formatting)
- [Resources](#resources)
<!-- tocstop -->

## Overview

Utilities to simplify fine tuning of LLMs, with an emphasis on teaching chemistry to LLMs. 
Input is jsonl formatted files, which are processed and used to train arbitrary LLMs in Hugging Face
safetensor format. Use of jsonl allows for flexible generation of arbitrary training tasks
independent of any software libraries.

## Getting Started

### Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for environment and dependency management
- Git for cloning the repository

### Install the environment

```bash
# Clone the repository
git clone https://github.com/reductionnist/l1nus.git
cd l1nus

# Create or update the virtual environment
uv sync

# Option B: activate the environment manually if you prefer
source .venv/bin/activate           # macOS/Linux
```

### Verify the installation

```bash
uv run python -m pytest -k smoke
```

## Development Guide

### Testing

```bash
uv run python -m pytest                 # Run the full suite
```

### Linting & formatting

```bash
uv run ruff check .
uv run ruff format .
```
## Resources

- [ChemData Team](https://chemdata.nist.gov/)
- [License](LICENSE.md)
