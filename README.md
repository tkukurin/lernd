# LERND - Learning Explanatory Rules from Noisy Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14630371.svg)](https://doi.org/10.5281/zenodo.14630371)

JAX implementation of differentiable Inductive Logic Programming.
Based on the TensorFlow implementation [crunchiness/lernd](https://github.com/crunchiness/lernd).

## Installation

Requires Python 3.12+.

```bash
# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Usage

```bash
# Run predecessor problem
uv run python -m lernd.experiments predecessor

# Run even number problem
uv run python -m lernd.experiments even
```

Or use as a library:

```python
from lernd.experiments import setup_predecessor
from lernd.lernd_loss import Lernd
from lernd.main import main_loop

ilp, program_template = setup_predecessor()
main_loop(ilp, program_template, steps=100)
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run linter
uv run ruff check lernd/

# Run type checker
uv run mypy lernd/
```

## References

- Evans, R., & Grefenstette, E. (2018). Learning Explanatory Rules from Noisy Data. *Journal of Artificial Intelligence Research*, 61, 1-64.
