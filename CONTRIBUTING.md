# Contributing to Epistemic Ecology Runtime (EER)

Thank you for your interest in contributing to **EER**. We welcome
contributions from researchers and developers in numerical optimization,
theoretical computer science, and graph theory.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/myosettnaing/epistemic-ecology-runtime.git
cd epistemic-ecology-runtime
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate           # Linux / macOS
# venv\Scripts\activate            # Windows
```

### 3. Install in editable mode

```bash
pip install -e ".[dev,bench]"
```

## Coding & Testing Guidelines

- **Linting & formatting** — we enforce style with `ruff`:

  ```bash
  ruff check .
  ruff format .
  ```

- **Test suite** — ensure all tests pass before submitting a PR:

  ```bash
  pytest --cov=eer --cov-report=term-missing tests/
  ```

- **Mathematical accuracy** — any modification to Hessian assembly or
  regularizer matrices (cascade, cycle, SCC) **must** be accompanied by a
  regression test under `tests/`. In particular, changes to
  `build_cycle_matrix_fundamental` must preserve the alternating-sign
  convention \(b_\sigma(v_k) = (-1)^k\).

## Submitting Pull Requests

1. Fork the repository and create a feature branch:

   ```bash
   git checkout -b feature/amazing-feature
   ```

2. Commit your changes with a clear message:

   ```bash
   git commit -m "Add feature X"
   ```

3. Push the branch and open a Pull Request against `main`.

## Reporting Issues

Please include:

- Python version (`python --version`)
- `numba`, `numpy`, `scipy` versions
- Minimal reproducible example
- Full traceback (if applicable)
