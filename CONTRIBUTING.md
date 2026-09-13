# Contributing to EER

Thanks for your interest in the Epistemic Ecology Runtime. This document
explains how to report bugs, propose features, and submit pull requests.

---

## Reporting Bugs

Open a [GitHub issue](https://github.com/myosettnaing1992-alt/Epistemic-Ecology-Runtime/issues)
with:

1. **Environment**: output of `python --version`, `pip list | grep -E "numpy|scipy|numba|networkx"`, and OS.
2. **Minimal reproduction**: a small Python snippet that triggers the bug.
3. **Expected vs actual**: what you expected and what you observed.
4. **Traceback**: the full error message if applicable.

For **performance regressions**, include the output of:

```bash
python benchmarks/run_benchmarks.py --smoke --quiet
