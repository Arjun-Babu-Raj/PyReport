# Contributing to pyreport

Thank you for considering a contribution to **pyreport**! This document explains how to get set up, what we expect, and how to submit changes.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Running the Tests](#running-the-tests)
5. [Coding Conventions](#coding-conventions)
6. [Submitting Changes](#submitting-changes)
7. [Reporting Bugs](#reporting-bugs)
8. [Requesting Features](#requesting-features)

---

## Code of Conduct

All contributors are expected to be respectful and constructive. Harassment of any kind will not be tolerated.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork:

   ```bash
   git clone https://github.com/<your-username>/PyReport.git
   cd PyReport
   ```

3. Create a **feature branch**:

   ```bash
   git checkout -b feature/my-improvement
   ```

---

## Development Setup

Install the package in editable mode with all development dependencies:

```bash
pip install -e ".[dev]"
```

Optional extras (needed for pingouin-based reporters):

```bash
pip install -e ".[dev,pingouin]"
```

---

## Running the Tests

```bash
pytest tests/ --cov=pyreport
```

All pull requests must pass the full test suite. New functionality should include corresponding tests. We aim to maintain **≥ 90 % line coverage**.

---

## Coding Conventions

- **Python 3.9+** syntax only.
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- All public functions and classes must have docstrings.
- Statistical output text must conform to **APA 7th edition** formatting rules (see `pyreport/formatters.py`).
- Avoid adding new required dependencies without discussion. Optional dependencies (e.g., `pingouin`) should remain optional.

---

## Submitting Changes

1. Ensure all tests pass locally (`pytest tests/ --cov=pyreport`).
2. Push your branch and open a **Pull Request** against `main`.
3. Fill in the PR description with:
   - A summary of what changed and why.
   - Any related issue numbers.
   - Notes on new tests added.
4. A maintainer will review and may request changes.

---

## Adding a New Reporter

If you are adding support for a new statistical test or object type:

1. Create `pyreport/reporters/<name>.py` subclassing `BaseReporter`.
2. Add a type-detection helper in `pyreport/utils.py`.
3. Register the new type in `pyreport/core.py`'s `report()` dispatcher.
4. Add tests in `tests/test_reporters.py` or a new file.

---

## Reporting Bugs

Please open a [GitHub Issue](https://github.com/Arjun-Babu-Raj/PyReport/issues) and include:

- Python version and OS.
- Minimal reproducible example.
- Full traceback.

---

## Requesting Features

Open a GitHub Issue with the label **enhancement** describing:

- The use case.
- The statistical test or object type involved.
- Example of the expected output text.
