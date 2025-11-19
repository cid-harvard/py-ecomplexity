# Contributing to py-ecomplexity

Thank you for your interest in contributing to py-ecomplexity!

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Git Workflow](#git-workflow)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone git@github.com:YOUR-USERNAME/py-ecomplexity.git
   cd py-ecomplexity
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream git@github.com:cid-harvard/py-ecomplexity.git
   ```

## Development Environment Setup

1. Install dependencies using uv:
   ```bash
   uv sync
   ```

2. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

   This will automatically run linting, formatting, and type checking before each commit.

3. Verify your setup:
   ```bash
   uv run pytest
   uv run ruff check .
   uv run mypy .
   ```

## Code Style Guidelines

We follow standard Python best practices and enforce them through automated tooling:

### Python Style

- **Python Version**: 3.9+ with type hints required throughout
- **Line Length**: 88 characters (Black/Ruff default)
- **String Quotes**: Double quotes preferred
- **Formatting**: Automated via `ruff format`
- **Linting**: Automated via `ruff check`
- **Type Checking**: Enforced via `mypy`

### Documentation

- Use **Google-style docstrings** for all public functions, classes, and modules
- Include type hints in function signatures
- Document parameters, return values, and exceptions
- Example:
  ```python
  def calculate_complexity(data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
      """Calculate economic complexity indices.

      Args:
          data: DataFrame containing trade data with location, product, and value columns.
          threshold: RCA threshold for determining product presence. Defaults to 1.0.

      Returns:
          DataFrame with added ECI and PCI columns.

      Raises:
          ValueError: If required columns are missing from data.
      """
      pass
  ```

### Import Organization

Organize imports in the following order:
1. Future imports (e.g., `from __future__ import annotations`)
2. Standard library imports
3. Third-party imports
4. First-party imports (e.g., `from ecomplexity import ...`)
5. Local folder imports

Use `ruff` to automatically organize imports.

### Code Quality

- Implement comprehensive error handling with meaningful error messages
- Use the `logging` module for any diagnostic output
- Write defensive code that validates inputs
- Avoid premature optimization; prioritize readability

## Testing

We prioritize integration tests over unit tests, focusing on core logical components.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ecomplexity

# Run specific test file
uv run pytest tests/path/to/test_file.py

# Run specific test function
uv run pytest tests/path/to/test_file.py::test_function
```

### Writing Tests

- Place tests in the `tests/` directory, mirroring the package structure
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Test both happy paths and edge cases
- Use pytest fixtures for common setup
- Focus on testing public APIs and integration points

Example test structure:
```python
import pytest
import pandas as pd
from ecomplexity import ecomplexity

def test_ecomplexity_basic_calculation():
    """Test that ecomplexity runs successfully on valid input."""
    # Setup
    data = create_sample_data()
    trade_cols = {'time': 'year', 'loc': 'country', 'prod': 'product', 'val': 'value'}

    # Execute
    result = ecomplexity(data, trade_cols)

    # Assert
    assert 'eci' in result.columns
    assert 'pci' in result.columns
    assert not result['eci'].isna().all()
```

## Git Workflow

We use a Git Flow-style workflow with two main branches:

- **`master`**: Stable releases only
- **`develop`**: Integration branch for ongoing development

### Contributing Changes

1. **Sync with upstream**:
   ```bash
   git checkout develop
   git fetch upstream
   git merge upstream/develop
   ```

2. **Create a feature branch** from `develop`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

   Use descriptive branch names:
   - `feature/add-new-complexity-metric`
   - `fix/proximity-calculation-bug`
   - `docs/improve-api-documentation`

3. **Make your changes**:
   - Write code following style guidelines
   - Add/update tests
   - Update documentation as needed
   - Commit with clear, descriptive messages

4. **Commit messages**:
   - Use present tense: "Add feature" not "Added feature"
   - Be descriptive but concise
   - Reference issue numbers when applicable
   - Example: `Fix density calculation for KNN method (#42)`

5. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

6. **Run pre-commit checks**:
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run mypy .
   uv run pytest
   ```

## Submitting Changes

### Before Submitting a Pull Request

Ensure your contribution meets these criteria:

- [ ] Code follows style guidelines (enforced by pre-commit hooks)
- [ ] All tests pass locally
- [ ] Type hints are included for new functions
- [ ] Documentation is updated (docstrings, README if needed)
- [ ] No new warnings from mypy or ruff
- [ ] Commit messages are clear and descriptive

### Pull Request Process

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Target the `develop` branch (not `master`)
   - Fill out the PR template
   - Link related issues
   - Provide a clear description of changes

3. **Code Review**:
   - Respond to feedback promptly
   - Make requested changes in new commits (don't force push during review)
   - Update tests and documentation as requested

4. **Merge**:
   - Maintainers will merge your PR when approved
   - Your contribution will be included in the next release

### Pull Request Guidelines

- **Keep PRs focused**: One feature or fix per PR
- **Write clear descriptions**: Explain what, why, and how
- **Reference issues**: Use "Fixes #123" or "Closes #123"
- **Update documentation**: Include docstring and README changes
- **Add tests**: Especially for bug fixes and new features

## Release Process

Releases are managed by maintainers:

1. Changes accumulate on `develop`
2. When ready for release, `develop` is merged to `master`
3. Version number is updated in `pyproject.toml`
4. Release is tagged and published to PyPI

Contributors don't need to update version numbers or manage releases.

## Getting Help

- **Issues**: Check [existing issues](https://github.com/cid-harvard/py-ecomplexity/issues) or create a new one
- **Discussions**: Use GitHub Issues for questions and discussions
- **Documentation**: See [README.md](README.md) for usage documentation

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code. Please report unacceptable behavior to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Recognition

Contributors are recognized through:
- GitHub contributor statistics
- Release notes acknowledgments
- Author attribution in significant contributions

Thank you for contributing to py-ecomplexity!
