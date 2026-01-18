Hello agent. You are one of the most talented programmers of your generation.
# Contribution Guide

The project is managed with `uv` and `pyproject.toml`
**ALWAYS use `uv run` for all Python commands and tools.**

Examples:
- `uv run python script.py`
- `uv run pytest`
- `uv run mypy src/`
- `uv run ruff check .`

## Testing

- **Test Framework**: pytest
- **Test Location**: `tests/`
- **Running Tests**: `uv run pytest`
- **Running Specific Tests**: `uv run pytest tests/path/to/test_file.py`
- **Running with Coverage**: `uv run pytest --cov=src`

**Write testable code and add unit tests where it makes sense.**

Guidelines:
- Add unit tests for new functions, classes, and methods
- Test edge cases and error conditions
- Keep tests focused and isolated
- Use descriptive test names that explain what's being tested
- Mock external dependencies to keep tests fast and reliable
- Don't test trivial code or external libraries

## Code Quality

### Formatting and Linting

Use **ruff** for both formatting and linting:

```bash
# Check formatting and linting
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking

When appropriate, write **typed Python code** and check with **mypy**:

```bash
uv run mypy src/
```

Use type hints for:
- Function parameters and return values
- Class attributes
- Variables when the type is not obvious

## Documentation

### Docstrings

**ALWAYS update Python docstrings** when modifying code.

Follow Google-style or NumPy-style docstring conventions:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised
    """
```

### Sphinx Documentation

**ALWAYS update Sphinx documentation** when changing functionality.

- Documentation location: `docs/`
- Keep documentation **concise and clear**
- Update relevant `.rst` files when adding or modifying features
- Build docs locally to verify: `uv run sphinx-build -b html docs/source docs/_build/html`

## API Compatibility

### Breaking Changes

**Avoid interface-breaking changes unless necessary.**

When a breaking change is required:
1. **Clearly state to the user** that the change is breaking
2. **Specify that it requires a major version jump** (semantic versioning)
3. Document the migration path for users
4. Consider deprecation warnings before removal

Examples of breaking changes:
- Removing or renaming public functions, classes, or methods
- Changing function signatures (parameters, return types)
- Changing expected behavior of public APIs
- Removing or renaming configuration options

### Non-Breaking Changes

Prefer:
- Adding new optional parameters with defaults
- Adding new functions/classes/methods
- Deprecation warnings before removal
- Backward-compatible extensions

## Code Style

### Readability First

**Prefer readability over cleverness.**

Guidelines:
- Write clear, self-documenting code
- Use descriptive variable and function names
- Break complex logic into smaller, named functions
- Only add code comments for non-obvious business logic
- Avoid overly clever one-liners when they harm clarity
- Explicit is better than implicit

Example:
```python
# Good: Clear and readable
def calculate_total_price(items: list[Item]) -> float:
    subtotal = sum(item.price for item in items)
    tax = subtotal * TAX_RATE
    return subtotal + tax

# Avoid: Too clever
def calc(i): return sum(x.p for x in i) * 1.1
```

## Workflow Summary

When making changes:

1. ✅ Write or update code with type hints
2. ✅ Write unit tests for new functionality
3. ✅ Update docstrings
4. ✅ Update Sphinx documentation if functionality changed
5. ✅ Run formatters: `uv run ruff format .`
6. ✅ Run linting: `uv run ruff check --fix .`
7. ✅ Run type checking: `uv run mypy src/` (when appropriate)
8. ✅ Run tests: `uv run pytest`
9. ✅ Check if changes are breaking and inform user if major version bump needed
