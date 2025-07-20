# Python Coding Standards

## Code Style and Formatting

### Indentation
- Use 2 spaces for indentation (not the standard 4 spaces)
- Never mix tabs and spaces
- Be consistent throughout the entire codebase

### Line Length
- Maximum line length: 88 characters (Black formatter default)
- Break long lines using parentheses for natural line continuation
- Prefer breaking after operators rather than before

### Imports
- Group imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library imports
- Separate each group with a blank line
- Use absolute imports when possible
- Avoid wildcard imports (`from module import *`)

```python
import os
import sys
from pathlib import Path

import requests
import pandas as pd

from myapp.utils import helper_function
from myapp.models import User
```

### Naming Conventions
- **Variables and functions**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Classes**: PascalCase
- **Private attributes/methods**: prefix with single underscore
- **Modules**: lowercase with underscores if needed

### String Formatting
- Prefer f-strings for string interpolation
- Use double quotes for strings by default
- Use single quotes for strings containing double quotes

```python
name = "Alice"
message = f"Hello, {name}!"
sql_query = 'SELECT * FROM users WHERE name = "Alice"'
```

## Code Organization

### Function and Class Structure
- Keep functions small and focused on a single responsibility
- Use type hints for function parameters and return values
- Include docstrings for all public functions and classes

```python
def calculate_total(items: list[dict]) -> float:
  """Calculate the total price of items.
  
  Args:
    items: List of item dictionaries with 'price' key
    
  Returns:
    Total price as float
  """
  return sum(item["price"] for item in items)
```

### Error Handling
- Use specific exception types rather than bare `except:`
- Handle exceptions at the appropriate level
- Use context managers (`with` statements) for resource management

### Documentation
- Use Google-style docstrings
- Include type information in docstrings when type hints aren't sufficient
- Document complex algorithms and business logic

## Testing Standards

### Test Structure
- Use pytest as the testing framework
- Follow the Arrange-Act-Assert pattern
- Use descriptive test names that explain the scenario

```python
def test_calculate_total_returns_sum_of_item_prices():
  # Arrange
  items = [{"price": 10.0}, {"price": 15.5}]
  
  # Act
  result = calculate_total(items)
  
  # Assert
  assert result == 25.5
```

### Test Organization
- Mirror the source code structure in test directories
- Use fixtures for common test data
- Group related tests in classes when appropriate

## Dependencies and Environment

### Package Management
- Use `requirements.txt` for production dependencies
- Use `requirements-dev.txt` for development dependencies
- Pin exact versions for production, allow ranges for development

### Virtual Environments
- Always use virtual environments for projects
- Include `.venv/` in `.gitignore`
- Document environment setup in README

## Security Best Practices

### Input Validation
- Validate and sanitize all user inputs
- Use parameterized queries for database operations
- Never execute user-provided code directly

### Secrets Management
- Never commit secrets to version control
- Use environment variables for configuration
- Use dedicated secret management tools for production

## Performance Considerations

### General Guidelines
- Profile before optimizing
- Use appropriate data structures (sets for membership tests, etc.)
- Consider memory usage for large datasets
- Use generators for large sequences when possible

### Database Operations
- Use connection pooling
- Implement proper indexing strategies
- Avoid N+1 query problems

## File Structure

### Project Layout
```
project/
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── main.py
│       ├── models/
│       ├── services/
│       └── utils/
├── tests/
├── docs/
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

### Configuration
- Keep configuration separate from code
- Use environment-specific config files
- Validate configuration on startup