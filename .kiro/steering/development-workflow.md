# Development Workflow Guidelines

## Development Environment Setup

### Virtual Environment
- Always use virtual environments for development
- Use `python -m venv venv` for consistency
- Include activation instructions in documentation
- Add `.venv/` and `venv/` to `.gitignore`

### Dependencies Management
- Use `requirements.txt` for production dependencies
- Use `requirements-dev.txt` for development tools
- Pin exact versions in production requirements
- Include version ranges in development requirements
- Update dependencies regularly and test compatibility

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting (configured for 2-space indentation)
- **flake8**: Linting and style checking
- **mypy**: Type checking
- **coverage**: Code coverage reporting

## Git Workflow

### Branch Naming
- `feature/description`: New features
- `bugfix/description`: Bug fixes
- `refactor/description`: Code refactoring
- `docs/description`: Documentation updates

### Commit Messages
- Use conventional commit format
- Examples:
  - `feat: add OpenAI provider integration`
  - `fix: handle missing API key gracefully`
  - `docs: update installation instructions`
  - `test: add questionnaire loader tests`

### Pull Request Process
1. Create feature branch from main
2. Implement changes with tests
3. Run full test suite locally
4. Update documentation if needed
5. Create PR with descriptive title and description
6. Address review feedback
7. Squash and merge when approved

## Testing Standards

### Test Organization
- Mirror source code structure in `tests/` directory
- Use descriptive test file names: `test_questionnaire_loader.py`
- Group related tests in classes when appropriate
- Use fixtures for common test data

### Test Naming Convention
```python
def test_load_questionnaire_returns_valid_structure():
  """Test that questionnaire loader returns expected structure."""
  pass

def test_load_questionnaire_raises_error_for_invalid_yaml():
  """Test that loader raises appropriate error for malformed YAML."""
  pass
```

### Test Coverage Requirements
- Maintain >90% code coverage
- Focus on critical paths and error handling
- Use coverage reports to identify gaps
- Exclude trivial code from coverage requirements

### Mock Strategy
- Mock external API calls in unit tests
- Use `pytest-mock` for mocking
- Create reusable mock fixtures
- Test both success and failure scenarios

## Code Quality Standards

### Type Hints
- Use type hints for all function parameters and return values
- Import types from `typing` module when needed
- Use `Optional` for nullable parameters
- Define custom types for complex data structures

### Documentation
- Include docstrings for all public functions and classes
- Use Google-style docstrings
- Document complex algorithms and business logic
- Keep README.md updated with new features

### Error Handling
- Use specific exception types
- Create custom exceptions for domain-specific errors
- Log errors with appropriate context
- Provide helpful error messages to users

## Configuration Management

### YAML Configuration
- Use consistent indentation (2 spaces)
- Validate configuration schemas for both questionnaires and LLM configs
- Provide example configuration files in both folders
- Document all configuration options
- Support multiple questionnaire files in `config/questionnaires/`
- Support multiple LLM config files in `config/llms/` with default merging
- Test configuration loading and merging logic

### Environment Variables
- Use uppercase with underscores: `OPENAI_API_KEY`
- Provide default values when appropriate
- Document required environment variables
- Use `.env.example` file for reference

## CLI Development

### Command Structure
- Use Click framework for CLI implementation
- Provide helpful help text for all commands and options
- Use consistent option naming conventions
- Support both short and long option forms
- Support listing available questionnaires and LLM configurations
- Handle multiple questionnaire selection
- Support configuration directory discovery

### User Experience
- Provide progress indicators for long-running operations
- Use colored output for better readability
- Implement verbose mode for debugging
- Handle keyboard interrupts gracefully

### Output Formats
- Default to human-readable output
- Support JSON output for programmatic use
- Use consistent formatting across commands
- Provide summary information

## Performance Guidelines

### API Rate Limiting
- Implement exponential backoff for retries
- Respect provider rate limits
- Use connection pooling for HTTP clients
- Monitor and log API usage

### Memory Usage
- Stream large datasets when possible
- Clean up resources in finally blocks
- Use generators for large sequences
- Monitor memory usage in tests

### Async Operations
- Use asyncio for concurrent API calls
- Implement proper error handling in async code
- Use semaphores to limit concurrent requests
- Handle timeouts appropriately

## Release Process

### Version Management
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `setup.py` and `__init__.py`
- Create git tags for releases
- Maintain CHANGELOG.md

### Pre-release Checklist
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version numbers are bumped
- [ ] CHANGELOG.md is updated
- [ ] Example configurations are tested
- [ ] Performance benchmarks are run

### Release Notes
- Highlight new features and improvements
- Document breaking changes
- Include migration instructions when needed
- Acknowledge contributors

## Debugging and Troubleshooting

### Logging Strategy
- Use structured logging with JSON format
- Include request IDs for tracing
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Avoid logging sensitive information

### Debug Mode
- Implement verbose/debug flags
- Provide detailed error messages in debug mode
- Include stack traces for unexpected errors
- Log configuration values (excluding secrets)

### Common Issues
- Document common setup issues and solutions
- Provide troubleshooting guides
- Include FAQ section in documentation
- Monitor and address user-reported issues