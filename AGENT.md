# Agent Guidelines

This document consolidates all development guidelines and standards for the rationale benchmark project. These guidelines ensure consistency, quality, and maintainability across the codebase.

## Python Coding Standards

### Code Style and Formatting

#### Indentation
- Use 2 spaces for indentation (not the standard 4 spaces)
- Never mix tabs and spaces
- Be consistent throughout the entire codebase

#### Line Length
- Maximum line length: 88 characters (Black formatter default)
- Break long lines using parentheses for natural line continuation
- Prefer breaking after operators rather than before

#### Imports
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

#### Naming Conventions
- **Variables and functions**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Classes**: PascalCase
- **Private attributes/methods**: prefix with single underscore
- **Modules**: lowercase with underscores if needed

#### String Formatting
- Prefer f-strings for string interpolation
- Use double quotes for strings by default
- Use single quotes for strings containing double quotes

```python
name = "Alice"
message = f"Hello, {name}!"
sql_query = 'SELECT * FROM users WHERE name = "Alice"'
```

### Code Organization

#### Function and Class Structure
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

#### Error Handling
- Use specific exception types rather than bare `except:`
- Handle exceptions at the appropriate level
- Use context managers (`with` statements) for resource management

#### Documentation
- Use Google-style docstrings
- Include type information in docstrings when type hints aren't sufficient
- Document complex algorithms and business logic

### Testing Standards

#### Test-First Development
- **MANDATORY**: When generating tasks, always create unit tests and mock tests as subtasks before actual code implementation
- **CRITICAL**: Tests must be created as subtasks before any implementation work begins
- Write tests before implementing functionality
- Use Test-Driven Development (TDD) approach where appropriate
- **ALWAYS** check `pyproject.toml` for existing test dependencies before adding new ones with `uv add --dev`

#### Test Structure
- Use pytest as the testing framework (check `pyproject.toml` first, add with `uv add --dev pytest` if not present)
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

#### Test Organization
- Mirror the source code structure in test directories
- Use fixtures for common test data
- Group related tests in classes when appropriate

### Dependencies and Environment

#### Package Management
- **ALWAYS** use `uv` for dependency management with `pyproject.toml` - never use `pip`
- **CRITICAL**: Before writing any code, always check whether dependencies have already been added to `pyproject.toml`
- **MANDATORY**: Check `pyproject.toml` for existing dependencies before adding new ones
- Use `uv add <package>` to add new production dependencies
- Use `uv add --dev <package>` for development dependencies
- Pin exact versions for production, allow ranges for development
- Use `uv sync` to install dependencies from lock file
- Use `uv run <command>` to execute commands in the virtual environment

#### Virtual Environments
- Use `uv venv` to create virtual environments
- Use `uv run` to execute commands in the virtual environment
- Include `.venv/` in `.gitignore`
- Document environment setup in README

### Security Best Practices

#### Input Validation
- Validate and sanitize all user inputs
- Use parameterized queries for database operations
- Never execute user-provided code directly

#### Secrets Management
- Never commit secrets to version control
- Use environment variables for configuration
- Use dedicated secret management tools for production

### Performance Considerations

#### General Guidelines
- Profile before optimizing
- Use appropriate data structures (sets for membership tests, etc.)
- Consider memory usage for large datasets
- Use generators for large sequences when possible

#### Database Operations
- Use connection pooling
- Implement proper indexing strategies
- Avoid N+1 query problems

### File Structure

#### Project Layout
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
├── pyproject.toml
├── uv.lock
└── README.md
```

#### Configuration
- Keep configuration separate from code
- Use environment-specific config files
- Validate configuration on startup

## Project Architecture Guidelines

### Project Overview
This is a rationale benchmark tool for evaluating LLMs based on human-like reasoning patterns. The architecture follows a modular design with clear separation of concerns.

### Core Modules

#### CLI Module (`cli.py`)
- Entry point for command-line interface
- Handle argument parsing and validation for multiple questionnaires and LLM configs
- Support listing available questionnaires and LLM configurations
- Coordinate between questionnaire loading, LLM clients, and benchmark execution
- Use Click framework for CLI implementation
- Support running single or multiple questionnaires
- Handle configuration directory discovery and validation

#### Questionnaire Module (`questionnaire/`)
- **loader.py**: YAML questionnaire parsing and loading from multiple files
- **validator.py**: Questionnaire structure validation
- Support loading questionnaires by filename (without extension)
- Support loading multiple questionnaires in a single execution
- Support nested sections and multiple question types
- Validate required fields and data types
- Discover and list available questionnaire files

#### LLM Module (`llm/`)
- **client.py**: Abstract LLM client interface
- **providers/**: Provider-specific implementations (OpenAI, Anthropic, local)
- Handle API authentication and rate limiting
- Standardize response formats across providers

#### Benchmark Module (`benchmark/`)
- **runner.py**: Execute benchmarks across multiple models
- **evaluator.py**: Analyze and score LLM responses
- Support parallel execution when possible
- Generate structured results

### Configuration Management

#### Multi-File Configuration Structure
- Use `config/questionnaires/` folder for multiple questionnaire definitions
- Use `config/llms/` folder for multiple LLM provider configurations
- Each questionnaire file represents one complete benchmark
- Support default LLM configuration with custom overrides
- Support environment variable substitution
- Validate configuration schemas on load

#### Questionnaire Configuration
- Each YAML file in `config/questionnaires/` defines one questionnaire
- Files are referenced by filename without extension
- Support loading multiple questionnaires in a single run
- Validate questionnaire structure and required fields

#### LLM Configuration
- `default-llms.yaml` provides base configuration and defaults
- Additional YAML files can extend or override default settings
- Support provider-specific configurations
- Merge configurations with custom files taking precedence
- Files are referenced by filename without extension

#### Environment Variables
- Use for sensitive data (API keys)
- Follow naming convention: `{PROVIDER}_API_KEY`
- Document all required environment variables
- Support environment variable substitution in all config files

### Data Models

#### Questionnaire Structure
```python
@dataclass
class Question:
  id: str
  type: str  # "scenario", "choice", "open"
  prompt: str
  options: Optional[list[str]] = None
  expected_reasoning: Optional[str] = None
  bias_type: Optional[str] = None

@dataclass
class Section:
  name: str
  questions: list[Question]

@dataclass
class Questionnaire:
  name: str
  description: str
  sections: list[Section]
```

#### LLM Response Structure
```python
@dataclass
class LLMResponse:
  model: str
  question_id: str
  response_text: str
  reasoning: Optional[str] = None
  confidence: Optional[float] = None
  timestamp: datetime
  latency_ms: int
```

### Error Handling Strategy

#### Configuration Errors
- Validate YAML structure on load
- Provide clear error messages for missing fields
- Fail fast with descriptive error messages

#### API Errors
- Implement retry logic with exponential backoff
- Handle rate limiting gracefully
- Log API errors with context

#### Benchmark Execution Errors
- Continue execution if single model fails
- Collect and report all errors at the end
- Provide partial results when possible

### Output and Results

#### JSON Output Format
```json
{
  "benchmark_info": {
    "questionnaires": ["string"],
    "llm_config": "string",
    "execution_timestamp": "ISO8601",
    "models_tested": ["string"]
  },
  "results": [
    {
      "questionnaire": "string",
      "model": "string",
      "question_id": "string",
      "response": "string",
      "evaluation_score": "float",
      "reasoning_alignment": "float"
    }
  ],
  "summary": {
    "questionnaires_run": "int",
    "total_questions": "int",
    "models_tested": "int",
    "average_scores_by_questionnaire": {},
    "average_scores_by_model": {}
  }
}
```

#### Logging
- Use structured logging with JSON format
- Log levels: DEBUG, INFO, WARNING, ERROR
- Include execution context in log messages
- Separate log files for different components

### Future Extensibility

#### Web Interface Preparation
- Keep business logic separate from CLI
- Design APIs that can be exposed via REST
- Use dependency injection for easier testing

#### Plugin Architecture
- Design provider interface for easy extension
- Support custom evaluators
- Allow custom question types

### Performance Considerations

#### Async Operations
- Use asyncio for concurrent LLM API calls
- Implement connection pooling for HTTP clients
- Handle rate limiting across concurrent requests

#### Memory Management
- Stream large result sets when possible
- Implement pagination for large questionnaires
- Clean up resources properly

### Security Guidelines

#### API Key Management
- Never log API keys
- Use environment variables only
- Validate API key format before use

#### Input Validation
- Sanitize all user inputs
- Validate file paths and prevent directory traversal
- Limit file sizes for uploaded questionnaires

## LLM Integration Guidelines

### Provider Integration Standards

#### Abstract Provider Interface
All LLM providers must implement a common interface to ensure consistency:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class LLMProvider(ABC):
  """Abstract base class for LLM providers."""
  
  @abstractmethod
  async def generate_response(
    self, 
    prompt: str, 
    model: str,
    **kwargs
  ) -> LLMResponse:
    """Generate response from LLM model."""
    pass
  
  @abstractmethod
  def list_available_models(self) -> List[str]:
    """Return list of available models for this provider."""
    pass
  
  @abstractmethod
  def validate_config(self) -> bool:
    """Validate provider configuration."""
    pass
```

#### Provider Configuration
Each provider should support standardized configuration:

```yaml
providers:
  provider_name:
    api_key: "${PROVIDER_API_KEY}"
    base_url: "https://api.provider.com"  # Optional
    timeout: 30  # seconds
    max_retries: 3
    models:
      - "model-name-1"
      - "model-name-2"
    default_params:
      temperature: 0.7
      max_tokens: 1000
```

### Supported Providers

#### OpenAI Integration
- Support GPT-4, GPT-3.5-turbo, and future models
- Handle rate limiting (RPM and TPM limits)
- Implement proper error handling for API errors
- Support streaming responses for long outputs

#### Anthropic Integration
- Support Claude models (Opus, Sonnet, Haiku)
- Handle Anthropic-specific rate limits
- Implement proper message formatting
- Support system prompts appropriately

#### Local Model Integration
- Support local inference servers (Ollama, vLLM, etc.)
- Handle connection errors gracefully
- Support custom model names and parameters
- Implement health checks for local servers

#### Custom Provider Support
- Allow easy addition of new providers
- Provide template for new provider implementations
- Support provider-specific parameters
- Document provider integration process

### Request/Response Handling

#### Request Standardization
```python
@dataclass
class LLMRequest:
  prompt: str
  model: str
  temperature: float = 0.7
  max_tokens: int = 1000
  system_prompt: Optional[str] = None
  stop_sequences: Optional[List[str]] = None
  provider_specific: Dict[str, Any] = field(default_factory=dict)
```

#### Response Standardization
```python
@dataclass
class LLMResponse:
  text: str
  model: str
  provider: str
  timestamp: datetime
  latency_ms: int
  token_count: Optional[int] = None
  finish_reason: Optional[str] = None
  cost_estimate: Optional[float] = None
  metadata: Dict[str, Any] = field(default_factory=dict)
```

### Error Handling

#### API Error Categories
- **Authentication Errors**: Invalid API keys, expired tokens
- **Rate Limit Errors**: Too many requests, quota exceeded
- **Model Errors**: Model not found, model overloaded
- **Network Errors**: Connection timeouts, DNS failures
- **Validation Errors**: Invalid parameters, malformed requests

#### Retry Strategy
```python
async def make_request_with_retry(
  provider: LLMProvider,
  request: LLMRequest,
  max_retries: int = 3
) -> LLMResponse:
  """Make LLM request with exponential backoff retry."""
  for attempt in range(max_retries + 1):
    try:
      return await provider.generate_response(request)
    except RateLimitError:
      if attempt == max_retries:
        raise
      wait_time = 2 ** attempt
      await asyncio.sleep(wait_time)
    except (NetworkError, TemporaryError):
      if attempt == max_retries:
        raise
      await asyncio.sleep(1)
```

#### Error Logging
- Log all API errors with context
- Include request parameters (excluding sensitive data)
- Log retry attempts and backoff times
- Provide actionable error messages to users

### Rate Limiting and Quotas

#### Rate Limit Management
- Track requests per minute/hour for each provider
- Implement token bucket or sliding window algorithms
- Queue requests when approaching limits
- Provide rate limit status in responses

#### Cost Tracking
- Estimate costs for each request when possible
- Track token usage across models
- Provide cost summaries in benchmark results
- Warn users about high-cost operations

#### Concurrent Request Handling
```python
import asyncio
from asyncio import Semaphore

class RateLimitedProvider:
  def __init__(self, provider: LLMProvider, max_concurrent: int = 5):
    self.provider = provider
    self.semaphore = Semaphore(max_concurrent)
  
  async def generate_response(self, request: LLMRequest) -> LLMResponse:
    async with self.semaphore:
      return await self.provider.generate_response(request)
```

### Prompt Engineering

#### Prompt Templates
Create reusable prompt templates for different question types:

```python
SCENARIO_PROMPT_TEMPLATE = """
You are participating in a reasoning benchmark. Please read the following scenario carefully and provide your response.

Scenario: {scenario}

Please provide:
1. Your decision or choice
2. Your reasoning process
3. Any assumptions you made

Response:
"""

CHOICE_PROMPT_TEMPLATE = """
Please select the best option from the choices below and explain your reasoning.

Question: {question}

Options:
{options}

Your choice and reasoning:
"""
```

#### Prompt Validation
- Validate prompt length against model limits
- Check for potentially harmful content
- Ensure prompts are clear and unambiguous
- Test prompts across different models

### Model-Specific Considerations

#### Context Length Limits
- Track context length for each model
- Truncate or summarize long prompts when needed
- Warn users about context length issues
- Implement sliding window for long conversations

#### Model Capabilities
- Document model-specific features and limitations
- Handle models that don't support certain parameters
- Provide fallback options for unsupported features
- Test compatibility across model versions

#### Response Parsing
- Handle different response formats across models
- Extract structured data from free-form responses
- Implement robust parsing with error handling
- Validate response completeness

### Testing LLM Integrations

#### Test-First Development
- **MANDATORY**: When generating tasks, always create unit tests and mock tests as subtasks before actual code implementation
- **CRITICAL**: Always create unit tests and mock tests before implementing LLM provider code
- Write tests for error handling scenarios before implementing error handling
- **ALWAYS** check `pyproject.toml` for existing test dependencies before adding new ones with `uv add --dev`

#### Mock Responses
Create realistic mock responses for testing:

```python
@pytest.fixture
def mock_openai_response():
  return LLMResponse(
    text="This is a test response from GPT-4.",
    model="gpt-4",
    provider="openai",
    timestamp=datetime.now(),
    latency_ms=1500,
    token_count=25,
    finish_reason="stop"
  )
```

#### Integration Tests
- Test against real APIs in CI/CD (with rate limiting)
- Use dedicated test API keys
- Validate response formats and error handling
- Test timeout and retry mechanisms

#### Performance Testing
- Measure response times across providers
- Test concurrent request handling
- Monitor memory usage during batch processing
- Validate rate limiting effectiveness

### Security Considerations

#### API Key Management
- Never log API keys or tokens
- Use environment variables for sensitive data
- Implement key rotation procedures
- Monitor for key exposure in logs or errors

#### Request Sanitization
- Sanitize user inputs before sending to APIs
- Prevent prompt injection attacks
- Validate request parameters
- Limit request sizes and frequencies

#### Response Handling
- Sanitize responses before processing
- Handle potentially harmful content appropriately
- Log security-relevant events
- Implement content filtering when required

### Monitoring and Observability

#### Metrics Collection
- Track request/response times by provider and model
- Monitor error rates and types
- Measure token usage and costs
- Track rate limit utilization

#### Health Checks
- Implement provider health checks
- Monitor API endpoint availability
- Check model availability and performance
- Alert on service degradation

#### Logging Standards
```python
import structlog

logger = structlog.get_logger()

async def log_llm_request(request: LLMRequest, response: LLMResponse):
  logger.info(
    "llm_request_completed",
    provider=response.provider,
    model=response.model,
    latency_ms=response.latency_ms,
    token_count=response.token_count,
    success=True
  )
```

## Development Workflow Guidelines

### Development Environment Setup

#### Virtual Environment
- Always use virtual environments for development
- Use `uv venv` to create virtual environments
- Use `uv run` to execute commands in the virtual environment
- Include activation instructions in documentation
- Add `.venv/` to `.gitignore`

#### Dependencies Management
- **ALWAYS** use `uv` for dependency management with `pyproject.toml` - never use `pip`
- **CRITICAL**: Before writing any code, always check whether dependencies have already been added to `pyproject.toml`
- **MANDATORY**: Check `pyproject.toml` for existing dependencies before adding new ones
- Use `uv add <package>` to add new production dependencies
- Use `uv add --dev <package>` for development tools
- Use `uv sync` to install dependencies from lock file
- Use `uv run <command>` to execute commands in the virtual environment
- Pin exact versions for production requirements
- Include version ranges in development requirements
- Update dependencies regularly with `uv lock --upgrade` and test compatibility

#### Development Tools
- **pytest**: Testing framework (check `pyproject.toml` first, add with `uv add --dev pytest` if not present)
- **black**: Code formatting (check `pyproject.toml` first, add with `uv add --dev black` if not present - configured for 2-space indentation)
- **ruff**: Linting and style checking (check `pyproject.toml` first, add with `uv add --dev ruff` if not present)
- **mypy**: Type checking (check `pyproject.toml` first, add with `uv add --dev mypy` if not present)
- **coverage**: Code coverage reporting (check `pyproject.toml` first, add with `uv add --dev coverage` if not present)

### Git Workflow

#### Branch Naming
- `feature/description`: New features
- `bugfix/description`: Bug fixes
- `refactor/description`: Code refactoring
- `docs/description`: Documentation updates

#### Commit Messages
- Use conventional commit format
- Examples:
  - `feat: add OpenAI provider integration`
  - `fix: handle missing API key gracefully`
  - `docs: update installation instructions`
  - `test: add questionnaire loader tests`

#### Pull Request Process
1. Create feature branch from main
2. Implement changes with tests
3. Run full test suite locally
4. Update documentation if needed
5. Create PR with descriptive title and description
6. Address review feedback
7. Squash and merge when approved

### Code Quality Standards

#### Type Hints
- Use type hints for all function parameters and return values
- Import types from `typing` module when needed
- Use `Optional` for nullable parameters
- Define custom types for complex data structures

#### Documentation
- Include docstrings for all public functions and classes
- Use Google-style docstrings
- Document complex algorithms and business logic
- Keep README.md updated with new features

#### Error Handling
- Use specific exception types
- Create custom exceptions for domain-specific errors
- Log errors with appropriate context
- Provide helpful error messages to users

### Configuration Management

#### YAML Configuration
- Use consistent indentation (2 spaces)
- Validate configuration schemas for both questionnaires and LLM configs
- Provide example configuration files in both folders
- Document all configuration options
- Support multiple questionnaire files in `config/questionnaires/`
- Support multiple LLM config files in `config/llms/` with default merging
- Test configuration loading and merging logic

#### Environment Variables
- Use uppercase with underscores: `OPENAI_API_KEY`
- Provide default values when appropriate
- Document required environment variables
- Use `.env.example` file for reference

### CLI Development

#### Command Structure
- Use Click framework for CLI implementation
- Provide helpful help text for all commands and options
- Use consistent option naming conventions
- Support both short and long option forms
- Support listing available questionnaires and LLM configurations
- Handle multiple questionnaire selection
- Support configuration directory discovery

#### User Experience
- Provide progress indicators for long-running operations
- Use colored output for better readability
- Implement verbose mode for debugging
- Handle keyboard interrupts gracefully

#### Output Formats
- Default to human-readable output
- Support JSON output for programmatic use
- Use consistent formatting across commands
- Provide summary information

### Performance Guidelines

#### API Rate Limiting
- Implement exponential backoff for retries
- Respect provider rate limits
- Use connection pooling for HTTP clients
- Monitor and log API usage

#### Memory Usage
- Stream large datasets when possible
- Clean up resources in finally blocks
- Use generators for large sequences
- Monitor memory usage in tests

#### Async Operations
- Use asyncio for concurrent API calls
- Implement proper error handling in async code
- Use semaphores to limit concurrent requests
- Handle timeouts appropriately

### Release Process

#### Version Management
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `setup.py` and `__init__.py`
- Create git tags for releases
- Maintain CHANGELOG.md

#### Pre-release Checklist
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version numbers are bumped
- [ ] CHANGELOG.md is updated
- [ ] Example configurations are tested
- [ ] Performance benchmarks are run

#### Release Notes
- Highlight new features and improvements
- Document breaking changes
- Include migration instructions when needed
- Acknowledge contributors

### Debugging and Troubleshooting

#### Logging Strategy
- Use structured logging with JSON format
- Include request IDs for tracing
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Avoid logging sensitive information

#### Debug Mode
- Implement verbose/debug flags
- Provide detailed error messages in debug mode
- Include stack traces for unexpected errors
- Log configuration values (excluding secrets)

#### Common Issues
- Document common setup issues and solutions
- Provide troubleshooting guides
- Include FAQ section in documentation
- Monitor and address user-reported issues

## Working with Kiro Specs

### Overview

Kiro uses a structured specification system for feature development located in `.kiro/specs/`. Each spec represents a complete feature development cycle with three core documents: requirements, design, and implementation tasks. This section provides comprehensive guidance for reading, understanding, and working with these specifications.

### Spec Directory Structure

```
.kiro/specs/
├── {feature-name}/
│   ├── requirements.md    # User stories and acceptance criteria
│   ├── design.md         # Technical design and architecture
│   └── tasks.md          # Implementation task list with checkboxes
```

### Reading Spec Documents

#### 1. Requirements Document (requirements.md)

The requirements document contains user stories and acceptance criteria in EARS format (Easy Approach to Requirements Syntax). When reading this document:

**Structure to Expect:**
- **Introduction**: High-level feature summary and context
- **Requirements**: Numbered list of requirements, each containing:
  - **User Story**: Format "As a [role], I want [feature], so that [benefit]"
  - **Acceptance Criteria**: EARS format statements using WHEN/THEN/IF patterns

**How to Read:**
```markdown
### Requirement 1
**User Story:** As a developer, I want to configure multiple LLM providers, so that I can benchmark across different models.

#### Acceptance Criteria
1. WHEN the system loads configuration THEN it SHALL read from config/llms/ directory
2. WHEN no config is specified THEN the system SHALL use default-llms.yaml
3. IF configuration is malformed THEN the system SHALL raise descriptive error
```

**Key Points:**
- Each requirement maps to specific functionality you need to implement
- Acceptance criteria define exact behavior expectations
- EARS format provides clear conditional logic (WHEN/THEN/IF)
- Requirements are referenced in design and tasks documents

#### 2. Design Document (design.md)

The design document provides technical architecture and implementation details. When reading this document:

**Structure to Expect:**
- **Overview**: High-level architecture summary
- **Development Standards**: Project-specific coding standards and practices
- **Architecture**: System architecture diagrams and component relationships
- **Components and Interfaces**: Detailed class/module specifications
- **Data Models**: Data structures and their relationships
- **Error Handling**: Exception handling strategies
- **Testing Strategy**: Testing approaches and requirements

**How to Read:**
```markdown
## Components and Interfaces

### Configuration Management

#### Configuration Data Models
```python
@dataclass
class ProviderConfig:
  name: str
  api_key: str
  # ... other fields
```

**Key Points:**
- Contains actual code examples and interfaces to implement
- Includes architectural decisions and rationales
- Provides data models with type hints and structure
- References requirements that each component addresses
- May include Mermaid diagrams for visual architecture understanding

#### 3. Tasks Document (tasks.md)

The tasks document contains the implementation plan as a checklist. When reading this document:

**Structure to Expect:**
- Numbered hierarchical task list with checkboxes
- Each task includes:
  - Clear implementation objective
  - Sub-bullets with additional context
  - Requirements references (e.g., "_Requirements: 1.1, 2.3_")
  - Completion status indicators

**How to Read:**
```markdown
- [x] 1. Set up core project structure
  - [x] 1.1 Create module directory structure
    - Create rationale_benchmark/llm/ directory
    - Set up __init__.py files for all modules
    - _Requirements: 1.1, 1.3, 5.1_
  
- [ ] 2. Implement configuration management
  - [ ] 2.1 Create configuration loader
    - Write ConfigLoader class with YAML parsing
    - _Requirements: 1.1, 1.4, 10.1_
```

**Task Status Indicators:**
- `[x]` = Completed task
- `[ ]` = Pending task
- `[-]` = In progress task (when using task management tools)

### Working with Existing Specs

#### Before Starting Implementation

1. **Read All Three Documents**: Always read requirements.md, design.md, and tasks.md in that order
2. **Understand the Context**: Review the introduction and overview sections
3. **Identify Dependencies**: Look for references between requirements, design components, and tasks
4. **Check Current Status**: Review task completion status in tasks.md

#### Finding Relevant Information

**For Understanding What to Build:**
- Start with requirements.md for user stories and acceptance criteria
- Look for specific requirement numbers referenced in tasks

**For Understanding How to Build:**
- Refer to design.md for architecture, data models, and interfaces
- Look for code examples and class definitions
- Check error handling and testing strategies

**For Understanding Implementation Order:**
- Follow the task hierarchy in tasks.md
- Look for task dependencies and prerequisites
- Check which tasks are already completed

#### Working with Task References

Tasks reference requirements using this format: `_Requirements: 1.1, 2.3, 5.2_`

**To trace a task back to requirements:**
1. Find the requirement numbers in the task description
2. Look up those numbers in requirements.md
3. Read the corresponding acceptance criteria
4. Ensure your implementation satisfies those criteria

**Example Workflow:**
```
Task: "2.1 Create configuration loader - _Requirements: 1.1, 1.4, 10.1_"
↓
Look up Requirements 1.1, 1.4, and 10.1 in requirements.md
↓
Read acceptance criteria for those requirements
↓
Implement functionality that satisfies all criteria
```

### Implementing from Specs

#### Test-First Development

**CRITICAL**: All specs follow Test-Driven Development (TDD):
- Always create unit tests BEFORE implementing functionality
- Look for test-related tasks in the implementation plan
- Tests must be created as subtasks before code implementation

#### Following the Implementation Plan

1. **Work One Task at a Time**: Never implement multiple tasks simultaneously
2. **Follow Task Order**: Respect the hierarchical task structure
3. **Complete Sub-tasks First**: If a task has sub-tasks, complete all sub-tasks before marking the parent complete
4. **Update Task Status**: Mark tasks as complete when finished
5. **Reference Requirements**: Ensure implementation satisfies referenced requirements

#### Code Standards from Specs

Each spec includes development standards specific to the project:
- **Indentation**: Use 2 spaces (not 4) for Python code
- **Dependencies**: Always check `pyproject.toml` before adding new dependencies
- **Package Management**: Use `uv` commands, never `pip`
- **Testing**: Use pytest with comprehensive test coverage
- **Type Hints**: Include type hints for all functions and parameters

### Updating Existing Specs

#### When to Update Specs

- Requirements change or new requirements are discovered
- Design needs modification based on implementation learnings
- Tasks need to be added, modified, or reordered

#### How to Update Specs

1. **Requirements Changes**: Update requirements.md with new user stories and acceptance criteria
2. **Design Changes**: Update design.md with new architecture, components, or data models
3. **Task Changes**: Update tasks.md with new tasks, modified descriptions, or status changes

#### Maintaining Consistency

- Ensure task references to requirements remain accurate after updates
- Update design components when requirements change
- Add new tasks when new requirements or design components are added
- Keep requirement numbering consistent across all documents

### Common Spec Patterns

#### File References

Specs may include references to other files using this format:
```markdown
#[[file:config/llms/default-llms.yaml]]
```

This indicates the spec references external files that should be considered during implementation.

#### Requirement Traceability

Every implementation task should trace back to specific requirements:
- Tasks reference requirements using `_Requirements: X.Y_` format
- Design components should address specific requirements
- Test cases should validate requirement acceptance criteria

#### Hierarchical Task Structure

Tasks follow a hierarchical structure:
- Top-level tasks (1, 2, 3) represent major implementation phases
- Sub-tasks (1.1, 1.2, 2.1) represent specific implementation steps
- Sub-tasks should be completed before parent tasks
- Maximum two levels of hierarchy for clarity

### Troubleshooting Spec Issues

#### Missing Information

If specs lack necessary implementation details:
1. Check if information exists in other spec documents
2. Look for referenced external files
3. Review related requirements for additional context
4. Consider if the spec needs updating

#### Conflicting Information

If specs contain conflicting information:
1. Requirements take precedence over design
2. Design takes precedence over tasks
3. More recent updates take precedence over older content
4. Consider updating specs to resolve conflicts

#### Unclear Requirements

If requirements are unclear or ambiguous:
1. Look for additional context in the introduction
2. Check related requirements for clarification
3. Review acceptance criteria for specific behavior expectations
4. Consider updating requirements for clarity

### Best Practices for Spec-Driven Development

1. **Always Read First**: Never start implementing without reading all spec documents
2. **Follow the Plan**: Implement tasks in the specified order
3. **Test First**: Create tests before implementation as specified in tasks
4. **Reference Requirements**: Ensure every implementation satisfies its referenced requirements
5. **Update Status**: Keep task completion status current
6. **Maintain Traceability**: Preserve the connection between requirements, design, and implementation
7. **Document Changes**: Update specs when implementation reveals necessary changes

This comprehensive approach to working with Kiro specs ensures consistent, requirement-driven development that maintains traceability from user needs through design to implementation.