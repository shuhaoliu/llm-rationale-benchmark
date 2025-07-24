# Project Architecture Guidelines

## Project Overview
This is a rationale benchmark tool for evaluating LLMs based on human-like reasoning patterns. The architecture follows a modular design with clear separation of concerns.

## Core Modules

### CLI Module (`cli.py`)
- Entry point for command-line interface
- Handle argument parsing and validation for multiple questionnaires and LLM configs
- Support listing available questionnaires and LLM configurations
- Coordinate between questionnaire loading, LLM clients, and benchmark execution
- Use Click framework for CLI implementation
- Support running single or multiple questionnaires
- Handle configuration directory discovery and validation

### Questionnaire Module (`questionnaire/`)
- **loader.py**: YAML questionnaire parsing and loading from multiple files
- **validator.py**: Questionnaire structure validation
- Support loading questionnaires by filename (without extension)
- Support loading multiple questionnaires in a single execution
- Support nested sections and multiple question types
- Validate required fields and data types
- Discover and list available questionnaire files

### LLM Module (`llm/`)
- **client.py**: Abstract LLM client interface
- **providers/**: Provider-specific implementations (OpenAI, Anthropic, local)
- Handle API authentication and rate limiting
- Standardize response formats across providers

### Benchmark Module (`benchmark/`)
- **runner.py**: Execute benchmarks across multiple models
- **evaluator.py**: Analyze and score LLM responses
- Support parallel execution when possible
- Generate structured results

## Configuration Management

### Multi-File Configuration Structure
- Use `config/questionnaires/` folder for multiple questionnaire definitions
- Use `config/llms/` folder for multiple LLM provider configurations
- Each questionnaire file represents one complete benchmark
- Support default LLM configuration with custom overrides
- Support environment variable substitution
- Validate configuration schemas on load

### Questionnaire Configuration
- Each YAML file in `config/questionnaires/` defines one questionnaire
- Files are referenced by filename without extension
- Support loading multiple questionnaires in a single run
- Validate questionnaire structure and required fields

### LLM Configuration
- `default-llms.yaml` provides base configuration and defaults
- Additional YAML files can extend or override default settings
- Support provider-specific configurations
- Merge configurations with custom files taking precedence
- Files are referenced by filename without extension

### Environment Variables
- Use for sensitive data (API keys)
- Follow naming convention: `{PROVIDER}_API_KEY`
- Document all required environment variables
- Support environment variable substitution in all config files

## Data Models

### Questionnaire Structure
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

### LLM Response Structure
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

## Error Handling Strategy

### Configuration Errors
- Validate YAML structure on load
- Provide clear error messages for missing fields
- Fail fast with descriptive error messages

### API Errors
- Implement retry logic with exponential backoff
- Handle rate limiting gracefully
- Log API errors with context

### Benchmark Execution Errors
- Continue execution if single model fails
- Collect and report all errors at the end
- Provide partial results when possible

## Testing Strategy

### Test-First Development
- **MANDATORY**: When generating tasks, always create unit tests and mock tests as subtasks before actual code implementation
- **CRITICAL**: Tests must be created as subtasks before any implementation work begins
- Write tests before implementing functionality
- Use Test-Driven Development (TDD) approach where appropriate

### Unit Tests
- Test each module independently
- Mock external API calls
- Use pytest fixtures for common test data
- Aim for >90% code coverage
- **ALWAYS** check dependencies in `pyproject.toml` before adding test dependencies with `uv add --dev`

### Integration Tests
- Test full benchmark execution flow
- Use sample questionnaires and mock LLM responses
- Validate output format and structure

### Configuration Tests
- Test YAML loading and validation for multiple files
- Test questionnaire discovery and loading
- Test LLM configuration merging (default + custom)
- Test environment variable substitution
- Test error handling for invalid configurations
- Test configuration file listing functionality

## Output and Results

### JSON Output Format
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

### Logging
- Use structured logging with JSON format
- Log levels: DEBUG, INFO, WARNING, ERROR
- Include execution context in log messages
- Separate log files for different components

## Future Extensibility

### Web Interface Preparation
- Keep business logic separate from CLI
- Design APIs that can be exposed via REST
- Use dependency injection for easier testing

### Plugin Architecture
- Design provider interface for easy extension
- Support custom evaluators
- Allow custom question types

## Performance Considerations

### Async Operations
- Use asyncio for concurrent LLM API calls
- Implement connection pooling for HTTP clients
- Handle rate limiting across concurrent requests

### Memory Management
- Stream large result sets when possible
- Implement pagination for large questionnaires
- Clean up resources properly

## Security Guidelines

### API Key Management
- Never log API keys
- Use environment variables only
- Validate API key format before use

### Input Validation
- Sanitize all user inputs
- Validate file paths and prevent directory traversal
- Limit file sizes for uploaded questionnaires