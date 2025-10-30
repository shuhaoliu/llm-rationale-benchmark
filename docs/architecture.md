# Project Architecture Guidelines

## Project Overview
This rationale benchmark tool evaluates LLMs against human-like reasoning patterns. The architecture follows a modular design with a clear separation of concerns, enabling each component to evolve independently while remaining cohesive as a system.

## Core Modules

### CLI Module (`cli.py`)
- Entry point for the command-line interface
- Handles argument parsing and validation for multiple questionnaires and LLM configs
- Supports listing available questionnaires and LLM configurations
- Coordinates between questionnaire loading, LLM clients, and benchmark execution
- Uses the Click framework for CLI implementation
- Supports running single or multiple questionnaires
- Handles configuration directory discovery and validation

### Questionnaire Module (`questionnaire/`)
- `loader.py`: YAML questionnaire parsing and loading from multiple files
- `validator.py`: Questionnaire structure validation
- Supports loading questionnaires by filename (without extension)
- Supports loading multiple questionnaires in a single execution
- Supports nested sections and multiple question types
- Validates required fields and data types
- Discovers and lists available questionnaire files

### LLM Module (`llm/`)
- `client.py`: Abstract LLM client interface
- `providers/`: Provider-specific implementations (OpenAI, Anthropic, local)
- Handles API authentication and rate limiting
- Standardizes response formats across providers

### Benchmark Module (`benchmark/`)
- `runner.py`: Executes benchmarks across multiple models
- `evaluator.py`: Analyzes and scores LLM responses
- Supports parallel execution when possible
- Generates structured results

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
- Continue execution if a single model fails
- Collect and report all errors at the end
- Provide partial results when possible

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
- Keep business logic separate from the CLI
- Design APIs that can be exposed via REST
- Use dependency injection for easier testing

### Plugin Architecture
- Design the provider interface for easy extension
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
