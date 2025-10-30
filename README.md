# Rationale Benchmark for Large Language Models

A configurable benchmark tool designed to evaluate large language models (LLMs) based on their similarity to human rationale, including human intuitions and psychological reasoning patterns.

## Overview

This project provides a command-line tool for testing different LLMs against human-like reasoning benchmarks. Unlike traditional benchmarks that focus on accuracy, this tool evaluates how closely LLM reasoning aligns with human cognitive processes and decision-making patterns.

## Features

- **Configurable Questionnaires**: Define custom questionnaires using YAML configuration files
- **Multi-Provider LLM Support**: Connect to various LLM inference services from different providers
- **Human Rationale Testing**: Evaluate models on human intuition and psychological reasoning
- **Flexible Configuration**: Separate configuration files for questionnaires and LLM providers
- **Command-Line Interface**: Easy-to-use CLI for running benchmarks
- **Extensible Architecture**: Built with future web interface and visualization in mind

## Documentation Practices

- Place all project documentation under `/doc`
- Review the relevant documentation before starting work on any coding prompt or implementation task

## Installation

### Prerequisites
- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd rationale-benchmark

# Create virtual environment and install dependencies
uv sync

# Install the package in development mode
uv add -e .
```

### Alternative: Development Setup

```bash
# Create virtual environment
uv venv

# Install all dependencies including development tools
uv sync --dev

# Activate the virtual environment (optional, uv run handles this automatically)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Quick Start

1. **Create questionnaires** in `config/questionnaires/` folder
2. **Configure LLM providers** in `config/llms/` folder (starting with `default-llms.yaml`)
3. **Run the benchmark**:

```bash
# Run all questionnaires with default LLM config
uv run rationale-benchmark

# Run specific questionnaire with default LLMs
uv run rationale-benchmark --questionnaire moral-reasoning

# Run specific questionnaire with custom LLM config
uv run rationale-benchmark --questionnaire cognitive-biases --llm-config research-models

# Run multiple questionnaires
uv run rationale-benchmark --questionnaires moral-reasoning,cognitive-biases
```

## Configuration

The tool supports multiple questionnaire files and LLM configuration files organized in separate folders for better modularity and reusability.

### Configuration Structure

```
config/
├── questionnaires/
│   ├── moral-reasoning.yaml
│   ├── cognitive-biases.yaml
│   ├── decision-making.yaml
│   └── custom-benchmark.yaml
└── llms/
    ├── default-llms.yaml
    ├── openai-models.yaml
    ├── anthropic-models.yaml
    └── local-models.yaml
```

### Questionnaire Configuration

Create multiple YAML files in the `config/questionnaires/` folder. Each file defines one complete questionnaire:

```yaml
# config/questionnaires/moral-reasoning.yaml
questionnaire:
  name: "Moral Reasoning Benchmark"
  description: "Testing ethical decision-making patterns"
  
  sections:
    - name: "Trolley Problems"
      questions:
        - id: "moral_001"
          type: "scenario"
          prompt: "A runaway trolley is heading towards five people..."
          expected_reasoning: "utilitarian vs deontological"
          
    - name: "Justice Scenarios"
      questions:
        - id: "moral_002"
          type: "choice"
          prompt: "In a resource allocation scenario..."
          options: ["Equal distribution", "Merit-based", "Need-based"]
          bias_type: "fairness_bias"
```

```yaml
# config/questionnaires/cognitive-biases.yaml
questionnaire:
  name: "Cognitive Bias Detection"
  description: "Evaluating susceptibility to common cognitive biases"
  
  sections:
    - name: "Availability Heuristic"
      questions:
        - id: "bias_001"
          type: "choice"
          prompt: "Which cause of death is more common?"
          options: ["Shark attacks", "Dog bites"]
          bias_type: "availability_heuristic"
```

### LLM Provider Configuration

The LLM configuration supports a default configuration file with custom overrides:

#### Default Configuration
```yaml
# config/llms/default-llms.yaml
defaults:
  timeout: 30
  max_retries: 3
  temperature: 0.7
  max_tokens: 1000

providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    models:
      - "gpt-4"
      - "gpt-3.5-turbo"
    
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com"
    models:
      - "claude-3-opus-20240229"
      - "claude-3-sonnet-20240229"
```

#### Custom Configuration Files
```yaml
# config/llms/research-models.yaml
# This file extends/overrides the default configuration
providers:
  openai:
    models:
      - "gpt-4-turbo"
      - "gpt-4o"
    default_params:
      temperature: 0.1  # Lower temperature for research
      
  local:
    endpoint: "http://localhost:8000"
    timeout: 60  # Longer timeout for local models
    models:
      - "llama-2-7b-chat"
      - "mistral-7b-instruct"
```

```yaml
# config/llms/production-models.yaml
providers:
  openai:
    models:
      - "gpt-4"
    default_params:
      temperature: 0.5
      max_tokens: 2000
      
  anthropic:
    models:
      - "claude-3-opus-20240229"
    default_params:
      temperature: 0.3
```

#### Configuration Merging Behavior

When using custom LLM configuration files:
1. The system loads `default-llms.yaml` first as the base configuration
2. Custom configuration files extend or override the defaults
3. Provider-level settings are merged (custom providers are added, existing ones are updated)
4. Model lists are replaced entirely if specified in custom config
5. Default parameters are merged at the provider level
6. Environment variables are resolved in both default and custom configs

Example of how configurations merge:
- `default-llms.yaml` defines OpenAI with `gpt-4` and `gpt-3.5-turbo`
- `research-models.yaml` specifies OpenAI with `gpt-4-turbo` only
- Result: OpenAI provider uses only `gpt-4-turbo` (model list is replaced)
- Default timeout and retry settings from `default-llms.yaml` are preserved unless overridden

## Usage

### Basic Usage

```bash
# Run all questionnaires with default LLM configuration
rationale-benchmark

# Run specific questionnaire (by filename without extension)
rationale-benchmark --questionnaire moral-reasoning

# Run multiple questionnaires
rationale-benchmark --questionnaires moral-reasoning,cognitive-biases

# Use custom LLM configuration
rationale-benchmark --llm-config research-models

# Run specific models only
rationale-benchmark --models gpt-4,claude-3-opus

# Output results to specific file
rationale-benchmark --output results.json

# List available questionnaires and LLM configs
rationale-benchmark --list-questionnaires
rationale-benchmark --list-llm-configs
```

### Command-Line Options

```
Options:
  --questionnaire TEXT       Single questionnaire to run (filename without .yaml)
  --questionnaires TEXT      Comma-separated list of questionnaires to run
  --llm-config TEXT         LLM configuration to use (filename without .yaml, defaults to 'default-llms')
  --models TEXT             Comma-separated list of specific models to test
  --output PATH             Output file for results (JSON format)
  --list-questionnaires     List all available questionnaires
  --list-llm-configs        List all available LLM configurations
  --config-dir PATH         Custom path to configuration directory (default: ./config)
  --verbose                 Enable verbose logging
  --help                    Show this message and exit
```

## Project Structure

- `config/`: Houses LLM configuration files and questionnaire definitions consumed by the CLI.
- `docs/`: Authoritative references for architecture, configuration workflows, and interface contracts.
- `rationale_benchmark/`: Core implementation of the CLI and LLM interaction layers.
- `tests/`: Pytest suite validating configuration loading, provider integrations, and concurrency helpers.
- `AGENTS.md`, `README.md`, `pyproject.toml`, `uv.lock`: Contributor handbook, project overview, and uv dependency manifests.

## Development

### Setting up Development Environment

```bash
# Create virtual environment and install all dependencies
uv sync --dev

# Run tests
uv run pytest

# Run code formatting
uv run black rationale_benchmark/

# Run linting
uv run ruff check rationale_benchmark/

# Run type checking
uv run mypy rationale_benchmark/
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=rationale_benchmark

# Run specific test file
uv run pytest tests/test_questionnaire.py

# Generate coverage report
uv run coverage report
uv run coverage html  # Generate HTML coverage report
```

## Roadmap

### Phase 1: CLI Tool (Current)
- [x] Basic project structure
- [ ] Questionnaire YAML loader
- [ ] LLM provider integrations
- [ ] Benchmark runner
- [ ] Results output

### Phase 2: Web Interface
- [ ] REST API server
- [ ] Web-based questionnaire configuration
- [ ] Real-time benchmark execution
- [ ] Results dashboard

### Phase 3: Advanced Features
- [ ] Results visualization
- [ ] Statistical analysis
- [ ] Comparative reporting
- [ ] Export capabilities

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{rationale_benchmark,
  title={Rationale Benchmark for Large Language Models},
  author={Shuhao Liu, Xiaotian Wang, et al.},
  year={2025},
  url={https://github.com/shuhaoliu/rationale-benchmark}
}
```

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Join our discussions

---

**Note**: This project is under active development. The API and configuration formats may change in future versions.
