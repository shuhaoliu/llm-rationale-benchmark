# Configuration Management Specifications

## Configuration Practices
- Keep configuration separate from code
- Use environment-specific config files
- Validate configuration on startup

## Multi-File Configuration Structure
- Use `config/questionnaires/` for multiple questionnaire definitions
- Use `config/llms/` for multiple LLM provider configurations
- Treat each questionnaire file as one complete benchmark
- Support default LLM configuration with custom overrides
- Support environment variable substitution
- Validate configuration schemas on load

## Questionnaire Configuration
- Store each questionnaire in YAML within `config/questionnaires/`
- Reference files by filename without extension
- Support loading multiple questionnaires in a single run
- Validate questionnaire structure and required fields

## LLM Configuration
- Use `default-llms.yaml` for base configuration and defaults
- Extend or override defaults with additional YAML files
- Support provider-specific configurations
- Merge configurations with custom files taking precedence
- Reference files by filename without extension

## Environment Variables
- Use environment variables for sensitive data such as API keys
- Follow naming convention `{PROVIDER}_API_KEY`
- Document all required environment variables
- Support environment variable substitution in all config files

## YAML Configuration Standards
- Use consistent 2-space indentation
- Validate configuration schemas for both questionnaires and LLM configs
- Provide example configuration files in both folders
- Document all configuration options
- Support multiple questionnaire files in `config/questionnaires/`
- Support multiple LLM config files in `config/llms/` with default merging
- Test configuration loading and merging logic

## Environment Variable Management
- Use uppercase names with underscores, for example `OPENAI_API_KEY`
- Provide default values when appropriate
- Document required environment variables
- Maintain a `.env.example` file for reference
