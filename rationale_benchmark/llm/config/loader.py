"""Configuration loader for LLM connector with YAML parsing and environment variable resolution."""

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from ..exceptions import ConfigurationError
from .models import LLMConfig


class ConfigLoader:
  """Handles loading of LLM configuration files with YAML parsing and environment variable resolution.
  
  This class provides functionality to:
  - Load configuration from single YAML files
  - Discover available configuration files in the config/llms directory
  - Resolve environment variables in ${VAR} format
  - Validate YAML structure and provide detailed error messages
  """

  def __init__(self, config_dir: Path):
    """Initialize ConfigLoader with configuration directory.
    
    Args:
      config_dir: Path to directory containing LLM configuration files
    """
    self.config_dir = Path(config_dir)

  def load_config(self, config_name: str = "default-llms") -> LLMConfig:
    """Load configuration from specified file.
    
    Args:
      config_name: Name of configuration file without .yaml extension
      
    Returns:
      LLMConfig instance with resolved environment variables
      
    Raises:
      ConfigurationError: If file not found, invalid YAML, or environment variables missing
    """
    config_file = self.config_dir / f"{config_name}.yaml"
    
    if not config_file.exists():
      raise ConfigurationError(
        f"Configuration file not found: {config_file}",
        config_file=str(config_file)
      )
    
    try:
      with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
      raise ConfigurationError(
        f"Invalid YAML in configuration file: {e}",
        config_file=str(config_file)
      )
    except Exception as e:
      raise ConfigurationError(
        f"Error reading configuration file: {e}",
        config_file=str(config_file)
      )
    
    if config_dict is None:
      raise ConfigurationError(
        "Configuration file is empty",
        config_file=str(config_file)
      )
    
    if not isinstance(config_dict, dict):
      raise ConfigurationError(
        "Configuration file must contain a YAML dictionary",
        config_file=str(config_file)
      )
    
    # Resolve environment variables
    try:
      resolved_config = self._resolve_environment_variables(config_dict)
    except ConfigurationError as e:
      # Add config file context to environment variable errors
      e.config_file = str(config_file)
      raise e
    
    # Create LLMConfig from resolved dictionary
    return LLMConfig.from_dict(resolved_config, str(config_file))

  def list_available_configs(self) -> list[str]:
    """List all available configuration files in the config directory.
    
    Returns:
      List of configuration file names without .yaml extension
      
    Raises:
      ConfigurationError: If config directory doesn't exist
    """
    if not self.config_dir.exists():
      raise ConfigurationError(
        f"Configuration directory not found: {self.config_dir}"
      )
    
    if not self.config_dir.is_dir():
      raise ConfigurationError(
        f"Configuration path is not a directory: {self.config_dir}"
      )
    
    config_files = []
    for file_path in self.config_dir.glob("*.yaml"):
      if file_path.is_file():
        config_files.append(file_path.stem)
    
    # Also check for .yml extension
    for file_path in self.config_dir.glob("*.yml"):
      if file_path.is_file():
        config_files.append(file_path.stem)
    
    return sorted(config_files)

  def discover_config_files(self) -> dict[str, Path]:
    """Discover and return mapping of config names to file paths.
    
    Returns:
      Dictionary mapping config names to their file paths
      
    Raises:
      ConfigurationError: If config directory doesn't exist
    """
    if not self.config_dir.exists():
      raise ConfigurationError(
        f"Configuration directory not found: {self.config_dir}"
      )
    
    config_files = {}
    
    # Check for .yaml files
    for file_path in self.config_dir.glob("*.yaml"):
      if file_path.is_file():
        config_files[file_path.stem] = file_path
    
    # Check for .yml files (prefer .yaml if both exist)
    for file_path in self.config_dir.glob("*.yml"):
      if file_path.is_file() and file_path.stem not in config_files:
        config_files[file_path.stem] = file_path
    
    return config_files

  def validate_config_file(self, config_name: str) -> tuple[bool, Optional[str]]:
    """Validate a configuration file without fully loading it.
    
    Args:
      config_name: Name of configuration file without .yaml extension
      
    Returns:
      Tuple of (is_valid, error_message)
    """
    try:
      self.load_config(config_name)
      return True, None
    except ConfigurationError as e:
      return False, str(e)
    except Exception as e:
      return False, f"Unexpected error: {e}"

  def _resolve_environment_variables(self, config_dict: dict[str, Any]) -> dict[str, Any]:
    """Resolve ${VAR} patterns in configuration with environment variables.
    
    This method recursively traverses the configuration dictionary and replaces
    ${VAR} patterns with corresponding environment variable values.
    
    Args:
      config_dict: Configuration dictionary potentially containing ${VAR} patterns
      
    Returns:
      Dictionary with environment variables resolved
      
    Raises:
      ConfigurationError: If required environment variable is missing
    """
    def resolve_value(value: Any, path: str = "") -> Any:
      """Recursively resolve environment variables in configuration values.
      
      Args:
        value: Value to resolve (can be string, dict, list, or other)
        path: Current path in config for error reporting
        
      Returns:
        Resolved value with environment variables substituted
      """
      if isinstance(value, str):
        # Find all ${VAR} patterns
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        
        if not matches:
          return value
        
        resolved_value = value
        for var_name in matches:
          env_value = os.getenv(var_name)
          if env_value is None:
            error_path = f" at {path}" if path else ""
            raise ConfigurationError(
              f"Environment variable '{var_name}' is not set{error_path}"
            )
          resolved_value = resolved_value.replace(f"${{{var_name}}}", env_value)
        
        return resolved_value
      
      elif isinstance(value, dict):
        resolved_dict = {}
        for key, val in value.items():
          new_path = f"{path}.{key}" if path else key
          resolved_dict[key] = resolve_value(val, new_path)
        return resolved_dict
      
      elif isinstance(value, list):
        resolved_list = []
        for i, item in enumerate(value):
          new_path = f"{path}[{i}]" if path else f"[{i}]"
          resolved_list.append(resolve_value(item, new_path))
        return resolved_list
      
      else:
        # Return other types unchanged (int, float, bool, None, etc.)
        return value
    
    return resolve_value(config_dict)

  def get_config_file_path(self, config_name: str) -> Path:
    """Get the full path to a configuration file.
    
    Args:
      config_name: Name of configuration file without extension
      
    Returns:
      Path to the configuration file
      
    Raises:
      ConfigurationError: If file doesn't exist
    """
    # Try .yaml first, then .yml
    yaml_path = self.config_dir / f"{config_name}.yaml"
    yml_path = self.config_dir / f"{config_name}.yml"
    
    if yaml_path.exists():
      return yaml_path
    elif yml_path.exists():
      return yml_path
    else:
      raise ConfigurationError(
        f"Configuration file not found: {config_name}.yaml or {config_name}.yml",
        config_file=str(yaml_path)
      )

  def load_single_config(self, config_name: Optional[str] = None) -> LLMConfig:
    """Load a single specified configuration file with default selection support.
    
    This method implements the core functionality for loading a single configuration
    file as specified in task 2.2. It provides:
    - Loading of a single specified configuration file
    - Default configuration file selection when none specified
    - Proper error handling for missing configuration files
    - Complete configuration structure validation
    
    Args:
      config_name: Name of configuration file without .yaml extension.
                  If None, uses "default-llms" as default.
      
    Returns:
      LLMConfig instance with resolved environment variables and validated structure
      
    Raises:
      ConfigurationError: If file not found, invalid YAML, environment variables missing,
                         or configuration structure is invalid
    """
    # Use default configuration name if none specified (Requirement 1.2)
    if config_name is None:
      config_name = "default-llms"
    
    # Determine the configuration file path, preferring .yaml over .yml
    config_file = None
    yaml_path = self.config_dir / f"{config_name}.yaml"
    yml_path = self.config_dir / f"{config_name}.yml"
    
    if yaml_path.exists():
      config_file = yaml_path
    elif yml_path.exists():
      config_file = yml_path
    else:
      # Proper error handling for missing configuration files (Task requirement)
      raise ConfigurationError(
        f"Configuration file not found: {config_name}.yaml or {config_name}.yml",
        config_file=str(yaml_path)
      )
    
    # Load and parse the YAML file
    try:
      with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
      raise ConfigurationError(
        f"Invalid YAML in configuration file: {e}",
        config_file=str(config_file)
      )
    except Exception as e:
      raise ConfigurationError(
        f"Error reading configuration file: {e}",
        config_file=str(config_file)
      )
    
    # Validate that the file is not empty
    if config_dict is None:
      raise ConfigurationError(
        "Configuration file is empty",
        config_file=str(config_file)
      )
    
    # Validate that the content is a dictionary
    if not isinstance(config_dict, dict):
      raise ConfigurationError(
        "Configuration file must contain a YAML dictionary",
        config_file=str(config_file)
      )
    
    # Resolve environment variables with proper error context
    try:
      resolved_config = self._resolve_environment_variables(config_dict)
    except ConfigurationError as e:
      # Add config file context to environment variable errors
      e.config_file = str(config_file)
      raise e
    
    # Create and validate LLMConfig from resolved dictionary (Requirement 5.4)
    # This ensures complete configuration structure validation
    return LLMConfig.from_dict(resolved_config, str(config_file))

  def create_config_directory(self) -> None:
    """Create the configuration directory if it doesn't exist.
    
    Raises:
      ConfigurationError: If directory cannot be created
    """
    try:
      self.config_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
      raise ConfigurationError(
        f"Failed to create configuration directory {self.config_dir}: {e}"
      )

  def discover_configuration_files(self) -> list[dict[str, Any]]:
    """Discover and list available configuration files with validation status.
    
    This method implements configuration file discovery as specified in task 12.1.
    It provides:
    - Discovery of all YAML configuration files in the config directory
    - Validation status reporting for each configuration file
    - Streaming parameter detection and warnings
    - Provider and model information for each configuration
    
    Returns:
      List of dictionaries containing configuration file information:
      - name: Configuration name (without extension)
      - file_path: Full path to configuration file
      - is_valid: Boolean indicating if configuration is valid
      - error_message: Error message if configuration is invalid
      - has_streaming_params: Boolean indicating if streaming parameters found
      - streaming_params: List of streaming parameter names found
      - providers: Dictionary of provider information
      - total_providers: Total number of providers configured
      - total_models: Total number of models across all providers
      - warnings: List of warning messages
      
    Raises:
      ConfigurationError: If config directory doesn't exist
    """
    if not self.config_dir.exists():
      raise ConfigurationError(
        f"Configuration directory not found: {self.config_dir}"
      )
    
    if not self.config_dir.is_dir():
      raise ConfigurationError(
        f"Configuration path is not a directory: {self.config_dir}"
      )
    
    discovered_configs = []
    config_files = self.discover_config_files()
    
    for config_name, config_path in config_files.items():
      config_info = {
        "name": config_name,
        "file_path": str(config_path),
        "is_valid": False,
        "error_message": None,
        "has_streaming_params": False,
        "streaming_params": [],
        "providers": {},
        "total_providers": 0,
        "total_models": 0,
        "warnings": []
      }
      
      try:
        # Load and validate configuration
        config = self.load_config(config_name)
        
        # Create validator to check for streaming parameters and other issues
        from .validator import ConfigValidator
        validator = ConfigValidator()
        
        # Run validation
        validation_errors = validator.validate_config(config)
        env_errors = validator.validate_environment_variables(config)
        
        # Check if configuration is valid
        if not validation_errors and not env_errors:
          config_info["is_valid"] = True
        else:
          config_info["error_message"] = "; ".join(validation_errors + env_errors)
        
        # Get streaming parameter information
        streaming_params = validator.get_streaming_parameters_found()
        if streaming_params:
          config_info["has_streaming_params"] = True
          config_info["streaming_params"] = streaming_params
        
        # Get warnings
        config_info["warnings"] = validator.get_warnings()
        
        # Extract provider and model information
        config_info["total_providers"] = len(config.providers)
        config_info["providers"] = {}
        total_models = 0
        
        for provider_name, provider_config in config.providers.items():
          provider_info = {
            "models": provider_config.models,
            "model_count": len(provider_config.models),
            "base_url": provider_config.base_url,
            "timeout": provider_config.timeout,
            "max_retries": provider_config.max_retries
          }
          config_info["providers"][provider_name] = provider_info
          total_models += len(provider_config.models)
        
        config_info["total_models"] = total_models
        
      except Exception as e:
        # Handle any errors during configuration loading or validation
        config_info["is_valid"] = False
        config_info["error_message"] = str(e)
      
      discovered_configs.append(config_info)
    
    # Sort by configuration name for consistent ordering
    return sorted(discovered_configs, key=lambda x: x["name"])

  def create_example_configuration_directory(self) -> None:
    """Create configuration directory with example files if it doesn't exist.
    
    This method implements configuration directory creation as specified in task 12.1.
    It provides:
    - Creation of the configuration directory if it doesn't exist
    - Generation of example configuration files for reference
    - Proper error handling for directory creation failures
    
    Raises:
      ConfigurationError: If directory cannot be created
    """
    # Create directory if it doesn't exist
    self.create_config_directory()
    
    # Create example configuration files if they don't exist
    default_config_path = self.config_dir / "default-llms.yaml"
    example_config_path = self.config_dir / "example-config.yaml"
    
    # Default configuration template
    default_config_template = {
      "defaults": {
        "timeout": 30,
        "max_retries": 3,
        "temperature": 0.7,
        "max_tokens": 1000
      },
      "providers": {
        "openai": {
          "api_key": "${OPENAI_API_KEY}",
          "base_url": "https://api.openai.com/v1",
          "models": ["gpt-4", "gpt-3.5-turbo"],
          "default_params": {
            "temperature": 0.7
          }
        },
        "anthropic": {
          "api_key": "${ANTHROPIC_API_KEY}",
          "base_url": "https://api.anthropic.com",
          "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
          "default_params": {
            "max_tokens": 2000
          }
        }
      }
    }
    
    # Example configuration with more providers
    example_config_template = {
      "defaults": {
        "timeout": 45,
        "max_retries": 5,
        "temperature": 0.8,
        "max_tokens": 2000
      },
      "providers": {
        "openai": {
          "api_key": "${OPENAI_API_KEY}",
          "base_url": "https://api.openai.com/v1",
          "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
          "default_params": {
            "temperature": 0.8,
            "top_p": 0.9
          }
        },
        "anthropic": {
          "api_key": "${ANTHROPIC_API_KEY}",
          "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        },
        "gemini": {
          "api_key": "${GEMINI_API_KEY}",
          "models": ["gemini-pro", "gemini-pro-vision"],
          "default_params": {
            "temperature": 0.9
          }
        },
        "openrouter": {
          "api_key": "${OPENROUTER_API_KEY}",
          "base_url": "https://openrouter.ai/api/v1",
          "models": ["openai/gpt-4", "anthropic/claude-3-opus"],
          "provider_specific": {
            "http_referer": "https://your-app.com",
            "x_title": "Your App Name"
          }
        }
      }
    }
    
    # Write default configuration if it doesn't exist
    if not default_config_path.exists():
      try:
        with open(default_config_path, 'w', encoding='utf-8') as f:
          yaml.dump(default_config_template, f, indent=2, default_flow_style=False)
      except Exception as e:
        raise ConfigurationError(
          f"Failed to create default configuration file: {e}"
        )
    
    # Write example configuration if it doesn't exist
    if not example_config_path.exists():
      try:
        with open(example_config_path, 'w', encoding='utf-8') as f:
          yaml.dump(example_config_template, f, indent=2, default_flow_style=False)
      except Exception as e:
        raise ConfigurationError(
          f"Failed to create example configuration file: {e}"
        )

  def get_configuration_summary(self) -> dict[str, Any]:
    """Get summary information about all available configurations.
    
    Returns:
      Dictionary containing summary statistics:
      - total_configurations: Total number of configuration files
      - valid_configurations: Number of valid configurations
      - invalid_configurations: Number of invalid configurations
      - configurations_with_streaming: Number of configs with streaming parameters
      - all_providers: List of all unique providers across configurations
      - total_models: Total number of models across all configurations
    """
    try:
      configs = self.discover_configuration_files()
    except ConfigurationError:
      # Return empty summary if directory doesn't exist
      return {
        "total_configurations": 0,
        "valid_configurations": 0,
        "invalid_configurations": 0,
        "configurations_with_streaming": 0,
        "all_providers": [],
        "total_models": 0
      }
    
    total_configs = len(configs)
    valid_configs = sum(1 for c in configs if c["is_valid"])
    invalid_configs = total_configs - valid_configs
    streaming_configs = sum(1 for c in configs if c["has_streaming_params"])
    
    # Collect all unique providers
    all_providers = set()
    total_models = 0
    
    for config in configs:
      all_providers.update(config["providers"].keys())
      total_models += config["total_models"]
    
    return {
      "total_configurations": total_configs,
      "valid_configurations": valid_configs,
      "invalid_configurations": invalid_configs,
      "configurations_with_streaming": streaming_configs,
      "all_providers": sorted(list(all_providers)),
      "total_models": total_models
    }

  def validate_all_configurations(self) -> list[dict[str, Any]]:
    """Validate all configuration files and return detailed results.
    
    Returns:
      List of dictionaries containing detailed validation results for each configuration
    """
    try:
      configs = self.discover_configuration_files()
    except ConfigurationError:
      return []
    
    detailed_results = []
    
    for config_info in configs:
      # Create detailed validation report
      detailed_result = {
        "name": config_info["name"],
        "file_path": config_info["file_path"],
        "is_valid": config_info["is_valid"],
        "error_message": config_info["error_message"],
        "has_streaming_params": config_info["has_streaming_params"],
        "streaming_params": config_info["streaming_params"],
        "warnings": config_info["warnings"],
        "validation_details": None
      }
      
      # Try to get detailed validation report
      try:
        config = self.load_config(config_info["name"])
        from .validator import ConfigValidator
        validator = ConfigValidator()
        detailed_result["validation_details"] = validator.create_validation_report(config)
      except Exception:
        # If we can't load config, create a basic validation report
        detailed_result["validation_details"] = {
          "validation_passed": False,
          "total_errors": 1,
          "configuration_errors": [config_info["error_message"]] if config_info["error_message"] else ["Configuration could not be loaded"],
          "environment_errors": [],
          "warnings": config_info["warnings"],
          "streaming_parameters_found": config_info["streaming_params"],
          "statistics": {
            "total_providers": 0,
            "total_models": 0,
            "providers_configured": [],
          }
        }
      
      detailed_results.append(detailed_result)
    
    return detailed_results

  def list_configurations_with_details(self) -> list[dict[str, Any]]:
    """List all configurations with comprehensive details.
    
    Returns:
      List of dictionaries containing comprehensive configuration information
    """
    return self.discover_configuration_files()

  def display_configuration(self, config_name: str) -> dict[str, Any]:
    """Display detailed information for a specific configuration.
    
    This method implements configuration display functionality as specified in task 12.2.
    It provides:
    - Detailed provider and model information display
    - Configuration validation status in listing output
    - Warnings for any streaming-related configuration found
    - Comprehensive configuration information display
    
    Args:
      config_name: Name of configuration to display (without .yaml extension)
      
    Returns:
      Dictionary containing detailed configuration display information:
      - name: Configuration name
      - file_path: Full path to configuration file
      - is_valid: Boolean indicating if configuration is valid
      - error_message: Error message if configuration is invalid
      - has_streaming_params: Boolean indicating if streaming parameters found
      - streaming_params: List of streaming parameter names found
      - warnings: List of warning messages
      - defaults: Default configuration parameters
      - providers: Detailed provider information
      - total_providers: Total number of providers
      - total_models: Total number of models
      
    Raises:
      ConfigurationError: If configuration file not found
    """
    # Get the configuration file path
    try:
      config_path = self.get_config_file_path(config_name)
    except ConfigurationError:
      raise ConfigurationError(
        f"Configuration file not found: {config_name}.yaml or {config_name}.yml"
      )
    
    display_info = {
      "name": config_name,
      "file_path": str(config_path),
      "is_valid": False,
      "error_message": None,
      "has_streaming_params": False,
      "streaming_params": [],
      "warnings": [],
      "defaults": {},
      "providers": {},
      "total_providers": 0,
      "total_models": 0
    }
    
    try:
      # Load configuration
      config = self.load_config(config_name)
      
      # Create validator for streaming detection and validation
      from .validator import ConfigValidator
      validator = ConfigValidator()
      
      # Run validation
      validation_errors = validator.validate_config(config)
      env_errors = validator.validate_environment_variables(config)
      
      # Set validation status
      if not validation_errors and not env_errors:
        display_info["is_valid"] = True
      else:
        display_info["error_message"] = "; ".join(validation_errors + env_errors)
      
      # Get streaming parameter information
      streaming_params = validator.get_streaming_parameters_found()
      if streaming_params:
        display_info["has_streaming_params"] = True
        display_info["streaming_params"] = streaming_params
      
      # Get warnings
      display_info["warnings"] = validator.get_warnings()
      
      # Extract defaults information
      display_info["defaults"] = config.defaults.copy()
      
      # Extract detailed provider information
      display_info["total_providers"] = len(config.providers)
      total_models = 0
      
      for provider_name, provider_config in config.providers.items():
        provider_info = {
          "models": provider_config.models.copy(),
          "model_count": len(provider_config.models),
          "base_url": provider_config.base_url,
          "timeout": provider_config.timeout,
          "max_retries": provider_config.max_retries,
          "default_params": provider_config.default_params.copy(),
          "provider_specific": provider_config.provider_specific.copy(),
          "has_custom_params": bool(provider_config.default_params or provider_config.provider_specific)
        }
        display_info["providers"][provider_name] = provider_info
        total_models += len(provider_config.models)
      
      display_info["total_models"] = total_models
      
    except Exception as e:
      # Handle any errors during configuration loading
      display_info["is_valid"] = False
      display_info["error_message"] = str(e)
    
    return display_info

  def display_all_configurations(self) -> list[dict[str, Any]]:
    """Display summary information for all available configurations.
    
    Returns:
      List of dictionaries containing summary information for each configuration
    """
    try:
      configs = self.discover_configuration_files()
    except ConfigurationError:
      return []
    
    # Return simplified summary information for all configurations
    summary_configs = []
    for config in configs:
      summary_info = {
        "name": config["name"],
        "file_path": config["file_path"],
        "is_valid": config["is_valid"],
        "error_message": config["error_message"],
        "has_streaming_params": config["has_streaming_params"],
        "streaming_params": config["streaming_params"],
        "warnings": config["warnings"],
        "total_providers": config["total_providers"],
        "total_models": config["total_models"],
        "provider_names": list(config["providers"].keys())
      }
      summary_configs.append(summary_info)
    
    return summary_configs

  def format_configuration_display(self, config_name: str) -> str:
    """Format configuration information as human-readable text.
    
    Args:
      config_name: Name of configuration to format
      
    Returns:
      Formatted text string with configuration information
      
    Raises:
      ConfigurationError: If configuration file not found
    """
    display_info = self.display_configuration(config_name)
    
    lines = []
    lines.append(f"Configuration: {display_info['name']}")
    lines.append(f"File Path: {display_info['file_path']}")
    lines.append(f"Status: {'Valid' if display_info['is_valid'] else 'Invalid'}")
    
    if not display_info["is_valid"] and display_info["error_message"]:
      lines.append(f"ERROR: {display_info['error_message']}")
    
    lines.append(f"Total Providers: {display_info['total_providers']}")
    lines.append(f"Total Models: {display_info['total_models']}")
    
    # Show warnings if present
    if display_info["warnings"]:
      lines.append("")
      lines.append("WARNINGS:")
      for warning in display_info["warnings"]:
        lines.append(f"  - {warning}")
    
    # Show streaming parameter warnings
    if display_info["has_streaming_params"]:
      lines.append("")
      lines.append("STREAMING PARAMETERS DETECTED:")
      lines.append("  The following streaming parameters were found and will be removed:")
      for param in display_info["streaming_params"]:
        lines.append(f"    - {param}")
      lines.append("  Streaming is not supported by this LLM connector.")
    
    # Show defaults if configuration is valid
    if display_info["is_valid"] and display_info["defaults"]:
      lines.append("")
      lines.append("Default Parameters:")
      for key, value in display_info["defaults"].items():
        lines.append(f"  {key}: {value}")
    
    # Show provider details if configuration is valid
    if display_info["is_valid"] and display_info["providers"]:
      lines.append("")
      lines.append("Providers:")
      
      for provider_name, provider_info in display_info["providers"].items():
        # Format provider name properly (handle special cases like OpenAI)
        formatted_name = provider_name.title()
        if provider_name.lower() == "openai":
          formatted_name = "OpenAI"
        elif provider_name.lower() == "openrouter":
          formatted_name = "OpenRouter"
        
        lines.append(f"  {formatted_name} Provider:")
        lines.append(f"    Models ({provider_info['model_count']}): {', '.join(provider_info['models'])}")
        
        if provider_info["base_url"]:
          lines.append(f"    Base URL: {provider_info['base_url']}")
        
        lines.append(f"    Timeout: {provider_info['timeout']}s")
        lines.append(f"    Max Retries: {provider_info['max_retries']}")
        
        if provider_info["default_params"]:
          lines.append("    Default Parameters:")
          for key, value in provider_info["default_params"].items():
            lines.append(f"      {key}: {value}")
        
        if provider_info["provider_specific"]:
          lines.append("    Provider-Specific Parameters:")
          for key, value in provider_info["provider_specific"].items():
            lines.append(f"      {key}: {value}")
        
        lines.append("")  # Empty line between providers
    
    return "\n".join(lines)

  def format_all_configurations_display(self) -> str:
    """Format summary information for all configurations as human-readable text.
    
    Returns:
      Formatted text string with summary of all configurations
    """
    summary = self.get_configuration_summary()
    all_configs = self.display_all_configurations()
    
    lines = []
    lines.append("Configuration Summary")
    lines.append("=" * 50)
    lines.append(f"Total Configurations: {summary['total_configurations']}")
    lines.append(f"Valid Configurations: {summary['valid_configurations']}")
    lines.append(f"Invalid Configurations: {summary['invalid_configurations']}")
    lines.append(f"Configurations with Streaming Parameters: {summary['configurations_with_streaming']}")
    lines.append(f"Total Models Available: {summary['total_models']}")
    
    if summary["all_providers"]:
      lines.append(f"All Providers: {', '.join(summary['all_providers'])}")
    
    if all_configs:
      lines.append("")
      lines.append("Individual Configurations:")
      lines.append("-" * 30)
      
      for config in all_configs:
        lines.append(f"  {config['name']}:")
        lines.append(f"    Status: {'Valid' if config['is_valid'] else 'Invalid'}")
        lines.append(f"    Providers: {config['total_providers']} ({', '.join(config['provider_names'])})")
        lines.append(f"    Models: {config['total_models']}")
        
        if config["has_streaming_params"]:
          lines.append(f"    Streaming Parameters: {', '.join(config['streaming_params'])}")
        
        if not config["is_valid"] and config["error_message"]:
          lines.append(f"    Error: {config['error_message']}")
        
        lines.append("")  # Empty line between configurations
    else:
      lines.append("")
      lines.append("No configuration files found.")
      lines.append("Use create_example_configuration_directory() to create example files.")
    
    return "\n".join(lines)