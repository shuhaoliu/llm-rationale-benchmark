"""Configuration validator for LLM connector with comprehensive validation and streaming detection."""

import os
import re
import warnings
from typing import Any, Optional

from ..exceptions import ConfigurationError, StreamingNotSupportedError
from .models import LLMConfig, ProviderConfig


class ConfigValidator:
  """Validates LLM configuration structure and values with streaming parameter detection.
  
  This class provides comprehensive validation for LLM configurations including:
  - Required field validation for configuration structure
  - Data type validation for all configuration values
  - Value range validation for numeric parameters
  - Environment variable validation without exposing sensitive values
  - Streaming parameter detection and removal with warnings
  - Provider-specific configuration validation
  """

  # Known streaming-related parameters that should be blocked
  STREAMING_PARAMETERS = {
    "stream", "streaming", "stream_options", "stream_usage", 
    "stream_callback", "stream_handler", "incremental", "stream_mode"
  }

  # Valid provider types
  SUPPORTED_PROVIDERS = {
    "openai", "anthropic", "gemini", "openrouter", "local"
  }

  # Valid parameter ranges
  PARAMETER_RANGES = {
    "temperature": (0.0, 2.0),
    "max_tokens": (1, 100000),
    "timeout": (1, 300),
    "max_retries": (0, 10),
    "top_p": (0.0, 1.0),
    "frequency_penalty": (-2.0, 2.0),
    "presence_penalty": (-2.0, 2.0),
  }

  # Required environment variable patterns by provider
  ENV_VAR_PATTERNS = {
    "openai": r"^sk-[A-Za-z0-9]{48}$",
    "anthropic": r"^sk-ant-[A-Za-z0-9\-_]{95}$",
    "gemini": r"^[A-Za-z0-9\-_]{39}$",
    "openrouter": r"^sk-or-[A-Za-z0-9\-_]{48}$",
  }

  def __init__(self):
    """Initialize ConfigValidator."""
    self.validation_errors: list[str] = []
    self.warnings: list[str] = []
    self.streaming_params_found: list[str] = []

  def validate_config(self, config: LLMConfig) -> list[str]:
    """Validate complete configuration, return list of errors.
    
    Args:
      config: LLMConfig instance to validate
      
    Returns:
      List of validation error messages
    """
    self.validation_errors = []
    self.warnings = []
    self.streaming_params_found = []

    # Validate top-level structure
    self._validate_config_structure(config)
    
    # Validate defaults section
    self._validate_defaults(config.defaults)
    
    # Validate each provider
    for provider_name, provider_config in config.providers.items():
      self._validate_provider(provider_name, provider_config)
    
    # Check for streaming parameters across the entire configuration
    self._detect_streaming_parameters(config)
    
    # Issue warnings for streaming parameters found
    if self.streaming_params_found:
      warning_msg = f"Streaming parameters detected and will be removed: {', '.join(self.streaming_params_found)}"
      self.warnings.append(warning_msg)
      warnings.warn(warning_msg, UserWarning)

    return self.validation_errors

  def validate_environment_variables(self, config: LLMConfig) -> list[str]:
    """Validate required environment variables are present without exposing values.
    
    Args:
      config: LLMConfig instance to validate
      
    Returns:
      List of environment variable validation errors
    """
    env_errors = []
    
    for provider_name, provider_config in config.providers.items():
      # Check if API key looks like an environment variable reference
      api_key = provider_config.api_key
      
      if api_key.startswith("${") and api_key.endswith("}"):
        # Extract environment variable name
        env_var_match = re.match(r'\$\{([^}]+)\}', api_key)
        if env_var_match:
          env_var_name = env_var_match.group(1)
          
          # Check if environment variable exists
          env_value = os.getenv(env_var_name)
          if env_value is None:
            env_errors.append(
              f"Environment variable '{env_var_name}' required for provider '{provider_name}' is not set"
            )
          else:
            # Validate format without exposing the actual value
            validation_error = self._validate_api_key_format(
              provider_name, env_var_name, env_value
            )
            if validation_error:
              env_errors.append(validation_error)
      else:
        # Direct API key - validate format
        validation_error = self._validate_api_key_format(
          provider_name, "api_key", api_key
        )
        if validation_error:
          env_errors.append(validation_error)

    return env_errors

  def remove_streaming_parameters(self, config_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove streaming parameters from configuration dictionary with warnings.
    
    Args:
      config_dict: Configuration dictionary that may contain streaming parameters
      
    Returns:
      Configuration dictionary with streaming parameters removed
    """
    cleaned_config = {}
    removed_params = []

    for key, value in config_dict.items():
      if key.lower() in self.STREAMING_PARAMETERS:
        removed_params.append(key)
        warnings.warn(
          f"Removed streaming parameter '{key}' from configuration",
          UserWarning
        )
      elif isinstance(value, dict):
        # Recursively clean nested dictionaries
        cleaned_value = self.remove_streaming_parameters(value)
        cleaned_config[key] = cleaned_value
      elif isinstance(value, list):
        # Clean list items if they are dictionaries
        cleaned_list = []
        for item in value:
          if isinstance(item, dict):
            cleaned_item = self.remove_streaming_parameters(item)
            cleaned_list.append(cleaned_item)
          else:
            cleaned_list.append(item)
        cleaned_config[key] = cleaned_list
      else:
        cleaned_config[key] = value

    if removed_params:
      self.streaming_params_found.extend(removed_params)

    return cleaned_config

  def get_warnings(self) -> list[str]:
    """Get list of validation warnings.
    
    Returns:
      List of warning messages
    """
    return self.warnings.copy()

  def get_streaming_parameters_found(self) -> list[str]:
    """Get list of streaming parameters that were detected.
    
    Returns:
      List of streaming parameter names found
    """
    return self.streaming_params_found.copy()

  def _validate_config_structure(self, config: LLMConfig) -> None:
    """Validate top-level configuration structure.
    
    Args:
      config: LLMConfig instance to validate
    """
    if not isinstance(config.defaults, dict):
      self.validation_errors.append("Configuration 'defaults' must be a dictionary")

    if not isinstance(config.providers, dict):
      self.validation_errors.append("Configuration 'providers' must be a dictionary")

    if not config.providers:
      self.validation_errors.append("At least one provider must be configured")

  def _validate_defaults(self, defaults: dict[str, Any]) -> None:
    """Validate defaults section of configuration.
    
    Args:
      defaults: Defaults dictionary to validate
    """
    if not isinstance(defaults, dict):
      self.validation_errors.append("Defaults section must be a dictionary")
      return

    # Validate common parameters in defaults
    for param_name, param_value in defaults.items():
      self._validate_parameter_value(param_name, param_value, "defaults")

    # Check for streaming parameters in defaults
    streaming_in_defaults = [
      param for param in defaults.keys() 
      if param.lower() in self.STREAMING_PARAMETERS
    ]
    if streaming_in_defaults:
      self.streaming_params_found.extend(streaming_in_defaults)

  def _validate_provider(self, provider_name: str, provider_config: ProviderConfig) -> None:
    """Validate individual provider configuration.
    
    Args:
      provider_name: Name of the provider
      provider_config: ProviderConfig instance to validate
    """
    # Validate required fields
    if not provider_config.name:
      self.validation_errors.append(f"Provider '{provider_name}' name cannot be empty")

    if not provider_config.api_key:
      self.validation_errors.append(f"Provider '{provider_name}' must have an API key")

    # Validate data types
    if not isinstance(provider_config.timeout, int):
      self.validation_errors.append(
        f"Provider '{provider_name}' timeout must be an integer"
      )

    if not isinstance(provider_config.max_retries, int):
      self.validation_errors.append(
        f"Provider '{provider_name}' max_retries must be an integer"
      )

    if not isinstance(provider_config.models, list):
      self.validation_errors.append(
        f"Provider '{provider_name}' models must be a list"
      )

    if not isinstance(provider_config.default_params, dict):
      self.validation_errors.append(
        f"Provider '{provider_name}' default_params must be a dictionary"
      )

    if not isinstance(provider_config.provider_specific, dict):
      self.validation_errors.append(
        f"Provider '{provider_name}' provider_specific must be a dictionary"
      )

    # Validate value ranges (only if they are the correct type)
    if isinstance(provider_config.timeout, int) and provider_config.timeout <= 0:
      self.validation_errors.append(
        f"Provider '{provider_name}' timeout must be positive"
      )

    if isinstance(provider_config.max_retries, int) and provider_config.max_retries < 0:
      self.validation_errors.append(
        f"Provider '{provider_name}' max_retries cannot be negative"
      )

    # Validate base_url format if provided
    if provider_config.base_url is not None:
      if not isinstance(provider_config.base_url, str):
        self.validation_errors.append(
          f"Provider '{provider_name}' base_url must be a string"
        )
      elif not (provider_config.base_url.startswith("http://") or 
                provider_config.base_url.startswith("https://")):
        self.validation_errors.append(
          f"Provider '{provider_name}' base_url must start with http:// or https://"
        )

    # Validate models list
    for i, model in enumerate(provider_config.models):
      if not isinstance(model, str):
        self.validation_errors.append(
          f"Provider '{provider_name}' model[{i}] must be a string"
        )
      elif not model.strip():
        self.validation_errors.append(
          f"Provider '{provider_name}' model[{i}] cannot be empty"
        )

    # Validate default_params (only if it's a dict)
    if isinstance(provider_config.default_params, dict):
      for param_name, param_value in provider_config.default_params.items():
        self._validate_parameter_value(
          param_name, param_value, f"providers.{provider_name}.default_params"
        )

    # Check for streaming parameters in provider configuration
    streaming_in_provider = []
    if isinstance(provider_config.default_params, dict):
      streaming_in_provider.extend([
        param for param in provider_config.default_params.keys()
        if param.lower() in self.STREAMING_PARAMETERS
      ])
    if isinstance(provider_config.provider_specific, dict):
      streaming_in_provider.extend([
        param for param in provider_config.provider_specific.keys()
        if param.lower() in self.STREAMING_PARAMETERS
      ])
    
    if streaming_in_provider:
      self.streaming_params_found.extend(streaming_in_provider)

  def _validate_parameter_value(self, param_name: str, param_value: Any, context: str) -> None:
    """Validate individual parameter value and range.
    
    Args:
      param_name: Name of the parameter
      param_value: Value of the parameter
      context: Context string for error reporting
    """
    # Check if parameter is in known ranges
    if param_name in self.PARAMETER_RANGES:
      min_val, max_val = self.PARAMETER_RANGES[param_name]
      
      # Validate numeric parameters
      if param_name in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
        if not isinstance(param_value, (int, float)):
          self.validation_errors.append(
            f"Parameter '{param_name}' in {context} must be a number"
          )
        elif not (min_val <= param_value <= max_val):
          self.validation_errors.append(
            f"Parameter '{param_name}' in {context} must be between {min_val} and {max_val}"
          )
      
      elif param_name in ["max_tokens", "timeout", "max_retries"]:
        if not isinstance(param_value, int):
          self.validation_errors.append(
            f"Parameter '{param_name}' in {context} must be an integer"
          )
        elif not (min_val <= param_value <= max_val):
          self.validation_errors.append(
            f"Parameter '{param_name}' in {context} must be between {min_val} and {max_val}"
          )

    # Validate specific parameter types
    if param_name == "system_prompt" and param_value is not None:
      if not isinstance(param_value, str):
        self.validation_errors.append(
          f"Parameter 'system_prompt' in {context} must be a string"
        )

    if param_name == "stop_sequences" and param_value is not None:
      if not isinstance(param_value, list):
        self.validation_errors.append(
          f"Parameter 'stop_sequences' in {context} must be a list"
        )
      else:
        for i, seq in enumerate(param_value):
          if not isinstance(seq, str):
            self.validation_errors.append(
              f"Parameter 'stop_sequences[{i}]' in {context} must be a string"
            )

  def _detect_streaming_parameters(self, config: LLMConfig) -> None:
    """Detect streaming parameters throughout the configuration.
    
    Args:
      config: LLMConfig instance to check for streaming parameters
    """
    # Check defaults (only if it's a dict)
    if isinstance(config.defaults, dict):
      for param in config.defaults.keys():
        if param.lower() in self.STREAMING_PARAMETERS:
          if param not in self.streaming_params_found:
            self.streaming_params_found.append(param)

    # Check each provider
    for provider_name, provider_config in config.providers.items():
      # Check default_params (only if it's a dict)
      if isinstance(provider_config.default_params, dict):
        for param in provider_config.default_params.keys():
          if param.lower() in self.STREAMING_PARAMETERS:
            if param not in self.streaming_params_found:
              self.streaming_params_found.append(param)

      # Check provider_specific (only if it's a dict)
      if isinstance(provider_config.provider_specific, dict):
        for param in provider_config.provider_specific.keys():
          if param.lower() in self.STREAMING_PARAMETERS:
            if param not in self.streaming_params_found:
              self.streaming_params_found.append(param)

  def _validate_api_key_format(
    self, provider_name: str, key_name: str, api_key: str
  ) -> Optional[str]:
    """Validate API key format without exposing the actual key value.
    
    Args:
      provider_name: Name of the provider
      key_name: Name of the key field (for error reporting)
      api_key: API key value to validate
      
    Returns:
      Error message if validation fails, None if valid
    """
    if not api_key or not isinstance(api_key, str):
      return f"Provider '{provider_name}' {key_name} must be a non-empty string"

    # Check for common API key format issues without exposing the key
    if len(api_key.strip()) != len(api_key):
      return f"Provider '{provider_name}' {key_name} contains leading/trailing whitespace"

    if len(api_key) < 10:
      return f"Provider '{provider_name}' {key_name} appears to be too short"

    # Validate against known patterns if available
    provider_type = provider_name.lower()
    if provider_type in self.ENV_VAR_PATTERNS:
      pattern = self.ENV_VAR_PATTERNS[provider_type]
      if not re.match(pattern, api_key):
        return f"Provider '{provider_name}' {key_name} format appears invalid for {provider_type}"

    return None

  def validate_streaming_prevention(self, request_params: dict[str, Any]) -> None:
    """Validate that no streaming parameters are present in request parameters.
    
    Args:
      request_params: Request parameters dictionary to validate
      
    Raises:
      StreamingNotSupportedError: If streaming parameters are detected
    """
    streaming_found = []
    
    for param_name in request_params.keys():
      if param_name.lower() in self.STREAMING_PARAMETERS:
        streaming_found.append(param_name)
    
    if streaming_found:
      raise StreamingNotSupportedError(
        f"Streaming parameters are not supported: {', '.join(streaming_found)}",
        blocked_params=streaming_found
      )

  def create_validation_report(self, config: LLMConfig) -> dict[str, Any]:
    """Create a comprehensive validation report for the configuration.
    
    Args:
      config: LLMConfig instance to validate
      
    Returns:
      Dictionary containing validation results and statistics
    """
    # Run validation
    errors = self.validate_config(config)
    env_errors = self.validate_environment_variables(config)
    
    # Collect statistics
    total_providers = len(config.providers)
    total_models = sum(len(provider.models) for provider in config.providers.values())
    
    return {
      "validation_passed": len(errors) == 0 and len(env_errors) == 0,
      "total_errors": len(errors) + len(env_errors),
      "configuration_errors": errors,
      "environment_errors": env_errors,
      "warnings": self.get_warnings(),
      "streaming_parameters_found": self.get_streaming_parameters_found(),
      "statistics": {
        "total_providers": total_providers,
        "total_models": total_models,
        "providers_configured": list(config.providers.keys()),
      }
    }