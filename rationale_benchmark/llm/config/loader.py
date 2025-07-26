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