"""Data models for LLM connector configuration and request/response handling."""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from .exceptions import ConfigurationError


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    name: str
    api_key: str
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    models: list[str] = field(default_factory=list)
    default_params: dict[str, Any] = field(default_factory=dict)
    provider_specific: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate provider configuration after initialization."""
        if not self.name:
            raise ConfigurationError("Provider name cannot be empty")

        if not self.api_key:
            raise ConfigurationError(f"API key is required for provider '{self.name}'")

        if self.timeout <= 0:
            raise ConfigurationError(
                f"Timeout must be positive for provider '{self.name}'"
            )

        if self.max_retries < 0:
            raise ConfigurationError(
                f"Max retries cannot be negative for provider '{self.name}'"
            )


@dataclass
class LLMConfig:
    """Complete LLM configuration with all providers."""

    defaults: dict[str, Any]
    providers: dict[str, ProviderConfig]

    @classmethod
    def from_file(
        cls, config_dir: Path, config_name: str = "default-llms"
    ) -> "LLMConfig":
        """Load configuration from a single file.

        Args:
          config_dir: Directory containing configuration files
          config_name: Name of configuration file (without .yaml extension)

        Returns:
          LLMConfig instance loaded from file

        Raises:
          ConfigurationError: If file not found or invalid
        """
        config_file = config_dir / f"{config_name}.yaml"

        if not config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_file}",
                config_file=str(config_file),
            )

        try:
            with open(config_file) as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}", config_file=str(config_file)
            )
        except Exception as e:
            raise ConfigurationError(
                f"Error reading configuration file: {e}", config_file=str(config_file)
            )

        if not isinstance(config_dict, dict):
            raise ConfigurationError(
                "Configuration file must contain a YAML dictionary",
                config_file=str(config_file),
            )

        # Resolve environment variables
        resolved_config = cls._resolve_environment_variables(config_dict)

        return cls.from_dict(resolved_config, str(config_file))

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], config_file: str = ""
    ) -> "LLMConfig":
        """Create LLMConfig from dictionary.

        Args:
          config_dict: Configuration dictionary
          config_file: Source file path for error reporting

        Returns:
          LLMConfig instance

        Raises:
          ConfigurationError: If configuration is invalid
        """
        # Extract defaults
        defaults = config_dict.get("defaults", {})
        if not isinstance(defaults, dict):
            raise ConfigurationError(
                "Defaults section must be a dictionary",
                config_file=config_file,
                field="defaults",
            )

        # Extract providers
        providers_dict = config_dict.get("providers", {})
        if not isinstance(providers_dict, dict):
            raise ConfigurationError(
                "Providers section must be a dictionary",
                config_file=config_file,
                field="providers",
            )

        if not providers_dict:
            raise ConfigurationError(
                "At least one provider must be configured",
                config_file=config_file,
                field="providers",
            )

        # Create provider configurations
        providers = {}
        for provider_name, provider_config in providers_dict.items():
            if not isinstance(provider_config, dict):
                raise ConfigurationError(
                    f"Provider '{provider_name}' configuration must be a dictionary",
                    config_file=config_file,
                    field=f"providers.{provider_name}",
                )

            try:
                # Merge with defaults
                merged_config = {**defaults, **provider_config}
                merged_config["name"] = provider_name

                providers[provider_name] = ProviderConfig(**merged_config)
            except TypeError as e:
                raise ConfigurationError(
                    f"Invalid configuration for provider '{provider_name}': {e}",
                    config_file=config_file,
                    field=f"providers.{provider_name}",
                )

        return cls(defaults=defaults, providers=providers)

    @staticmethod
    def _resolve_environment_variables(config_dict: dict[str, Any]) -> dict[str, Any]:
        """Resolve ${VAR} patterns in configuration with environment variables.

        Args:
          config_dict: Configuration dictionary potentially containing ${VAR} patterns

        Returns:
          Dictionary with environment variables resolved

        Raises:
          ConfigurationError: If required environment variable is missing
        """

        def resolve_value(value: Any) -> Any:
            if isinstance(value, str):
                # Find all ${VAR} patterns
                pattern = r"\$\{([^}]+)\}"
                matches = re.findall(pattern, value)

                resolved_value = value
                for var_name in matches:
                    env_value = os.getenv(var_name)
                    if env_value is None:
                        raise ConfigurationError(
                            f"Environment variable '{var_name}' is not set"
                        )
                    resolved_value = resolved_value.replace(
                        f"${{{var_name}}}", env_value
                    )

                return resolved_value
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value

        return resolve_value(config_dict)


@dataclass
class ModelRequest:
    """Request parameters for LLM generation."""

    prompt: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    stop_sequences: Optional[list[str]] = None
    provider_specific: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate request parameters after initialization."""
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

        if not self.model:
            raise ValueError("Model cannot be empty")

        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")

        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

        if self.stop_sequences is not None and not isinstance(
            self.stop_sequences, list
        ):
            raise ValueError("Stop sequences must be a list")


@dataclass
class ModelResponse:
    """Standardized response from LLM providers."""

    text: str
    model: str
    provider: str
    timestamp: datetime
    latency_ms: int
    token_count: Optional[int] = None
    finish_reason: Optional[str] = None
    cost_estimate: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate response parameters after initialization."""
        if not self.text:
            raise ValueError("Response text cannot be empty")

        if not self.model:
            raise ValueError("Model cannot be empty")

        if not self.provider:
            raise ValueError("Provider cannot be empty")

        if self.latency_ms < 0:
            raise ValueError("Latency cannot be negative")

        if self.token_count is not None and self.token_count < 0:
            raise ValueError("Token count cannot be negative")

        if self.cost_estimate is not None and self.cost_estimate < 0:
            raise ValueError("Cost estimate cannot be negative")
