"""Unit tests for LLM data models."""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rationale_benchmark.llm.exceptions import ConfigurationError
from rationale_benchmark.llm.models import (
  LLMConfig,
  ModelRequest,
  ModelResponse,
  ProviderConfig,
)


class TestProviderConfig:
  """Test cases for ProviderConfig data class."""

  def test_provider_config_creation_with_valid_data(self):
    """Test that ProviderConfig can be created with valid data."""
    # Arrange
    config = ProviderConfig(
      name="openai",
      api_key="sk-test123",
      base_url="https://api.openai.com/v1",
      timeout=60,
      max_retries=5,
      models=["gpt-4", "gpt-3.5-turbo"],
      default_params={"temperature": 0.7},
      provider_specific={"organization": "test-org"}
    )

    # Act & Assert
    assert config.name == "openai"
    assert config.api_key == "sk-test123"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.timeout == 60
    assert config.max_retries == 5
    assert config.models == ["gpt-4", "gpt-3.5-turbo"]
    assert config.default_params == {"temperature": 0.7}
    assert config.provider_specific == {"organization": "test-org"}

  def test_provider_config_creation_with_minimal_data(self):
    """Test that ProviderConfig can be created with minimal required data."""
    # Arrange & Act
    config = ProviderConfig(name="test", api_key="key123")

    # Assert
    assert config.name == "test"
    assert config.api_key == "key123"
    assert config.base_url is None
    assert config.timeout == 30
    assert config.max_retries == 3
    assert config.models == []
    assert config.default_params == {}
    assert config.provider_specific == {}

  def test_provider_config_raises_error_for_empty_name(self):
    """Test that ProviderConfig raises error for empty name."""
    # Act & Assert
    with pytest.raises(ConfigurationError, match="Provider name cannot be empty"):
      ProviderConfig(name="", api_key="key123")

  def test_provider_config_raises_error_for_empty_api_key(self):
    """Test that ProviderConfig raises error for empty API key."""
    # Act & Assert
    with pytest.raises(ConfigurationError, match="API key is required"):
      ProviderConfig(name="test", api_key="")

  def test_provider_config_raises_error_for_negative_timeout(self):
    """Test that ProviderConfig raises error for negative timeout."""
    # Act & Assert
    with pytest.raises(ConfigurationError, match="Timeout must be positive"):
      ProviderConfig(name="test", api_key="key123", timeout=-1)

  def test_provider_config_raises_error_for_zero_timeout(self):
    """Test that ProviderConfig raises error for zero timeout."""
    # Act & Assert
    with pytest.raises(ConfigurationError, match="Timeout must be positive"):
      ProviderConfig(name="test", api_key="key123", timeout=0)

  def test_provider_config_raises_error_for_negative_max_retries(self):
    """Test that ProviderConfig raises error for negative max_retries."""
    # Act & Assert
    with pytest.raises(ConfigurationError, match="Max retries cannot be negative"):
      ProviderConfig(name="test", api_key="key123", max_retries=-1)

  def test_provider_config_allows_zero_max_retries(self):
    """Test that ProviderConfig allows zero max_retries."""
    # Arrange & Act
    config = ProviderConfig(name="test", api_key="key123", max_retries=0)

    # Assert
    assert config.max_retries == 0

  def test_provider_config_field_constraints(self):
    """Test various field constraint validations."""
    # Test valid edge cases
    config = ProviderConfig(
      name="a",  # Single character name
      api_key="x",  # Single character key
      timeout=1,  # Minimum positive timeout
      max_retries=0  # Zero retries allowed
    )
    assert config.name == "a"
    assert config.api_key == "x"
    assert config.timeout == 1
    assert config.max_retries == 0


class TestLLMConfig:
  """Test cases for LLMConfig data class."""

  def test_llm_config_creation_with_valid_data(self):
    """Test that LLMConfig can be created with valid data."""
    # Arrange
    defaults = {"temperature": 0.7, "max_tokens": 1000}
    providers = {
      "openai": ProviderConfig(name="openai", api_key="sk-test123"),
      "anthropic": ProviderConfig(name="anthropic", api_key="ant-test456")
    }

    # Act
    config = LLMConfig(defaults=defaults, providers=providers)

    # Assert
    assert config.defaults == defaults
    assert config.providers == providers
    assert len(config.providers) == 2

  def test_llm_config_from_dict_with_valid_data(self):
    """Test LLMConfig.from_dict with valid configuration dictionary."""
    # Arrange
    config_dict = {
      "defaults": {
        "timeout": 60,
        "max_retries": 5
      },
      "providers": {
        "openai": {
          "api_key": "sk-test123",
          "models": ["gpt-4"]
        },
        "anthropic": {
          "api_key": "ant-test456",
          "base_url": "https://api.anthropic.com"
        }
      }
    }

    # Act
    config = LLMConfig.from_dict(config_dict)

    # Assert
    assert config.defaults == {"timeout": 60, "max_retries": 5}
    assert len(config.providers) == 2
    assert "openai" in config.providers
    assert "anthropic" in config.providers
    assert config.providers["openai"].name == "openai"
    assert config.providers["openai"].api_key == "sk-test123"
    assert config.providers["openai"].models == ["gpt-4"]
    assert config.providers["openai"].timeout == 60  # From defaults
    assert config.providers["openai"].max_retries == 5  # From defaults
    assert config.providers["anthropic"].name == "anthropic"
    assert config.providers["anthropic"].api_key == "ant-test456"
    assert config.providers["anthropic"].base_url == "https://api.anthropic.com"

  def test_llm_config_from_dict_merges_defaults_with_provider_config(self):
    """Test that from_dict merges defaults with provider-specific config."""
    # Arrange
    config_dict = {
      "defaults": {
        "timeout": 30,
        "max_retries": 3
      },
      "providers": {
        "openai": {
          "api_key": "sk-test123",
          "timeout": 60  # Override default timeout
        }
      }
    }

    # Act
    config = LLMConfig.from_dict(config_dict)

    # Assert
    provider = config.providers["openai"]
    assert provider.timeout == 60  # Overridden
    assert provider.max_retries == 3  # From defaults

  def test_llm_config_from_dict_raises_error_for_invalid_defaults(self):
    """Test that from_dict raises error for invalid defaults section."""
    # Arrange
    config_dict = {
      "defaults": "not a dict",
      "providers": {
        "openai": {"api_key": "sk-test123"}
      }
    }

    # Act & Assert
    with pytest.raises(ConfigurationError, match="Defaults section must be a dictionary"):
      LLMConfig.from_dict(config_dict)

  def test_llm_config_from_dict_raises_error_for_invalid_providers(self):
    """Test that from_dict raises error for invalid providers section."""
    # Arrange
    config_dict = {
      "defaults": {},
      "providers": "not a dict"
    }

    # Act & Assert
    with pytest.raises(ConfigurationError, match="Providers section must be a dictionary"):
      LLMConfig.from_dict(config_dict)

  def test_llm_config_from_dict_raises_error_for_empty_providers(self):
    """Test that from_dict raises error for empty providers section."""
    # Arrange
    config_dict = {
      "defaults": {},
      "providers": {}
    }

    # Act & Assert
    with pytest.raises(ConfigurationError, match="At least one provider must be configured"):
      LLMConfig.from_dict(config_dict)

  def test_llm_config_from_dict_raises_error_for_invalid_provider_config(self):
    """Test that from_dict raises error for invalid provider configuration."""
    # Arrange
    config_dict = {
      "defaults": {},
      "providers": {
        "openai": "not a dict"
      }
    }

    # Act & Assert
    with pytest.raises(ConfigurationError, match="Provider 'openai' configuration must be a dictionary"):
      LLMConfig.from_dict(config_dict)

  def test_llm_config_from_dict_handles_unknown_fields_gracefully(self):
    """Test that from_dict handles unknown fields by putting them in default_params."""
    # Arrange - Create config with unknown field
    config_dict = {
      "defaults": {},
      "providers": {
        "openai": {
          "api_key": "sk-test123",
          "unknown_field": "value"  # This should go to default_params
        }
      }
    }

    # Act
    config = LLMConfig.from_dict(config_dict)

    # Assert - Unknown field should be in default_params
    assert "unknown_field" in config.providers["openai"].default_params
    assert config.providers["openai"].default_params["unknown_field"] == "value"

  def test_llm_config_from_file_loads_valid_yaml(self):
    """Test that from_file loads valid YAML configuration."""
    # Arrange
    config_data = {
      "defaults": {"timeout": 45},
      "providers": {
        "openai": {"api_key": "sk-test123"}
      }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
      config_dir = Path(temp_dir)
      config_file = config_dir / "test-config.yaml"
      
      with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

      # Act
      config = LLMConfig.from_file(config_dir, "test-config")

      # Assert
      assert config.defaults == {"timeout": 45}
      assert len(config.providers) == 1
      assert "openai" in config.providers
      assert config.providers["openai"].timeout == 45

  def test_llm_config_from_file_raises_error_for_missing_file(self):
    """Test that from_file raises error for missing configuration file."""
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
      config_dir = Path(temp_dir)

      # Act & Assert
      with pytest.raises(ConfigurationError, match="Configuration file not found"):
        LLMConfig.from_file(config_dir, "nonexistent")

  def test_llm_config_from_file_raises_error_for_invalid_yaml(self):
    """Test that from_file raises error for invalid YAML."""
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
      config_dir = Path(temp_dir)
      config_file = config_dir / "invalid.yaml"
      
      with open(config_file, 'w') as f:
        f.write("invalid: yaml: content: [")

      # Act & Assert
      with pytest.raises(ConfigurationError, match="Invalid YAML in configuration file"):
        LLMConfig.from_file(config_dir, "invalid")

  def test_llm_config_from_file_raises_error_for_non_dict_yaml(self):
    """Test that from_file raises error for non-dictionary YAML."""
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
      config_dir = Path(temp_dir)
      config_file = config_dir / "list.yaml"
      
      with open(config_file, 'w') as f:
        yaml.dump(["not", "a", "dict"], f)

      # Act & Assert
      with pytest.raises(ConfigurationError, match="Configuration file must contain a YAML dictionary"):
        LLMConfig.from_file(config_dir, "list")

  def test_llm_config_from_file_raises_error_for_file_read_error(self):
    """Test that from_file raises error for general file reading errors."""
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
      config_dir = Path(temp_dir)
      config_file = config_dir / "test.yaml"
      
      # Create file and then make it unreadable
      with open(config_file, 'w') as f:
        f.write("test: value")
      
      # Make file unreadable (this might not work on all systems)
      import stat
      os.chmod(config_file, 0o000)
      
      try:
        # Act & Assert
        with pytest.raises(ConfigurationError, match="Error reading configuration file"):
          LLMConfig.from_file(config_dir, "test")
      finally:
        # Restore permissions for cleanup
        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

  @patch.dict(os.environ, {"TEST_API_KEY": "resolved-key-123"})
  def test_llm_config_resolves_environment_variables(self):
    """Test that LLMConfig resolves environment variables in configuration."""
    # Arrange
    config_data = {
      "defaults": {},
      "providers": {
        "openai": {"api_key": "${TEST_API_KEY}"}
      }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
      config_dir = Path(temp_dir)
      config_file = config_dir / "env-test.yaml"
      
      with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

      # Act
      config = LLMConfig.from_file(config_dir, "env-test")

      # Assert
      assert config.providers["openai"].api_key == "resolved-key-123"

  def test_llm_config_raises_error_for_missing_environment_variable(self):
    """Test that LLMConfig raises error for missing environment variable."""
    # Arrange
    config_data = {
      "defaults": {},
      "providers": {
        "openai": {"api_key": "${NONEXISTENT_VAR}"}
      }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
      config_dir = Path(temp_dir)
      config_file = config_dir / "missing-env.yaml"
      
      with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

      # Act & Assert
      with pytest.raises(ConfigurationError, match="Environment variable 'NONEXISTENT_VAR' is not set"):
        LLMConfig.from_file(config_dir, "missing-env")

  @patch.dict(os.environ, {"NESTED_VAR": "nested-value"})
  def test_llm_config_resolves_nested_environment_variables(self):
    """Test that LLMConfig resolves environment variables in nested structures."""
    # Arrange
    config_data = {
      "defaults": {
        "timeout": 30,
        "provider_specific": {
          "nested_value": "${NESTED_VAR}"
        }
      },
      "providers": {
        "openai": {"api_key": "sk-test123"}
      }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
      config_dir = Path(temp_dir)
      config_file = config_dir / "nested-env.yaml"
      
      with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

      # Act
      config = LLMConfig.from_file(config_dir, "nested-env")

      # Assert
      assert config.defaults["provider_specific"]["nested_value"] == "nested-value"
      assert config.providers["openai"].provider_specific["nested_value"] == "nested-value"

  def test_llm_config_resolves_environment_variables_with_non_string_values(self):
    """Test that LLMConfig handles non-string values in environment variable resolution."""
    # Arrange
    config_data = {
      "defaults": {
        "timeout": 30,  # Integer value
        "models": ["gpt-4", "gpt-3.5-turbo"],  # List value
        "default_params": {
          "temperature": 0.7,  # Nested dict with float
          "enabled": True  # Boolean value
        }
      },
      "providers": {
        "openai": {"api_key": "sk-test123"}
      }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
      config_dir = Path(temp_dir)
      config_file = config_dir / "non-string-env.yaml"
      
      with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

      # Act
      config = LLMConfig.from_file(config_dir, "non-string-env")

      # Assert - Non-string values should be preserved as-is
      assert config.defaults["timeout"] == 30
      assert config.defaults["models"] == ["gpt-4", "gpt-3.5-turbo"]
      assert config.defaults["default_params"]["temperature"] == 0.7
      assert config.defaults["default_params"]["enabled"] is True
      # Check that these values are also passed to the provider
      assert config.providers["openai"].timeout == 30
      assert config.providers["openai"].models == ["gpt-4", "gpt-3.5-turbo"]
      assert config.providers["openai"].default_params["temperature"] == 0.7
      assert config.providers["openai"].default_params["enabled"] is True


class TestModelRequest:
  """Test cases for ModelRequest data class."""

  def test_model_request_creation_with_valid_data(self):
    """Test that ModelRequest can be created with valid data."""
    # Arrange & Act
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.8,
      max_tokens=2000,
      system_prompt="You are a helpful assistant",
      stop_sequences=["END", "STOP"],
      provider_specific={"top_p": 0.9}
    )

    # Assert
    assert request.prompt == "Test prompt"
    assert request.model == "gpt-4"
    assert request.temperature == 0.8
    assert request.max_tokens == 2000
    assert request.system_prompt == "You are a helpful assistant"
    assert request.stop_sequences == ["END", "STOP"]
    assert request.provider_specific == {"top_p": 0.9}

  def test_model_request_creation_with_minimal_data(self):
    """Test that ModelRequest can be created with minimal required data."""
    # Arrange & Act
    request = ModelRequest(prompt="Test", model="gpt-3.5-turbo")

    # Assert
    assert request.prompt == "Test"
    assert request.model == "gpt-3.5-turbo"
    assert request.temperature == 0.7
    assert request.max_tokens == 1000
    assert request.system_prompt is None
    assert request.stop_sequences is None
    assert request.provider_specific == {}

  def test_model_request_raises_error_for_empty_prompt(self):
    """Test that ModelRequest raises error for empty prompt."""
    # Act & Assert
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
      ModelRequest(prompt="", model="gpt-4")

  def test_model_request_raises_error_for_empty_model(self):
    """Test that ModelRequest raises error for empty model."""
    # Act & Assert
    with pytest.raises(ValueError, match="Model cannot be empty"):
      ModelRequest(prompt="Test", model="")

  def test_model_request_raises_error_for_invalid_temperature_too_low(self):
    """Test that ModelRequest raises error for temperature below 0.0."""
    # Act & Assert
    with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
      ModelRequest(prompt="Test", model="gpt-4", temperature=-0.1)

  def test_model_request_raises_error_for_invalid_temperature_too_high(self):
    """Test that ModelRequest raises error for temperature above 2.0."""
    # Act & Assert
    with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
      ModelRequest(prompt="Test", model="gpt-4", temperature=2.1)

  def test_model_request_allows_temperature_boundary_values(self):
    """Test that ModelRequest allows temperature boundary values 0.0 and 2.0."""
    # Arrange & Act
    request_min = ModelRequest(prompt="Test", model="gpt-4", temperature=0.0)
    request_max = ModelRequest(prompt="Test", model="gpt-4", temperature=2.0)

    # Assert
    assert request_min.temperature == 0.0
    assert request_max.temperature == 2.0

  def test_model_request_raises_error_for_negative_max_tokens(self):
    """Test that ModelRequest raises error for negative max_tokens."""
    # Act & Assert
    with pytest.raises(ValueError, match="Max tokens must be positive"):
      ModelRequest(prompt="Test", model="gpt-4", max_tokens=-1)

  def test_model_request_raises_error_for_zero_max_tokens(self):
    """Test that ModelRequest raises error for zero max_tokens."""
    # Act & Assert
    with pytest.raises(ValueError, match="Max tokens must be positive"):
      ModelRequest(prompt="Test", model="gpt-4", max_tokens=0)

  def test_model_request_raises_error_for_invalid_stop_sequences_type(self):
    """Test that ModelRequest raises error for non-list stop_sequences."""
    # Act & Assert
    with pytest.raises(ValueError, match="Stop sequences must be a list"):
      ModelRequest(prompt="Test", model="gpt-4", stop_sequences="not a list")

  def test_model_request_allows_empty_stop_sequences_list(self):
    """Test that ModelRequest allows empty stop_sequences list."""
    # Arrange & Act
    request = ModelRequest(prompt="Test", model="gpt-4", stop_sequences=[])

    # Assert
    assert request.stop_sequences == []

  def test_model_request_field_constraints(self):
    """Test various field constraint validations."""
    # Test valid edge cases
    request = ModelRequest(
      prompt="x",  # Single character prompt
      model="m",  # Single character model
      temperature=1.0,  # Mid-range temperature
      max_tokens=1,  # Minimum positive tokens
      stop_sequences=["a", "b", "c"]  # Multiple stop sequences
    )
    assert request.prompt == "x"
    assert request.model == "m"
    assert request.temperature == 1.0
    assert request.max_tokens == 1
    assert len(request.stop_sequences) == 3


class TestModelResponse:
  """Test cases for ModelResponse data class."""

  def test_model_response_creation_with_valid_data(self):
    """Test that ModelResponse can be created with valid data."""
    # Arrange
    timestamp = datetime.now()

    # Act
    response = ModelResponse(
      text="Generated response text",
      model="gpt-4",
      provider="openai",
      timestamp=timestamp,
      latency_ms=1500,
      token_count=25,
      finish_reason="stop",
      cost_estimate=0.003,
      metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 15}}
    )

    # Assert
    assert response.text == "Generated response text"
    assert response.model == "gpt-4"
    assert response.provider == "openai"
    assert response.timestamp == timestamp
    assert response.latency_ms == 1500
    assert response.token_count == 25
    assert response.finish_reason == "stop"
    assert response.cost_estimate == 0.003
    assert response.metadata == {"usage": {"prompt_tokens": 10, "completion_tokens": 15}}

  def test_model_response_creation_with_minimal_data(self):
    """Test that ModelResponse can be created with minimal required data."""
    # Arrange
    timestamp = datetime.now()

    # Act
    response = ModelResponse(
      text="Response",
      model="gpt-3.5-turbo",
      provider="openai",
      timestamp=timestamp,
      latency_ms=800
    )

    # Assert
    assert response.text == "Response"
    assert response.model == "gpt-3.5-turbo"
    assert response.provider == "openai"
    assert response.timestamp == timestamp
    assert response.latency_ms == 800
    assert response.token_count is None
    assert response.finish_reason is None
    assert response.cost_estimate is None
    assert response.metadata == {}

  def test_model_response_raises_error_for_empty_text(self):
    """Test that ModelResponse raises error for empty text."""
    # Act & Assert
    with pytest.raises(ValueError, match="Response text cannot be empty"):
      ModelResponse(
        text="",
        model="gpt-4",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=1000
      )

  def test_model_response_raises_error_for_empty_model(self):
    """Test that ModelResponse raises error for empty model."""
    # Act & Assert
    with pytest.raises(ValueError, match="Model cannot be empty"):
      ModelResponse(
        text="Response",
        model="",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=1000
      )

  def test_model_response_raises_error_for_empty_provider(self):
    """Test that ModelResponse raises error for empty provider."""
    # Act & Assert
    with pytest.raises(ValueError, match="Provider cannot be empty"):
      ModelResponse(
        text="Response",
        model="gpt-4",
        provider="",
        timestamp=datetime.now(),
        latency_ms=1000
      )

  def test_model_response_raises_error_for_negative_latency(self):
    """Test that ModelResponse raises error for negative latency."""
    # Act & Assert
    with pytest.raises(ValueError, match="Latency cannot be negative"):
      ModelResponse(
        text="Response",
        model="gpt-4",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=-1
      )

  def test_model_response_allows_zero_latency(self):
    """Test that ModelResponse allows zero latency."""
    # Arrange & Act
    response = ModelResponse(
      text="Response",
      model="gpt-4",
      provider="openai",
      timestamp=datetime.now(),
      latency_ms=0
    )

    # Assert
    assert response.latency_ms == 0

  def test_model_response_raises_error_for_negative_token_count(self):
    """Test that ModelResponse raises error for negative token_count."""
    # Act & Assert
    with pytest.raises(ValueError, match="Token count cannot be negative"):
      ModelResponse(
        text="Response",
        model="gpt-4",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=1000,
        token_count=-1
      )

  def test_model_response_allows_zero_token_count(self):
    """Test that ModelResponse allows zero token_count."""
    # Arrange & Act
    response = ModelResponse(
      text="Response",
      model="gpt-4",
      provider="openai",
      timestamp=datetime.now(),
      latency_ms=1000,
      token_count=0
    )

    # Assert
    assert response.token_count == 0

  def test_model_response_raises_error_for_negative_cost_estimate(self):
    """Test that ModelResponse raises error for negative cost_estimate."""
    # Act & Assert
    with pytest.raises(ValueError, match="Cost estimate cannot be negative"):
      ModelResponse(
        text="Response",
        model="gpt-4",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=1000,
        cost_estimate=-0.001
      )

  def test_model_response_allows_zero_cost_estimate(self):
    """Test that ModelResponse allows zero cost_estimate."""
    # Arrange & Act
    response = ModelResponse(
      text="Response",
      model="gpt-4",
      provider="openai",
      timestamp=datetime.now(),
      latency_ms=1000,
      cost_estimate=0.0
    )

    # Assert
    assert response.cost_estimate == 0.0

  def test_model_response_field_constraints(self):
    """Test various field constraint validations."""
    # Test valid edge cases
    timestamp = datetime.now()
    response = ModelResponse(
      text="x",  # Single character text
      model="m",  # Single character model
      provider="p",  # Single character provider
      timestamp=timestamp,
      latency_ms=1,  # Minimum positive latency
      token_count=1,  # Minimum positive token count
      cost_estimate=0.001  # Small positive cost
    )
    assert response.text == "x"
    assert response.model == "m"
    assert response.provider == "p"
    assert response.latency_ms == 1
    assert response.token_count == 1
    assert response.cost_estimate == 0.001


class TestDataModelSerialization:
  """Test cases for data model serialization and deserialization."""

  def test_provider_config_serialization(self):
    """Test that ProviderConfig can be serialized and deserialized."""
    # Arrange
    original = ProviderConfig(
      name="openai",
      api_key="sk-test123",
      base_url="https://api.openai.com/v1",
      timeout=60,
      max_retries=5,
      models=["gpt-4", "gpt-3.5-turbo"],
      default_params={"temperature": 0.7},
      provider_specific={"organization": "test-org"}
    )

    # Act - Convert to dict and back
    config_dict = {
      "name": original.name,
      "api_key": original.api_key,
      "base_url": original.base_url,
      "timeout": original.timeout,
      "max_retries": original.max_retries,
      "models": original.models,
      "default_params": original.default_params,
      "provider_specific": original.provider_specific
    }
    
    recreated = ProviderConfig(**config_dict)

    # Assert
    assert recreated.name == original.name
    assert recreated.api_key == original.api_key
    assert recreated.base_url == original.base_url
    assert recreated.timeout == original.timeout
    assert recreated.max_retries == original.max_retries
    assert recreated.models == original.models
    assert recreated.default_params == original.default_params
    assert recreated.provider_specific == original.provider_specific

  def test_model_request_serialization(self):
    """Test that ModelRequest can be serialized and deserialized."""
    # Arrange
    original = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.8,
      max_tokens=2000,
      system_prompt="You are a helpful assistant",
      stop_sequences=["END", "STOP"],
      provider_specific={"top_p": 0.9}
    )

    # Act - Convert to dict and back
    request_dict = {
      "prompt": original.prompt,
      "model": original.model,
      "temperature": original.temperature,
      "max_tokens": original.max_tokens,
      "system_prompt": original.system_prompt,
      "stop_sequences": original.stop_sequences,
      "provider_specific": original.provider_specific
    }
    
    recreated = ModelRequest(**request_dict)

    # Assert
    assert recreated.prompt == original.prompt
    assert recreated.model == original.model
    assert recreated.temperature == original.temperature
    assert recreated.max_tokens == original.max_tokens
    assert recreated.system_prompt == original.system_prompt
    assert recreated.stop_sequences == original.stop_sequences
    assert recreated.provider_specific == original.provider_specific

  def test_model_response_serialization(self):
    """Test that ModelResponse can be serialized and deserialized."""
    # Arrange
    timestamp = datetime.now()
    original = ModelResponse(
      text="Generated response text",
      model="gpt-4",
      provider="openai",
      timestamp=timestamp,
      latency_ms=1500,
      token_count=25,
      finish_reason="stop",
      cost_estimate=0.003,
      metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 15}}
    )

    # Act - Convert to dict and back
    response_dict = {
      "text": original.text,
      "model": original.model,
      "provider": original.provider,
      "timestamp": original.timestamp,
      "latency_ms": original.latency_ms,
      "token_count": original.token_count,
      "finish_reason": original.finish_reason,
      "cost_estimate": original.cost_estimate,
      "metadata": original.metadata
    }
    
    recreated = ModelResponse(**response_dict)

    # Assert
    assert recreated.text == original.text
    assert recreated.model == original.model
    assert recreated.provider == original.provider
    assert recreated.timestamp == original.timestamp
    assert recreated.latency_ms == original.latency_ms
    assert recreated.token_count == original.token_count
    assert recreated.finish_reason == original.finish_reason
    assert recreated.cost_estimate == original.cost_estimate
    assert recreated.metadata == original.metadata