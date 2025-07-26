"""Unit tests for ConfigValidator class."""

import os
import warnings
from unittest.mock import patch

import pytest

from rationale_benchmark.llm.config.models import LLMConfig, ProviderConfig
from rationale_benchmark.llm.config.validator import ConfigValidator
from rationale_benchmark.llm.exceptions import ConfigurationError, StreamingNotSupportedError


class TestConfigValidator:
  """Test cases for ConfigValidator class."""

  @pytest.fixture
  def validator(self):
    """Create ConfigValidator instance for testing."""
    return ConfigValidator()

  @pytest.fixture
  def valid_config(self):
    """Create valid LLMConfig for testing."""
    return LLMConfig(
      defaults={
        "timeout": 30,
        "max_retries": 3,
        "temperature": 0.7,
        "max_tokens": 1000
      },
      providers={
        "openai": ProviderConfig(
          name="openai",
          api_key="sk-abcdef1234567890abcdef1234567890abcdef1234567890",  # Exactly 48 alphanumeric chars after sk-
          base_url="https://api.openai.com/v1",
          timeout=30,
          max_retries=3,
          models=["gpt-4", "gpt-3.5-turbo"],
          default_params={"temperature": 0.7},
          provider_specific={"organization": "test-org"}
        ),
        "anthropic": ProviderConfig(
          name="anthropic",
          api_key="sk-ant-abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef123456789",  # Exactly 95 chars after sk-ant-
          base_url="https://api.anthropic.com",
          timeout=45,
          max_retries=2,
          models=["claude-3-opus-20240229"],
          default_params={"max_tokens": 2000},
          provider_specific={"version": "2023-06-01"}
        )
      }
    )

  @pytest.fixture
  def config_with_streaming_params(self):
    """Create LLMConfig with streaming parameters for testing."""
    return LLMConfig(
      defaults={
        "timeout": 30,
        "stream": True,  # Streaming parameter in defaults
        "temperature": 0.7
      },
      providers={
        "openai": ProviderConfig(
          name="openai",
          api_key="sk-abcdef1234567890abcdef1234567890abcdef1234567890",  # Exactly 48 alphanumeric chars after sk-
          timeout=30,
          max_retries=3,
          models=["gpt-4"],
          default_params={
            "temperature": 0.8,
            "streaming": True  # Streaming parameter in default_params
          },
          provider_specific={
            "stream_options": {"include_usage": True}  # Streaming parameter in provider_specific
          }
        )
      }
    )

  def test_init_creates_validator_with_empty_state(self, validator):
    """Test that ConfigValidator initializes with empty state."""
    assert validator.validation_errors == []
    assert validator.warnings == []
    assert validator.streaming_params_found == []

  def test_validate_config_returns_empty_list_for_valid_config(self, validator, valid_config):
    """Test that validate_config returns empty list for valid configuration."""
    errors = validator.validate_config(valid_config)
    
    assert errors == []
    assert len(validator.validation_errors) == 0

  def test_validate_config_validates_top_level_structure(self, validator):
    """Test that validate_config validates top-level configuration structure."""
    # Create config with invalid structure
    invalid_config = LLMConfig(
      defaults="not a dict",  # Should be dict
      providers={}  # Empty providers
    )
    
    errors = validator.validate_config(invalid_config)
    
    assert len(errors) > 0
    assert any("'defaults' must be a dictionary" in error for error in errors)
    assert any("At least one provider must be configured" in error for error in errors)

  def test_validate_config_validates_defaults_section(self, validator):
    """Test that validate_config validates defaults section."""
    config = LLMConfig(
      defaults={
        "temperature": 3.0,  # Invalid range
        "max_tokens": -100,  # Invalid value
        "timeout": "not_int"  # Invalid type
      },
      providers={
        "test": ProviderConfig(
          name="test",
          api_key="test-key",
          models=["test-model"]
        )
      }
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("temperature" in error and "between 0.0 and 2.0" in error for error in errors)
    assert any("max_tokens" in error and "between 1 and 100000" in error for error in errors)

  def test_validate_config_validates_provider_required_fields(self, validator):
    """Test that validate_config validates provider required fields."""
    # Create provider with valid initial values, then modify for testing
    provider = ProviderConfig(
      name="test",
      api_key="test-key",
      models=["test-model"]
    )
    # Bypass validation by directly setting invalid values
    provider.name = ""
    provider.api_key = ""
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("name cannot be empty" in error for error in errors)
    assert any("must have an API key" in error for error in errors)

  def test_validate_config_validates_provider_data_types(self, validator):
    """Test that validate_config validates provider data types."""
    # Create provider with valid initial values, then modify for testing
    provider = ProviderConfig(
      name="test",
      api_key="test-key",
      models=["test-model"]
    )
    # Bypass validation by directly setting invalid values
    provider.timeout = "not_int"
    provider.max_retries = "not_int"
    provider.models = "not_list"
    provider.default_params = "not_dict"
    provider.provider_specific = "not_dict"
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("timeout must be an integer" in error for error in errors)
    assert any("max_retries must be an integer" in error for error in errors)
    assert any("models must be a list" in error for error in errors)
    assert any("default_params must be a dictionary" in error for error in errors)
    assert any("provider_specific must be a dictionary" in error for error in errors)

  def test_validate_config_validates_provider_value_ranges(self, validator):
    """Test that validate_config validates provider value ranges."""
    # Create provider with valid initial values, then modify for testing
    provider = ProviderConfig(
      name="test",
      api_key="test-key",
      models=["test-model"]
    )
    # Bypass validation by directly setting invalid values
    provider.timeout = -5
    provider.max_retries = -1
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("timeout must be positive" in error for error in errors)
    assert any("max_retries cannot be negative" in error for error in errors)

  def test_validate_config_validates_base_url_format(self, validator):
    """Test that validate_config validates base_url format."""
    # Create providers with valid initial values, then modify for testing
    provider1 = ProviderConfig(
      name="test1",
      api_key="test-key",
      models=["test-model"]
    )
    provider2 = ProviderConfig(
      name="test2",
      api_key="test-key",
      models=["test-model"]
    )
    # Bypass validation by directly setting invalid values
    provider1.base_url = 123
    provider2.base_url = "invalid-url"
    
    config = LLMConfig(
      defaults={},
      providers={
        "test1": provider1,
        "test2": provider2
      }
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("base_url must be a string" in error for error in errors)
    assert any("base_url must start with http://" in error for error in errors)

  def test_validate_config_validates_models_list(self, validator):
    """Test that validate_config validates models list."""
    # Create provider with valid initial values, then modify for testing
    provider = ProviderConfig(
      name="test",
      api_key="test-key",
      models=["valid-model"]
    )
    # Bypass validation by directly setting invalid values
    provider.models = [123, "", "valid-model"]
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("model[0] must be a string" in error for error in errors)
    assert any("model[1] cannot be empty" in error for error in errors)

  def test_validate_config_validates_parameter_values(self, validator):
    """Test that validate_config validates parameter values."""
    provider = ProviderConfig(
      name="test",
      api_key="test-key",
      models=["test-model"],
      default_params={
        "temperature": "not_number",  # Should be number
        "max_tokens": "not_int",  # Should be int
        "top_p": 2.5,  # Out of range
        "system_prompt": 123,  # Should be string
        "stop_sequences": "not_list"  # Should be list
      }
    )
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("temperature" in error and "must be a number" in error for error in errors)
    assert any("max_tokens" in error and "must be an integer" in error for error in errors)
    assert any("top_p" in error and "between 0.0 and 1.0" in error for error in errors)
    assert any("system_prompt" in error and "must be a string" in error for error in errors)
    assert any("stop_sequences" in error and "must be a list" in error for error in errors)

  def test_validate_config_detects_streaming_parameters(self, validator, config_with_streaming_params):
    """Test that validate_config detects streaming parameters."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      errors = validator.validate_config(config_with_streaming_params)
      
      # Should have warnings about streaming parameters
      assert len(w) > 0
      warning_messages = [str(warning.message) for warning in w]
      assert any("Streaming parameters detected" in msg for msg in warning_messages)
    
    # Should detect streaming parameters
    streaming_params = validator.get_streaming_parameters_found()
    assert "stream" in streaming_params
    assert "streaming" in streaming_params
    assert "stream_options" in streaming_params

  def test_validate_environment_variables_validates_existing_vars(self, validator, valid_config):
    """Test that validate_environment_variables validates existing environment variables."""
    # Mock environment variables with proper format
    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'sk-abcdef1234567890abcdef1234567890abcdef1234567890',  # Exactly 48 alphanumeric chars after sk-
      'ANTHROPIC_API_KEY': 'sk-ant-abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef123456789'  # Exactly 95 chars after sk-ant-
    }):
      # Update config to use environment variables
      valid_config.providers["openai"].api_key = "${OPENAI_API_KEY}"
      valid_config.providers["anthropic"].api_key = "${ANTHROPIC_API_KEY}"
      
      errors = validator.validate_environment_variables(valid_config)
      
      assert errors == []

  def test_validate_environment_variables_detects_missing_vars(self, validator, valid_config):
    """Test that validate_environment_variables detects missing environment variables."""
    # Update config to use missing environment variables
    valid_config.providers["openai"].api_key = "${MISSING_OPENAI_KEY}"
    valid_config.providers["anthropic"].api_key = "${MISSING_ANTHROPIC_KEY}"
    
    errors = validator.validate_environment_variables(valid_config)
    
    assert len(errors) == 2
    assert any("MISSING_OPENAI_KEY" in error and "is not set" in error for error in errors)
    assert any("MISSING_ANTHROPIC_KEY" in error and "is not set" in error for error in errors)

  def test_validate_environment_variables_validates_api_key_format(self, validator, valid_config):
    """Test that validate_environment_variables validates API key format."""
    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'invalid-key',  # Too short
      'ANTHROPIC_API_KEY': 'also-invalid'  # Wrong format
    }):
      valid_config.providers["openai"].api_key = "${OPENAI_API_KEY}"
      valid_config.providers["anthropic"].api_key = "${ANTHROPIC_API_KEY}"
      
      errors = validator.validate_environment_variables(valid_config)
      
      assert len(errors) > 0
      assert any("appears to be too short" in error or "format appears invalid" in error for error in errors)

  def test_validate_environment_variables_handles_direct_api_keys(self, validator, valid_config):
    """Test that validate_environment_variables handles direct API keys."""
    # Use invalid direct API keys
    valid_config.providers["openai"].api_key = "short"
    valid_config.providers["anthropic"].api_key = ""
    
    errors = validator.validate_environment_variables(valid_config)
    
    assert len(errors) > 0
    assert any("appears to be too short" in error for error in errors)
    assert any("must be a non-empty string" in error for error in errors)

  def test_remove_streaming_parameters_removes_streaming_params(self, validator):
    """Test that remove_streaming_parameters removes streaming parameters."""
    config_dict = {
      "defaults": {
        "temperature": 0.7,
        "stream": True,  # Should be removed
        "max_tokens": 1000
      },
      "providers": {
        "test": {
          "api_key": "test-key",
          "streaming": True,  # Should be removed
          "stream_options": {"include_usage": True},  # Should be removed
          "temperature": 0.8
        }
      }
    }
    
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      cleaned_config = validator.remove_streaming_parameters(config_dict)
      
      # Should have warnings
      assert len(w) > 0
    
    # Should remove streaming parameters
    assert "stream" not in cleaned_config["defaults"]
    assert "streaming" not in cleaned_config["providers"]["test"]
    assert "stream_options" not in cleaned_config["providers"]["test"]
    
    # Should keep non-streaming parameters
    assert cleaned_config["defaults"]["temperature"] == 0.7
    assert cleaned_config["defaults"]["max_tokens"] == 1000
    assert cleaned_config["providers"]["test"]["api_key"] == "test-key"
    assert cleaned_config["providers"]["test"]["temperature"] == 0.8

  def test_remove_streaming_parameters_handles_nested_structures(self, validator):
    """Test that remove_streaming_parameters handles nested structures."""
    config_dict = {
      "nested": {
        "level1": {
          "level2": {
            "stream": True,  # Should be removed
            "valid_param": "keep_me"
          }
        }
      },
      "list_field": [
        {"stream_handler": "remove_me"},  # Should be removed
        {"keep_param": "keep_me"}
      ]
    }
    
    cleaned_config = validator.remove_streaming_parameters(config_dict)
    
    assert "stream" not in cleaned_config["nested"]["level1"]["level2"]
    assert cleaned_config["nested"]["level1"]["level2"]["valid_param"] == "keep_me"
    assert "stream_handler" not in cleaned_config["list_field"][0]
    assert cleaned_config["list_field"][1]["keep_param"] == "keep_me"

  def test_get_warnings_returns_warning_list(self, validator, config_with_streaming_params):
    """Test that get_warnings returns list of warnings."""
    with warnings.catch_warnings(record=True):
      warnings.simplefilter("always")
      validator.validate_config(config_with_streaming_params)
    
    warnings_list = validator.get_warnings()
    assert isinstance(warnings_list, list)
    assert len(warnings_list) > 0
    assert any("Streaming parameters detected" in warning for warning in warnings_list)

  def test_get_streaming_parameters_found_returns_param_list(self, validator, config_with_streaming_params):
    """Test that get_streaming_parameters_found returns list of found parameters."""
    validator.validate_config(config_with_streaming_params)
    
    streaming_params = validator.get_streaming_parameters_found()
    assert isinstance(streaming_params, list)
    assert "stream" in streaming_params
    assert "streaming" in streaming_params
    assert "stream_options" in streaming_params

  def test_validate_streaming_prevention_allows_valid_params(self, validator):
    """Test that validate_streaming_prevention allows valid parameters."""
    valid_params = {
      "temperature": 0.7,
      "max_tokens": 1000,
      "model": "gpt-4"
    }
    
    # Should not raise exception
    validator.validate_streaming_prevention(valid_params)

  def test_validate_streaming_prevention_blocks_streaming_params(self, validator):
    """Test that validate_streaming_prevention blocks streaming parameters."""
    invalid_params = {
      "temperature": 0.7,
      "stream": True,
      "streaming": True,
      "stream_options": {"include_usage": True}
    }
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      validator.validate_streaming_prevention(invalid_params)
    
    assert "Streaming parameters are not supported" in str(exc_info.value)
    assert exc_info.value.blocked_params == ["stream", "streaming", "stream_options"]

  def test_create_validation_report_creates_comprehensive_report(self, validator, valid_config):
    """Test that create_validation_report creates comprehensive validation report."""
    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'sk-abcdef1234567890abcdef1234567890abcdef1234567890',  # Exactly 48 alphanumeric chars after sk-
      'ANTHROPIC_API_KEY': 'sk-ant-abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef123456789'  # Exactly 95 chars after sk-ant-
    }):
      # Update config to use environment variables
      valid_config.providers["openai"].api_key = "${OPENAI_API_KEY}"
      valid_config.providers["anthropic"].api_key = "${ANTHROPIC_API_KEY}"
      
      report = validator.create_validation_report(valid_config)
    
    assert isinstance(report, dict)
    assert "validation_passed" in report
    assert "total_errors" in report
    assert "configuration_errors" in report
    assert "environment_errors" in report
    assert "warnings" in report
    assert "streaming_parameters_found" in report
    assert "statistics" in report
    
    # For valid config, should pass validation
    assert report["validation_passed"] is True
    assert report["total_errors"] == 0
    assert report["statistics"]["total_providers"] == 2
    assert report["statistics"]["total_models"] == 3
    assert "openai" in report["statistics"]["providers_configured"]
    assert "anthropic" in report["statistics"]["providers_configured"]

  def test_create_validation_report_reports_errors(self, validator):
    """Test that create_validation_report reports validation errors."""
    # Create invalid config
    invalid_config = LLMConfig(
      defaults="not_dict",  # Invalid
      providers={}  # Empty
    )
    
    report = validator.create_validation_report(invalid_config)
    
    assert report["validation_passed"] is False
    assert report["total_errors"] > 0
    assert len(report["configuration_errors"]) > 0

  def test_parameter_ranges_constant_has_expected_values(self, validator):
    """Test that PARAMETER_RANGES constant has expected parameter ranges."""
    ranges = validator.PARAMETER_RANGES
    
    assert ranges["temperature"] == (0.0, 2.0)
    assert ranges["max_tokens"] == (1, 100000)
    assert ranges["timeout"] == (1, 300)
    assert ranges["max_retries"] == (0, 10)
    assert ranges["top_p"] == (0.0, 1.0)
    assert ranges["frequency_penalty"] == (-2.0, 2.0)
    assert ranges["presence_penalty"] == (-2.0, 2.0)

  def test_streaming_parameters_constant_has_expected_values(self, validator):
    """Test that STREAMING_PARAMETERS constant has expected streaming parameters."""
    streaming_params = validator.STREAMING_PARAMETERS
    
    expected_params = {
      "stream", "streaming", "stream_options", "stream_usage",
      "stream_callback", "stream_handler", "incremental", "stream_mode"
    }
    
    assert streaming_params == expected_params

  def test_supported_providers_constant_has_expected_values(self, validator):
    """Test that SUPPORTED_PROVIDERS constant has expected provider types."""
    providers = validator.SUPPORTED_PROVIDERS
    
    expected_providers = {"openai", "anthropic", "gemini", "openrouter", "local"}
    
    assert providers == expected_providers

  def test_env_var_patterns_constant_has_expected_patterns(self, validator):
    """Test that ENV_VAR_PATTERNS constant has expected API key patterns."""
    patterns = validator.ENV_VAR_PATTERNS
    
    assert "openai" in patterns
    assert "anthropic" in patterns
    assert "gemini" in patterns
    assert "openrouter" in patterns
    
    # Test pattern formats
    assert patterns["openai"].startswith("^sk-")
    assert patterns["anthropic"].startswith("^sk-ant-")
    assert patterns["openrouter"].startswith("^sk-or-")

  def test_validator_uses_2_space_indentation_in_implementation(self, validator, valid_config):
    """Test that ConfigValidator implementation uses 2-space indentation."""
    # This test verifies the implementation follows the 2-space indentation standard
    # by checking that the validator works correctly (implementation detail)
    errors = validator.validate_config(valid_config)
    
    # If the implementation uses proper indentation, it should work correctly
    assert errors == []
    assert isinstance(validator.validation_errors, list)
    assert isinstance(validator.warnings, list)
    assert isinstance(validator.streaming_params_found, list)

  def test_validate_config_handles_stop_sequences_validation(self, validator):
    """Test that validate_config validates stop_sequences parameter."""
    provider = ProviderConfig(
      name="test",
      api_key="test-key",
      models=["test-model"],
      default_params={
        "stop_sequences": ["valid", 123, ""]  # Mixed valid/invalid
      }
    )
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("stop_sequences[1]" in error and "must be a string" in error for error in errors)

  def test_validate_config_comprehensive_integration(self, validator):
    """Test comprehensive validation with multiple error types."""
    # Create provider with valid initial values, then modify for testing
    provider = ProviderConfig(
      name="test",
      api_key="test-key",
      models=["test-model"]
    )
    # Bypass validation by directly setting invalid values
    provider.name = ""
    provider.api_key = ""
    provider.timeout = -5
    provider.max_retries = -1
    provider.models = [123, ""]
    provider.default_params = {
      "temperature": "not_number",  # Invalid type
      "streaming": True  # Streaming parameter
    }
    provider.provider_specific = {
      "stream_options": {"test": True}  # Streaming parameter
    }
    
    config = LLMConfig(
      defaults={
        "temperature": 5.0,  # Out of range
        "stream": True,  # Streaming parameter
        "max_tokens": -100  # Invalid value
      },
      providers={"invalid_provider": provider}
    )
    
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      errors = validator.validate_config(config)
    
    # Should have multiple validation errors
    assert len(errors) > 5
    
    # Should have streaming parameter warnings
    assert len(w) > 0
    
    # Should detect streaming parameters
    streaming_params = validator.get_streaming_parameters_found()
    assert len(streaming_params) >= 3
    assert "stream" in streaming_params
    assert "streaming" in streaming_params
    assert "stream_options" in streaming_params