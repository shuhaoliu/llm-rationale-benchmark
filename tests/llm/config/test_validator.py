"""Unit tests for ConfigValidator class."""

import os
import warnings
from unittest.mock import patch

import pytest

from rationale_benchmark.llm.config.models import LLMConfig, ProviderConfig
from rationale_benchmark.llm.config.validator import ConfigValidator
from rationale_benchmark.llm.exceptions import (
  ConfigurationError,
  StreamingNotSupportedError,
)


class TestConfigValidator:
  """Test cases for ConfigValidator class."""

  @pytest.fixture
  def validator(self):
    """Create ConfigValidator instance for testing."""
    return ConfigValidator()

  @pytest.fixture
  def valid_provider_config(self):
    """Create valid ProviderConfig for testing."""
    return ProviderConfig(
      name="test_provider",
      api_key="sk-test123456789012345678901234567890123456",
      base_url="https://api.test.com",
      timeout=30,
      max_retries=3,
      models=["model1", "model2"],
      default_params={"temperature": 0.7, "max_tokens": 1000},
      provider_specific={"custom_param": "value"}
    )

  @pytest.fixture
  def valid_llm_config(self, valid_provider_config):
    """Create valid LLMConfig for testing."""
    return LLMConfig(
      defaults={"timeout": 30, "temperature": 0.7},
      providers={"test_provider": valid_provider_config}
    )

  @pytest.fixture
  def config_with_streaming_params(self):
    """Create LLMConfig with streaming parameters for testing."""
    provider = ProviderConfig(
      name="streaming_provider",
      api_key="sk-test123456789012345678901234567890123456",
      default_params={"temperature": 0.7, "stream": True, "streaming": True},
      provider_specific={"stream_options": {"include_usage": True}}
    )
    return LLMConfig(
      defaults={"stream_mode": True, "timeout": 30},
      providers={"streaming_provider": provider}
    )

  def test_init_creates_validator_with_empty_state(self, validator):
    """Test that ConfigValidator initializes with empty state."""
    assert validator.validation_errors == []
    assert validator.warnings == []
    assert validator.streaming_params_found == []

  def test_validate_config_returns_empty_list_for_valid_config(self, validator, valid_llm_config):
    """Test that validate_config returns empty list for valid configuration."""
    errors = validator.validate_config(valid_llm_config)
    
    assert errors == []
    assert len(validator.validation_errors) == 0

  def test_validate_config_validates_top_level_structure(self, validator):
    """Test that validate_config validates top-level configuration structure."""
    # Test with invalid defaults type
    invalid_config = LLMConfig(
      defaults="not_a_dict",  # Should be dict
      providers={}
    )
    
    errors = validator.validate_config(invalid_config)
    
    assert len(errors) > 0
    assert any("'defaults' must be a dictionary" in error for error in errors)

  def test_validate_config_requires_at_least_one_provider(self, validator):
    """Test that validate_config requires at least one provider."""
    config = LLMConfig(
      defaults={},
      providers={}  # Empty providers
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("At least one provider must be configured" in error for error in errors)

  def test_validate_config_validates_provider_structure(self, validator):
    """Test that validate_config validates individual provider structure."""
    # Create provider with invalid fields by bypassing __post_init__ validation
    invalid_provider = ProviderConfig.__new__(ProviderConfig)
    invalid_provider.name = ""  # Empty name
    invalid_provider.api_key = ""  # Empty API key
    invalid_provider.base_url = None
    invalid_provider.timeout = -1  # Invalid timeout
    invalid_provider.max_retries = -1  # Invalid max_retries
    invalid_provider.models = "not_a_list"  # Should be list
    invalid_provider.default_params = "not_a_dict"  # Should be dict
    invalid_provider.provider_specific = "not_a_dict"  # Should be dict
    
    config = LLMConfig(
      defaults={},
      providers={"invalid": invalid_provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("name cannot be empty" in error for error in errors)
    assert any("must have an API key" in error for error in errors)
    assert any("timeout must be positive" in error for error in errors)
    assert any("max_retries cannot be negative" in error for error in errors)
    assert any("models must be a list" in error for error in errors)
    assert any("default_params must be a dictionary" in error for error in errors)
    assert any("provider_specific must be a dictionary" in error for error in errors)

  def test_validate_config_validates_parameter_ranges(self, validator):
    """Test that validate_config validates parameter value ranges."""
    provider = ProviderConfig(
      name="test",
      api_key="sk-test123456789012345678901234567890123456",
      default_params={
        "temperature": 3.0,  # Out of range (0.0-2.0)
        "max_tokens": -100,  # Out of range (1-100000)
        "top_p": 1.5,  # Out of range (0.0-1.0)
        "frequency_penalty": -3.0,  # Out of range (-2.0-2.0)
        "presence_penalty": 3.0,  # Out of range (-2.0-2.0)
      }
    )
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("temperature" in error and "between 0.0 and 2.0" in error for error in errors)
    assert any("max_tokens" in error and "between 1 and 100000" in error for error in errors)
    assert any("top_p" in error and "between 0.0 and 1.0" in error for error in errors)
    assert any("frequency_penalty" in error and "between -2.0 and 2.0" in error for error in errors)
    assert any("presence_penalty" in error and "between -2.0 and 2.0" in error for error in errors)

  def test_validate_config_validates_parameter_types(self, validator):
    """Test that validate_config validates parameter data types."""
    provider = ProviderConfig(
      name="test",
      api_key="sk-test123456789012345678901234567890123456",
      default_params={
        "temperature": "not_a_number",  # Should be float
        "max_tokens": "not_an_integer",  # Should be int
        "system_prompt": 123,  # Should be string
        "stop_sequences": "not_a_list",  # Should be list
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
    assert any("system_prompt" in error and "must be a string" in error for error in errors)
    assert any("stop_sequences" in error and "must be a list" in error for error in errors)

  def test_validate_config_validates_stop_sequences_content(self, validator):
    """Test that validate_config validates stop_sequences list content."""
    provider = ProviderConfig(
      name="test",
      api_key="sk-test123456789012345678901234567890123456",
      default_params={
        "stop_sequences": ["valid_string", 123, None]  # Mixed types
      }
    )
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("stop_sequences[1]" in error and "must be a string" in error for error in errors)
    assert any("stop_sequences[2]" in error and "must be a string" in error for error in errors)

  def test_validate_config_validates_base_url_format(self, validator):
    """Test that validate_config validates base_url format."""
    provider = ProviderConfig(
      name="test",
      api_key="sk-test123456789012345678901234567890123456",
      base_url="invalid-url"  # Should start with http:// or https://
    )
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("base_url must start with http://" in error for error in errors)

  def test_validate_config_validates_models_list_content(self, validator):
    """Test that validate_config validates models list content."""
    provider = ProviderConfig(
      name="test",
      api_key="sk-test123456789012345678901234567890123456",
      models=["valid_model", 123, "", "  "]  # Mixed types and empty strings
    )
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("model[1] must be a string" in error for error in errors)
    assert any("model[2] cannot be empty" in error for error in errors)
    assert any("model[3] cannot be empty" in error for error in errors)

  def test_validate_config_detects_streaming_parameters(self, validator, config_with_streaming_params):
    """Test that validate_config detects streaming parameters."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      errors = validator.validate_config(config_with_streaming_params)
    
    # Should detect streaming parameters
    assert len(validator.streaming_params_found) > 0
    assert "stream" in validator.streaming_params_found
    assert "streaming" in validator.streaming_params_found
    assert "stream_options" in validator.streaming_params_found
    assert "stream_mode" in validator.streaming_params_found
    
    # Should issue warnings
    assert len(w) > 0
    warning_message = str(w[0].message)
    assert "Streaming parameters detected" in warning_message

  def test_validate_config_validates_defaults_section(self, validator):
    """Test that validate_config validates defaults section parameters."""
    config = LLMConfig(
      defaults={
        "temperature": 3.0,  # Out of range
        "max_tokens": "not_an_integer",  # Wrong type
        "stream": True  # Streaming parameter
      },
      providers={"test": ProviderConfig(
        name="test",
        api_key="sk-test123456789012345678901234567890123456"
      )}
    )
    
    with warnings.catch_warnings(record=True):
      warnings.simplefilter("always")
      errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("temperature" in error and "between 0.0 and 2.0" in error for error in errors)
    assert any("max_tokens" in error and "must be an integer" in error for error in errors)
    assert "stream" in validator.streaming_params_found

  def test_validate_environment_variables_validates_present_variables(self, validator, valid_llm_config):
    """Test that validate_environment_variables validates present environment variables."""
    # Modify config to use environment variable
    valid_llm_config.providers["test_provider"].api_key = "${TEST_API_KEY}"
    
    with patch.dict(os.environ, {"TEST_API_KEY": "sk-test123456789012345678901234567890123456"}):
      errors = validator.validate_environment_variables(valid_llm_config)
    
    assert errors == []

  def test_validate_environment_variables_detects_missing_variables(self, validator, valid_llm_config):
    """Test that validate_environment_variables detects missing environment variables."""
    # Modify config to use missing environment variable
    valid_llm_config.providers["test_provider"].api_key = "${MISSING_API_KEY}"
    
    errors = validator.validate_environment_variables(valid_llm_config)
    
    assert len(errors) > 0
    assert any("Environment variable 'MISSING_API_KEY'" in error and "is not set" in error for error in errors)

  def test_validate_environment_variables_validates_api_key_format(self, validator, valid_llm_config):
    """Test that validate_environment_variables validates API key format."""
    # Modify config to use environment variable with invalid format
    valid_llm_config.providers["test_provider"].api_key = "${INVALID_API_KEY}"
    
    with patch.dict(os.environ, {"INVALID_API_KEY": "short"}):  # Use a clearly invalid key
      errors = validator.validate_environment_variables(valid_llm_config)
    
    assert len(errors) > 0
    assert any("appears to be too short" in error for error in errors)

  def test_validate_environment_variables_handles_direct_api_keys(self, validator, valid_llm_config):
    """Test that validate_environment_variables handles direct API keys."""
    # Config already has direct API key
    errors = validator.validate_environment_variables(valid_llm_config)
    
    assert errors == []

  def test_validate_environment_variables_detects_short_api_keys(self, validator, valid_llm_config):
    """Test that validate_environment_variables detects API keys that are too short."""
    valid_llm_config.providers["test_provider"].api_key = "short"
    
    errors = validator.validate_environment_variables(valid_llm_config)
    
    assert len(errors) > 0
    assert any("appears to be too short" in error for error in errors)

  def test_validate_environment_variables_detects_whitespace_in_keys(self, validator, valid_llm_config):
    """Test that validate_environment_variables detects whitespace in API keys."""
    valid_llm_config.providers["test_provider"].api_key = " sk-test123456789012345678901234567890123456 "
    
    errors = validator.validate_environment_variables(valid_llm_config)
    
    assert len(errors) > 0
    assert any("contains leading/trailing whitespace" in error for error in errors)

  def test_validate_environment_variables_validates_provider_specific_patterns(self, validator):
    """Test that validate_environment_variables validates provider-specific API key patterns."""
    # Test OpenAI pattern
    openai_provider = ProviderConfig(
      name="openai",
      api_key="${OPENAI_API_KEY}"
    )
    config = LLMConfig(defaults={}, providers={"openai": openai_provider})
    
    # Valid OpenAI key
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 48}):
      errors = validator.validate_environment_variables(config)
      assert errors == []
    
    # Invalid OpenAI key
    with patch.dict(os.environ, {"OPENAI_API_KEY": "invalid-openai-key"}):
      errors = validator.validate_environment_variables(config)
      assert len(errors) > 0
      assert any("format appears invalid for openai" in error for error in errors)

  def test_remove_streaming_parameters_removes_streaming_params(self, validator):
    """Test that remove_streaming_parameters removes streaming parameters."""
    config_dict = {
      "defaults": {
        "temperature": 0.7,
        "stream": True,  # Should be removed
        "streaming": True  # Should be removed
      },
      "providers": {
        "test": {
          "api_key": "test",
          "stream_options": {"include_usage": True},  # Should be removed
          "valid_param": "keep_this"
        }
      }
    }
    
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      cleaned = validator.remove_streaming_parameters(config_dict)
    
    # Check that streaming parameters were removed
    assert "stream" not in cleaned["defaults"]
    assert "streaming" not in cleaned["defaults"]
    assert "stream_options" not in cleaned["providers"]["test"]
    
    # Check that valid parameters were kept
    assert cleaned["defaults"]["temperature"] == 0.7
    assert cleaned["providers"]["test"]["valid_param"] == "keep_this"
    assert cleaned["providers"]["test"]["api_key"] == "test"
    
    # Check that warnings were issued
    assert len(w) >= 3  # At least 3 streaming parameters removed
    assert any("Removed streaming parameter" in str(warning.message) for warning in w)

  def test_remove_streaming_parameters_handles_nested_structures(self, validator):
    """Test that remove_streaming_parameters handles nested data structures."""
    config_dict = {
      "level1": {
        "level2": {
          "stream": True,  # Should be removed
          "valid_param": "keep"
        },
        "list_field": [
          {"stream_mode": True, "keep": "this"},  # stream_mode should be removed
          {"normal": "param"}
        ]
      }
    }
    
    with warnings.catch_warnings(record=True):
      warnings.simplefilter("always")
      cleaned = validator.remove_streaming_parameters(config_dict)
    
    assert "stream" not in cleaned["level1"]["level2"]
    assert cleaned["level1"]["level2"]["valid_param"] == "keep"
    assert "stream_mode" not in cleaned["level1"]["list_field"][0]
    assert cleaned["level1"]["list_field"][0]["keep"] == "this"
    assert cleaned["level1"]["list_field"][1]["normal"] == "param"

  def test_get_warnings_returns_copy_of_warnings(self, validator, config_with_streaming_params):
    """Test that get_warnings returns a copy of warnings list."""
    with warnings.catch_warnings(record=True):
      warnings.simplefilter("always")
      validator.validate_config(config_with_streaming_params)
    
    warnings_list = validator.get_warnings()
    original_length = len(warnings_list)
    
    # Modify returned list
    warnings_list.append("new warning")
    
    # Original should be unchanged
    assert len(validator.get_warnings()) == original_length

  def test_get_streaming_parameters_found_returns_copy(self, validator, config_with_streaming_params):
    """Test that get_streaming_parameters_found returns a copy of streaming parameters list."""
    with warnings.catch_warnings(record=True):
      warnings.simplefilter("always")
      validator.validate_config(config_with_streaming_params)
    
    streaming_params = validator.get_streaming_parameters_found()
    original_length = len(streaming_params)
    
    # Modify returned list
    streaming_params.append("new_param")
    
    # Original should be unchanged
    assert len(validator.get_streaming_parameters_found()) == original_length

  def test_validate_streaming_prevention_raises_error_for_streaming_params(self, validator):
    """Test that validate_streaming_prevention raises error for streaming parameters."""
    request_params = {
      "model": "gpt-4",
      "temperature": 0.7,
      "stream": True,  # Should trigger error
      "streaming": True  # Should trigger error
    }
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      validator.validate_streaming_prevention(request_params)
    
    assert "Streaming parameters are not supported" in str(exc_info.value)
    assert exc_info.value.blocked_params == ["stream", "streaming"]

  def test_validate_streaming_prevention_passes_for_valid_params(self, validator):
    """Test that validate_streaming_prevention passes for valid parameters."""
    request_params = {
      "model": "gpt-4",
      "temperature": 0.7,
      "max_tokens": 1000
    }
    
    # Should not raise any exception
    validator.validate_streaming_prevention(request_params)

  def test_validate_streaming_prevention_is_case_insensitive(self, validator):
    """Test that validate_streaming_prevention is case insensitive."""
    request_params = {
      "model": "gpt-4",
      "STREAM": True,  # Uppercase should still be detected
      "Stream_Options": {"include_usage": True}  # Mixed case
    }
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      validator.validate_streaming_prevention(request_params)
    
    assert len(exc_info.value.blocked_params) == 2

  def test_create_validation_report_creates_comprehensive_report(self, validator, valid_llm_config):
    """Test that create_validation_report creates comprehensive validation report."""
    report = validator.create_validation_report(valid_llm_config)
    
    assert isinstance(report, dict)
    assert "validation_passed" in report
    assert "total_errors" in report
    assert "configuration_errors" in report
    assert "environment_errors" in report
    assert "warnings" in report
    assert "streaming_parameters_found" in report
    assert "statistics" in report
    
    # Check statistics
    stats = report["statistics"]
    assert stats["total_providers"] == 1
    assert stats["total_models"] == 2
    assert "test_provider" in stats["providers_configured"]

  def test_create_validation_report_reports_errors(self, validator):
    """Test that create_validation_report reports validation errors."""
    # Create invalid config
    invalid_config = LLMConfig(
      defaults={},
      providers={}  # No providers
    )
    
    report = validator.create_validation_report(invalid_config)
    
    assert report["validation_passed"] is False
    assert report["total_errors"] > 0
    assert len(report["configuration_errors"]) > 0

  def test_create_validation_report_reports_environment_errors(self, validator, valid_llm_config):
    """Test that create_validation_report reports environment variable errors."""
    # Modify config to use missing environment variable
    valid_llm_config.providers["test_provider"].api_key = "${MISSING_KEY}"
    
    report = validator.create_validation_report(valid_llm_config)
    
    assert report["validation_passed"] is False
    assert report["total_errors"] > 0
    assert len(report["environment_errors"]) > 0

  def test_create_validation_report_reports_streaming_warnings(self, validator, config_with_streaming_params):
    """Test that create_validation_report reports streaming parameter warnings."""
    with warnings.catch_warnings(record=True):
      warnings.simplefilter("always")
      report = validator.create_validation_report(config_with_streaming_params)
    
    assert len(report["streaming_parameters_found"]) > 0
    assert len(report["warnings"]) > 0

  def test_validate_config_resets_state_between_calls(self, validator, valid_llm_config, config_with_streaming_params):
    """Test that validate_config resets state between calls."""
    # First validation with streaming config
    with warnings.catch_warnings(record=True):
      warnings.simplefilter("always")
      validator.validate_config(config_with_streaming_params)
    
    assert len(validator.streaming_params_found) > 0
    
    # Second validation with valid config should reset state
    validator.validate_config(valid_llm_config)
    
    assert len(validator.streaming_params_found) == 0
    assert len(validator.validation_errors) == 0
    assert len(validator.warnings) == 0

  def test_validate_config_handles_none_values_gracefully(self, validator):
    """Test that validate_config handles None values gracefully."""
    provider = ProviderConfig(
      name="test",
      api_key="sk-test123456789012345678901234567890123456",
      base_url=None,  # None is valid for base_url
      default_params={"system_prompt": None}  # None is valid for system_prompt
    )
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    # Should not have errors for None values in optional fields
    assert not any("base_url" in error for error in errors)
    assert not any("system_prompt" in error for error in errors)

  def test_validate_config_validates_integer_parameters_strictly(self, validator):
    """Test that validate_config validates integer parameters strictly."""
    provider = ProviderConfig(
      name="test",
      api_key="sk-test123456789012345678901234567890123456",
      default_params={
        "max_tokens": 1.5,  # Float instead of int
        "timeout": "30",  # String instead of int
      }
    )
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("max_tokens" in error and "must be an integer" in error for error in errors)
    assert any("timeout" in error and "must be an integer" in error for error in errors)

  def test_validate_config_validates_float_parameters_strictly(self, validator):
    """Test that validate_config validates float parameters strictly."""
    provider = ProviderConfig(
      name="test",
      api_key="sk-test123456789012345678901234567890123456",
      default_params={
        "temperature": "0.7",  # String instead of number
        "top_p": [0.9],  # List instead of number
      }
    )
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    assert len(errors) > 0
    assert any("temperature" in error and "must be a number" in error for error in errors)
    assert any("top_p" in error and "must be a number" in error for error in errors)

  def test_streaming_parameters_constant_contains_expected_values(self, validator):
    """Test that STREAMING_PARAMETERS constant contains expected streaming parameter names."""
    expected_params = {
      "stream", "streaming", "stream_options", "stream_usage",
      "stream_callback", "stream_handler", "incremental", "stream_mode"
    }
    
    assert validator.STREAMING_PARAMETERS == expected_params

  def test_supported_providers_constant_contains_expected_values(self, validator):
    """Test that SUPPORTED_PROVIDERS constant contains expected provider names."""
    expected_providers = {"openai", "anthropic", "gemini", "openrouter", "local"}
    
    assert validator.SUPPORTED_PROVIDERS == expected_providers

  def test_parameter_ranges_constant_contains_expected_ranges(self, validator):
    """Test that PARAMETER_RANGES constant contains expected parameter ranges."""
    expected_ranges = {
      "temperature": (0.0, 2.0),
      "max_tokens": (1, 100000),
      "timeout": (1, 300),
      "max_retries": (0, 10),
      "top_p": (0.0, 1.0),
      "frequency_penalty": (-2.0, 2.0),
      "presence_penalty": (-2.0, 2.0),
    }
    
    assert validator.PARAMETER_RANGES == expected_ranges

  def test_env_var_patterns_constant_contains_expected_patterns(self, validator):
    """Test that ENV_VAR_PATTERNS constant contains expected API key patterns."""
    expected_patterns = {
      "openai": r"^sk-[A-Za-z0-9]{48}$",
      "anthropic": r"^sk-ant-[A-Za-z0-9\-_]{95}$",
      "gemini": r"^[A-Za-z0-9\-_]{39}$",
      "openrouter": r"^sk-or-[A-Za-z0-9\-_]{48}$",
    }
    
    assert validator.ENV_VAR_PATTERNS == expected_patterns

  def test_validate_api_key_format_validates_openai_keys(self, validator):
    """Test that _validate_api_key_format validates OpenAI API key format."""
    # Valid OpenAI key
    error = validator._validate_api_key_format("openai", "api_key", "sk-" + "a" * 48)
    assert error is None
    
    # Invalid OpenAI key
    error = validator._validate_api_key_format("openai", "api_key", "invalid-key")
    assert error is not None
    assert "format appears invalid for openai" in error

  def test_validate_api_key_format_validates_anthropic_keys(self, validator):
    """Test that _validate_api_key_format validates Anthropic API key format."""
    # Valid Anthropic key
    valid_key = "sk-ant-" + "a" * 95
    error = validator._validate_api_key_format("anthropic", "api_key", valid_key)
    assert error is None
    
    # Invalid Anthropic key
    error = validator._validate_api_key_format("anthropic", "api_key", "sk-invalid")
    assert error is not None
    assert "format appears invalid for anthropic" in error

  def test_validate_api_key_format_validates_gemini_keys(self, validator):
    """Test that _validate_api_key_format validates Gemini API key format."""
    # Valid Gemini key
    valid_key = "a" * 39
    error = validator._validate_api_key_format("gemini", "api_key", valid_key)
    assert error is None
    
    # Invalid Gemini key (wrong format, but long enough to pass length check)
    error = validator._validate_api_key_format("gemini", "api_key", "invalid-key-that-is-long-enough-but-wrong-format")
    assert error is not None
    assert "format appears invalid for gemini" in error

  def test_validate_api_key_format_validates_openrouter_keys(self, validator):
    """Test that _validate_api_key_format validates OpenRouter API key format."""
    # Valid OpenRouter key
    valid_key = "sk-or-" + "a" * 48
    error = validator._validate_api_key_format("openrouter", "api_key", valid_key)
    assert error is None
    
    # Invalid OpenRouter key
    error = validator._validate_api_key_format("openrouter", "api_key", "sk-or-short")
    assert error is not None
    assert "format appears invalid for openrouter" in error

  def test_validate_api_key_format_handles_unknown_providers(self, validator):
    """Test that _validate_api_key_format handles unknown providers gracefully."""
    # Unknown provider should only do basic validation
    error = validator._validate_api_key_format("unknown", "api_key", "some-long-enough-key")
    assert error is None
    
    # But still catch obviously invalid keys
    error = validator._validate_api_key_format("unknown", "api_key", "short")
    assert error is not None
    assert "appears to be too short" in error

  def test_validate_api_key_format_handles_empty_and_none_keys(self, validator):
    """Test that _validate_api_key_format handles empty and None API keys."""
    # Empty string
    error = validator._validate_api_key_format("test", "api_key", "")
    assert error is not None
    assert "must be a non-empty string" in error
    
    # None value
    error = validator._validate_api_key_format("test", "api_key", None)
    assert error is not None
    assert "must be a non-empty string" in error

  def test_detect_streaming_parameters_finds_all_streaming_params(self, validator, config_with_streaming_params):
    """Test that _detect_streaming_parameters finds all streaming parameters."""
    validator._detect_streaming_parameters(config_with_streaming_params)
    
    found_params = set(validator.streaming_params_found)
    expected_params = {"stream_mode", "stream", "streaming", "stream_options"}
    
    assert expected_params.issubset(found_params)

  def test_detect_streaming_parameters_avoids_duplicates(self, validator):
    """Test that _detect_streaming_parameters avoids duplicate entries."""
    # Create config with duplicate streaming parameters
    provider = ProviderConfig(
      name="test",
      api_key="sk-test123456789012345678901234567890123456",
      default_params={"stream": True},
      provider_specific={"stream": False}  # Same parameter in different sections
    )
    config = LLMConfig(
      defaults={"stream": True},  # Same parameter again
      providers={"test": provider}
    )
    
    validator._detect_streaming_parameters(config)
    
    # Should only appear once despite being in multiple places
    assert validator.streaming_params_found.count("stream") == 1

  def test_validate_config_handles_malformed_provider_configs_gracefully(self, validator):
    """Test that validate_config handles malformed provider configs gracefully."""
    # Create config with provider that has wrong type for timeout by bypassing validation
    provider = ProviderConfig.__new__(ProviderConfig)
    provider.name = "test"
    provider.api_key = "sk-test123456789012345678901234567890123456"
    provider.base_url = None
    provider.timeout = "not_an_integer"  # Wrong type
    provider.max_retries = "not_an_integer"  # Wrong type
    provider.models = []
    provider.default_params = {}
    provider.provider_specific = {}
    
    config = LLMConfig(
      defaults={},
      providers={"test": provider}
    )
    
    errors = validator.validate_config(config)
    
    # Should have specific errors for type mismatches
    assert len(errors) > 0
    assert any("timeout must be an integer" in error for error in errors)
    assert any("max_retries must be an integer" in error for error in errors)

  def test_validate_config_provides_context_in_error_messages(self, validator):
    """Test that validate_config provides context in error messages."""
    provider = ProviderConfig(
      name="test_provider",
      api_key="sk-test123456789012345678901234567890123456",
      default_params={"temperature": 5.0}  # Out of range
    )
    
    config = LLMConfig(
      defaults={},
      providers={"test_provider": provider}
    )
    
    errors = validator.validate_config(config)
    
    # Error message should include context
    assert len(errors) > 0
    temp_error = next((error for error in errors if "temperature" in error), None)
    assert temp_error is not None
    assert "providers.test_provider.default_params" in temp_error

  def test_validate_environment_variables_handles_complex_env_var_references(self, validator):
    """Test that validate_environment_variables handles complex environment variable references."""
    provider = ProviderConfig(
      name="test",
      api_key="${API_KEY_PREFIX}_${API_KEY_SUFFIX}"  # Multiple env vars in one value
    )
    config = LLMConfig(defaults={}, providers={"test": provider})
    
    # Missing one of the variables
    with patch.dict(os.environ, {"API_KEY_PREFIX": "sk-test"}):
      errors = validator.validate_environment_variables(config)
      
      # Should detect that the resolved key is invalid (because suffix is missing)
      assert len(errors) > 0

  def test_remove_streaming_parameters_preserves_non_streaming_params(self, validator):
    """Test that remove_streaming_parameters preserves non-streaming parameters."""
    config_dict = {
      "temperature": 0.7,
      "max_tokens": 1000,
      "stream": True,  # Should be removed
      "model": "gpt-4",
      "streaming": True,  # Should be removed
      "timeout": 30
    }
    
    with warnings.catch_warnings(record=True):
      warnings.simplefilter("always")
      cleaned = validator.remove_streaming_parameters(config_dict)
    
    # Non-streaming parameters should be preserved
    assert cleaned["temperature"] == 0.7
    assert cleaned["max_tokens"] == 1000
    assert cleaned["model"] == "gpt-4"
    assert cleaned["timeout"] == 30
    
    # Streaming parameters should be removed
    assert "stream" not in cleaned
    assert "streaming" not in cleaned

  def test_validate_config_integration_with_all_validations(self, validator):
    """Test validate_config integration with all validation types."""
    # Create a config that has multiple types of issues by bypassing validation
    provider = ProviderConfig.__new__(ProviderConfig)
    provider.name = ""  # Empty name
    provider.api_key = "short"  # Too short
    provider.base_url = None
    provider.timeout = -1  # Invalid range
    provider.max_retries = 3
    provider.models = ["valid_model", ""]  # Mixed valid/invalid
    provider.default_params = {
      "temperature": 3.0,  # Out of range
      "stream": True,  # Streaming parameter
      "stop_sequences": ["valid", 123]  # Mixed types
    }
    provider.provider_specific = {}
    
    config = LLMConfig(
      defaults={"streaming": True},  # Streaming in defaults
      providers={"test": provider}
    )
    
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      errors = validator.validate_config(config)
    
    # Should have multiple types of errors
    assert len(errors) >= 5  # Multiple validation errors
    assert len(validator.streaming_params_found) > 0  # Streaming detection
    assert len(w) > 0  # Warnings issued
    
    # Check specific error types are present
    error_text = " ".join(errors)
    assert "name cannot be empty" in error_text
    assert "appears to be too short" in error_text or "timeout must be positive" in error_text
    assert "temperature" in error_text and "between 0.0 and 2.0" in error_text
    assert "stop_sequences" in error_text and "must be a string" in error_text