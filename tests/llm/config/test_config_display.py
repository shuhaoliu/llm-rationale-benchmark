"""Unit tests for configuration display functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rationale_benchmark.llm.config.loader import ConfigLoader
from rationale_benchmark.llm.exceptions import ConfigurationError


class TestConfigurationDisplay:
  """Test cases for configuration display functionality."""

  @pytest.fixture
  def temp_config_dir(self):
    """Create temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
      yield Path(temp_dir)

  @pytest.fixture
  def sample_multi_provider_config(self):
    """Sample configuration with multiple providers for testing."""
    return {
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
          "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
          "default_params": {
            "temperature": 0.8
          }
        },
        "anthropic": {
          "api_key": "${ANTHROPIC_API_KEY}",
          "base_url": "https://api.anthropic.com",
          "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
          "default_params": {
            "max_tokens": 2000
          }
        },
        "gemini": {
          "api_key": "${GEMINI_API_KEY}",
          "models": ["gemini-pro", "gemini-pro-vision"]
        }
      }
    }

  @pytest.fixture
  def sample_config_with_streaming_warnings(self):
    """Sample configuration with streaming parameters for warning tests."""
    return {
      "defaults": {
        "timeout": 30,
        "stream": True  # Should trigger warning
      },
      "providers": {
        "openai": {
          "api_key": "${OPENAI_API_KEY}",
          "models": ["gpt-4"],
          "default_params": {
            "temperature": 0.7,
            "streaming": True  # Should trigger warning
          },
          "provider_specific": {
            "stream_options": {"include_usage": True},  # Should trigger warning
            "valid_param": "valid_value"
          }
        }
      }
    }

  @pytest.fixture
  def config_loader(self, temp_config_dir):
    """Create ConfigLoader instance with temporary directory."""
    return ConfigLoader(temp_config_dir)

  def test_display_configuration_shows_provider_and_model_info(self, config_loader, temp_config_dir, sample_multi_provider_config):
    """Test that display_configuration shows detailed provider and model information."""
    config_path = temp_config_dir / "multi-provider.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(sample_multi_provider_config, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-openai-key',
      'ANTHROPIC_API_KEY': 'test-anthropic-key',
      'GEMINI_API_KEY': 'test-gemini-key'
    }):
      display_info = config_loader.display_configuration("multi-provider")

    # Check basic configuration info
    assert display_info["name"] == "multi-provider"
    assert display_info["is_valid"] is True
    assert display_info["total_providers"] == 3
    assert display_info["total_models"] == 8

    # Check provider details
    assert len(display_info["providers"]) == 3
    assert "openai" in display_info["providers"]
    assert "anthropic" in display_info["providers"]
    assert "gemini" in display_info["providers"]

    # Check OpenAI provider details
    openai_info = display_info["providers"]["openai"]
    assert openai_info["model_count"] == 3
    assert openai_info["models"] == ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
    assert openai_info["base_url"] == "https://api.openai.com/v1"
    assert openai_info["has_custom_params"] is True

    # Check Anthropic provider details
    anthropic_info = display_info["providers"]["anthropic"]
    assert anthropic_info["model_count"] == 3
    assert anthropic_info["models"] == ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]

    # Check Gemini provider details
    gemini_info = display_info["providers"]["gemini"]
    assert gemini_info["model_count"] == 2
    assert gemini_info["models"] == ["gemini-pro", "gemini-pro-vision"]

  def test_display_configuration_shows_validation_status(self, config_loader, temp_config_dir):
    """Test that display_configuration shows validation status for invalid configurations."""
    # Create invalid configuration (missing providers)
    invalid_config = {
      "defaults": {"timeout": 30},
      "providers": {}
    }
    
    config_path = temp_config_dir / "invalid.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(invalid_config, f, indent=2)

    display_info = config_loader.display_configuration("invalid")

    assert display_info["name"] == "invalid"
    assert display_info["is_valid"] is False
    assert display_info["error_message"] is not None
    assert "At least one provider must be configured" in display_info["error_message"]
    assert display_info["total_providers"] == 0
    assert display_info["total_models"] == 0

  def test_display_configuration_shows_streaming_warnings(self, config_loader, temp_config_dir, sample_config_with_streaming_warnings):
    """Test that display_configuration shows warnings for streaming-related configuration."""
    config_path = temp_config_dir / "streaming-warnings.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(sample_config_with_streaming_warnings, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      display_info = config_loader.display_configuration("streaming-warnings")

    assert display_info["name"] == "streaming-warnings"
    assert display_info["has_streaming_params"] is True
    assert len(display_info["streaming_params"]) >= 3
    assert "stream" in display_info["streaming_params"]
    assert "streaming" in display_info["streaming_params"]
    assert "stream_options" in display_info["streaming_params"]
    
    # Check that warnings are included
    assert len(display_info["warnings"]) > 0
    warning_text = " ".join(display_info["warnings"])
    assert "streaming" in warning_text.lower()

  def test_display_configuration_handles_missing_config(self, config_loader):
    """Test that display_configuration handles missing configuration files."""
    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.display_configuration("nonexistent")

    assert "Configuration file not found" in str(exc_info.value)

  def test_display_configuration_includes_defaults_info(self, config_loader, temp_config_dir, sample_multi_provider_config):
    """Test that display_configuration includes information about default parameters."""
    config_path = temp_config_dir / "with-defaults.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(sample_multi_provider_config, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key',
      'GEMINI_API_KEY': 'test-key'
    }):
      display_info = config_loader.display_configuration("with-defaults")

    # Check defaults information
    assert "defaults" in display_info
    defaults_info = display_info["defaults"]
    assert defaults_info["timeout"] == 30
    assert defaults_info["max_retries"] == 3
    assert defaults_info["temperature"] == 0.7
    assert defaults_info["max_tokens"] == 1000

  def test_display_all_configurations_returns_summary_for_all_configs(self, config_loader, temp_config_dir, sample_multi_provider_config):
    """Test that display_all_configurations returns summary information for all configurations."""
    # Create multiple configuration files
    config1_path = temp_config_dir / "config1.yaml"
    config2_path = temp_config_dir / "config2.yaml"
    
    # Modify config for variety
    config2_data = sample_multi_provider_config.copy()
    config2_data["providers"] = {
      "openai": sample_multi_provider_config["providers"]["openai"]
    }
    
    with open(config1_path, 'w') as f:
      yaml.dump(sample_multi_provider_config, f, indent=2)
    with open(config2_path, 'w') as f:
      yaml.dump(config2_data, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key',
      'GEMINI_API_KEY': 'test-key'
    }):
      all_configs = config_loader.display_all_configurations()

    assert len(all_configs) == 2
    
    # Find configs by name
    config1_info = next(c for c in all_configs if c["name"] == "config1")
    config2_info = next(c for c in all_configs if c["name"] == "config2")
    
    # Check config1 (multi-provider)
    assert config1_info["total_providers"] == 3
    assert config1_info["total_models"] == 8
    assert config1_info["is_valid"] is True
    
    # Check config2 (single provider)
    assert config2_info["total_providers"] == 1
    assert config2_info["total_models"] == 3
    assert config2_info["is_valid"] is True

  def test_display_all_configurations_includes_validation_status_for_each(self, config_loader, temp_config_dir, sample_multi_provider_config):
    """Test that display_all_configurations includes validation status for each configuration."""
    # Create valid and invalid configurations
    valid_config_path = temp_config_dir / "valid.yaml"
    invalid_config_path = temp_config_dir / "invalid.yaml"
    
    invalid_config = {"defaults": {}, "providers": {}}  # Invalid: no providers
    
    with open(valid_config_path, 'w') as f:
      yaml.dump(sample_multi_provider_config, f, indent=2)
    with open(invalid_config_path, 'w') as f:
      yaml.dump(invalid_config, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key',
      'GEMINI_API_KEY': 'test-key'
    }):
      all_configs = config_loader.display_all_configurations()

    assert len(all_configs) == 2
    
    # Find configs by name
    valid_info = next(c for c in all_configs if c["name"] == "valid")
    invalid_info = next(c for c in all_configs if c["name"] == "invalid")
    
    assert valid_info["is_valid"] is True
    assert valid_info["error_message"] is None
    
    assert invalid_info["is_valid"] is False
    assert invalid_info["error_message"] is not None

  def test_display_configuration_with_detailed_provider_info(self, config_loader, temp_config_dir):
    """Test that display_configuration includes detailed provider-specific information."""
    detailed_config = {
      "defaults": {"timeout": 30},
      "providers": {
        "openai": {
          "api_key": "${OPENAI_API_KEY}",
          "base_url": "https://custom.openai.com/v1",
          "timeout": 45,  # Override default
          "max_retries": 5,  # Override default
          "models": ["gpt-4", "gpt-3.5-turbo"],
          "default_params": {
            "temperature": 0.8,
            "max_tokens": 2000,
            "top_p": 0.9
          },
          "provider_specific": {
            "organization": "org-123",
            "custom_header": "value"
          }
        }
      }
    }
    
    config_path = temp_config_dir / "detailed.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(detailed_config, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      display_info = config_loader.display_configuration("detailed")

    openai_info = display_info["providers"]["openai"]
    
    # Check provider-specific settings
    assert openai_info["base_url"] == "https://custom.openai.com/v1"
    assert openai_info["timeout"] == 45
    assert openai_info["max_retries"] == 5
    
    # Check default parameters
    assert openai_info["default_params"]["temperature"] == 0.8
    assert openai_info["default_params"]["max_tokens"] == 2000
    assert openai_info["default_params"]["top_p"] == 0.9
    
    # Check provider-specific parameters
    assert openai_info["provider_specific"]["organization"] == "org-123"
    assert openai_info["provider_specific"]["custom_header"] == "value"

  def test_format_configuration_display_returns_human_readable_text(self, config_loader, temp_config_dir, sample_multi_provider_config):
    """Test that format_configuration_display returns human-readable text format."""
    config_path = temp_config_dir / "format-test.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(sample_multi_provider_config, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key',
      'GEMINI_API_KEY': 'test-key'
    }):
      formatted_text = config_loader.format_configuration_display("format-test")

    # Check that formatted text contains expected information
    assert "Configuration: format-test" in formatted_text
    assert "Status: Valid" in formatted_text
    assert "Total Providers: 3" in formatted_text
    assert "Total Models: 8" in formatted_text
    
    # Check provider sections
    assert "OpenAI Provider:" in formatted_text
    assert "Anthropic Provider:" in formatted_text
    assert "Gemini Provider:" in formatted_text
    
    # Check model listings
    assert "gpt-4" in formatted_text
    assert "claude-3-opus-20240229" in formatted_text
    assert "gemini-pro" in formatted_text

  def test_format_configuration_display_includes_warnings_for_streaming(self, config_loader, temp_config_dir, sample_config_with_streaming_warnings):
    """Test that format_configuration_display includes warnings for streaming parameters."""
    config_path = temp_config_dir / "streaming-format.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(sample_config_with_streaming_warnings, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      formatted_text = config_loader.format_configuration_display("streaming-format")

    # Check that warnings are included in formatted output
    assert "WARNINGS:" in formatted_text
    assert "streaming" in formatted_text.lower()
    assert "stream" in formatted_text.lower()

  def test_format_configuration_display_handles_invalid_config(self, config_loader, temp_config_dir):
    """Test that format_configuration_display handles invalid configurations gracefully."""
    invalid_config = {"defaults": {}, "providers": {}}
    
    config_path = temp_config_dir / "invalid-format.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(invalid_config, f, indent=2)

    formatted_text = config_loader.format_configuration_display("invalid-format")

    assert "Configuration: invalid-format" in formatted_text
    assert "Status: Invalid" in formatted_text
    assert "ERROR:" in formatted_text
    assert "At least one provider must be configured" in formatted_text

  def test_format_all_configurations_display_returns_summary_text(self, config_loader, temp_config_dir, sample_multi_provider_config):
    """Test that format_all_configurations_display returns formatted summary text."""
    # Create multiple configs
    config1_path = temp_config_dir / "summary1.yaml"
    config2_path = temp_config_dir / "summary2.yaml"
    
    with open(config1_path, 'w') as f:
      yaml.dump(sample_multi_provider_config, f, indent=2)
    with open(config2_path, 'w') as f:
      yaml.dump(sample_multi_provider_config, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key',
      'GEMINI_API_KEY': 'test-key'
    }):
      formatted_text = config_loader.format_all_configurations_display()

    # Check summary information
    assert "Configuration Summary" in formatted_text
    assert "Total Configurations: 2" in formatted_text
    assert "Valid Configurations: 2" in formatted_text
    assert "Invalid Configurations: 0" in formatted_text
    
    # Check individual config listings
    assert "summary1" in formatted_text
    assert "summary2" in formatted_text

  def test_display_configuration_handles_environment_variable_errors(self, config_loader, temp_config_dir, sample_multi_provider_config):
    """Test that display_configuration handles missing environment variables gracefully."""
    config_path = temp_config_dir / "env-error.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(sample_multi_provider_config, f, indent=2)

    # Don't set environment variables
    display_info = config_loader.display_configuration("env-error")

    assert display_info["name"] == "env-error"
    assert display_info["is_valid"] is False
    assert display_info["error_message"] is not None
    assert "Environment variable" in display_info["error_message"]

  def test_display_configuration_includes_file_path_info(self, config_loader, temp_config_dir, sample_multi_provider_config):
    """Test that display_configuration includes file path information."""
    config_path = temp_config_dir / "path-test.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(sample_multi_provider_config, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key',
      'GEMINI_API_KEY': 'test-key'
    }):
      display_info = config_loader.display_configuration("path-test")

    assert "file_path" in display_info
    assert display_info["file_path"].endswith("path-test.yaml")
    assert Path(display_info["file_path"]).exists()

  def test_display_configuration_with_empty_providers(self, config_loader, temp_config_dir):
    """Test display_configuration with configuration that has empty provider models."""
    config_with_empty_models = {
      "defaults": {"timeout": 30},
      "providers": {
        "openai": {
          "api_key": "${OPENAI_API_KEY}",
          "models": []  # Empty models list
        },
        "anthropic": {
          "api_key": "${ANTHROPIC_API_KEY}",
          "models": ["claude-3-opus-20240229"]
        }
      }
    }
    
    config_path = temp_config_dir / "empty-models.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(config_with_empty_models, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'ANTHROPIC_API_KEY': 'test-key'}):
      display_info = config_loader.display_configuration("empty-models")

    assert display_info["total_providers"] == 2
    assert display_info["total_models"] == 1  # Only anthropic has models
    
    openai_info = display_info["providers"]["openai"]
    anthropic_info = display_info["providers"]["anthropic"]
    
    assert openai_info["model_count"] == 0
    assert openai_info["models"] == []
    assert anthropic_info["model_count"] == 1
    assert anthropic_info["models"] == ["claude-3-opus-20240229"]

  def test_display_configuration_performance_with_large_config(self, config_loader, temp_config_dir):
    """Test display_configuration performance with large configuration files."""
    # Create a large configuration with many providers and models
    large_config = {
      "defaults": {"timeout": 30},
      "providers": {}
    }
    
    # Add many providers with many models each
    for i in range(10):
      provider_name = f"provider_{i}"
      large_config["providers"][provider_name] = {
        "api_key": f"${{PROVIDER_{i}_API_KEY}}",
        "models": [f"model_{i}_{j}" for j in range(20)]  # 20 models per provider
      }

    config_path = temp_config_dir / "large-config.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(large_config, f, indent=2)

    # Set environment variables
    env_vars = {f"PROVIDER_{i}_API_KEY": f"test-key-{i}" for i in range(10)}
    
    with patch.dict(os.environ, env_vars):
      display_info = config_loader.display_configuration("large-config")

    assert display_info["total_providers"] == 10
    assert display_info["total_models"] == 200  # 10 providers * 20 models each
    assert display_info["is_valid"] is True
    assert len(display_info["providers"]) == 10

  def test_display_configuration_with_malformed_yaml(self, config_loader, temp_config_dir):
    """Test display_configuration with malformed YAML files."""
    # Create malformed YAML
    malformed_path = temp_config_dir / "malformed-display.yaml"
    with open(malformed_path, 'w') as f:
      f.write("invalid: yaml: content: [\n")

    display_info = config_loader.display_configuration("malformed-display")

    assert display_info["name"] == "malformed-display"
    assert display_info["is_valid"] is False
    assert display_info["error_message"] is not None
    assert "Invalid YAML" in display_info["error_message"]
    assert display_info["total_providers"] == 0
    assert display_info["total_models"] == 0