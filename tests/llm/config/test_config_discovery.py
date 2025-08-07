"""Unit tests for configuration file discovery functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rationale_benchmark.llm.config.loader import ConfigLoader
from rationale_benchmark.llm.config.validator import ConfigValidator
from rationale_benchmark.llm.exceptions import ConfigurationError


class TestConfigurationDiscovery:
  """Test cases for configuration file discovery functionality."""

  @pytest.fixture
  def temp_config_dir(self):
    """Create temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
      yield Path(temp_dir)

  @pytest.fixture
  def sample_valid_config(self):
    """Sample valid configuration for testing."""
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
          "models": ["gpt-4", "gpt-3.5-turbo"]
        }
      }
    }

  @pytest.fixture
  def sample_invalid_config(self):
    """Sample invalid configuration for testing."""
    return {
      "defaults": {},
      "providers": {}  # Invalid: no providers configured
    }

  @pytest.fixture
  def sample_config_with_streaming(self):
    """Sample configuration with streaming parameters."""
    return {
      "defaults": {
        "timeout": 30,
        "stream": True  # Streaming parameter that should be detected
      },
      "providers": {
        "openai": {
          "api_key": "${OPENAI_API_KEY}",
          "models": ["gpt-4"],
          "default_params": {
            "temperature": 0.7,
            "streaming": True  # Another streaming parameter
          },
          "provider_specific": {
            "stream_options": {"include_usage": True}  # More streaming params
          }
        }
      }
    }

  @pytest.fixture
  def config_loader(self, temp_config_dir):
    """Create ConfigLoader instance with temporary directory."""
    return ConfigLoader(temp_config_dir)

  def test_discover_configuration_files_returns_empty_for_empty_directory(self, config_loader):
    """Test that discover_configuration_files returns empty list for empty directory."""
    configs = config_loader.discover_configuration_files()
    assert configs == []

  def test_discover_configuration_files_finds_yaml_files(self, config_loader, temp_config_dir, sample_valid_config):
    """Test that discover_configuration_files finds all YAML configuration files."""
    # Create test configuration files
    config1_path = temp_config_dir / "config1.yaml"
    config2_path = temp_config_dir / "config2.yaml"
    config3_path = temp_config_dir / "config3.yml"
    
    with open(config1_path, 'w') as f:
      yaml.dump(sample_valid_config, f, indent=2)
    with open(config2_path, 'w') as f:
      yaml.dump(sample_valid_config, f, indent=2)
    with open(config3_path, 'w') as f:
      yaml.dump(sample_valid_config, f, indent=2)
    
    # Create non-YAML file that should be ignored
    (temp_config_dir / "not-config.txt").touch()

    configs = config_loader.discover_configuration_files()
    
    assert len(configs) == 3
    config_names = [config["name"] for config in configs]
    assert sorted(config_names) == ["config1", "config2", "config3"]

  def test_discover_configuration_files_includes_validation_status(self, config_loader, temp_config_dir, sample_valid_config, sample_invalid_config):
    """Test that discover_configuration_files includes validation status for each config."""
    # Create valid config
    valid_config_path = temp_config_dir / "valid.yaml"
    with open(valid_config_path, 'w') as f:
      yaml.dump(sample_valid_config, f, indent=2)
    
    # Create invalid config
    invalid_config_path = temp_config_dir / "invalid.yaml"
    with open(invalid_config_path, 'w') as f:
      yaml.dump(sample_invalid_config, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      configs = config_loader.discover_configuration_files()

    assert len(configs) == 2
    
    # Find configs by name
    valid_config = next(c for c in configs if c["name"] == "valid")
    invalid_config = next(c for c in configs if c["name"] == "invalid")
    
    assert valid_config["is_valid"] is True
    assert valid_config["error_message"] is None
    
    assert invalid_config["is_valid"] is False
    assert invalid_config["error_message"] is not None
    assert "At least one provider must be configured" in invalid_config["error_message"]

  def test_discover_configuration_files_detects_streaming_parameters(self, config_loader, temp_config_dir, sample_config_with_streaming):
    """Test that discover_configuration_files detects streaming parameters in configurations."""
    # Create config with streaming parameters
    streaming_config_path = temp_config_dir / "streaming.yaml"
    with open(streaming_config_path, 'w') as f:
      yaml.dump(sample_config_with_streaming, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      configs = config_loader.discover_configuration_files()

    assert len(configs) == 1
    config = configs[0]
    
    assert config["name"] == "streaming"
    assert config["has_streaming_params"] is True
    assert len(config["streaming_params"]) > 0
    assert "stream" in config["streaming_params"]
    assert "streaming" in config["streaming_params"]
    assert "stream_options" in config["streaming_params"]

  def test_discover_configuration_files_includes_provider_and_model_info(self, config_loader, temp_config_dir, sample_valid_config):
    """Test that discover_configuration_files includes provider and model information."""
    # Create config with multiple providers
    multi_provider_config = {
      "defaults": {"timeout": 30},
      "providers": {
        "openai": {
          "api_key": "${OPENAI_API_KEY}",
          "models": ["gpt-4", "gpt-3.5-turbo"]
        },
        "anthropic": {
          "api_key": "${ANTHROPIC_API_KEY}",
          "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]
        }
      }
    }
    
    config_path = temp_config_dir / "multi.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(multi_provider_config, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'ANTHROPIC_API_KEY': 'test-key'}):
      configs = config_loader.discover_configuration_files()

    assert len(configs) == 1
    config = configs[0]
    
    assert config["name"] == "multi"
    assert len(config["providers"]) == 2
    assert "openai" in config["providers"]
    assert "anthropic" in config["providers"]
    
    assert config["providers"]["openai"]["models"] == ["gpt-4", "gpt-3.5-turbo"]
    assert config["providers"]["anthropic"]["models"] == ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]
    
    assert config["total_models"] == 4

  def test_discover_configuration_files_handles_malformed_yaml(self, config_loader, temp_config_dir):
    """Test that discover_configuration_files handles malformed YAML files gracefully."""
    # Create malformed YAML file
    malformed_path = temp_config_dir / "malformed.yaml"
    with open(malformed_path, 'w') as f:
      f.write("invalid: yaml: content: [\n")

    configs = config_loader.discover_configuration_files()
    
    assert len(configs) == 1
    config = configs[0]
    
    assert config["name"] == "malformed"
    assert config["is_valid"] is False
    assert config["error_message"] is not None
    assert "Invalid YAML" in config["error_message"]

  def test_discover_configuration_files_handles_missing_env_vars(self, config_loader, temp_config_dir, sample_valid_config):
    """Test that discover_configuration_files handles missing environment variables."""
    config_path = temp_config_dir / "missing-env.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(sample_valid_config, f, indent=2)

    # Explicitly unset OPENAI_API_KEY environment variable
    with patch.dict(os.environ, {}, clear=False):
      # Remove OPENAI_API_KEY if it exists
      if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
      
      configs = config_loader.discover_configuration_files()
    
    assert len(configs) == 1
    config = configs[0]
    
    assert config["name"] == "missing-env"
    assert config["is_valid"] is False
    assert config["error_message"] is not None
    assert "Environment variable 'OPENAI_API_KEY' is not set" in config["error_message"]

  def test_discover_configuration_files_raises_error_for_missing_directory(self):
    """Test that discover_configuration_files raises error for missing directory."""
    loader = ConfigLoader(Path("/nonexistent/directory"))
    
    with pytest.raises(ConfigurationError) as exc_info:
      loader.discover_configuration_files()
    
    assert "Configuration directory not found" in str(exc_info.value)

  def test_create_example_configuration_directory_creates_directory_and_files(self, temp_config_dir):
    """Test that create_example_configuration_directory creates directory with example files."""
    # Use a subdirectory that doesn't exist
    new_config_dir = temp_config_dir / "new_config"
    loader = ConfigLoader(new_config_dir)
    
    loader.create_example_configuration_directory()
    
    # Check that directory was created
    assert new_config_dir.exists()
    assert new_config_dir.is_dir()
    
    # Check that example files were created
    default_config = new_config_dir / "default-llms.yaml"
    example_config = new_config_dir / "example-config.yaml"
    
    assert default_config.exists()
    assert example_config.exists()
    
    # Verify example files contain valid YAML
    with open(default_config, 'r') as f:
      default_content = yaml.safe_load(f)
    with open(example_config, 'r') as f:
      example_content = yaml.safe_load(f)
    
    assert isinstance(default_content, dict)
    assert isinstance(example_content, dict)
    assert "providers" in default_content
    assert "providers" in example_content

  def test_create_example_configuration_directory_succeeds_if_directory_exists(self, config_loader):
    """Test that create_example_configuration_directory succeeds if directory already exists."""
    # Directory already exists from fixture
    config_loader.create_example_configuration_directory()
    # Should not raise an error

  def test_create_example_configuration_directory_does_not_overwrite_existing_files(self, config_loader, temp_config_dir):
    """Test that create_example_configuration_directory does not overwrite existing files."""
    # Create existing file with custom content
    existing_file = temp_config_dir / "default-llms.yaml"
    custom_content = {"custom": "content"}
    with open(existing_file, 'w') as f:
      yaml.dump(custom_content, f)
    
    config_loader.create_example_configuration_directory()
    
    # Verify original content is preserved
    with open(existing_file, 'r') as f:
      content = yaml.safe_load(f)
    
    assert content == custom_content

  def test_get_configuration_summary_returns_overview_info(self, config_loader, temp_config_dir, sample_valid_config):
    """Test that get_configuration_summary returns overview information."""
    # Create multiple config files
    config1_path = temp_config_dir / "config1.yaml"
    config2_path = temp_config_dir / "config2.yaml"
    
    with open(config1_path, 'w') as f:
      yaml.dump(sample_valid_config, f, indent=2)
    with open(config2_path, 'w') as f:
      yaml.dump(sample_valid_config, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      summary = config_loader.get_configuration_summary()

    assert summary["total_configurations"] == 2
    assert summary["valid_configurations"] == 2
    assert summary["invalid_configurations"] == 0
    assert summary["configurations_with_streaming"] == 0
    assert len(summary["all_providers"]) >= 1
    assert "openai" in summary["all_providers"]

  def test_get_configuration_summary_handles_mixed_validity(self, config_loader, temp_config_dir, sample_valid_config, sample_invalid_config, sample_config_with_streaming):
    """Test that get_configuration_summary handles mixed configuration validity."""
    # Create valid, invalid, and streaming configs
    valid_path = temp_config_dir / "valid.yaml"
    invalid_path = temp_config_dir / "invalid.yaml"
    streaming_path = temp_config_dir / "streaming.yaml"
    
    with open(valid_path, 'w') as f:
      yaml.dump(sample_valid_config, f, indent=2)
    with open(invalid_path, 'w') as f:
      yaml.dump(sample_invalid_config, f, indent=2)
    with open(streaming_path, 'w') as f:
      yaml.dump(sample_config_with_streaming, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      summary = config_loader.get_configuration_summary()

    assert summary["total_configurations"] == 3
    assert summary["valid_configurations"] == 2  # valid and streaming (streaming is still valid)
    assert summary["invalid_configurations"] == 1
    assert summary["configurations_with_streaming"] == 1

  def test_validate_all_configurations_returns_detailed_results(self, config_loader, temp_config_dir, sample_valid_config, sample_invalid_config):
    """Test that validate_all_configurations returns detailed validation results."""
    # Create valid and invalid configs
    valid_path = temp_config_dir / "valid.yaml"
    invalid_path = temp_config_dir / "invalid.yaml"
    
    with open(valid_path, 'w') as f:
      yaml.dump(sample_valid_config, f, indent=2)
    with open(invalid_path, 'w') as f:
      yaml.dump(sample_invalid_config, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      results = config_loader.validate_all_configurations()

    assert len(results) == 2
    
    # Find results by name
    valid_result = next(r for r in results if r["name"] == "valid")
    invalid_result = next(r for r in results if r["name"] == "invalid")
    
    # Check valid config result
    assert valid_result["is_valid"] is True
    assert valid_result["error_message"] is None
    assert valid_result["validation_details"]["validation_passed"] is True
    assert valid_result["validation_details"]["total_errors"] == 0
    
    # Check invalid config result
    assert invalid_result["is_valid"] is False
    assert invalid_result["error_message"] is not None
    assert invalid_result["validation_details"]["validation_passed"] is False
    assert invalid_result["validation_details"]["total_errors"] > 0

  def test_list_configurations_with_details_includes_comprehensive_info(self, config_loader, temp_config_dir, sample_valid_config):
    """Test that list_configurations_with_details includes comprehensive configuration information."""
    # Create config with detailed information
    detailed_config = {
      "defaults": {
        "timeout": 30,
        "temperature": 0.7
      },
      "providers": {
        "openai": {
          "api_key": "${OPENAI_API_KEY}",
          "base_url": "https://api.openai.com/v1",
          "models": ["gpt-4", "gpt-3.5-turbo"],
          "default_params": {
            "max_tokens": 1000
          }
        },
        "anthropic": {
          "api_key": "${ANTHROPIC_API_KEY}",
          "models": ["claude-3-opus-20240229"]
        }
      }
    }
    
    config_path = temp_config_dir / "detailed.yaml"
    with open(config_path, 'w') as f:
      yaml.dump(detailed_config, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'ANTHROPIC_API_KEY': 'test-key'}):
      configs = config_loader.list_configurations_with_details()

    assert len(configs) == 1
    config = configs[0]
    
    assert config["name"] == "detailed"
    assert config["file_path"].endswith("detailed.yaml")
    assert config["is_valid"] is True
    assert config["total_providers"] == 2
    assert config["total_models"] == 3
    assert len(config["providers"]) == 2
    
    # Check provider details
    openai_provider = config["providers"]["openai"]
    assert openai_provider["models"] == ["gpt-4", "gpt-3.5-turbo"]
    assert openai_provider["base_url"] == "https://api.openai.com/v1"
    
    anthropic_provider = config["providers"]["anthropic"]
    assert anthropic_provider["models"] == ["claude-3-opus-20240229"]

  def test_streaming_parameter_detection_across_all_configs(self, config_loader, temp_config_dir, sample_config_with_streaming):
    """Test streaming parameter detection across all configuration files."""
    # Create multiple configs with different streaming parameters
    streaming_config1 = {
      "defaults": {"stream": True},
      "providers": {
        "openai": {
          "api_key": "${OPENAI_API_KEY}",
          "models": ["gpt-4"]
        }
      }
    }
    
    streaming_config2 = {
      "defaults": {},
      "providers": {
        "anthropic": {
          "api_key": "${ANTHROPIC_API_KEY}",
          "models": ["claude-3-opus-20240229"],
          "provider_specific": {
            "streaming": True,
            "stream_options": {"include_usage": True}
          }
        }
      }
    }
    
    config1_path = temp_config_dir / "stream1.yaml"
    config2_path = temp_config_dir / "stream2.yaml"
    
    with open(config1_path, 'w') as f:
      yaml.dump(streaming_config1, f, indent=2)
    with open(config2_path, 'w') as f:
      yaml.dump(streaming_config2, f, indent=2)

    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'ANTHROPIC_API_KEY': 'test-key'}):
      configs = config_loader.discover_configuration_files()

    # Both configs should have streaming parameters detected
    stream1_config = next(c for c in configs if c["name"] == "stream1")
    stream2_config = next(c for c in configs if c["name"] == "stream2")
    
    assert stream1_config["has_streaming_params"] is True
    assert "stream" in stream1_config["streaming_params"]
    
    assert stream2_config["has_streaming_params"] is True
    assert "streaming" in stream2_config["streaming_params"]
    assert "stream_options" in stream2_config["streaming_params"]

  def test_configuration_directory_creation_with_permissions_error(self, temp_config_dir):
    """Test configuration directory creation handles permission errors."""
    # Create a path that would cause permission error
    restricted_path = temp_config_dir / "restricted"
    loader = ConfigLoader(restricted_path)
    
    # Mock mkdir to raise permission error
    with patch.object(Path, 'mkdir', side_effect=PermissionError("Permission denied")):
      with pytest.raises(ConfigurationError) as exc_info:
        loader.create_example_configuration_directory()
    
    assert "Failed to create configuration directory" in str(exc_info.value)
    assert "Permission denied" in str(exc_info.value)

  def test_discover_configuration_files_handles_file_read_errors(self, config_loader, temp_config_dir):
    """Test that discover_configuration_files handles file read errors gracefully."""
    # Create a config file
    config_path = temp_config_dir / "unreadable.yaml"
    config_path.write_text("defaults: {}\nproviders: {}")
    
    # Mock open to raise permission error for this specific file
    original_open = open
    def mock_open(file, *args, **kwargs):
      if str(file).endswith("unreadable.yaml"):
        raise PermissionError("Permission denied")
      return original_open(file, *args, **kwargs)
    
    with patch('builtins.open', side_effect=mock_open):
      configs = config_loader.discover_configuration_files()
    
    assert len(configs) == 1
    config = configs[0]
    
    assert config["name"] == "unreadable"
    assert config["is_valid"] is False
    assert config["error_message"] is not None
    assert "Permission denied" in config["error_message"]

  def test_configuration_summary_with_no_configurations(self, config_loader):
    """Test configuration summary when no configuration files exist."""
    summary = config_loader.get_configuration_summary()
    
    assert summary["total_configurations"] == 0
    assert summary["valid_configurations"] == 0
    assert summary["invalid_configurations"] == 0
    assert summary["configurations_with_streaming"] == 0
    assert summary["all_providers"] == []
    assert summary["total_models"] == 0