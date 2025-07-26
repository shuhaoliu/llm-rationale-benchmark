"""Unit tests for ConfigLoader class."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rationale_benchmark.llm.config.loader import ConfigLoader
from rationale_benchmark.llm.config.models import LLMConfig
from rationale_benchmark.llm.exceptions import ConfigurationError


class TestConfigLoader:
  """Test cases for ConfigLoader class."""

  @pytest.fixture
  def temp_config_dir(self):
    """Create temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
      yield Path(temp_dir)

  @pytest.fixture
  def sample_config_dict(self):
    """Sample configuration dictionary for testing."""
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
        },
        "anthropic": {
          "api_key": "${ANTHROPIC_API_KEY}",
          "base_url": "https://api.anthropic.com",
          "models": ["claude-3-opus-20240229"]
        }
      }
    }

  @pytest.fixture
  def config_loader(self, temp_config_dir):
    """Create ConfigLoader instance with temporary directory."""
    return ConfigLoader(temp_config_dir)

  def test_init_creates_config_loader_with_directory(self, temp_config_dir):
    """Test that ConfigLoader initializes with correct directory."""
    loader = ConfigLoader(temp_config_dir)
    assert loader.config_dir == temp_config_dir

  def test_load_config_loads_valid_yaml_file(self, config_loader, temp_config_dir, sample_config_dict):
    """Test that load_config successfully loads a valid YAML configuration file."""
    # Create test config file
    config_file = temp_config_dir / "test-config.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(sample_config_dict, f, indent=2)

    # Mock environment variables
    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-openai-key',
      'ANTHROPIC_API_KEY': 'test-anthropic-key'
    }):
      config = config_loader.load_config("test-config")

    assert isinstance(config, LLMConfig)
    assert config.defaults["timeout"] == 30
    assert "openai" in config.providers
    assert "anthropic" in config.providers
    assert config.providers["openai"].api_key == "test-openai-key"
    assert config.providers["anthropic"].api_key == "test-anthropic-key"

  def test_load_config_uses_default_config_name(self, config_loader, temp_config_dir, sample_config_dict):
    """Test that load_config uses 'default-llms' as default config name."""
    # Create default config file
    config_file = temp_config_dir / "default-llms.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(sample_config_dict, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key'
    }):
      config = config_loader.load_config()

    assert isinstance(config, LLMConfig)
    assert "openai" in config.providers

  def test_load_config_raises_error_for_missing_file(self, config_loader):
    """Test that load_config raises ConfigurationError for missing file."""
    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_config("nonexistent")

    assert "Configuration file not found" in str(exc_info.value)
    assert "nonexistent.yaml" in str(exc_info.value)
    assert exc_info.value.config_file is not None

  def test_load_config_raises_error_for_invalid_yaml(self, config_loader, temp_config_dir):
    """Test that load_config raises ConfigurationError for invalid YAML."""
    # Create invalid YAML file
    config_file = temp_config_dir / "invalid.yaml"
    with open(config_file, 'w') as f:
      f.write("invalid: yaml: content: [\n")

    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_config("invalid")

    assert "Invalid YAML" in str(exc_info.value)
    assert exc_info.value.config_file is not None

  def test_load_config_raises_error_for_empty_file(self, config_loader, temp_config_dir):
    """Test that load_config raises ConfigurationError for empty file."""
    # Create empty file
    config_file = temp_config_dir / "empty.yaml"
    config_file.touch()

    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_config("empty")

    assert "Configuration file is empty" in str(exc_info.value)

  def test_load_config_raises_error_for_non_dict_yaml(self, config_loader, temp_config_dir):
    """Test that load_config raises ConfigurationError for non-dictionary YAML."""
    # Create YAML file with list instead of dict
    config_file = temp_config_dir / "list.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(["item1", "item2"], f)

    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_config("list")

    assert "must contain a YAML dictionary" in str(exc_info.value)

  def test_load_config_resolves_environment_variables(self, config_loader, temp_config_dir):
    """Test that load_config resolves environment variables in ${VAR} format."""
    config_dict = {
      "defaults": {"timeout": 30},
      "providers": {
        "test": {
          "api_key": "${TEST_API_KEY}",
          "base_url": "https://${TEST_HOST}/api",
          "custom_field": "prefix-${TEST_VALUE}-suffix"
        }
      }
    }

    config_file = temp_config_dir / "env-test.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(config_dict, f, indent=2)

    with patch.dict(os.environ, {
      'TEST_API_KEY': 'secret-key',
      'TEST_HOST': 'example.com',
      'TEST_VALUE': 'middle'
    }):
      config = config_loader.load_config("env-test")

    assert config.providers["test"].api_key == "secret-key"
    assert config.providers["test"].base_url == "https://example.com/api"
    assert config.providers["test"].default_params["custom_field"] == "prefix-middle-suffix"

  def test_load_config_raises_error_for_missing_env_var(self, config_loader, temp_config_dir):
    """Test that load_config raises ConfigurationError for missing environment variables."""
    config_dict = {
      "defaults": {},
      "providers": {
        "test": {
          "api_key": "${MISSING_API_KEY}"
        }
      }
    }

    config_file = temp_config_dir / "missing-env.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(config_dict, f, indent=2)

    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_config("missing-env")

    assert "Environment variable 'MISSING_API_KEY' is not set" in str(exc_info.value)
    assert exc_info.value.config_file is not None

  def test_list_available_configs_returns_yaml_files(self, config_loader, temp_config_dir):
    """Test that list_available_configs returns all YAML files in directory."""
    # Create test config files
    (temp_config_dir / "config1.yaml").touch()
    (temp_config_dir / "config2.yaml").touch()
    (temp_config_dir / "config3.yml").touch()
    (temp_config_dir / "not-yaml.txt").touch()

    configs = config_loader.list_available_configs()

    assert sorted(configs) == ["config1", "config2", "config3"]

  def test_list_available_configs_returns_empty_for_no_files(self, config_loader):
    """Test that list_available_configs returns empty list when no YAML files exist."""
    configs = config_loader.list_available_configs()
    assert configs == []

  def test_list_available_configs_raises_error_for_missing_directory(self):
    """Test that list_available_configs raises ConfigurationError for missing directory."""
    loader = ConfigLoader(Path("/nonexistent/directory"))

    with pytest.raises(ConfigurationError) as exc_info:
      loader.list_available_configs()

    assert "Configuration directory not found" in str(exc_info.value)

  def test_list_available_configs_raises_error_for_file_instead_of_directory(self, temp_config_dir):
    """Test that list_available_configs raises ConfigurationError when path is a file."""
    # Create a file instead of directory
    file_path = temp_config_dir / "not-a-directory"
    file_path.touch()

    loader = ConfigLoader(file_path)

    with pytest.raises(ConfigurationError) as exc_info:
      loader.list_available_configs()

    assert "Configuration path is not a directory" in str(exc_info.value)

  def test_discover_config_files_returns_name_to_path_mapping(self, config_loader, temp_config_dir):
    """Test that discover_config_files returns mapping of config names to paths."""
    # Create test config files
    config1 = temp_config_dir / "config1.yaml"
    config2 = temp_config_dir / "config2.yml"
    config1.touch()
    config2.touch()

    configs = config_loader.discover_config_files()

    assert len(configs) == 2
    assert configs["config1"] == config1
    assert configs["config2"] == config2

  def test_discover_config_files_prefers_yaml_over_yml(self, config_loader, temp_config_dir):
    """Test that discover_config_files prefers .yaml over .yml when both exist."""
    # Create both .yaml and .yml files with same name
    yaml_file = temp_config_dir / "config.yaml"
    yml_file = temp_config_dir / "config.yml"
    yaml_file.touch()
    yml_file.touch()

    configs = config_loader.discover_config_files()

    assert len(configs) == 1
    assert configs["config"] == yaml_file

  def test_validate_config_file_returns_true_for_valid_config(self, config_loader, temp_config_dir, sample_config_dict):
    """Test that validate_config_file returns True for valid configuration."""
    config_file = temp_config_dir / "valid.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(sample_config_dict, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key'
    }):
      is_valid, error = config_loader.validate_config_file("valid")

    assert is_valid is True
    assert error is None

  def test_validate_config_file_returns_false_for_invalid_config(self, config_loader, temp_config_dir):
    """Test that validate_config_file returns False with error message for invalid configuration."""
    # Create invalid config (missing providers)
    config_dict = {"defaults": {}}
    config_file = temp_config_dir / "invalid.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(config_dict, f, indent=2)

    is_valid, error = config_loader.validate_config_file("invalid")

    assert is_valid is False
    assert error is not None
    assert "At least one provider must be configured" in error

  def test_get_config_file_path_returns_yaml_path(self, config_loader, temp_config_dir):
    """Test that get_config_file_path returns path to .yaml file."""
    config_file = temp_config_dir / "test.yaml"
    config_file.touch()

    path = config_loader.get_config_file_path("test")
    assert path == config_file

  def test_get_config_file_path_returns_yml_path_when_yaml_missing(self, config_loader, temp_config_dir):
    """Test that get_config_file_path returns path to .yml file when .yaml doesn't exist."""
    config_file = temp_config_dir / "test.yml"
    config_file.touch()

    path = config_loader.get_config_file_path("test")
    assert path == config_file

  def test_get_config_file_path_prefers_yaml_over_yml(self, config_loader, temp_config_dir):
    """Test that get_config_file_path prefers .yaml over .yml when both exist."""
    yaml_file = temp_config_dir / "test.yaml"
    yml_file = temp_config_dir / "test.yml"
    yaml_file.touch()
    yml_file.touch()

    path = config_loader.get_config_file_path("test")
    assert path == yaml_file

  def test_get_config_file_path_raises_error_for_missing_file(self, config_loader):
    """Test that get_config_file_path raises ConfigurationError for missing file."""
    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.get_config_file_path("nonexistent")

    assert "Configuration file not found" in str(exc_info.value)
    assert "nonexistent.yaml or nonexistent.yml" in str(exc_info.value)

  def test_create_config_directory_creates_directory(self, temp_config_dir):
    """Test that create_config_directory creates the configuration directory."""
    # Use a subdirectory that doesn't exist
    new_dir = temp_config_dir / "new" / "config"
    loader = ConfigLoader(new_dir)

    loader.create_config_directory()

    assert new_dir.exists()
    assert new_dir.is_dir()

  def test_create_config_directory_succeeds_if_directory_exists(self, config_loader):
    """Test that create_config_directory succeeds if directory already exists."""
    # Directory already exists from fixture
    config_loader.create_config_directory()
    # Should not raise an error

  def test_resolve_environment_variables_handles_nested_structures(self, config_loader):
    """Test that _resolve_environment_variables handles nested dictionaries and lists."""
    config_dict = {
      "level1": {
        "level2": {
          "api_key": "${TEST_KEY}",
          "list_field": ["${TEST_ITEM1}", "${TEST_ITEM2}"]
        }
      },
      "top_level_list": [
        {"nested_key": "${TEST_NESTED}"},
        "${TEST_SIMPLE}"
      ]
    }

    with patch.dict(os.environ, {
      'TEST_KEY': 'secret',
      'TEST_ITEM1': 'item1',
      'TEST_ITEM2': 'item2',
      'TEST_NESTED': 'nested_value',
      'TEST_SIMPLE': 'simple_value'
    }):
      resolved = config_loader._resolve_environment_variables(config_dict)

    assert resolved["level1"]["level2"]["api_key"] == "secret"
    assert resolved["level1"]["level2"]["list_field"] == ["item1", "item2"]
    assert resolved["top_level_list"][0]["nested_key"] == "nested_value"
    assert resolved["top_level_list"][1] == "simple_value"

  def test_resolve_environment_variables_preserves_non_string_values(self, config_loader):
    """Test that _resolve_environment_variables preserves non-string values."""
    config_dict = {
      "string_with_var": "${TEST_VAR}",
      "plain_string": "no variables here",
      "integer": 42,
      "float": 3.14,
      "boolean": True,
      "null_value": None,
      "list": [1, 2, 3],
      "dict": {"key": "value"}
    }

    with patch.dict(os.environ, {'TEST_VAR': 'resolved'}):
      resolved = config_loader._resolve_environment_variables(config_dict)

    assert resolved["string_with_var"] == "resolved"
    assert resolved["plain_string"] == "no variables here"
    assert resolved["integer"] == 42
    assert resolved["float"] == 3.14
    assert resolved["boolean"] is True
    assert resolved["null_value"] is None
    assert resolved["list"] == [1, 2, 3]
    assert resolved["dict"] == {"key": "value"}

  def test_resolve_environment_variables_handles_multiple_vars_in_string(self, config_loader):
    """Test that _resolve_environment_variables handles multiple variables in one string."""
    config_dict = {
      "complex_string": "https://${HOST}:${PORT}/api/v${VERSION}"
    }

    with patch.dict(os.environ, {
      'HOST': 'api.example.com',
      'PORT': '8080',
      'VERSION': '2'
    }):
      resolved = config_loader._resolve_environment_variables(config_dict)

    assert resolved["complex_string"] == "https://api.example.com:8080/api/v2"

  def test_resolve_environment_variables_provides_path_context_in_errors(self, config_loader):
    """Test that _resolve_environment_variables provides path context in error messages."""
    config_dict = {
      "providers": {
        "openai": {
          "api_key": "${MISSING_KEY}"
        }
      }
    }

    with pytest.raises(ConfigurationError) as exc_info:
      config_loader._resolve_environment_variables(config_dict)

    error_message = str(exc_info.value)
    assert "Environment variable 'MISSING_KEY' is not set" in error_message
    assert "at providers.openai.api_key" in error_message

  def test_load_config_handles_file_encoding(self, config_loader, temp_config_dir, sample_config_dict):
    """Test that load_config handles UTF-8 encoding correctly."""
    # Create config file with UTF-8 content
    config_file = temp_config_dir / "utf8-test.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
      yaml.dump(sample_config_dict, f, indent=2, allow_unicode=True)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key'
    }):
      config = config_loader.load_config("utf8-test")

    assert isinstance(config, LLMConfig)

  def test_load_single_config_loads_specified_config(self, config_loader, temp_config_dir, sample_config_dict):
    """Test that load_single_config loads the specified configuration file."""
    # Create test config file
    config_file = temp_config_dir / "custom-config.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(sample_config_dict, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-openai-key',
      'ANTHROPIC_API_KEY': 'test-anthropic-key'
    }):
      config = config_loader.load_single_config("custom-config")

    assert isinstance(config, LLMConfig)
    assert config.defaults["timeout"] == 30
    assert "openai" in config.providers
    assert "anthropic" in config.providers
    assert config.providers["openai"].api_key == "test-openai-key"

  def test_load_single_config_uses_default_when_none_specified(self, config_loader, temp_config_dir, sample_config_dict):
    """Test that load_single_config uses 'default-llms' when config_name is None."""
    # Create default config file
    config_file = temp_config_dir / "default-llms.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(sample_config_dict, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key'
    }):
      config = config_loader.load_single_config(None)

    assert isinstance(config, LLMConfig)
    assert "openai" in config.providers

  def test_load_single_config_prefers_yaml_over_yml(self, config_loader, temp_config_dir, sample_config_dict):
    """Test that load_single_config prefers .yaml over .yml when both exist."""
    # Create both .yaml and .yml files
    yaml_file = temp_config_dir / "test-config.yaml"
    yml_file = temp_config_dir / "test-config.yml"
    
    # Different content to verify which file is loaded - use deep copy to avoid modifying original
    import copy
    yaml_config = copy.deepcopy(sample_config_dict)
    yaml_config["defaults"]["timeout"] = 45
    
    yml_config = copy.deepcopy(sample_config_dict)
    yml_config["defaults"]["timeout"] = 60

    with open(yaml_file, 'w') as f:
      yaml.dump(yaml_config, f, indent=2)
    with open(yml_file, 'w') as f:
      yaml.dump(yml_config, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key'
    }):
      config = config_loader.load_single_config("test-config")

    # Should load .yaml file (timeout=45), not .yml file (timeout=60)
    assert config.defaults["timeout"] == 45

  def test_load_single_config_loads_yml_when_yaml_missing(self, config_loader, temp_config_dir, sample_config_dict):
    """Test that load_single_config loads .yml file when .yaml doesn't exist."""
    # Create only .yml file
    yml_file = temp_config_dir / "yml-only.yml"
    with open(yml_file, 'w') as f:
      yaml.dump(sample_config_dict, f, indent=2)

    with patch.dict(os.environ, {
      'OPENAI_API_KEY': 'test-key',
      'ANTHROPIC_API_KEY': 'test-key'
    }):
      config = config_loader.load_single_config("yml-only")

    assert isinstance(config, LLMConfig)
    assert "openai" in config.providers

  def test_load_single_config_raises_error_for_missing_file(self, config_loader):
    """Test that load_single_config raises ConfigurationError for missing file."""
    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_single_config("nonexistent")

    assert "Configuration file not found" in str(exc_info.value)
    assert "nonexistent.yaml or nonexistent.yml" in str(exc_info.value)
    assert exc_info.value.config_file is not None

  def test_load_single_config_raises_error_for_invalid_yaml(self, config_loader, temp_config_dir):
    """Test that load_single_config raises ConfigurationError for invalid YAML."""
    # Create invalid YAML file
    config_file = temp_config_dir / "invalid-single.yaml"
    with open(config_file, 'w') as f:
      f.write("invalid: yaml: content: [\n")

    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_single_config("invalid-single")

    assert "Invalid YAML" in str(exc_info.value)
    assert exc_info.value.config_file is not None

  def test_load_single_config_raises_error_for_empty_file(self, config_loader, temp_config_dir):
    """Test that load_single_config raises ConfigurationError for empty file."""
    # Create empty file
    config_file = temp_config_dir / "empty-single.yaml"
    config_file.touch()

    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_single_config("empty-single")

    assert "Configuration file is empty" in str(exc_info.value)
    assert exc_info.value.config_file is not None

  def test_load_single_config_raises_error_for_non_dict_yaml(self, config_loader, temp_config_dir):
    """Test that load_single_config raises ConfigurationError for non-dictionary YAML."""
    # Create YAML file with list instead of dict
    config_file = temp_config_dir / "list-single.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(["item1", "item2"], f)

    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_single_config("list-single")

    assert "must contain a YAML dictionary" in str(exc_info.value)
    assert exc_info.value.config_file is not None

  def test_load_single_config_resolves_environment_variables(self, config_loader, temp_config_dir):
    """Test that load_single_config resolves environment variables correctly."""
    config_dict = {
      "defaults": {"timeout": 30},
      "providers": {
        "test": {
          "api_key": "${TEST_SINGLE_API_KEY}",
          "base_url": "https://${TEST_SINGLE_HOST}/api"
        }
      }
    }

    config_file = temp_config_dir / "env-single-test.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(config_dict, f, indent=2)

    with patch.dict(os.environ, {
      'TEST_SINGLE_API_KEY': 'single-secret-key',
      'TEST_SINGLE_HOST': 'single.example.com'
    }):
      config = config_loader.load_single_config("env-single-test")

    assert config.providers["test"].api_key == "single-secret-key"
    assert config.providers["test"].base_url == "https://single.example.com/api"

  def test_load_single_config_raises_error_for_missing_env_var(self, config_loader, temp_config_dir):
    """Test that load_single_config raises ConfigurationError for missing environment variables."""
    config_dict = {
      "defaults": {},
      "providers": {
        "test": {
          "api_key": "${MISSING_SINGLE_API_KEY}"
        }
      }
    }

    config_file = temp_config_dir / "missing-env-single.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(config_dict, f, indent=2)

    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_single_config("missing-env-single")

    assert "Environment variable 'MISSING_SINGLE_API_KEY' is not set" in str(exc_info.value)
    assert exc_info.value.config_file is not None

  def test_load_single_config_handles_file_read_errors(self, config_loader, temp_config_dir):
    """Test that load_single_config handles file read errors gracefully."""
    # Create a file and then make it unreadable (simulate permission error)
    config_file = temp_config_dir / "unreadable.yaml"
    config_file.write_text("defaults: {}\nproviders: {}")
    
    # Mock open to raise a permission error
    with patch('builtins.open', side_effect=PermissionError("Permission denied")):
      with pytest.raises(ConfigurationError) as exc_info:
        config_loader.load_single_config("unreadable")

    assert "Error reading configuration file" in str(exc_info.value)
    assert "Permission denied" in str(exc_info.value)
    assert exc_info.value.config_file is not None

  def test_load_config_handles_file_read_errors(self, config_loader, temp_config_dir):
    """Test that load_config handles file read errors gracefully."""
    # Create a file and then make it unreadable (simulate permission error)
    config_file = temp_config_dir / "unreadable-main.yaml"
    config_file.write_text("defaults: {}\nproviders: {}")
    
    # Mock open to raise a permission error
    with patch('builtins.open', side_effect=PermissionError("Permission denied")):
      with pytest.raises(ConfigurationError) as exc_info:
        config_loader.load_config("unreadable-main")

    assert "Error reading configuration file" in str(exc_info.value)
    assert "Permission denied" in str(exc_info.value)
    assert exc_info.value.config_file is not None

  def test_resolve_environment_variables_handles_empty_strings(self, config_loader):
    """Test that _resolve_environment_variables handles empty environment variables."""
    config_dict = {
      "test_field": "${EMPTY_VAR}"
    }

    with patch.dict(os.environ, {'EMPTY_VAR': ''}):
      resolved = config_loader._resolve_environment_variables(config_dict)

    assert resolved["test_field"] == ""

  def test_resolve_environment_variables_handles_special_characters(self, config_loader):
    """Test that _resolve_environment_variables handles special characters in environment variables."""
    config_dict = {
      "special_chars": "${SPECIAL_VAR}"
    }

    with patch.dict(os.environ, {'SPECIAL_VAR': 'value with spaces & symbols!@#$%^&*()'}):
      resolved = config_loader._resolve_environment_variables(config_dict)

    assert resolved["special_chars"] == "value with spaces & symbols!@#$%^&*()"

  def test_resolve_environment_variables_handles_numeric_env_vars(self, config_loader):
    """Test that _resolve_environment_variables handles numeric environment variables."""
    config_dict = {
      "numeric_field": "${NUMERIC_VAR}",
      "mixed_field": "port-${PORT_VAR}"
    }

    with patch.dict(os.environ, {
      'NUMERIC_VAR': '12345',
      'PORT_VAR': '8080'
    }):
      resolved = config_loader._resolve_environment_variables(config_dict)

    assert resolved["numeric_field"] == "12345"
    assert resolved["mixed_field"] == "port-8080"

  def test_resolve_environment_variables_handles_boolean_like_env_vars(self, config_loader):
    """Test that _resolve_environment_variables handles boolean-like environment variables."""
    config_dict = {
      "bool_field": "${BOOL_VAR}",
      "flag_field": "${FLAG_VAR}"
    }

    with patch.dict(os.environ, {
      'BOOL_VAR': 'true',
      'FLAG_VAR': 'false'
    }):
      resolved = config_loader._resolve_environment_variables(config_dict)

    assert resolved["bool_field"] == "true"
    assert resolved["flag_field"] == "false"

  def test_resolve_environment_variables_handles_deeply_nested_structures(self, config_loader):
    """Test that _resolve_environment_variables handles deeply nested data structures."""
    config_dict = {
      "level1": {
        "level2": {
          "level3": {
            "level4": {
              "deep_var": "${DEEP_VAR}",
              "deep_list": ["${ITEM1}", {"nested_deep": "${ITEM2}"}]
            }
          }
        }
      }
    }

    with patch.dict(os.environ, {
      'DEEP_VAR': 'deep_value',
      'ITEM1': 'list_item1',
      'ITEM2': 'nested_item2'
    }):
      resolved = config_loader._resolve_environment_variables(config_dict)

    assert resolved["level1"]["level2"]["level3"]["level4"]["deep_var"] == "deep_value"
    assert resolved["level1"]["level2"]["level3"]["level4"]["deep_list"][0] == "list_item1"
    assert resolved["level1"]["level2"]["level3"]["level4"]["deep_list"][1]["nested_deep"] == "nested_item2"

  def test_resolve_environment_variables_provides_detailed_path_context(self, config_loader):
    """Test that _resolve_environment_variables provides detailed path context for nested errors."""
    config_dict = {
      "providers": {
        "openai": {
          "auth": {
            "api_key": "${MISSING_NESTED_KEY}"
          }
        }
      }
    }

    with pytest.raises(ConfigurationError) as exc_info:
      config_loader._resolve_environment_variables(config_dict)

    error_message = str(exc_info.value)
    assert "Environment variable 'MISSING_NESTED_KEY' is not set" in error_message
    assert "at providers.openai.auth.api_key" in error_message

  def test_resolve_environment_variables_handles_list_index_paths(self, config_loader):
    """Test that _resolve_environment_variables provides correct path context for list items."""
    config_dict = {
      "models": [
        {"name": "model1", "key": "${VALID_KEY}"},
        {"name": "model2", "key": "${MISSING_LIST_KEY}"}
      ]
    }

    with patch.dict(os.environ, {'VALID_KEY': 'valid_value'}):
      with pytest.raises(ConfigurationError) as exc_info:
        config_loader._resolve_environment_variables(config_dict)

    error_message = str(exc_info.value)
    assert "Environment variable 'MISSING_LIST_KEY' is not set" in error_message
    assert "at models[1].key" in error_message

  def test_load_config_only_supports_yaml_extension(self, config_loader, temp_config_dir, sample_config_dict):
    """Test that load_config only supports .yaml extension, not .yml."""
    # Create config file with .yml extension only
    config_file = temp_config_dir / "yml-config.yml"
    with open(config_file, 'w') as f:
      yaml.dump(sample_config_dict, f, indent=2)

    # load_config should fail because it only looks for .yaml files
    with pytest.raises(ConfigurationError) as exc_info:
      config_loader.load_config("yml-config")

    assert "Configuration file not found" in str(exc_info.value)
    assert "yml-config.yaml" in str(exc_info.value)

  def test_list_available_configs_handles_mixed_extensions(self, config_loader, temp_config_dir):
    """Test that list_available_configs handles both .yaml and .yml extensions correctly."""
    # Create files with both extensions
    (temp_config_dir / "config1.yaml").touch()
    (temp_config_dir / "config2.yml").touch()
    (temp_config_dir / "config3.yaml").touch()
    (temp_config_dir / "config4.yml").touch()
    (temp_config_dir / "not-config.txt").touch()
    (temp_config_dir / "also-not-config.json").touch()

    configs = config_loader.list_available_configs()

    assert sorted(configs) == ["config1", "config2", "config3", "config4"]

  def test_discover_config_files_handles_empty_directory(self, config_loader):
    """Test that discover_config_files returns empty dict for directory with no YAML files."""
    configs = config_loader.discover_config_files()
    assert configs == {}

  def test_validate_config_file_handles_missing_file(self, config_loader):
    """Test that validate_config_file returns False for missing file."""
    is_valid, error = config_loader.validate_config_file("nonexistent")

    assert is_valid is False
    assert error is not None
    assert "Configuration file not found" in error

  def test_validate_config_file_handles_unexpected_errors(self, config_loader, temp_config_dir):
    """Test that validate_config_file handles unexpected errors gracefully."""
    # Create a minimal config that won't fail on environment variables
    minimal_config = {
      "defaults": {},
      "providers": {
        "test": {
          "api_key": "test-key"  # No environment variable to avoid that error
        }
      }
    }
    
    config_file = temp_config_dir / "test-unexpected.yaml"
    with open(config_file, 'w') as f:
      yaml.dump(minimal_config, f, indent=2)

    # Mock the load_config method to raise an unexpected error
    with patch.object(config_loader, 'load_config', side_effect=RuntimeError("Unexpected error")):
      is_valid, error = config_loader.validate_config_file("test-unexpected")

    assert is_valid is False
    assert error is not None
    assert "Unexpected error" in error

  def test_create_config_directory_handles_permission_errors(self, temp_config_dir):
    """Test that create_config_directory handles permission errors gracefully."""
    # Create a path that would cause permission error
    restricted_dir = temp_config_dir / "restricted" / "config"
    loader = ConfigLoader(restricted_dir)

    # Mock mkdir to raise permission error
    with patch.object(Path, 'mkdir', side_effect=PermissionError("Permission denied")):
      with pytest.raises(ConfigurationError) as exc_info:
        loader.create_config_directory()

    assert "Failed to create configuration directory" in str(exc_info.value)
    assert "Permission denied" in str(exc_info.value)