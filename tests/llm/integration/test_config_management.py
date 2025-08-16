"""Integration tests for the configuration management system."""

import os
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from rationale_benchmark.llm.client import LLMClient
from rationale_benchmark.llm.exceptions import ConfigurationError

@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory."""
    config_path = tmp_path / "config"
    config_path.mkdir()
    return config_path

@pytest.fixture
def valid_config_file(config_dir: Path) -> Path:
    """Create a valid default-llms.yaml configuration file."""
    config = {
        "defaults": {
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "providers": {
            "openai": {
                "name": "openai",
                "api_key": "test-key-openai",
                "models": ["gpt-4", "gpt-3.5-turbo"],
            },
            "anthropic": {
                "name": "anthropic",
                "api_key": "test-key-anthropic",
                "models": ["claude-3-opus"],
            },
        },
    }
    config_file = config_dir / "default-llms.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def env_var_config_file(config_dir: Path) -> Path:
    """Create a config file with an environment variable placeholder."""
    config = {
        "providers": {
            "openai": {
                "name": "openai",
                "api_key": "${TEST_API_KEY}",
                "models": ["gpt-4"],
            },
        },
    }
    config_file = config_dir / "env-var-config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def streaming_config_file(config_dir: Path) -> Path:
    """Create a config file with streaming parameters."""
    config = {
        "defaults": {
            "temperature": 0.7,
            "stream": True,  # Streaming param in defaults
        },
        "providers": {
            "openai": {
                "name": "openai",
                "api_key": "test-key-openai",
                "models": ["gpt-4"],
                "default_params": {
                    "streaming": True,  # Streaming param in provider defaults
                },
            },
        },
    }
    config_file = config_dir / "streaming-config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def invalid_yaml_config_file(config_dir: Path) -> Path:
    """Create a config file with invalid YAML syntax."""
    config_file = config_dir / "invalid-yaml.yaml"
    with open(config_file, "w") as f:
        f.write("providers: { openai: { api_key: 'test' }")  # Missing closing brace
    return config_file


@pytest.fixture
def semantically_invalid_config_file(config_dir: Path) -> Path:
    """Create a config file that is syntactically correct but semantically invalid."""
    config = {
        "providers": {
            "openai": {
                # Missing 'name' and 'api_key'
                "models": ["gpt-4"],
            },
        },
    }
    config_file = config_dir / "semantic-invalid.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


@pytest.mark.integration
class TestConfigManagementIntegration:
    """Integration tests for configuration loading and validation."""

    @pytest.mark.asyncio
    async def test_load_valid_configuration(self, config_dir: Path, valid_config_file: Path):
        """Test end-to-end loading of a valid configuration file."""
        with patch("rationale_benchmark.llm.client.ProviderFactory") as mock_provider_factory:
            # Mock the provider factory to prevent actual provider initialization
            mock_factory_instance = mock_provider_factory.return_value
            mock_factory_instance.initialize_providers.return_value = None
            mock_factory_instance.get_model_to_provider_mapping.return_value = {}
            mock_factory_instance.list_providers.return_value = {}

            client = LLMClient(config_dir=config_dir)
            await client.initialize()

            assert client._is_initialized
            assert client.config is not None
            assert "openai" in client.config.providers
            assert "anthropic" in client.config.providers
            assert client.config.providers["openai"].api_key == "test-key-openai"

            await client.shutdown()

    @pytest.mark.asyncio
    async def test_environment_variable_substitution(self, config_dir: Path, env_var_config_file: Path, monkeypatch):
        """Test that environment variables in the config are correctly substituted."""
        monkeypatch.setenv("TEST_API_KEY", "test-my-secret-key")

        with patch("rationale_benchmark.llm.client.ProviderFactory") as mock_provider_factory:
            mock_factory_instance = mock_provider_factory.return_value
            mock_factory_instance.initialize_providers.return_value = None
            mock_factory_instance.get_model_to_provider_mapping.return_value = {}
            mock_factory_instance.list_providers.return_value = {}

            client = LLMClient(config_dir=config_dir, config_name="env-var-config")
            await client.initialize()

            assert client.config.providers["openai"].api_key == "test-my-secret-key"

            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_parameter_removal(self, config_dir: Path, streaming_config_file: Path, caplog):
        """Test that streaming parameters are removed from the config and a warning is logged."""
        with patch("rationale_benchmark.llm.client.ProviderFactory") as mock_provider_factory:
            mock_factory_instance = mock_provider_factory.return_value
            mock_factory_instance.initialize_providers.return_value = None
            mock_factory_instance.get_model_to_provider_mapping.return_value = {}
            mock_factory_instance.list_providers.return_value = {}

            client = LLMClient(config_dir=config_dir, config_name="streaming-config")
            await client.initialize()

            # Check that streaming params are removed
            assert "stream" not in client.config.defaults
            assert "streaming" not in client.config.providers["openai"].default_params

            # Check that warnings were logged
            assert "removing streaming parameters from configuration" in caplog.text.lower()
            assert "does not support streaming responses" in caplog.text.lower()

            await client.shutdown()

    @pytest.mark.asyncio
    async def test_invalid_yaml_config(self, config_dir: Path, invalid_yaml_config_file: Path):
        """Test that loading a malformed YAML file raises a ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Invalid YAML in configuration file"):
            client = LLMClient(config_dir=config_dir, config_name="invalid-yaml")
            await client.initialize()

    @pytest.mark.asyncio
    async def test_semantically_invalid_config(self, config_dir: Path, semantically_invalid_config_file: Path):
        """Test that a semantically invalid config raises a ConfigurationError during validation."""
        with pytest.raises(ConfigurationError, match="is missing required 'api_key' field"):
            client = LLMClient(config_dir=config_dir, config_name="semantic-invalid")
            await client.initialize()
