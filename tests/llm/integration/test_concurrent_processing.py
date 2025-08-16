"""Integration tests for concurrent processing."""

import pytest
import pytest_asyncio
import yaml
import json
from pathlib import Path
from aresponses import ResponsesMockServer

from rationale_benchmark.llm.client import LLMClient
from rationale_benchmark.llm.models import ModelRequest

@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory."""
    config_path = tmp_path / "config"
    config_path.mkdir()
    return config_path

@pytest_asyncio.fixture
async def multi_provider_llm_client(config_dir: Path):
    """Factory fixture to create an LLMClient for a specific provider test."""
    clients = []

    async def _factory(providers: dict):
        config = {"providers": {}}
        for provider_name, provider_config in providers.items():
            config["providers"][provider_name] = {
                "name": provider_name,
                "api_key": provider_config["api_key"],
                "models": provider_config["models"],
                "base_url": provider_config["base_url"],
            }

        config_file = config_dir / "multi-provider-config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        client = LLMClient(config_dir=config_dir, config_name="multi-provider-config")
        await client.initialize()
        clients.append(client)
        return client

    yield _factory

    for client in clients:
        await client.shutdown()

@pytest.mark.integration
class TestConcurrentProcessing:
    @pytest.mark.asyncio
    async def test_concurrent_successful_responses(self, multi_provider_llm_client, aresponses: ResponsesMockServer):
        """Test concurrent processing of successful responses from multiple providers."""
        openai_host = "openai.test.com"
        anthropic_host = "anthropic.test.com"

        # Mock OpenAI response
        aresponses.add(
            openai_host,
            "/v1/chat/completions",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from OpenAI!"}, "finish_reason": "stop"}],
                    "model": "openai-model", "object": "chat.completion",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        # Mock Anthropic response
        aresponses.add(
            anthropic_host,
            "/v1/messages",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "content": [{"type": "text", "text": "Hello from Anthropic!"}],
                    "model": "anthropic-model", "role": "assistant", "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        client = await multi_provider_llm_client({
            "openai": {
                "api_key": "sk-" + "a" * 48,
                "models": ["openai-model"],
                "base_url": f"https://{openai_host}/v1",
            },
            "anthropic": {
                "api_key": "sk-ant-" + "a" * 95,
                "models": ["anthropic-model"],
                "base_url": f"https://{anthropic_host}",
            },
        })

        requests = [
            ModelRequest(prompt="Hello OpenAI", model="openai-model"),
            ModelRequest(prompt="Hello Anthropic", model="anthropic-model"),
        ]

        responses = await client.generate_responses_concurrent(requests)

        assert len(responses) == 2
        assert responses[0].provider == "openai"
        assert responses[0].text == "Hello from OpenAI!"
        assert responses[1].provider == "anthropic"
        assert responses[1].text == "Hello from Anthropic!"

    @pytest.mark.asyncio
    async def test_concurrent_with_failures(self, multi_provider_llm_client, aresponses: ResponsesMockServer):
        """Test that concurrent processing handles partial failures gracefully."""
        openai_host = "openai.test.com"
        anthropic_host = "anthropic.test.com"

        # Mock OpenAI successful response
        aresponses.add(
            openai_host,
            "/v1/chat/completions",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from OpenAI!"}, "finish_reason": "stop"}],
                    "model": "openai-model", "object": "chat.completion",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        # Mock Anthropic error response
        aresponses.add(
            anthropic_host,
            "/v1/messages",
            "post",
            aresponses.Response(status=500, text="Internal Server Error"),
        )

        client = await multi_provider_llm_client({
            "openai": {
                "api_key": "sk-" + "a" * 48,
                "models": ["openai-model"],
                "base_url": f"https://{openai_host}/v1",
            },
            "anthropic": {
                "api_key": "sk-ant-" + "a" * 95,
                "models": ["anthropic-model"],
                "base_url": f"https://{anthropic_host}",
            },
        })

        requests = [
            ModelRequest(prompt="Hello OpenAI", model="openai-model"),
            ModelRequest(prompt="Hello Anthropic", model="anthropic-model"),
        ]

        responses = await client.generate_responses_concurrent(requests)

        assert len(responses) == 2
        assert responses[0].provider == "openai"
        assert responses[0].text == "Hello from OpenAI!"
        assert responses[1].provider == "anthropic"
        assert responses[1].text.startswith("ERROR:")
