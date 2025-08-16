"""End-to-end integration tests for the complete LLM connector system."""

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
async def e2e_llm_client_factory(config_dir: Path):
    """Factory fixture to create an LLMClient for end-to-end tests."""
    clients = []

    async def _factory(providers: dict, config_name: str = "e2e-config"):
        config = {"providers": {}}
        if "defaults" in providers:
            config["defaults"] = providers.pop("defaults")

        for provider_name, provider_config in providers.items():
            config["providers"][provider_name] = {
                "name": provider_name,
                "api_key": provider_config["api_key"],
                "models": provider_config["models"],
                "base_url": provider_config["base_url"],
            }

        config_file = config_dir / f"{config_name}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        client = LLMClient(config_dir=config_dir, config_name=config_name)
        await client.initialize()
        clients.append(client)
        return client

    yield _factory

    for client in clients:
        await client.shutdown()

@pytest.mark.integration
class TestEndToEndIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_successful_flow(self, e2e_llm_client_factory, aresponses: ResponsesMockServer):
        """Test a full end-to-end flow with multiple providers."""
        openai_host = "openai-e2e.test.com"
        anthropic_host = "anthropic-e2e.test.com"

        # Mock responses
        aresponses.add(
            openai_host, "/v1/chat/completions", "post",
            aresponses.Response(status=200, text=json.dumps({
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "E2E OpenAI response"}, "finish_reason": "stop"}],
                "model": "openai-model", "object": "chat.completion",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }), headers={"Content-Type": "application/json"})
        )
        aresponses.add(
            anthropic_host, "/v1/messages", "post",
            aresponses.Response(status=200, text=json.dumps({
                "content": [{"type": "text", "text": "E2E Anthropic response"}],
                "model": "anthropic-model", "role": "assistant", "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 20}
            }), headers={"Content-Type": "application/json"})
        )

        client = await e2e_llm_client_factory({
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
            ModelRequest(prompt="Test OpenAI", model="openai-model"),
            ModelRequest(prompt="Test Anthropic", model="anthropic-model"),
        ]
        responses = await client.generate_responses_concurrent(requests)

        assert len(responses) == 2
        assert responses[0].text == "E2E OpenAI response"
        assert responses[1].text == "E2E Anthropic response"

    @pytest.mark.asyncio
    async def test_end_to_end_with_config_streaming_params(self, e2e_llm_client_factory, aresponses: ResponsesMockServer, caplog):
        """Test that streaming parameters in the config are handled correctly at the system level."""
        host = "openai-e2e.test.com"

        # Mock response
        aresponses.add(
            host, "/v1/chat/completions", "post",
            aresponses.Response(status=200, text=json.dumps({
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "E2E OpenAI response"}, "finish_reason": "stop"}],
                "model": "openai-model", "object": "chat.completion",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }), headers={"Content-Type": "application/json"})
        )

        client = await e2e_llm_client_factory(
            {
                "defaults": {"stream": True},
                "openai": {
                    "api_key": "sk-" + "a" * 48,
                    "models": ["openai-model"],
                    "base_url": f"https://{host}/v1",
                },
            },
            config_name="streaming-config"
        )

        # Check that a warning was logged
        assert "removing streaming parameters from configuration" in caplog.text.lower()

        request = ModelRequest(prompt="Test OpenAI", model="openai-model")
        await client.generate_response(request)

        assert len(aresponses.history) == 1
        sent_body = await aresponses.history[0].request.json()
        assert sent_body["stream"] is False
