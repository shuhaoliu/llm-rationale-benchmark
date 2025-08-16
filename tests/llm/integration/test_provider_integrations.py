"""Integration tests for LLM providers."""

import pytest
import pytest_asyncio
import yaml
import json
from pathlib import Path
from aresponses import ResponsesMockServer

from rationale_benchmark.llm.client import LLMClient
from rationale_benchmark.llm.models import ModelRequest
from rationale_benchmark.llm.exceptions import LLMConnectorError

@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory."""
    config_path = tmp_path / "config"
    config_path.mkdir()
    return config_path

@pytest_asyncio.fixture
async def llm_client_factory(config_dir: Path):
    """Factory fixture to create an LLMClient for a specific provider test."""
    clients = []

    async def _factory(provider_name: str, base_url: str, api_key: str = "test-key"):
        config = {
            "providers": {
                provider_name: {
                    "name": provider_name,
                    "api_key": api_key,
                    "models": [f"{provider_name}-model"],
                    "base_url": base_url,
                }
            }
        }
        config_file = config_dir / f"{provider_name}-config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        client = LLMClient(config_dir=config_dir, config_name=f"{provider_name}-config")
        await client.initialize()
        clients.append(client)
        return client

    yield _factory

    for client in clients:
        await client.shutdown()

@pytest.mark.integration
class TestOpenAIProviderIntegration:
    """Integration tests for the OpenAI provider."""

    @pytest.mark.asyncio
    async def test_openai_successful_response(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test a successful request/response cycle with the OpenAI provider."""
        host = "openai.test.com"
        aresponses.add(
            host,
            "/v1/chat/completions",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello, world!"}, "finish_reason": "stop"}],
                    "model": "openai-model",
                    "object": "chat.completion",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        client = await llm_client_factory(
            "openai", f"https://{host}/v1", api_key="test-sk-" + "a" * 43
        )
        request = ModelRequest(prompt="Hello", model="openai-model")
        response = await client.generate_response(request)

        assert response.text == "Hello, world!"
        assert response.model == "openai-model"
        assert response.provider == "openai"

        assert len(aresponses.history) == 1
        sent_request = aresponses.history[0].request
        assert sent_request.headers["Authorization"] == "Bearer sk-" + "a" * 48
        sent_body = await sent_request.json()
        assert sent_body["model"] == "openai-model"
        assert sent_body["messages"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_openai_api_error(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test that an API error is handled correctly."""
        host = "openai.test.com"
        aresponses.add(
            host,
            "/v1/chat/completions",
            "post",
            aresponses.Response(status=500, text="Internal Server Error"),
        )

        client = await llm_client_factory(
            "openai", f"https://{host}/v1", api_key="test-sk-" + "a" * 43
        )
        request = ModelRequest(prompt="Hello", model="openai-model")

        with pytest.raises(LLMConnectorError, match="Request failed for model openai-model"):
            await client.generate_response(request)

    @pytest.mark.asyncio
    async def test_openai_streaming_param_filtering(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test that streaming parameters are filtered from the request."""
        host = "openai.test.com"
        aresponses.add(
            host,
            "/v1/chat/completions",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Streamed response"}, "finish_reason": "stop"}],
                    "model": "openai-model",
                    "object": "chat.completion",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        client = await llm_client_factory(
            "openai", f"https://{host}/v1", api_key="test-sk-" + "a" * 43
        )
        request = ModelRequest(
            prompt="Hello",
            model="openai-model",
            provider_specific={"stream": True, "temperature": 0.5}
        )

        await client.generate_response(request)

        assert len(aresponses.history) == 1
        sent_body = await aresponses.history[0].request.json()
        assert sent_body["stream"] is False
        assert sent_body["temperature"] == 0.5


@pytest.mark.integration
class TestAnthropicProviderIntegration:
    """Integration tests for the Anthropic provider."""

    @pytest.mark.asyncio
    async def test_anthropic_successful_response(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test a successful request/response cycle with the Anthropic provider."""
        host = "anthropic.test.com"
        aresponses.add(
            host,
            "/v1/messages",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "content": [{"type": "text", "text": "Hello from Anthropic!"}],
                    "model": "anthropic-model",
                    "role": "assistant",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        client = await llm_client_factory(
            "anthropic", f"https://{host}", api_key="test-sk-ant-api" + "a" * 80
        )
        request = ModelRequest(prompt="Hello", model="anthropic-model")
        response = await client.generate_response(request)

        assert response.text == "Hello from Anthropic!"
        assert response.model == "anthropic-model"
        assert response.provider == "anthropic"

        assert len(aresponses.history) == 1
        sent_request = aresponses.history[0].request
        assert sent_request.headers["x-api-key"] == "sk-ant-api" + "a" * 85
        sent_body = await sent_request.json()
        assert sent_body["model"] == "anthropic-model"
        assert sent_body["messages"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_anthropic_api_error(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test that an API error is handled correctly by the Anthropic provider."""
        host = "anthropic.test.com"
        aresponses.add(
            host,
            "/v1/messages",
            "post",
            aresponses.Response(status=500, text="Internal Server Error"),
        )

        client = await llm_client_factory(
            "anthropic", f"https://{host}", api_key="test-sk-ant-api" + "a" * 80
        )
        request = ModelRequest(prompt="Hello", model="anthropic-model")

        with pytest.raises(LLMConnectorError, match="Request failed for model anthropic-model"):
            await client.generate_response(request)

    @pytest.mark.asyncio
    async def test_anthropic_streaming_param_filtering(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test that streaming parameters are filtered from the request for the Anthropic provider."""
        host = "anthropic.test.com"
        aresponses.add(
            host,
            "/v1/messages",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "content": [{"type": "text", "text": "Streamed response"}],
                    "model": "anthropic-model",
                    "role": "assistant",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        client = await llm_client_factory(
            "anthropic", f"https://{host}", api_key="test-sk-ant-api" + "a" * 80
        )
        request = ModelRequest(
            prompt="Hello",
            model="anthropic-model",
            provider_specific={"stream": True, "temperature": 0.5}
        )

        await client.generate_response(request)

        assert len(aresponses.history) == 1
        sent_body = await aresponses.history[0].request.json()
        assert "stream" not in sent_body
        assert sent_body["temperature"] == 0.5


@pytest.mark.integration
class TestGeminiProviderIntegration:
    """Integration tests for the Gemini provider."""

    @pytest.mark.asyncio
    async def test_gemini_successful_response(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test a successful request/response cycle with the Gemini provider."""
        host = "gemini.test.com"
        model = "gemini-model"
        aresponses.add(
            host,
            f"/models/{model}:generateContent",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "candidates": [{
                        "content": {"parts": [{"text": "Hello from Gemini!"}], "role": "model"},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        client = await llm_client_factory(
            "gemini", f"https://{host}", api_key="test-AIza" + "a" * 30
        )
        request = ModelRequest(prompt="Hello", model=model)
        response = await client.generate_response(request)

        assert response.text == "Hello from Gemini!"
        assert response.model == model
        assert response.provider == "gemini"

        assert len(aresponses.history) == 1
        sent_request = aresponses.history[0].request
        assert sent_request.query["key"] == "AIza" + "a" * 35
        sent_body = await sent_request.json()
        assert sent_body["contents"][0]["parts"][0]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_gemini_api_error(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test that an API error is handled correctly by the Gemini provider."""
        host = "gemini.test.com"
        model = "gemini-model"
        aresponses.add(
            host,
            f"/models/{model}:generateContent",
            "post",
            aresponses.Response(status=500, text="Internal Server Error"),
        )

        client = await llm_client_factory(
            "gemini", f"https://{host}", api_key="test-AIza" + "a" * 30
        )
        request = ModelRequest(prompt="Hello", model=model)

        with pytest.raises(LLMConnectorError, match="Request failed for model gemini-model"):
            await client.generate_response(request)

    @pytest.mark.asyncio
    async def test_gemini_streaming_param_filtering(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test that streaming parameters are filtered from the request for the Gemini provider."""
        host = "gemini.test.com"
        model = "gemini-model"
        aresponses.add(
            host,
            f"/models/{model}:generateContent",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "candidates": [{
                        "content": {"parts": [{"text": "Streamed response"}], "role": "model"},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        client = await llm_client_factory(
            "gemini", f"https://{host}", api_key="test-AIza" + "a" * 30
        )
        request = ModelRequest(
            prompt="Hello",
            model=model,
            provider_specific={"stream": True, "temperature": 0.5}
        )

        await client.generate_response(request)

        assert len(aresponses.history) == 1
        sent_body = await aresponses.history[0].request.json()
        assert "stream" not in sent_body["generationConfig"]


@pytest.mark.integration
class TestOpenRouterProviderIntegration:
    """Integration tests for the OpenRouter provider."""

    @pytest.mark.asyncio
    async def test_openrouter_successful_response(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test a successful request/response cycle with the OpenRouter provider."""
        host = "openrouter.test.com"
        model = "openrouter-model"
        aresponses.add(
            host,
            "/api/v1/chat/completions",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from OpenRouter!"}, "finish_reason": "stop"}],
                    "model": model,
                    "object": "chat.completion",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        client = await llm_client_factory(
            "openrouter", f"https://{host}/api/v1", api_key="sk-or-" + "a" * 48
        )
        request = ModelRequest(prompt="Hello", model=model)
        response = await client.generate_response(request)

        assert response.text == "Hello from OpenRouter!"
        assert response.model == model
        assert response.provider == "openrouter"

        assert len(aresponses.history) == 1
        sent_request = aresponses.history[0].request
        assert sent_request.headers["Authorization"] == "Bearer sk-or-" + "a" * 48
        sent_body = await sent_request.json()
        assert sent_body["model"] == model
        assert sent_body["messages"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_openrouter_api_error(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test that an API error is handled correctly by the OpenRouter provider."""
        host = "openrouter.test.com"
        aresponses.add(
            host,
            "/api/v1/chat/completions",
            "post",
            aresponses.Response(status=500, text="Internal Server Error"),
        )

        client = await llm_client_factory(
            "openrouter", f"https://{host}/api/v1", api_key="sk-or-" + "a" * 48
        )
        request = ModelRequest(prompt="Hello", model="openrouter-model")

        with pytest.raises(LLMConnectorError, match="Request failed for model openrouter-model"):
            await client.generate_response(request)

    @pytest.mark.asyncio
    async def test_openrouter_streaming_param_filtering(self, aresponses: ResponsesMockServer, llm_client_factory):
        """Test that streaming parameters are filtered from the request for the OpenRouter provider."""
        host = "openrouter.test.com"
        aresponses.add(
            host,
            "/api/v1/chat/completions",
            "post",
            aresponses.Response(
                status=200,
                text=json.dumps({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Streamed response"}, "finish_reason": "stop"}],
                    "model": "openrouter-model",
                    "object": "chat.completion",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }),
                headers={"Content-Type": "application/json"},
            ),
        )

        client = await llm_client_factory(
            "openrouter", f"https://{host}/api/v1", api_key="sk-or-" + "a" * 48
        )
        request = ModelRequest(
            prompt="Hello",
            model="openrouter-model",
            provider_specific={"stream": True, "temperature": 0.5}
        )

        await client.generate_response(request)

        assert len(aresponses.history) == 1
        sent_body = await aresponses.history[0].request.json()
        assert sent_body["stream"] is False
        assert sent_body["temperature"] == 0.5
