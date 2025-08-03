"""Unit tests for Gemini provider implementation."""

import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponse

from rationale_benchmark.llm.exceptions import (
  AuthenticationError,
  ModelNotFoundError,
  ProviderError,
  ResponseValidationError,
  StreamingNotSupportedError,
)
from rationale_benchmark.llm.http.client import HTTPClient
from rationale_benchmark.llm.models import ModelRequest, ModelResponse, ProviderConfig
from rationale_benchmark.llm.providers.gemini import GeminiProvider


class TestGeminiProvider:
  """Test cases for Gemini provider implementation."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="gemini",
      api_key="AIzaSyDfGhJkLmNoPqRsTuVwXyZ123456789012",  # Use proper Google API key format (39 chars)
      base_url="https://generativelanguage.googleapis.com/v1beta",
      timeout=30,
      max_retries=3,
      models=["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
      default_params={"temperature": 0.7, "max_tokens": 1000},
      provider_specific={}
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return AsyncMock(spec=HTTPClient)

  @pytest.fixture
  def gemini_provider(self, provider_config, http_client):
    """Create Gemini provider instance."""
    return GeminiProvider(provider_config, http_client)

  @pytest.fixture
  def model_request(self):
    """Create test model request."""
    return ModelRequest(
      prompt="What is the capital of France?",
      model="gemini-1.5-pro",
      temperature=0.7,
      max_tokens=1000,
      system_prompt="You are a helpful assistant.",
      stop_sequences=["END"],
      provider_specific={}
    )

  @pytest.fixture
  def valid_gemini_response(self):
    """Create valid Gemini API response."""
    return {
      "candidates": [
        {
          "content": {
            "parts": [
              {
                "text": "The capital of France is Paris."
              }
            ],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0,
          "safetyRatings": [
            {
              "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_HATE_SPEECH",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_HARASSMENT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
              "probability": "NEGLIGIBLE"
            }
          ]
        }
      ],
      "usageMetadata": {
        "promptTokenCount": 15,
        "candidatesTokenCount": 8,
        "totalTokenCount": 23
      },
      "modelVersion": "gemini-1.5-pro-001"
    }

  def test_init_sets_correct_attributes(self, provider_config, http_client):
    """Test that Gemini provider initializes with correct attributes."""
    provider = GeminiProvider(provider_config, http_client)
    
    assert provider.config == provider_config
    assert provider.http_client == http_client
    assert provider.name == "gemini"
    assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta"

  def test_init_uses_default_base_url_when_none_provided(self, http_client):
    """Test that provider uses default base URL when none is configured."""
    config = ProviderConfig(
      name="gemini",
      api_key="test-key",
      base_url=None
    )
    provider = GeminiProvider(config, http_client)
    
    assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta"

  def test_validate_config_returns_empty_list_for_valid_config(self, gemini_provider):
    """Test that validate_config returns empty list for valid configuration."""
    errors = gemini_provider.validate_config()
    assert errors == []

  def test_validate_config_returns_errors_for_missing_api_key(self, http_client):
    """Test that validate_config returns errors for missing API key."""
    # Create config with invalid API key that bypasses __post_init__ validation
    config = ProviderConfig.__new__(ProviderConfig)
    config.name = "gemini"
    config.api_key = ""  # Empty API key
    config.base_url = "https://generativelanguage.googleapis.com/v1beta"
    config.timeout = 30
    config.max_retries = 3
    config.models = []
    config.default_params = {}
    config.provider_specific = {}
    
    provider = GeminiProvider(config, http_client)
    
    errors = provider.validate_config()
    assert len(errors) > 0
    assert any("api key" in error.lower() for error in errors)

  def test_validate_config_returns_errors_for_invalid_api_key_format(self, http_client):
    """Test that validate_config returns errors for invalid API key format."""
    config = ProviderConfig.__new__(ProviderConfig)
    config.name = "gemini"
    config.api_key = "invalid-key-format"  # Invalid format
    config.base_url = "https://generativelanguage.googleapis.com/v1beta"
    config.timeout = 30
    config.max_retries = 3
    config.models = []
    config.default_params = {}
    config.provider_specific = {}
    
    provider = GeminiProvider(config, http_client)
    
    errors = provider.validate_config()
    assert len(errors) > 0
    assert any("api key format" in error.lower() for error in errors)

  @pytest.mark.asyncio
  async def test_generate_response_success(
    self, 
    gemini_provider, 
    model_request, 
    valid_gemini_response
  ):
    """Test successful response generation."""
    # Mock HTTP response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=valid_gemini_response)
    
    gemini_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request
    response = await gemini_provider.generate_response(model_request)
    
    # Verify response
    assert isinstance(response, ModelResponse)
    assert response.text == "The capital of France is Paris."
    assert response.model == "gemini-1.5-pro"
    assert response.provider == "gemini"
    assert response.token_count == 23
    assert response.finish_reason == "STOP"
    assert response.latency_ms >= 0  # Allow 0 for very fast mock responses

  @pytest.mark.asyncio
  async def test_generate_response_handles_http_error(
    self, 
    gemini_provider, 
    model_request
  ):
    """Test that HTTP errors are handled properly."""
    # Mock HTTP error response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 401
    mock_response.json = AsyncMock(return_value={
      "error": {
        "code": 401,
        "message": "API key not valid. Please pass a valid API key.",
        "status": "UNAUTHENTICATED"
      }
    })
    
    gemini_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request and expect error
    with pytest.raises(AuthenticationError) as exc_info:
      await gemini_provider.generate_response(model_request)
    
    assert "API key not valid" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_handles_model_not_found(
    self, 
    gemini_provider, 
    model_request
  ):
    """Test handling of model not found errors."""
    # Mock model not found response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 404
    mock_response.json = AsyncMock(return_value={
      "error": {
        "code": 404,
        "message": "Model 'gemini-nonexistent' not found",
        "status": "NOT_FOUND"
      }
    })
    
    gemini_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request and expect error
    with pytest.raises(ModelNotFoundError) as exc_info:
      await gemini_provider.generate_response(model_request)
    
    assert "not found" in str(exc_info.value).lower()

  @pytest.mark.asyncio
  async def test_generate_response_handles_rate_limit_error(
    self, 
    gemini_provider, 
    model_request
  ):
    """Test handling of rate limit errors."""
    # Mock rate limit response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 429
    mock_response.json = AsyncMock(return_value={
      "error": {
        "code": 429,
        "message": "Quota exceeded for requests per minute",
        "status": "RESOURCE_EXHAUSTED"
      }
    })
    
    gemini_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request and expect error
    with pytest.raises(ProviderError) as exc_info:
      await gemini_provider.generate_response(model_request)
    
    assert "429" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_validates_response_structure(
    self, 
    gemini_provider, 
    model_request
  ):
    """Test that response structure validation is performed."""
    # Mock invalid response (missing required fields)
    invalid_response = {
      "candidates": []  # Empty candidates array
    }
    
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=invalid_response)
    
    gemini_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request and expect validation error
    with pytest.raises(ResponseValidationError) as exc_info:
      await gemini_provider.generate_response(model_request)
    
    assert "candidates" in str(exc_info.value).lower()

  @pytest.mark.asyncio
  async def test_generate_response_blocks_streaming_parameters(
    self, 
    gemini_provider
  ):
    """Test that streaming parameters are blocked."""
    request_with_streaming = ModelRequest(
      prompt="Test prompt",
      model="gemini-1.5-pro",
      provider_specific={"stream": True}  # Streaming parameter
    )
    
    # Execute request and expect streaming error
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      await gemini_provider.generate_response(request_with_streaming)
    
    assert "stream" in str(exc_info.value).lower()

  @pytest.mark.asyncio
  async def test_list_models_success(self, gemini_provider):
    """Test successful model listing."""
    mock_models_response = {
      "models": [
        {
          "name": "models/gemini-1.5-pro",
          "displayName": "Gemini 1.5 Pro",
          "description": "Mid-size multimodal model",
          "inputTokenLimit": 2097152,
          "outputTokenLimit": 8192,
          "supportedGenerationMethods": ["generateContent", "countTokens"]
        },
        {
          "name": "models/gemini-1.5-flash",
          "displayName": "Gemini 1.5 Flash",
          "description": "Fast and versatile multimodal model",
          "inputTokenLimit": 1048576,
          "outputTokenLimit": 8192,
          "supportedGenerationMethods": ["generateContent", "countTokens"]
        }
      ]
    }
    
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=mock_models_response)
    
    gemini_provider.http_client.get = AsyncMock(return_value=mock_response)
    
    # Execute request
    models = await gemini_provider.list_models()
    
    # Verify response
    assert isinstance(models, list)
    assert len(models) == 2
    assert "gemini-1.5-pro" in models
    assert "gemini-1.5-flash" in models

  @pytest.mark.asyncio
  async def test_list_models_handles_error(self, gemini_provider):
    """Test that model listing handles errors properly."""
    # Mock HTTP error response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 403
    mock_response.json = AsyncMock(return_value={
      "error": {
        "code": 403,
        "message": "Permission denied",
        "status": "PERMISSION_DENIED"
      }
    })
    
    gemini_provider.http_client.get = AsyncMock(return_value=mock_response)
    
    # Execute request and expect error
    with pytest.raises(ProviderError) as exc_info:
      await gemini_provider.list_models()
    
    assert "403" in str(exc_info.value)

  def test_prepare_request_formats_correctly(self, gemini_provider, model_request):
    """Test that _prepare_request formats request correctly."""
    request_payload = gemini_provider._prepare_request(model_request)
    
    # Verify basic structure
    assert "contents" in request_payload
    assert "generationConfig" in request_payload
    
    # Verify contents structure
    contents = request_payload["contents"]
    assert isinstance(contents, list)
    assert len(contents) >= 1  # At least user message
    
    # Check for system prompt
    if model_request.system_prompt:
      assert any(content.get("role") == "system" for content in contents)
    
    # Check user message
    user_content = next(content for content in contents if content.get("role") == "user")
    assert user_content["parts"][0]["text"] == model_request.prompt
    
    # Verify generation config
    gen_config = request_payload["generationConfig"]
    assert gen_config["temperature"] == model_request.temperature
    assert gen_config["maxOutputTokens"] == model_request.max_tokens
    
    # Verify stop sequences
    if model_request.stop_sequences:
      assert gen_config["stopSequences"] == model_request.stop_sequences

  def test_prepare_request_blocks_streaming_parameters(self, gemini_provider):
    """Test that _prepare_request blocks streaming parameters."""
    request_with_streaming = ModelRequest(
      prompt="Test prompt",
      model="gemini-1.5-pro",
      provider_specific={"stream": True, "streaming": True}
    )
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      gemini_provider._prepare_request(request_with_streaming)
    
    assert "stream" in str(exc_info.value).lower()

  def test_prepare_request_includes_provider_specific_params(self, gemini_provider):
    """Test that _prepare_request includes valid provider-specific parameters."""
    request_with_specific = ModelRequest(
      prompt="Test prompt",
      model="gemini-1.5-pro",
      provider_specific={
        "topK": 40,
        "topP": 0.95,
        "candidateCount": 1
      }
    )
    
    request_payload = gemini_provider._prepare_request(request_with_specific)
    
    gen_config = request_payload["generationConfig"]
    assert gen_config["topK"] == 40
    assert gen_config["topP"] == 0.95
    assert gen_config["candidateCount"] == 1

  def test_parse_response_extracts_data_correctly(
    self, 
    gemini_provider, 
    model_request, 
    valid_gemini_response
  ):
    """Test that _parse_response extracts data correctly."""
    latency_ms = 1500
    
    response = gemini_provider._parse_response(
      valid_gemini_response, 
      model_request, 
      latency_ms
    )
    
    # Verify response structure
    assert isinstance(response, ModelResponse)
    assert response.text == "The capital of France is Paris."
    assert response.model == "gemini-1.5-pro"
    assert response.provider == "gemini"
    assert response.latency_ms == latency_ms
    assert response.token_count == 23
    assert response.finish_reason == "STOP"
    
    # Verify metadata
    assert "promptTokenCount" in response.metadata
    assert "candidatesTokenCount" in response.metadata
    assert "modelVersion" in response.metadata
    assert response.metadata["promptTokenCount"] == 15
    assert response.metadata["candidatesTokenCount"] == 8

  def test_parse_response_handles_missing_usage_metadata(
    self, 
    gemini_provider, 
    model_request
  ):
    """Test that _parse_response handles missing usage metadata."""
    response_without_usage = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "Test response"}],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ],
      "modelVersion": "gemini-1.5-pro-001"
    }
    
    response = gemini_provider._parse_response(
      response_without_usage, 
      model_request, 
      1000
    )
    
    assert response.token_count is None
    assert "promptTokenCount" not in response.metadata

  def test_validate_response_structure_passes_for_valid_response(
    self, 
    gemini_provider, 
    valid_gemini_response
  ):
    """Test that _validate_response_structure passes for valid response."""
    # Should not raise any exception
    gemini_provider._validate_response_structure(valid_gemini_response)

  def test_validate_response_structure_fails_for_missing_candidates(
    self, 
    gemini_provider
  ):
    """Test that _validate_response_structure fails for missing candidates."""
    invalid_response = {
      "usageMetadata": {
        "totalTokenCount": 10
      }
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "candidates" in str(exc_info.value).lower()

  def test_validate_response_structure_fails_for_empty_candidates(
    self, 
    gemini_provider
  ):
    """Test that _validate_response_structure fails for empty candidates."""
    invalid_response = {
      "candidates": []
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "candidates" in str(exc_info.value).lower()

  def test_validate_response_structure_fails_for_missing_content(
    self, 
    gemini_provider
  ):
    """Test that _validate_response_structure fails for missing content."""
    invalid_response = {
      "candidates": [
        {
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "content" in str(exc_info.value).lower()

  def test_validate_response_structure_fails_for_empty_text(
    self, 
    gemini_provider
  ):
    """Test that _validate_response_structure fails for empty text."""
    invalid_response = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": ""}],  # Empty text
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "empty" in str(exc_info.value).lower()

  def test_is_valid_gemini_parameter_accepts_valid_params(self, gemini_provider):
    """Test that _is_valid_gemini_parameter accepts valid parameters."""
    valid_params = [
      ("topK", 40),
      ("topP", 0.95),
      ("candidateCount", 1),
      ("maxOutputTokens", 1000),
      ("temperature", 0.7),
      ("stopSequences", ["END"]),
      ("responseMimeType", "text/plain"),
      ("responseSchema", {}),
      ("presencePenalty", 0.5),
      ("frequencyPenalty", 0.5)
    ]
    
    for param_name, param_value in valid_params:
      assert gemini_provider._is_valid_gemini_parameter(param_name, param_value)

  def test_is_valid_gemini_parameter_rejects_invalid_params(self, gemini_provider):
    """Test that _is_valid_gemini_parameter rejects invalid parameters."""
    invalid_params = [
      ("invalidParam", "value"),
      ("stream", True),
      ("streaming", True),
      ("unknownParameter", 123)
    ]
    
    for param_name, param_value in invalid_params:
      assert not gemini_provider._is_valid_gemini_parameter(param_name, param_value)

  def test_estimate_cost_returns_none_for_unknown_model(self, gemini_provider):
    """Test that _estimate_cost returns None for unknown models."""
    usage_metadata = {
      "promptTokenCount": 100,
      "candidatesTokenCount": 50,
      "totalTokenCount": 150
    }
    
    cost = gemini_provider._estimate_cost(usage_metadata, "unknown-model")
    assert cost is None

  def test_estimate_cost_calculates_correctly_for_known_model(self, gemini_provider):
    """Test that _estimate_cost calculates correctly for known models."""
    usage_metadata = {
      "promptTokenCount": 1000,
      "candidatesTokenCount": 500,
      "totalTokenCount": 1500
    }
    
    cost = gemini_provider._estimate_cost(usage_metadata, "gemini-1.5-pro")
    assert cost is not None
    assert cost > 0
    assert isinstance(cost, float)

  def test_str_representation(self, gemini_provider):
    """Test string representation of provider."""
    str_repr = str(gemini_provider)
    assert "GeminiProvider" in str_repr
    assert "gemini" in str_repr

  def test_repr_representation(self, gemini_provider):
    """Test detailed string representation of provider."""
    repr_str = repr(gemini_provider)
    assert "GeminiProvider" in repr_str
    assert "name='gemini'" in repr_str
    assert "base_url=" in repr_str
    assert "timeout=" in repr_str


class TestGeminiRequestResponseHandling:
  """Test cases for Gemini request/response handling methods."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="gemini",
      api_key="AIzaSyDfGhJkLmNoPqRsTuVwXyZ123456789012",  # Use proper Google API key format (39 chars)
      base_url="https://generativelanguage.googleapis.com/v1beta",
      timeout=30,
      max_retries=3,
      models=["gemini-1.5-pro", "gemini-1.5-flash"],
      default_params={"temperature": 0.7, "max_tokens": 1000},
      provider_specific={}
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return AsyncMock(spec=HTTPClient)

  @pytest.fixture
  def gemini_provider(self, provider_config, http_client):
    """Create Gemini provider instance."""
    return GeminiProvider(provider_config, http_client)

  def test_prepare_request_basic_structure(self, gemini_provider):
    """Test that _prepare_request creates correct basic structure."""
    request = ModelRequest(
      prompt="What is AI?",
      model="gemini-1.5-pro",
      temperature=0.8,
      max_tokens=500
    )
    
    payload = gemini_provider._prepare_request(request)
    
    # Verify top-level structure
    assert "contents" in payload
    assert "generationConfig" in payload
    assert isinstance(payload["contents"], list)
    assert isinstance(payload["generationConfig"], dict)

  def test_prepare_request_user_message_format(self, gemini_provider):
    """Test that user message is formatted correctly."""
    request = ModelRequest(
      prompt="Explain quantum computing",
      model="gemini-1.5-pro"
    )
    
    payload = gemini_provider._prepare_request(request)
    
    # Find user message
    user_message = next(
      content for content in payload["contents"] 
      if content.get("role") == "user"
    )
    
    assert user_message["role"] == "user"
    assert "parts" in user_message
    assert len(user_message["parts"]) == 1
    assert user_message["parts"][0]["text"] == "Explain quantum computing"

  def test_prepare_request_system_prompt_handling(self, gemini_provider):
    """Test that system prompt is handled correctly."""
    request = ModelRequest(
      prompt="What is the weather?",
      model="gemini-1.5-pro",
      system_prompt="You are a weather expert."
    )
    
    payload = gemini_provider._prepare_request(request)
    
    # Check for system message
    system_messages = [
      content for content in payload["contents"] 
      if content.get("role") == "system"
    ]
    
    assert len(system_messages) == 1
    system_message = system_messages[0]
    assert system_message["role"] == "system"
    assert system_message["parts"][0]["text"] == "You are a weather expert."

  def test_prepare_request_generation_config_basic_params(self, gemini_provider):
    """Test that basic generation config parameters are set correctly."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gemini-1.5-pro",
      temperature=0.9,
      max_tokens=2000
    )
    
    payload = gemini_provider._prepare_request(request)
    
    gen_config = payload["generationConfig"]
    assert gen_config["temperature"] == 0.9
    assert gen_config["maxOutputTokens"] == 2000

  def test_prepare_request_stop_sequences(self, gemini_provider):
    """Test that stop sequences are handled correctly."""
    request = ModelRequest(
      prompt="Count to 10",
      model="gemini-1.5-pro",
      stop_sequences=["STOP", "END", "FINISH"]
    )
    
    payload = gemini_provider._prepare_request(request)
    
    gen_config = payload["generationConfig"]
    assert "stopSequences" in gen_config
    assert gen_config["stopSequences"] == ["STOP", "END", "FINISH"]

  def test_prepare_request_no_stop_sequences(self, gemini_provider):
    """Test that stop sequences are omitted when not provided."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gemini-1.5-pro",
      stop_sequences=None
    )
    
    payload = gemini_provider._prepare_request(request)
    
    gen_config = payload["generationConfig"]
    assert "stopSequences" not in gen_config

  def test_prepare_request_provider_specific_params(self, gemini_provider):
    """Test that valid provider-specific parameters are included."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gemini-1.5-pro",
      provider_specific={
        "topK": 40,
        "topP": 0.95,
        "candidateCount": 1,
        "presencePenalty": 0.5,
        "frequencyPenalty": 0.3
      }
    )
    
    payload = gemini_provider._prepare_request(request)
    
    gen_config = payload["generationConfig"]
    assert gen_config["topK"] == 40
    assert gen_config["topP"] == 0.95
    assert gen_config["candidateCount"] == 1
    assert gen_config["presencePenalty"] == 0.5
    assert gen_config["frequencyPenalty"] == 0.3

  def test_prepare_request_filters_invalid_params(self, gemini_provider):
    """Test that invalid provider-specific parameters are filtered out."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gemini-1.5-pro",
      provider_specific={
        "topK": 40,  # Valid
        "invalidParam": "value",  # Invalid
        "unknownSetting": 123  # Invalid
      }
    )
    
    payload = gemini_provider._prepare_request(request)
    
    gen_config = payload["generationConfig"]
    assert gen_config["topK"] == 40
    assert "invalidParam" not in gen_config
    assert "unknownSetting" not in gen_config

  def test_prepare_request_blocks_streaming_params(self, gemini_provider):
    """Test that streaming parameters are blocked and raise error."""
    streaming_params = ["stream", "streaming", "stream_options"]
    
    for param in streaming_params:
      request = ModelRequest(
        prompt="Test prompt",
        model="gemini-1.5-pro",
        provider_specific={param: True}
      )
      
      with pytest.raises(StreamingNotSupportedError) as exc_info:
        gemini_provider._prepare_request(request)
      
      assert param in str(exc_info.value).lower()

  def test_prepare_request_multiple_streaming_params_blocked(self, gemini_provider):
    """Test that multiple streaming parameters are all blocked."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gemini-1.5-pro",
      provider_specific={
        "stream": True,
        "streaming": True,
        "stream_options": {"include_usage": True}
      }
    )
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      gemini_provider._prepare_request(request)
    
    error_msg = str(exc_info.value).lower()
    assert "stream" in error_msg

  def test_parse_response_basic_extraction(self, gemini_provider):
    """Test basic response data extraction."""
    response_data = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "Paris is the capital of France."}],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ],
      "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 8,
        "totalTokenCount": 18
      },
      "modelVersion": "gemini-1.5-pro-001"
    }
    
    request = ModelRequest(prompt="Test", model="gemini-1.5-pro")
    latency_ms = 1200
    
    response = gemini_provider._parse_response(response_data, request, latency_ms)
    
    assert response.text == "Paris is the capital of France."
    assert response.model == "gemini-1.5-pro"
    assert response.provider == "gemini"
    assert response.latency_ms == 1200
    assert response.token_count == 18
    assert response.finish_reason == "STOP"

  def test_parse_response_metadata_extraction(self, gemini_provider):
    """Test that metadata is extracted correctly."""
    response_data = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "Test response"}],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0,
          "safetyRatings": [
            {
              "category": "HARM_CATEGORY_HARASSMENT",
              "probability": "NEGLIGIBLE"
            }
          ]
        }
      ],
      "usageMetadata": {
        "promptTokenCount": 5,
        "candidatesTokenCount": 3,
        "totalTokenCount": 8
      },
      "modelVersion": "gemini-1.5-flash-001"
    }
    
    request = ModelRequest(prompt="Test", model="gemini-1.5-flash")
    
    response = gemini_provider._parse_response(response_data, request, 800)
    
    # Check metadata
    assert "promptTokenCount" in response.metadata
    assert "candidatesTokenCount" in response.metadata
    assert "modelVersion" in response.metadata
    assert "safetyRatings" in response.metadata
    
    assert response.metadata["promptTokenCount"] == 5
    assert response.metadata["candidatesTokenCount"] == 3
    assert response.metadata["modelVersion"] == "gemini-1.5-flash-001"

  def test_parse_response_missing_usage_metadata(self, gemini_provider):
    """Test parsing response without usage metadata."""
    response_data = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "Response without usage"}],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ],
      "modelVersion": "gemini-1.5-pro-001"
    }
    
    request = ModelRequest(prompt="Test", model="gemini-1.5-pro")
    
    response = gemini_provider._parse_response(response_data, request, 1000)
    
    assert response.text == "Response without usage"
    assert response.token_count is None
    assert "promptTokenCount" not in response.metadata
    assert "candidatesTokenCount" not in response.metadata

  def test_parse_response_multiple_parts_concatenation(self, gemini_provider):
    """Test that multiple parts are concatenated correctly."""
    response_data = {
      "candidates": [
        {
          "content": {
            "parts": [
              {"text": "First part. "},
              {"text": "Second part. "},
              {"text": "Third part."}
            ],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    request = ModelRequest(prompt="Test", model="gemini-1.5-pro")
    
    response = gemini_provider._parse_response(response_data, request, 500)
    
    assert response.text == "First part. Second part. Third part."

  def test_parse_response_cost_estimation(self, gemini_provider):
    """Test that cost estimation is included when available."""
    response_data = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "Test response"}],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ],
      "usageMetadata": {
        "promptTokenCount": 100,
        "candidatesTokenCount": 50,
        "totalTokenCount": 150
      }
    }
    
    request = ModelRequest(prompt="Test", model="gemini-1.5-pro")
    
    response = gemini_provider._parse_response(response_data, request, 1000)
    
    # Cost estimation should be present for known models
    assert response.cost_estimate is not None
    assert response.cost_estimate >= 0

  def test_parse_response_timestamp_is_recent(self, gemini_provider):
    """Test that response timestamp is recent."""
    response_data = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "Test response"}],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    request = ModelRequest(prompt="Test", model="gemini-1.5-pro")
    
    before_time = datetime.now()
    response = gemini_provider._parse_response(response_data, request, 1000)
    after_time = datetime.now()
    
    assert before_time <= response.timestamp <= after_time

  def test_validate_response_structure_valid_response(self, gemini_provider):
    """Test validation passes for valid response structure."""
    valid_response = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "Valid response"}],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ],
      "usageMetadata": {
        "totalTokenCount": 10
      }
    }
    
    # Should not raise any exception
    gemini_provider._validate_response_structure(valid_response)

  def test_validate_response_structure_missing_candidates(self, gemini_provider):
    """Test validation fails for missing candidates field."""
    invalid_response = {
      "usageMetadata": {
        "totalTokenCount": 10
      }
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "candidates" in str(exc_info.value).lower()

  def test_validate_response_structure_empty_candidates(self, gemini_provider):
    """Test validation fails for empty candidates array."""
    invalid_response = {
      "candidates": []
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "candidates" in str(exc_info.value).lower()
    assert "empty" in str(exc_info.value).lower()

  def test_validate_response_structure_missing_content(self, gemini_provider):
    """Test validation fails for missing content in candidate."""
    invalid_response = {
      "candidates": [
        {
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "content" in str(exc_info.value).lower()

  def test_validate_response_structure_missing_parts(self, gemini_provider):
    """Test validation fails for missing parts in content."""
    invalid_response = {
      "candidates": [
        {
          "content": {
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "parts" in str(exc_info.value).lower()

  def test_validate_response_structure_empty_parts(self, gemini_provider):
    """Test validation fails for empty parts array."""
    invalid_response = {
      "candidates": [
        {
          "content": {
            "parts": [],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "parts" in str(exc_info.value).lower()
    assert "empty" in str(exc_info.value).lower()

  def test_validate_response_structure_missing_text_in_part(self, gemini_provider):
    """Test validation fails for missing text in part."""
    invalid_response = {
      "candidates": [
        {
          "content": {
            "parts": [{}],  # Part without text
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "text" in str(exc_info.value).lower()

  def test_validate_response_structure_empty_text(self, gemini_provider):
    """Test validation fails for empty text content."""
    invalid_response = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": ""}],  # Empty text
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "empty" in str(exc_info.value).lower()

  def test_validate_response_structure_whitespace_only_text(self, gemini_provider):
    """Test validation fails for whitespace-only text content."""
    invalid_response = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "   \n\t  "}],  # Whitespace only
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "empty" in str(exc_info.value).lower() or "whitespace" in str(exc_info.value).lower()

  def test_validate_response_structure_invalid_role(self, gemini_provider):
    """Test validation fails for invalid role in content."""
    invalid_response = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "Valid text"}],
            "role": "user"  # Should be "model"
          },
          "finishReason": "STOP",
          "index": 0
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "role" in str(exc_info.value).lower()

  def test_validate_response_structure_missing_finish_reason(self, gemini_provider):
    """Test validation fails for missing finish reason."""
    invalid_response = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "Valid text"}],
            "role": "model"
          },
          "index": 0
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "finishReason" in str(exc_info.value) or "finish_reason" in str(exc_info.value).lower()

  def test_validate_response_structure_invalid_index(self, gemini_provider):
    """Test validation fails for invalid index."""
    invalid_response = {
      "candidates": [
        {
          "content": {
            "parts": [{"text": "Valid text"}],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": -1  # Invalid negative index
        }
      ]
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_provider._validate_response_structure(invalid_response)
    
    assert "index" in str(exc_info.value).lower()

  def test_is_valid_gemini_parameter_comprehensive_valid_params(self, gemini_provider):
    """Test comprehensive list of valid Gemini parameters."""
    valid_params = {
      "topK": 40,
      "topP": 0.95,
      "temperature": 0.7,
      "maxOutputTokens": 1000,
      "candidateCount": 1,
      "stopSequences": ["STOP"],
      "responseMimeType": "text/plain",
      "responseSchema": {"type": "object"},
      "presencePenalty": 0.5,
      "frequencyPenalty": 0.3,
      "seed": 12345
    }
    
    for param_name, param_value in valid_params.items():
      assert gemini_provider._is_valid_gemini_parameter(param_name, param_value), \
        f"Parameter {param_name} should be valid"

  def test_is_valid_gemini_parameter_invalid_params(self, gemini_provider):
    """Test that invalid parameters are rejected."""
    invalid_params = {
      "stream": True,
      "streaming": True,
      "stream_options": {},
      "invalidParam": "value",
      "unknownSetting": 123,
      "customParam": "test"
    }
    
    for param_name, param_value in invalid_params.items():
      assert not gemini_provider._is_valid_gemini_parameter(param_name, param_value), \
        f"Parameter {param_name} should be invalid"

  def test_estimate_cost_known_models(self, gemini_provider):
    """Test cost estimation for known Gemini models."""
    usage_metadata = {
      "promptTokenCount": 1000,
      "candidatesTokenCount": 500,
      "totalTokenCount": 1500
    }
    
    known_models = [
      "gemini-1.5-pro",
      "gemini-1.5-flash",
      "gemini-pro",
      "gemini-pro-vision"
    ]
    
    for model in known_models:
      cost = gemini_provider._estimate_cost(usage_metadata, model)
      assert cost is not None, f"Cost should be estimated for {model}"
      assert cost > 0, f"Cost should be positive for {model}"
      assert isinstance(cost, float), f"Cost should be float for {model}"

  def test_estimate_cost_unknown_model(self, gemini_provider):
    """Test cost estimation returns None for unknown models."""
    usage_metadata = {
      "promptTokenCount": 1000,
      "candidatesTokenCount": 500,
      "totalTokenCount": 1500
    }
    
    cost = gemini_provider._estimate_cost(usage_metadata, "unknown-model")
    assert cost is None

  def test_estimate_cost_missing_usage_metadata(self, gemini_provider):
    """Test cost estimation with missing usage metadata."""
    incomplete_usage = {
      "totalTokenCount": 1500
      # Missing promptTokenCount and candidatesTokenCount
    }
    
    cost = gemini_provider._estimate_cost(incomplete_usage, "gemini-1.5-pro")
    assert cost is None

  def test_estimate_cost_zero_tokens(self, gemini_provider):
    """Test cost estimation with zero tokens."""
    usage_metadata = {
      "promptTokenCount": 0,
      "candidatesTokenCount": 0,
      "totalTokenCount": 0
    }
    
    cost = gemini_provider._estimate_cost(usage_metadata, "gemini-1.5-pro")
    assert cost == 0.0