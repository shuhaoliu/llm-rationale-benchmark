"""OpenAI provider implementation for LLM connector."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from rationale_benchmark.llm.exceptions import (
  AuthenticationError,
  ModelNotFoundError,
  ProviderError,
  ResponseValidationError,
  StreamingNotSupportedError,
)
from rationale_benchmark.llm.http.client import HTTPClient
from rationale_benchmark.llm.models import ModelRequest, ModelResponse, ProviderConfig
from rationale_benchmark.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
  """OpenAI API provider implementation with structured output validation.
  
  This provider implements the LLMProvider interface for OpenAI's API,
  supporting GPT-4, GPT-3.5-turbo, and other OpenAI models with comprehensive
  response validation and explicit streaming prevention.
  """

  def __init__(self, config: ProviderConfig, http_client: HTTPClient):
    """Initialize OpenAI provider with configuration and HTTP client.
    
    Args:
      config: Provider-specific configuration
      http_client: HTTP client for making API requests
    """
    super().__init__(config, http_client)
    self.base_url = config.base_url or "https://api.openai.com/v1"
    
    # OpenAI-specific model pricing (per 1K tokens) - approximate values
    self._model_pricing = {
      "gpt-4": {"prompt": 0.03, "completion": 0.06},
      "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
      "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
      "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
      "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
      "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
      "text-davinci-003": {"prompt": 0.02, "completion": 0.02},
      "text-davinci-002": {"prompt": 0.02, "completion": 0.02},
    }

  async def generate_response(self, request: ModelRequest) -> ModelResponse:
    """Generate response using OpenAI API with no streaming and strict validation.
    
    Args:
      request: Standardized request parameters
      
    Returns:
      Standardized response from OpenAI
      
    Raises:
      AuthenticationError: For invalid API keys
      ModelNotFoundError: For unavailable models
      ProviderError: For other API errors
      ResponseValidationError: For invalid response structure
      StreamingNotSupportedError: If streaming parameters are detected
    """
    headers = {
      "Authorization": f"Bearer {self.config.api_key}",
      "Content-Type": "application/json"
    }
    
    payload = self._prepare_request(request)
    # Explicitly disable streaming - this is critical
    payload["stream"] = False
    
    start_time = time.time()
    
    try:
      response = await self.http_client.post(
        f"{self.base_url}/chat/completions",
        json=payload,
        headers=headers
      )
      
      # Handle HTTP errors
      if response.status >= 400:
        await self._handle_api_error(response, request.model)
      
      response_data = await response.json()
      
    except Exception as e:
      if isinstance(e, (AuthenticationError, ModelNotFoundError, ProviderError)):
        raise
      raise ProviderError("openai", f"Request failed: {str(e)}", e)
    
    latency_ms = self._measure_latency(start_time)
    
    # Validate response structure BEFORE parsing
    self._validate_response_structure(response_data)
    
    # Parse response after validation
    parsed_response = self._parse_response(response_data, request, latency_ms)
    
    logger.debug(
      f"OpenAI response generated successfully for model {request.model} "
      f"in {latency_ms}ms"
    )
    
    return parsed_response

  async def list_models(self) -> List[str]:
    """List available models for OpenAI provider.
    
    Returns:
      List of model names available from OpenAI
      
    Raises:
      ProviderError: For API errors
    """
    headers = {
      "Authorization": f"Bearer {self.config.api_key}",
      "Content-Type": "application/json"
    }
    
    try:
      response = await self.http_client.get(
        f"{self.base_url}/models",
        headers=headers
      )
      
      if response.status >= 400:
        await self._handle_api_error(response)
      
      response_data = await response.json()
      
      # Extract model IDs from response
      if "data" in response_data and isinstance(response_data["data"], list):
        models = [model["id"] for model in response_data["data"] if "id" in model]
        return sorted(models)
      else:
        raise ProviderError("openai", "Invalid models list response format")
        
    except Exception as e:
      if isinstance(e, ProviderError):
        raise
      raise ProviderError("openai", f"Failed to list models: {str(e)}", e)

  def validate_config(self) -> List[str]:
    """Validate OpenAI provider configuration.
    
    Returns:
      List of validation error messages (empty if valid)
    """
    errors = []
    
    if not self.config.api_key:
      errors.append("OpenAI API key is required")
    elif not self.config.api_key.startswith("sk-"):
      errors.append("OpenAI API key should start with 'sk-'")
    
    if self.config.base_url and not self.config.base_url.startswith("https://"):
      errors.append("OpenAI base URL should use HTTPS")
    
    if self.config.timeout <= 0:
      errors.append("Timeout must be positive")
    
    if self.config.max_retries < 0:
      errors.append("Max retries cannot be negative")
    
    return errors

  def _prepare_request(self, request: ModelRequest) -> Dict[str, Any]:
    """Convert ModelRequest to OpenAI format, ensuring no streaming.
    
    This method creates the request payload for OpenAI API with strict
    streaming prevention and comprehensive parameter validation.
    
    Args:
      request: The ModelRequest to convert
      
    Returns:
      Dict containing the OpenAI API request payload
      
    Raises:
      StreamingNotSupportedError: If streaming parameters are detected
    """
    # Build base payload with explicit streaming disabled
    payload = {
      "model": request.model,
      "messages": [{"role": "user", "content": request.prompt}],
      "temperature": request.temperature,
      "max_tokens": request.max_tokens,
      "stream": False  # Explicitly disable streaming - CRITICAL requirement
    }
    
    # Add system prompt if provided
    if request.system_prompt:
      payload["messages"].insert(0, {"role": "system", "content": request.system_prompt})
    
    # Add stop sequences if provided
    if request.stop_sequences:
      payload["stop"] = request.stop_sequences
    
    # Strictly filter out ALL streaming-related parameters
    streaming_params = {
      "stream", "streaming", "stream_options", "stream_usage", 
      "stream_callback", "stream_handler", "incremental"
    }
    
    blocked_params = []
    for key, value in request.provider_specific.items():
      if key in streaming_params:
        blocked_params.append(key)
        logger.warning(f"Blocked streaming parameter '{key}' in OpenAI request")
      else:
        # Validate parameter before adding
        if self._is_valid_openai_parameter(key, value):
          payload[key] = value
        else:
          logger.warning(f"Skipped invalid OpenAI parameter '{key}': {value}")
    
    # Raise error if streaming was attempted
    if blocked_params:
      raise StreamingNotSupportedError(
        f"Streaming parameters not supported: {blocked_params}",
        blocked_params=blocked_params
      )
    
    # Final validation that no streaming is enabled
    if payload.get("stream", False):
      raise StreamingNotSupportedError("Stream parameter cannot be True")
    
    logger.debug(f"Prepared OpenAI request for {request.model} with {len(payload)} parameters")
    return payload

  def _parse_response(
    self, 
    response_data: Dict[str, Any], 
    request: ModelRequest,
    latency_ms: int
  ) -> ModelResponse:
    """Parse OpenAI response into standardized ModelResponse.
    
    Args:
      response_data: Raw response from OpenAI API
      request: Original request for context
      latency_ms: Request latency in milliseconds
      
    Returns:
      Standardized ModelResponse
    """
    choice = response_data["choices"][0]
    message = choice["message"]
    usage = response_data["usage"]
    
    return ModelResponse(
      text=message["content"],
      model=response_data["model"],
      provider="openai",
      timestamp=datetime.now(),
      latency_ms=latency_ms,
      token_count=usage["total_tokens"],
      finish_reason=choice["finish_reason"],
      cost_estimate=self._estimate_cost(usage, response_data["model"]),
      metadata={
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "finish_reason": choice["finish_reason"],
        "choice_index": choice["index"]
      }
    )

  def _validate_response_structure(self, response_data: Dict[str, Any]) -> None:
    """Comprehensive validation of OpenAI response structure.
    
    This method performs exhaustive validation of the OpenAI API response
    to ensure all required fields are present and properly formatted.
    
    Args:
      response_data: Raw response dictionary from OpenAI API
      
    Raises:
      ResponseValidationError: If any validation check fails
    """
    # Validate response is not empty
    self._validate_response_not_empty(response_data)
    
    # Check top-level required fields
    required_fields = ["choices", "model", "usage", "object"]
    for field in required_fields:
      if field not in response_data:
        raise ResponseValidationError(
          f"Missing required field '{field}' in OpenAI response",
          provider="openai",
          response_data=response_data
        )
    
    # Validate object type
    if response_data["object"] != "chat.completion":
      raise ResponseValidationError(
        f"Invalid object type in OpenAI response: expected 'chat.completion', "
        f"got '{response_data['object']}'",
        provider="openai",
        response_data=response_data
      )
    
    # Validate choices array
    if not isinstance(response_data["choices"], list) or not response_data["choices"]:
      raise ResponseValidationError(
        "OpenAI response 'choices' must be a non-empty array",
        provider="openai",
        response_data=response_data
      )
    
    # Validate first choice structure (we only use the first choice)
    choice = response_data["choices"][0]
    required_choice_fields = ["message", "finish_reason", "index"]
    for field in required_choice_fields:
      if field not in choice:
        raise ResponseValidationError(
          f"Missing required field '{field}' in OpenAI choice",
          provider="openai",
          response_data=response_data
        )
    
    # Validate choice index
    if not isinstance(choice["index"], int) or choice["index"] < 0:
      raise ResponseValidationError(
        f"Invalid choice index in OpenAI response: {choice['index']}",
        provider="openai",
        response_data=response_data
      )
    
    # Validate message structure
    message = choice["message"]
    if not isinstance(message, dict):
      raise ResponseValidationError(
        "OpenAI response message must be a dictionary",
        provider="openai",
        response_data=response_data
      )
    
    required_message_fields = ["content", "role"]
    for field in required_message_fields:
      if field not in message:
        raise ResponseValidationError(
          f"Missing required field '{field}' in OpenAI message",
          provider="openai",
          response_data=response_data
        )
    
    # Validate message role
    if message["role"] != "assistant":
      raise ResponseValidationError(
        f"Invalid message role in OpenAI response: expected 'assistant', "
        f"got '{message['role']}'",
        provider="openai",
        response_data=response_data
      )
    
    # Validate content is not empty
    if not message["content"] or not isinstance(message["content"], str):
      raise ResponseValidationError(
        "OpenAI response message content must be a non-empty string",
        provider="openai",
        response_data=response_data
      )
    
    if len(message["content"].strip()) == 0:
      raise ResponseValidationError(
        "OpenAI response message content cannot be empty or whitespace only",
        provider="openai",
        response_data=response_data
      )
    
    # Validate finish reason
    valid_finish_reasons = ["stop", "length", "function_call", "content_filter", "null"]
    if choice["finish_reason"] not in valid_finish_reasons:
      logger.warning(f"Unexpected finish reason in OpenAI response: {choice['finish_reason']}")
    
    # Validate usage information
    usage = response_data["usage"]
    if not isinstance(usage, dict):
      raise ResponseValidationError(
        "OpenAI usage must be a dictionary",
        provider="openai",
        response_data=response_data
      )
    
    required_usage_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
    for field in required_usage_fields:
      if field not in usage:
        raise ResponseValidationError(
          f"Missing usage field '{field}' in OpenAI response",
          provider="openai",
          response_data=response_data
        )
      if not isinstance(usage[field], int) or usage[field] < 0:
        raise ResponseValidationError(
          f"OpenAI usage field '{field}' must be a non-negative integer, got {usage[field]}",
          provider="openai",
          response_data=response_data
        )
    
    # Validate token count consistency
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
      raise ResponseValidationError(
        f"OpenAI token count inconsistency: total={usage['total_tokens']}, "
        f"sum={usage['prompt_tokens'] + usage['completion_tokens']}",
        provider="openai",
        response_data=response_data
      )
    
    # Validate model field
    if not isinstance(response_data["model"], str) or not response_data["model"]:
      raise ResponseValidationError(
        "OpenAI response model must be a non-empty string",
        provider="openai",
        response_data=response_data
      )
    
    # Additional validation for streaming indicators (should never be present)
    if "stream" in response_data and response_data["stream"]:
      raise ResponseValidationError(
        "OpenAI response indicates streaming mode, but streaming is not supported",
        provider="openai",
        response_data=response_data
      )
    
    logger.debug(f"OpenAI response structure validation passed for model {response_data['model']}")

  def _is_valid_openai_parameter(self, key: str, value: Any) -> bool:
    """Validate that a parameter is valid for OpenAI API.
    
    Args:
      key: Parameter name
      value: Parameter value
      
    Returns:
      True if parameter is valid for OpenAI API
    """
    # List of known valid OpenAI parameters
    valid_params = {
      "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
      "max_tokens", "n", "presence_penalty", "response_format",
      "seed", "stop", "temperature", "top_p", "tools", "tool_choice",
      "user", "function_call", "functions"
    }
    
    return key in valid_params

  def _estimate_cost(self, usage: Dict[str, int], model: str) -> Optional[float]:
    """Estimate cost for the request based on token usage and model.
    
    Args:
      usage: Token usage information from OpenAI response
      model: Model name used for the request
      
    Returns:
      Estimated cost in USD, or None if model pricing is unknown
    """
    if model not in self._model_pricing:
      return None
    
    pricing = self._model_pricing[model]
    prompt_cost = (usage["prompt_tokens"] / 1000) * pricing["prompt"]
    completion_cost = (usage["completion_tokens"] / 1000) * pricing["completion"]
    
    return prompt_cost + completion_cost

  async def _handle_api_error(
    self, 
    response, 
    model: Optional[str] = None
  ) -> None:
    """Handle API error responses with specific error types.
    
    Args:
      response: HTTP response object
      model: Model name for context (optional)
      
    Raises:
      AuthenticationError: For 401 errors
      ModelNotFoundError: For 404 errors with model context
      ProviderError: For other API errors
    """
    try:
      error_data = await response.json()
      error_info = error_data.get("error", {})
      error_message = error_info.get("message", f"HTTP {response.status}")
      error_type = error_info.get("type", "unknown")
      error_code = error_info.get("code", "unknown")
    except Exception:
      # If we can't parse error details, use the status code
      error_message = f"HTTP {response.status}"
      error_type = "unknown"
      error_code = "unknown"
    
    # Handle specific error types
    if response.status == 401:
      raise AuthenticationError("openai", error_message)
    elif response.status == 404 and model and "model" in error_message.lower():
      raise ModelNotFoundError("openai", model)
    else:
      raise ProviderError("openai", f"API request failed: {error_message}")