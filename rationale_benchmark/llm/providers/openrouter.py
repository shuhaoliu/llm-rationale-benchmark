"""OpenRouter provider implementation for LLM connector."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from rationale_benchmark.llm.exceptions import (
  AuthenticationError,
  ModelNotFoundError,
  NetworkError,
  ProviderError,
  RateLimitError,
  ResponseValidationError,
  StreamingNotSupportedError,
  TimeoutError,
)
from rationale_benchmark.llm.http.client import HTTPClient
from rationale_benchmark.llm.models import ModelRequest, ModelResponse, ProviderConfig
from rationale_benchmark.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
  """OpenRouter API provider implementation with structured output validation.
  
  This provider implements the LLMProvider interface for OpenRouter's OpenAI-compatible API,
  supporting custom base URLs, configurable authentication headers, and strict no-streaming
  validation. OpenRouter provides access to multiple LLM models through a unified interface.
  """

  def __init__(self, config: ProviderConfig, http_client: HTTPClient):
    """Initialize OpenRouter provider with configuration and HTTP client.
    
    Args:
      config: Provider-specific configuration
      http_client: HTTP client for making API requests
    """
    super().__init__(config, http_client)
    self.base_url = config.base_url or "https://openrouter.ai/api/v1"
    
    # OpenRouter supports configurable authentication headers
    self.auth_headers = self._setup_auth_headers()
    
    # OpenRouter uses OpenAI-compatible pricing structure
    # Pricing varies by model and is handled by OpenRouter's billing system
    self._default_headers = {
      "Content-Type": "application/json",
      "HTTP-Referer": "https://github.com/rationale-benchmark/rationale-benchmark",
      "X-Title": "Rationale Benchmark Tool"
    }

  def _map_openrouter_error(self, response, error_data: Optional[Dict[str, Any]] = None) -> Exception:
    """Map OpenRouter API errors to specific exception types with provider-specific guidance.
    
    This method provides comprehensive error mapping for OpenRouter API responses,
    including specific guidance for common issues and authentication problems.
    
    Args:
      response: HTTP response object
      error_data: Parsed error data from response (if available)
      
    Returns:
      Appropriate exception instance with detailed error information
    """
    status_code = response.status
    
    # Extract error details if available
    error_message = "Unknown error"
    error_type = "unknown"
    error_code = None
    
    if error_data and isinstance(error_data, dict):
      if "error" in error_data:
        error_info = error_data["error"]
        if isinstance(error_info, dict):
          error_message = error_info.get("message", error_message)
          error_type = error_info.get("type", error_type)
          error_code = error_info.get("code", error_code)
        elif isinstance(error_info, str):
          error_message = error_info
      elif "message" in error_data:
        error_message = error_data["message"]
        error_type = error_data.get("type", error_type)
    
    # Map specific HTTP status codes to exceptions
    if status_code == 401:
      guidance = (
        "OpenRouter authentication failed. Please check:\n"
        "1. Ensure OPENROUTER_API_KEY environment variable is set\n"
        "2. Verify your API key is valid and active\n"
        "3. Check if your API key has the required permissions\n"
        "4. Ensure your account has sufficient credits\n"
        "5. Verify the API key format is correct"
      )
      return AuthenticationError("openrouter", f"{error_message}. {guidance}")
    
    elif status_code == 402:
      # Payment required - specific to OpenRouter
      guidance = (
        "OpenRouter payment required. Please:\n"
        "1. Check your OpenRouter account balance\n"
        "2. Add credits to your account\n"
        "3. Verify your payment method is valid\n"
        "4. Check if you've exceeded your spending limits\n"
        "5. Review your usage and billing settings"
      )
      return RateLimitError("openrouter", f"Payment required: {error_message}. {guidance}")
    
    elif status_code == 403:
      if "billing" in error_message.lower() or "quota" in error_message.lower():
        guidance = (
          "OpenRouter billing/quota issue. Please:\n"
          "1. Check your OpenRouter account billing status\n"
          "2. Verify you have sufficient credits\n"
          "3. Review your usage limits and quotas\n"
          "4. Consider adding more credits to your account"
        )
        return RateLimitError("openrouter", f"{error_message}. {guidance}")
      else:
        guidance = (
          "OpenRouter access forbidden. This may indicate:\n"
          "1. Your API key lacks required permissions\n"
          "2. The requested model is not available to your account\n"
          "3. Your account may be restricted or suspended\n"
          "4. Geographic restrictions may apply"
        )
        return ProviderError("openrouter", f"{error_message}. {guidance}")
    
    elif status_code == 404:
      if "model" in error_message.lower():
        guidance = (
          "OpenRouter model not found. Please:\n"
          "1. Check the model name spelling\n"
          "2. Verify the model is available on OpenRouter\n"
          "3. Ensure your account has access to the requested model\n"
          "4. Use list_models() to see available models\n"
          "5. Check OpenRouter's model documentation"
        )
        # Try to extract model name from error message
        model_name = error_code or "unknown"
        if "model" in error_message.lower():
          words = error_message.split()
          for i, word in enumerate(words):
            if "model" in word.lower() and i + 1 < len(words):
              model_name = words[i + 1].strip("'\"")
              break
        return ModelNotFoundError("openrouter", model_name)
      else:
        return ProviderError("openrouter", f"Resource not found: {error_message}")
    
    elif status_code == 429:
      # Rate limiting
      retry_after = None
      if hasattr(response, 'headers') and 'retry-after' in response.headers:
        try:
          retry_after = int(response.headers['retry-after'])
        except (ValueError, TypeError):
          pass
      
      if "rate limit" in error_message.lower():
        guidance = (
          "OpenRouter rate limit exceeded. Please:\n"
          "1. Implement exponential backoff in your requests\n"
          "2. Reduce your request frequency\n"
          "3. Check your rate limits in OpenRouter dashboard\n"
          "4. Consider upgrading your plan for higher limits\n"
          "5. Monitor your requests per minute/hour"
        )
        return RateLimitError("openrouter", f"{error_message}. {guidance}", retry_after)
      else:
        guidance = (
          "OpenRouter quota exceeded. Please:\n"
          "1. Check your account usage and limits\n"
          "2. Wait for your quota to reset\n"
          "3. Add more credits to your account\n"
          "4. Monitor your usage to avoid future overages"
        )
        return RateLimitError("openrouter", f"{error_message}. {guidance}", retry_after)
    
    elif status_code == 500:
      guidance = (
        "OpenRouter server error. This is typically temporary:\n"
        "1. Retry your request after a brief delay\n"
        "2. Check OpenRouter status page for service issues\n"
        "3. Implement exponential backoff for retries\n"
        "4. Contact OpenRouter support if the issue persists"
      )
      return ProviderError("openrouter", f"Server error: {error_message}. {guidance}")
    
    elif status_code == 502 or status_code == 503 or status_code == 504:
      guidance = (
        "OpenRouter service temporarily unavailable:\n"
        "1. Retry your request after a delay\n"
        "2. Implement exponential backoff\n"
        "3. Check OpenRouter status page for outages\n"
        "4. Consider using a different model if available"
      )
      return NetworkError(f"OpenRouter service unavailable (HTTP {status_code}): {error_message}. {guidance}")
    
    # Check for streaming-related errors (critical for OpenRouter)
    if "stream" in error_message.lower() or "streaming" in error_message.lower():
      guidance = (
        "Streaming is not supported by this connector:\n"
        "1. Remove any 'stream' parameters from your configuration\n"
        "2. Ensure no streaming options are set in provider_specific section\n"
        "3. This connector only supports complete responses\n"
        "4. OpenRouter streaming is explicitly blocked by this implementation"
      )
      return StreamingNotSupportedError(f"OpenRouter streaming error: {error_message}. {guidance}")
    
    # Check for model-specific errors
    if "model" in error_message.lower() and ("unavailable" in error_message.lower() or "offline" in error_message.lower()):
      guidance = (
        "OpenRouter model temporarily unavailable:\n"
        "1. Try a different model from the same provider\n"
        "2. Check OpenRouter status page for model availability\n"
        "3. Wait and retry later as models may come back online\n"
        "4. Use list_models() to see currently available models"
      )
      return ProviderError("openrouter", f"Model unavailable: {error_message}. {guidance}")
    
    # Generic error with troubleshooting guidance
    guidance = (
      "General OpenRouter API error troubleshooting:\n"
      "1. Check your request parameters and format\n"
      "2. Verify your API key and account status\n"
      "3. Review OpenRouter API documentation\n"
      "4. Check OpenRouter status page for service issues\n"
      "5. Ensure you have sufficient credits"
    )
    
    return ProviderError("openrouter", f"API error (HTTP {status_code}): {error_message}. {guidance}")

  def _setup_auth_headers(self) -> Dict[str, str]:
    """Setup authentication headers for OpenRouter API.
    
    OpenRouter supports multiple authentication methods:
    - Authorization header with Bearer token (default)
    - Custom headers specified in provider_specific config
    
    Returns:
      Dictionary of authentication headers
    """
    auth_headers = {}
    
    # Default Bearer token authentication
    if self.config.api_key:
      auth_headers["Authorization"] = f"Bearer {self.config.api_key}"
    
    # Support custom authentication headers from provider_specific config
    provider_specific = getattr(self.config, 'provider_specific', {})
    custom_headers = provider_specific.get('auth_headers', {})
    
    if isinstance(custom_headers, dict):
      for key, value in custom_headers.items():
        if key.lower() not in ['authorization'] or not auth_headers.get('Authorization'):
          auth_headers[key] = value
    
    return auth_headers

  async def generate_response(self, request: ModelRequest) -> ModelResponse:
    """Generate response using OpenRouter API with no streaming and strict validation.
    
    Args:
      request: Standardized request parameters
      
    Returns:
      Standardized response from OpenRouter
      
    Raises:
      AuthenticationError: For invalid API keys
      ModelNotFoundError: For unavailable models
      ProviderError: For other API errors
      ResponseValidationError: For invalid response structure
      StreamingNotSupportedError: If streaming parameters are detected
    """
    headers = {**self._default_headers, **self.auth_headers}
    
    payload = self._prepare_request(request)
    # Explicitly disable streaming - this is critical for OpenRouter
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
      raise ProviderError("openrouter", f"Request failed: {str(e)}", e)
    
    latency_ms = self._measure_latency(start_time)
    
    # Validate response structure BEFORE parsing
    self._validate_response_structure(response_data)
    
    # Parse response after validation
    parsed_response = self._parse_response(response_data, request, latency_ms)
    
    logger.debug(
      f"OpenRouter response generated successfully for model {request.model} "
      f"in {latency_ms}ms"
    )
    
    return parsed_response

  async def list_models(self) -> List[str]:
    """List available models for OpenRouter provider.
    
    Returns:
      List of model names available from OpenRouter
      
    Raises:
      ProviderError: For API errors
    """
    headers = {**self._default_headers, **self.auth_headers}
    
    try:
      response = await self.http_client.get(
        f"{self.base_url}/models",
        headers=headers
      )
      
      if response.status >= 400:
        await self._handle_api_error(response)
      
      response_data = await response.json()
      
      # Extract model IDs from OpenAI-compatible response
      if "data" in response_data and isinstance(response_data["data"], list):
        models = [model["id"] for model in response_data["data"] if "id" in model]
        return sorted(models)
      else:
        raise ProviderError("openrouter", "Invalid models list response format")
        
    except Exception as e:
      if isinstance(e, ProviderError):
        raise
      raise ProviderError("openrouter", f"Failed to list models: {str(e)}", e)

  def validate_config(self) -> List[str]:
    """Validate OpenRouter provider configuration.
    
    Returns:
      List of validation error messages (empty if valid)
    """
    errors = []
    
    if not self.config.api_key:
      errors.append("OpenRouter API key is required")
    
    if self.config.base_url and not self.config.base_url.startswith("https://"):
      errors.append("OpenRouter base URL should use HTTPS")
    
    if self.config.timeout <= 0:
      errors.append("Timeout must be positive")
    
    if self.config.max_retries < 0:
      errors.append("Max retries cannot be negative")
    
    # Validate custom authentication headers if present
    provider_specific = getattr(self.config, 'provider_specific', {})
    custom_headers = provider_specific.get('auth_headers', {})
    
    if custom_headers and not isinstance(custom_headers, dict):
      errors.append("Custom auth_headers must be a dictionary")
    
    # Check for streaming parameters in configuration
    streaming_params = {
      "stream", "streaming", "stream_options", "stream_usage",
      "stream_callback", "stream_handler", "incremental"
    }
    
    found_streaming_params = []
    for key in provider_specific:
      if key in streaming_params:
        found_streaming_params.append(key)
    
    if found_streaming_params:
      errors.append(
        f"Streaming parameters not supported in OpenRouter configuration: {found_streaming_params}"
      )
    
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
    
    # Add provider-specific parameters (excluding streaming and base params)
    streaming_params = {
      "stream", "streaming", "stream_options", "stream_usage", 
      "stream_callback", "stream_handler", "incremental"
    }
    
    for key, value in request.provider_specific.items():
      if key not in streaming_params and key not in payload:
        # Validate parameter before adding
        if self._is_valid_openrouter_parameter(key, value):
          payload[key] = value
        else:
          logger.warning(f"Skipped invalid OpenRouter parameter '{key}': {value}")
      elif key in payload:
        # Don't override base parameters
        logger.warning(f"Skipped provider_specific parameter '{key}' that would override base parameter")
    
    # Use base class streaming detection for comprehensive validation
    self._detect_streaming_parameters(request, payload)
    
    # Additional OpenRouter-specific validation
    self._validate_openrouter_request(payload)
    
    logger.debug(f"Prepared OpenRouter request for {request.model} with {len(payload)} parameters")
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
    
    # OpenRouter may include additional metadata
    openrouter_metadata = {
      "prompt_tokens": usage["prompt_tokens"],
      "completion_tokens": usage["completion_tokens"],
      "finish_reason": choice["finish_reason"],
      "choice_index": choice["index"]
    }
    
    # Add OpenRouter-specific metadata if present
    if "id" in response_data:
      openrouter_metadata["request_id"] = response_data["id"]
    
    if "created" in response_data:
      openrouter_metadata["created"] = response_data["created"]
    
    # OpenRouter may include provider information
    if "provider" in response_data:
      openrouter_metadata["upstream_provider"] = response_data["provider"]
    
    return ModelResponse(
      text=message["content"],
      model=response_data["model"],
      provider="openrouter",
      timestamp=datetime.now(),
      latency_ms=latency_ms,
      token_count=usage["total_tokens"],
      finish_reason=choice["finish_reason"],
      cost_estimate=self._estimate_cost(usage, response_data["model"]),
      metadata=openrouter_metadata
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
          f"Missing required field '{field}' in OpenRouter response",
          provider="openrouter",
          response_data=response_data
        )
    
    # Validate object type
    if response_data["object"] != "chat.completion":
      raise ResponseValidationError(
        f"Invalid object type in OpenRouter response: expected 'chat.completion', "
        f"got '{response_data['object']}'",
        provider="openrouter",
        response_data=response_data
      )
    
    # Validate choices array
    if not isinstance(response_data["choices"], list) or not response_data["choices"]:
      raise ResponseValidationError(
        "OpenRouter response 'choices' must be a non-empty array",
        provider="openrouter",
        response_data=response_data
      )
    
    # Validate first choice structure (we only use the first choice)
    choice = response_data["choices"][0]
    required_choice_fields = ["message", "finish_reason", "index"]
    for field in required_choice_fields:
      if field not in choice:
        raise ResponseValidationError(
          f"Missing required field '{field}' in OpenRouter choice",
          provider="openrouter",
          response_data=response_data
        )
    
    # Validate choice index
    if not isinstance(choice["index"], int) or choice["index"] < 0:
      raise ResponseValidationError(
        f"Invalid choice index in OpenRouter response: {choice['index']}",
        provider="openrouter",
        response_data=response_data
      )
    
    # Validate message structure
    message = choice["message"]
    if not isinstance(message, dict):
      raise ResponseValidationError(
        "OpenRouter response message must be a dictionary",
        provider="openrouter",
        response_data=response_data
      )
    
    required_message_fields = ["content", "role"]
    for field in required_message_fields:
      if field not in message:
        raise ResponseValidationError(
          f"Missing required field '{field}' in OpenRouter message",
          provider="openrouter",
          response_data=response_data
        )
    
    # Validate message role
    if message["role"] != "assistant":
      raise ResponseValidationError(
        f"Invalid message role in OpenRouter response: expected 'assistant', "
        f"got '{message['role']}'",
        provider="openrouter",
        response_data=response_data
      )
    
    # Validate content is not empty
    if not message["content"] or not isinstance(message["content"], str):
      raise ResponseValidationError(
        "OpenRouter response message content must be a non-empty string",
        provider="openrouter",
        response_data=response_data
      )
    
    if len(message["content"].strip()) == 0:
      raise ResponseValidationError(
        "OpenRouter response message content cannot be empty or whitespace only",
        provider="openrouter",
        response_data=response_data
      )
    
    # Validate finish reason
    valid_finish_reasons = ["stop", "length", "function_call", "content_filter", "null"]
    if choice["finish_reason"] not in valid_finish_reasons:
      logger.warning(f"Unexpected finish reason in OpenRouter response: {choice['finish_reason']}")
    
    # Validate usage information
    usage = response_data["usage"]
    if not isinstance(usage, dict):
      raise ResponseValidationError(
        "OpenRouter usage must be a dictionary",
        provider="openrouter",
        response_data=response_data
      )
    
    required_usage_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
    for field in required_usage_fields:
      if field not in usage:
        raise ResponseValidationError(
          f"Missing usage field '{field}' in OpenRouter response",
          provider="openrouter",
          response_data=response_data
        )
      if not isinstance(usage[field], int) or usage[field] < 0:
        raise ResponseValidationError(
          f"OpenRouter usage field '{field}' must be a non-negative integer, got {usage[field]}",
          provider="openrouter",
          response_data=response_data
        )
    
    # Validate token count consistency
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
      raise ResponseValidationError(
        f"OpenRouter token count inconsistency: total={usage['total_tokens']}, "
        f"sum={usage['prompt_tokens'] + usage['completion_tokens']}",
        provider="openrouter",
        response_data=response_data
      )
    
    # Validate model field
    if not isinstance(response_data["model"], str) or not response_data["model"]:
      raise ResponseValidationError(
        "OpenRouter response model must be a non-empty string",
        provider="openrouter",
        response_data=response_data
      )
    
    # Additional validation for streaming indicators (should never be present)
    if "stream" in response_data and response_data["stream"]:
      raise ResponseValidationError(
        "OpenRouter response indicates streaming mode, but streaming is not supported",
        provider="openrouter",
        response_data=response_data
      )
    
    logger.debug(f"OpenRouter response structure validation passed for model {response_data['model']}")

  def _is_valid_openrouter_parameter(self, key: str, value: Any) -> bool:
    """Validate that a parameter is valid for OpenRouter API.
    
    OpenRouter supports OpenAI-compatible parameters plus some additional ones.
    
    Args:
      key: Parameter name
      value: Parameter value
      
    Returns:
      True if parameter is valid for OpenRouter API
    """
    # OpenAI-compatible parameters
    openai_params = {
      "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
      "max_tokens", "n", "presence_penalty", "response_format",
      "seed", "stop", "temperature", "top_p", "tools", "tool_choice",
      "user", "function_call", "functions"
    }
    
    # OpenRouter-specific parameters
    openrouter_params = {
      "provider", "route", "models", "fallbacks"
    }
    
    valid_params = openai_params | openrouter_params
    return key in valid_params

  def _validate_openrouter_request(self, payload: Dict[str, Any]) -> None:
    """Validate OpenRouter-specific request requirements.
    
    Args:
      payload: Request payload to validate
      
    Raises:
      StreamingNotSupportedError: If streaming is detected
      ProviderError: For other validation errors
    """
    # Ensure no streaming parameters are present
    streaming_indicators = ["stream", "streaming", "stream_options"]
    for indicator in streaming_indicators:
      if payload.get(indicator):
        raise StreamingNotSupportedError(
          f"OpenRouter does not support streaming: {indicator}={payload[indicator]}"
        )
    
    # Validate model is specified
    if not payload.get("model"):
      raise ProviderError("openrouter", "Model must be specified for OpenRouter requests")
    
    # Validate messages format
    messages = payload.get("messages", [])
    if not messages or not isinstance(messages, list):
      raise ProviderError("openrouter", "Messages must be a non-empty list")

  def _estimate_cost(self, usage: Dict[str, int], model: str) -> Optional[float]:
    """Estimate cost for the request based on token usage and model.
    
    OpenRouter handles pricing dynamically based on the upstream provider
    and model. We cannot provide accurate cost estimates without access
    to OpenRouter's pricing API.
    
    Args:
      usage: Token usage information from OpenRouter response
      model: Model name used for the request
      
    Returns:
      None (cost estimation not available for OpenRouter)
    """
    # OpenRouter pricing is dynamic and model-dependent
    # Cost estimation would require additional API calls to OpenRouter's pricing endpoint
    return None

  async def _handle_api_error(
    self, 
    response, 
    model: Optional[str] = None
  ) -> None:
    """Handle API error responses with comprehensive error mapping and guidance.
    
    This method uses the provider-specific error mapping to provide detailed
    error information and troubleshooting guidance for OpenRouter API errors.
    
    Args:
      response: HTTP response object
      model: Model name for context (optional)
      
    Raises:
      Various specific exceptions based on error type and status code
    """
    try:
      error_data = await response.json()
    except Exception:
      error_data = None
    
    # Use the comprehensive error mapping
    exception = self._map_openrouter_error(response, error_data)
    
    # Add model context to ModelNotFoundError if available
    if isinstance(exception, ModelNotFoundError) and model:
      exception.model = model
    
    logger.error(
      f"OpenRouter API error (HTTP {response.status}): {exception}",
      extra={
        "provider": "openrouter",
        "status_code": response.status,
        "model": model,
        "error_type": type(exception).__name__
      }
    )
    
    raise exception