"""OpenAI provider implementation for LLM connector."""

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
from rationale_benchmark.llm.validation import ResponseValidator
from rationale_benchmark.llm.logging import get_llm_logger

logger = get_llm_logger(__name__, "openai")


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

  def _map_openai_error(self, response, error_data: Optional[Dict[str, Any]] = None) -> Exception:
    """Map OpenAI API errors to specific exception types with provider-specific guidance.
    
    This method provides comprehensive error mapping for OpenAI API responses,
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
    
    # Map specific HTTP status codes to exceptions
    if status_code == 401:
      guidance = (
        "OpenAI authentication failed. Please check:\n"
        "1. Ensure OPENAI_API_KEY environment variable is set\n"
        "2. Verify your API key is valid and active\n"
        "3. Check if your API key has the required permissions\n"
        "4. Ensure your account has sufficient credits"
      )
      return AuthenticationError("openai", f"{error_message}. {guidance}")
    
    elif status_code == 403:
      if "billing" in error_message.lower() or "quota" in error_message.lower():
        guidance = (
          "OpenAI billing/quota issue. Please:\n"
          "1. Check your OpenAI account billing status\n"
          "2. Verify you have sufficient credits\n"
          "3. Review your usage limits and quotas\n"
          "4. Consider upgrading your plan if needed"
        )
        return AuthenticationError("openai", f"{error_message}. {guidance}")
      else:
        guidance = (
          "OpenAI access forbidden. This may indicate:\n"
          "1. Your API key lacks required permissions\n"
          "2. The requested model is not available to your account\n"
          "3. Your account may be restricted or suspended"
        )
        return ProviderError("openai", f"{error_message}. {guidance}")
    
    elif status_code == 404:
      if "model" in error_message.lower():
        guidance = (
          "OpenAI model not found. Please:\n"
          "1. Check the model name spelling\n"
          "2. Verify the model is available in your region\n"
          "3. Ensure your account has access to the requested model\n"
          "4. Use list_models() to see available models"
        )
        model_name = error_code or "unknown"
        return ModelNotFoundError("openai", model_name)
      else:
        return ProviderError("openai", f"Resource not found: {error_message}")
    
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
          "OpenAI rate limit exceeded. Please:\n"
          "1. Implement exponential backoff in your requests\n"
          "2. Reduce your request frequency\n"
          "3. Consider upgrading to a higher tier plan\n"
          "4. Distribute requests across multiple API keys if allowed"
        )
        return RateLimitError("openai", f"{error_message}. {guidance}", retry_after)
      else:
        guidance = (
          "OpenAI quota exceeded. Please:\n"
          "1. Check your account usage and limits\n"
          "2. Wait for your quota to reset\n"
          "3. Consider upgrading your plan\n"
          "4. Monitor your usage to avoid future overages"
        )
        return RateLimitError("openai", f"{error_message}. {guidance}", retry_after)
    
    elif status_code == 500:
      guidance = (
        "OpenAI server error. This is typically temporary:\n"
        "1. Retry your request after a brief delay\n"
        "2. Check OpenAI status page for service issues\n"
        "3. Implement exponential backoff for retries\n"
        "4. Contact OpenAI support if the issue persists"
      )
      return ProviderError("openai", f"Server error: {error_message}. {guidance}")
    
    elif status_code == 502 or status_code == 503 or status_code == 504:
      guidance = (
        "OpenAI service temporarily unavailable:\n"
        "1. Retry your request after a delay\n"
        "2. Implement exponential backoff\n"
        "3. Check OpenAI status page for outages\n"
        "4. Consider using a different model if available"
      )
      return NetworkError(f"OpenAI service unavailable (HTTP {status_code}): {error_message}. {guidance}")
    
    # Check for streaming-related errors
    if "stream" in error_message.lower() or "streaming" in error_message.lower():
      guidance = (
        "Streaming is not supported by this connector:\n"
        "1. Remove any 'stream' parameters from your configuration\n"
        "2. Ensure no streaming options are set in provider_specific section\n"
        "3. This connector only supports complete responses"
      )
      return StreamingNotSupportedError(f"OpenAI streaming error: {error_message}. {guidance}")
    
    # Generic error with troubleshooting guidance
    guidance = (
      "General OpenAI API error troubleshooting:\n"
      "1. Check your request parameters and format\n"
      "2. Verify your API key and account status\n"
      "3. Review OpenAI API documentation for requirements\n"
      "4. Check OpenAI status page for service issues"
    )
    
    return ProviderError("openai", f"API error (HTTP {status_code}): {error_message}. {guidance}")

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
    
    # Log the request with context
    logger.log_request(
      method="POST",
      url=f"{self.base_url}/chat/completions",
      model=request.model,
      request_data=payload,
      temperature=request.temperature,
      max_tokens=request.max_tokens,
    )
    
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
      latency_ms = self._measure_latency(start_time)
      logger.log_response(
        status_code=getattr(response, 'status', 0) if 'response' in locals() else 0,
        model=request.model,
        latency_ms=latency_ms,
        error=str(e),
      )
      
      if isinstance(e, (AuthenticationError, ModelNotFoundError, ProviderError)):
        raise
      raise ProviderError("openai", f"Request failed: {str(e)}", e)
    
    latency_ms = self._measure_latency(start_time)
    
    # Validate response structure BEFORE parsing
    try:
      self._validate_response_structure(response_data)
    except ResponseValidationError as e:
      logger.log_validation_error(
        error_type="response_structure",
        field_errors=e.field_errors,
        provider="openai",
        model=request.model,
        latency_ms=latency_ms,
      )
      raise
    
    # Parse response after validation
    parsed_response = self._parse_response(response_data, request, latency_ms)
    
    # Log successful response
    logger.log_response(
      status_code=response.status,
      model=request.model,
      latency_ms=latency_ms,
      response_data=response_data,
      token_count=parsed_response.token_count,
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
    
    # Add provider-specific parameters (excluding streaming)
    streaming_params = {
      "stream", "streaming", "stream_options", "stream_usage", 
      "stream_callback", "stream_handler", "incremental"
    }
    
    for key, value in request.provider_specific.items():
      if key not in streaming_params:
        # Validate parameter before adding
        if self._is_valid_openai_parameter(key, value):
          payload[key] = value
        else:
          logger.warning(f"Skipped invalid OpenAI parameter '{key}': {value}")
    
    # Use base class streaming detection for comprehensive validation
    self._detect_streaming_parameters(request, payload)
    
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
    """Comprehensive validation of OpenAI response structure using enhanced validator.
    
    This method uses the ResponseValidator utility to perform exhaustive validation
    of the OpenAI API response with detailed error reporting and recovery suggestions.
    
    Args:
      response_data: Raw response dictionary from OpenAI API
      
    Raises:
      ResponseValidationError: If any validation check fails with detailed context
    """
    # Validate response is not empty first
    self._validate_response_not_empty(response_data)
    
    # Use the enhanced ResponseValidator for comprehensive validation
    validator = ResponseValidator("openai")
    
    try:
      validator.validate_openai_response(response_data)
      
      # Additional validation for streaming indicators (should never be present)
      if "stream" in response_data and response_data["stream"]:
        error = validator.create_validation_error_with_context(
          "OpenAI response indicates streaming mode, but streaming is not supported",
          response_data,
          {"streaming_detected": True, "stream_value": response_data["stream"]}
        )
        error.add_recovery_suggestion("Ensure stream parameter is set to False in API requests")
        error.add_recovery_suggestion("Check request preparation logic for streaming parameter removal")
        raise error
      
      logger.debug(f"OpenAI response structure validation passed for model {response_data.get('model', 'unknown')}")
      
    except ResponseValidationError as e:
      # Add OpenAI-specific context and recovery suggestions if not already present
      if not e.recovery_suggestions:
        e.add_recovery_suggestion("Check OpenAI API documentation for correct response format")
        e.add_recovery_suggestion("Verify API key has proper permissions for the requested model")
        e.add_recovery_suggestion("Ensure request parameters match OpenAI API requirements")
      
      # Add additional context about the validation failure
      if not e.validation_context.get("model"):
        e.validation_context["model"] = response_data.get("model", "unknown")
      
      raise e

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
    """Handle API error responses with comprehensive error mapping and guidance.
    
    This method uses the provider-specific error mapping to provide detailed
    error information and troubleshooting guidance for OpenAI API errors.
    
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
    exception = self._map_openai_error(response, error_data)
    
    # Add model context to ModelNotFoundError if available
    if isinstance(exception, ModelNotFoundError) and model:
      exception.model = model
    
    logger.error(
      f"OpenAI API error (HTTP {response.status}): {exception}",
      extra={
        "provider": "openai",
        "status_code": response.status,
        "model": model,
        "error_type": type(exception).__name__
      }
    )
    
    raise exception