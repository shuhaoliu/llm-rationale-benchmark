"""Anthropic API provider implementation with structured output validation."""

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


class AnthropicProvider(LLMProvider):
  """Anthropic API provider implementation with structured output validation.
  
  This provider implements the LLMProvider interface for Anthropic's Claude models,
  ensuring no streaming support and comprehensive response validation.
  """

  def __init__(self, config: ProviderConfig, http_client: HTTPClient):
    """Initialize Anthropic provider with configuration and HTTP client.
    
    Args:
      config: Provider configuration including API key and settings
      http_client: HTTP client for making API requests
    """
    super().__init__(config, http_client)
    self.base_url = config.base_url or "https://api.anthropic.com"

  def _map_anthropic_error(self, response, error_data: Optional[Dict[str, Any]] = None) -> Exception:
    """Map Anthropic API errors to specific exception types with provider-specific guidance.
    
    This method provides comprehensive error mapping for Anthropic API responses,
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
    
    if error_data and isinstance(error_data, dict):
      if "error" in error_data:
        error_info = error_data["error"]
        if isinstance(error_info, dict):
          error_message = error_info.get("message", error_message)
          error_type = error_info.get("type", error_type)
        elif isinstance(error_info, str):
          error_message = error_info
      elif "message" in error_data:
        error_message = error_data["message"]
        error_type = error_data.get("type", error_type)
    
    # Map specific HTTP status codes to exceptions
    if status_code == 401:
      guidance = (
        "Anthropic authentication failed. Please check:\n"
        "1. Ensure ANTHROPIC_API_KEY environment variable is set\n"
        "2. Verify your API key starts with 'sk-ant-api'\n"
        "3. Check if your API key is valid and active\n"
        "4. Ensure your account has sufficient credits"
      )
      return AuthenticationError("anthropic", f"{error_message}. {guidance}")
    
    elif status_code == 403:
      if "billing" in error_message.lower() or "quota" in error_message.lower():
        guidance = (
          "Anthropic billing/quota issue. Please:\n"
          "1. Check your Anthropic account billing status\n"
          "2. Verify you have sufficient credits\n"
          "3. Review your usage limits and quotas\n"
          "4. Consider upgrading your plan if needed"
        )
        return AuthenticationError("anthropic", f"{error_message}. {guidance}")
      else:
        guidance = (
          "Anthropic access forbidden. This may indicate:\n"
          "1. Your API key lacks required permissions\n"
          "2. The requested model is not available to your account\n"
          "3. Your account may be restricted or suspended"
        )
        return ProviderError("anthropic", f"{error_message}. {guidance}")
    
    elif status_code == 404:
      if "model" in error_message.lower():
        guidance = (
          "Anthropic model not found. Please:\n"
          "1. Check the model name spelling (e.g., 'claude-3-opus-20240229')\n"
          "2. Verify the model is available in your region\n"
          "3. Ensure your account has access to the requested Claude model\n"
          "4. Use list_models() to see available models"
        )
        # Try to extract model name from error message
        model_name = "unknown"
        if "model" in error_message.lower():
          # Simple extraction - could be improved with regex
          words = error_message.split()
          for i, word in enumerate(words):
            if "model" in word.lower() and i + 1 < len(words):
              model_name = words[i + 1].strip("'\"")
              break
        return ModelNotFoundError("anthropic", model_name)
      else:
        return ProviderError("anthropic", f"Resource not found: {error_message}")
    
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
          "Anthropic rate limit exceeded. Please:\n"
          "1. Implement exponential backoff in your requests\n"
          "2. Reduce your request frequency\n"
          "3. Consider upgrading to a higher tier plan\n"
          "4. Monitor your requests per minute/hour limits"
        )
        return RateLimitError("anthropic", f"{error_message}. {guidance}", retry_after)
      else:
        guidance = (
          "Anthropic quota exceeded. Please:\n"
          "1. Check your account usage and limits\n"
          "2. Wait for your quota to reset\n"
          "3. Consider upgrading your plan\n"
          "4. Monitor your usage to avoid future overages"
        )
        return RateLimitError("anthropic", f"{error_message}. {guidance}", retry_after)
    
    elif status_code == 500:
      guidance = (
        "Anthropic server error. This is typically temporary:\n"
        "1. Retry your request after a brief delay\n"
        "2. Check Anthropic status page for service issues\n"
        "3. Implement exponential backoff for retries\n"
        "4. Contact Anthropic support if the issue persists"
      )
      return ProviderError("anthropic", f"Server error: {error_message}. {guidance}")
    
    elif status_code == 502 or status_code == 503 or status_code == 504:
      guidance = (
        "Anthropic service temporarily unavailable:\n"
        "1. Retry your request after a delay\n"
        "2. Implement exponential backoff\n"
        "3. Check Anthropic status page for outages\n"
        "4. Consider using a different Claude model if available"
      )
      return NetworkError(f"Anthropic service unavailable (HTTP {status_code}): {error_message}. {guidance}")
    
    # Check for streaming-related errors
    if "stream" in error_message.lower() or "streaming" in error_message.lower():
      guidance = (
        "Streaming is not supported by this connector:\n"
        "1. Remove any streaming parameters from your configuration\n"
        "2. Ensure no streaming options are set in provider_specific section\n"
        "3. This connector only supports complete responses"
      )
      return StreamingNotSupportedError(f"Anthropic streaming error: {error_message}. {guidance}")
    
    # Generic error with troubleshooting guidance
    guidance = (
      "General Anthropic API error troubleshooting:\n"
      "1. Check your request parameters and format\n"
      "2. Verify your API key and account status\n"
      "3. Review Anthropic API documentation for requirements\n"
      "4. Check Anthropic status page for service issues"
    )
    
    return ProviderError("anthropic", f"API error (HTTP {status_code}): {error_message}. {guidance}")

  def validate_config(self) -> List[str]:
    """Validate Anthropic-specific configuration.
    
    Returns:
      List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate API key presence
    if not self.config.api_key:
      errors.append("Anthropic API key is required")
    elif not self.config.api_key.startswith("sk-ant-api"):
      errors.append("Anthropic API key format is invalid (should start with 'sk-ant-api')")
    
    # Validate base URL if provided
    if self.config.base_url and not self.config.base_url.startswith("https://"):
      errors.append("Anthropic base URL must use HTTPS")
    
    # Validate timeout
    if self.config.timeout <= 0:
      errors.append("Timeout must be positive")
    
    # Validate max_retries
    if self.config.max_retries < 0:
      errors.append("Max retries cannot be negative")
    
    return errors

  async def generate_response(self, request: ModelRequest) -> ModelResponse:
    """Generate response using Anthropic API with no streaming and strict validation.
    
    Args:
      request: Standardized request parameters
      
    Returns:
      Standardized response from Anthropic API
      
    Raises:
      ProviderError: For provider-specific errors
      ResponseValidationError: For invalid response structure
      StreamingNotSupportedError: If streaming parameters are detected
      AuthenticationError: For authentication failures
      ModelNotFoundError: If the requested model is not available
    """
    headers = {
      "x-api-key": self.config.api_key,
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01"
    }
    
    payload = self._prepare_request(request)
    
    start_time = time.time()
    
    try:
      response = await self.http_client.post(
        f"{self.base_url}/v1/messages",
        json=payload,
        headers=headers
      )
      
      if response.status != 200:
        await self._handle_api_error(response, request.model)
      
      response_data = await response.json()
      
    except Exception as e:
      if isinstance(e, (ProviderError, AuthenticationError, ModelNotFoundError)):
        raise
      raise ProviderError("anthropic", f"Request failed: {str(e)}", e)
    
    latency_ms = self._measure_latency(start_time)
    
    # Validate response structure BEFORE parsing
    self._validate_response_structure(response_data)
    
    # Parse response after validation
    parsed_response = self._parse_response(response_data, request, latency_ms)
    
    return parsed_response

  async def list_models(self) -> List[str]:
    """List available models for Anthropic provider.
    
    Note: Anthropic doesn't provide a models endpoint, so this returns
    the configured models from the provider configuration.
    
    Returns:
      List of model names available from this provider
    """
    # Anthropic doesn't have a public models endpoint like OpenAI
    # Return the configured models from the provider config
    return self.config.models

  async def _handle_api_error(self, response, model: str) -> None:
    """Handle API error responses with comprehensive error mapping and guidance.
    
    This method uses the provider-specific error mapping to provide detailed
    error information and troubleshooting guidance for Anthropic API errors.
    
    Args:
      response: HTTP response object
      model: Model name for context
      
    Raises:
      Various specific exceptions based on error type and status code
    """
    try:
      error_data = await response.json()
    except Exception:
      error_data = None
    
    # Use the comprehensive error mapping
    exception = self._map_anthropic_error(response, error_data)
    
    # Add model context to ModelNotFoundError if available
    if isinstance(exception, ModelNotFoundError) and model:
      exception.model = model
    
    logger.error(
      f"Anthropic API error (HTTP {response.status}): {exception}",
      extra={
        "provider": "anthropic",
        "status_code": response.status,
        "model": model,
        "error_type": type(exception).__name__
      }
    )
    
    raise exception

  def _prepare_request(self, request: ModelRequest) -> Dict[str, Any]:
    """Convert ModelRequest to Anthropic format, ensuring no streaming.
    
    This method creates the request payload for Anthropic API with strict
    streaming prevention and comprehensive parameter validation.
    
    Args:
      request: The ModelRequest to convert
      
    Returns:
      Dict containing the Anthropic API request payload
      
    Raises:
      StreamingNotSupportedError: If streaming parameters are detected
    """
    # Build base payload - Anthropic uses different structure than OpenAI
    messages = [{"role": "user", "content": request.prompt}]
    
    payload = {
      "model": request.model,
      "messages": messages,
      "max_tokens": request.max_tokens,
      "temperature": request.temperature
    }
    
    # Add system prompt if provided (separate field in Anthropic)
    if request.system_prompt:
      payload["system"] = request.system_prompt
    
    # Add stop sequences if provided
    if request.stop_sequences:
      payload["stop_sequences"] = request.stop_sequences
    
    # Add provider-specific parameters (excluding streaming and protected params)
    streaming_params = {
      "stream", "streaming", "stream_options", "stream_usage", 
      "stream_callback", "stream_handler", "incremental"
    }
    
    # Parameters that should not be overridden by provider_specific
    protected_params = {"model", "messages", "max_tokens", "temperature", "system", "stop_sequences"}
    
    for key, value in request.provider_specific.items():
      if key not in streaming_params and key not in protected_params:
        # Validate parameter before adding
        if self._is_valid_anthropic_parameter(key, value):
          payload[key] = value
    
    # Use base class streaming detection for comprehensive validation
    self._detect_streaming_parameters(request, payload)
    
    return payload

  def _is_valid_anthropic_parameter(self, key: str, value: Any) -> bool:
    """Validate that a parameter is valid for Anthropic API.
    
    Args:
      key: Parameter name
      value: Parameter value
      
    Returns:
      True if parameter is valid for Anthropic API
    """
    # List of known valid Anthropic parameters
    valid_params = {
      "temperature", "max_tokens", "top_p", "top_k", "stop_sequences",
      "metadata", "system"
    }
    
    return key in valid_params

  def _parse_response(
    self, 
    response_data: Dict[str, Any], 
    request: ModelRequest, 
    latency_ms: int
  ) -> ModelResponse:
    """Parse Anthropic response into standardized ModelResponse.
    
    Args:
      response_data: Raw response from Anthropic API
      request: Original request for context
      latency_ms: Request latency in milliseconds
      
    Returns:
      Standardized ModelResponse
    """
    # Extract text content from Anthropic's content array
    content_blocks = response_data["content"]
    text_content = content_blocks[0]["text"]  # We validated this exists
    
    usage = response_data["usage"]
    total_tokens = usage["input_tokens"] + usage["output_tokens"]
    
    return ModelResponse(
      text=text_content,
      model=response_data["model"],
      provider="anthropic",
      timestamp=datetime.now(),
      latency_ms=latency_ms,
      token_count=total_tokens,
      finish_reason=response_data["stop_reason"],
      cost_estimate=self._estimate_cost(usage, response_data["model"]),
      metadata={
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "stop_reason": response_data["stop_reason"],
        "stop_sequence": response_data.get("stop_sequence"),
        "message_id": response_data.get("id")
      }
    )

  def _estimate_cost(self, usage: Dict[str, Any], model: str) -> Optional[float]:
    """Estimate cost for Anthropic API usage.
    
    Args:
      usage: Usage information from API response
      model: Model name used
      
    Returns:
      Estimated cost in USD, or None if model pricing unknown
    """
    # Anthropic pricing (as of 2024) - prices per 1M tokens
    pricing = {
      "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
      "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
      "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    if model not in pricing:
      return None
    
    model_pricing = pricing[model]
    input_cost = (usage["input_tokens"] / 1_000_000) * model_pricing["input"]
    output_cost = (usage["output_tokens"] / 1_000_000) * model_pricing["output"]
    
    return input_cost + output_cost

  def _validate_response_structure(self, response_data: Dict[str, Any]) -> None:
    """Comprehensive validation of Anthropic response structure.
    
    This method performs exhaustive validation of the Anthropic API response
    to ensure all required fields are present and properly formatted.
    
    Args:
      response_data: Raw response dictionary from Anthropic API
      
    Raises:
      ResponseValidationError: If any validation check fails
    """
    # Validate response is not empty
    self._validate_response_not_empty(response_data)
    
    # Check top-level required fields
    required_fields = ["content", "model", "role", "stop_reason", "usage"]
    for field in required_fields:
      if field not in response_data:
        raise ResponseValidationError(
          f"Missing required field '{field}' in Anthropic response",
          provider="anthropic",
          response_data=response_data
        )
    
    # Validate role
    if response_data["role"] != "assistant":
      raise ResponseValidationError(
        f"Invalid role in Anthropic response: expected 'assistant', got '{response_data['role']}'",
        provider="anthropic",
        response_data=response_data
      )
    
    # Validate content array
    content = response_data["content"]
    if not isinstance(content, list) or not content:
      raise ResponseValidationError(
        "Anthropic response 'content' must be a non-empty array",
        provider="anthropic",
        response_data=response_data
      )
    
    # Validate first content block
    content_block = content[0]
    if not isinstance(content_block, dict):
      raise ResponseValidationError(
        "Anthropic content block must be a dictionary",
        provider="anthropic",
        response_data=response_data
      )
    
    required_content_fields = ["text", "type"]
    for field in required_content_fields:
      if field not in content_block:
        raise ResponseValidationError(
          f"Missing required field '{field}' in Anthropic content block",
          provider="anthropic",
          response_data=response_data
        )
    
    # Validate content type
    if content_block["type"] != "text":
      raise ResponseValidationError(
        f"Anthropic content block type must be 'text', got '{content_block['type']}'",
        provider="anthropic",
        response_data=response_data
      )
    
    # Validate content text is not empty
    text_content = content_block["text"]
    if not text_content or not isinstance(text_content, str):
      raise ResponseValidationError(
        "Anthropic content text must be a non-empty string",
        provider="anthropic",
        response_data=response_data
      )
    
    if len(text_content.strip()) == 0:
      raise ResponseValidationError(
        "Anthropic content text cannot be empty or whitespace only",
        provider="anthropic",
        response_data=response_data
      )
    
    # Validate usage information
    usage = response_data["usage"]
    if not isinstance(usage, dict):
      raise ResponseValidationError(
        "Anthropic usage must be a dictionary",
        provider="anthropic",
        response_data=response_data
      )
    
    required_usage_fields = ["input_tokens", "output_tokens"]
    for field in required_usage_fields:
      if field not in usage:
        raise ResponseValidationError(
          f"Missing usage field '{field}' in Anthropic response",
          provider="anthropic",
          response_data=response_data
        )
      if not isinstance(usage[field], int) or usage[field] < 0:
        raise ResponseValidationError(
          f"Anthropic usage field '{field}' must be a non-negative integer, got {usage[field]}",
          provider="anthropic",
          response_data=response_data
        )
    
    # Validate model field
    if not isinstance(response_data["model"], str) or not response_data["model"]:
      raise ResponseValidationError(
        "Anthropic response model must be a non-empty string",
        provider="anthropic",
        response_data=response_data
      )
    
    # Validate stop_reason
    valid_stop_reasons = ["end_turn", "max_tokens", "stop_sequence"]
    if response_data["stop_reason"] not in valid_stop_reasons:
      # Log warning but don't fail - Anthropic might add new stop reasons
      pass