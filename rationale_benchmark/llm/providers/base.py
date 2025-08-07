"""Abstract base class for LLM providers."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from rationale_benchmark.llm.exceptions import (
    ProviderError,
    ResponseValidationError,
    StreamingNotSupportedError,
)
from rationale_benchmark.llm.http.client import HTTPClient
from rationale_benchmark.llm.models import ModelRequest, ModelResponse, ProviderConfig

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
  """Abstract base class for all LLM providers.
  
  This class defines the contract that all LLM providers must implement,
  ensuring consistent behavior across different provider implementations.
  All providers must support non-streaming responses only.
  """

  def __init__(self, config: ProviderConfig, http_client: HTTPClient):
    """Initialize the provider with configuration and HTTP client.
    
    Args:
      config: Provider-specific configuration
      http_client: HTTP client for making API requests
    """
    self.config = config
    self.http_client = http_client
    self.name = config.name

  @abstractmethod
  async def generate_response(self, request: ModelRequest) -> ModelResponse:
    """Generate response from the LLM.
    
    This method must be implemented by all providers to handle LLM requests
    and return standardized responses. Streaming is explicitly not supported.
    
    Args:
      request: Standardized request parameters
      
    Returns:
      Standardized response from the LLM
      
    Raises:
      ProviderError: For provider-specific errors
      ResponseValidationError: For invalid response structure
      StreamingNotSupportedError: If streaming parameters are detected
    """
    pass

  @abstractmethod
  async def list_models(self) -> List[str]:
    """List available models for this provider.
    
    Returns:
      List of model names available from this provider
      
    Raises:
      ProviderError: For provider-specific errors
    """
    pass

  @abstractmethod
  def validate_config(self) -> List[str]:
    """Validate provider-specific configuration.
    
    Returns:
      List of validation error messages (empty if valid)
    """
    pass

  @abstractmethod
  def _prepare_request(self, request: ModelRequest) -> Dict[str, Any]:
    """Convert ModelRequest to provider-specific format.
    
    This method must filter out any streaming-related parameters
    and ensure the request is formatted correctly for the provider's API.
    
    Args:
      request: Standardized request parameters
      
    Returns:
      Provider-specific request payload
      
    Raises:
      StreamingNotSupportedError: If streaming parameters are detected
    """
    pass

  @abstractmethod
  def _parse_response(
    self, 
    response_data: Dict[str, Any], 
    request: ModelRequest,
    latency_ms: int
  ) -> ModelResponse:
    """Convert provider response to ModelResponse.
    
    Args:
      response_data: Raw response from provider API
      request: Original request for context
      latency_ms: Request latency in milliseconds
      
    Returns:
      Standardized ModelResponse
      
    Raises:
      ResponseValidationError: If response structure is invalid
    """
    pass

  def _validate_no_streaming_params(
    self, 
    params: Dict[str, Any], 
    context: str = "request"
  ) -> None:
    """Validate that no streaming parameters are present.
    
    Args:
      params: Parameters to validate
      context: Context for error messages
      
    Raises:
      StreamingNotSupportedError: If streaming parameters are found
    """
    streaming_params = {
      "stream", "streaming", "stream_options", "stream_usage",
      "stream_callback", "stream_handler", "incremental"
    }
    
    blocked_params = []
    for key in params:
      if key in streaming_params:
        blocked_params.append(key)
    
    if blocked_params:
      raise StreamingNotSupportedError(
        f"Streaming parameters not supported in {context}: {blocked_params}",
        blocked_params=blocked_params
      )

  def _filter_streaming_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out streaming parameters from request parameters.
    
    Args:
      params: Parameters to filter
      
    Returns:
      Filtered parameters with streaming params removed
    """
    streaming_params = {
      "stream", "streaming", "stream_options", "stream_usage",
      "stream_callback", "stream_handler", "incremental"
    }
    
    filtered = {}
    for key, value in params.items():
      if key not in streaming_params:
        filtered[key] = value
    
    return filtered

  def _measure_latency(self, start_time: float) -> int:
    """Calculate request latency in milliseconds.
    
    Args:
      start_time: Start time from time.time()
      
    Returns:
      Latency in milliseconds
    """
    return int((time.time() - start_time) * 1000)

  def _validate_response_not_empty(self, response_data: Dict[str, Any]) -> None:
    """Validate that response data is not empty.
    
    Args:
      response_data: Response data to validate
      
    Raises:
      ResponseValidationError: If response is empty or invalid
    """
    if not response_data:
      raise ResponseValidationError(
        f"Empty response from {self.name} provider",
        provider=self.name,
        response_data=response_data
      )
    
    if not isinstance(response_data, dict):
      raise ResponseValidationError(
        f"Response from {self.name} provider must be a dictionary",
        provider=self.name,
        response_data=response_data
      )

  def _handle_http_error(self, response, context: str = "API request") -> None:
    """Handle HTTP error responses.
    
    Args:
      response: HTTP response object
      context: Context for error message
      
    Raises:
      ProviderError: For HTTP errors
    """
    if response.status >= 400:
      error_msg = f"{context} failed with status {response.status}"
      
      # Try to get error details from response
      try:
        error_data = response.json() if hasattr(response, 'json') else {}
        if isinstance(error_data, dict) and 'error' in error_data:
          error_msg += f": {error_data['error']}"
      except Exception:
        # If we can't parse error details, use the status code
        pass
      
      raise ProviderError(self.name, error_msg)

  def __str__(self) -> str:
    """String representation of the provider."""
    return f"{self.__class__.__name__}(name='{self.name}')"

  def _detect_streaming_parameters(self, request: ModelRequest, payload: Dict[str, Any]) -> None:
    """Detect and block streaming parameters in requests.
    
    This method checks for any streaming-related parameters in the request
    or payload and raises an error if found, as streaming is not supported.
    
    Args:
      request: The original model request
      payload: The prepared request payload
      
    Raises:
      StreamingNotSupportedError: If streaming parameters are detected
    """
    # Common streaming parameter names across providers
    streaming_params = {
      "stream", "streaming", "stream_options", "stream_usage",
      "stream_callback", "stream_handler", "incremental",
      "server_sent_events", "sse", "event_stream"
    }
    
    blocked_params = []
    
    # Check payload for streaming parameters
    for param in streaming_params:
      if param in payload:
        if payload[param] is True or (isinstance(payload[param], str) and payload[param].lower() == "true"):
          blocked_params.append(param)
    
    # Check provider_specific section in request
    if request.provider_specific:
      for param in streaming_params:
        if param in request.provider_specific:
          if (request.provider_specific[param] is True or 
              (isinstance(request.provider_specific[param], str) and 
               request.provider_specific[param].lower() == "true")):
            blocked_params.append(f"provider_specific.{param}")
    
    # Raise error if any streaming parameters were found
    if blocked_params:
      guidance = (
        f"Streaming parameters detected but not supported by {self.name} provider:\n"
        f"Blocked parameters: {blocked_params}\n"
        "Please:\n"
        "1. Remove all streaming-related parameters from your configuration\n"
        "2. Ensure no streaming options are set in provider_specific section\n"
        "3. This connector only supports complete, non-streaming responses\n"
        "4. Check your configuration files for any streaming settings"
      )
      raise StreamingNotSupportedError(
        f"Streaming not supported: {guidance}",
        blocked_params=blocked_params
      )

  def __repr__(self) -> str:
    """Detailed string representation of the provider."""
    return (
      f"{self.__class__.__name__}("
      f"name='{self.name}', "
      f"base_url='{getattr(self.config, 'base_url', None)}', "
      f"timeout={self.config.timeout})"
    )