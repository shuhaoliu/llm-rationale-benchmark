"""Google Gemini API provider implementation."""

import logging
import re
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


class GeminiProvider(LLMProvider):
  """Google Gemini API provider implementation with structured output validation.
  
  This provider implements the LLMProvider interface for Google's Gemini models,
  including Gemini 1.5 Pro, Gemini 1.5 Flash, and other Gemini variants.
  
  The provider uses Google Cloud authentication patterns and supports the full
  range of Gemini API parameters while explicitly blocking streaming responses.
  """

  def __init__(self, config: ProviderConfig, http_client: HTTPClient):
    """Initialize Gemini provider with configuration and HTTP client.
    
    Args:
      config: Provider configuration including API key and settings
      http_client: HTTP client for making API requests
    """
    super().__init__(config, http_client)
    self.base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta"
    
    # Validate API key format on initialization
    if not self._is_valid_api_key_format(config.api_key):
      logger.warning(f"API key for {self.name} may have invalid format")

  def _is_valid_api_key_format(self, api_key: str) -> bool:
    """Validate Google API key format.
    
    Google API keys typically start with 'AIza' and are 39 characters long.
    
    Args:
      api_key: API key to validate
      
    Returns:
      True if format appears valid, False otherwise
    """
    if not api_key:
      return False
    
    # Google API keys typically start with 'AIza' and are 39 characters
    return api_key.startswith('AIza') and len(api_key) == 39

  async def generate_response(self, request: ModelRequest) -> ModelResponse:
    """Generate response using Gemini API with no streaming and strict validation.
    
    Args:
      request: Standardized request parameters
      
    Returns:
      Standardized response from Gemini
      
    Raises:
      ProviderError: For API errors
      AuthenticationError: For authentication failures
      ModelNotFoundError: For invalid model names
      ResponseValidationError: For invalid response structure
      StreamingNotSupportedError: If streaming parameters are detected
    """
    headers = {
      "Content-Type": "application/json",
      "x-goog-api-key": self.config.api_key
    }
    
    # Prepare request payload (this will validate no streaming params)
    payload = self._prepare_request(request)
    
    # Construct API endpoint URL
    url = f"{self.base_url}/models/{request.model}:generateContent"
    
    start_time = time.time()
    
    try:
      response = await self.http_client.post(
        url,
        json=payload,
        headers=headers
      )
      
      # Handle HTTP errors
      if response.status != 200:
        await self._handle_api_error(response, request.model)
      
      response_data = await response.json()
      
    except Exception as e:
      if isinstance(e, (ProviderError, AuthenticationError, ModelNotFoundError)):
        raise
      raise ProviderError(self.name, f"Request failed: {str(e)}", e)
    
    latency_ms = self._measure_latency(start_time)
    
    # Validate response structure BEFORE parsing
    self._validate_response_not_empty(response_data)
    self._validate_response_structure(response_data)
    
    # Parse response after validation
    parsed_response = self._parse_response(response_data, request, latency_ms)
    
    logger.debug(f"Gemini response generated successfully for {request.model}")
    return parsed_response

  async def _handle_api_error(self, response, model: str) -> None:
    """Handle API error responses with appropriate exception types.
    
    Args:
      response: HTTP response object
      model: Model name for context
      
    Raises:
      AuthenticationError: For authentication failures
      ModelNotFoundError: For model not found errors
      ProviderError: For other API errors
    """
    try:
      error_data = await response.json()
      error_info = error_data.get("error", {})
      error_message = error_info.get("message", f"API request failed with status {response.status}")
      error_code = error_info.get("code", response.status)
      
    except Exception:
      error_message = f"API request failed with status {response.status}"
      error_code = response.status
    
    # Map specific error codes to appropriate exceptions
    if response.status == 401 or error_code == 401:
      raise AuthenticationError(self.name, error_message)
    elif response.status == 404 or error_code == 404:
      raise ModelNotFoundError(self.name, model)
    elif response.status == 429 or error_code == 429:
      raise ProviderError(self.name, f"Rate limit exceeded (429): {error_message}")
    else:
      raise ProviderError(self.name, f"API error ({response.status}): {error_message}")

  async def list_models(self) -> List[str]:
    """List available models for Gemini provider.
    
    Returns:
      List of available model names
      
    Raises:
      ProviderError: For API errors
      AuthenticationError: For authentication failures
    """
    headers = {
      "Content-Type": "application/json",
      "x-goog-api-key": self.config.api_key
    }
    
    url = f"{self.base_url}/models"
    
    try:
      response = await self.http_client.get(url, headers=headers)
      
      if response.status != 200:
        await self._handle_api_error(response, "models")
      
      response_data = await response.json()
      
      # Extract model names from response
      models = []
      for model_info in response_data.get("models", []):
        model_name = model_info.get("name", "")
        # Extract just the model name from the full path (e.g., "models/gemini-1.5-pro" -> "gemini-1.5-pro")
        if model_name.startswith("models/"):
          model_name = model_name[7:]  # Remove "models/" prefix
        
        # Only include models that support generateContent
        supported_methods = model_info.get("supportedGenerationMethods", [])
        if "generateContent" in supported_methods and model_name:
          models.append(model_name)
      
      logger.debug(f"Retrieved {len(models)} models from Gemini API")
      return models
      
    except Exception as e:
      if isinstance(e, (ProviderError, AuthenticationError)):
        raise
      raise ProviderError(self.name, f"Failed to list models: {str(e)}", e)

  def validate_config(self) -> List[str]:
    """Validate Gemini provider configuration.
    
    Returns:
      List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate API key presence
    if not self.config.api_key:
      errors.append("Gemini API key is required")
    elif not self._is_valid_api_key_format(self.config.api_key):
      errors.append("Gemini API key format appears invalid (should start with 'AIza' and be 39 characters)")
    
    # Validate base URL format
    if self.config.base_url and not self.config.base_url.startswith(('http://', 'https://')):
      errors.append("Base URL must start with http:// or https://")
    
    # Validate timeout
    if self.config.timeout <= 0:
      errors.append("Timeout must be positive")
    
    # Validate max retries
    if self.config.max_retries < 0:
      errors.append("Max retries cannot be negative")
    
    return errors

  def _prepare_request(self, request: ModelRequest) -> Dict[str, Any]:
    """Convert ModelRequest to Gemini format, ensuring no streaming.
    
    This method creates the request payload for Gemini API with strict
    streaming prevention and comprehensive parameter validation.
    
    Args:
      request: The ModelRequest to convert
      
    Returns:
      Dict containing the Gemini API request payload
      
    Raises:
      StreamingNotSupportedError: If streaming parameters are detected
    """
    # Validate no streaming parameters in provider_specific
    self._validate_no_streaming_params(request.provider_specific, "provider_specific")
    
    # Build contents array (messages)
    contents = []
    
    # Add system prompt if provided
    if request.system_prompt:
      contents.append({
        "role": "system",
        "parts": [{"text": request.system_prompt}]
      })
    
    # Add user message
    contents.append({
      "role": "user", 
      "parts": [{"text": request.prompt}]
    })
    
    # Build generation configuration
    generation_config = {
      "temperature": request.temperature,
      "maxOutputTokens": request.max_tokens
    }
    
    # Add stop sequences if provided
    if request.stop_sequences:
      generation_config["stopSequences"] = request.stop_sequences
    
    # Add valid provider-specific parameters
    filtered_specific = self._filter_streaming_params(request.provider_specific)
    for key, value in filtered_specific.items():
      if self._is_valid_gemini_parameter(key, value):
        generation_config[key] = value
      else:
        logger.warning(f"Skipped invalid Gemini parameter '{key}': {value}")
    
    # Build final payload
    payload = {
      "contents": contents,
      "generationConfig": generation_config
    }
    
    logger.debug(f"Prepared Gemini request for {request.model} with {len(contents)} messages")
    return payload

  def _is_valid_gemini_parameter(self, key: str, value: Any) -> bool:
    """Validate that a parameter is valid for Gemini API.
    
    Args:
      key: Parameter name
      value: Parameter value
      
    Returns:
      True if parameter is valid for Gemini API
    """
    # List of known valid Gemini parameters
    valid_params = {
      "topK", "topP", "temperature", "maxOutputTokens", "candidateCount",
      "stopSequences", "responseMimeType", "responseSchema", 
      "presencePenalty", "frequencyPenalty", "seed"
    }
    
    return key in valid_params

  def _parse_response(
    self, 
    response_data: Dict[str, Any], 
    request: ModelRequest,
    latency_ms: int
  ) -> ModelResponse:
    """Parse Gemini response into standardized ModelResponse.
    
    Args:
      response_data: Raw response from Gemini API
      request: Original request for context
      latency_ms: Request latency in milliseconds
      
    Returns:
      Standardized ModelResponse
      
    Raises:
      ResponseValidationError: If response structure is invalid
    """
    # Extract the first candidate
    candidate = response_data["candidates"][0]
    content = candidate["content"]
    
    # Concatenate all text parts
    text_parts = []
    for part in content["parts"]:
      if "text" in part:
        text_parts.append(part["text"])
    
    response_text = "".join(text_parts)
    
    # Extract usage metadata if available
    usage_metadata = response_data.get("usageMetadata", {})
    token_count = usage_metadata.get("totalTokenCount")
    
    # Extract finish reason
    finish_reason = candidate.get("finishReason", "UNKNOWN")
    
    # Estimate cost if usage data is available
    cost_estimate = self._estimate_cost(usage_metadata, request.model)
    
    # Build metadata
    metadata = {}
    if "promptTokenCount" in usage_metadata:
      metadata["promptTokenCount"] = usage_metadata["promptTokenCount"]
    if "candidatesTokenCount" in usage_metadata:
      metadata["candidatesTokenCount"] = usage_metadata["candidatesTokenCount"]
    if "modelVersion" in response_data:
      metadata["modelVersion"] = response_data["modelVersion"]
    if "safetyRatings" in candidate:
      metadata["safetyRatings"] = candidate["safetyRatings"]
    
    return ModelResponse(
      text=response_text,
      model=request.model,
      provider=self.name,
      timestamp=datetime.now(),
      latency_ms=latency_ms,
      token_count=token_count,
      finish_reason=finish_reason,
      cost_estimate=cost_estimate,
      metadata=metadata
    )

  def _validate_response_structure(self, response_data: Dict[str, Any]) -> None:
    """Comprehensive validation of Gemini response structure.
    
    This method performs exhaustive validation of the Gemini API response
    to ensure all required fields are present and properly formatted.
    
    Args:
      response_data: Raw response dictionary from Gemini API
      
    Raises:
      ResponseValidationError: If any validation check fails
    """
    # Check top-level required fields
    if "candidates" not in response_data:
      raise ResponseValidationError(
        "Missing required field 'candidates' in Gemini response",
        provider=self.name,
        response_data=response_data
      )
    
    # Validate candidates array
    candidates = response_data["candidates"]
    if not isinstance(candidates, list) or not candidates:
      raise ResponseValidationError(
        "Gemini response 'candidates' must be a non-empty array",
        provider=self.name,
        response_data=response_data
      )
    
    # Validate first candidate structure
    candidate = candidates[0]
    if not isinstance(candidate, dict):
      raise ResponseValidationError(
        "Gemini candidate must be a dictionary",
        provider=self.name,
        response_data=response_data
      )
    
    # Check required candidate fields
    required_candidate_fields = ["content", "finishReason", "index"]
    for field in required_candidate_fields:
      if field not in candidate:
        raise ResponseValidationError(
          f"Missing required field '{field}' in Gemini candidate",
          provider=self.name,
          response_data=response_data
        )
    
    # Validate candidate index
    if not isinstance(candidate["index"], int) or candidate["index"] < 0:
      raise ResponseValidationError(
        f"Invalid candidate index in Gemini response: {candidate['index']}",
        provider=self.name,
        response_data=response_data
      )
    
    # Validate content structure
    content = candidate["content"]
    if not isinstance(content, dict):
      raise ResponseValidationError(
        "Gemini candidate content must be a dictionary",
        provider=self.name,
        response_data=response_data
      )
    
    # Check required content fields
    required_content_fields = ["parts", "role"]
    for field in required_content_fields:
      if field not in content:
        raise ResponseValidationError(
          f"Missing required field '{field}' in Gemini content",
          provider=self.name,
          response_data=response_data
        )
    
    # Validate content role
    if content["role"] != "model":
      raise ResponseValidationError(
        f"Invalid content role in Gemini response: expected 'model', got '{content['role']}'",
        provider=self.name,
        response_data=response_data
      )
    
    # Validate parts array
    parts = content["parts"]
    if not isinstance(parts, list) or not parts:
      raise ResponseValidationError(
        "Gemini content parts must be a non-empty array",
        provider=self.name,
        response_data=response_data
      )
    
    # Validate at least one part has text
    has_text = False
    for part in parts:
      if not isinstance(part, dict):
        raise ResponseValidationError(
          "Gemini content part must be a dictionary",
          provider=self.name,
          response_data=response_data
        )
      
      if "text" in part:
        if not isinstance(part["text"], str):
          raise ResponseValidationError(
            "Gemini part text must be a string",
            provider=self.name,
            response_data=response_data
          )
        
        if part["text"].strip():  # Has non-whitespace content
          has_text = True
    
    if not has_text:
      raise ResponseValidationError(
        "Gemini response must contain at least one part with non-empty text",
        provider=self.name,
        response_data=response_data
      )
    
    logger.debug(f"Gemini response structure validation passed for model response")

  def _estimate_cost(self, usage_metadata: Dict[str, Any], model: str) -> Optional[float]:
    """Estimate cost for Gemini API usage.
    
    Args:
      usage_metadata: Usage metadata from API response
      model: Model name used
      
    Returns:
      Estimated cost in USD, or None if cannot estimate
    """
    if not usage_metadata or "promptTokenCount" not in usage_metadata or "candidatesTokenCount" not in usage_metadata:
      return None
    
    prompt_tokens = usage_metadata["promptTokenCount"]
    output_tokens = usage_metadata["candidatesTokenCount"]
    
    # Gemini pricing (as of 2024, subject to change)
    # These are approximate rates and should be updated based on current pricing
    pricing = {
      "gemini-1.5-pro": {
        "input": 0.00125 / 1000,   # $1.25 per 1M input tokens
        "output": 0.005 / 1000     # $5.00 per 1M output tokens
      },
      "gemini-1.5-flash": {
        "input": 0.000075 / 1000,  # $0.075 per 1M input tokens
        "output": 0.0003 / 1000    # $0.30 per 1M output tokens
      },
      "gemini-pro": {
        "input": 0.0005 / 1000,    # $0.50 per 1M input tokens
        "output": 0.0015 / 1000    # $1.50 per 1M output tokens
      },
      "gemini-pro-vision": {
        "input": 0.00025 / 1000,   # $0.25 per 1M input tokens
        "output": 0.0005 / 1000    # $0.50 per 1M output tokens
      }
    }
    
    # Find matching pricing
    model_pricing = None
    for price_model, rates in pricing.items():
      if price_model in model.lower():
        model_pricing = rates
        break
    
    if not model_pricing:
      return None
    
    # Calculate cost
    input_cost = prompt_tokens * model_pricing["input"]
    output_cost = output_tokens * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    return round(total_cost, 6)  # Round to 6 decimal places for precision