"""Response validator for comprehensive response structure validation."""

import logging
import re
from typing import Any, Dict, Optional

from rationale_benchmark.llm.models import ModelResponse
from rationale_benchmark.llm.exceptions import ResponseValidationError

logger = logging.getLogger(__name__)


class ResponseValidator:
  """Comprehensive response structure validator for LLM responses.
  
  This class provides validation methods for basic structure, content quality,
  and metadata validation with provider-specific logic for different response formats.
  """

  def __init__(self):
    """Initialize the response validator."""
    self._provider_validators = {
      "openai": self._validate_openai_response,
      "anthropic": self._validate_anthropic_response,
      "gemini": self._validate_gemini_response,
      "openrouter": self._validate_openai_response,  # OpenRouter uses OpenAI format
    }

  def validate_response(self, response: Any) -> None:
    """Validate a model response comprehensively.
    
    This method performs exhaustive validation of the response structure,
    content quality, and provider-specific requirements.
    
    Args:
      response: The response to validate (should be ModelResponse)
      
    Raises:
      ResponseValidationError: If any validation check fails
    """
    # Basic type validation
    if response is None:
      raise ResponseValidationError("Response cannot be None")
    
    if not isinstance(response, ModelResponse):
      raise ResponseValidationError(
        "Response must be a ModelResponse instance",
        response_data=response
      )
    
    # Validate basic structure
    self._validate_basic_structure(response)
    
    # Validate content quality
    self._validate_content_quality(response)
    
    # Validate metadata
    self._validate_metadata(response)
    
    # Provider-specific validation
    self._validate_provider_specific(response)
    
    logger.debug(
      f"Response validation passed for {response.provider} provider, "
      f"model {response.model} (text length: {len(response.text)}, "
      f"tokens: {response.token_count})"
    )

  def _validate_basic_structure(self, response: ModelResponse) -> None:
    """Validate basic response structure and required fields.
    
    Args:
      response: The response to validate
      
    Raises:
      ResponseValidationError: If basic structure validation fails
    """
    # Validate text content
    if not response.text:
      raise ResponseValidationError(
        "Response text cannot be empty",
        provider=response.provider,
        response_data=response
      )
    
    if not response.text.strip():
      raise ResponseValidationError(
        "Response text cannot be empty or whitespace only",
        provider=response.provider,
        response_data=response
      )
    
    # Validate model name
    if not response.model:
      raise ResponseValidationError(
        "Model name cannot be empty",
        provider=response.provider,
        response_data=response
      )
    
    # Validate provider name
    if not response.provider:
      raise ResponseValidationError(
        "Provider name cannot be empty",
        provider=response.provider,
        response_data=response
      )
    
    # Validate timestamp
    if response.timestamp is None:
      raise ResponseValidationError(
        "Timestamp cannot be None",
        provider=response.provider,
        response_data=response
      )
    
    # Validate latency
    if response.latency_ms < 0:
      raise ResponseValidationError(
        f"Latency cannot be negative: {response.latency_ms}",
        provider=response.provider,
        response_data=response
      )
    
    # Validate optional numeric fields
    if response.token_count is not None and response.token_count < 0:
      raise ResponseValidationError(
        f"Token count cannot be negative: {response.token_count}",
        provider=response.provider,
        response_data=response
      )
    
    if response.cost_estimate is not None and response.cost_estimate < 0:
      raise ResponseValidationError(
        f"Cost estimate cannot be negative: {response.cost_estimate}",
        provider=response.provider,
        response_data=response
      )

  def _validate_content_quality(self, response: ModelResponse) -> None:
    """Validate content quality and detect potential issues.
    
    Args:
      response: The response to validate
      
    Note:
      This method performs quality checks but doesn't raise errors for
      quality issues, only logs warnings for potential problems.
    """
    text = response.text.strip()
    
    # Check for very short responses (may indicate truncation)
    if len(text) < 5:
      logger.warning(
        f"Very short response from {response.provider} ({len(text)} chars): '{text}'"
      )
    
    # Check for highly repetitive content
    if len(text) > 50:
      # Simple repetition detection: check if any 10-char substring appears > 5 times
      substrings = {}
      for i in range(len(text) - 9):
        substring = text[i:i+10]
        substrings[substring] = substrings.get(substring, 0) + 1
      
      max_repetitions = max(substrings.values()) if substrings else 0
      if max_repetitions > 5:
        logger.warning(
          f"Highly repetitive content detected in {response.provider} response "
          f"(max repetitions: {max_repetitions})"
        )
    
    # Check for common error patterns
    error_patterns = [
      r"error\s*:\s*",
      r"exception\s*:",
      r"failed\s*to\s*",
      r"unable\s*to\s*",
      r"cannot\s*",
      r"invalid\s*",
    ]
    
    for pattern in error_patterns:
      if re.search(pattern, text.lower()):
        logger.warning(
          f"Potential error content detected in {response.provider} response: "
          f"pattern '{pattern}' found"
        )
        break

  def _validate_metadata(self, response: ModelResponse) -> None:
    """Validate response metadata structure and content.
    
    Args:
      response: The response to validate
      
    Raises:
      ResponseValidationError: If metadata validation fails
    """
    if not isinstance(response.metadata, dict):
      raise ResponseValidationError(
        f"Metadata must be a dictionary, got {type(response.metadata)}",
        provider=response.provider,
        response_data=response
      )
    
    # Validate metadata values are JSON-serializable types
    for key, value in response.metadata.items():
      if not self._is_json_serializable(value):
        logger.warning(
          f"Non-JSON-serializable value in metadata for {response.provider}: "
          f"key '{key}' has type {type(value)}"
        )

  def _validate_provider_specific(self, response: ModelResponse) -> None:
    """Validate provider-specific response requirements.
    
    Args:
      response: The response to validate
      
    Raises:
      ResponseValidationError: If provider-specific validation fails
    """
    provider = response.provider.lower()
    
    if provider in self._provider_validators:
      self._provider_validators[provider](response)
    else:
      logger.debug(f"No specific validation for provider '{provider}'")

  def _validate_openai_response(self, response: ModelResponse) -> None:
    """Validate OpenAI-specific response requirements.
    
    Args:
      response: The response to validate
      
    Raises:
      ResponseValidationError: If OpenAI validation fails
    """
    metadata = response.metadata
    
    # Check for required OpenAI metadata fields
    required_fields = ["prompt_tokens", "completion_tokens"]
    for field in required_fields:
      if field not in metadata:
        raise ResponseValidationError(
          f"OpenAI response missing required metadata field '{field}'",
          provider=response.provider,
          response_data=response
        )
      
      if not isinstance(metadata[field], int) or metadata[field] < 0:
        raise ResponseValidationError(
          f"OpenAI metadata field '{field}' must be a non-negative integer, "
          f"got {metadata[field]}",
          provider=response.provider,
          response_data=response
        )
    
    # Validate token count consistency
    if response.token_count is not None:
      expected_total = metadata["prompt_tokens"] + metadata["completion_tokens"]
      if response.token_count != expected_total:
        raise ResponseValidationError(
          f"OpenAI token count mismatch: response.token_count={response.token_count}, "
          f"metadata sum={expected_total}",
          provider=response.provider,
          response_data=response
        )
    
    # Validate finish reason if present
    if response.finish_reason:
      valid_finish_reasons = ["stop", "length", "function_call", "content_filter", "null"]
      if response.finish_reason not in valid_finish_reasons:
        logger.warning(
          f"Unexpected OpenAI finish reason: {response.finish_reason} "
          f"(expected one of: {valid_finish_reasons})"
        )

  def _validate_anthropic_response(self, response: ModelResponse) -> None:
    """Validate Anthropic-specific response requirements.
    
    Args:
      response: The response to validate
      
    Raises:
      ResponseValidationError: If Anthropic validation fails
    """
    metadata = response.metadata
    
    # Check for required Anthropic metadata fields
    required_fields = ["input_tokens", "output_tokens"]
    for field in required_fields:
      if field not in metadata:
        raise ResponseValidationError(
          f"Anthropic response missing required metadata field '{field}'",
          provider=response.provider,
          response_data=response
        )
      
      if not isinstance(metadata[field], int) or metadata[field] < 0:
        raise ResponseValidationError(
          f"Anthropic metadata field '{field}' must be a non-negative integer, "
          f"got {metadata[field]}",
          provider=response.provider,
          response_data=response
        )
    
    # Validate token count consistency
    if response.token_count is not None:
      expected_total = metadata["input_tokens"] + metadata["output_tokens"]
      if response.token_count != expected_total:
        raise ResponseValidationError(
          f"Anthropic token count mismatch: response.token_count={response.token_count}, "
          f"metadata sum={expected_total}",
          provider=response.provider,
          response_data=response
        )
    
    # Validate stop reason if present
    if response.finish_reason:
      valid_stop_reasons = ["end_turn", "max_tokens", "stop_sequence"]
      if response.finish_reason not in valid_stop_reasons:
        logger.warning(
          f"Unexpected Anthropic stop reason: {response.finish_reason} "
          f"(expected one of: {valid_stop_reasons})"
        )

  def _validate_gemini_response(self, response: ModelResponse) -> None:
    """Validate Gemini-specific response requirements.
    
    Args:
      response: The response to validate
      
    Raises:
      ResponseValidationError: If Gemini validation fails
    """
    metadata = response.metadata
    
    # Check for required Gemini metadata fields
    required_fields = ["total_token_count"]
    for field in required_fields:
      if field not in metadata:
        raise ResponseValidationError(
          f"Gemini response missing required metadata field '{field}'",
          provider=response.provider,
          response_data=response
        )
      
      if not isinstance(metadata[field], int) or metadata[field] < 0:
        raise ResponseValidationError(
          f"Gemini metadata field '{field}' must be a non-negative integer, "
          f"got {metadata[field]}",
          provider=response.provider,
          response_data=response
        )
    
    # Validate token count consistency
    if response.token_count is not None:
      if response.token_count != metadata["total_token_count"]:
        raise ResponseValidationError(
          f"Gemini token count mismatch: response.token_count={response.token_count}, "
          f"metadata.total_token_count={metadata['total_token_count']}",
          provider=response.provider,
          response_data=response
        )
    
    # Validate finish reason if present
    if response.finish_reason:
      valid_finish_reasons = ["STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"]
      if response.finish_reason not in valid_finish_reasons:
        logger.warning(
          f"Unexpected Gemini finish reason: {response.finish_reason} "
          f"(expected one of: {valid_finish_reasons})"
        )

  def _is_json_serializable(self, value: Any) -> bool:
    """Check if a value is JSON serializable.
    
    Args:
      value: The value to check
      
    Returns:
      True if the value is JSON serializable, False otherwise
    """
    try:
      import json
      json.dumps(value)
      return True
    except (TypeError, ValueError):
      return False

  def get_validation_summary(self, response: ModelResponse) -> Dict[str, Any]:
    """Get a summary of validation results without raising errors.
    
    Args:
      response: The response to analyze
      
    Returns:
      Dictionary containing validation summary information
    """
    summary = {
      "is_valid": False,
      "errors": [],
      "warnings": [],
      "provider": response.provider if hasattr(response, 'provider') else None,
      "model": response.model if hasattr(response, 'model') else None,
    }
    
    try:
      self.validate_response(response)
      summary["is_valid"] = True
    except ResponseValidationError as e:
      summary["errors"].append(str(e))
    except Exception as e:
      summary["errors"].append(f"Unexpected validation error: {e}")
    
    return summary