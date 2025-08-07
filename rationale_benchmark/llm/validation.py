"""Response validation utilities for LLM connector module."""

import logging
from typing import Any, Dict, List, Optional, Union

from .exceptions import ResponseValidationError

logger = logging.getLogger(__name__)


class ResponseValidator:
  """Comprehensive response validation utility for LLM providers."""
  
  def __init__(self, provider: str):
    """Initialize ResponseValidator for a specific provider.
    
    Args:
      provider: Name of the LLM provider
    """
    self.provider = provider
    self.validation_errors: List[ResponseValidationError] = []
  
  def validate_response_structure(
    self, 
    response_data: Dict[str, Any],
    required_fields: List[str],
    field_validators: Optional[Dict[str, callable]] = None,
  ) -> None:
    """Validate basic response structure and required fields.
    
    Args:
      response_data: Raw response data from provider
      required_fields: List of required field names
      field_validators: Optional dict mapping field names to validation functions
      
    Raises:
      ResponseValidationError: If validation fails
    """
    errors = []
    field_errors = {}
    validation_context = {
      "response_type": type(response_data).__name__,
      "response_keys": list(response_data.keys()) if isinstance(response_data, dict) else [],
      "required_fields": required_fields,
    }
    
    # Check if response is a dictionary
    if not isinstance(response_data, dict):
      error = ResponseValidationError(
        message=f"Response must be a dictionary, got {type(response_data).__name__}",
        provider=self.provider,
        response_data=response_data,
        validation_context=validation_context,
      )
      error.add_recovery_suggestion("Check API response format and ensure it returns a JSON object")
      raise error
    
    # Check for required fields
    for field in required_fields:
      if field not in response_data:
        field_errors[field] = f"Required field '{field}' is missing from response"
    
    # Run custom field validators if provided
    if field_validators:
      for field, validator in field_validators.items():
        if field in response_data:
          try:
            validator(response_data[field])
          except Exception as e:
            field_errors[field] = f"Field validation failed: {str(e)}"
    
    # If there are field errors, create comprehensive error
    if field_errors:
      error = ResponseValidationError(
        message=f"Response validation failed for {len(field_errors)} fields",
        provider=self.provider,
        response_data=response_data,
        field_errors=field_errors,
        validation_context=validation_context,
      )
      
      # Add recovery suggestions based on common issues
      if "choices" in field_errors:
        error.add_recovery_suggestion("Ensure the API request includes proper model and message parameters")
      if "content" in field_errors:
        error.add_recovery_suggestion("Check if the model response was truncated or incomplete")
      if "usage" in field_errors:
        error.add_recovery_suggestion("Verify the API response includes token usage information")
      
      raise error
  
  def validate_openai_response(self, response_data: Dict[str, Any]) -> None:
    """Validate OpenAI-specific response structure.
    
    Args:
      response_data: Raw OpenAI API response
      
    Raises:
      ResponseValidationError: If validation fails
    """
    required_fields = ["choices", "model", "usage", "object"]
    
    def validate_choices(choices: Any) -> None:
      if not isinstance(choices, list) or not choices:
        raise ValueError("must be a non-empty array")
      
      choice = choices[0]
      if not isinstance(choice, dict):
        raise ValueError("first choice must be a dictionary")
      
      required_choice_fields = ["message", "finish_reason", "index"]
      for field in required_choice_fields:
        if field not in choice:
          raise ValueError(f"missing required field '{field}'")
      
      # Validate message structure
      message = choice["message"]
      if not isinstance(message, dict):
        raise ValueError("message must be a dictionary")
      
      if "content" not in message or "role" not in message:
        raise ValueError("message must have 'content' and 'role' fields")
      
      if not isinstance(message["content"], str) or not message["content"].strip():
        raise ValueError("message content must be a non-empty string")
      
      if message["role"] != "assistant":
        raise ValueError(f"expected role 'assistant', got '{message['role']}'")
    
    def validate_usage(usage: Any) -> None:
      if not isinstance(usage, dict):
        raise ValueError("must be a dictionary")
      
      required_usage_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
      for field in required_usage_fields:
        if field not in usage:
          raise ValueError(f"missing required field '{field}'")
        if not isinstance(usage[field], int) or usage[field] < 0:
          raise ValueError(f"field '{field}' must be a non-negative integer")
      
      # Validate token count consistency
      expected_total = usage["prompt_tokens"] + usage["completion_tokens"]
      if usage["total_tokens"] != expected_total:
        raise ValueError(f"token count inconsistency: total={usage['total_tokens']}, expected={expected_total}")
    
    def validate_object_type(obj_type: Any) -> None:
      if obj_type != "chat.completion":
        raise ValueError(f"expected 'chat.completion', got '{obj_type}'")
    
    def validate_model(model: Any) -> None:
      if not isinstance(model, str) or not model:
        raise ValueError("must be a non-empty string")
    
    field_validators = {
      "choices": validate_choices,
      "usage": validate_usage,
      "object": validate_object_type,
      "model": validate_model,
    }
    
    try:
      self.validate_response_structure(response_data, required_fields, field_validators)
    except ResponseValidationError as e:
      # Add OpenAI-specific recovery suggestions
      e.add_recovery_suggestion("Check OpenAI API documentation for correct response format")
      e.add_recovery_suggestion("Verify API key has proper permissions for the requested model")
      e.add_recovery_suggestion("Ensure request parameters match OpenAI API requirements")
      raise
  
  def validate_anthropic_response(self, response_data: Dict[str, Any]) -> None:
    """Validate Anthropic-specific response structure.
    
    Args:
      response_data: Raw Anthropic API response
      
    Raises:
      ResponseValidationError: If validation fails
    """
    required_fields = ["content", "model", "role", "stop_reason", "usage"]
    
    def validate_content(content: Any) -> None:
      if not isinstance(content, list) or not content:
        raise ValueError("must be a non-empty array")
      
      content_block = content[0]
      if not isinstance(content_block, dict):
        raise ValueError("content block must be a dictionary")
      
      if "text" not in content_block or "type" not in content_block:
        raise ValueError("content block must have 'text' and 'type' fields")
      
      if content_block["type"] != "text":
        raise ValueError(f"expected type 'text', got '{content_block['type']}'")
      
      if not isinstance(content_block["text"], str) or not content_block["text"]:
        raise ValueError("text content must be a non-empty string")
    
    def validate_usage(usage: Any) -> None:
      if not isinstance(usage, dict):
        raise ValueError("must be a dictionary")
      
      required_usage_fields = ["input_tokens", "output_tokens"]
      for field in required_usage_fields:
        if field not in usage:
          raise ValueError(f"missing required field '{field}'")
        if not isinstance(usage[field], int) or usage[field] < 0:
          raise ValueError(f"field '{field}' must be a non-negative integer")
    
    def validate_role(role: Any) -> None:
      if role != "assistant":
        raise ValueError(f"expected 'assistant', got '{role}'")
    
    def validate_model(model: Any) -> None:
      if not isinstance(model, str) or not model:
        raise ValueError("must be a non-empty string")
    
    field_validators = {
      "content": validate_content,
      "usage": validate_usage,
      "role": validate_role,
      "model": validate_model,
    }
    
    try:
      self.validate_response_structure(response_data, required_fields, field_validators)
    except ResponseValidationError as e:
      # Add Anthropic-specific recovery suggestions
      e.add_recovery_suggestion("Check Anthropic API documentation for correct response format")
      e.add_recovery_suggestion("Verify API key is valid and has proper permissions")
      e.add_recovery_suggestion("Ensure message format follows Anthropic requirements")
      raise
  
  def validate_gemini_response(self, response_data: Dict[str, Any]) -> None:
    """Validate Gemini-specific response structure.
    
    Args:
      response_data: Raw Gemini API response
      
    Raises:
      ResponseValidationError: If validation fails
    """
    required_fields = ["candidates", "usageMetadata"]
    
    def validate_candidates(candidates: Any) -> None:
      if not isinstance(candidates, list) or not candidates:
        raise ValueError("must be a non-empty array")
      
      candidate = candidates[0]
      if not isinstance(candidate, dict):
        raise ValueError("candidate must be a dictionary")
      
      if "content" not in candidate:
        raise ValueError("candidate must have 'content' field")
      
      content = candidate["content"]
      if not isinstance(content, dict) or "parts" not in content:
        raise ValueError("content must have 'parts' field")
      
      parts = content["parts"]
      if not isinstance(parts, list) or not parts:
        raise ValueError("parts must be a non-empty array")
      
      part = parts[0]
      if not isinstance(part, dict) or "text" not in part:
        raise ValueError("part must have 'text' field")
      
      if not isinstance(part["text"], str) or not part["text"]:
        raise ValueError("text must be a non-empty string")
    
    def validate_usage_metadata(usage: Any) -> None:
      if not isinstance(usage, dict):
        raise ValueError("must be a dictionary")
      
      # Gemini uses different field names for token counts
      if "promptTokenCount" in usage and not isinstance(usage["promptTokenCount"], int):
        raise ValueError("promptTokenCount must be an integer")
      if "candidatesTokenCount" in usage and not isinstance(usage["candidatesTokenCount"], int):
        raise ValueError("candidatesTokenCount must be an integer")
    
    field_validators = {
      "candidates": validate_candidates,
      "usageMetadata": validate_usage_metadata,
    }
    
    try:
      self.validate_response_structure(response_data, required_fields, field_validators)
    except ResponseValidationError as e:
      # Add Gemini-specific recovery suggestions
      e.add_recovery_suggestion("Check Google AI API documentation for correct response format")
      e.add_recovery_suggestion("Verify API key and project configuration")
      e.add_recovery_suggestion("Ensure request follows Gemini API requirements")
      raise
  
  def validate_provider_response(self, response_data: Dict[str, Any]) -> None:
    """Validate response based on provider type.
    
    Args:
      response_data: Raw provider response
      
    Raises:
      ResponseValidationError: If validation fails
    """
    provider_validators = {
      "openai": self.validate_openai_response,
      "anthropic": self.validate_anthropic_response,
      "gemini": self.validate_gemini_response,
      "openrouter": self.validate_openai_response,  # OpenRouter uses OpenAI format
    }
    
    validator = provider_validators.get(self.provider.lower())
    if validator:
      validator(response_data)
    else:
      # Generic validation for unknown providers
      logger.warning(f"No specific validator for provider '{self.provider}', using generic validation")
      self.validate_response_structure(response_data, ["text", "model"])
  
  def create_validation_error_with_context(
    self,
    message: str,
    response_data: Dict[str, Any],
    additional_context: Optional[Dict[str, Any]] = None,
  ) -> ResponseValidationError:
    """Create a ResponseValidationError with comprehensive context.
    
    Args:
      message: Primary error message
      response_data: Raw response data
      additional_context: Additional context information
      
    Returns:
      ResponseValidationError with full context
    """
    validation_context = {
      "provider": self.provider,
      "response_size": len(str(response_data)),
      "response_keys": list(response_data.keys()) if isinstance(response_data, dict) else [],
    }
    
    if additional_context:
      validation_context.update(additional_context)
    
    error = ResponseValidationError(
      message=message,
      provider=self.provider,
      response_data=response_data,
      validation_context=validation_context,
    )
    
    # Add common recovery suggestions
    error.add_recovery_suggestion("Check provider API documentation for expected response format")
    error.add_recovery_suggestion("Verify API credentials and permissions")
    error.add_recovery_suggestion("Review request parameters for correctness")
    
    return error


def aggregate_validation_errors(
  errors: List[ResponseValidationError],
  provider: Optional[str] = None,
) -> ResponseValidationError:
  """Aggregate multiple validation errors into a single comprehensive error.
  
  Args:
    errors: List of ResponseValidationError instances
    provider: Provider name for the aggregated error
    
  Returns:
    Aggregated ResponseValidationError
    
  Raises:
    ValueError: If errors list is empty
  """
  return ResponseValidationError.create_aggregated_error(errors, provider)


def validate_response_content_quality(
  content: str,
  min_length: int = 1,
  max_length: Optional[int] = None,
) -> List[str]:
  """Validate response content quality and return list of issues.
  
  Args:
    content: Response content to validate
    min_length: Minimum acceptable content length
    max_length: Maximum acceptable content length (optional)
    
  Returns:
    List of validation issues (empty if content is valid)
  """
  issues = []
  
  if not isinstance(content, str):
    issues.append(f"Content must be a string, got {type(content).__name__}")
    return issues
  
  if len(content.strip()) < min_length:
    issues.append(f"Content too short: {len(content.strip())} characters (minimum: {min_length})")
  
  if max_length and len(content) > max_length:
    issues.append(f"Content too long: {len(content)} characters (maximum: {max_length})")
  
  # Check for common quality issues
  if content.strip() == "":
    issues.append("Content is empty or contains only whitespace")
  
  if content.count("Error") > 0 or content.count("error") > 0:
    issues.append("Content appears to contain error messages")
  
  if len(content.strip()) < 10 and not content.strip().endswith((".", "!", "?")):
    issues.append("Very short content without proper sentence ending")
  
  return issues