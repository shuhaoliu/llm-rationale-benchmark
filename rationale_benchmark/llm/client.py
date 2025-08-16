"""Main LLM client interface with concurrent processing support."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config.loader import ConfigLoader
from .config.validator import ConfigValidator
from .concurrent.manager import ConcurrentLLMManager
from .concurrent.validator import ResponseValidator
from .exceptions import (
    ConfigurationError,
    LLMConnectorError,
    StreamingNotSupportedError,
)
from .factory import ProviderFactory
from .http.client import HTTPClient
from .models import LLMConfig, ModelRequest, ModelResponse

logger = logging.getLogger(__name__)


class LLMClient:
  """Main interface for LLM operations with concurrent processing support.
  
  This class provides the primary interface for interacting with multiple LLM
  providers through a unified API. It handles configuration loading, provider
  initialization, request validation, and concurrent processing coordination.
  
  Key features:
  - Configuration-driven provider setup
  - Streaming parameter validation and removal
  - Concurrent processing across different providers
  - Sequential processing within each provider
  - Comprehensive error handling and validation
  """

  def __init__(
    self,
    config_dir: Optional[Path] = None,
    config_name: Optional[str] = None,
    max_connections: int = 100,
    request_timeout: Optional[float] = None
  ):
    """Initialize the LLM client with configuration and HTTP settings.
    
    Args:
      config_dir: Directory containing LLM configuration files.
                 Defaults to 'config/llms' relative to current directory.
      config_name: Name of configuration file without .yaml extension.
                  Defaults to 'default-llms'.
      max_connections: Maximum number of HTTP connections in the pool.
      request_timeout: Optional timeout for individual requests in seconds.
      
    Raises:
      ConfigurationError: If configuration loading or validation fails
      LLMConnectorError: If client initialization fails
    """
    # Set default configuration directory
    if config_dir is None:
      config_dir = Path("config/llms")
    
    # Set default configuration name
    if config_name is None:
      config_name = "default-llms"
    
    self.config_dir = Path(config_dir)
    self.config_name = config_name
    self.max_connections = max_connections
    self.request_timeout = request_timeout
    
    # Initialize components
    self.config_loader = ConfigLoader(self.config_dir)
    self.config_validator = ConfigValidator()
    self.http_client: Optional[HTTPClient] = None
    self.provider_factory: Optional[ProviderFactory] = None
    self.concurrent_manager: Optional[ConcurrentLLMManager] = None
    self.config: Optional[LLMConfig] = None
    
    # Track initialization state
    self._is_initialized = False
    self._is_shutdown = False
    
    logger.info(
      f"LLMClient created with config_dir={config_dir}, "
      f"config_name={config_name}, max_connections={max_connections}"
    )

  async def initialize(self) -> None:
    """Initialize the client by loading configuration and setting up providers.
    
    This method performs the complete initialization sequence:
    1. Load and validate configuration
    2. Initialize HTTP client and provider factory
    3. Set up concurrent manager with provider queues
    4. Validate streaming parameter removal
    
    Raises:
      ConfigurationError: If configuration loading or validation fails
      LLMConnectorError: If provider initialization fails
    """
    if self._is_initialized:
      logger.warning("LLMClient is already initialized")
      return
    
    if self._is_shutdown:
      raise LLMConnectorError("Cannot initialize a shut down LLMClient")
    
    logger.info("Initializing LLMClient")
    
    try:
      # Load configuration
      self.config = self.config_loader.load_config(self.config_name)
      logger.debug(f"Loaded configuration with {len(self.config.providers)} providers")
      
      # Validate configuration
      validation_errors = self.config_validator.validate_config(self.config)
      if validation_errors:
        raise ConfigurationError(
          f"Configuration validation failed:\n" + "\n".join(validation_errors)
        )
      
      # Validate environment variables
      env_errors = self.config_validator.validate_environment_variables(self.config)
      if env_errors:
        raise ConfigurationError(
          f"Environment variable validation failed:\n" + "\n".join(env_errors)
        )
      
      # Initialize HTTP client
      self.http_client = HTTPClient(
        max_connections=self.max_connections,
        timeout=self.request_timeout or 30
      )
      
      # Initialize provider factory
      self.provider_factory = ProviderFactory(self.http_client)
      self.provider_factory.initialize_providers(self.config)
      
      # Initialize response validator
      response_validator = ResponseValidator()
      
      # Initialize concurrent manager
      providers = self.provider_factory.list_providers()
      model_to_provider = self.provider_factory.get_model_to_provider_mapping()
      
      self.concurrent_manager = ConcurrentLLMManager(
        providers=providers,
        response_validator=response_validator,
        model_to_provider=model_to_provider,
        request_timeout=self.request_timeout
      )
      
      # Validate streaming parameter removal at client level
      self._validate_no_streaming_configuration()
      
      self._is_initialized = True
      
      logger.info(
        f"LLMClient initialized successfully with {len(providers)} providers "
        f"and {len(model_to_provider)} models"
      )
      
    except Exception as e:
      # Clean up on initialization failure
      await self._cleanup_on_error()
      
      if isinstance(e, (ConfigurationError, LLMConnectorError)):
        raise
      else:
        raise LLMConnectorError(f"Failed to initialize LLMClient: {str(e)}") from e

  def _validate_no_streaming_configuration(self) -> None:
    """Validate that no streaming parameters are present in configuration.
    
    This method implements streaming parameter validation and removal at the
    client level as an additional safety layer (Requirement 11.8).
    
    Raises:
      StreamingNotSupportedError: If streaming parameters are found
    """
    if not self.config:
      return
    
    streaming_params = {
      "stream", "streaming", "stream_options", "stream_usage",
      "stream_callback", "stream_handler", "incremental"
    }
    
    found_streaming_params = []
    
    # Check defaults section
    for param in streaming_params:
      if param in self.config.defaults:
        found_streaming_params.append(f"defaults.{param}")
    
    # Check each provider configuration
    for provider_name, provider_config in self.config.providers.items():
      # Check default_params
      for param in streaming_params:
        if param in provider_config.default_params:
          found_streaming_params.append(f"providers.{provider_name}.default_params.{param}")
      
      # Check provider_specific
      for param in streaming_params:
        if param in provider_config.provider_specific:
          found_streaming_params.append(f"providers.{provider_name}.provider_specific.{param}")
    
    if found_streaming_params:
      logger.warning(f"Removing streaming parameters from configuration: {found_streaming_params}")
      
      # Remove streaming parameters from configuration
      for param in streaming_params:
        self.config.defaults.pop(param, None)
        
        for provider_config in self.config.providers.values():
          provider_config.default_params.pop(param, None)
          provider_config.provider_specific.pop(param, None)
      
      # Log warning about streaming limitations
      logger.warning(
        "Streaming parameters have been removed from configuration. "
        "This LLM connector does not support streaming responses."
      )

  def _validate_request_no_streaming(self, request: ModelRequest) -> ModelRequest:
    """Validate and clean streaming parameters from a request.
    
    This method implements comprehensive request validation including streaming
    parameter removal as specified in the requirements.
    
    Args:
      request: The model request to validate
      
    Returns:
      Cleaned ModelRequest with streaming parameters removed
      
    Raises:
      StreamingNotSupportedError: If streaming parameters are detected
    """
    streaming_params = {
      "stream", "streaming", "stream_options", "stream_usage",
      "stream_callback", "stream_handler", "incremental"
    }
    
    # Check for streaming parameters in provider_specific
    found_streaming_params = []
    cleaned_provider_specific = {}
    
    for key, value in request.provider_specific.items():
      if key in streaming_params:
        found_streaming_params.append(key)
        logger.warning(f"Blocked streaming parameter '{key}' in request")
      else:
        cleaned_provider_specific[key] = value
    
    if found_streaming_params:
      # Create new request with cleaned provider_specific
      cleaned_request = ModelRequest(
        prompt=request.prompt,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        system_prompt=request.system_prompt,
        stop_sequences=request.stop_sequences,
        provider_specific=cleaned_provider_specific
      )
      
      logger.warning(
        f"Removed streaming parameters from request: {found_streaming_params}"
      )
      
      return cleaned_request
    
    return request

  async def _cleanup_on_error(self) -> None:
    """Clean up resources on initialization error."""
    try:
      if self.concurrent_manager:
        await self.concurrent_manager.shutdown()
      if self.http_client:
        await self.http_client.close()
    except Exception as e:
      logger.error(f"Error during cleanup: {e}")

  def _ensure_initialized(self) -> None:
    """Ensure the client is initialized before operations.
    
    Raises:
      LLMConnectorError: If client is not initialized or has been shut down
    """
    if self._is_shutdown:
      raise LLMConnectorError("LLMClient has been shut down")
    
    if not self._is_initialized:
      raise LLMConnectorError(
        "LLMClient is not initialized. Call initialize() first."
      )

  def get_available_models(self) -> List[str]:
    """Get list of all available models across all providers.
    
    Returns:
      List of model names available across all providers
      
    Raises:
      LLMConnectorError: If client is not initialized
    """
    self._ensure_initialized()
    
    if not self.provider_factory:
      raise LLMConnectorError("Provider factory not initialized")
    
    return self.provider_factory.list_models()

  def get_provider_for_model(self, model: str) -> Optional[str]:
    """Get the provider name for a specific model.
    
    Args:
      model: The model name
      
    Returns:
      Provider name for the model, or None if not found
      
    Raises:
      LLMConnectorError: If client is not initialized
    """
    self._ensure_initialized()
    
    if not self.concurrent_manager:
      raise LLMConnectorError("Concurrent manager not initialized")
    
    return self.concurrent_manager.get_provider_for_model(model)

  def get_client_status(self) -> Dict[str, Any]:
    """Get comprehensive client status information.
    
    Returns:
      Dictionary containing client status and provider information
      
    Raises:
      LLMConnectorError: If client is not initialized
    """
    self._ensure_initialized()
    
    status = {
      "is_initialized": self._is_initialized,
      "is_shutdown": self._is_shutdown,
      "config_name": self.config_name,
      "config_dir": str(self.config_dir),
      "max_connections": self.max_connections,
      "request_timeout": self.request_timeout,
      "available_models": len(self.get_available_models()) if self.provider_factory else 0,
      "providers": {}
    }
    
    if self.concurrent_manager:
      status.update(self.concurrent_manager.get_overall_status())
    
    if self.provider_factory:
      status["provider_discovery"] = self.provider_factory.get_provider_discovery_info()
    
    return status

  async def shutdown(self) -> None:
    """Shutdown the client and clean up all resources.
    
    This method performs graceful shutdown of all components:
    1. Shutdown concurrent manager and provider queues
    2. Close HTTP client connections
    3. Mark client as shut down
    """
    if self._is_shutdown:
      logger.warning("LLMClient is already shut down")
      return
    
    logger.info("Shutting down LLMClient")
    
    try:
      # Shutdown concurrent manager first
      if self.concurrent_manager:
        await self.concurrent_manager.shutdown()
      
      # Close HTTP client
      if self.http_client:
        await self.http_client.close()
      
      # Mark as shut down
      self._is_shutdown = True
      self._is_initialized = False
      
      logger.info("LLMClient shutdown complete")
      
    except Exception as e:
      logger.error(f"Error during LLMClient shutdown: {e}")
      raise LLMConnectorError(f"Failed to shutdown LLMClient: {str(e)}") from e

  def __str__(self) -> str:
    """String representation of the client."""
    return (
      f"LLMClient("
      f"config={self.config_name}, "
      f"initialized={self._is_initialized}, "
      f"shutdown={self._is_shutdown})"
    )

  def __repr__(self) -> str:
    """Detailed string representation of the client."""
    return (
      f"LLMClient("
      f"config_dir={self.config_dir}, "
      f"config_name={self.config_name}, "
      f"max_connections={self.max_connections}, "
      f"initialized={self._is_initialized}, "
      f"shutdown={self._is_shutdown})"
    )

  async def __aenter__(self):
    """Async context manager entry."""
    await self.initialize()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Async context manager exit."""
    await self.shutdown()

  async def generate_response(self, request: ModelRequest) -> ModelResponse:
    """Generate a single response from an LLM provider.
    
    This method routes single requests through provider queues with comprehensive
    validation and error handling. It implements:
    - Request validation including streaming parameter removal
    - Provider routing based on model name
    - Response validation at client level as additional safety layer
    - Detailed error handling with provider context
    
    Args:
      request: The model request to process
      
    Returns:
      Model response from the appropriate provider
      
    Raises:
      LLMConnectorError: If client is not initialized or request fails
      StreamingNotSupportedError: If streaming parameters are detected
      ValueError: If model is not supported or request is invalid
    """
    self._ensure_initialized()
    
    if not self.concurrent_manager:
      raise LLMConnectorError("Concurrent manager not initialized")
    
    # Validate and clean request (remove streaming parameters)
    try:
      cleaned_request = self._validate_request_no_streaming(request)
    except Exception as e:
      raise LLMConnectorError(f"Request validation failed: {str(e)}") from e
    
    # Validate model is supported
    provider_name = self.get_provider_for_model(cleaned_request.model)
    if provider_name is None:
      available_models = self.get_available_models()
      raise ValueError(
        f"Model '{cleaned_request.model}' is not supported. "
        f"Available models: {available_models}"
      )
    
    logger.debug(
      f"Routing request for model {cleaned_request.model} to provider {provider_name}"
    )
    
    try:
      # Submit request to concurrent manager
      future = await self.concurrent_manager.submit_request(cleaned_request)
      
      # Wait for response
      response = await future
      
      # Additional response validation at client level (safety layer)
      self._validate_response_structure(response, cleaned_request, provider_name)
      
      logger.debug(
        f"Successfully generated response for model {cleaned_request.model} "
        f"(latency: {response.latency_ms}ms, tokens: {response.token_count})"
      )
      
      return response
      
    except asyncio.TimeoutError as e:
      raise LLMConnectorError(
        f"Request timeout for model {cleaned_request.model} "
        f"(provider: {provider_name})"
      ) from e
    
    except Exception as e:
      # Add provider context to error
      error_msg = f"Request failed for model {cleaned_request.model} (provider: {provider_name}): {str(e)}"
      
      if "rate limit" in str(e).lower():
        raise LLMConnectorError(f"Rate limit exceeded - {error_msg}") from e
      elif "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
        raise LLMConnectorError(f"Authentication failed - {error_msg}") from e
      elif "not found" in str(e).lower() or "model" in str(e).lower():
        raise LLMConnectorError(f"Model not available - {error_msg}") from e
      else:
        raise LLMConnectorError(error_msg) from e

  def _validate_response_structure(
    self, 
    response: ModelResponse, 
    request: ModelRequest, 
    provider_name: str
  ) -> None:
    """Validate response structure at client level as additional safety layer.
    
    This method provides comprehensive response validation to ensure the response
    meets all requirements and contains valid data.
    
    Args:
      response: The response to validate
      request: The original request for context
      provider_name: Name of the provider that generated the response
      
    Raises:
      LLMConnectorError: If response validation fails
    """
    try:
      # Validate basic response structure
      if not response.text or not response.text.strip():
        raise ValueError("Response text is empty or whitespace only")
      
      if response.model != request.model:
        raise ValueError(
          f"Response model '{response.model}' does not match "
          f"request model '{request.model}'"
        )
      
      if response.provider != provider_name:
        raise ValueError(
          f"Response provider '{response.provider}' does not match "
          f"expected provider '{provider_name}'"
        )
      
      if response.latency_ms < 0:
        raise ValueError(f"Invalid latency: {response.latency_ms}ms")
      
      if response.token_count is not None and response.token_count < 0:
        raise ValueError(f"Invalid token count: {response.token_count}")
      
      if response.cost_estimate is not None and response.cost_estimate < 0:
        raise ValueError(f"Invalid cost estimate: {response.cost_estimate}")
      
      # Validate timestamp is reasonable (not too far in past/future)
      import time
      current_time = time.time()
      response_time = response.timestamp.timestamp()
      
      # Allow 1 hour tolerance for clock differences
      if abs(current_time - response_time) > 3600:
        logger.warning(
          f"Response timestamp seems incorrect: {response.timestamp} "
          f"(current time: {time.ctime(current_time)})"
        )
      
      logger.debug(f"Response validation passed for {response.model}")
      
    except Exception as e:
      raise LLMConnectorError(
        f"Response validation failed for {request.model} "
        f"(provider: {provider_name}): {str(e)}"
      ) from e

  async def generate_responses_concurrent(
    self, 
    requests: List[ModelRequest]
  ) -> List[ModelResponse]:
    """Generate responses for multiple requests concurrently.
    
    This method processes multiple requests with concurrent execution across
    different LLMs while maintaining sequential processing per LLM. It implements:
    - Comprehensive validation for all requests before processing
    - Concurrent execution across different providers
    - Sequential processing within each provider's queue
    - Response order preservation regardless of completion timing
    - Detailed error handling for individual request failures
    
    Args:
      requests: List of model requests to process concurrently
      
    Returns:
      List of model responses in the same order as input requests
      
    Raises:
      LLMConnectorError: If client is not initialized or processing fails
      ValueError: If any request is invalid or uses unsupported model
    """
    self._ensure_initialized()
    
    if not self.concurrent_manager:
      raise LLMConnectorError("Concurrent manager not initialized")
    
    if not requests:
      return []
    
    logger.info(f"Processing {len(requests)} requests concurrently")
    
    # Comprehensive validation for all requests before processing
    validated_requests = []
    validation_errors = []
    
    for i, request in enumerate(requests):
      try:
        # Validate and clean each request
        cleaned_request = self._validate_request_no_streaming(request)
        
        # Validate model is supported
        provider_name = self.get_provider_for_model(cleaned_request.model)
        if provider_name is None:
          available_models = self.get_available_models()
          validation_errors.append(
            f"Request {i}: Model '{cleaned_request.model}' is not supported. "
            f"Available models: {available_models}"
          )
          continue
        
        validated_requests.append((i, cleaned_request, provider_name))
        
      except Exception as e:
        validation_errors.append(f"Request {i}: Validation failed - {str(e)}")
    
    # Raise error if any validation failed
    if validation_errors:
      raise ValueError(
        f"Request validation failed:\n" + "\n".join(validation_errors)
      )
    
    # Log provider distribution
    provider_counts = {}
    for _, request, provider_name in validated_requests:
      provider_counts[provider_name] = provider_counts.get(provider_name, 0) + 1
    
    logger.info(
      f"Distributing {len(validated_requests)} requests across "
      f"{len(provider_counts)} providers: {provider_counts}"
    )
    
    try:
      # Submit all requests to concurrent manager
      request_list = [request for _, request, _ in validated_requests]
      responses = await self.concurrent_manager.process_requests_concurrent(request_list)
      
      # Validate all responses at client level
      validated_responses = []
      for i, (original_index, request, provider_name) in enumerate(validated_requests):
        response = responses[i]
        
        if isinstance(response, Exception):
          # Create error response for exceptions returned by gather
          error_response = self._create_error_response(
            request, provider_name, f"Request failed with exception: {str(response)}"
          )
          validated_responses.append(error_response)
          logger.error(
            f"Request {original_index} failed with exception "
            f"(model: {request.model}): {response}"
          )
        else:
          try:
            # Additional response validation at client level
            self._validate_response_structure(response, request, provider_name)
            validated_responses.append(response)

          except Exception as e:
            # Create error response for failed validation
            error_response = self._create_error_response(
              request, provider_name, f"Response validation failed: {str(e)}"
            )
            validated_responses.append(error_response)
            logger.error(
              f"Response validation failed for request {original_index} "
              f"(model: {request.model}): {e}"
            )
      
      # Log completion statistics
      successful_responses = sum(
        1 for r in validated_responses 
        if not r.text.startswith("ERROR:")
      )
      
      logger.info(
        f"Completed concurrent processing: {successful_responses}/{len(requests)} "
        f"successful responses"
      )
      
      return validated_responses
      
    except Exception as e:
      # Handle catastrophic failures
      error_msg = f"Concurrent processing failed: {str(e)}"
      logger.error(error_msg)
      
      # Create error responses for all requests
      error_responses = []
      for _, request, provider_name in validated_requests:
        error_response = self._create_error_response(request, provider_name, error_msg)
        error_responses.append(error_response)
      
      return error_responses

  def _create_error_response(
    self, 
    request: ModelRequest, 
    provider_name: str, 
    error_message: str
  ) -> ModelResponse:
    """Create an error response for a failed request.
    
    Args:
      request: The original request that failed
      provider_name: Name of the provider that should have handled the request
      error_message: Description of the error
      
    Returns:
      ModelResponse with error information
    """
    from datetime import datetime
    
    return ModelResponse(
      text=f"ERROR: {error_message}",
      model=request.model,
      provider=provider_name,
      timestamp=datetime.now(),
      latency_ms=0,
      token_count=0,
      finish_reason="error",
      cost_estimate=0.0,
      metadata={
        "error": True,
        "error_message": error_message,
        "original_request": {
          "prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt,
          "model": request.model,
          "temperature": request.temperature,
          "max_tokens": request.max_tokens
        }
      }
    )

  async def list_all_models(self) -> Dict[str, Any]:
    """Query all configured providers concurrently to list available models.
    
    This method queries all configured providers concurrently to retrieve their
    available models, handling connectivity issues gracefully without affecting
    other providers. It returns structured results with provider grouping and
    validation status.
    
    Returns:
      Dictionary containing model information grouped by provider with status
      
    Raises:
      LLMConnectorError: If client is not initialized
    """
    self._ensure_initialized()
    
    if not self.provider_factory:
      raise LLMConnectorError("Provider factory not initialized")
    
    providers = self.provider_factory.list_providers()
    
    if not providers:
      return {
        "providers": {},
        "total_providers": 0,
        "successful_providers": 0,
        "total_models": 0,
        "errors": []
      }
    
    logger.info(f"Querying {len(providers)} providers concurrently for available models")
    
    # Create concurrent tasks for each provider
    provider_tasks = {}
    for provider_name, provider in providers.items():
      task = asyncio.create_task(
        self._query_provider_models(provider_name, provider),
        name=f"query_models_{provider_name}"
      )
      provider_tasks[provider_name] = task
    
    # Wait for all tasks to complete (with timeout)
    try:
      # Use a reasonable timeout for model listing
      timeout = 30.0  # 30 seconds total timeout
      results = await asyncio.wait_for(
        asyncio.gather(*provider_tasks.values(), return_exceptions=True),
        timeout=timeout
      )
    except asyncio.TimeoutError:
      logger.warning(f"Model listing timed out after {timeout}s")
      # Cancel remaining tasks
      for task in provider_tasks.values():
        if not task.done():
          task.cancel()
      
      # Get partial results
      results = []
      for task in provider_tasks.values():
        if task.done() and not task.cancelled():
          try:
            results.append(task.result())
          except Exception as e:
            results.append(e)
        else:
          results.append(Exception("Query timed out"))
    
    # Process results
    provider_results = {}
    successful_providers = 0
    total_models = 0
    errors = []
    
    for provider_name, result in zip(provider_tasks.keys(), results):
      if isinstance(result, Exception):
        # Handle provider query failure
        error_msg = f"Provider '{provider_name}': {str(result)}"
        errors.append(error_msg)
        
        provider_results[provider_name] = {
          "status": "error",
          "error": str(result),
          "models": [],
          "model_count": 0,
          "response_time_ms": None
        }
        
        logger.warning(f"Failed to query models from {provider_name}: {result}")
      else:
        # Successful provider query
        provider_results[provider_name] = result
        if result["status"] == "success":
          successful_providers += 1
          total_models += result["model_count"]
    
    # Build final response
    response = {
      "providers": provider_results,
      "total_providers": len(providers),
      "successful_providers": successful_providers,
      "failed_providers": len(providers) - successful_providers,
      "total_models": total_models,
      "errors": errors,
      "query_timestamp": datetime.now().isoformat(),
      "configured_models": self.get_available_models()  # Models from configuration
    }
    
    logger.info(
      f"Model listing complete: {successful_providers}/{len(providers)} providers "
      f"successful, {total_models} total models discovered"
    )
    
    return response

  async def _query_provider_models(
    self, 
    provider_name: str, 
    provider: Any
  ) -> Dict[str, Any]:
    """Query a single provider for available models.
    
    This method handles individual provider queries with proper error handling
    and response validation.
    
    Args:
      provider_name: Name of the provider
      provider: Provider instance
      
    Returns:
      Dictionary containing provider query results
    """
    import time
    
    start_time = time.time()
    
    try:
      # Query provider for models
      models = await provider.list_models()
      
      # Validate response
      if not isinstance(models, list):
        raise ValueError(f"Provider returned invalid model list type: {type(models)}")
      
      # Filter out empty or invalid model names
      valid_models = []
      for model in models:
        if isinstance(model, str) and model.strip():
          valid_models.append(model.strip())
        else:
          logger.warning(f"Provider {provider_name} returned invalid model: {model}")
      
      response_time_ms = int((time.time() - start_time) * 1000)
      
      result = {
        "status": "success",
        "models": valid_models,
        "model_count": len(valid_models),
        "response_time_ms": response_time_ms,
        "provider_info": {
          "base_url": getattr(provider.config, 'base_url', None),
          "timeout": provider.config.timeout,
          "max_retries": provider.config.max_retries
        }
      }
      
      logger.debug(
        f"Successfully queried {len(valid_models)} models from {provider_name} "
        f"in {response_time_ms}ms"
      )
      
      return result
      
    except asyncio.TimeoutError:
      response_time_ms = int((time.time() - start_time) * 1000)
      return {
        "status": "timeout",
        "error": "Query timed out",
        "models": [],
        "model_count": 0,
        "response_time_ms": response_time_ms
      }
    
    except Exception as e:
      response_time_ms = int((time.time() - start_time) * 1000)
      
      # Categorize error types
      error_type = "unknown"
      if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
        error_type = "authentication"
      elif "network" in str(e).lower() or "connection" in str(e).lower():
        error_type = "network"
      elif "rate limit" in str(e).lower():
        error_type = "rate_limit"
      elif "not found" in str(e).lower():
        error_type = "not_found"
      
      return {
        "status": "error",
        "error": str(e),
        "error_type": error_type,
        "models": [],
        "model_count": 0,
        "response_time_ms": response_time_ms
      }