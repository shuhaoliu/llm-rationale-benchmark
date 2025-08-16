"""Concurrent LLM manager for coordinating execution across providers."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from rationale_benchmark.llm.models import ModelRequest, ModelResponse
from rationale_benchmark.llm.providers.base import LLMProvider
from rationale_benchmark.llm.concurrent.queue import ProviderRequestQueue
from rationale_benchmark.llm.concurrent.validator import ResponseValidator
from rationale_benchmark.llm.logging import get_llm_logger

logger = get_llm_logger(__name__)


class ConcurrentLLMManager:
  """Coordinates concurrent execution across multiple LLM providers.
  
  This class manages multiple provider queues, distributing requests to the
  appropriate provider while maintaining order preservation within each provider
  and enabling concurrent execution across different providers.
  """

  def __init__(
    self,
    providers: Dict[str, LLMProvider],
    response_validator: ResponseValidator,
    model_to_provider: Dict[str, str],
    request_timeout: Optional[float] = None
  ):
    """Initialize the concurrent LLM manager.
    
    Args:
      providers: Dictionary mapping provider names to LLMProvider instances
      response_validator: Validator for response structure validation
      model_to_provider: Mapping from model names to provider names
      request_timeout: Optional timeout for individual requests in seconds
    """
    self.providers = providers
    self.response_validator = response_validator
    self.model_to_provider = model_to_provider
    self.request_timeout = request_timeout
    
    # Create provider queues
    self._provider_queues: Dict[str, ProviderRequestQueue] = {}
    for provider_name, provider in providers.items():
      self._provider_queues[provider_name] = ProviderRequestQueue(
        provider=provider,
        response_validator=response_validator,
        request_timeout=request_timeout
      )
    
    # Track manager state
    self._is_shutdown = False
    
    logger.info(
      f"Initialized ConcurrentLLMManager with {len(providers)} providers: "
      f"{list(providers.keys())}"
    )

  async def submit_request(self, request: ModelRequest) -> asyncio.Future[ModelResponse]:
    """Submit a request for processing and return a future for the response.
    
    This method routes the request to the appropriate provider queue based on
    the model name and returns a future that will contain the response.
    
    Args:
      request: The model request to process
      
    Returns:
      Future that will contain the model response when processing completes
      
    Raises:
      ValueError: If the model is not mapped to any provider
      RuntimeError: If the manager has been shut down
    """
    if self._is_shutdown:
      raise RuntimeError("ConcurrentLLMManager has been shut down")
    
    # Determine which provider to use
    provider_name = self._get_provider_for_model(request.model)
    
    if provider_name not in self._provider_queues:
      raise ValueError(
        f"Provider '{provider_name}' for model '{request.model}' is not available. "
        f"Available providers: {list(self._provider_queues.keys())}"
      )
    
    # Submit to appropriate provider queue
    provider_queue = self._provider_queues[provider_name]
    future = await provider_queue.submit_request(request)
    
    logger.debug(
      f"Submitted request for model {request.model} to {provider_name} provider queue"
    )
    
    return future

  def _get_provider_for_model(self, model: str) -> str:
    """Get the provider name for a given model.
    
    Args:
      model: The model name
      
    Returns:
      The provider name for the model
      
    Raises:
      ValueError: If the model is not mapped to any provider
    """
    if model not in self.model_to_provider:
      available_models = list(self.model_to_provider.keys())
      raise ValueError(
        f"Unknown model '{model}'. Available models: {available_models}"
      )
    
    return self.model_to_provider[model]

  async def submit_multiple_requests(
    self, 
    requests: List[ModelRequest]
  ) -> List[asyncio.Future[ModelResponse]]:
    """Submit multiple requests concurrently and return futures for all responses.
    
    This method submits all requests concurrently and returns a list of futures
    in the same order as the input requests. Requests to different providers
    will execute concurrently, while requests to the same provider will execute
    sequentially within that provider's queue.
    
    Args:
      requests: List of model requests to process
      
    Returns:
      List of futures for the responses in the same order as input requests
      
    Raises:
      ValueError: If any model is not mapped to a provider
      RuntimeError: If the manager has been shut down
    """
    if self._is_shutdown:
      raise RuntimeError("ConcurrentLLMManager has been shut down")
    
    if not requests:
      return []
    
    # Submit all requests and collect futures
    futures = []
    for request in requests:
      future = await self.submit_request(request)
      futures.append(future)
    
    logger.info(
      f"Submitted {len(requests)} requests across "
      f"{len(set(self._get_provider_for_model(req.model) for req in requests))} providers"
    )
    
    return futures

  async def process_requests_concurrent(
    self, 
    requests: List[ModelRequest]
  ) -> List[ModelResponse]:
    """Process multiple requests concurrently and return responses in order.
    
    This is a convenience method that submits requests and waits for all
    responses, returning them in the same order as the input requests.
    
    Args:
      requests: List of model requests to process
      
    Returns:
      List of model responses in the same order as input requests
      
    Raises:
      ValueError: If any model is not mapped to a provider
      RuntimeError: If the manager has been shut down
    """
    if not requests:
      return []
    
    # Log initial queue status
    for provider_name, queue in self._provider_queues.items():
      status = queue.get_status()
      logger.log_queue_status(
        provider=provider_name,
        queue_size=status["queue_size"],
        active_requests=1 if status["worker_running"] else 0,
        total_processed=status["total_processed"],
      )
    
    # Submit all requests
    futures = await self.submit_multiple_requests(requests)
    
    # Wait for all responses
    responses = await asyncio.gather(*futures, return_exceptions=True)
    
    # Log final status
    successful_responses = len([r for r in responses if r is not None])
    logger.info(
      "Concurrent processing completed",
      total_requests=len(requests),
      successful_responses=successful_responses,
      failed_requests=len(requests) - successful_responses,
    )
    
    # Log final queue status
    for provider_name, queue in self._provider_queues.items():
      status = queue.get_status()
      logger.log_queue_status(
        provider=provider_name,
        queue_size=status["queue_size"],
        active_requests=1 if status["worker_running"] else 0,
        total_processed=status["total_processed"],
      )
    
    return responses

  def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
    """Get status information for all provider queues.
    
    Returns:
      Dictionary mapping provider names to their status information
    """
    status = {}
    for provider_name, queue in self._provider_queues.items():
      status[provider_name] = queue.get_status()
    
    return status

  def get_overall_status(self) -> Dict[str, Any]:
    """Get overall manager status information.
    
    Returns:
      Dictionary containing overall manager status
    """
    provider_statuses = self.get_provider_status()
    
    total_queue_size = sum(
      status["queue_size"] for status in provider_statuses.values()
    )
    
    total_processed = sum(
      status["total_processed"] for status in provider_statuses.values()
    )
    
    active_providers = sum(
      1 for status in provider_statuses.values() 
      if status["worker_running"]
    )
    
    return {
      "total_providers": len(self._provider_queues),
      "active_providers": active_providers,
      "total_queue_size": total_queue_size,
      "total_processed": total_processed,
      "is_shutdown": self._is_shutdown,
      "supported_models": list(self.model_to_provider.keys()),
      "providers": provider_statuses
    }

  def get_provider_for_model(self, model: str) -> Optional[str]:
    """Get the provider name for a model without raising an error.
    
    Args:
      model: The model name
      
    Returns:
      The provider name for the model, or None if not found
    """
    return self.model_to_provider.get(model)

  def get_supported_models(self) -> List[str]:
    """Get list of all supported models.
    
    Returns:
      List of supported model names
    """
    return list(self.model_to_provider.keys())

  def get_models_for_provider(self, provider_name: str) -> List[str]:
    """Get list of models supported by a specific provider.
    
    Args:
      provider_name: The provider name
      
    Returns:
      List of model names supported by the provider
    """
    return [
      model for model, provider in self.model_to_provider.items()
      if provider == provider_name
    ]

  async def shutdown(self) -> None:
    """Shutdown the manager and all provider queues.
    
    This method stops all provider queues and cancels any pending requests.
    After shutdown, no new requests can be submitted.
    """
    if self._is_shutdown:
      logger.warning("ConcurrentLLMManager is already shut down")
      return
    
    logger.info("Shutting down ConcurrentLLMManager")
    
    # Mark as shutdown to prevent new requests
    self._is_shutdown = True
    
    # Shutdown all provider queues
    shutdown_tasks = []
    for provider_name, queue in self._provider_queues.items():
      task = asyncio.create_task(queue.shutdown())
      shutdown_tasks.append(task)
    
    # Wait for all queues to shutdown
    if shutdown_tasks:
      await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
    logger.info(
      f"ConcurrentLLMManager shutdown complete "
      f"({len(self._provider_queues)} provider queues stopped)"
    )

  async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
    """Wait for all queues to become empty (all requests processed).
    
    Args:
      timeout: Optional timeout in seconds
      
    Returns:
      True if all queues are empty, False if timeout occurred
    """
    async def all_queues_empty():
      while True:
        status = self.get_provider_status()
        if all(s["queue_size"] == 0 and not s["is_processing"] for s in status.values()):
          return True
        await asyncio.sleep(0.1)
    
    try:
      if timeout:
        await asyncio.wait_for(all_queues_empty(), timeout=timeout)
      else:
        await all_queues_empty()
      return True
    except asyncio.TimeoutError:
      return False

  def __str__(self) -> str:
    """String representation of the manager."""
    return (
      f"ConcurrentLLMManager("
      f"providers={len(self._provider_queues)}, "
      f"models={len(self.model_to_provider)})"
    )

  def __repr__(self) -> str:
    """Detailed string representation of the manager."""
    return (
      f"ConcurrentLLMManager("
      f"providers={list(self._provider_queues.keys())}, "
      f"models={len(self.model_to_provider)}, "
      f"is_shutdown={self._is_shutdown})"
    )