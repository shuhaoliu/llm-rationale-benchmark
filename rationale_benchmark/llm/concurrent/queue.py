"""Provider request queue for sequential processing per provider."""

import asyncio
import logging
from typing import Any, Dict, Optional

from rationale_benchmark.llm.models import ModelRequest, ModelResponse
from rationale_benchmark.llm.providers.base import LLMProvider
from rationale_benchmark.llm.concurrent.validator import ResponseValidator

logger = logging.getLogger(__name__)


class ProviderRequestQueue:
  """Manages sequential processing of requests for a single LLM provider.
  
  This class implements a FIFO queue for processing requests to a single LLM provider
  sequentially, ensuring that only one request is processed at a time per provider.
  It uses asyncio.Queue for thread-safe request queuing and asyncio.Future for
  coordinating between request submission and response delivery.
  """

  def __init__(
    self, 
    provider: LLMProvider, 
    response_validator: ResponseValidator,
    request_timeout: Optional[float] = None
  ):
    """Initialize the provider request queue.
    
    Args:
      provider: The LLM provider to process requests for
      response_validator: Validator for response structure validation
      request_timeout: Optional timeout for individual requests in seconds
    """
    self.provider = provider
    self.response_validator = response_validator
    self.request_timeout = request_timeout
    
    # FIFO queue for request processing
    self._request_queue: asyncio.Queue = asyncio.Queue()
    
    # Statistics tracking
    self._total_processed = 0
    self._is_processing = False
    self._current_request: Optional[ModelRequest] = None
    self._current_future: Optional[asyncio.Future] = None
    
    # Worker task for processing requests
    self._worker_task = asyncio.create_task(self._process_requests())
    
    logger.debug(f"Initialized ProviderRequestQueue for {provider.name}")

  async def submit_request(self, request: ModelRequest) -> asyncio.Future[ModelResponse]:
    """Submit a request for processing and return a future for the response.
    
    Args:
      request: The model request to process
      
    Returns:
      Future that will contain the model response when processing completes
      
    Raises:
      RuntimeError: If the queue has been shut down
    """
    if self._worker_task.done():
      raise RuntimeError("Queue has been shut down")
    
    # Create future for response coordination
    response_future: asyncio.Future[ModelResponse] = asyncio.Future()
    
    # Package request with its future
    request_item = {
      "request": request,
      "future": response_future
    }
    
    # Add to queue
    await self._request_queue.put(request_item)
    
    logger.debug(
      f"Submitted request for model {request.model} to {self.provider.name} queue "
      f"(queue size: {self._request_queue.qsize()})"
    )
    
    return response_future

  async def _process_requests(self) -> None:
    """Worker coroutine that processes requests sequentially from the queue.
    
    This method runs continuously, processing requests one at a time in FIFO order.
    It handles errors gracefully and ensures that futures are properly resolved.
    """
    logger.info(f"Started request processing worker for {self.provider.name}")
    
    try:
      while True:
        try:
          # Get next request from queue
          request_item = await self._request_queue.get()
          request = request_item["request"]
          future = request_item["future"]
          
          # Check if future was cancelled while waiting in queue
          if future.cancelled():
            logger.debug(f"Skipping cancelled request for {self.provider.name}")
            self._request_queue.task_done()
            continue
          
          # Mark as processing
          self._is_processing = True
          self._current_request = request
          self._current_future = future
          
          logger.debug(
            f"Processing request for model {request.model} on {self.provider.name}"
          )
          
          try:
            # Process the request with optional timeout
            if self.request_timeout:
              response = await asyncio.wait_for(
                self.provider.generate_response(request),
                timeout=self.request_timeout
              )
            else:
              response = await self.provider.generate_response(request)
            
            # Validate response before returning
            self.response_validator.validate_response(response)
            
            # Set successful result
            if not future.cancelled():
              future.set_result(response)
            
            logger.debug(
              f"Successfully processed request for model {request.model} "
              f"on {self.provider.name} (latency: {response.latency_ms}ms)"
            )
            
          except asyncio.CancelledError:
            # Worker was cancelled, cancel the future too
            if not future.cancelled():
              future.cancel()
            raise  # Re-raise to exit the worker loop
          except Exception as e:
            # Set error result
            if not future.cancelled():
              future.set_exception(e)
            
            logger.error(
              f"Error processing request for model {request.model} "
              f"on {self.provider.name}: {e}"
            )
          
          finally:
            # Update statistics and mark as done
            self._total_processed += 1
            self._is_processing = False
            self._current_request = None
            self._current_future = None
            self._request_queue.task_done()
            
        except asyncio.CancelledError:
          logger.info(f"Request processing worker for {self.provider.name} cancelled")
          # Cancel the current future if it exists
          if hasattr(self, '_current_future') and self._current_future and not self._current_future.done():
            self._current_future.cancel()
          break
        except Exception as e:
          logger.error(
            f"Unexpected error in request processing worker for {self.provider.name}: {e}"
          )
          # Continue processing other requests
          continue
          
    except Exception as e:
      logger.error(
        f"Fatal error in request processing worker for {self.provider.name}: {e}"
      )
    finally:
      logger.info(f"Request processing worker for {self.provider.name} stopped")

  async def shutdown(self) -> None:
    """Shutdown the queue and cancel all pending requests.
    
    This method stops the worker task and cancels all pending request futures.
    """
    logger.info(f"Shutting down ProviderRequestQueue for {self.provider.name}")
    
    # Cancel the worker task
    if not self._worker_task.done():
      self._worker_task.cancel()
      try:
        await self._worker_task
      except asyncio.CancelledError:
        pass
    
    # Cancel all pending requests in the queue
    pending_requests = []
    while not self._request_queue.empty():
      try:
        request_item = self._request_queue.get_nowait()
        pending_requests.append(request_item)
        self._request_queue.task_done()
      except asyncio.QueueEmpty:
        break
    
    # Cancel futures for pending requests
    cancelled_count = 0
    for request_item in pending_requests:
      future = request_item["future"]
      if not future.done():
        future.cancel()
        cancelled_count += 1
    
    # Cancel currently processing request if any
    if self._current_future and not self._current_future.done():
      self._current_future.cancel()
      cancelled_count += 1
    
    logger.info(
      f"Shutdown complete for {self.provider.name} "
      f"(cancelled {cancelled_count} pending requests)"
    )

  def get_status(self) -> Dict[str, Any]:
    """Get current status information for the queue.
    
    Returns:
      Dictionary containing queue status information
    """
    return {
      "provider": self.provider.name,
      "queue_size": self._request_queue.qsize(),
      "is_processing": self._is_processing,
      "total_processed": self._total_processed,
      "current_request_model": self._current_request.model if self._current_request else None,
      "worker_running": not self._worker_task.done()
    }

  def __str__(self) -> str:
    """String representation of the queue."""
    return f"ProviderRequestQueue(provider='{self.provider.name}', queue_size={self._request_queue.qsize()})"

  def __repr__(self) -> str:
    """Detailed string representation of the queue."""
    return (
      f"ProviderRequestQueue("
      f"provider='{self.provider.name}', "
      f"queue_size={self._request_queue.qsize()}, "
      f"is_processing={self._is_processing}, "
      f"total_processed={self._total_processed})"
    )