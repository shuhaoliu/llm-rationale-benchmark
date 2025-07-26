"""Retry logic with exponential backoff for HTTP requests."""

import asyncio
import random
import time
from typing import Any, Callable, TypeVar
from aiohttp.client_exceptions import ClientConnectorError, ClientResponseError

from rationale_benchmark.llm.exceptions import RetryExhaustedError

T = TypeVar('T')


class RetryHandler:
  """Handles retry logic with exponential backoff for transient failures."""

  def __init__(
    self,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True
  ):
    """Initialize retry handler with configuration parameters.
    
    Args:
      max_retries: Maximum number of retry attempts
      base_delay: Base delay in seconds for first retry
      max_delay: Maximum delay in seconds between retries
      backoff_factor: Multiplier for exponential backoff
      jitter: Whether to add random jitter to delays
    """
    self.max_retries = max_retries
    self.base_delay = base_delay
    self.max_delay = max_delay
    self.backoff_factor = backoff_factor
    self.jitter = jitter

  async def execute(
    self, 
    func: Callable[..., T], 
    *args: Any, 
    **kwargs: Any
  ) -> T:
    """Execute function with retry logic on transient failures.
    
    Args:
      func: Async function to execute
      *args: Positional arguments to pass to function
      **kwargs: Keyword arguments to pass to function
      
    Returns:
      Result of successful function execution
      
    Raises:
      RetryExhaustedError: When max retries are exceeded
      Exception: Non-retryable exceptions are re-raised immediately
    """
    last_exception = None
    
    for attempt in range(self.max_retries + 1):
      try:
        return await func(*args, **kwargs)
      except Exception as e:
        last_exception = e
        
        # Check if this exception should trigger a retry
        if not self.is_retryable(e):
          raise
        
        # If this was the last attempt, raise RetryExhaustedError
        if attempt == self.max_retries:
          raise RetryExhaustedError(
            f"Max retries ({self.max_retries}) exceeded. Last error: {str(e)}",
            max_retries=self.max_retries,
            last_exception=e
          )
        
        # Calculate delay for next attempt
        delay = self._calculate_delay(attempt + 1)
        await asyncio.sleep(delay)
    
    # This should never be reached, but included for completeness
    raise RetryExhaustedError(
      f"Max retries ({self.max_retries}) exceeded",
      max_retries=self.max_retries,
      last_exception=last_exception
    )

  def is_retryable(self, exception: Exception) -> bool:
    """Determine if an exception should trigger a retry.
    
    Args:
      exception: Exception to check
      
    Returns:
      True if the exception is retryable, False otherwise
    """
    # Network connection errors are retryable
    if isinstance(exception, ClientConnectorError):
      return True
    
    # Timeout errors are retryable
    if isinstance(exception, asyncio.TimeoutError):
      return True
    
    # General OS errors (network issues) are retryable
    if isinstance(exception, OSError):
      return True
    
    # HTTP response errors - only 5xx server errors are retryable
    if isinstance(exception, ClientResponseError):
      return 500 <= exception.status < 600
    
    # All other exceptions are not retryable
    return False

  def _calculate_delay(self, attempt: int) -> float:
    """Calculate delay for retry attempt using exponential backoff.
    
    Args:
      attempt: Current attempt number (1-based)
      
    Returns:
      Delay in seconds
    """
    # Calculate exponential backoff delay
    delay = self.base_delay * (self.backoff_factor ** attempt)
    
    # Cap at maximum delay
    delay = min(delay, self.max_delay)
    
    # Add jitter if enabled
    if self.jitter:
      # Add random jitter of ±50% of the delay
      jitter_range = delay * 0.5
      delay += random.uniform(-jitter_range, jitter_range)
      
      # Ensure delay is not negative
      delay = max(0.0, delay)
    
    return delay