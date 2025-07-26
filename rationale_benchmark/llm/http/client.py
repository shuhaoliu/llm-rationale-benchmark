"""HTTP client with connection pooling and retry logic."""

import asyncio
from typing import Any, Dict, Optional
import aiohttp
from aiohttp import ClientTimeout, TCPConnector, ClientResponse
from aiohttp.client_exceptions import ClientConnectorError, ClientError

from rationale_benchmark.llm.http.retry import RetryHandler
from rationale_benchmark.llm.exceptions import NetworkError, TimeoutError


class HTTPClient:
  """HTTP client with connection pooling and automatic retry logic."""

  def __init__(
    self,
    max_connections: int = 100,
    timeout: int = 30
  ):
    """Initialize HTTP client with connection pooling.
    
    Args:
      max_connections: Maximum number of connections in the pool
      timeout: Request timeout in seconds
    """
    self.max_connections = max_connections
    self.timeout = timeout
    self.session: Optional[aiohttp.ClientSession] = None
    self.retry_handler = RetryHandler()

  async def _get_session(self) -> aiohttp.ClientSession:
    """Get or create aiohttp session with connection pooling.
    
    Returns:
      Configured ClientSession instance
    """
    if self.session is None:
      connector = TCPConnector(limit=self.max_connections)
      timeout = ClientTimeout(total=self.timeout)
      
      self.session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout
      )
    
    return self.session

  async def post(
    self,
    url: str,
    json: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    **kwargs: Any
  ) -> ClientResponse:
    """Make POST request with retry logic.
    
    Args:
      url: Request URL
      json: JSON data to send in request body
      data: Raw data to send in request body
      headers: HTTP headers
      params: Query parameters
      **kwargs: Additional arguments passed to aiohttp
      
    Returns:
      HTTP response
      
    Raises:
      NetworkError: For network-related errors
      TimeoutError: For timeout errors
    """
    return await self.retry_handler.execute(
      self._make_request,
      "POST",
      url,
      json=json,
      data=data,
      headers=headers,
      params=params,
      **kwargs
    )

  async def get(
    self,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    **kwargs: Any
  ) -> ClientResponse:
    """Make GET request with retry logic.
    
    Args:
      url: Request URL
      headers: HTTP headers
      params: Query parameters
      **kwargs: Additional arguments passed to aiohttp
      
    Returns:
      HTTP response
      
    Raises:
      NetworkError: For network-related errors
      TimeoutError: For timeout errors
    """
    return await self.retry_handler.execute(
      self._make_request,
      "GET",
      url,
      headers=headers,
      params=params,
      **kwargs
    )

  async def put(
    self,
    url: str,
    json: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    **kwargs: Any
  ) -> ClientResponse:
    """Make PUT request with retry logic.
    
    Args:
      url: Request URL
      json: JSON data to send in request body
      data: Raw data to send in request body
      headers: HTTP headers
      params: Query parameters
      **kwargs: Additional arguments passed to aiohttp
      
    Returns:
      HTTP response
      
    Raises:
      NetworkError: For network-related errors
      TimeoutError: For timeout errors
    """
    return await self.retry_handler.execute(
      self._make_request,
      "PUT",
      url,
      json=json,
      data=data,
      headers=headers,
      params=params,
      **kwargs
    )

  async def delete(
    self,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    **kwargs: Any
  ) -> ClientResponse:
    """Make DELETE request with retry logic.
    
    Args:
      url: Request URL
      headers: HTTP headers
      params: Query parameters
      **kwargs: Additional arguments passed to aiohttp
      
    Returns:
      HTTP response
      
    Raises:
      NetworkError: For network-related errors
      TimeoutError: For timeout errors
    """
    return await self.retry_handler.execute(
      self._make_request,
      "DELETE",
      url,
      headers=headers,
      params=params,
      **kwargs
    )

  async def _make_request(
    self,
    method: str,
    url: str,
    **kwargs: Any
  ) -> ClientResponse:
    """Make HTTP request using the session.
    
    Args:
      method: HTTP method (GET, POST, PUT, DELETE)
      url: Request URL
      **kwargs: Additional arguments passed to aiohttp
      
    Returns:
      HTTP response
      
    Raises:
      NetworkError: For network-related errors
      TimeoutError: For timeout errors
    """
    session = await self._get_session()
    
    try:
      # Note: We don't use async with here because we want to return the response
      # The caller is responsible for reading the response content
      response = await session.request(method, url, **kwargs)
      return response
    except asyncio.TimeoutError as e:
      raise TimeoutError(f"Request timeout: {str(e)}")
    except ClientConnectorError as e:
      raise NetworkError(f"Connection failed: {str(e)}")
    except ClientError as e:
      raise NetworkError(f"HTTP request failed: {str(e)}")

  async def close(self) -> None:
    """Close the HTTP session and clean up resources."""
    if self.session is not None:
      await self.session.close()
      self.session = None

  async def __aenter__(self) -> "HTTPClient":
    """Async context manager entry."""
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Async context manager exit with cleanup."""
    await self.close()