"""Unit tests for HTTP client with connection pooling."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from aiohttp import ClientError, ClientTimeout, ClientResponse
from aiohttp.client_exceptions import ClientConnectorError, ClientResponseError

from rationale_benchmark.llm.http.client import HTTPClient
from rationale_benchmark.llm.http.retry import RetryHandler
from rationale_benchmark.llm.exceptions import NetworkError, TimeoutError


class TestHTTPClient:
  """Test cases for HTTPClient class."""

  @pytest.fixture
  def http_client(self):
    """Create HTTPClient instance for testing."""
    return HTTPClient(max_connections=10, timeout=30)

  @pytest.fixture
  def mock_session(self):
    """Create mock aiohttp ClientSession."""
    session = AsyncMock()
    return session

  @pytest.fixture
  def mock_response(self):
    """Create mock aiohttp ClientResponse."""
    response = Mock(spec=ClientResponse)
    response.status = 200
    response.json = AsyncMock(return_value={"test": "data"})
    response.text = AsyncMock(return_value="test response")
    response.headers = {"Content-Type": "application/json"}
    return response

  def test_http_client_initialization(self):
    """Test HTTPClient initialization with default parameters."""
    client = HTTPClient()
    assert client.max_connections == 100
    assert client.timeout == 30
    assert client.session is None
    assert client.retry_handler is not None
    assert isinstance(client.retry_handler, RetryHandler)

  def test_http_client_initialization_with_custom_params(self):
    """Test HTTPClient initialization with custom parameters."""
    client = HTTPClient(max_connections=50, timeout=60)
    assert client.max_connections == 50
    assert client.timeout == 60
    assert client.session is None

  @pytest.mark.asyncio
  async def test_session_creation_on_first_use(self, http_client):
    """Test that session is created lazily on first use."""
    assert http_client.session is None
    
    with patch('aiohttp.ClientSession') as mock_session_class:
      mock_session = AsyncMock()
      mock_session_class.return_value = mock_session
      mock_session.post.return_value.__aenter__.return_value.status = 200
      
      await http_client._get_session()
      
      assert http_client.session is not None
      mock_session_class.assert_called_once()

  @pytest.mark.asyncio
  async def test_session_reuse(self, http_client):
    """Test that session is reused across multiple calls."""
    with patch('aiohttp.ClientSession') as mock_session_class:
      mock_session = AsyncMock()
      mock_session_class.return_value = mock_session
      
      session1 = await http_client._get_session()
      session2 = await http_client._get_session()
      
      assert session1 is session2
      mock_session_class.assert_called_once()

  @pytest.mark.asyncio
  async def test_post_request_success(self, http_client, mock_response):
    """Test successful POST request."""
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      mock_session.request.return_value = mock_response
      
      response = await http_client.post("https://api.example.com/test", json={"key": "value"})
      
      assert response.status == 200
      mock_session.request.assert_called_once_with(
        "POST",
        "https://api.example.com/test",
        json={"key": "value"},
        data=None,
        headers=None,
        params=None
      )

  @pytest.mark.asyncio
  async def test_post_request_with_headers(self, http_client, mock_response):
    """Test POST request with custom headers."""
    headers = {"Authorization": "Bearer token", "Content-Type": "application/json"}
    
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      mock_session.request.return_value = mock_response
      
      response = await http_client.post(
        "https://api.example.com/test",
        json={"key": "value"},
        headers=headers
      )
      
      assert response.status == 200
      mock_session.request.assert_called_once_with(
        "POST",
        "https://api.example.com/test",
        json={"key": "value"},
        data=None,
        headers=headers,
        params=None
      )

  @pytest.mark.asyncio
  async def test_get_request_success(self, http_client, mock_response):
    """Test successful GET request."""
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      mock_session.request.return_value = mock_response
      
      response = await http_client.get("https://api.example.com/test")
      
      assert response.status == 200
      mock_session.request.assert_called_once_with(
        "GET",
        "https://api.example.com/test",
        headers=None,
        params=None
      )

  @pytest.mark.asyncio
  async def test_timeout_handling(self, http_client):
    """Test timeout handling in HTTP requests."""
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      mock_session.request.side_effect = asyncio.TimeoutError("Request timeout")
      
      with pytest.raises(TimeoutError, match="Request timeout"):
        await http_client.post("https://api.example.com/test")

  @pytest.mark.asyncio
  async def test_connection_error_handling(self, http_client):
    """Test connection error handling."""
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      mock_session.request.side_effect = ClientConnectorError(
        connection_key=Mock(), os_error=OSError("Connection failed")
      )
      
      with pytest.raises(NetworkError, match="Connection failed"):
        await http_client.post("https://api.example.com/test")

  @pytest.mark.asyncio
  async def test_client_error_handling(self, http_client):
    """Test general client error handling."""
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      mock_session.request.side_effect = ClientError("General client error")
      
      with pytest.raises(NetworkError, match="HTTP request failed"):
        await http_client.post("https://api.example.com/test")

  @pytest.mark.asyncio
  async def test_retry_integration_success_after_failure(self, http_client, mock_response):
    """Test integration with RetryHandler for successful retry."""
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      
      # First call fails, second succeeds
      mock_session.request.side_effect = [
        ClientConnectorError(connection_key=Mock(), os_error=OSError("Connection failed")),
        mock_response
      ]
      
      with patch.object(http_client.retry_handler, 'execute') as mock_retry:
        mock_retry.return_value = mock_response
        
        response = await http_client.post("https://api.example.com/test")
        
        assert response.status == 200
        mock_retry.assert_called_once()

  @pytest.mark.asyncio
  async def test_retry_integration_max_retries_exceeded(self, http_client):
    """Test integration with RetryHandler when max retries exceeded."""
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      
      with patch.object(http_client.retry_handler, 'execute') as mock_retry:
        mock_retry.side_effect = NetworkError("Max retries exceeded")
        
        with pytest.raises(NetworkError, match="Max retries exceeded"):
          await http_client.post("https://api.example.com/test")

  @pytest.mark.asyncio
  async def test_session_cleanup_on_close(self, http_client):
    """Test proper session cleanup when client is closed."""
    with patch('aiohttp.ClientSession') as mock_session_class:
      mock_session = AsyncMock()
      mock_session_class.return_value = mock_session
      
      # Create session
      await http_client._get_session()
      assert http_client.session is not None
      
      # Close client
      await http_client.close()
      
      # Verify session was closed and set to None
      mock_session.close.assert_called_once()
      assert http_client.session is None

  @pytest.mark.asyncio
  async def test_close_without_session(self, http_client):
    """Test closing client when no session was created."""
    # Should not raise any errors
    await http_client.close()
    assert http_client.session is None

  @pytest.mark.asyncio
  async def test_context_manager_usage(self):
    """Test HTTPClient as async context manager."""
    with patch('aiohttp.ClientSession') as mock_session_class:
      mock_session = AsyncMock()
      mock_session_class.return_value = mock_session
      
      async with HTTPClient() as client:
        assert isinstance(client, HTTPClient)
        # Session should be created when used
        await client._get_session()
        assert client.session is not None
      
      # Session should be closed after context exit
      mock_session.close.assert_called_once()

  @pytest.mark.asyncio
  async def test_connection_pooling_configuration(self):
    """Test that connection pooling is properly configured."""
    with patch('rationale_benchmark.llm.http.client.aiohttp.ClientSession') as mock_session_class, \
         patch('rationale_benchmark.llm.http.client.TCPConnector') as mock_connector_class:
      
      mock_connector = Mock()
      mock_connector_class.return_value = mock_connector
      mock_session = AsyncMock()
      mock_session_class.return_value = mock_session
      
      client = HTTPClient(max_connections=50, timeout=60)
      await client._get_session()
      
      # Verify connector was created with correct limit
      mock_connector_class.assert_called_once_with(limit=50)
      
      # Verify session was created with connector and timeout
      mock_session_class.assert_called_once_with(
        connector=mock_connector,
        timeout=ClientTimeout(total=60)
      )

  @pytest.mark.asyncio
  async def test_multiple_concurrent_requests(self, http_client, mock_response):
    """Test handling multiple concurrent requests."""
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      mock_session.request.return_value = mock_response
      
      # Make multiple concurrent requests
      tasks = [
        http_client.post(f"https://api.example.com/test{i}")
        for i in range(5)
      ]
      
      responses = await asyncio.gather(*tasks)
      
      # All requests should succeed
      assert len(responses) == 5
      assert all(r.status == 200 for r in responses)
      assert mock_session.request.call_count == 5

  @pytest.mark.asyncio
  async def test_request_with_params(self, http_client, mock_response):
    """Test request with query parameters."""
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      mock_session.request.return_value = mock_response
      
      params = {"key1": "value1", "key2": "value2"}
      response = await http_client.get("https://api.example.com/test", params=params)
      
      assert response.status == 200
      mock_session.request.assert_called_once_with(
        "GET",
        "https://api.example.com/test",
        headers=None,
        params=params
      )

  @pytest.mark.asyncio
  async def test_response_error_status_handling(self, http_client):
    """Test handling of HTTP error status codes."""
    mock_response = Mock(spec=ClientResponse)
    mock_response.status = 500
    mock_response.text = AsyncMock(return_value="Internal Server Error")
    
    with patch.object(http_client, '_get_session') as mock_get_session:
      mock_session = AsyncMock()
      mock_get_session.return_value = mock_session
      mock_session.request.return_value = mock_response
      
      # Should not raise exception for error status codes
      # (error handling is responsibility of the caller)
      response = await http_client.post("https://api.example.com/test")
      assert response.status == 500