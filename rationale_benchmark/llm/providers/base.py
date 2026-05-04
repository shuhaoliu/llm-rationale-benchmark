"""Shared helpers for HTTP-based provider clients."""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, Tuple
from urllib import error, request

from ..exceptions import (
  ConfigurationError,
  NetworkError,
  RetryableProviderError,
  TimeoutError as ConnectorTimeoutError,
)
from ..provider_client import BaseProviderClient


class JSONHTTPProvider(BaseProviderClient):
  """Base class implementing JSON over HTTP helpers."""

  def _require_api_key(self) -> str:
    api_key = self.config.api_key
    if not api_key:
      raise ConfigurationError(
        f"Provider '{self.config.provider.value}' requires an API key"
      )
    return api_key

  def _post_json(
    self,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
  ) -> Tuple[dict[str, Any], Dict[str, str]]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    timeout = (
      None
      if self.config.timeout_seconds == 0
      else self.config.timeout_seconds
    )
    try:
      with request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
        parsed = self._safe_load_json(text)
        return parsed, dict(resp.headers)
    except error.HTTPError as exc:
      body = exc.read().decode("utf-8", "ignore")
      parsed = self._safe_load_json(body)
      raise self._handle_http_error(exc, parsed) from exc
    except error.URLError as exc:
      if isinstance(exc.reason, socket.timeout):
        raise ConnectorTimeoutError(
          timeout_seconds=self.config.timeout_seconds
        ) from exc
      raise NetworkError(str(exc.reason)) from exc

  def _safe_load_json(self, text: str) -> dict[str, Any]:
    if not text:
      return {}
    try:
      value = json.loads(text)
    except json.JSONDecodeError:
      return {"message": text}
    if isinstance(value, dict):
      return value
    return {"value": value}

  def _handle_http_error(
    self,
    exc: error.HTTPError,
    payload: dict[str, Any],
  ) -> RetryableProviderError:
    message = self._extract_error_message(payload) or exc.reason or str(exc.code)
    retry_after = None
    if exc.headers and "retry-after" in exc.headers:
      try:
        retry_after = int(exc.headers["retry-after"])
      except ValueError:
        retry_after = None
    return RetryableProviderError(
      self.config.provider.value,
      message,
      retry_after=retry_after,
    )

  def _extract_error_message(self, payload: dict[str, Any]) -> str:
    if "error" in payload and isinstance(payload["error"], dict):
      return str(payload["error"].get("message") or payload["error"])
    if "message" in payload:
      return str(payload["message"])
    return ""
