"""Tests for shared HTTP provider transport helpers."""

from __future__ import annotations

import socket
from contextlib import contextmanager

from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
)
from rationale_benchmark.llm.exceptions import TimeoutError
from rationale_benchmark.llm.providers.base import JSONHTTPProvider


class DummyHTTPProvider(JSONHTTPProvider):
  """Minimal concrete provider for testing shared HTTP behavior."""

  def _generate(self, messages, *, output_schema):  # pragma: no cover
    raise NotImplementedError


def test_zero_timeout_disables_urlopen_timeout(monkeypatch):
  captured = {}

  class FakeResponse:
    headers = {}

    def read(self):
      return b"{}"

  @contextmanager
  def fake_urlopen(req, timeout):
    captured["timeout"] = timeout
    yield FakeResponse()

  monkeypatch.setattr(
    "rationale_benchmark.llm.providers.base.request.urlopen",
    fake_urlopen,
  )
  config = LLMConnectorConfig(
    provider=ProviderType.OPENAI_COMPATIBLE,
    model="demo",
    api_key="token",
    timeout_seconds=0,
  )
  provider = DummyHTTPProvider(config)

  provider._post_json("https://example.test", {}, {})

  assert captured["timeout"] is None


def test_read_timeout_is_wrapped_as_connector_timeout(monkeypatch):
  class FakeResponse:
    headers = {}

    def read(self):
      raise socket.timeout("The read operation timed out")

  @contextmanager
  def fake_urlopen(req, timeout):
    yield FakeResponse()

  monkeypatch.setattr(
    "rationale_benchmark.llm.providers.base.request.urlopen",
    fake_urlopen,
  )
  config = LLMConnectorConfig(
    provider=ProviderType.OPENAI_COMPATIBLE,
    model="demo",
    api_key="token",
    timeout_seconds=30,
  )
  provider = DummyHTTPProvider(config)

  try:
    provider._post_json("https://example.test", {}, {})
  except TimeoutError as exc:
    assert exc.timeout_seconds == 30
    assert str(exc) == "Request timed out"
  else:  # pragma: no cover - assertion guard
    raise AssertionError("Expected TimeoutError for socket read timeout")
