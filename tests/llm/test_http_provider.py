"""Tests for shared HTTP provider transport helpers."""

from __future__ import annotations

from contextlib import contextmanager

from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
)
from rationale_benchmark.llm.providers.base import JSONHTTPProvider


class DummyHTTPProvider(JSONHTTPProvider):
  """Minimal concrete provider for testing shared HTTP behavior."""

  def _generate(self, messages, *, response_format):  # pragma: no cover
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
