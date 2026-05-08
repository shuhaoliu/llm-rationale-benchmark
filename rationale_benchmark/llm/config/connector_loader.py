"""Load and validate connector configurations from YAML documents."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from ..exceptions import ConfigurationError
from .connector_models import (
  LLMConnectorConfig,
  ProviderType,
  format_validation_error,
)

STREAMING_KEYS = {
  "stream",
  "streaming",
  "stream_options",
  "stream_usage",
  "stream_callback",
  "stream_handler",
  "incremental",
}

ALIYUN_MODEL_SUFFIX_PATTERN = re.compile(
  r"^(?P<model>.+?)\s*\((?P<thinking_budget>-?\d+)\)\s*$"
)


class ConnectorConfigLoader:
  """Load connector configurations keyed by ``provider/model`` selectors."""

  def __init__(self, *, base_filename: str = "default-llms.yaml") -> None:
    self.base_filename = base_filename

  def load(
    self,
    path: Path,
    *,
    base_path: Path | None = None,
    merge_default: bool = True,
  ) -> Dict[str, LLMConnectorConfig]:
    """Return validated connector configurations from the supplied YAML file."""

    merged = self._load_and_merge(
      path,
      base_path=base_path,
      merge_default=merge_default,
    )
    return self._build_configurations(merged, source_path=path)

  def _load_and_merge(
    self,
    path: Path,
    *,
    base_path: Path | None,
    merge_default: bool,
  ) -> dict[str, Any]:
    overlay = self._read_document(path)

    candidate_base = base_path
    if merge_default and candidate_base is None:
      default_candidate = path.with_name(self.base_filename)
      if default_candidate.exists() and default_candidate != path:
        candidate_base = default_candidate

    if candidate_base is None:
      return overlay

    base_document = self._read_document(candidate_base)
    return self._merge_documents(base_document, overlay)

  def _read_document(self, path: Path) -> dict[str, Any]:
    try:
      raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
      raise ConfigurationError(
        f"Configuration file not found: {path}",
        config_file=str(path),
      ) from exc
    except OSError as exc:
      raise ConfigurationError(
        f"Failed to read configuration file: {exc}",
        config_file=str(path),
      ) from exc

    try:
      loaded = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
      raise ConfigurationError(
        f"Invalid YAML in configuration file: {exc}",
        config_file=str(path),
      ) from exc

    if not isinstance(loaded, dict):
      raise ConfigurationError(
        "Configuration document must be a mapping",
        config_file=str(path),
      )

    return self._resolve_environment(loaded, source=str(path))

  def _merge_documents(
    self,
    base: dict[str, Any],
    overlay: dict[str, Any],
  ) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    base_defaults = base.get("defaults", {})
    overlay_defaults = overlay.get("defaults", {})
    if base_defaults or overlay_defaults:
      merged["defaults"] = self._merge_mapping(
        base_defaults,
        overlay_defaults,
        field="defaults",
      )

    base_providers = base.get("providers", {})
    overlay_providers = overlay.get("providers", {})
    providers: dict[str, Any] = {}

    for name, value in base_providers.items():
      if value is None:
        continue
      providers[name] = copy.deepcopy(value)

    for name, value in overlay_providers.items():
      overlay_value = copy.deepcopy(value)
      if overlay_value is None:
        providers.pop(name, None)
        continue
      base_value = providers.get(name, {})
      providers[name] = self._merge_provider_entry(
        base_value,
        overlay_value,
        provider=name,
      )

    if providers:
      merged["providers"] = providers

    return merged

  def _merge_provider_entry(
    self,
    base: Any,
    overlay: Any,
    *,
    provider: str,
  ) -> dict[str, Any]:
    if base is None:
      base = {}
    if not isinstance(base, dict):
      raise ConfigurationError(
        f"Provider '{provider}' configuration must be a mapping",
        field=f"providers.{provider}",
      )

    if not isinstance(overlay, dict):
      raise ConfigurationError(
        f"Provider '{provider}' configuration must be a mapping",
        field=f"providers.{provider}",
      )

    merged = copy.deepcopy(base)
    for key, value in overlay.items():
      if key in {"default_params", "provider_specific", "metadata"}:
        merged[key] = self._merge_mapping(
          merged.get(key, {}),
          value,
          field=f"providers.{provider}.{key}",
        )
      elif key == "models":
        if value is None:
          merged.pop("models", None)
        elif isinstance(value, list):
          merged["models"] = list(value)
        else:
          raise ConfigurationError(
            f"'models' for provider '{provider}' must be a list",
            field=f"providers.{provider}.models",
          )
      else:
        merged[key] = value
    return merged

  def _build_configurations(
    self,
    document: dict[str, Any],
    *,
    source_path: Path,
  ) -> Dict[str, LLMConnectorConfig]:
    defaults = document.get("defaults", {}) or {}
    providers = document.get("providers")

    if defaults and not isinstance(defaults, dict):
      raise ConfigurationError(
        "defaults must be a mapping when provided",
        config_file=str(source_path),
        field="defaults",
      )

    if not isinstance(providers, dict) or not providers:
      raise ConfigurationError(
        "providers section must be a non-empty mapping",
        config_file=str(source_path),
        field="providers",
      )

    (
      default_common,
      default_params,
      default_specific,
      default_metadata,
    ) = self._partition_entry(defaults, label="defaults")

    configurations: Dict[str, LLMConnectorConfig] = {}
    errors: list[str] = []

    for provider_name, raw_entry in providers.items():
      try:
        provider_type = self._provider_type(
          provider_name,
          str(source_path),
        )
      except ConfigurationError as exc:
        errors.append(str(exc))
        continue
      (
        provider_common,
        provider_params,
        provider_specific,
        provider_metadata,
      ) = self._partition_entry(
        raw_entry,
        label=f"providers.{provider_name}",
      )

      merged_common = self._merge_mapping(
        default_common,
        provider_common,
        field=f"providers.{provider_name}",
      )
      merged_params = self._merge_mapping(
        default_params,
        provider_params,
        field=f"providers.{provider_name}.default_params",
      )
      merged_specific = self._merge_mapping(
        default_specific,
        provider_specific,
        field=f"providers.{provider_name}.provider_specific",
      )
      merged_metadata = self._merge_mapping(
        default_metadata,
        provider_metadata,
        field=f"providers.{provider_name}.metadata",
      )

      model_entries = merged_common.pop("models", None)
      if not isinstance(model_entries, list) or not model_entries:
        errors.append(
          f"{provider_name}: 'models' must be a non-empty list"
        )
        continue

      for index, model_entry in enumerate(model_entries):
        try:
          selector_name, model_name, model_overrides = self._parse_model_entry(
            model_entry,
            provider=provider_name,
            index=index,
          )
        except ConfigurationError as exc:
          errors.append(str(exc))
          continue

        (
          model_common,
          model_params,
          model_specific,
          model_metadata,
        ) = self._partition_entry(
          model_overrides,
          label=f"providers.{provider_name}.models[{index}]",
        )

        config_common = self._merge_mapping(
          merged_common,
          model_common,
          field=f"providers.{provider_name}.models[{index}]",
        )
        config_params = self._merge_mapping(
          merged_params,
          model_params,
          field=f"providers.{provider_name}.models[{index}].default_params",
        )
        config_specific = self._merge_mapping(
          merged_specific,
          model_specific,
          field=f"providers.{provider_name}.models[{index}].provider_specific",
        )
        config_metadata = self._merge_mapping(
          merged_metadata,
          model_metadata,
          field=f"providers.{provider_name}.models[{index}].metadata",
        )

        try:
          payload = self._normalise_payload(
            provider_name=provider_name,
            model_name=model_name,
            provider_type=provider_type,
            common=config_common,
            default_params=config_params,
            provider_specific=config_specific,
            metadata=config_metadata,
            source=str(source_path),
          )
          config = LLMConnectorConfig.model_validate(payload)
        except Exception as exc:
          if hasattr(exc, "errors"):
            message = format_validation_error(exc)  # type: ignore[arg-type]
          else:
            message = str(exc)
          errors.append(
            f"{provider_name}/{selector_name}: {message}"
          )
          continue

        selector = f"{provider_name}/{selector_name}"
        if config.provider is not provider_type:
          errors.append(
            f"{selector}: provider '{config.provider.value}' "
            f"does not match expected '{provider_type.value}'"
          )
          continue
        configurations[selector] = config

    if errors:
      message = "Configuration validation failed:\n" + "\n".join(errors)
      raise ConfigurationError(
        message,
        config_file=str(source_path),
      )

    return configurations

  def _partition_entry(
    self,
    entry: Any,
    *,
    label: str,
  ) -> Tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if entry is None:
      return {}, {}, {}, {}
    if not isinstance(entry, dict):
      raise ConfigurationError(
        f"{label} must be a mapping",
        field=label,
      )

    common = dict(entry)
    default_params = self._pop_mapping(common, "default_params", label)
    provider_specific = self._pop_mapping(
      common,
      "provider_specific",
      label,
    )
    metadata = self._pop_mapping(common, "metadata", label)
    return common, default_params, provider_specific, metadata

  def _pop_mapping(
    self,
    entry: dict[str, Any],
    key: str,
    label: str,
  ) -> dict[str, Any]:
    if key not in entry:
      return {}
    value = entry.pop(key)
    if value is None:
      return {}
    if not isinstance(value, dict):
      raise ConfigurationError(
        f"{label}.{key} must be a mapping",
        field=f"{label}.{key}",
      )
    return copy.deepcopy(value)

  def _parse_model_entry(
    self,
    entry: Any,
    *,
    provider: str,
    index: int,
  ) -> tuple[str, str, dict[str, Any]]:
    if isinstance(entry, str):
      return self._parse_string_model_entry(
        entry,
        provider=provider,
        index=index,
      )
    if isinstance(entry, dict):
      model_name = entry.get("name")
      if not isinstance(model_name, str) or not model_name.strip():
        raise ConfigurationError(
          f"providers.{provider}.models[{index}].name must be a string",
          field=f"providers.{provider}.models[{index}].name",
        )
      overrides = dict(entry)
      overrides.pop("name", None)
      if self._is_aliyun_provider(provider):
        selector_name, model_name, thinking_overrides = (
          self._parse_aliyun_model_name(
          model_name,
          provider=provider,
          index=index,
        )
        )
        overrides["default_params"] = self._merge_mapping(
          thinking_overrides,
          overrides.get("default_params", {}),
          field=f"providers.{provider}.models[{index}].default_params",
        )
        return selector_name, model_name, overrides
      return model_name, model_name, overrides

    raise ConfigurationError(
      f"providers.{provider}.models[{index}] must be a string or mapping",
      field=f"providers.{provider}.models[{index}]",
    )

  def _parse_string_model_entry(
    self,
    entry: str,
    *,
    provider: str,
    index: int,
  ) -> tuple[str, str, dict[str, Any]]:
    if self._is_aliyun_provider(provider):
      return self._parse_aliyun_model_name(
        entry,
        provider=provider,
        index=index,
      )
    return entry, entry, {}

  def _parse_aliyun_model_name(
    self,
    entry: str,
    *,
    provider: str,
    index: int,
  ) -> tuple[str, str, dict[str, Any]]:
    selector_name = entry.strip()
    match = ALIYUN_MODEL_SUFFIX_PATTERN.match(selector_name)
    if not match:
      return selector_name, selector_name, {}

    model_name = match.group("model").strip()
    budget = int(match.group("thinking_budget"))
    default_params: dict[str, Any] = {}

    if budget == 0:
      default_params["enable_thinking"] = False
    elif budget == -1:
      default_params["enable_thinking"] = True
    elif budget > 0:
      default_params["enable_thinking"] = True
      default_params["thinking_budget"] = budget
    else:
      raise ConfigurationError(
        (
          "Aliyun thinking suffix must be 0, -1, or a positive integer "
          f"for providers.{provider}.models[{index}]"
        ),
        field=f"providers.{provider}.models[{index}]",
      )

    return selector_name, model_name, {"default_params": default_params}

  def _normalise_payload(
    self,
    *,
    provider_name: str,
    model_name: str,
    provider_type: ProviderType,
    common: dict[str, Any],
    default_params: dict[str, Any],
    provider_specific: dict[str, Any],
    metadata: dict[str, Any],
    source: str,
  ) -> dict[str, Any]:
    payload = dict(common)
    payload["model"] = model_name
    payload["provider"] = provider_type

    timeout = payload.pop("timeout", None)
    if timeout is not None:
      payload["timeout_seconds"] = timeout

    max_retries = payload.pop("max_retries", None)
    retry_block = payload.pop("retry", {}) or {}
    if not isinstance(retry_block, dict):
      raise ConfigurationError(
        f"{provider_name}/{model_name}: retry must be a mapping",
        config_file=source,
        field=f"providers.{provider_name}.retry",
      )
    retry_payload = dict(retry_block)
    if max_retries is not None and "max_attempts" not in retry_payload:
      retry_payload["max_attempts"] = max_retries
    if retry_payload:
      payload["retry"] = retry_payload

    payload["default_params"] = default_params
    payload["provider_specific"] = provider_specific
    payload["metadata"] = metadata

    self._assert_no_streaming(
      payload,
      provider=provider_name,
      model=model_name,
      field="configuration",
    )
    self._assert_no_streaming(
      default_params,
      provider=provider_name,
      model=model_name,
      field="default_params",
    )
    self._assert_no_streaming(
      provider_specific,
      provider=provider_name,
      model=model_name,
      field="provider_specific",
    )

    return payload

  def _provider_type(
    self,
    provider: str,
    source: str,
  ) -> ProviderType:
    if self._is_aliyun_provider(provider):
      return ProviderType.OPENAI_COMPATIBLE
    if provider.endswith("_openai_compatible"):
      return ProviderType.OPENAI_COMPATIBLE

    try:
      return ProviderType(provider)
    except ValueError as exc:
      raise ConfigurationError(
        f"Unknown provider '{provider}'",
        config_file=source,
        field=f"providers.{provider}",
      ) from exc

  def _is_aliyun_provider(self, provider: str) -> bool:
    return provider == "aliyun" or provider.endswith("_aliyun")

  def _assert_no_streaming(
    self,
    value: dict[str, Any],
    *,
    provider: str,
    model: str,
    field: str,
  ) -> None:
    blocked = STREAMING_KEYS.intersection(self._iter_all_keys(value))
    if blocked:
      joined = ", ".join(sorted(blocked))
      raise ConfigurationError(
        f"Streaming parameters not supported ({joined}) for {provider}/{model}",
        field=f"{provider}.{model}.{field}",
      )

  def _iter_all_keys(self, value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
      for key, item in value.items():
        keys.add(str(key))
        keys.update(self._iter_all_keys(item))
    elif isinstance(value, list):
      for item in value:
        keys.update(self._iter_all_keys(item))
    return keys

  def _merge_mapping(
    self,
    base: Any,
    overlay: Any,
    *,
    field: str,
  ) -> dict[str, Any]:
    if base is None and overlay is None:
      return {}
    if base is None:
      base = {}
    if overlay is None:
      return copy.deepcopy(base)
    if not isinstance(base, dict) or not isinstance(overlay, dict):
      raise ConfigurationError(
        f"{field} must be a mapping",
        field=field,
      )
    merged: dict[str, Any] = {}
    for key, value in base.items():
      if isinstance(value, dict):
        merged[key] = copy.deepcopy(value)
      else:
        merged[key] = value
    for key, value in overlay.items():
      if (
        key in merged
        and isinstance(merged[key], dict)
        and isinstance(value, dict)
      ):
        merged[key] = self._merge_mapping(
          merged[key],
          value,
          field=f"{field}.{key}",
        )
      else:
        merged[key] = copy.deepcopy(value)
    return merged

  def _resolve_environment(
    self,
    value: Any,
    *,
    source: str,
  ) -> Any:
    if isinstance(value, str):
      return self._substitute_env(value, source=source)

    if isinstance(value, dict):
      return {
        key: self._resolve_environment(val, source=source)
        for key, val in value.items()
      }

    if isinstance(value, list):
      return [
        self._resolve_environment(item, source=source)
        for item in value
      ]

    return value

  def _substitute_env(self, text: str, *, source: str) -> str:
    result = ""
    index = 0
    while index < len(text):
      start = text.find("${", index)
      if start == -1:
        result += text[index:]
        break
      result += text[index:start]
      end = text.find("}", start)
      if end == -1:
        result += text[start:]
        break
      name = text[start + 2:end]
      if not name:
        result += "${}"
        index = end + 1
        continue
      value = os.getenv(name)
      if value is None:
        raise ConfigurationError(
          f"Environment variable '{name}' referenced but not set",
          config_file=source,
          field=name,
        )
      result += value
      index = end + 1
    return result
