# LLM Configuration Specification

## Purpose
- Define the canonical shape of YAML configuration files stored in `config/llms/`.
- Document how defaults, provider entries, and overrides are merged at runtime.
- Capture validation expectations so configuration errors are caught before any network traffic.
- Reinforce that sensitive values (API keys, organization IDs, custom endpoints) belong in environment variables, not hard-coded YAML.

Review this guide before authoring or updating LLM configuration files. It complements the higher-level connector architecture in `docs/01-llm/design.md`.

## Directory and Naming Conventions
- Store all LLM configuration files under `config/llms/`.
- The CLI selects configurations by filename stem (e.g., `--llm-config research-models` loads `config/llms/research-models.yaml`).
- `default-llms.yaml` acts as the base configuration and **must** exist. Custom files extend or override it.
- Keep filenames descriptive of their intended environment or usage (`production-models.yaml`, `local-testing.yaml`, etc.).

## YAML Structure Overview
Each configuration file is a YAML document with two top-level sections:

```yaml
defaults:
  timeout: 30
  max_retries: 3
  temperature: 0.7
  max_tokens: 1000

providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    models:
      - "gpt-4"
      - "gpt-3.5-turbo"
    default_params:
      response_format: "json"
    provider_specific:
      organization: "${OPENAI_ORG_ID}"
```

- `defaults` declares reusable baseline values applied to every provider unless explicitly overridden.
- `providers` contains named maps of provider identifiers (e.g., `openai`, `anthropic`, `gemini`, `openrouter`) to their configuration.
- Use `${ENV_VAR}` placeholders wherever the value varies per deployment or contains secrets; the loader resolves them at runtime.

## Defaults Section
`defaults` is optional in override files but recommended. Recognized keys align with the connector configuration model:
- `timeout`: Integer seconds before requests are aborted (default: 30).
- `max_retries`: Maximum retry attempts for transient errors (default: 3).
- `temperature`: Floating-point value `0.0–2.0`. Controls creativity; defaults to `0.7`.
- `max_tokens`: Maximum tokens requested per completion (default: 1000).
- `system_prompt`: Fallback system prompt if none is supplied via CLI.
- Any additional fields under `defaults` become part of the shared baseline and are merged into each provider unless overridden.

## Provider Entries
Each provider entry **must** define the following:
- `api_key`: Usually expressed via `${ENV_VAR}`. Validation ensures the environment variable resolves.
- `models`: List of model identifiers to exercise for this provider. Lists replace the defaults entirely when overridden.
- Prefer environment variables for every credential-like value (`api_key`, `organization`, custom headers) and for endpoints that differ between environments (sandbox vs. production).

Optional fields:
- `endpoint`: Full URL for providers requiring explicit endpoint configuration.
- `base_url`: For OpenAI-compatible services (e.g., OpenRouter). Mutually exclusive with `endpoint`; use whichever the adapter expects.
- `timeout`: Provider-specific override of the default timeout.
- `max_retries`: Provider-specific retry override.
- `default_params`: Map of request parameters applied to every call. Use for `temperature`, `max_tokens`, `top_p`, `stop_sequences`, etc. These values override anything inherited from `defaults`.
- `provider_specific`: Adapter-specific settings (e.g., Anthropic `version`, Gemini `safety_settings`, custom headers). Streaming parameters (`stream`, `streaming`, `stream_options`) are forbidden and raise validation errors.
- `metadata`: Optional free-form dictionary stored with archives for reporting.

### Supported Provider Keys
| Provider key      | Description                                                |
|-------------------|------------------------------------------------------------|
| `openai`          | Official OpenAI endpoint (`https://api.openai.com/v1`).    |
| `openai_compatible` / `openrouter` | OpenAI-compatible APIs with custom `base_url`. |
| `anthropic`       | Claude family models via `https://api.anthropic.com`.      |
| `gemini`          | Google Gemini models; requires `endpoint` and `api_key`.   |

Provider keys must match the enum supported by the connector; unknown keys raise validation errors.

## Environment Variable Resolution
- Express secrets as `${ENV_VAR}` placeholders. At load time, the configuration loader performs shell-style substitution.
- Missing environment variables trigger `ConfigurationError` with the name of the unresolved variable.
- Non-secret literals (e.g., `timeout`, `model` names) should remain inline.
- Avoid embedding raw secrets in configuration files. If unavoidable, document why and restrict file access.

## Configuration Merging Rules
When a custom configuration name is supplied, the loader performs a two-step merge:
1. Load `default-llms.yaml` as the base.
2. Load the requested file and overlay it.

Overlay semantics:
- Provider maps merge by key: new providers are appended; existing providers are updated.
- Scalar fields (`timeout`, `max_retries`, etc.) replace the base value when redefined.
- `models` lists are **replaced** in full when the overlay declares them.
- Nested dictionaries (`default_params`, `provider_specific`) merge recursively, with overlay values winning.
- Absence of a provider in the overlay leaves the default definition intact.

This behavior matches the examples in `README.md` (`research-models.yaml`, `production-models.yaml`) and allows minimal override files.

## Validation Expectations
The loader validates every configuration before exposing it:
- YAML must parse cleanly; malformed documents raise `ConfigurationError`.
- Required sections (`providers`, provider-level `api_key`, `models`) must be present.
- Temperatures must be within `0.0–2.0`; negative token limits or retry counts are rejected.
- Provider identifiers must be in the supported registry.
- Environment references must resolve.
- Streaming controls are disallowed and rejected.
- Validation errors report the file path and the offending field to aid debugging.

## Example Override File

```yaml
# config/llms/research-models.yaml
providers:
  openai:
    models:
      - "gpt-4-turbo"
      - "gpt-4o"
    default_params:
      temperature: 0.1
  local:
    endpoint: "http://localhost:8000/v1"
    timeout: 60
    models:
      - "llama-2-7b-chat"
      - "mistral-7b-instruct"
    provider_specific:
      authentication: "none"
```

This file leaves all default providers untouched except for OpenAI. It introduces a new `local` provider with a longer timeout and no authentication. Runtime merging yields:
- OpenAI inherits global defaults but replaces its model list and temperature.
- Anthropic (defined in `default-llms.yaml`) remains available unless explicitly removed.
- The new `local` provider becomes selectable in CLI commands.

## CLI Usage Recap
- Run with defaults: `uv run rationale-benchmark --llm-config default-llms`
- Switch configs: `uv run rationale-benchmark --llm-config research-models`
- Combine with questionnaires: `uv run rationale-benchmark --questionnaires moral-reasoning,cognitive-biases --llm-config production-models`

Refer to `README.md` for a full command reference and `docs/01-llm/design.md` for architectural context when integrating new provider adapters.
