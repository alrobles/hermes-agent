# Changelog

## Unreleased

### Added

- **`hermes-fast` model alias** — bypass the AIAgent loop for chat-only prompts.
  Send `model: "hermes-fast"` to proxy directly to the upstream provider without
  the ~16K-token agentic system prompt or tool-calling iterations. Supports
  provider/model override syntax: `hermes-fast:<provider>[:<upstream_model>]`.
  TTFT drops from ~4.5s to sub-second for simple prompts.

- **`hermes-reasoner` model alias** — same bypass as `hermes-fast`, but
  automatically injects `thinking: {type: enabled}` and `reasoning_effort: high`
  into the upstream request. Upstream `reasoning_content` is preserved as a
  separate field in both non-streaming responses and SSE deltas (not merged
  into `content`). Client-provided thinking params take precedence (setdefault).

- **`/v1/models` expanded** — now lists `hermes-agent`, `hermes-fast`, and
  `hermes-reasoner` as available models.

- **`/v1/capabilities` expanded** — `features.fast_bypass` and
  `features.reasoner_bypass` flags advertise bypass support.

- **`usage.prompt_tokens_details` passthrough** — bypass responses preserve
  upstream `prompt_tokens_details` (including `cached_tokens`) and
  `completion_tokens_details` instead of stripping them during normalization.

- Tests for all bypass paths in `tests/gateway/test_api_server_fast_bypass.py`.
