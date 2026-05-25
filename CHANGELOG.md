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

- **`hermes_trace` opt-in telemetry** — send `"hermes": {"trace": true}` in the
  request body to receive agent-loop telemetry in the response. Exposes per-LLM-call
  token counts (prompt, cached, completion), timing, tool call names/durations,
  iteration count, and gateway metadata. Non-streaming: `hermes_trace` top-level
  field in JSON response. Streaming: `event: hermes.trace` SSE event before `[DONE]`.
  Bounded at 50 entries; no PII (tool names only, never arguments or results).
  Tests in `tests/gateway/test_hermes_trace.py`.
