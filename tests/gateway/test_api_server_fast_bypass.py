"""
Tests for the hermes-fast / hermes-reasoner bypass path.

Covers:
- Model name parsing (_parse_bypass_model)
- _is_reasoner_model helper
- /v1/models listing includes all three model aliases
- hermes-fast non-streaming bypass
- hermes-fast streaming bypass
- hermes-reasoner injects thinking parameters
- hermes-reasoner preserves reasoning_content in SSE deltas
- Provider override via model name segments (hermes-fast:deepseek:deepseek-chat)
- Upstream error forwarding
- Missing credentials error
"""

import asyncio
import json
import os
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _parse_bypass_model,
    _is_reasoner_model,
    _FAST_MODEL_PREFIXES,
    _REASONER_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(api_key: str = "", cors_origins=None) -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    if cors_origins is not None:
        extra["cors_origins"] = cors_origins
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    from gateway.platforms.api_server import cors_middleware, security_headers_middleware
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


# ---------------------------------------------------------------------------
# Unit tests: _parse_bypass_model
# ---------------------------------------------------------------------------

class TestParseBypassModel:
    def test_hermes_fast_plain(self):
        result = _parse_bypass_model("hermes-fast")
        assert result is not None
        assert result["alias"] == "hermes-fast"
        assert result["provider"] is None
        assert result["upstream_model"] is None

    def test_hermes_fast_with_provider(self):
        result = _parse_bypass_model("hermes-fast:deepseek")
        assert result["alias"] == "hermes-fast"
        assert result["provider"] == "deepseek"
        assert result["upstream_model"] is None

    def test_hermes_fast_with_provider_and_model(self):
        result = _parse_bypass_model("hermes-fast:deepseek:deepseek-chat")
        assert result["alias"] == "hermes-fast"
        assert result["provider"] == "deepseek"
        assert result["upstream_model"] == "deepseek-chat"

    def test_hermes_reasoner_plain(self):
        result = _parse_bypass_model("hermes-reasoner")
        assert result is not None
        assert result["alias"] == "hermes-reasoner"

    def test_hermes_reasoner_with_provider_and_model(self):
        result = _parse_bypass_model("hermes-reasoner:deepseek:deepseek-reasoner")
        assert result["alias"] == "hermes-reasoner"
        assert result["provider"] == "deepseek"
        assert result["upstream_model"] == "deepseek-reasoner"

    def test_hermes_agent_not_bypass(self):
        assert _parse_bypass_model("hermes-agent") is None

    def test_random_model_not_bypass(self):
        assert _parse_bypass_model("gpt-4") is None

    def test_empty_string(self):
        assert _parse_bypass_model("") is None

    def test_empty_provider_segment(self):
        result = _parse_bypass_model("hermes-fast:")
        assert result is not None
        assert result["provider"] is None


class TestIsReasonerModel:
    def test_plain_reasoner(self):
        assert _is_reasoner_model("hermes-reasoner") is True

    def test_reasoner_with_segments(self):
        assert _is_reasoner_model("hermes-reasoner:deepseek:model") is True

    def test_fast_is_not_reasoner(self):
        assert _is_reasoner_model("hermes-fast") is False

    def test_agent_is_not_reasoner(self):
        assert _is_reasoner_model("hermes-agent") is False


# ---------------------------------------------------------------------------
# /v1/models lists all three aliases
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_models_lists_three_aliases(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "list"
            model_ids = [m["id"] for m in data["data"]]
            assert "hermes-agent" in model_ids
            assert "hermes-fast" in model_ids
            assert "hermes-reasoner" in model_ids
            assert len(data["data"]) == 3

    @pytest.mark.asyncio
    async def test_models_all_owned_by_hermes(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            data = await resp.json()
            for model in data["data"]:
                assert model["owned_by"] == "hermes"
                assert model["object"] == "model"

    @pytest.mark.asyncio
    async def test_models_with_valid_auth_lists_three(self):
        """Auth + model listing combined (auth-only tests live in test_api_server.py)."""
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(
                "/v1/models",
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 200
            data = await resp.json()
            model_ids = [m["id"] for m in data["data"]]
            assert "hermes-fast" in model_ids
            assert "hermes-reasoner" in model_ids
            assert len(data["data"]) == 3


# ---------------------------------------------------------------------------
# /v1/capabilities advertises bypass features
# ---------------------------------------------------------------------------

class TestCapabilitiesBypass:
    @pytest.mark.asyncio
    async def test_capabilities_includes_bypass_features(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["features"]["fast_bypass"] is True
            assert data["features"]["reasoner_bypass"] is True


# ---------------------------------------------------------------------------
# hermes-fast: non-streaming bypass
# ---------------------------------------------------------------------------

class TestFastBypassNonStreaming:
    @pytest.mark.asyncio
    async def test_fast_bypass_proxies_to_upstream(self):
        """hermes-fast should bypass the agent loop and proxy to upstream."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        mock_upstream_response = {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
        }

        async def mock_post(self_session, url, json=None, headers=None):
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_upstream_response)
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            return mock_resp

        with patch.object(adapter, "_resolve_upstream_credentials", return_value={
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-test",
            "model": "deepseek-chat",
            "provider": "deepseek",
        }):
            import aiohttp
            with patch("aiohttp.ClientSession") as MockSession:
                mock_session = AsyncMock()
                mock_resp_ctx = AsyncMock()
                mock_resp_ctx.status = 200
                mock_resp_ctx.json = AsyncMock(return_value=mock_upstream_response)
                mock_resp_ctx.__aenter__ = AsyncMock(return_value=mock_resp_ctx)
                mock_resp_ctx.__aexit__ = AsyncMock(return_value=False)
                mock_session.post = MagicMock(return_value=mock_resp_ctx)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=False)
                MockSession.return_value = mock_session

                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-fast",
                            "messages": [{"role": "user", "content": "ping"}],
                        },
                    )
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["model"] == "hermes-fast"
                    assert data["choices"][0]["message"]["content"] == "pong"
                    # Verify usage passthrough including details
                    assert data["usage"]["prompt_tokens"] == 10
                    assert data["usage"]["prompt_tokens_details"]["cached_tokens"] == 5

    @pytest.mark.asyncio
    async def test_fast_bypass_no_credentials_returns_502(self):
        """Missing upstream credentials should return 502."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        with patch.object(adapter, "_resolve_upstream_credentials", return_value={
            "base_url": "",
            "api_key": "",
            "model": "",
            "provider": "",
        }):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-fast",
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                )
                assert resp.status == 502


# ---------------------------------------------------------------------------
# hermes-reasoner: non-streaming with thinking params
# ---------------------------------------------------------------------------

class TestReasonerBypassNonStreaming:
    @pytest.mark.asyncio
    async def test_reasoner_injects_thinking_params(self):
        """hermes-reasoner should inject thinking and reasoning_effort."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        captured_body = {}

        mock_upstream_response = {
            "id": "chatcmpl-upstream",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "QED",
                    "reasoning_content": "Step 1: assume sqrt(2) is rational...",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }

        import aiohttp
        with patch.object(adapter, "_resolve_upstream_credentials", return_value={
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-test",
            "model": "deepseek-reasoner",
            "provider": "deepseek",
        }):
            with patch("aiohttp.ClientSession") as MockSession:
                mock_session = AsyncMock()
                mock_resp_ctx = AsyncMock()
                mock_resp_ctx.status = 200
                mock_resp_ctx.json = AsyncMock(return_value=mock_upstream_response)
                mock_resp_ctx.__aenter__ = AsyncMock(return_value=mock_resp_ctx)
                mock_resp_ctx.__aexit__ = AsyncMock(return_value=False)

                def capture_post(url, json=None, headers=None):
                    captured_body.update(json or {})
                    return mock_resp_ctx

                mock_session.post = MagicMock(side_effect=capture_post)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=False)
                MockSession.return_value = mock_session

                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-reasoner",
                            "messages": [{"role": "user", "content": "Prove sqrt(2) is irrational"}],
                        },
                    )
                    assert resp.status == 200
                    data = await resp.json()

                    # Verify thinking params were injected
                    assert captured_body.get("thinking") == {"type": "enabled"}
                    assert captured_body.get("reasoning_effort") == "high"

                    # Verify reasoning_content is preserved
                    assert data["choices"][0]["message"]["reasoning_content"] == "Step 1: assume sqrt(2) is rational..."
                    assert data["choices"][0]["message"]["content"] == "QED"

    @pytest.mark.asyncio
    async def test_reasoner_client_override_preserved(self):
        """Client-provided thinking params should not be overwritten."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        captured_body = {}

        mock_upstream_response = {
            "id": "chatcmpl-upstream",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }

        with patch.object(adapter, "_resolve_upstream_credentials", return_value={
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-test",
            "model": "deepseek-reasoner",
            "provider": "deepseek",
        }):
            import aiohttp
            with patch("aiohttp.ClientSession") as MockSession:
                mock_session = AsyncMock()
                mock_resp_ctx = AsyncMock()
                mock_resp_ctx.status = 200
                mock_resp_ctx.json = AsyncMock(return_value=mock_upstream_response)
                mock_resp_ctx.__aenter__ = AsyncMock(return_value=mock_resp_ctx)
                mock_resp_ctx.__aexit__ = AsyncMock(return_value=False)

                def capture_post(url, json=None, headers=None):
                    captured_body.update(json or {})
                    return mock_resp_ctx

                mock_session.post = MagicMock(side_effect=capture_post)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=False)
                MockSession.return_value = mock_session

                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-reasoner",
                            "messages": [{"role": "user", "content": "hi"}],
                            "reasoning_effort": "low",
                            "thinking": {"type": "disabled"},
                        },
                    )
                    assert resp.status == 200
                    # Client override should be preserved (setdefault)
                    assert captured_body["reasoning_effort"] == "low"
                    assert captured_body["thinking"] == {"type": "disabled"}


# ---------------------------------------------------------------------------
# hermes-fast: streaming bypass
# ---------------------------------------------------------------------------

class TestFastBypassStreaming:
    @pytest.mark.asyncio
    async def test_fast_bypass_streaming_returns_sse(self):
        """hermes-fast with stream=true should return SSE chunks."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        sse_lines = [
            b'data: {"choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"po"},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"ng"},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n',
            b'data: [DONE]\n\n',
        ]

        with patch.object(adapter, "_resolve_upstream_credentials", return_value={
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-test",
            "model": "deepseek-chat",
            "provider": "deepseek",
        }):
            import aiohttp
            with patch("aiohttp.ClientSession") as MockSession:
                mock_session = AsyncMock()
                mock_resp_ctx = AsyncMock()
                mock_resp_ctx.status = 200

                async def line_iter():
                    for line in sse_lines:
                        yield line

                mock_resp_ctx.content = line_iter()
                mock_resp_ctx.__aenter__ = AsyncMock(return_value=mock_resp_ctx)
                mock_resp_ctx.__aexit__ = AsyncMock(return_value=False)
                mock_session.post = MagicMock(return_value=mock_resp_ctx)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=False)
                MockSession.return_value = mock_session

                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-fast",
                            "messages": [{"role": "user", "content": "ping"}],
                            "stream": True,
                        },
                    )
                    assert resp.status == 200
                    assert "text/event-stream" in resp.headers.get("Content-Type", "")

                    body = await resp.read()
                    text = body.decode("utf-8")
                    # Should contain role chunk, content chunks, and [DONE]
                    assert "assistant" in text
                    assert "[DONE]" in text


# ---------------------------------------------------------------------------
# hermes-reasoner: streaming with reasoning_content
# ---------------------------------------------------------------------------

class TestReasonerBypassStreaming:
    @pytest.mark.asyncio
    async def test_reasoner_streaming_preserves_reasoning_content(self):
        """hermes-reasoner streaming should preserve reasoning_content deltas."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        sse_lines = [
            b'data: {"choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"reasoning_content":"Let me think..."},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"reasoning_content":" step 1"},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"The answer is 42"},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n',
            b'data: [DONE]\n\n',
        ]

        with patch.object(adapter, "_resolve_upstream_credentials", return_value={
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-test",
            "model": "deepseek-reasoner",
            "provider": "deepseek",
        }):
            import aiohttp
            with patch("aiohttp.ClientSession") as MockSession:
                mock_session = AsyncMock()
                mock_resp_ctx = AsyncMock()
                mock_resp_ctx.status = 200

                async def line_iter():
                    for line in sse_lines:
                        yield line

                mock_resp_ctx.content = line_iter()
                mock_resp_ctx.__aenter__ = AsyncMock(return_value=mock_resp_ctx)
                mock_resp_ctx.__aexit__ = AsyncMock(return_value=False)
                mock_session.post = MagicMock(return_value=mock_resp_ctx)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=False)
                MockSession.return_value = mock_session

                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-reasoner",
                            "messages": [{"role": "user", "content": "Think hard"}],
                            "stream": True,
                        },
                    )
                    assert resp.status == 200
                    body = await resp.read()
                    text = body.decode("utf-8")
                    # Should contain reasoning_content in the SSE stream
                    assert "reasoning_content" in text
                    assert "Let me think..." in text
                    assert "The answer is 42" in text
                    assert "[DONE]" in text


# ---------------------------------------------------------------------------
# Upstream error forwarding
# ---------------------------------------------------------------------------

class TestUpstreamErrorForwarding:
    @pytest.mark.asyncio
    async def test_upstream_error_forwarded(self):
        """Upstream 4xx/5xx should be forwarded to the client."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        with patch.object(adapter, "_resolve_upstream_credentials", return_value={
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-test",
            "model": "deepseek-chat",
            "provider": "deepseek",
        }):
            import aiohttp
            with patch("aiohttp.ClientSession") as MockSession:
                mock_session = AsyncMock()
                mock_resp_ctx = AsyncMock()
                mock_resp_ctx.status = 429
                mock_resp_ctx.text = AsyncMock(return_value='{"error": "rate limited"}')
                mock_resp_ctx.__aenter__ = AsyncMock(return_value=mock_resp_ctx)
                mock_resp_ctx.__aexit__ = AsyncMock(return_value=False)
                mock_session.post = MagicMock(return_value=mock_resp_ctx)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=False)
                MockSession.return_value = mock_session

                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-fast",
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                    assert resp.status == 429


# ---------------------------------------------------------------------------
# Agent-path NOT triggered for bypass models
# ---------------------------------------------------------------------------

class TestBypassSkipsAgentLoop:
    @pytest.mark.asyncio
    async def test_hermes_agent_still_uses_agent_loop(self):
        """hermes-agent model should NOT use the fast bypass."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        # Patch _run_agent to verify it gets called for hermes-agent
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (
                {"final_response": "hello", "completed": True},
                {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
            )
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status == 200
                mock_run.assert_called_once()
