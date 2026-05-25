"""
Tests for cached_tokens passthrough (P3).

Covers:
- _build_usage_block helper
- Non-streaming: prompt_tokens_details in response when agent reports cached_tokens
- Non-streaming: no prompt_tokens_details when no cache hits
- Streaming: prompt_tokens_details in finish chunk
- completion_tokens_details.reasoning_tokens passthrough
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _build_usage_block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={})
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    from gateway.platforms.api_server import cors_middleware, security_headers_middleware
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


# ---------------------------------------------------------------------------
# Unit tests: _build_usage_block
# ---------------------------------------------------------------------------


class TestBuildUsageBlock:
    def test_basic_fields(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        out = _build_usage_block(usage)
        assert out == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_prompt_tokens_details_passthrough(self):
        usage = {
            "input_tokens": 16188,
            "output_tokens": 412,
            "total_tokens": 16600,
            "prompt_tokens_details": {"cached_tokens": 14336},
        }
        out = _build_usage_block(usage)
        assert out["prompt_tokens_details"] == {"cached_tokens": 14336}

    def test_completion_tokens_details_passthrough(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 200},
        }
        out = _build_usage_block(usage)
        assert out["completion_tokens_details"] == {"reasoning_tokens": 200}

    def test_both_details_passthrough(self):
        usage = {
            "input_tokens": 16188,
            "output_tokens": 412,
            "total_tokens": 16600,
            "prompt_tokens_details": {"cached_tokens": 14336, "cache_creation_tokens": 500},
            "completion_tokens_details": {"reasoning_tokens": 200},
        }
        out = _build_usage_block(usage)
        assert out["prompt_tokens_details"]["cached_tokens"] == 14336
        assert out["prompt_tokens_details"]["cache_creation_tokens"] == 500
        assert out["completion_tokens_details"]["reasoning_tokens"] == 200

    def test_no_details_when_absent(self):
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        out = _build_usage_block(usage)
        assert "prompt_tokens_details" not in out
        assert "completion_tokens_details" not in out

    def test_empty_usage(self):
        out = _build_usage_block({})
        assert out["prompt_tokens"] == 0
        assert out["completion_tokens"] == 0
        assert out["total_tokens"] == 0


# ---------------------------------------------------------------------------
# Integration tests: non-streaming
# ---------------------------------------------------------------------------


class TestCachedTokensNonStreaming:
    @pytest.mark.asyncio
    async def test_cached_tokens_in_response(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_result = {
                "final_response": "Hello!",
                "messages": [],
                "api_calls": 1,
            }
            usage_with_cache = {
                "input_tokens": 16188,
                "output_tokens": 412,
                "total_tokens": 16600,
                "prompt_tokens_details": {"cached_tokens": 14336},
            }
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage_with_cache)
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            assert resp.status == 200
            data = await resp.json()
            assert "prompt_tokens_details" in data["usage"]
            assert data["usage"]["prompt_tokens_details"]["cached_tokens"] == 14336

    @pytest.mark.asyncio
    async def test_no_details_when_no_cache(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_result = {
                "final_response": "Hello!",
                "messages": [],
                "api_calls": 1,
            }
            usage_no_cache = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage_no_cache)
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            assert resp.status == 200
            data = await resp.json()
            assert "prompt_tokens_details" not in data["usage"]

    @pytest.mark.asyncio
    async def test_reasoning_tokens_in_response(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_result = {
                "final_response": "Proof...",
                "messages": [],
                "api_calls": 1,
            }
            usage_with_reasoning = {
                "input_tokens": 500,
                "output_tokens": 300,
                "total_tokens": 800,
                "completion_tokens_details": {"reasoning_tokens": 200},
            }
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage_with_reasoning)
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "prove it"}],
                    },
                )
            assert resp.status == 200
            data = await resp.json()
            assert data["usage"]["completion_tokens_details"]["reasoning_tokens"] == 200


# ---------------------------------------------------------------------------
# Integration tests: streaming
# ---------------------------------------------------------------------------


class TestCachedTokensStreaming:
    @pytest.mark.asyncio
    async def test_cached_tokens_in_finish_chunk(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello!")
                    cb(None)
                return (
                    {"final_response": "Hello!", "messages": [], "api_calls": 1},
                    {
                        "input_tokens": 16188,
                        "output_tokens": 412,
                        "total_tokens": 16600,
                        "prompt_tokens_details": {"cached_tokens": 14336},
                    },
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
            assert resp.status == 200
            body = await resp.text()

            # Find the finish chunk (has finish_reason: "stop")
            for line in body.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    if chunk.get("choices", [{}])[0].get("finish_reason") == "stop":
                        assert "prompt_tokens_details" in chunk["usage"]
                        assert chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 14336
                        break
            else:
                pytest.fail("No finish chunk found in SSE stream")

    @pytest.mark.asyncio
    async def test_no_details_in_finish_chunk_when_no_cache(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello!")
                    cb(None)
                return (
                    {"final_response": "Hello!", "messages": [], "api_calls": 1},
                    {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
            assert resp.status == 200
            body = await resp.text()

            for line in body.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    if chunk.get("choices", [{}])[0].get("finish_reason") == "stop":
                        assert "prompt_tokens_details" not in chunk["usage"]
                        break
