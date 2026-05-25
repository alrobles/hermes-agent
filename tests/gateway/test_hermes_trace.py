"""
Tests for the hermes_trace opt-in telemetry extension (P2).

Covers:
- HermesTrace dataclass: recording, serialization, truncation
- Non-streaming: hermes_trace attached to response JSON when opt-in
- Non-streaming: no trace when opt-in is absent or false
- Streaming: hermes.trace SSE event emitted before [DONE]
- Streaming: no trace event when opt-in is absent
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.hermes_trace import HermesTrace, LLMCallTrace, ToolCallTrace, MAX_ENTRIES
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


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
# Unit tests: HermesTrace dataclass
# ---------------------------------------------------------------------------


class TestHermesTraceDataclass:
    def test_empty_trace_serializes(self):
        t = HermesTrace(model="hermes-agent")
        d = t.to_dict()
        assert d["model"] == "hermes-agent"
        assert d["agent_loop"]["iterations"] == 0
        assert d["agent_loop"]["llm_calls"] == []
        assert d["agent_loop"]["tool_calls"] == []
        assert d["agent_loop"]["total_ms"] >= 0

    def test_record_llm_call(self):
        t = HermesTrace(model="hermes-agent")
        t.record_llm_call(
            iteration=1,
            upstream="deepseek",
            model="deepseek-v4-pro",
            prompt_tokens=16188,
            cached_tokens=14336,
            completion_tokens=412,
            total_ms=2840,
        )
        d = t.to_dict()
        assert len(d["agent_loop"]["llm_calls"]) == 1
        call = d["agent_loop"]["llm_calls"][0]
        assert call["iter"] == 1
        assert call["upstream"] == "deepseek"
        assert call["model"] == "deepseek-v4-pro"
        assert call["prompt_tokens"] == 16188
        assert call["cached_tokens"] == 14336
        assert call["total_ms"] == 2840

    def test_record_tool_call(self):
        t = HermesTrace(model="hermes-agent")
        t.record_tool_call(name="skill_manage", duration_ms=412, status="ok")
        t.record_tool_call(name="memory_search", duration_ms=87, status="ok")
        d = t.to_dict()
        assert len(d["agent_loop"]["tool_calls"]) == 2
        assert d["agent_loop"]["tool_calls"][0]["name"] == "skill_manage"
        assert d["agent_loop"]["tool_calls"][1]["duration_ms"] == 87

    def test_iterations_count(self):
        t = HermesTrace(model="hermes-agent")
        t.record_llm_call(iteration=1, upstream="deepseek", model="m")
        t.record_llm_call(iteration=2, upstream="deepseek", model="m")
        t.record_llm_call(iteration=3, upstream="deepseek", model="m")
        d = t.to_dict()
        assert d["agent_loop"]["iterations"] == 3

    def test_session_id_in_gateway(self):
        t = HermesTrace(model="hermes-agent", session_id="abc-123")
        d = t.to_dict()
        assert d["gateway"]["session_id"] == "abc-123"

    def test_system_prompt_tokens_in_gateway(self):
        t = HermesTrace(model="hermes-agent", system_prompt_tokens=16188)
        d = t.to_dict()
        assert d["gateway"]["system_prompt_tokens"] == 16188

    def test_truncation_at_max_entries(self):
        t = HermesTrace(model="hermes-agent")
        for i in range(MAX_ENTRIES + 5):
            t.record_llm_call(iteration=i, upstream="deepseek", model="m")
        d = t.to_dict()
        assert len(d["agent_loop"]["llm_calls"]) == MAX_ENTRIES
        assert d["agent_loop"]["truncated"] is True

    def test_tool_call_truncation(self):
        t = HermesTrace(model="hermes-agent")
        for i in range(MAX_ENTRIES + 3):
            t.record_tool_call(name=f"tool_{i}", duration_ms=10)
        d = t.to_dict()
        assert len(d["agent_loop"]["tool_calls"]) == MAX_ENTRIES
        assert d["agent_loop"]["truncated"] is True

    def test_ttft_ms_optional(self):
        t = HermesTrace(model="m")
        t.record_llm_call(iteration=1, upstream="x", model="m", ttft_ms=920, total_ms=2840)
        d = t.to_dict()
        assert d["agent_loop"]["llm_calls"][0]["ttft_ms"] == 920

    def test_ttft_ms_absent_when_none(self):
        t = HermesTrace(model="m")
        t.record_llm_call(iteration=1, upstream="x", model="m", total_ms=2840)
        d = t.to_dict()
        assert "ttft_ms" not in d["agent_loop"]["llm_calls"][0]


# ---------------------------------------------------------------------------
# Integration tests: non-streaming
# ---------------------------------------------------------------------------


class TestHermesTraceNonStreaming:
    @pytest.mark.asyncio
    async def test_trace_present_when_opt_in(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_trace_dict = {
                "model": "hermes-agent",
                "agent_loop": {
                    "iterations": 1,
                    "llm_calls": [{"iter": 1, "upstream": "deepseek", "model": "m",
                                   "prompt_tokens": 100, "cached_tokens": 50, "total_ms": 500}],
                    "tool_calls": [],
                    "total_ms": 600,
                },
                "gateway": {},
            }
            mock_result = {
                "final_response": "Hello!",
                "messages": [],
                "api_calls": 1,
                "hermes_trace": mock_trace_dict,
            }
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                        "hermes": {"trace": True},
                    },
                )
            assert resp.status == 200
            data = await resp.json()
            assert "hermes_trace" in data
            assert data["hermes_trace"]["model"] == "hermes-agent"
            assert data["hermes_trace"]["agent_loop"]["iterations"] == 1
            assert len(data["hermes_trace"]["agent_loop"]["llm_calls"]) == 1

    @pytest.mark.asyncio
    async def test_no_trace_when_not_requested(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_result = {
                "final_response": "Hello!",
                "messages": [],
                "api_calls": 1,
            }
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            assert resp.status == 200
            data = await resp.json()
            assert "hermes_trace" not in data

    @pytest.mark.asyncio
    async def test_no_trace_when_false(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_result = {
                "final_response": "Hello!",
                "messages": [],
                "api_calls": 1,
            }
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                        "hermes": {"trace": False},
                    },
                )
            assert resp.status == 200
            data = await resp.json()
            assert "hermes_trace" not in data

    @pytest.mark.asyncio
    async def test_trace_obj_passed_to_run_agent(self):
        """Verify _run_agent receives a HermesTrace when trace is requested."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            captured_trace = []

            async def _mock_run_agent(**kwargs):
                captured_trace.append(kwargs.get("hermes_trace"))
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                        "hermes": {"trace": True},
                    },
                )

            assert len(captured_trace) == 1
            assert isinstance(captured_trace[0], HermesTrace)

    @pytest.mark.asyncio
    async def test_no_trace_obj_when_not_requested(self):
        """Verify _run_agent receives None for hermes_trace when not opt-in."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            captured_trace = []

            async def _mock_run_agent(**kwargs):
                captured_trace.append(kwargs.get("hermes_trace"))
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )

            assert len(captured_trace) == 1
            assert captured_trace[0] is None


# ---------------------------------------------------------------------------
# Integration tests: streaming
# ---------------------------------------------------------------------------


class TestHermesTraceStreaming:
    @pytest.mark.asyncio
    async def test_trace_sse_event_emitted_before_done(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_trace_dict = {
                "model": "hermes-agent",
                "agent_loop": {
                    "iterations": 2,
                    "llm_calls": [
                        {"iter": 1, "upstream": "deepseek", "model": "m",
                         "prompt_tokens": 100, "cached_tokens": 50, "total_ms": 500},
                    ],
                    "tool_calls": [
                        {"name": "skill_manage", "duration_ms": 412, "status": "ok"},
                    ],
                    "total_ms": 1200,
                },
                "gateway": {"session_id": "test-123"},
            }

            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello world!")
                    cb(None)
                return (
                    {
                        "final_response": "Hello world!",
                        "messages": [],
                        "api_calls": 2,
                        "hermes_trace": mock_trace_dict,
                    },
                    {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                        "hermes": {"trace": True},
                    },
                )
            assert resp.status == 200
            body = await resp.text()

            # Verify hermes.trace event is present
            assert "event: hermes.trace" in body

            # Verify trace event appears before [DONE]
            trace_pos = body.index("event: hermes.trace")
            done_pos = body.index("[DONE]")
            assert trace_pos < done_pos

            # Parse the trace data from the SSE event
            trace_line_start = body.index("event: hermes.trace")
            data_start = body.index("data: ", trace_line_start) + len("data: ")
            data_end = body.index("\n", data_start)
            trace_data = json.loads(body[data_start:data_end])
            assert trace_data["model"] == "hermes-agent"
            assert trace_data["agent_loop"]["iterations"] == 2
            assert len(trace_data["agent_loop"]["tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_no_trace_sse_event_when_not_requested(self):
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
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
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
            assert "event: hermes.trace" not in body
            assert "[DONE]" in body


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestHermesTraceBackwardCompat:
    @pytest.mark.asyncio
    async def test_hermes_key_ignored_when_not_dict(self):
        """hermes as non-dict should not crash."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_result = {
                "final_response": "ok",
                "messages": [],
                "api_calls": 1,
            }
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                        "hermes": "not-a-dict",
                    },
                )
            assert resp.status == 200
            data = await resp.json()
            assert "hermes_trace" not in data

    @pytest.mark.asyncio
    async def test_hermes_trace_non_boolean_ignored(self):
        """hermes.trace must be exactly True (boolean)."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_result = {
                "final_response": "ok",
                "messages": [],
                "api_calls": 1,
            }
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hi"}],
                        "hermes": {"trace": 1},  # truthy but not True
                    },
                )
            assert resp.status == 200
            data = await resp.json()
            assert "hermes_trace" not in data
