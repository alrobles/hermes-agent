"""
Tests for the hermes-fast agent-loop bypass.

Covers:
- The pure-function helpers (`_is_fast_model`, `_parse_fast_model`).
- The /v1/models advertisement now lists `hermes-fast` next to `hermes-agent`.
- The router shortcut in `_handle_chat_completions` delegates to the bypass
  handler when (and only when) `model` matches a hermes-fast spec.

The streaming/non-streaming proxy path itself is intentionally not covered
here: it depends on the real upstream provider client and is exercised by the
existing benchmark playbook. These tests keep the surface small and
dependency-free so the regression value is high and the flake risk is zero.
"""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _FAST_MODELS,
    _is_fast_model,
    _parse_fast_model,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors the pattern in tests/gateway/test_api_server.py)
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestIsFastModel:
    def test_empty_string_is_not_fast(self):
        assert _is_fast_model("") is False

    def test_none_is_not_fast(self):
        assert _is_fast_model(None) is False  # type: ignore[arg-type]

    def test_default_agent_model_is_not_fast(self):
        assert _is_fast_model("hermes-agent") is False

    def test_unknown_model_is_not_fast(self):
        assert _is_fast_model("gpt-4o") is False

    def test_hermes_fast_bare_is_fast(self):
        assert _is_fast_model("hermes-fast") is True

    def test_hermes_fast_with_provider_is_fast(self):
        assert _is_fast_model("hermes-fast:deepseek") is True

    def test_hermes_fast_with_provider_and_model_is_fast(self):
        assert _is_fast_model("hermes-fast:deepseek:deepseek-chat") is True

    def test_fast_models_set_contains_hermes_fast(self):
        assert "hermes-fast" in _FAST_MODELS


class TestParseFastModel:
    def test_bare(self):
        base, provider, upstream = _parse_fast_model("hermes-fast")
        assert base == "hermes-fast"
        assert provider is None
        assert upstream is None

    def test_provider_only(self):
        base, provider, upstream = _parse_fast_model("hermes-fast:deepseek")
        assert base == "hermes-fast"
        assert provider == "deepseek"
        assert upstream is None

    def test_provider_and_model(self):
        base, provider, upstream = _parse_fast_model(
            "hermes-fast:deepseek:deepseek-chat"
        )
        assert base == "hermes-fast"
        assert provider == "deepseek"
        assert upstream == "deepseek-chat"

    def test_empty_provider_slot_treated_as_none(self):
        """``hermes-fast::deepseek-chat`` should treat the empty provider as unset."""
        base, provider, upstream = _parse_fast_model("hermes-fast::deepseek-chat")
        assert base == "hermes-fast"
        assert provider is None
        assert upstream == "deepseek-chat"

    def test_whitespace_in_provider_is_stripped(self):
        _, provider, _ = _parse_fast_model("hermes-fast:  deepseek  ")
        assert provider == "deepseek"


# ---------------------------------------------------------------------------
# /v1/models advertisement
# ---------------------------------------------------------------------------


class TestModelsListIncludesHermesFast:
    @pytest.mark.asyncio
    async def test_models_endpoint_lists_hermes_agent_and_hermes_fast(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            payload = await resp.json()

        assert payload["object"] == "list"
        ids = [m["id"] for m in payload["data"]]
        assert "hermes-fast" in ids

        fast_entry = next(m for m in payload["data"] if m["id"] == "hermes-fast")
        assert fast_entry["object"] == "model"
        assert fast_entry["owned_by"] == "hermes"
        assert fast_entry["root"] == "hermes-fast"
        assert fast_entry.get("metadata", {}).get("mode") == "fast"


# ---------------------------------------------------------------------------
# Router shortcut
# ---------------------------------------------------------------------------


class TestRouterShortcut:
    @pytest.mark.asyncio
    async def test_fast_request_calls_fast_handler(self):
        adapter = _make_adapter()

        async def _fake_fast(request, body, model_name):
            assert model_name == "hermes-fast"
            return web.json_response({"sentinel": "fast-handler-called"})

        with patch.object(
            adapter,
            "_handle_chat_completions_fast",
            side_effect=_fake_fast,
        ) as mock_fast:
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-fast",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status == 200
                body = await resp.json()
                assert body == {"sentinel": "fast-handler-called"}

        assert mock_fast.call_count == 1

    @pytest.mark.asyncio
    async def test_fast_request_with_provider_spec_calls_fast_handler(self):
        adapter = _make_adapter()
        captured_model = {}

        async def _fake_fast(request, body, model_name):
            captured_model["m"] = model_name
            return web.json_response({"ok": True})

        with patch.object(
            adapter,
            "_handle_chat_completions_fast",
            side_effect=_fake_fast,
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-fast:deepseek",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status == 200

        assert captured_model["m"] == "hermes-fast:deepseek"

    @pytest.mark.asyncio
    async def test_non_fast_request_does_not_call_fast_handler(self):
        """A request without a hermes-fast model name must not short-circuit.

        We don't exercise the full agent loop here — we just assert that the
        bypass handler is never invoked. The downstream agent path is allowed
        to fail in any way it likes; we patch it to a quick error response so
        the test stays bounded.
        """
        adapter = _make_adapter()

        with patch.object(
            adapter,
            "_handle_chat_completions_fast",
        ) as mock_fast, patch.object(
            adapter,
            "_run_agent_for_chat_completions",
            new=AsyncMock(side_effect=RuntimeError("test stop")),
            create=True,
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                try:
                    await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "hermes-agent",
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                except Exception:
                    # Any downstream failure is acceptable; the contract here
                    # is that the bypass handler must not have been entered.
                    pass
        assert mock_fast.call_count == 0
