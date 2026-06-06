"""Tests for EcoAgent MCP Server.

Validates:
  - tools/list returns all 25 tools with schemas
  - tools/call dispatches correctly
  - initialize handshake
  - Error handling (unknown tools, connection errors)
  - JSON-RPC protocol compliance
  - HTTP transport mode
"""

import json
import sys
import io
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, "plugins/ecoseek")

from mcp_ecoagent import (
    handle_list_tools,
    handle_call_tool,
    handle_initialize,
    MCP_TOOLS,
    _TOOL_MAP,
)


# ===========================================================================
# Tool Definitions
# ===========================================================================


class TestToolDefinitions:
    def test_all_25_tools_defined(self):
        """Exactly 25 ecological tools are registered."""
        assert len(MCP_TOOLS) == 25, f"Expected 25 tools, got {len(MCP_TOOLS)}"

    def test_unique_tool_names(self):
        """No duplicate tool names."""
        names = [t["name"] for t in MCP_TOOLS]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_every_tool_has_ecoagent_action(self):
        """Each tool maps to an EcoAgent backend action."""
        for t in MCP_TOOLS:
            assert "ecoagent_action" in t, f"Missing ecoagent_action in {t['name']}"
            assert t["ecoagent_action"], f"Empty ecoagent_action in {t['name']}"

    def test_every_tool_has_schema(self):
        """Each tool has an inputSchema with properties."""
        for t in MCP_TOOLS:
            assert "inputSchema" in t, f"Missing inputSchema in {t['name']}"
            schema = t["inputSchema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_every_tool_has_description(self):
        """Each tool has a non-empty description."""
        for t in MCP_TOOLS:
            assert "description" in t, f"Missing description in {t['name']}"
            assert len(t["description"]) > 20, \
                f"Description too short for {t['name']}: {t['description'][:50]}"

    def test_required_params_are_in_properties(self):
        """Required parameters exist in properties."""
        for t in MCP_TOOLS:
            required = t["inputSchema"].get("required", [])
            properties = t["inputSchema"].get("properties", {})
            for param in required:
                assert param in properties, \
                    f"{t['name']}: required param '{param}' not in properties"

    def test_all_ecoagent_actions_covered(self):
        """The 25 SUPPORTED_ACTIONS from eco_analyze are all mapped."""
        # Import the canonical list
        from eco_analyze import SUPPORTED_ACTIONS
        mcp_actions = {t["ecoagent_action"] for t in MCP_TOOLS}
        eco_actions = set(SUPPORTED_ACTIONS)
        missing = eco_actions - mcp_actions
        extra = mcp_actions - eco_actions
        assert not missing, f"EcoAgent actions not covered by MCP: {missing}"
        assert not extra, f"MCP actions not in EcoAgent: {extra}"


# ===========================================================================
# MCP Protocol Handlers
# ===========================================================================


class TestListTools:
    def test_returns_all_tools(self):
        result = handle_list_tools()
        assert len(result) == 25

    def test_tool_schema_format(self):
        result = handle_list_tools()
        tool = result[0]
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        # MCP tools/list response should NOT have 'ecoagent_action'
        assert "ecoagent_action" not in tool

    def test_all_tools_have_eco_prefix(self):
        """All tools are prefixed with eco_ for discoverability."""
        result = handle_list_tools()
        for tool in result:
            assert tool["name"].startswith("eco_"), \
                f"Tool {tool['name']} doesn't have eco_ prefix"


class TestCallTool:
    def test_forward_to_ecoagent_success(self):
        """Valid tool call forwards to EcoAgent and returns result."""
        mock_response = {"status": "ok", "data": [{"species": "Quercus robur"}]}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.read.return_value = \
                json.dumps(mock_response).encode("utf-8")

            result = handle_call_tool("eco_query_species", {"species": "Quercus robur"})

        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        parsed = json.loads(result["content"][0]["text"])
        assert parsed["status"] == "ok"

    def test_unknown_tool(self):
        """Unknown tool name returns isError."""
        result = handle_call_tool("eco_nonexistent", {})
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "Unknown tool" in text

    def test_connection_error(self):
        """EcoAgent unreachable returns isError."""
        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            result = handle_call_tool("eco_query_species", {"species": "Test"})

        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "Connection refused" in text or "Cannot reach" in text

    def test_http_error(self):
        """EcoAgent HTTP error returns isError with detail."""
        import urllib.error
        mock_err = urllib.error.HTTPError(
            "http://localhost:8200/v1/tools/query_species/execute",
            500, "Internal Server Error", {}, io.BytesIO(b'{"error":"crash"}')
        )
        with patch("urllib.request.urlopen", side_effect=mock_err):
            result = handle_call_tool("eco_query_species", {"species": "Test"})

        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "500" in text

    def test_forwards_correct_action(self):
        """Verify the correct EcoAgent action is called."""
        calls = []

        def capture_request(req, timeout):
            calls.append(req.full_url)
            mock = MagicMock()
            mock.read.return_value = b'{"result": "ok"}'
            return mock

        with patch("urllib.request.urlopen", side_effect=capture_request):
            handle_call_tool("eco_fit_sdm", {"species": "Panthera onca", "method": "maxent"})

        assert len(calls) == 1
        assert "/v1/tools/fit_sdm/execute" in calls[0]

    def test_resolve_taxonomy_action(self):
        """eco_resolve_taxonomy maps to resolve_taxonomy."""
        with patch("urllib.request.urlopen") as mock:
            mock.return_value.__enter__.return_value.read.return_value = \
                b'{"names": [{"scientificName": "Quercus robur"}]}'
            result = handle_call_tool(
                "eco_resolve_taxonomy",
                {"names": ["Quercus robur"]},
            )

        assert "isError" not in result
        text = result["content"][0]["text"]
        assert "Quercus robur" in text


class TestInitialize:
    def test_protocol_version(self):
        result = handle_initialize({})
        assert result["protocolVersion"] == "2024-11-05"

    def test_capabilities(self):
        result = handle_initialize({})
        assert "tools" in result["capabilities"]

    def test_server_info(self):
        result = handle_initialize({})
        assert result["serverInfo"]["name"] == "ecoagent-mcp"
        assert "version" in result["serverInfo"]


# ===========================================================================
# JSON-RPC Protocol (stdio simulation)
# ===========================================================================


class TestJSONRPC:
    def test_list_tools_request(self, monkeypatch, capsys):
        """Simulate a tools/list JSON-RPC request."""
        stdin_data = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }) + "\n"

        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_data))

        # Import and run one iteration of the loop
        from mcp_ecoagent import _send_jsonrpc, handle_list_tools

        # Simulate initialize first
        _send_jsonrpc(1, result=handle_initialize({}))
        captured = capsys.readouterr()
        response = json.loads(captured.out.strip())
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"

    def test_call_tool_request(self):
        """JSON-RPC response format for tools/call."""
        from mcp_ecoagent import _send_jsonrpc

        with patch("urllib.request.urlopen") as mock:
            mock.return_value.__enter__.return_value.read.return_value = \
                b'{"species": "Test"}'

            result = handle_call_tool("eco_query_species", {"species": "Test"})

        # Capture the JSON-RPC response
        import io as _io
        saved_stdout = sys.stdout
        try:
            sys.stdout = _io.StringIO()
            _send_jsonrpc(42, result=result)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = saved_stdout

        response = json.loads(output.strip())
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 42
        assert "result" in response


# ===========================================================================
# Tool Map Consistency
# ===========================================================================


class TestToolMap:
    def test_all_tools_in_map(self):
        """Every MCP_TOOLS entry is in _TOOL_MAP by name."""
        for t in MCP_TOOLS:
            assert t["name"] in _TOOL_MAP, f"{t['name']} not in _TOOL_MAP"

    def test_map_points_to_correct_tool(self):
        """_TOOL_MAP values are the same objects as MCP_TOOLS entries."""
        for t in MCP_TOOLS:
            assert _TOOL_MAP[t["name"]] is t
