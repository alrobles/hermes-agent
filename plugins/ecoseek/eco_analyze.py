"""eco_analyze — High-level EcoAgent tool wrapper for Beta executor.

Provides a structured interface to the EcoAgent MCP server at localhost:8000.
Wraps common ecological analysis operations (species queries, SDM fitting,
diversity metrics, taxonomy resolution, etc.) into a single tool.

Available on reumanlab where EcoAgent is running. The check function verifies
the server is reachable before exposing the tool.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_ECOAGENT_URL = os.environ.get("ECOAGENT_URL", "http://localhost:8000").rstrip("/")
_TIMEOUT = int(os.environ.get("ECOAGENT_TIMEOUT", "120"))

SUPPORTED_ACTIONS = (
    "query_species",
    "query_papers",
    "compute_diversity",
    "fit_sdm",
    "fit_maxent",
    "evaluate_niche",
    "resolve_taxonomy",
    "query_cofid",
    "extract_triplets",
    "build_knowledge_graph",
    "query_gbif_parquet",
    "compute_bioclim",
    "compute_effort_bias",
    "classify_abstract",
    "predict_susceptibility",
    "compute_ecological_distance",
    "fit_geotax",
)


def check_ecoagent_available() -> bool:
    """Return True when EcoAgent MCP server is reachable."""
    try:
        req = urllib.request.Request(f"{_ECOAGENT_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def eco_analyze(
    action: str,
    params: dict | None = None,
    task_id: Optional[str] = None,
) -> str:
    """Execute an ecological analysis action via EcoAgent.

    Parameters
    ----------
    action : str
        The EcoAgent action to execute (e.g. query_species, fit_sdm).
    params : dict, optional
        Parameters for the action. Structure depends on the action.
    task_id : str, optional
        Hermes task ID for tracing.
    """
    if action not in SUPPORTED_ACTIONS:
        return json.dumps({
            "success": False,
            "error": "invalid_action",
            "message": f"Unknown action: {action!r}. Supported: {', '.join(SUPPORTED_ACTIONS)}",
        })

    body = json.dumps({
        "action": action,
        "params": params or {},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{_ECOAGENT_URL}/v1/tools/{action}",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        logger.info("eco_analyze: action=%s status=ok", action)
        return json.dumps({
            "success": True,
            "action": action,
            "result": data,
        })

    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.warning("eco_analyze HTTP %s on %s: %s", exc.code, action, error_body[:200])
        return json.dumps({
            "success": False,
            "error": f"http_{exc.code}",
            "action": action,
            "message": f"EcoAgent returned HTTP {exc.code}",
            "detail": error_body[:500],
        })

    except urllib.error.URLError as exc:
        logger.warning("eco_analyze URL error on %s: %s", action, exc.reason)
        return json.dumps({
            "success": False,
            "error": "connection_error",
            "action": action,
            "message": f"Cannot reach EcoAgent at {_ECOAGENT_URL}: {exc.reason}",
        })

    except Exception as exc:
        logger.exception("eco_analyze unexpected error on %s", action)
        return json.dumps({
            "success": False,
            "error": "unexpected_error",
            "action": action,
            "message": str(exc)[:300],
        })


ECO_ANALYZE_SCHEMA = {
    "name": "eco_analyze",
    "description": (
        "Execute ecological analysis via the EcoAgent server on reumanlab. "
        "Supports species queries (GBIF), SDM fitting (MaxEnt, bioclim), "
        "diversity metrics, taxonomy resolution, knowledge graph operations, "
        "and more. Use this for structured ecological computations rather "
        "than raw shell commands."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(SUPPORTED_ACTIONS),
                "description": (
                    "The ecological analysis action to run. "
                    "query_species: search GBIF for occurrence data. "
                    "fit_sdm/fit_maxent: fit species distribution models. "
                    "compute_diversity: calculate diversity indices. "
                    "resolve_taxonomy: validate and resolve taxonomic names. "
                    "query_gbif_parquet: fast columnar GBIF queries. "
                    "compute_bioclim: extract bioclimatic variables. "
                    "extract_triplets/build_knowledge_graph: ecological knowledge extraction."
                ),
            },
            "params": {
                "type": "object",
                "description": (
                    "Parameters for the action. Examples:\n"
                    "  query_species: {\"species\": \"Quercus robur\", \"limit\": 100}\n"
                    "  fit_sdm: {\"species\": \"Panthera onca\", \"method\": \"maxent\"}\n"
                    "  compute_diversity: {\"community_matrix\": [[1,2],[3,4]], \"index\": \"shannon\"}\n"
                    "  resolve_taxonomy: {\"names\": [\"Quercus robur\", \"Panthera onca\"]}\n"
                    "  compute_bioclim: {\"lon\": -95.5, \"lat\": 38.9}"
                ),
            },
        },
        "required": ["action"],
    },
}
