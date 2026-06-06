"""EcoAgent MCP Server — exposes 25 ecological analysis tools via MCP stdio.

Implements the Model Context Protocol (MCP) over stdio JSON-RPC, exposing
each EcoAgent action as an individually discoverable MCP tool with its own
schema, description, and parameter contract.

Usage:
  Standalone:
    python mcp_ecoagent.py                          # stdio mode (default)
    python mcp_ecoagent.py --transport http --port 8201  # HTTP mode

  With Hermes:
    hermes mcp add ecoagent --command "python3 plugins/ecoseek/mcp_ecoagent.py"

  With Claude Desktop:
    {
      "mcpServers": {
        "ecoagent": {
          "command": "python3",
          "args": ["plugins/ecoseek/mcp_ecoagent.py"]
        }
      }
    }

Env vars:
  ECOAGENT_URL  — EcoAgent backend URL (default: http://localhost:8200)
  ECOAGENT_TIMEOUT — Request timeout in seconds (default: 120)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger("mcp-ecoagent")
logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ECOAGENT_URL = os.environ.get("ECOAGENT_URL", "http://localhost:8200").rstrip("/")
TIMEOUT = int(os.environ.get("ECOAGENT_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# Tool Definitions — one per EcoAgent action, with full MCP schemas
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "eco_query_species",
        "description": (
            "Query GBIF for species occurrence data. Returns georeferenced "
            "occurrence records with coordinates, dates, and metadata. "
            "Use this to get where a species has been observed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "string",
                    "description": "Scientific name of the species (e.g., 'Quercus robur', 'Panthera onca').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of records to return (default: 100).",
                    "default": 100,
                },
                "has_coordinate": {
                    "type": "boolean",
                    "description": "Only return records with coordinates (default: true).",
                    "default": True,
                },
            },
            "required": ["species"],
        },
        "ecoagent_action": "query_species",
    },
    {
        "name": "eco_query_gbif_literature",
        "description": (
            "Search GBIF's literature database for papers that reference "
            "specific taxa, datasets, or publishers. Returns bibliographic "
            "metadata and links to source publications."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term (scientific name, dataset key, publisher).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 20).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
        "ecoagent_action": "query_gbif_literature",
    },
    {
        "name": "eco_query_gbif_parquet",
        "description": (
            "Fast columnar queries on GBIF data using Parquet format. "
            "Much faster than the standard API for large queries. "
            "Filter by taxon, country, year range, or geometry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "string",
                    "description": "Scientific name to query.",
                },
                "country": {
                    "type": "string",
                    "description": "ISO country code (e.g., 'MX', 'BR', 'CO').",
                },
                "year_from": {
                    "type": "integer",
                    "description": "Start year for temporal filter.",
                },
                "year_to": {
                    "type": "integer",
                    "description": "End year for temporal filter.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum records (default: 1000).",
                    "default": 1000,
                },
            },
            "required": ["species"],
        },
        "ecoagent_action": "query_gbif_parquet",
    },
    {
        "name": "eco_fit_sdm",
        "description": (
            "Fit a Species Distribution Model (SDM) to predict habitat suitability "
            "based on occurrence data and environmental variables. "
            "Supports multiple algorithms: maxent, random forest, glm, bioclim."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "string",
                    "description": "Scientific name of the target species.",
                },
                "method": {
                    "type": "string",
                    "enum": ["maxent", "random_forest", "glm", "bioclim"],
                    "description": "SDM algorithm to use (default: maxent).",
                    "default": "maxent",
                },
                "predictors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Bioclimatic variables to use (e.g., ['bio1', 'bio12', 'bio7']). "
                        "If empty, uses a standard set of 19 WorldClim variables."
                    ),
                },
                "projection_scenario": {
                    "type": "string",
                    "description": (
                        "Future climate scenario for projection "
                        "(e.g., 'ssp245_2050', 'ssp585_2070'). Leave empty for current only."
                    ),
                },
            },
            "required": ["species"],
        },
        "ecoagent_action": "fit_sdm",
    },
    {
        "name": "eco_fit_maxent",
        "description": (
            "Fit a MaxEnt (Maximum Entropy) species distribution model. "
            "Industry standard for presence-only SDM. Returns habitat "
            "suitability maps, variable importance, and model diagnostics."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "string",
                    "description": "Scientific name.",
                },
                "background_points": {
                    "type": "integer",
                    "description": "Number of background/pseudo-absence points (default: 10000).",
                    "default": 10000,
                },
                "features": {
                    "type": "string",
                    "description": "Feature classes: 'l', 'lq', 'lqh', 'lqhp', 'lqphpt' (default: 'lqphpt').",
                    "default": "lqphpt",
                },
            },
            "required": ["species"],
        },
        "ecoagent_action": "fit_maxent",
    },
    {
        "name": "eco_compute_bioclim",
        "description": (
            "Extract bioclimatic variables (WorldClim BIO1-BIO19) for a given "
            "location. Returns annual mean temperature, precipitation, seasonality, "
            "and extreme indices."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "lon": {
                    "type": "number",
                    "description": "Longitude (decimal degrees, WGS84).",
                },
                "lat": {
                    "type": "number",
                    "description": "Latitude (decimal degrees, WGS84).",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["30s", "2.5m", "5m", "10m"],
                    "description": "Spatial resolution (default: '2.5m' = 2.5 arc-minutes).",
                    "default": "2.5m",
                },
            },
            "required": ["lon", "lat"],
        },
        "ecoagent_action": "compute_bioclim",
    },
    {
        "name": "eco_evaluate_niche",
        "description": (
            "Evaluate ecological niche characteristics for a species. "
            "Computes niche breadth, overlap, marginality, and specialization "
            "metrics from occurrence and environmental data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "string",
                    "description": "Scientific name.",
                },
                "method": {
                    "type": "string",
                    "enum": ["ecospat", "density", "mahalanobis"],
                    "description": "Niche evaluation method (default: ecospat).",
                    "default": "ecospat",
                },
            },
            "required": ["species"],
        },
        "ecoagent_action": "evaluate_niche",
    },
    {
        "name": "eco_compute_diversity",
        "description": (
            "Compute biodiversity indices from a community matrix. "
            "Supports Shannon, Simpson, inverse Simpson, Chao1, ACE, "
            "phylogenetic diversity (PD), and functional diversity (FD)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "community_data": {
                    "type": "string",
                    "description": (
                        "Community matrix as CSV string or JSON array of arrays. "
                        "Rows = sites, columns = species, values = abundances. "
                        "Example CSV: 'site,sp1,sp2,sp3\\nA,5,0,2\\nB,1,10,0'"
                    ),
                },
                "index": {
                    "type": "string",
                    "enum": ["shannon", "simpson", "inv_simpson", "chao1", "ace", "pd", "fd"],
                    "description": "Diversity index to compute (default: shannon).",
                    "default": "shannon",
                },
            },
            "required": ["community_data"],
        },
        "ecoagent_action": "compute_diversity",
    },
    {
        "name": "eco_compute_clusters",
        "description": (
            "Cluster sites or species based on community composition. "
            "Returns dendrograms, cluster assignments, and ordination plots. "
            "Useful for identifying biogeographic regions or species assemblages."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "community_data": {
                    "type": "string",
                    "description": "Community matrix as CSV string or JSON.",
                },
                "method": {
                    "type": "string",
                    "enum": ["ward", "complete", "average", "kmeans", "nmds"],
                    "description": "Clustering method (default: ward).",
                    "default": "ward",
                },
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters for k-means (ignored for hierarchical methods).",
                },
            },
            "required": ["community_data"],
        },
        "ecoagent_action": "compute_clusters",
    },
    {
        "name": "eco_compute_ecological_distance",
        "description": (
            "Compute ecological (environmental) distance between two or more "
            "locations based on bioclimatic variables. Returns Euclidean, "
            "Mahalanobis, and niche overlap metrics."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "lon": {"type": "number"},
                            "lat": {"type": "number"},
                            "label": {"type": "string"},
                        },
                        "required": ["lon", "lat"],
                    },
                    "description": "List of points to compare (min 2).",
                },
                "metric": {
                    "type": "string",
                    "enum": ["euclidean", "mahalanobis", "schoener_d", "hellinger"],
                    "description": "Distance/overlap metric (default: euclidean).",
                    "default": "euclidean",
                },
            },
            "required": ["points"],
        },
        "ecoagent_action": "compute_ecological_distance",
    },
    {
        "name": "eco_compute_effort_bias",
        "description": (
            "Compute spatial sampling effort bias for GBIF occurrence data. "
            "Identifies over-sampled and under-sampled regions to guide "
            "future field surveys. Critical for robust SDM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "taxon_key": {
                    "type": "string",
                    "description": "GBIF taxon key for the group (e.g., class Aves = 212).",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["0.5", "1", "2"],
                    "description": "Grid resolution in degrees (default: '1').",
                    "default": "1",
                },
            },
            "required": ["taxon_key"],
        },
        "ecoagent_action": "compute_effort_bias",
    },
    {
        "name": "eco_resolve_taxonomy",
        "description": (
            "Resolve and validate taxonomic names against GBIF's backbone "
            "taxonomy. Returns accepted names, synonyms, higher classification, "
            "and taxonomic keys. Handles misspellings and obsolete names."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of scientific names to resolve.",
                },
                "strict": {
                    "type": "boolean",
                    "description": "If true, only return exact matches (default: false).",
                    "default": False,
                },
            },
            "required": ["names"],
        },
        "ecoagent_action": "resolve_taxonomy",
    },
    {
        "name": "eco_resolve_worms",
        "description": (
            "Resolve marine species names against the World Register of "
            "Marine Species (WoRMS). Returns accepted names, authority, "
            "classification, and aphia IDs for marine taxa."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of marine species names to resolve.",
                },
                "marine_only": {
                    "type": "boolean",
                    "description": "Only return marine taxa (default: true).",
                    "default": True,
                },
            },
            "required": ["names"],
        },
        "ecoagent_action": "resolve_worms",
    },
    {
        "name": "eco_search_literature",
        "description": (
            "Semantic vector search over indexed ecological literature. "
            "Find papers by meaning, not just keywords. Returns ranked "
            "results with abstracts, authors, and relevance scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language query describing the research topic "
                        "(e.g., 'effects of climate change on alpine plant phenology')."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 10).",
                    "default": 10,
                },
                "year_from": {
                    "type": "integer",
                    "description": "Filter papers published after this year.",
                },
            },
            "required": ["query"],
        },
        "ecoagent_action": "search_literature",
    },
    {
        "name": "eco_query_papers",
        "description": (
            "Keyword search in paper abstracts database. Faster than semantic "
            "search for specific terms. Returns papers matching exact keywords."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords to search for in abstracts.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 20).",
                    "default": 20,
                },
            },
            "required": ["keywords"],
        },
        "ecoagent_action": "query_papers",
    },
    {
        "name": "eco_classify_abstract",
        "description": (
            "Classify a scientific abstract into ecological research domains "
            "using a trained classifier. Returns domain labels with confidence "
            "scores (e.g., 'climate_change_ecology', 'conservation_biology')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "abstract": {
                    "type": "string",
                    "description": "The scientific abstract text to classify.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top domain labels to return (default: 3).",
                    "default": 3,
                },
            },
            "required": ["abstract"],
        },
        "ecoagent_action": "classify_abstract",
    },
    {
        "name": "eco_query_pubtator",
        "description": (
            "Query PubTator, NCBI's biomedical text mining tool, for ecological "
            "entities. Extracts species mentions, genes, diseases, and chemicals "
            "from biomedical literature. Useful for disease ecology."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term (species name, disease, gene).",
                },
                "entity_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["species", "gene", "disease", "chemical"],
                    },
                    "description": "Entity types to extract (default: all).",
                },
            },
            "required": ["query"],
        },
        "ecoagent_action": "query_pubtator",
    },
    {
        "name": "eco_query_cofid",
        "description": (
            "Query the COFID database for host-parasite interaction records. "
            "Search by host, parasite, location, or interaction type. "
            "Returns verified interaction data with geographic context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host species scientific name.",
                },
                "parasite": {
                    "type": "string",
                    "description": "Parasite species scientific name.",
                },
                "interaction_type": {
                    "type": "string",
                    "enum": ["parasitism", "commensalism", "mutualism", "predation"],
                    "description": "Type of ecological interaction.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum records (default: 100).",
                    "default": 100,
                },
            },
            "required": [],
        },
        "ecoagent_action": "query_cofid",
    },
    {
        "name": "eco_validate_cofid",
        "description": (
            "Validate a host-parasite interaction record in COFID. "
            "Checks taxonomic consistency, geographic plausibility, and "
            "literature support for the claimed interaction."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host species.",
                },
                "parasite": {
                    "type": "string",
                    "description": "Parasite species.",
                },
                "location": {
                    "type": "string",
                    "description": "Geographic location of the observation.",
                },
            },
            "required": ["host", "parasite"],
        },
        "ecoagent_action": "validate_cofid",
    },
    {
        "name": "eco_predict_susceptibility",
        "description": (
            "Predict disease susceptibility for a species based on ecological "
            "and phylogenetic traits. Uses machine learning models trained on "
            "known host-pathogen associations. Returns risk scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "string",
                    "description": "Scientific name of the species to assess.",
                },
                "disease": {
                    "type": "string",
                    "description": (
                        "Disease or pathogen to assess susceptibility for "
                        "(e.g., 'Batrachochytrium dendrobatidis', 'SARS-CoV-2')."
                    ),
                },
            },
            "required": ["species", "disease"],
        },
        "ecoagent_action": "predict_susceptibility",
    },
    {
        "name": "eco_extract_triplets",
        "description": (
            "Extract ecological relationship triplets (subject-predicate-object) "
            "from scientific text using NLP. Identifies interactions like "
            "'Panthera onca PREYS_ON Tapirus terrestris' from literature."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Scientific text to extract triplets from.",
                },
                "focus_taxon": {
                    "type": "string",
                    "description": "Optional: only extract triplets involving this taxon.",
                },
            },
            "required": ["text"],
        },
        "ecoagent_action": "extract_triplets",
    },
    {
        "name": "eco_build_knowledge_graph",
        "description": (
            "Build an ecological knowledge graph from extracted triplets. "
            "Nodes = species/locations/interactions, edges = ecological "
            "relationships. Returns graph in JSON-LD or NetworkX format."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "triplets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                        },
                    },
                    "description": "Array of subject-predicate-object triplets.",
                },
                "format": {
                    "type": "string",
                    "enum": ["json-ld", "networkx", "cytoscape"],
                    "description": "Output format (default: json-ld).",
                    "default": "json-ld",
                },
            },
            "required": ["triplets"],
        },
        "ecoagent_action": "build_knowledge_graph",
    },
    {
        "name": "eco_query_graph_hosts",
        "description": (
            "Query the ecological knowledge graph for hosts of a given parasite "
            "or pathogen. Returns all known host species with interaction metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "parasite": {
                    "type": "string",
                    "description": "Scientific name of the parasite/pathogen.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum host species to return (default: 50).",
                    "default": 50,
                },
            },
            "required": ["parasite"],
        },
        "ecoagent_action": "query_graph_hosts",
    },
    {
        "name": "eco_query_graph_parasites",
        "description": (
            "Query the ecological knowledge graph for parasites of a given host "
            "species. Returns all known parasites/pathogens with interaction data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Scientific name of the host species.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum parasite species to return (default: 50).",
                    "default": 50,
                },
            },
            "required": ["host"],
        },
        "ecoagent_action": "query_graph_parasites",
    },
    {
        "name": "eco_fit_geotax",
        "description": (
            "Fit a geo-taxonomic model that jointly estimates species distributions "
            "and taxonomic relationships. Combines phylogeny with spatial ecology."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "species_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of related species to model jointly.",
                },
                "phylogeny": {
                    "type": "string",
                    "description": "Newick-format phylogenetic tree for the species.",
                },
            },
            "required": ["species_list"],
        },
        "ecoagent_action": "fit_geotax",
    },
]

# Build lookup: tool_name → ecoagent_action
_TOOL_MAP = {t["name"]: t for t in MCP_TOOLS}

# ---------------------------------------------------------------------------
# MCP Protocol Handlers
# ---------------------------------------------------------------------------


def handle_list_tools() -> list[dict]:
    """MCP tools/list — return all available tools with schemas."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "inputSchema": t["inputSchema"],
        }
        for t in MCP_TOOLS
    ]


def handle_call_tool(tool_name: str, arguments: dict) -> dict:
    """MCP tools/call — execute a tool by forwarding to EcoAgent backend."""
    if tool_name not in _TOOL_MAP:
        return {
            "content": [{"type": "text", "text": json.dumps({
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(_TOOL_MAP.keys()),
            })}],
            "isError": True,
        }

    tool_def = _TOOL_MAP[tool_name]
    action = tool_def["ecoagent_action"]

    # Forward to EcoAgent backend
    body = json.dumps(arguments).encode("utf-8")
    req = urllib.request.Request(
        f"{ECOAGENT_URL}/v1/tools/{action}/execute",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        logger.info("mcp-ecoagent: %s → %s OK", tool_name, action)
        return {
            "content": [{"type": "text", "text": json.dumps(data, indent=2)}],
        }

    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.warning("mcp-ecoagent: %s → %s HTTP %s", tool_name, action, exc.code)
        return {
            "content": [{"type": "text", "text": json.dumps({
                "error": f"EcoAgent returned HTTP {exc.code}",
                "detail": error_body,
            })}],
            "isError": True,
        }

    except urllib.error.URLError as exc:
        logger.warning("mcp-ecoagent: %s → %s connection error: %s", tool_name, action, exc.reason)
        return {
            "content": [{"type": "text", "text": json.dumps({
                "error": f"Cannot reach EcoAgent at {ECOAGENT_URL}: {exc.reason}",
            })}],
            "isError": True,
        }

    except Exception as exc:
        logger.exception("mcp-ecoagent: %s → %s unexpected error", tool_name, action)
        return {
            "content": [{"type": "text", "text": json.dumps({
                "error": str(exc)[:300],
            })}],
            "isError": True,
        }


def handle_initialize(params: dict) -> dict:
    """MCP initialize — server handshake."""
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {},
        },
        "serverInfo": {
            "name": "ecoagent-mcp",
            "version": "0.1.0",
        },
    }


# ---------------------------------------------------------------------------
# JSON-RPC Loop (stdio transport)
# ---------------------------------------------------------------------------


def _send_jsonrpc(response_id, result=None, error=None):
    """Write a JSON-RPC response to stdout."""
    msg = {"jsonrpc": "2.0", "id": response_id}
    if error:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def run_stdio():
    """Main MCP stdio loop. Reads JSON-RPC requests from stdin."""
    logger.info("EcoAgent MCP server starting (stdio transport)")
    server_name = "ecoagent-mcp v0.1.0"
    sys.stderr.write(f"[{server_name}] Ready. Waiting for MCP requests...\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON on stdin: %s", line[:200])
            continue

        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "initialize":
            result = handle_initialize(params)
            _send_jsonrpc(req_id, result=result)

        elif method == "notifications/initialized":
            # No response needed for notifications
            pass

        elif method == "tools/list":
            result = {"tools": handle_list_tools()}
            _send_jsonrpc(req_id, result=result)

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = handle_call_tool(tool_name, arguments)
            _send_jsonrpc(req_id, result=result)

        elif method == "ping":
            _send_jsonrpc(req_id, result={})

        else:
            _send_jsonrpc(req_id, error={
                "code": -32601,
                "message": f"Method not found: {method}",
            })


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EcoAgent MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8201,
        help="HTTP port (default: 8201)",
    )
    args = parser.parse_args()

    if args.transport == "http":
        # Minimal HTTP transport for testing / non-stdio clients
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class MCPHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                request = json.loads(body)
                req_id = request.get("id")
                method = request.get("method", "")
                params = request.get("params", {})

                if method == "initialize":
                    result = handle_initialize(params)
                elif method == "tools/list":
                    result = {"tools": handle_list_tools()}
                elif method == "tools/call":
                    result = handle_call_tool(
                        params.get("name", ""),
                        params.get("arguments", {}),
                    )
                else:
                    result = None

                response = {"jsonrpc": "2.0", "id": req_id, "result": result}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            def log_message(self, format, *args):
                logger.info("HTTP %s", format % args)

        server = HTTPServer(("127.0.0.1", args.port), MCPHandler)
        sys.stderr.write(
            f"[ecoagent-mcp] HTTP server on http://127.0.0.1:{args.port}\n"
        )
        sys.stderr.flush()
        server.serve_forever()
    else:
        run_stdio()
