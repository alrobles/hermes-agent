# EcoSeek Plugin for Hermes Agent

Dual-agent architecture plugin for ecological intelligence. Supports the DiDAL
(Dialectical Dual-Agent Loop) protocol where Alpha (Emily local) plans and
Beta (Hermes remote on reumanlab) executes and critiques.

## Setup

```bash
hermes plugins enable ecoseek
```

Add to `~/.hermes/.env`:
```
ECOSEEK_BROKER_URL=https://broker.ecoseek.org
ECOSEEK_BROKER_KEY=<your-session-key>
```

## Tools

### Alpha tools (Emily local → Beta remote)

#### `escalate_remote`
Simple one-shot delegation to the remote Hermes agent via the broker.

**When to use:** heavy computation, HPC jobs, large dataset processing,
advanced reasoning, ecological pipelines.

#### `dialectical_exchange`
DiDAL structured debate: Alpha proposes a plan, Beta executes and critiques,
Alpha refines, loop continues until consensus.

**When to use:** complex multi-step tasks where iterative refinement
improves the result — SDM pipelines, HPC workflows, code review.

### Beta tools (Hermes remote on reumanlab)

#### `eco_analyze`
Structured interface to the EcoAgent MCP server at `localhost:8000`.
Supports: query_species, fit_sdm, fit_maxent, compute_diversity,
resolve_taxonomy, query_gbif_parquet, compute_bioclim, and more.

**Availability:** only on reumanlab where EcoAgent is running.

#### `ku_hpc`
KU HPC cluster operations via Slurm. Submit batch jobs, check status,
cancel jobs, view accounting, and query partition/GPU info.

**Available GPUs:** A100 (80GB), MI210 (64GB), V100 (16GB), Quadro RTX 8000 (48GB).

**Availability:** only on reumanlab with Slurm access.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ECOSEEK_BROKER_URL` | `https://broker.ecoseek.org` | Broker endpoint URL |
| `ECOSEEK_BROKER_KEY` | *(required for Alpha)* | Session key for authentication |
| `ECOSEEK_MODEL` | `openclaw/main` | Model name on the remote Hermes |
| `ECOSEEK_TIMEOUT` | `300` | Request timeout in seconds |
| `DIDAL_MAX_TURNS` | `20` | Max dialectical dialogue turns |
| `ECOAGENT_URL` | `http://localhost:8000` | EcoAgent MCP server URL |
| `ECOAGENT_TIMEOUT` | `120` | EcoAgent request timeout |
| `KU_HPC_TIMEOUT` | `30` | Slurm command timeout |

## Architecture

```
┌─ User Machine ─────────────────────┐    ┌─ reumanlab ─────────────────────┐
│                                     │    │                                  │
│  Emily (Alpha)                      │    │  Hermes (Beta)                   │
│    ├─ escalate_remote ──────────────┼──→ │    ├─ eco_analyze → EcoAgent MCP │
│    ├─ dialectical_exchange ─────────┼──→ │    ├─ ku_hpc → Slurm → HPC      │
│    └─ Emily personality             │    │    └─ Beta executor personality  │
│                                     │    │                                  │
│  Frontend (React SPA)               │    │  EcoAgent (localhost:8000)       │
│    └─ localhost:4000                │    │  Broker (broker.ecoseek.org)     │
│                                     │    │  KU HPC (A100, MI210, V100)     │
└─────────────────────────────────────┘    └──────────────────────────────────┘
```
