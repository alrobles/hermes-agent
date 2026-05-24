# EcoSeek Plugin for Hermes Agent

Remote escalation plugin that lets a local Emily agent delegate heavy tasks to
the remote Hermes instance on reumanlab.

## Setup

```bash
hermes plugins enable ecoseek
```

Add to `~/.hermes/.env`:
```
ECOSEEK_BROKER_URL=https://broker.ecoseek.org
ECOSEEK_BROKER_KEY=<your-session-key>
```

## Tool: `escalate_remote`

Sends a task to the remote Hermes agent via the AgenticPlug broker.

**When to use:**
- Heavy computation (HPC cluster, GPU jobs)
- Access to reumanlab resources (ku-hpc, deployed services)
- Advanced reasoning beyond your local model
- Large dataset processing (GBIF downloads, spatial analysis)
- Running ecological pipelines (SDMs, phylogenetics)

**When NOT to use:**
- Simple Q&A about ecology or species
- Generating short code snippets (R/Python)
- Explaining concepts or reviewing manuscripts
- Quick calculations or unit conversions

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ECOSEEK_BROKER_URL` | `https://broker.ecoseek.org` | Broker endpoint URL |
| `ECOSEEK_BROKER_KEY` | *(required)* | Session key for authentication |
| `ECOSEEK_MODEL` | `openclaw/main` | Model name on the remote Hermes |
| `ECOSEEK_TIMEOUT` | `300` | Request timeout in seconds |

## Architecture

```
Emily Local (Hermes) → escalate_remote tool
  → POST broker.ecoseek.org/v1/chat/completions
    → Hermes Remote (reumanlab, DeepSeek v4 Pro)
      → ku-hpc → KU HPC cluster (A100/MI210)
```
