#!/usr/bin/env bash
# Deploy the EcoSeek plugin to a Hermes installation.
#
# Usage (on reumanlab or any machine with Hermes installed):
#
#   # Option 1: Clone repo and run deploy script
#   git clone https://github.com/alrobles/hermes-agent.git /tmp/hermes-agent
#   bash /tmp/hermes-agent/plugins/ecoseek/deploy.sh
#
#   # Option 2: One-liner (clone + deploy + cleanup)
#   bash <(curl -sL https://raw.githubusercontent.com/alrobles/hermes-agent/main/plugins/ecoseek/deploy.sh)
#
# What it does:
#   1. Copies plugin files to ~/.hermes/plugins/ecoseek/
#   2. Enables the plugin via hermes config
#   3. Optionally copies beta-config.yaml for Beta executor mode
#
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGIN_DIR="${HERMES_HOME}/plugins/ecoseek"

# Find the source directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If running via curl pipe, we need to clone the repo first
if [ ! -f "${SCRIPT_DIR}/__init__.py" ]; then
    echo "[ecoseek] Cloning hermes-agent repo..."
    TMPDIR=$(mktemp -d)
    git clone --depth 1 https://github.com/alrobles/hermes-agent.git "${TMPDIR}/hermes-agent" 2>/dev/null
    SCRIPT_DIR="${TMPDIR}/hermes-agent/plugins/ecoseek"
    CLEANUP_TMP=1
fi

echo "[ecoseek] Deploying to ${PLUGIN_DIR}..."
mkdir -p "${PLUGIN_DIR}"

# Copy all plugin files
for f in __init__.py didal.py eco_analyze.py ku_hpc.py plugin.yaml; do
    if [ -f "${SCRIPT_DIR}/${f}" ]; then
        cp "${SCRIPT_DIR}/${f}" "${PLUGIN_DIR}/${f}"
        echo "  copied ${f}"
    else
        echo "  WARNING: ${f} not found in ${SCRIPT_DIR}"
    fi
done

# Enable the plugin in Hermes config
CONFIG_FILE="${HERMES_HOME}/config.yaml"
if command -v hermes &>/dev/null; then
    echo "[ecoseek] Enabling plugin..."
    hermes plugins enable ecoseek 2>/dev/null && echo "  enabled via hermes CLI" || echo "  hermes CLI enable failed, checking config manually..."
fi

# Check if config file exists and has plugins.enabled
if [ -f "${CONFIG_FILE}" ]; then
    if ! grep -q "ecoseek" "${CONFIG_FILE}" 2>/dev/null; then
        echo "[ecoseek] NOTE: Add 'ecoseek' to plugins.enabled in ${CONFIG_FILE}"
    fi
else
    echo "[ecoseek] NOTE: No config.yaml found. Run 'hermes plugins enable ecoseek' after setup."
fi

# Beta executor mode: optionally deploy the Beta personality config
echo ""
read -p "[ecoseek] Deploy Beta executor config? (for reumanlab only) [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "${SCRIPT_DIR}/beta-config.yaml" ]; then
        cp "${SCRIPT_DIR}/beta-config.yaml" "${HERMES_HOME}/config.yaml"
        echo "  beta-config.yaml → ${HERMES_HOME}/config.yaml"
        echo "  Beta executor personality is now active."
    fi
fi

# Verify
echo ""
echo "[ecoseek] Deployment complete!"
echo ""
echo "Files installed:"
ls -la "${PLUGIN_DIR}/" 2>/dev/null
echo ""
if command -v hermes &>/dev/null; then
    echo "Registered tools:"
    hermes tools 2>/dev/null | grep -E "eco_analyze|ku_hpc|escalate_remote|dialectical_exchange" || echo "  (run 'hermes tools' to verify)"
fi
echo ""
echo "Next steps:"
echo "  1. Add env vars to ~/.hermes/.env:"
echo "     HERMES_ECOSEEK_API_KEY=<your-api-key>"
echo "     ECOAGENT_URL=http://localhost:8200  (if EcoAgent is running)"
echo "  2. Deploy EcoAgent tool server (optional, enables eco_analyze):"
echo "     git clone https://github.com/alrobles/ecoagent.git /tmp/ecoagent"
echo "     pip install -e '/tmp/ecoagent[sdm]' --user"
echo "     ECOAGENT_PROFILE=full python -m ecoagent.tool_server --port 8200 &"
echo "  3. Restart gateway: hermes gateway run --replace"
echo "  4. Verify: HERMES_PLUGINS_DEBUG=1 hermes tools list 2>&1 | grep ecoseek"

# Cleanup temp dir if we cloned
if [ "${CLEANUP_TMP:-0}" = "1" ] && [ -n "${TMPDIR:-}" ]; then
    rm -rf "${TMPDIR}"
fi
