#!/usr/bin/env bash
# Create the isolated venv this experiment runs in and install everything
# it needs. Run from the repository root:
#   bash langgraph_experiment/setup_venv.sh
#
# See README.md's "Dependency isolation" section for why this is a
# separate venv from the project's main .venv rather than an extra on the
# main pyproject.toml.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m venv .venv-langgraph
source .venv-langgraph/Scripts/activate 2>/dev/null || source .venv-langgraph/bin/activate

python -m pip install -q -e . --no-deps

# The project's real runtime dependencies, minus the langchain group
# (langchain/langchain-community/langchain-text-splitters) and minus
# beautifulsoup4/lxml/pypdf/pdfplumber/python-docx (ingestion-only
# loaders; this experiment never ingests) -- see langgraph_experiment/
# __init__.py's module docstring for exactly which rag modules that
# keeps importable.
python -m pip install -q \
    psycopg2-binary pgvector sentence-transformers fastapi pydantic pyyaml \
    ollama python-dotenv rank-bm25 pyjwt prometheus-client \
    opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
    "mcp>=2.1,<3" httpx2

python -m pip install -q -r langgraph_experiment/requirements.txt
python -m pip install -q pytest

echo "Done. Activate with: source .venv-langgraph/Scripts/activate  (or .venv-langgraph/bin/activate on macOS/Linux)"
