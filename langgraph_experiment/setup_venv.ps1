# Create the isolated venv this experiment runs in and install everything
# it needs. Run from the repository root:
#   powershell -File langgraph_experiment/setup_venv.ps1
#
# See README.md's "Dependency isolation" section for why this is a
# separate venv from the project's main .venv rather than an extra on the
# main pyproject.toml.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python -m venv .venv-langgraph
$py = ".\.venv-langgraph\Scripts\python.exe"

& $py -m pip install -q -e . --no-deps

& $py -m pip install -q `
    psycopg2-binary pgvector sentence-transformers fastapi pydantic pyyaml `
    ollama python-dotenv rank-bm25 pyjwt prometheus-client `
    opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http `
    "mcp>=2.1,<3" httpx2

& $py -m pip install -q -r langgraph_experiment/requirements.txt
& $py -m pip install -q pytest

Write-Host "Done. Activate with: .venv-langgraph\Scripts\Activate.ps1"
