#!/usr/bin/env bash
# Smoke test for the containerized API: proves the rag-api container can
# reach Postgres+pgvector and native Ollama, then exercises a real
# ingest -> query round trip against them.
#
# Assumes `docker compose up -d` (or the prod-override form) is already
# running. Does not start/stop containers itself, so it's safe to point at
# either docker-compose.yml alone or the +docker-compose.prod.yml stack.
#
# Usage:
#   scripts/smoke_test_containers.sh
#   API_URL=http://localhost:8000 scripts/smoke_test_containers.sh
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
DATASET_ID="smoke_test_$(date +%s)"
SAMPLE_FILE="data/sample_docs/password-policy.md"
FAILED=0

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; FAILED=1; }

echo "== Smoke test: containerized rag-api against $API_URL =="

# ---------------------------------------------------------------------------
# 1. Wait for /health to respond at all (container may still be starting).
# ---------------------------------------------------------------------------
echo
echo "[1/4] Waiting for $API_URL/health to respond..."
READY=0
for _ in $(seq 1 30); do
    if curl -s -o /dev/null -w '%{http_code}' "$API_URL/health" 2>/dev/null | grep -q '^200$'; then
        READY=1
        break
    fi
    sleep 2
done
if [ "$READY" -ne 1 ]; then
    fail "API never responded on /health after 60s"
    echo
    echo "Smoke test aborted: $FAILED failure(s)."
    exit 1
fi
pass "API is responding"

# ---------------------------------------------------------------------------
# 2. /health dependency check: proves the container reaches Postgres AND
#    the native Windows Ollama process via host.docker.internal.
# ---------------------------------------------------------------------------
echo
echo "[2/4] Checking /health dependencies (Postgres + Ollama)..."
HEALTH_JSON="$(curl -s "$API_URL/health")"
echo "  Response: $HEALTH_JSON"

VECTORSTORE_OK="$(echo "$HEALTH_JSON" | python -c 'import json,sys; print(json.load(sys.stdin)["dependencies"]["vectorstore"])' 2>/dev/null || echo "parse_error")"
LLM_OK="$(echo "$HEALTH_JSON" | python -c 'import json,sys; print(json.load(sys.stdin)["dependencies"]["llm"])' 2>/dev/null || echo "parse_error")"

if [ "$VECTORSTORE_OK" = "ok" ]; then
    pass "rag-api container reaches Postgres+pgvector"
else
    fail "rag-api container cannot reach Postgres (vectorstore: $VECTORSTORE_OK)"
fi

if [ "$LLM_OK" = "ok" ]; then
    pass "rag-api container reaches native Ollama via host.docker.internal"
else
    fail "rag-api container cannot reach Ollama (llm: $LLM_OK) -- is Ollama running on the Windows host?"
fi

# ---------------------------------------------------------------------------
# 3. Ingest a sample document through the containerized API.
# ---------------------------------------------------------------------------
echo
echo "[3/4] Ingesting $SAMPLE_FILE under dataset_id=$DATASET_ID..."
if [ ! -f "$SAMPLE_FILE" ]; then
    fail "sample file $SAMPLE_FILE not found (run from repo root)"
else
    INGEST_JSON="$(curl -s -X POST "$API_URL/ingest" \
        -F "dataset_id=$DATASET_ID" \
        -F "files=@$SAMPLE_FILE")"
    echo "  Response: $INGEST_JSON"

    CHUNKS_WRITTEN="$(echo "$INGEST_JSON" | python -c 'import json,sys; d=json.load(sys.stdin); print(d[0]["chunks_written"] if d else 0)' 2>/dev/null || echo "0")"
    if [ "$CHUNKS_WRITTEN" -gt 0 ] 2>/dev/null; then
        pass "ingested document -> $CHUNKS_WRITTEN chunks written"
    else
        fail "ingestion returned no chunks written (response above)"
    fi
fi

# ---------------------------------------------------------------------------
# 4. Query the containerized API end-to-end (retrieval + Ollama generation).
# ---------------------------------------------------------------------------
echo
echo "[4/4] Querying dataset_id=$DATASET_ID..."
QUERY_JSON="$(curl -s -X POST "$API_URL/query" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"What is the password policy?\", \"filters\": {\"dataset_id\": \"$DATASET_ID\"}}")"
echo "  Response: $QUERY_JSON"

ANSWER="$(echo "$QUERY_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("answer",""))' 2>/dev/null || echo "")"
if [ -n "$ANSWER" ]; then
    pass "query returned a generated answer (retrieval -> reranker -> prompt builder -> Ollama all worked)"
else
    fail "query returned no answer (response above) -- check rag-api logs: docker compose logs rag-api"
fi

# ---------------------------------------------------------------------------
echo
if [ "$FAILED" -eq 0 ]; then
    echo "== All checks passed. =="
    exit 0
else
    echo "== $FAILED check group(s) failed. See FAIL lines above. =="
    exit 1
fi
