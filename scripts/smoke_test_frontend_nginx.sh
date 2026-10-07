#!/usr/bin/env bash
# Smoke test for the frontend nginx image's environment-portable resolver
# and worker-process configuration (see ISSUES.md's "The frontend nginx
# image silently no-op'd two of its own official opt-in features" entry).
#
# Proves, against a real running container, that:
#   1. nginx.conf.template's ${NGINX_LOCAL_RESOLVERS}/${RAG_API_UPSTREAM_HOST}
#      placeholders were actually substituted (not left literal, the
#      failure mode that made nginx refuse to start at all).
#   2. worker_processes was NOT left at the literal directive "auto" (the
#      autotune opt-in actually ran; the *value* it picks is
#      environment-dependent: a Kubernetes pod's 200m CPU limit
#      yields "1", an unconstrained Docker Compose container falls back to
#      the host's CPU count. This only asserts the placeholder was
#      resolved to *some* concrete integer, not a specific number).
#   3. A real proxied request (through nginx, not a raw pod-to-pod request)
#      reaches rag-api and returns a real response, not a resolver-related
#      502/timeout.
#
# Assumes `docker compose -f docker-compose.yml -f docker-compose.frontend.yml
# up -d` is already running (same convention as smoke_test_containers.sh).
#
# Usage:
#   scripts/smoke_test_frontend_nginx.sh
#   FRONTEND_URL=http://localhost:3001 scripts/smoke_test_frontend_nginx.sh
set -euo pipefail

FRONTEND_URL="${FRONTEND_URL:-http://localhost:3001}"
FRONTEND_CONTAINER="${FRONTEND_CONTAINER:-local-rag-frontend}"
FAILED=0

# On Windows Git Bash, an absolute Unix-style path argument (e.g.
# /etc/nginx/...) gets silently rewritten to a Windows path (e.g.
# C:/Program Files/Git/etc/nginx/...) before reaching `docker exec` --
# MSYS_NO_PATHCONV disables that rewrite. A no-op on Linux/macOS.
export MSYS_NO_PATHCONV=1

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; FAILED=1; }

echo "== Smoke test: frontend nginx resolver/worker-process config ($FRONTEND_CONTAINER) =="

# ---------------------------------------------------------------------------
# 1. The rendered config has no unexpanded envsubst placeholder.
# ---------------------------------------------------------------------------
echo
echo "[1/3] Checking the rendered nginx config for unexpanded template placeholders..."
RENDERED_CONF="$(docker exec "$FRONTEND_CONTAINER" cat /etc/nginx/conf.d/default.conf 2>/dev/null || echo "")"
if [ -z "$RENDERED_CONF" ]; then
    fail "could not read /etc/nginx/conf.d/default.conf from $FRONTEND_CONTAINER. Is it running?"
elif echo "$RENDERED_CONF" | grep -q '\${'; then
    fail "rendered config still contains an unexpanded \${...} placeholder:"
    echo "$RENDERED_CONF" | grep '\${'
else
    pass "resolver/upstream-host placeholders were substituted"
fi

RESOLVER_LINE="$(echo "$RENDERED_CONF" | grep -m1 '^resolver ' || echo "")"
echo "  resolver line: $RESOLVER_LINE"
if echo "$RESOLVER_LINE" | grep -qE '^resolver [0-9.]+ valid='; then
    pass "resolver resolved to a concrete address (whatever this container's own /etc/resolv.conf says)"
else
    fail "resolver line is not a concrete address: $RESOLVER_LINE"
fi

# ---------------------------------------------------------------------------
# 2. worker_processes was NOT left at the literal directive "auto".
# ---------------------------------------------------------------------------
echo
echo "[2/3] Checking worker_processes was autotuned (not left as the raw 'auto' directive)..."
WORKER_LINE="$(docker exec "$FRONTEND_CONTAINER" grep -m1 '^worker_processes' /etc/nginx/nginx.conf 2>/dev/null || echo "")"
echo "  worker_processes line: $WORKER_LINE"
if echo "$WORKER_LINE" | grep -qE '^worker_processes [0-9]+;$'; then
    pass "worker_processes was rewritten to a concrete integer by the cgroup-aware autotune script"
else
    fail "worker_processes was not autotuned (still 'auto', or the script didn't run): $WORKER_LINE"
fi

# ---------------------------------------------------------------------------
# 3. A real proxied request through nginx reaches rag-api.
# ---------------------------------------------------------------------------
echo
echo "[3/3] Checking a proxied request through nginx actually reaches rag-api..."
HEALTH_RESPONSE="$(curl -s -w '\n%{http_code}' "$FRONTEND_URL/health" 2>/dev/null || echo "")"
HTTP_CODE="$(echo "$HEALTH_RESPONSE" | tail -n1)"
BODY="$(echo "$HEALTH_RESPONSE" | sed '$d')"
echo "  $FRONTEND_URL/health -> HTTP $HTTP_CODE"
echo "  Response: $BODY"
if [ "$HTTP_CODE" = "200" ]; then
    pass "proxied /health reached rag-api and returned 200"
else
    fail "proxied /health did not return 200 (HTTP $HTTP_CODE). Check 'docker logs $FRONTEND_CONTAINER' for a resolver/connect error"
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
