#!/bin/sh
# Runs automatically at container start (nginx:alpine sources every
# executable script under /docker-entrypoint.d/). Rewrites the API base URL
# baked into the served index.html at runtime, so the same built image works
# whether the frontend proxies through its own nginx (the default; leave
# RAG_API_BASE_URL unset) or should call a backend directly at some other
# origin (set RAG_API_BASE_URL, e.g. for a remote rag-api. That mode
# requires CORS on that backend, which is not enabled by default).
set -e

HTML_FILE=/usr/share/nginx/html/index.html
API_BASE_URL="${RAG_API_BASE_URL:-}"

if [ -f "$HTML_FILE" ]; then
    sed -i "s#window.__RAG_API_BASE__ ?? \"\"#window.__RAG_API_BASE__ ?? \"${API_BASE_URL}\"#g" "$HTML_FILE"
fi
