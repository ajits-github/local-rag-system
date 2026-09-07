# Multi-stage build: Node builds the static Vite bundle, nginx serves it
# and reverse-proxies API paths to rag-api. Same-origin routing means the backend
# never needs CORS middleware added to it (see frontend/README.md).
FROM node:20-alpine AS builder
WORKDIR /app
# Selects the built-in "developer" vs. "production" UI mode (see
# src/config/uiMode.ts and frontend/README.md's "UI modes" section).
# Defaults to "production" -- the safe fallback -- so a plain
# `docker build` with no --build-arg never ships developer tooling by
# accident; pass --build-arg VITE_UI_MODE=developer for a demo image.
ARG VITE_UI_MODE=production
ENV VITE_UI_MODE=$VITE_UI_MODE
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.sh /docker-entrypoint.d/30-rag-api-base.sh
RUN chmod +x /docker-entrypoint.d/30-rag-api-base.sh

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -q -O - http://127.0.0.1/ >/dev/null || exit 1

EXPOSE 80
