# Multi-stage build: Node builds the static Vite bundle, nginx serves it
# and reverse-proxies API paths to rag-api. Same-origin routing means the backend
# never needs CORS middleware added to it (see frontend/README.md).
FROM node:20-alpine AS builder
WORKDIR /app
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
