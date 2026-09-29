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
# The official image's own docker-entrypoint.d/15-local-resolvers.envsh
# (which populates NGINX_LOCAL_RESOLVERS from this container's own
# /etc/resolv.conf, read by nginx.conf.template below) is itself opt-in:
# its first line is `[ "${NGINX_ENTRYPOINT_LOCAL_RESOLVERS:-}" ] || return 0`,
# confirmed by reading that script directly. Without this, the script is a
# silent no-op and envsubst leaves the ${NGINX_LOCAL_RESOLVERS} placeholder
# in nginx.conf.template completely unexpanded, which nginx then refuses to
# start on ("host not found in resolver ${NGINX_LOCAL_RESOLVERS}") -- caught
# by a local `docker run` smoke test before this was ever pushed to a
# cluster.
ENV NGINX_ENTRYPOINT_LOCAL_RESOLVERS=1
# The bare Compose service name (docker-compose.frontend.yml's "rag-api")
# resolves fine under Docker Compose's embedded DNS; a Kubernetes Deployment
# overrides this to the fully-qualified in-cluster name, since nginx's own
# resolver skips search-domain expansion (see nginx.conf.template's comment).
ENV RAG_API_UPSTREAM_HOST=rag-api
# Also opt-in, like the resolver script above: the official image's
# docker-entrypoint.d/30-tune-worker-processes.sh rewrites the outer
# /etc/nginx/nginx.conf's "worker_processes auto;" to a value derived from
# this container's own cgroup CPU quota, but only when
# NGINX_ENTRYPOINT_WORKER_PROCESSES_AUTOTUNE is set -- confirmed by reading
# that script directly. Left unset, "auto" resolves against the HOST's CPU
# count (getconf _NPROCESSORS_ONLN), completely ignoring a container's own
# resources.limits.cpu -- this project's k8s Deployment requests 50m/limits
# 200m, but nginx still spawned 8-10 worker processes on this host, all
# contending for a fifth of one core. Confirmed via a real cgroup v2
# cpu.stat read from a running pod: nr_throttled 25 of 177 periods,
# ~1.3s of accumulated throttled_usec -- real CPU starvation, not a
# hypothetical, and the direct cause of intermittent liveness/readiness
# probe timeouts under host load (distinct from, and found after, the DNS
# resolver bug above). Harmless under Docker Compose, where no CPU limit is
# normally set: the script's own cgroup-quota read fails closed in that
# case and it falls back to the host CPU count, identical to today's
# behavior. See ISSUES.md.
ENV NGINX_ENTRYPOINT_WORKER_PROCESSES_AUTOTUNE=1
COPY --from=builder /app/dist /usr/share/nginx/html
# .template (not conf.d directly) so the official image's own
# docker-entrypoint.d/20-envsubst-on-templates.sh renders it at container
# start, substituting NGINX_LOCAL_RESOLVERS (see nginx.conf.template's own
# comment) -- required for the resolver directive to be correct under both
# Docker Compose and Kubernetes.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY docker-entrypoint.sh /docker-entrypoint.d/30-rag-api-base.sh
RUN chmod +x /docker-entrypoint.d/30-rag-api-base.sh

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -q -O - http://127.0.0.1/ >/dev/null || exit 1

EXPOSE 80
