# rag-system Helm chart

Packages the workloads under `k8s/base/` (rag-api, frontend,
Postgres+pgvector, plus the ExternalName indirection to native-host Ollama)
as a configurable, releasable Helm chart. Same containers, same probes, same
security posture as `k8s/base` -- this chart is a *packaging* change, not an
application change. Read `../README.md` first for the full teaching
walkthrough (concepts, demos, interview questions); this file is the
chart-scoped reference: what's configurable, and the handful of decisions
that are specific to converting hand-written manifests into a chart.

## Quickstart

```bash
helm install rag . -n rag-helm --create-namespace
helm test rag -n rag-helm
```

See `../README.md`'s "Demos" section for the full install -> override ->
upgrade -> rollback walkthrough with real command output.

## Values reference

| Key | Default | What it controls |
|---|---|---|
| `ragApi.image.repository` / `.tag` | `local-rag-api` / `minikube` | rag-api image |
| `ragApi.replicaCount` | `1` | rag-api Deployment replicas |
| `ragApi.service.port` | `8000` | rag-api Service port |
| `ragApi.resources` | 250m/768Mi requests, 1/1536Mi limits | rag-api container resources |
| `ragApi.probes.*` | see values.yaml | startup/readiness/liveness probe timing |
| `ragApi.autoscaling.enabled` | `false` | HPA on/off |
| `ragApi.autoscaling.minReplicas` / `.maxReplicas` / `.targetCPUUtilizationPercentage` | `1` / `5` / `65` | HPA target |
| `ragApi.podDisruptionBudget.enabled` / `.minAvailable` | `true` / `1` | rag-api PDB |
| `ragApi.securityContext.readOnlyRootFilesystem` | `true` | root fs read-only (writable emptyDirs mounted at /tmp, /app/.cache, /app/data/uploads) |
| `ragApi.topologySpreadConstraints.enabled` | `true` | spread replicas across nodes |
| `frontend.image.repository` / `.tag` | `local-rag-frontend` / `minikube` | frontend image |
| `frontend.replicaCount` | `1` | frontend Deployment replicas |
| `frontend.service.port` | `80` | frontend Service port |
| `frontend.podDisruptionBudget.*` | `true` / `1` | frontend PDB |
| `postgres.image` | `pgvector/pgvector:pg16` | Postgres image |
| `postgres.persistence.enabled` / `.size` / `.storageClassName` | `true` / `1Gi` / `""` (cluster default) | pgdata PVC |
| `config.*` | see values.yaml | non-secret env (LOG_LEVEL, OLLAMA_BASE_URL, ...) |
| `secret.create` / `.existingSecret` | `true` / `""` | render an in-chart Secret, or point at one created out-of-band |
| `secret.postgresPassword` / `.jwtHs256Secret` / `.cohereApiKey` | dev-only placeholders | only used when `secret.create: true` |
| `ollama.externalName` | `host.docker.internal` | DNS target for the `ollama-host` ExternalName Service (`host.minikube.internal` on minikube) |
| `ingress.enabled` / `.host` / `.className` | `false` / `rag.local` / `nginx` | Ingress |
| `networkPolicy.enabled` | `true` | default-deny-ingress + explicit allow rules |
| `serviceAccount.create` | `true` | dedicated, no-token-mounted ServiceAccount |
| `rbac.create` | `false` | see values.yaml's comment -- nothing in this app calls the K8s API |
| `initDbJob.enabled` | `false` | run `scripts/init_db.py` as a post-install/post-upgrade Helm hook |
| `tests.enabled` | `true` | render the `helm test` connection check |

`values-dev.yaml` and `values-prod.yaml` are overlays (deltas only) layered
on top of `values.yaml` with `-f values.yaml -f values-<env>.yaml` -- see
each file's own header comment.

## Why object names are fixed, not release-prefixed

The typical `helm create` scaffold prefixes every object name with
`{{ .Release.Name }}-{{ .Chart.Name }}` (via `rag-system.fullname`) so
multiple releases can coexist in one namespace. This chart deliberately does
**not** do that for the four Service names (`rag-api`, `frontend`,
`postgres`, `ollama-host`) or the ConfigMap/workload names that key off
them, because several places outside this chart hardcode those exact DNS
names and changing them would be an actual application-behavior change,
which this migration was explicitly told not to make:

- `frontend/nginx.conf.template`'s `proxy_pass` targets literally build the
  upstream host from `RAG_API_UPSTREAM_HOST`, which this chart sets to
  `rag-api.<namespace>.svc.cluster.local` -- change the Service name and
  this breaks.
- The `wait-for-postgres` initContainers (`rag-api/deployment.yaml`,
  `jobs/init-db-job.yaml`) do a literal `nc -z postgres 5432`.
- `secret.yaml`'s `DATABASE_URL` and `ingress.yaml`'s backend both reference
  `postgres`/`frontend` by their fixed names.

`rag-system.fullname` is still used for the handful of objects that have no
such external reference (ConfigMap/Secret default names, ServiceAccount,
RBAC, the `helm test` Pod) -- multi-release-per-namespace was never a design
goal for this chart, but there's no reason to hardcode names that don't
need to be fixed. **Known limitation, stated plainly**: this chart cannot be
installed twice in the same namespace (a second release's `rag-api` Service
would collide with the first's). One release per namespace, same as
`k8s/base` itself.

## Why there's no `templates/namespace.yaml`

`k8s/base/namespace.yaml` creates the `rag` Namespace as one of the
manifests kustomize applies. This chart does not render a Namespace object
at all -- `helm install rag . -n rag-helm --create-namespace` creates it instead.
This is the standard Helm recommendation, for a concrete reason: `helm
uninstall` deletes every object *this chart* created, and a chart-owned
Namespace would be deleted right along with it -- taking down anything else
a cluster operator later put in that namespace that Helm doesn't know
about. Namespace lifecycle is deliberately left to the operator, not bundled
into the release.

## The `readOnlyRootFilesystem` writable-volume pattern

Both `ragApi` and `frontend` default to `securityContext.readOnlyRootFilesystem:
true`. Neither container's own code changed to support this -- the pattern
is entirely at the Kubernetes layer:

- **rag-api**: the Dockerfile's own `rag` user (uid/gid 999, confirmed
  against the built image) only ever writes to `$HOME` (`/app`, for
  Hugging Face/RAGAS model caches), `/app/data/uploads` (`POST /ingest`),
  and `/tmp`. Three `emptyDir` volumes mounted at exactly those paths cover
  every write the process makes; `/app/src` and `/app/config` stay on the
  genuinely read-only image layer.
- **frontend**: this one needs an extra step. The stock `nginx:1.27-alpine`
  entrypoint scripts (`envsubst`-on-templates, worker-process autotuning)
  and this project's own `docker-entrypoint.sh` (rewrites
  `window.__RAG_API_BASE__` into `index.html`) all write into
  `/usr/share/nginx/html` and `/etc/nginx` at container start -- both baked
  into the image layer. An `initContainer` running the *same* image (whose
  own filesystem is unrestricted, since the restriction is a Pod-spec
  property of the main container, not the image) copies both directories
  into two `emptyDir` volumes first; the main container mounts those
  volumes over the same paths. No entrypoint script needed to change --
  they run identically, just against writable volume-backed paths instead
  of the read-only image layer. `/var/cache/nginx`, `/var/run`, and `/tmp`
  get their own plain `emptyDir`s the same way.
- **postgres** deliberately does *not* get `readOnlyRootFilesystem: true`
  -- the official Postgres entrypoint chmods/initializes files a level or
  two above the `pgdata` mount during first-run setup, not only inside the
  mounted volume. Verified against a real init run, not assumed; see this
  chart's `templates/postgres/statefulset.yaml` comment. `/var/run/postgresql`
  and `/tmp` still get their own `emptyDir`s, since those are genuinely
  practical to isolate independently either way.

Every `emptyDir` above needs `podSecurityContext.fsGroup` set to the
container's own gid (999 for rag-api, 101 for frontend) -- a fresh
`emptyDir` is created root-owned by default, and a non-root process can't
write to it without `fsGroup` making it group-writable.

## What isn't templated, on purpose

Selector labels (`app: rag-api` / `app: frontend` / `app: postgres`) are
hardcoded, not built from `.Release.Name` -- `spec.selector` on a
Deployment/StatefulSet is immutable after creation, so templating it in
would only invite a broken `helm upgrade` the first time someone renamed a
release. Probe *paths* are values (`ragApi.probes.readiness.path`, etc.)
because the underlying endpoints (`/health` / `/readyz` / `/livez`) are a
real, documented three-endpoint design (see the root `CLAUDE.md`'s
"Observability" section) that's reasonable to point at a different path in
an unusual deployment; probe *logic* (which endpoint feeds which probe
type) is not a value, because getting that assignment wrong silently
breaks the safety properties the three-endpoint split exists for. Container
`command`/`args`, the app's own env var *names* (as opposed to their
*values*), and Kubernetes API versions are never values -- they're not
things a values override should be able to change without also changing
the template.

## Known gaps (flagged, not implemented)

- **No TLS.** `templates/ingress.yaml` has no `spec.tls` block and no
  cert-manager integration. `values-prod.yaml` flags this directly rather
  than pretending a `host: rag.example.com` value makes it production-ready.
- **In-chart Secret is dev-only.** `secret.create: true` (the `values.yaml`
  default) is fine for a local learning cluster; `values-prod.yaml` switches
  to `secret.existingSecret` and documents creating the real Secret
  out-of-band. This chart does not integrate with Vault/External Secrets/
  Sealed Secrets -- a real next step for an actual production chart, out of
  scope for this learning exercise.
- **No multi-namespace/multi-release support** -- see "Why object names are
  fixed" above.
- **`rbac.create: false` by default, with an empty rule set when enabled.**
  Nothing in this codebase calls the Kubernetes API today; the template
  exists to show the *shape* (a namespaced Role, never a ClusterRole), not
  to grant permissions speculatively.
