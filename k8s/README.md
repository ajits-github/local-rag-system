# Kubernetes Deployment Guide

This directory runs the existing RAG/Agentic RAG application on a local
Kubernetes cluster.

The goal is not to turn this repository into a full production platform.
The goal is to learn the Kubernetes concepts that matter in real
systems:

- Deployments and ReplicaSets
- StatefulSets and persistent storage
- Services, ClusterIP, DNS, and Ingress
- readiness and liveness probes
- rolling updates and rollback
- HPA and resource requests/limits
- Jobs and init containers
- NetworkPolicy
- observability
- failure recovery
- multi-node scheduling

The application itself is unchanged. Kubernetes is simply another way to
run the same container images that are already used by Docker Compose.

For the "why", read `LEARNING.md` (short reference) or `STUDY_GUIDE.md`
(the same material taught in full, from zero).

---

## 0. Choose your cluster: kind or minikube

Two local cluster options are documented here, sharing the same
application manifests (`k8s/base/`):

- **`kind/`**: the original, reference environment. 3 nodes
  (control-plane + 2 workers), which some exercises genuinely need (node
  failure/failover). Start with `kind/README.md`.
- **`minikube/`**: a working alternative for hosts where `kind` (and
  Docker Desktop's own built-in Kubernetes) can't bring up a control
  plane at all. Single node. Start with `minikube/README.md`.

If you don't already know which one you need: try `kind` first (it's the
richer environment). If `kind create cluster` hangs indefinitely at
"Starting control-plane" on your machine, that's a known, diagnosed
issue on at least one host in this project's history. See
`ISSUES.md`'s "kind (and Docker Desktop's own built-in Kubernetes)
cannot bring up a control plane on this host, but minikube's docker
driver can" entry before assuming it's something you misconfigured, and
switch to `minikube/README.md`.

Everything below this section is shared: it works identically once you
have a cluster up and `kubectl` pointed at it, regardless of which one
you chose.

---

## 1. Mental model before you start

Keep these layers separate:

```text
Dockerfile
    |
    | docker build
    v
Container image
    |
    +--------------------------+
    |                          |
    v                          v
Docker Compose             Kubernetes
(local multi-container)    (cluster orchestration)
                               |
                               v
                              Pod
                               |
                               v
                           container
```

Kubernetes does **not** execute your Dockerfile or your Compose files. It
runs already-built container images.

This repo therefore has two valid runtime styles:

```text
docker compose ...
```

for local/container development, and:

```text
kubectl apply ...
```

for the Kubernetes learning environment.

---

## 2. What is in `k8s/`

```text
k8s/
  base/
      Core application, shared by both clusters:
      - Namespace
      - ConfigMap
      - Secret
      - PostgreSQL StatefulSet
      - rag-api Deployment + Service
      - frontend Deployment + Service
      - Ingress
      - HPA
      - NetworkPolicy
      - kustomization.yaml

  kind/
      kind-cluster.yaml   : 3-node cluster definition
      README.md           : kind-specific setup steps
      EXERCISES.md         : the full 18-exercise set (reference)

  minikube/
      kustomization.yaml + two patches : overlay on top of k8s/base for
                                           the handful of things that
                                           genuinely differ under
                                           minikube's docker driver (host
                                           reachability, image tag,
                                           rag-api's startup probe budget)
      README.md           : minikube-specific setup steps
      EXERCISES.md         : delta from kind/EXERCISES.md (17 of 18
                             exercises are identical either way)

  jobs/
      One-shot database initialization Job.

  observability/
      Prometheus, Grafana, and Jaeger resources.

  README.md (this file)
      Shared setup and runbook.

  LEARNING.md
      Concept reference and interview preparation.

  STUDY_GUIDE.md (+ STUDY_GUIDE.html)
      The same material taught in full, from zero: a longer-form
      companion to README.md/EXERCISES.md/LEARNING.md above, for
      reading away from the keyboard.
```

---

## 3. Prerequisites

Common to both clusters:

- Docker Desktop
- `kubectl`
- Ollama running natively on the host, with the model the active RAG
  config expects
- optional: an HTTP load tool such as `hey`, `k6`, or a curl loop

Plus `kind` or `minikube` depending on which you're using. See that
tool's own README under `k8s/kind/` or `k8s/minikube/` for the exact
prerequisite check and cluster-creation steps.

---

## 4. Apply the application

Once your cluster is up (`kind/README.md` or `minikube/README.md`) and
images are built and loaded, applying the application itself differs
only in which kustomize target you use:

```bash
# kind: base's defaults already match kind's conventions
kubectl apply -k k8s/base

# minikube: a small overlay on top of base (see k8s/minikube/kustomization.yaml)
kubectl apply -k k8s/minikube
```

```bash
kubectl get pods -n rag -w
```

Useful checks:

```bash
kubectl get all -n rag
kubectl get svc -n rag
kubectl get pvc -n rag
kubectl get ingress -n rag
kubectl get hpa -n rag
```

Wait until the main Pods are running and ready.

---

## 5. Initialize the database

Create a ConfigMap containing the real schema-init script:

```bash
kubectl create configmap init-db-script -n rag \
  --from-file=scripts/init_db.py \
  --dry-run=client -o yaml | kubectl apply -f -
```

Run the one-shot Job:

```bash
kubectl apply -f k8s/jobs/init-db-job.yaml
kubectl wait --for=condition=complete job/init-db -n rag --timeout=60s
kubectl logs job/init-db -n rag
```

Why a `Job`?

```text
Deployment
    -> should keep running forever

Job
    -> should run to completion and stop
```

Database initialization belongs to the second category.

---

## 6. Ingest sample content

The helper scripts are not baked into the image, but the Python
ingestion package is available.

Copy the sample documents into one API Pod:

```bash
POD=$(kubectl get pod -n rag -l app=rag-api \
  -o jsonpath='{.items[0].metadata.name}')

kubectl cp data/sample_docs "rag/$POD:/tmp/sample_docs"
```

Run ingestion:

```bash
kubectl exec -n rag "$POD" -- \
  python -m rag.ingestion.pipeline \
  /tmp/sample_docs \
  --dataset-id sample_docs
```

Important production note: files copied or uploaded into a Pod's local
filesystem are ephemeral. If the Pod is replaced, those local files
disappear. A real replicated ingestion service would normally use shared
durable storage or object storage instead.

---

## 7. Reach the application

### Option A: port-forward

Fastest for local testing, and identical on both clusters:

```bash
kubectl port-forward -n rag svc/rag-api 8000:8000
```

Then:

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the deployment windows for production releases?",
    "filters": {"dataset_id": "sample_docs"}
  }'
```

This bypasses Ingress. The path is roughly:

```text
localhost:8000
    -> kubectl port-forward
    -> rag-api Service
    -> one ready rag-api Pod
```

### Option B: Ingress

The mechanics differ by cluster (kind uses a manual ingress-nginx
manifest; minikube uses an addon). See "Ingress on kind" in
`kind/README.md` or "Ingress on minikube" in `minikube/README.md`.

The browser path, once Ingress is up, is the same on both:

```text
Browser
  -> ingress-nginx
  -> frontend Service
  -> frontend Pod / nginx
```

For API calls, nginx performs another internal hop:

```text
frontend nginx
  -> rag-api Service
  -> one ready rag-api Pod
```

---

## 8. Service and ClusterIP: the most important networking distinction

A Kubernetes `Service` is **not** your FastAPI service. It is a
Kubernetes networking object.

For example:

```text
Service: rag-api
ClusterIP: 10.96.x.x
```

might currently point to:

```text
rag-api Pod A -> 10.244.1.12
rag-api Pod B -> 10.244.2.7
rag-api Pod C -> 10.244.2.9
```

The Pod IPs can change. The Service name and ClusterIP remain stable.

Inside the namespace, another Pod can call:

```text
http://rag-api:8000
```

CoreDNS resolves `rag-api` to the Service ClusterIP. Kubernetes Service
networking then selects one ready backend Pod.

`ClusterIP` is therefore:

> the stable internal virtual IP of one Kubernetes Service.

It is **not** the IP of the Kubernetes cluster itself.

---

## 9. Health probes: three endpoints, three different questions

The application (`src/rag/api/routers/health.py`) exposes three
separate endpoints, each answering one question, deliberately not one
endpoint reused three ways:

```text
/health
    Detailed diagnostics for a human/dashboard. Always HTTP 200;
    degradation is reported only in the body
    ({"status": "degraded", "dependencies": {...}}). Never wired to a
    Kubernetes probe.

/livez
    Is this process alive? No dependency checks at all. A Postgres or
    Ollama outage can never make this fail. Wired to startupProbe and
    livenessProbe: a failure here means the process itself is
    unresponsive, and restarting the container may actually help.

/readyz
    Can this Pod serve a normal request right now? Checks the same
    vectorstore/LLM reachability /health does, but returns a real
    non-2xx (503) when either is down. Wired to readinessProbe: a
    failure removes the Pod from Service traffic without restarting it.
```

Example during a database outage:

```text
/health  -> 200, body says "degraded"   (diagnostic only)
/livez   -> 200                          (process itself is fine)
/readyz  -> 503                          (Pod should stop receiving traffic)
```

**Note:** an earlier version of this deployment wired `readinessProbe`
directly at `/health`, on the reasoning that "degraded" in the body was
enough. It was not: Kubernetes' `httpGet` probes only ever look at the
status code, never the body, so a real Postgres outage never removed
the Pod from Service traffic at all (see `ISSUES.md`'s "readiness probe
can't see through /health" entry for the full diagnosis). The
three-endpoint design above is the fix, not a hypothetical. It is what's
actually wired into
`k8s/base/rag-api-deployment.yaml` today. `tests/unit/
test_health_endpoints.py` proves the endpoint semantics directly at the
application level; each cluster's `EXERCISES.md` (exercise 5) proves
the same thing end to end against a real cluster.

---

## 10. Optional observability stack

Create the existing Prometheus/Grafana configuration as ConfigMaps:

```bash
kubectl create configmap prometheus-config -n rag \
  --from-file=observability/prometheus/prometheus.yml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap grafana-datasources -n rag \
  --from-file=observability/grafana/provisioning/datasources/datasources.yml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap grafana-dashboard-provider -n rag \
  --from-file=observability/grafana/provisioning/dashboards/dashboards.yml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap grafana-dashboards -n rag \
  --from-file=observability/grafana/dashboards/agentic-rag-overview.json \
  --dry-run=client -o yaml | kubectl apply -f -
```

Apply the Kubernetes observability resources:

```bash
kubectl apply -f k8s/observability/
```

Port-forward the UIs:

```bash
kubectl port-forward -n rag svc/prometheus 9090:9090
kubectl port-forward -n rag svc/grafana 3000:3000
kubectl port-forward -n rag svc/jaeger 16686:16686
```

Then visit:

```text
Prometheus  http://localhost:9090
Grafana     http://localhost:3000
Jaeger      http://localhost:16686
```

If tracing is disabled in the active application config, enable it
through the repo's normal configuration mechanism, preferably an
experiment/production config selected through `RAG_CONFIG_PATH`, rather
than silently changing the shared default.

---

## 11. Install metrics-server for HPA

HPA needs metrics. The install differs slightly by cluster. See
"metrics-server on kind" / "metrics-server on minikube" in the
cluster-specific READMEs (minikube has a one-line addon; kind needs the
upstream manifest plus a TLS patch).

Verify either way:

```bash
kubectl top nodes
kubectl top pods -n rag
kubectl get hpa -n rag
```

Remember: CPU-based HPA is a useful learning example, but it may not be
the best production metric for a RAG workload whose latency depends
heavily on model inference and downstream calls.

---

## 12. Optional NetworkPolicy enforcement

The repo contains NetworkPolicy objects, but a policy object is only
useful if the cluster's networking implementation actually enforces it.

For this local setup, verify the CNI behavior rather than assuming that
"object exists" means "traffic is blocked."

Useful commands:

```bash
kubectl get networkpolicy -n rag
kubectl describe networkpolicy -n rag
```

For a policy-aware test cluster, use a CNI that supports NetworkPolicy,
such as Calico or Cilium.

The important interview lesson is:

> Kubernetes can accept a NetworkPolicy object even though actual packet
> enforcement is the job of the CNI/data plane.

---

## 13. Secrets in this learning cluster

The example Secret contains only local development values.

Do not treat this pattern as production secret management.

Base64 is encoding, not encryption.

```text
secret text
    -> base64
encoded text

Anyone with the encoded text can decode it.
```

In a real environment, important controls include:

- do not commit real secret values;
- restrict Secret access with RBAC;
- encrypt Kubernetes Secrets at rest;
- use a secret manager or controller where appropriate;
- rotate credentials;
- prefer workload identity over long-lived credentials where the
  platform supports it.

---

## 14. What would change in a real production deployment?

The Kubernetes concepts remain the same, but the surrounding platform
becomes much more serious.

Typical differences:

```text
Local kind/minikube               Real production
------------------------------   ---------------------------------------
kind/minikube node containers    VMs or physical servers
local image load                 private registry
1-3 replicas                     replica count based on SLO/load testing
local-path PVC                   cloud/on-prem storage backend
dummy Secret YAML                Vault/KMS/secret manager + RBAC
single laptop                    multi-node, often multi-zone
manual kubectl                   CI/CD or GitOps
CPU-only HPA demo                workload-specific scaling metrics
kind/minikube ingress             enterprise ingress/gateway/load balancer
learning NetworkPolicy           enforced CNI policies
local Postgres                   managed DB or HA DB/operator architecture
native host Ollama               hosted LLM or GPU model-serving platform
```

Cloud is **not mandatory** for Kubernetes. A production cluster can run
on:

- AWS/Azure/GCP
- OpenStack
- VMware/private cloud
- bare metal
- company data centers

---

## 15. First debugging commands to remember

If something does not work, start here:

```bash
kubectl get pods -n rag -o wide
kubectl describe pod <pod> -n rag
kubectl logs <pod> -n rag
kubectl get events -n rag --sort-by=.lastTimestamp
kubectl get svc -n rag
kubectl get endpointslices -n rag
```

If the Pod is not running, inspect:

```text
scheduling
image pull
init container
crash/restart
resource limits
```

If the Pod is running but not receiving traffic, inspect:

```text
readiness
Service selector
EndpointSlices
ports
Ingress
NetworkPolicy
```

That distinction is extremely useful in interviews.

---

## 16. k9s: a terminal UI for faster exploration

`kubectl` (section 15 above) is the ground truth and the thing to know
cold for an interview, but `k9s` is a live, keyboard-driven terminal UI
over the same API that makes iterating through pods/services/logs much
faster day to day. It reads the same kubeconfig `kubectl` already
uses, so no separate cluster setup is needed.

**Install**

```bash
# Windows (winget)
winget install derailed.k9s

# Windows (scoop)
scoop install k9s

# macOS
brew install k9s

# Manual (any OS): download the release for your platform from
# https://github.com/derailed/k9s/releases and put the binary on PATH.
```

**Launch**

```bash
k9s              # opens in the current kubeconfig context's default namespace
k9s -n rag       # jump straight into this project's rag namespace
```

**Core navigation**

| Key | Action |
|---|---|
| `:pods`, `:svc`, `:deploy`, `:ns`, `:hpa`, `:job`, ... | Jump to a resource type (command mode) |
| `/` | Filter the current list (fuzzy match on name) |
| `↑`/`↓` then `Enter` | Drill into the selected resource |
| `d` | Describe (same information as `kubectl describe`) |
| `l` | Stream logs; press again to toggle previous-container logs |
| `s` | Shell into a container |
| `y` | Show the resource's raw YAML |
| `ctrl-d` | Delete the selected resource (asks to confirm) |
| `Esc` | Back out one level |
| `:q` or `ctrl-c` | Quit |
| `?` | Full keybinding help |

**Where this is useful in this project**

- `:pods` then `/frontend`: jump straight to the frontend pod to watch
  restart counts live while working through the unresolved crash-loop
  follow-up (see `ISSUES.md`).
- `l` on a multi-container pod, then a number key to pick which
  container's logs to stream.
- `:deploy`, select `rag-api`, `ctrl-s` to scale replicas live. A
  hands-on way to watch the HPA fight (or not fight) a manual scale.
- The CPU/MEM columns on the `:pods` view make it easy to spot a Pod
  close to its resource limit without running `kubectl top` by hand
  repeatedly.

k9s never changes cluster state on its own beyond the action a key
explicitly triggers (delete, scale, edit); it is a faster window onto
the same objects `kubectl` already shows, not a different control plane.

---

## 17. Teardown

See "Teardown" in `kind/README.md` or `minikube/README.md` for the
exact command. Either destroys the local cluster and its local
storage.

---

## Recommended study order

1. Bring the cluster up once (kind or minikube, per section 0).
2. Trace one request from browser to `rag-api`.
3. Delete one API Pod and watch it come back.
4. Scale API replicas from 1 to 3.
5. Break readiness and inspect Service backends.
6. Perform a rolling restart.
7. deploy a bad image and roll back.
8. stop a worker node and observe recovery behavior (kind only. See
   `kind/EXERCISES.md` exercise 12).
9. inspect HPA and metrics-server.
10. only then move on to Helm, operators/controllers, distributed state,
    and Redis.

The hands-on drills are in `kind/EXERCISES.md` (the full reference set)
and `minikube/EXERCISES.md` (the same set, with one exercise's delta
noted).
