# minikube: cluster-specific steps

This is the `minikube`-specific half of the walkthrough. Read
`../README.md` first for the shared mental model, prerequisites, and
every step that works identically regardless of which local cluster you
use. This file only covers what's genuinely different about minikube:
cluster creation, image loading, ingress, and teardown -- plus the two
places its `docker` driver genuinely differs from kind at the manifest
level, both already handled by the `k8s/minikube` kustomize overlay (see
`kustomization.yaml` and `ollama-external-service-patch.yaml` in this
directory).

**Why this exists:** on this host, `kind create cluster` (and Docker
Desktop's own built-in Kubernetes) hangs indefinitely inside `kubeadm
init` -- a `containerd`/`runc` bug tied to this host's WSL2 kernel build,
documented in `ISSUES.md`. `minikube start --driver=docker` runs on the
identical kernel and Docker Desktop install but bundles a different
containerd build in its node image, and does not hit the hang -- verified
end to end in this session: every `kube-system` pod reached `Running`,
and a throwaway `nginx` deployment went through the full
`Pending -> Pulling -> Running` lifecycle and was reachable over its
`ClusterIP` service by DNS name from another pod. If `kind` works fine
on your machine, there's no reason to prefer this path -- use
`../kind/README.md` instead, since it keeps the reference 3-node
topology some exercises need.

**Single-node only.** This environment uses `minikube start` with no
`--nodes` flag, i.e. one node acting as both control-plane and worker.
Exercises that require a genuinely separate second node (killing a
worker node and observing scheduling/failover behavior) cannot be
demonstrated here -- see `../kind/EXERCISES.md` exercise 12 for that one,
which needs `kind`'s 3-node cluster. Everything else in this walkthrough
and in `EXERCISES.md` (in this directory) works the same way.

---

## Prerequisites

- Docker Desktop
- `kubectl`
- `minikube` (this was written against v1.39.0; `minikube version` to
  check yours)
- Ollama running natively on the host, with the model the active RAG
  config expects

```bash
docker version
kubectl version --client
minikube version
ollama list
```

---

## 1. Create the cluster

Use an explicit profile name (`rag-learning`), not minikube's default
`minikube` profile. This matters concretely on this host: a leftover
`minikube`-profile config from years of prior, unrelated minikube use
pinned an old Kubernetes version (`v1.27.4`) that current minikube
(v1.39+) refuses to start (`K8S_OLD_UNSUPPORTED`). A named profile
sidesteps that stale config entirely without touching or deleting it --
if you've never used minikube on this machine before, this still applies
as good practice (an explicit name is clearer than the implicit default
either way).

```bash
minikube start -p rag-learning --driver=docker
kubectl config use-context rag-learning
kubectl get nodes -o wide
```

Expected shape: one node, `Ready`, running `containerd`. CNI
initialization can take about a minute after `minikube start` reports
done -- if the node shows `NotReady` right away with `cni plugin not
initialized` in `kubectl describe node`, that's normal startup, not a
hang; give it a minute and check again.

The same node is also visible from Docker's point of view:

```bash
docker ps --filter "name=rag-learning"
```

---

## 2. Build and load the application images

Build the images exactly as Docker normally would:

```bash
docker build -t local-rag-api:minikube .
docker build -t local-rag-frontend:minikube ./frontend \
  --build-arg VITE_UI_MODE=developer
```

Load them into the minikube node (verified working this session with a
smaller test image; same mechanism applies to these):

```bash
minikube image load local-rag-api:minikube -p rag-learning
minikube image load local-rag-frontend:minikube -p rag-learning
```

Same reasoning as kind's equivalent step: Docker Desktop's own image
store and the minikube node's containerd image store are separate.
`minikube image load` copies the image across; `imagePullPolicy:
IfNotPresent` in the manifests then lets Kubernetes use what's already
on the node instead of trying to pull it.

---

## 3. Apply the application

Unlike kind, this goes through the `k8s/minikube` overlay, not
`k8s/base` directly -- it layers two small, minikube-specific patches on
top of the shared base manifests (see this directory's
`kustomization.yaml`):

```bash
kubectl apply -k k8s/minikube
kubectl get pods -n rag -w
```

For everything after this (initializing the database, ingesting sample
content, reaching the app via port-forward, observability, HPA,
NetworkPolicy, secrets, debugging commands), see `../README.md` -- those
steps are identical regardless of which cluster you're using.

---

## 4. Ingress on minikube

minikube's own addon is simpler than kind's manual manifest:

```bash
minikube addons enable ingress -p rag-learning
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

Reaching it depends on the driver. With `--driver=docker` on Windows,
`minikube tunnel -p rag-learning` (run in its own terminal, stays
attached) is the most reliable path -- it creates a routable IP for
`LoadBalancer`/`Ingress` resources. Add:

```text
127.0.0.1 rag.local
```

to the host's hosts file, then:

```bash
curl -H "Host: rag.local" http://localhost/health
```

Note this is port 80, not kind's `8080` -- `minikube tunnel` binds the
standard ports directly rather than going through a node-port mapping.

Alternatively, for quick access without setting up a hosts-file entry,
`minikube service` opens a direct URL to any Service (bypassing Ingress
entirely, similar in spirit to the port-forward option in the shared
README):

```bash
minikube service frontend -n rag -p rag-learning
```

---

## 5. metrics-server on minikube

An addon, no manual manifest or TLS patch needed:

```bash
minikube addons enable metrics-server -p rag-learning
kubectl wait \
  --for=condition=available \
  deployment/metrics-server \
  -n kube-system \
  --timeout=90s
```

---

## 6. Teardown

```bash
minikube delete -p rag-learning
```

This destroys the `rag-learning` cluster and its storage. It does not
touch the unrelated, stale `minikube` default profile mentioned in step
1 -- delete that separately (`minikube delete -p minikube`) only if you
know it's not needed for anything else.

---

Hands-on drills: `EXERCISES.md` in this directory.
