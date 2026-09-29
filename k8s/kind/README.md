# kind: cluster-specific steps

This is the `kind`-specific half of the walkthrough. Read `../README.md`
first for the shared mental model, prerequisites, and every step that
works identically regardless of which local cluster you use
(applying manifests, initializing the database, ingesting sample
content, reaching the app, observability, HPA, NetworkPolicy, secrets,
debugging, teardown concepts). This file only covers what's genuinely
different about `kind`: cluster creation, image loading, and
`kind`-specific ingress/teardown commands.

**Known issue on this host:** `kind create cluster` currently hangs
indefinitely inside `kubeadm init` on this specific machine (a
`containerd`/`runc` `CreateContainer` hang tied to this host's WSL2
kernel build, not a manifest or config problem). See `ISSUES.md`'s "kind
(and Docker Desktop's own built-in Kubernetes) cannot bring up a control
plane on this host, but minikube's docker driver can" entry for the full
diagnosis and evidence. This `kind` setup is kept as the reference
3-node topology (some exercises, like killing a worker node, genuinely
need a second node minikube's single-node setup can't provide) and is
not modified by that finding -- if you're on a different host where
`kind` works, or a future fix lands, everything below should work as
documented. If you're on this exact host, use `../minikube/README.md`
instead.

---

## Local architecture

```text
Windows host
|
|-- Browser / curl
|
|-- Ollama (native process on Windows)
|
`-- Docker Desktop
    |
    `-- kind cluster: rag-learning
        |
        |-- control-plane node
        |-- worker node 1
        |-- worker node 2
        |
        `-- namespace: rag
            |
            |-- frontend Pod(s)
            |     `-- nginx
            |
            |-- rag-api Pod(s)
            |     `-- FastAPI
            |
            |-- postgres-0
            |     `-- PostgreSQL + PVC
            |
            |-- optional Prometheus
            |-- optional Grafana
            `-- optional Jaeger
```

In `kind`, each Kubernetes **node** is itself a Docker container. That is
only a local-learning implementation detail. In production, nodes are
usually Linux VMs or physical machines.

---

## Prerequisites

- Docker Desktop
- `kubectl`
- `kind`
- Ollama running natively on the host, with the model the active RAG
  config expects

```bash
docker version
kubectl version --client
kind version
ollama list
```

---

## 1. Create the cluster

```bash
kind create cluster --config k8s/kind/kind-cluster.yaml
kubectl cluster-info --context kind-rag-learning
kubectl get nodes -o wide
```

Expected shape:

```text
rag-learning-control-plane
rag-learning-worker
rag-learning-worker2
```

The same kind nodes are also visible from Docker's point of view:

```bash
docker ps --filter "name=rag-learning"
```

This is useful for understanding the layers:

```text
docker ps
    -> kind node containers

kubectl get nodes
    -> the same objects as Kubernetes nodes
```

---

## 2. Build and load the application images

Build the images exactly as Docker normally would:

```bash
docker build -t local-rag-api:kind .
docker build -t local-rag-frontend:kind ./frontend \
  --build-arg VITE_UI_MODE=developer
```

Load them into the kind nodes:

```bash
kind load docker-image local-rag-api:kind --name rag-learning
kind load docker-image local-rag-frontend:kind --name rag-learning
```

Why this is necessary:

```text
Docker Desktop image store
        !=
kind node containerd image store
```

`kind load docker-image` copies the locally built image into the kind
nodes. The manifests use `imagePullPolicy: IfNotPresent`, so Kubernetes
can use the image already loaded into the node instead of trying to
fetch it from Docker Hub.

In a real organization this step normally looks different:

```text
CI pipeline
    -> build image
    -> scan/test image
    -> push to Harbor/ECR/ACR/GCR/etc.
    -> Kubernetes pulls immutable version/digest
```

---

## 3. Apply the base application

`k8s/base`'s defaults (`host.docker.internal`, image tag `:kind`) already
match kind's conventions, so kind applies the base manifests directly,
with no overlay needed:

```bash
kubectl apply -k k8s/base
kubectl get pods -n rag -w
```

For everything after this (initializing the database, ingesting sample
content, reaching the app via port-forward, observability, HPA,
NetworkPolicy, secrets, debugging commands), see `../README.md` -- those
steps are identical regardless of which cluster you're using.

---

## 4. Ingress on kind

Install the kind-compatible ingress-nginx controller:

```bash
kubectl apply -f \
  https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

Add:

```text
127.0.0.1 rag.local
```

to the host's hosts file. Then:

```bash
curl -H "Host: rag.local" http://localhost:8080/health
```

or open `http://rag.local:8080`.

---

## 5. metrics-server on kind

```bash
kubectl apply -f \
  https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

kind's kubelet certificate setup usually needs:

```bash
kubectl patch deployment metrics-server -n kube-system --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

```bash
kubectl wait \
  --for=condition=available \
  deployment/metrics-server \
  -n kube-system \
  --timeout=90s
```

---

## 6. Teardown

```bash
kind delete cluster --name rag-learning
```

This destroys the local cluster and its local storage.

---

Hands-on drills: `EXERCISES.md` in this directory (the full 18-exercise
set, including the ones that genuinely need kind's 3-node topology).
