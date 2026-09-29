# Kubernetes Study Guide

A from-zero Kubernetes study guide, taught through the real deployment
in this directory (`k8s/`) instead of toy examples. Every object below
is a file you already have. Every command is copy-pasteable against
your own `rag-learning` cluster.

This guide is the detailed study version. `README.md` (setup),
`EXERCISES.md` (failure-injection walkthroughs), and `LEARNING.md`
(short concept reference and interview questions) are quick-reference
versions of the same material. Use those when you are at the keyboard.

## How to use this

- **Part 1** covers the vocabulary and the one mental model that makes
  everything else click. Read it once, straight through.
- **Part 2** is the build log: thirteen steps, each one a real file in
  `k8s/`, each one answering "what is this, and why does it exist here
  specifically." This is the bulk of the guide.
- **Parts 3 to 5** are for closer to interview day: the failure-injection
  exercises, two real bugs the build surfaced, and interview questions
  with model answers.

## Part 1: Fundamentals

### 1. The problem Kubernetes actually solves

Start with what Docker and Docker Compose already do well.

A `Dockerfile` builds a container image. `docker-compose.yml` can start several
containers together on one machine, wire their environment variables and
networks, mount volumes, and even restart a crashed container with a restart
policy.

So Kubernetes was **not** invented simply because Docker cannot restart a
container.

The real problem appears when your application has to behave like a service
rather than a collection of containers on one machine:

- you want multiple machines so one machine is not a single point of failure;
- you want several copies of an API and a stable way to reach whichever copies
  are healthy right now;
- you want a failed workload recreated automatically, possibly on another node;
- you want rolling updates without taking every healthy copy down at once;
- you want replicas to increase or decrease with load;
- you want consistent service discovery even though Pod IPs constantly change;
- you want scheduling based on CPU, memory, GPUs, storage, affinity, and policy;
- you want one control plane continuously checking whether reality still
  matches the state you asked for.

That is the gap Kubernetes fills.

A useful way to remember the layers is:

```text
Dockerfile
    builds
Container image
    runs locally via Docker Compose
    OR
    runs in a Kubernetes Pod

Kubernetes then manages:
Pods -> replicas -> scheduling -> networking -> rollout -> recovery
across one or many nodes.
```

For this repo, Docker Compose is still useful for local development and
integration testing. Kubernetes is the separate orchestration layer used to
learn how the same application would be operated as a clustered service.

### 2. The one mental model that makes everything click

If you remember exactly one thing from this whole guide, make it this.
**Kubernetes is not a command runner. It is a continuous, closed-loop
controller.** You never tell it "start this container." You tell it
"this is the state I want to exist," and a background process, called a
*controller*, spends forever comparing that desired state against
reality and nudging reality toward it. This is called a **reconciliation
loop**, and the same pattern repeats for every object type in the
system.

```mermaid
flowchart LR
  A["You: kubectl apply -f rag-api-deployment.yaml"] --> B["API server writes\ndesired state to etcd"]
  B --> C["Deployment controller\n(a watch loop, running forever)"]
  C --> D{"desired replicas\n== actual running pods?"}
  D -- "no" --> E["create or delete Pods\n(via a ReplicaSet)"]
  E --> C
  D -- "yes" --> F["do nothing, keep watching"]
  F --> C
```

Every controller in Kubernetes runs this exact loop. Only "desired
state" and "how to fix a mismatch" change.

Once this model is clear, the common Kubernetes behaviors make sense.

- **"Why did a new pod appear when I deleted one?"** You never told
  Kubernetes to run 3 pods once. You told it "3 is the desired count,
  forever." Deleting one just created a mismatch the loop immediately
  corrected.
- **"Why does a rolling update not just replace the pod in place?"** A
  Pod's spec is mostly immutable. The controller's only tool for
  "change the code" is "delete the old, create a new one that matches
  the new desired spec," done gradually.
- **"Why didn't my NetworkPolicy do anything?"** The object was
  accepted and stored as desired state just fine. But the component
  actually responsible for reconciling that specific kind of state (the
  CNI plugin) doesn't implement it. Desired state with no controller
  behind it is just inert data. You'll hit a real example of this in
  Part 2.

Two supporting ideas go along with this.

**Declarative, not imperative.** A Compose file and a Kubernetes
manifest look similar at a glance, but Compose is closer to "a script
for what to start." Kubernetes YAML is "a description of an end state,"
checked and re-checked forever by many independent controllers, not run
once from top to bottom.

**Labels and selectors, not names.** A Deployment doesn't manage pods
by name, because it deletes and recreates them constantly, so names
keep changing. It manages them by **label**: every pod it creates gets
stamped `app: rag-api`, and its own `selector` field says "anything
wearing this label is mine." A Service does the same trick to decide
which pods it load-balances across. Stamp with a label, select by the
same label: that one convention is how almost every object in
Kubernetes points at another, with no database, no name registry, and
no fixed IP involved.

### 3. Every object type, in the order you'll meet them

Don't memorize this table. Read it once for the shape of the territory,
then come back to it as Part 2 uses each one for real.

| Object | One-line job | Analogy |
|---|---|---|
| `Namespace` | A named bucket that scopes object names, RBAC, and policy. | A folder that also enforces "no two files with the same name inside it." |
| `Pod` | The smallest deployable unit: one or more containers that always land on the same machine and share a network address. | A single shipping crate. You never move it partway; the whole crate goes or none of it does. |
| `ReplicaSet` | Keeps exactly N copies of a Pod template running. You almost never write one directly. | The dockworker whose only job is "count crates, add or remove until it's N." |
| `Deployment` | Manages ReplicaSets over time, so a spec change becomes a controlled, gradual replacement, with rollback history. | The shipping manifest revision log, swapping crate contents without ever leaving the dock empty. |
| `StatefulSet` | Like a Deployment, but each pod gets a stable name and its own permanently attached storage. | A numbered berth at the dock. Berth #0 always gets the same crane and warehouse slot, even if the ship in it is swapped. |
| `Service` | A stable virtual address that load-balances to whichever pods currently match a label, regardless of churn. | A phone number that always rings "whoever's on shift," never a specific person's desk line. |
| `Ingress` | HTTP(S) routing rules from outside the cluster to a Service, by hostname or path. | The port authority's gate, deciding which pier (Service) an incoming ship (request) gets sent to. |
| `ConfigMap` | Non-secret key/value config, injected into pods as env vars or files. | The manifest paperwork stapled to a crate, visible to anyone. |
| `Secret` | Same idea as ConfigMap, but for sensitive values, base64-encoded at rest. | The same paperwork, in a sealed (not locked, just sealed) envelope. |
| `PersistentVolumeClaim` | A request for durable disk that outlives any one pod. | A warehouse unit you lease. The ship (pod) using it can be swapped without emptying the unit. |
| `Job` | Run a pod to completion, then stop. Not a long-running service. | A one-off delivery run, not a standing shipping route. |
| `HorizontalPodAutoscaler` | Watches a metric and adjusts a Deployment's replica count automatically. | The harbor master calling in extra crews when ship traffic spikes. |
| `NetworkPolicy` | A firewall rule scoped to pod labels and namespaces, enforced by the CNI plugin, not the API server itself. | A checkpoint at a specific pier, only real if the port actually staffs it. |

Notice the pattern already forming. Almost every object is either "a
controller that keeps something running" (Deployment, StatefulSet,
ReplicaSet, Job, HPA), "a pointer or rule that connects things by
label" (Service, Ingress, NetworkPolicy), or "data injected into a pod"
(ConfigMap, Secret, PVC). That's the whole system: three buckets,
repeated with variations.

### 4. Reading a manifest

Every Kubernetes object, no matter its kind, is written in the same
YAML shape. Here's the smallest real one from your repo, annotated line
by line.

`k8s/base/frontend-service.yaml`
```yaml
apiVersion: v1          # which version of the Kubernetes API defines this kind
kind: Service           # what type of object this is
metadata:
  name: frontend         # its unique name within the namespace
  namespace: rag         # which bucket it lives in
  labels:
    app: frontend         # a tag other objects can select by
spec:                    # the actual "what do you want", unique per kind
  type: ClusterIP
  selector:
    app: frontend         # "load-balance to any pod wearing this label"
  ports:
    - name: http
      port: 80             # the Service's own port
      targetPort: 80        # the port the container actually listens on
```

That's it: four top-level fields, always: `apiVersion`, `kind`,
`metadata`, `spec` (a `Job` or Pod-based object also gets a `status`
field, but that one is written by Kubernetes, never by you; it's how a
controller reports what actually happened). The only real complexity is
that `spec`'s shape differs for every `kind`, because a Service's "what
do I want" (a selector and some ports) is a completely different
question from a Deployment's ("what pod template, how many copies, what
update strategy").

> **The trick that ties objects together.** Find `selector: { app:
> frontend }` above, then find every other object in
> `k8s/base/frontend-deployment.yaml` with `labels: { app: frontend }`
> in its pod template. That's the entire mechanism. There's no ID, no
> foreign key, no registry, just "this selector matches that label."
> Get a label wrong in one file and an object silently selects zero
> pods, or the wrong pods. This is the single most common real-world
> Kubernetes mistake.

### 5. `kubectl`: the core verbs

`kubectl` just talks to the API server over HTTP. Every command below
is really a REST call reading or writing an object's stored state.

| Command | What it actually does |
|---|---|
| `kubectl apply -f x.yaml` | Diffs your YAML against stored state and patches only what changed. The declarative one, safe to re-run. |
| `kubectl get pods -n rag` | Lists current objects and their high-level status. Add `-w` to watch changes live, `-o wide` for node and IP columns. |
| `kubectl describe pod X -n rag` | The single most useful debugging command: full spec, current status, and the event log (why a probe failed, why a pull failed, when it was scheduled). |
| `kubectl logs X -n rag` | stdout and stderr of a container. Add `--previous` to see the logs of a container that already crashed and got replaced. |
| `kubectl exec -it X -n rag -- sh` | A real shell inside a running container, for poking around, not for permanent fixes (anything you change here dies with the pod). |
| `kubectl port-forward svc/X 8000:8000 -n rag` | Tunnels a local port straight to a Service, bypassing Ingress entirely. The fastest way to hit an API during development. |
| `kubectl scale deployment X --replicas=3 -n rag` | Edits `spec.replicas` directly, the same field an HPA would edit automatically. |
| `kubectl rollout status / undo / restart` | Watch a rollout finish, revert to the previous ReplicaSet, or force a fresh rollout with no spec change at all. |
| `kubectl get events -n rag --sort-by=.lastTimestamp` | The cluster's own chronological diary: scheduling decisions, probe failures, image pulls, everything. |
| `kubectl top pods / nodes` | Live CPU and memory usage, but only if `metrics-server` is installed. It's not there by default. |

### 6. The hierarchy: cluster, node, Pod, container

These terms are easiest to understand from the outside in.

```text
Kubernetes cluster
|
+-- Control plane
|   +-- API server
|   +-- scheduler
|   +-- controller manager
|   +-- etcd
|
+-- Worker node A
|   +-- kubelet
|   +-- Pod 1
|   |   +-- container
|   +-- Pod 2
|       +-- container
|
+-- Worker node B
    +-- kubelet
    +-- Pod 3
        +-- container
```

A **node** is the machine Kubernetes can schedule Pods onto. In production it is
normally a VM or physical server. In `kind`, each node is simulated by a Docker
container on your laptop.

A **Pod** is the smallest unit Kubernetes schedules. A Pod can contain one or
more tightly coupled containers. Containers in the same Pod:

- always run on the same node;
- share the same Pod IP and network namespace;
- can call each other through `localhost`;
- must use different ports if they both listen on the same Pod IP.

A **replica** is another copy of the entire Pod template, not another container
inside the same Pod.

For example:

```text
Deployment: rag-api, replicas = 3

Pod A                    Pod B                    Pod C
10.244.1.8               10.244.2.5               10.244.2.9
+-- rag-api container    +-- rag-api container    +-- rag-api container
```

If the Pod template had an application container plus a sidecar, every replica
would contain both containers.

### 7. Service, ClusterIP, DNS, and the API server

Three different things are often mixed together.

**Kubernetes Service**
A Kubernetes `Service` is a networking object. It is not your FastAPI service.
For example, the `rag-api` Service gives the changing `rag-api` Pods one stable
network identity.

**ClusterIP**
`ClusterIP` is the default Service type, and the term is also used for the
virtual IP assigned to that Service.

Example:

```text
rag-api Service
ClusterIP: 10.96.20.50

Ready backends:
10.244.1.8:8000
10.244.2.5:8000
10.244.2.9:8000
```

`10.96.20.50` is **not the IP of the Kubernetes cluster**. One cluster can have
many Services, therefore many ClusterIPs.

**DNS**
DNS turns a stable name into an IP address. Inside Kubernetes, CoreDNS can
resolve:

```text
rag-api
    -> 10.96.20.50
```

DNS resolves the Service name because Services are stable. Pod IPs are
temporary and can change after a restart or rollout.

The request path is therefore:

```text
frontend Pod
    |
    | http://rag-api:8000
    v
CoreDNS
    |
    | rag-api -> Service ClusterIP
    v
10.96.20.50:8000
    |
    | Service networking selects a Ready backend
    v
one rag-api Pod
```

Modern Kubernetes represents the backend set using `EndpointSlice` objects.
A failed readiness probe removes that Pod from the usable Service backends.

**Kubernetes API server**
The API server is part of the control plane. `kubectl`, controllers, the
scheduler, and node kubelets all communicate with the cluster through it.

```text
kubectl -----------+
scheduler ---------+
controllers -------+--> kube-apiserver --> etcd
kubelet -----------+
```

When people say, "the kubelet stops reporting to the API server," they mean the
`kube-apiserver` belonging to that Kubernetes cluster.

## Part 2: The build, step by step

### Step 0: Inspect before building anything

Before writing any YAML, I read what your app already does, because a
Kubernetes manifest for an existing app is a *translation* exercise,
not a fresh design. I read, in order:

- `docker-compose.yml`, `.prod.yml`, `.observability.yml`,
  `.frontend.yml`: what containers exist, what env vars they need, what
  ports they publish, what depends on what.
- `Dockerfile`, `frontend/Dockerfile`: exactly what's inside each
  image, since a Kubernetes Pod runs the same image, unmodified.
- `frontend/nginx.conf`: the frontend hardcodes `proxy_pass
  http://rag-api:8000`. That single line drove a real design decision
  later: name the Kubernetes Service `rag-api` too, so nothing in the
  app needs to change.
- `src/rag/api/routers/health.py`: what does the health endpoint
  actually check, and what HTTP status does it return on failure? This
  later exposed a readiness-probe limitation; see the debugging note in
  Part 4.
- `config/default.yaml` and `config.py`'s `*_env_var` convention: every
  secret or host is named by an environment variable, never hardcoded.
  That convention maps almost one-to-one onto Kubernetes
  ConfigMaps and Secrets.

> **Why this order matters, as an interview answer.** "I inspected the
> existing runtime contract first" is a stronger answer than "I know
> Kubernetes objects." It's the difference between someone who can
> write YAML and someone who can actually migrate a real system without
> breaking it. Every decision in the rest of this guide traces back to
> something read in this step.

### Step 1: The cluster itself, Minikube now; kind kept as the 3-node reference

The repo started with a 3-node `kind` topology because it is ideal for
learning scheduling and node-failure behavior. On this Windows/WSL2 machine,
the Docker-backed kind path hit a runtime/kernel issue, so the working
hands-on environment is now a single-node Minikube cluster named
`rag-learning`. The kind files were deliberately kept rather than rewritten,
because they still document the intended multi-node topology.

`kind` stands for **Kubernetes IN Docker**. It is a tool for creating disposable
Kubernetes clusters on a developer machine.

In production, a Kubernetes node is normally a Linux VM or physical server.
`kind` replaces those real machines with Docker containers:

```text
Windows laptop
|
+-- Docker Desktop
    |
    +-- rag-learning-control-plane   <- Docker container acting as a K8s node
    +-- rag-learning-worker          <- Docker container acting as a K8s node
    +-- rag-learning-worker2         <- Docker container acting as a K8s node
```

Inside those node containers, Kubernetes runs normally and schedules your
application Pods.

That nesting is intentional:

```text
Docker Desktop
    -> kind node containers
        -> Kubernetes
            -> Pods
                -> application containers
```

This is a local-learning trick. Production does not normally put Kubernetes
nodes inside Docker containers.

`kind` and `minikube` are local cluster tools. EKS, AKS, and GKE are managed
Kubernetes offerings from AWS, Azure, and Google Cloud. The application
manifests are still Kubernetes manifests; what changes is the infrastructure
providing the cluster, networking, storage, identity, and load balancers.

| Option | Typical use |
|---|---|
| `kind` | local development, CI, disposable multi-node testing |
| minikube | local learning and development |
| EKS | managed Kubernetes on AWS |
| AKS | managed Kubernetes on Azure |
| GKE | managed Kubernetes on Google Cloud |
| self-managed Kubernetes | company data center, private cloud, bare metal, VMs |

|  | Minikube, current hands-on path | kind, retained reference |
|---|---|---|
| What runs today | One real local Kubernetes node, visible as `rag-learning` | Three node-containers when the host/runtime supports it |
| Best for | Pods, Services, probes, StatefulSet/PVC, rollouts, HPA, Ingress, failure drills | Scheduling across nodes, node failure, topology spreading |
| Application manifests | Mostly the same Kubernetes API objects; only environment-specific details belong in overlays/patches | Mostly the same Kubernetes API objects; only environment-specific details belong in overlays/patches |

```bash
minikube start -p rag-learning
kubectl config current-context
kubectl get nodes -o wide
```

The important production lesson is portability: the application manifests
should describe Kubernetes workloads, while environment-specific
assumptions such as image loading, host access, Ingress setup, and local
startup tolerances should stay isolated. That is why the kind setup
remains intact and Minikube-specific behavior lives separately.

The kind reference cluster still uses this cluster:

`k8s/kind-cluster.yaml`
```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: rag-learning
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - containerPort: 80
        hostPort: 8080
      - containerPort: 443
        hostPort: 8443
  - role: worker
  - role: worker
```

After starting the cluster, compare the two views:

```bash
docker ps
kubectl get nodes -o wide
```

`docker ps` shows the kind node **containers**. `kubectl get nodes` shows those
same objects from Kubernetes' point of view as **nodes**.

**Why three nodes?** It lets you see scheduling and node failure instead of
learning everything on one fake machine.

**Why the port mappings?** They expose ports from the kind control-plane node
back to your Windows host so a browser can reach the local Ingress controller.

**Why `ingress-ready=true`?** It is a label used by the kind-oriented
ingress-nginx setup to place the controller on the node whose ports are exposed
to the host.

### Step 2: A Namespace, before anything else

Everything below lives in one Namespace, `rag`. This is the first
object created, deliberately, because every other object's
`metadata.namespace: rag` field depends on it existing first. It's also
the boundary `NetworkPolicy` selectors and DNS search paths key off
later: a bare Service name like `rag-api` only resolves automatically
for another pod in the same namespace. A pod in a different namespace
needs the full `rag-api.rag.svc.cluster.local`.

### Step 3: ConfigMap and Secret, translating `*_env_var`

Your app already never hardcodes a host or credential. `config.py`
resolves everything through an environment variable named in YAML
(`connection_env_var: DATABASE_URL`, and so on). That convention
translates almost mechanically.

- **Non-secret values** (log level, which Ollama URL to use, the OTLP
  endpoint) become a `ConfigMap`, injected as env vars.
- **Sensitive values** (the Postgres password, the full `DATABASE_URL`
  which embeds it, the JWT signing secret) become a `Secret`. Same
  mechanism, base64-encoded at rest, and excluded from a plain `kubectl
  get` listing by default.

A container consumes either one the same way, as an environment
variable, through two different wiring styles you'll see throughout
`k8s/base/rag-api-deployment.yaml`.

```yaml
# style 1: dump every key in the ConfigMap in as an env var
envFrom:
  - configMapRef:
      name: rag-config

# style 2: pull one specific key from a Secret, name it explicitly
env:
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: rag-secret
        key: DATABASE_URL
```

> **Important, and a real interview detail.** A Kubernetes `Secret` is
> **encoded, not encrypted**. `kubectl get secret rag-secret -o yaml`
> shows base64 text that anyone can decode in one command. It's still
> worth using over a plain ConfigMap (RBAC can restrict who reads
> Secrets specifically, tooling treats them differently, and
> encryption-at-rest can be layered on at the cluster level), but
> "Secrets are secure" is the wrong claim. `README.md`'s "Secret
> hardening notes" section spells out what a real deployment adds on
> top: never commit rendered Secret YAML, enable etcd encryption at
> rest, and restrict RBAC read access.

### Step 4: Postgres, and why it's a StatefulSet

Your API replicas should ideally be stateless: any `rag-api` Pod should be able
to handle the next request because shared durable state lives elsewhere.

PostgreSQL is different. Its data must survive Pod replacement, and a database
replica may need a stable identity and its own storage.

That is the problem a StatefulSet is designed for.

A StatefulSet gives its Pods predictable identities:

```text
postgres-0
postgres-1
postgres-2
```

and, when `volumeClaimTemplates` is used, each Pod ordinal gets a stable PVC.

```text
postgres-0 -> its PVC
postgres-1 -> its PVC
postgres-2 -> its PVC
```

If `postgres-0` is recreated, Kubernetes can attach the same claim again
instead of treating the replacement as an unrelated stateless copy.

`k8s/base/postgres-statefulset.yaml` (trimmed)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: rag
spec:
  clusterIP: None
  selector:
    app: postgres
  ports:
    - port: 5432
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: rag
spec:
  serviceName: postgres
  replicas: 1
  volumeClaimTemplates:
    - metadata:
        name: pgdata
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
```

`clusterIP: None` creates a **headless Service**. Instead of giving clients one
shared virtual ClusterIP, DNS can expose the StatefulSet Pods with stable
network identities such as:

```text
postgres-0.postgres.rag.svc.cluster.local
```

For this repo there is one Postgres replica, so the StatefulSet is mostly about
expressing the correct stateful model and keeping persistent storage attached
across Pod replacement.

One important production distinction: using a StatefulSet does **not** magically
make PostgreSQL highly available. Real production might use a managed database
or a dedicated PostgreSQL operator/HA setup with replication, failover, backups,
and recovery procedures.

### Step 5: Reaching Ollama, which lives outside the cluster entirely

Your architecture deliberately keeps Ollama native on the host (Docker
Desktop has no GPU passthrough into a Linux container). Kubernetes
still needs a clean way for a pod to address it, and this is where
`Service` reveals it isn't really "one thing." It's a small family with
different `type` values.

| `type:` | What it points at |
|---|---|
| `ClusterIP` (default) | A set of pods, inside the cluster, via a virtual IP. |
| `NodePort` | The same, plus a fixed port opened on every node's real IP. |
| `LoadBalancer` | The same, plus a cloud provider provisions a real external load balancer. |
| `ExternalName` | Nothing inside the cluster at all. Pure DNS: "this name is actually a CNAME for that other hostname." |

```yaml
apiVersion: v1
kind: Service
metadata: { name: ollama-host, namespace: rag }
spec:
  type: ExternalName
  externalName: host.docker.internal
```

No selector, no pods, no ClusterIP. A pod resolving
`ollama-host.rag.svc.cluster.local` gets told, by CoreDNS, "that's
actually `host.docker.internal`, go resolve that instead." Docker
Desktop's own DNS then answers with the Windows host's real address.
The app's config just points `OLLAMA_BASE_URL` at this Service name; it
has no idea the target isn't a pod.

### Step 6: The rag-api Deployment, field by field

This is the file with the most individually defensible decisions in
it. Here is `k8s/base/rag-api-deployment.yaml`, field by field.

**`initContainers`: waiting for Postgres.** Compose has `depends_on:
condition: service_healthy`. Kubernetes has no direct equivalent; a
Deployment has no idea another workload even exists. The fix is an
**init container**: a container that must exit successfully before the
main container is even started.

```yaml
initContainers:
  - name: wait-for-postgres
    image: busybox:1.36
    command: ["sh","-c","until nc -z postgres 5432; do sleep 2; done"]
```

**startup, liveness, and readiness: three different questions.**

- `startupProbe`: has this slow-starting app finished starting?
- `livenessProbe`: should kubelet restart this container?
- `readinessProbe`: should this Pod receive normal traffic?

The first version used `/health` for both liveness and readiness. Hands-on
testing exposed two problems: the API can take a long time to cold-start on
this laptop, and `/health` returns HTTP 200 even when a dependency is
degraded. The implementation was therefore changed instead of merely hiding
the symptoms with a huge liveness delay.

```yaml
startupProbe:
  httpGet: { path: /livez, port: 8000 }

readinessProbe:
  httpGet: { path: /readyz, port: 8000 }
```

`/livez` deliberately avoids downstream dependency checks, so a temporary
Postgres or Ollama outage does not restart a healthy FastAPI process.
`/readyz` returns a real non-2xx response when a critical dependency is
unavailable, so the Pod is removed from normal Service traffic. The
Minikube overlay carries the unusually long startup tolerance needed by
this resource-constrained laptop; that host-specific value is not baked
into the common base.

> **Observed, not theoretical.** The original liveness probe actually
> restarted the API during a slow cold start. After adding `startupProbe`
> and separating `/livez` from `/readyz`, the behavior matched the
> intended lifecycle: slow startup is tolerated, dependency loss affects
> readiness, and liveness remains about the process itself.

**`resources`: requests vs. limits, not the same axis.**

```yaml
resources:
  requests: { cpu: "250m", memory: "768Mi" }
  limits:   { cpu: "1",    memory: "1536Mi" }
```

`requests` is what the **scheduler** uses to decide which node has room
for this pod, and what the **HPA** divides actual usage by to get a
utilization percentage. `limits` is a hard ceiling enforced by the
kernel. This is the part interviewers often probe: **CPU and memory are
enforced completely differently** at the limit. Exceed a CPU limit and
the container is *throttled* (slowed down, never killed). Exceed a
memory limit and it's *OOMKilled* outright, because memory can't be
throttled the way CPU time can; there's no way to partially deny an
allocation.

**`strategy`: RollingUpdate, maxSurge and maxUnavailable.**

```yaml
strategy:
  rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }
```

Kubernetes' default rolling-update percentages are useful general
defaults, but for a small API I prefer to make the availability choice
explicit. `maxUnavailable: 0` means the rollout must not intentionally
reduce available capacity below the desired replica count. A new pod
must pass readiness first, and only then does an old one get removed.
The cost is running `replicas + 1` pods for a few seconds during a
rollout, which is what `maxSurge: 1` permits. It's a trade most people
should make, and most default configs don't.

**`lifecycle`: graceful shutdown with preStop and
terminationGracePeriodSeconds.**

```yaml
lifecycle:
  preStop:
    exec: { command: ["sh","-c","sleep 5"] }
terminationGracePeriodSeconds: 40
```

Your app already handles `SIGTERM` gracefully; uvicorn finishes
in-flight requests before exiting. So why the extra sleep? Removing a
pod from a Service's routing table (updating iptables rules across the
cluster) happens asynchronously, in parallel with the pod being told to
shut down, not strictly before it. Without a small delay, a request
routed a few milliseconds before shutdown could arrive at a process
that's already exiting. `preStop` delays the actual `SIGTERM` just long
enough for that propagation to finish first.
`terminationGracePeriodSeconds` must be longer than `preStop`'s own
delay plus real shutdown time, or Kubernetes sends `SIGKILL` before the
process finishes exiting cleanly.

### Step 7: How traffic actually reaches one `rag-api` Pod

Assume the frontend container calls:

```text
http://rag-api:8000/query
```

`rag-api` is the name of a **Kubernetes Service**, not the FastAPI process.

#### 1. DNS resolves the stable Service name

CoreDNS resolves:

```text
rag-api
    -> 10.96.20.50
```

Assume `10.96.20.50` is the Service's ClusterIP.

A ClusterIP is a virtual, internal IP belonging to **that Service**. It is not
the IP of the whole Kubernetes cluster.

#### 2. Kubernetes tracks the usable Pod backends

Suppose the Deployment currently has three Pods:

```text
10.244.1.20   Ready
10.244.2.15   Ready
10.244.3.8    NotReady
```

The Service selector matches all three, but the not-ready Pod is not considered
a normal traffic backend. Modern Kubernetes stores this backend information in
`EndpointSlice` resources.

You can inspect the mapping with commands such as:

```bash
kubectl get svc rag-api -n rag
kubectl get endpointslices -n rag
```

#### 3. Service networking sends the connection to one ready Pod

The caller connects to:

```text
10.96.20.50:8000
```

No FastAPI process is actually listening on the ClusterIP itself. The ClusterIP
is a virtual Service address.

The cluster's Service networking then forwards the connection to one usable Pod
IP, for example:

```text
10.96.20.50:8000
        ->
10.244.2.15:8000
```

Traditionally this behavior is implemented using `kube-proxy` with iptables or
IPVS. Some modern networking stacks implement equivalent Service routing using
eBPF instead, so `kube-proxy + iptables` should not be treated as the only
possible implementation.

```mermaid
flowchart LR
  F["frontend Pod"] -->|"1. call http://rag-api:8000"| D["CoreDNS"]
  D -->|"2. rag-api -> 10.96.20.50"| S["Service: rag-api
ClusterIP 10.96.20.50"]
  S -->|"3. forward connection"| P1["rag-api Pod
10.244.1.20
Ready"]
  S -->|"3. or this one"| P2["rag-api Pod
10.244.2.15
Ready"]
  S -. "Not a normal backend" .-> P3["rag-api Pod
10.244.3.8
NotReady"]
```

The mental model is:

```text
DNS:
Which stable Service name/IP?

Service + EndpointSlices:
Which Pods currently belong to this Service and are usable?

Service networking:
Which usable Pod receives this connection?
```

This is why scaling from one replica to five does not require changing DNS.
The Service identity remains stable while the backend Pod set changes.

### Step 8: Autoscaling, and its real limits for this workload

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef: { kind: Deployment, name: rag-api }
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 65 } }
```

HPA isn't a separate execution path. It polls `metrics-server` (which
isn't installed on kind by default, and needs a patch to trust kind's
self-signed kubelet certificate), divides each pod's real CPU usage by
its own `requests.cpu`, averages across pods, and, if that's off
target, writes a new number straight into `spec.replicas`. That's the
same field `kubectl scale` edits by hand.

> **The interview-favorite nuance.** CPU is a weak signal for this
> specific workload. Generation latency is dominated by Ollama, an
> external process HPA can't see at all. A request can sit queued
> waiting on a slow model call while the rag-api pod's own CPU sits
> nearly idle, so HPA sees no scaling signal at exactly the moment
> latency is degrading. The correct fix is a custom metric via the
> Prometheus Adapter (for example, in-flight request count, or p95
> latency, already exported as
> `rag_http_request_duration_seconds`), not raw CPU. CPU-based HPA is
> used here to demonstrate the mechanism. In the current Minikube
> cluster the HPA shows `<unknown>/65%` because the Metrics API is not
> available; `kubectl top` fails until metrics-server is installed and
> working. Saying this out loud, unprompted, is exactly the kind
> of answer that separates "used HPA once" from "understands HPA."

### Step 9: Frontend and Ingress, two proxy hops, not one

Your frontend's nginx already reverse-proxies API paths to the backend.
That's how the Compose deployment avoids ever needing CORS. The
Kubernetes Ingress doesn't replace that; it sits in front of it.

```mermaid
flowchart LR
  Br["Browser"] -->|":8080"| Ing["Ingress\n(host: rag.local)"]
  Ing --> SF["Service: frontend"]
  SF --> PF["Pod: frontend (nginx)"]
  PF -->|"proxy_pass\nhttp://rag-api:8000\n(a SECOND, in-cluster hop)"| SA["Service: rag-api"]
  SA --> PA["Pod: rag-api"]
```

The Ingress only ever knows about one backend, "frontend." It has no
idea rag-api exists; that hop happens entirely inside nginx, one layer
further in. Keeping the backend Service name `rag-api` preserved the
application-level proxy target, but the move still exposed one
portability bug: nginx had a Docker-Compose-specific DNS resolver
assumption. The fix was to derive the resolver from the container's
real `/etc/resolv.conf` at startup instead of hardcoding one
environment. That let the same frontend image work across Compose and
Kubernetes without hardcoding a Kubernetes DNS IP.

### Step 10: NetworkPolicy, and the gotcha that makes it a useful interview topic

`NetworkPolicy` is a pod-level firewall: default-deny, then explicit
allow rules by label. Yours does this.

```yaml
podSelector: {}          # every pod in the namespace
policyTypes: [Ingress]   # deny all inbound by default...
# ...then explicit allows, e.g.:
podSelector: { matchLabels: { app: rag-api } }
ingress:
  - from: [{ podSelector: { matchLabels: { app: frontend } } }]
    ports: [{ port: 8000 }]
```

> **The gotcha.** The API server accepts and stores a `NetworkPolicy`
> object unconditionally; it never fails to apply. But nothing about
> the object itself blocks a single packet. That's the CNI plugin's
> job, and kind's default CNI, **kindnet**, doesn't implement
> NetworkPolicy enforcement at all. Apply this exact file on a stock
> kind cluster and every rule is silently a no-op. `kubectl get
> networkpolicy` shows it "successfully created," and traffic flows
> exactly as if it didn't exist. Swapping in Calico as the CNI is what
> actually turns these into real firewall rules.

This is a strong interview answer because it shows you understand
Kubernetes' architecture, not just its API. The API server is a
trusting store of desired state. Every kind of enforcement is delegated
to a separate controller or plugin that has to actually exist and
implement that kind. "Did I create the object" and "is anything acting
on it" are always two different questions.

### Step 11: The init-db Job, and a Job vs. Deployment distinction

`make up`'s schema-init step (`python scripts/init_db.py`) needs to run
exactly once and then stop, the opposite of what a Deployment
guarantees (a Deployment that exits is treated as a failure to be
restarted). That's what a `Job` is for.

```yaml
apiVersion: batch/v1
kind: Job
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: init-db
          command: ["python", "/scripts/init_db.py"]
```

`scripts/init_db.py` isn't baked into the image at all (the Dockerfile
only copies `src/` and `config/`). It's mounted in from a ConfigMap
built directly from the real file, so nothing is duplicated by hand.

### Step 12: Observability, reused verbatim

Prometheus, Grafana, and Jaeger get their own Deployments and Services
in `k8s/observability/`, but every config file they read
(`prometheus.yml`, Grafana's datasource and dashboard provisioning) is
the **exact same file** your Compose stack already uses. I confirmed,
rather than assumed, that their hardcoded target hostnames
(`rag-api:8000`, `prometheus:9090`, `jaeger:16686`) already resolve
correctly once those Services exist in the same namespace under those
same names. Zero content duplicated.

### Step 13: What `kustomize` is doing here

There is no mystery here: `kustomize` is being used as a convenient way to
treat several normal Kubernetes YAML files as one deployable set.

Without it, you could do this:

```bash
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/configmap-app.yaml
kubectl apply -f k8s/base/postgres-statefulset.yaml
kubectl apply -f k8s/base/rag-api-deployment.yaml
kubectl apply -f k8s/base/rag-api-service.yaml
...
```

With `k8s/base/kustomization.yaml`, you instead run:

```bash
kubectl apply -k k8s/base
```

The `resources:` list tells kustomize which normal manifests belong to the
bundle.

This repo also uses one useful transformation:

```yaml
images:
  - name: local-rag-api
    newTag: kind
  - name: local-rag-frontend
    newTag: kind
```

That means the image tag can be changed centrally instead of editing several
Deployment manifests by hand.

#### Kustomize vs. Helm

They solve related but different problems.

```text
Plain manifests
    = direct Kubernetes YAML

Kustomize
    = compose and patch plain YAML
      no template language required

Helm
    = package and template Kubernetes YAML
      values.yaml + {{ template expressions }} + release history
```

For learning, plain manifests plus a small amount of Kustomize are useful
because you can still see the real Kubernetes objects directly. Later, a Helm
chart can package the same concepts for repeatable dev/staging/prod releases.

## Part 3: Operating it

### Pod death and self-healing

Pod lifecycle states you'll actually see while running these:

- `Pending`: scheduled, image pulling, or waiting on init containers
- `Running` + `Ready`: passing its readiness probe, listed in Service Endpoints
- `Running`, `NotReady`: alive but failing readiness, benched from traffic
- `CrashLoopBackOff`: container keeps exiting, backoff delay growing each retry
- `ImagePullBackOff`: the image tag doesn't exist or can't be reached

```bash
kubectl delete pod -n rag -l app=rag-api --wait=false
kubectl get pods -n rag -l app=rag-api -w
```

**What you're actually watching:** the ReplicaSet controller notices
that the number of matching Pods is below the desired count and
creates a replacement Pod. The Deployment manages the ReplicaSet and
rollout history above it; the Scheduler then chooses a node for the
new Pod. It never "restarts" the old one. Compare this against deleting
`postgres-0`: same mechanism, but the StatefulSet gives the replacement
the identical name and reattaches the identical PVC, so data survives.

### Scaling 1 to 3

```bash
kubectl scale deployment rag-api -n rag --replicas=3
kubectl get endpoints rag-api -n rag
```

The payoff line is that second command. Watch the Endpoints object
grow from one pod IP to three, live. That list is the load-balancing
set; nothing else about the Service object changes at all.

### Making readiness fail for the right reason

The application now has separate health contracts, so a real dependency
outage can demonstrate readiness correctly rather than relying on a fake
broken path.

```bash
kubectl scale statefulset postgres -n rag --replicas=0
kubectl get pods -n rag -w
kubectl get endpointslice -n rag -l kubernetes.io/service-name=rag-api -o yaml
```

With Postgres down, the expected behavior is:

```text
/livez   -> HTTP 200   # FastAPI process itself is alive
/readyz  -> HTTP 503   # cannot serve normal requests correctly
/health  -> HTTP 200   # detailed diagnostics can still report degraded state
```

The API container should keep running, but the Pod becomes `Ready=False`
and its EndpointSlice condition becomes unready, so normal Service
traffic stops using it. Restore Postgres with `kubectl scale statefulset
postgres -n rag --replicas=1` and watch readiness recover without
restarting a healthy API process.

### A real rolling update

```bash
kubectl rollout restart deployment/rag-api -n rag
kubectl get pods -n rag -l app=rag-api -w
```

Watch the pod count briefly hit `replicas + 1` before settling. That is
`maxSurge: 1` visibly working. To see the availability trade-off
directly, temporarily set `maxUnavailable: 1` on a multi-replica test
deployment and observe that Kubernetes is allowed to proceed with one
fewer ready Pod. Percentage values are converted to integers by
Kubernetes, so always reason about the effective number at your actual
replica count rather than assuming that `25%` means one Pod.

### Deploying a bad tag

```bash
kubectl set image deployment/rag-api -n rag rag-api=local-rag-api:does-not-exist
kubectl get pods -n rag -l app=rag-api   # ImagePullBackOff
curl -s http://localhost:8000/health      # still works, old pod untouched
kubectl rollout undo deployment rag-api -n rag
```

Because `maxUnavailable: 0`, the rollout gets **stuck**, not
**broken**. The working old pod is never touched while Kubernetes
retries the pull with growing backoff. This is the whole point of the
strategy choice made back in Step 6, made concrete.

### Killing a node

```bash
docker stop rag-learning-worker2
kubectl get nodes -w   # NotReady within ~40s...
```

Pods on a lost node are not necessarily recreated elsewhere immediately.
The control plane first has to decide that the node is genuinely unavailable,
not merely experiencing a short network problem. Node health and eviction are
controlled through node status, taints, and Pod tolerations, so there is
normally a grace period before workloads are replaced elsewhere.

The important lesson is not to memorize one timeout value. It is to understand
why Kubernetes is conservative: during a network partition, the old node may
still be running even though the control plane cannot see it. Stateful
workloads therefore need storage and application-level designs that tolerate
failover safely rather than assuming "NotReady" means "the old process is
definitely dead."

## Part 4: Debugging notes

### The health-contract bug we found and fixed

**The original setup.** Both readiness and liveness pointed at `/health`.
That endpoint correctly reported dependency degradation in JSON, but
still returned HTTP 200.

```text
readinessProbe -> /health
livenessProbe  -> /health
```

```json
{
  "status": "degraded",
  "dependencies": {
    "vectorstore": "down",
    "llm": "ok"
  }
}
```

**Why that was a problem.** Kubernetes HTTP probes judge success from
the HTTP result; they do not parse the JSON body. A Postgres outage
therefore left the Pod marked ready even though the application
reported itself degraded. Separately, the application's heavy cold
start was long enough that liveness could restart it before startup
had legitimately finished.

**The fix.** The application now exposes three deliberately different
contracts:

```text
/livez   -> process-level liveness, no downstream dependency checks
/readyz  -> dependency-aware readiness, non-2xx when normal service is unavailable
/health  -> detailed diagnostics, preserving the existing response contract
```

Kubernetes now uses `startupProbe -> /livez`, `livenessProbe -> /livez`,
and `readinessProbe -> /readyz`. Nine unit tests were added for the
endpoint semantics. The common base has a generic startup budget, while
the much longer tolerance measured on this low-resource laptop is
isolated in the Minikube overlay.

> **Interview takeaway.** A health endpoint is an operational contract,
> not just a URL that returns JSON. Liveness, readiness, and startup
> failures should trigger different actions, so they should be designed
> around different failure domains. The useful part of this story is
> that the Kubernetes exercise exposed the application contract problem,
> and the fix was made at the correct layer rather than masking it with
> a probe workaround.

### kustomize's security sandbox

**The plan.** Generate the init-db Job's ConfigMap straight from the
real `scripts/init_db.py`, using kustomize's `configMapGenerator`, so
nothing is ever duplicated by hand.

**What happened.** `kubectl kustomize k8s/jobs` failed immediately.

```
error: loading KV pairs: file sources: [../../scripts/init_db.py]:
security; file '.../scripts/init_db.py' is not in or below '.../k8s/jobs'
```

**The diagnosis.** kustomize enforces a load restrictor that refuses to
read generator input from outside the directory tree rooted at the
kustomization.yaml itself. This is a deliberate security boundary, so a
kustomization can't silently reach out and read arbitrary files from
elsewhere on disk when someone else applies it. `scripts/` and
`observability/` are siblings of `k8s/`, not descendants of
`k8s/jobs/`. They sit structurally outside the allowed tree no matter
how the relative path is written. Newer standalone `kustomize` builds
expose `--load-restrictor LoadRestrictionsNone`; the `kubectl` version
available here (1.24) exposes no such flag at all, which I checked
directly against its `--help`, not assumed.

**The fix.** I dropped the generator for those two layers rather than
restructure the repo to please the tool. The same source files are
created imperatively instead.

```bash
kubectl create configmap init-db-script -n rag \
  --from-file=scripts/init_db.py \
  --dry-run=client -o yaml | kubectl apply -f -
```

> **The trade-off worth stating in an interview.** The generator form
> isn't strictly better, just different. It content-hashes the
> ConfigMap's name, so an edit produces a genuinely new object and
> forces anything referencing it to roll out automatically, the same
> "config change forces a new object" guarantee a Deployment gets for
> free. The imperative form uses a fixed name, so a content change
> needs an explicit delete-and-recreate (or a manual `kubectl rollout
> restart`) to take effect. Knowing exactly why a reasonable-looking
> approach fails, and picking a working alternative without bending the
> repo to fit the tool, reads better in an interview than never hitting
> the wall at all.

### What can break when you scale from 1 replica to 3

**Kubernetes can create more Pods; it cannot make application state
distributed for you.** Scaling a stateless HTTP handler is usually
straightforward. The problems appear wherever the application quietly
assumes that one process owns all state, all locks, all files, or all
background work.

1. **Process-local mutable state can become inconsistent.** The
   synthetic MCP business-case store uses an in-memory Python `dict`.
   If that state is expected to be shared, three Pods mean three
   independent copies: a write handled by Pod A is not automatically
   visible to Pods B and C. The fix is to put shared mutable state in a
   shared system such as Postgres or another appropriate datastore.
2. **Process-local locks stop protecting the system globally.** A
   `threading.Lock` or `asyncio.Lock` coordinates callers only inside
   one Python process. It does nothing between Pods. If an operation
   must be mutually exclusive across replicas, the coordination
   mechanism must also be distributed, for example a database
   constraint/transaction, a distributed lock where appropriate, or a
   queue/ownership design.
3. **Rate limiting can multiply.** An in-memory limiter enforces a quota
   independently in every process. With three replicas, the effective
   global allowance can rise toward roughly three times the intended
   per-client limit, depending on how traffic is distributed. A truly
   global limit needs shared state or enforcement at a common gateway.
4. **Local files are neither durable nor shared.** A file written to
   one Pod's container filesystem is invisible to the other replicas
   and disappears when that Pod is replaced. If uploads must survive
   restarts or be visible across replicas, use object storage or an
   appropriate shared persistent-storage design.
5. **Database connection pools multiply with replicas.** If each Pod
   can open, for example, 20 database connections, scaling from 1 to 10
   Pods can turn that into roughly 200 possible connections. The
   application may scale while the database becomes the bottleneck.
   Replica count and per-process pool size therefore need to be
   designed together.
6. **Process-local caches can diverge.** Each Pod can hold a different
   cached value or freshness window. That may be acceptable for a
   best-effort cache, or incorrect if the cache is being treated as
   authoritative state. The consistency requirement determines whether
   the cache can remain local or needs shared invalidation/state.
7. **Background work can accidentally run multiple times.** If every
   API replica starts the same scheduler or periodic job, scaling to
   three can execute the same task three times. Scheduled/queued work
   needs explicit ownership, idempotency, leader election, or a
   dedicated worker/Job/CronJob design.
8. **Per-replica model memory scales roughly linearly.** Every
   `rag-api` Pod loads its own embedding/model libraries. That is not a
   correctness bug, but three replicas can mean roughly three copies of
   those weights and runtime state in memory, which directly affects
   requests, limits, node capacity, and startup contention.

> **Scaling can move the bottleneck.** More Pods do not guarantee more
> throughput. The database, Ollama/model server, connection pools,
> locks, storage, or an external API can saturate first. A good scaling
> test measures end-to-end throughput and latency, not just whether
> Kubernetes successfully created more replicas.

The design goal for a horizontally scaled API is not "no state
anywhere." It is: **keep replica-local state disposable, and put
correctness-critical shared state in systems whose consistency and
durability semantics are explicit.**

## Part 5: Interview prep

### 25 interview questions, with interview-ready answers

Answer each one out loud first. Keep the first answer to roughly 30-60
seconds, then go deeper only if the interviewer asks.

**1. Why is Postgres a StatefulSet here instead of a Deployment?**
A Deployment assumes its Pods are interchangeable. That is a good fit
for `rag-api`, but not for a database. A StatefulSet gives each Pod a
stable ordinal identity such as `postgres-0` and, through
`volumeClaimTemplates`, a stable PVC associated with that identity. If
`postgres-0` is recreated, its Pod IP can change, but it can come back
with the same name and the same persistent storage.

**2. What is the difference between startup, liveness, and readiness
probes?**
**Startup** asks whether a slow-starting application has finished
starting. Until it succeeds, Kubernetes does not let liveness kill the
container for being slow. **Liveness** asks whether the process is
unhealthy enough that kubelet should restart the container.
**Readiness** asks whether this Pod should receive normal Service
traffic right now. In this project, `/livez` checks only the
application process, while `/readyz` can fail when a critical
dependency is unavailable.

**3. Walk through how a request to the rag-api Service reaches one
specific Pod.**
A client inside the namespace calls `http://rag-api:8000`. CoreDNS
resolves `rag-api` to the Service's stable ClusterIP. The Service
selector identifies the matching Pods, and EndpointSlices contain the
currently usable backend Pod IPs. The cluster's Service data plane,
traditionally kube-proxy with iptables/IPVS or, in some clusters, an
eBPF/CNI implementation, forwards that connection to one ready Pod IP.
The Scheduler is not involved in request routing; it only chooses which
node a Pod runs on.

**4. Why is the Postgres Service headless with `clusterIP: None`?**
A normal ClusterIP Service gives clients one virtual IP in front of a
set of interchangeable backends. A headless Service skips that virtual
IP and lets DNS expose the actual StatefulSet endpoints, which is
useful when individual members need stable network identity such as
`postgres-0.postgres...`. A StatefulSet does not universally require a
headless Service, but the combination is common when stable per-Pod
DNS matters.

**5. What do `maxSurge: 1` and `maxUnavailable: 0` mean?**
With `replicas: 2`, `maxSurge: 1` allows Kubernetes to temporarily run
at most 3 Pods during a rollout. `maxUnavailable: 0` means Kubernetes
should not intentionally reduce the number of available Pods below the
desired count. In our bad-image exercise, the two old Pods remained
ready while one new Pod entered `ImagePullBackOff`. The rollout got
stuck safely instead of taking the working version down.

**6. When would you allow `maxUnavailable: 1`?**
When temporarily losing one ready replica is acceptable and I want to
avoid extra surge capacity. For example, with 20 replicas, running 19
for a short period may be fine. With only one or two replicas, losing
one is a much larger availability hit, so I would be more conservative.
The right value depends on traffic, spare capacity, startup time, and
the service's availability target.

**7. What is the difference between an HTTP probe and an exec probe?**
An HTTP probe makes an HTTP request to the Pod and judges success from
the response. That is natural for FastAPI, so `rag-api` uses HTTP
endpoints. An exec probe runs a command inside the container and checks
its exit status. Postgres uses an exec probe with `pg_isready`. The
question is the same, but the checking mechanism is different.

**8. What happens after I delete a rag-api Pod?**
The Pod is terminated. The ReplicaSet controller sees that the actual
Pod count is now below the desired replica count and creates a
replacement Pod object. The Scheduler assigns that new Pod to a node.
The kubelet on that node starts its init container and main container
through the container runtime. Once the new Pod passes readiness, its
IP becomes a ready backend in the Service's EndpointSlice and it can
receive traffic.

**9. What is the difference between a container restart and a Pod
replacement?**
If the application process exits or fails liveness, kubelet can
restart the container inside the same Pod. The Pod name normally stays
the same and the `RESTARTS` count increases. If the whole Pod is
deleted, the controller creates a different Pod object. A Deployment
replacement normally gets a new Pod name and IP. A StatefulSet
replacement can come back with the same ordinal name, such as
`postgres-0`, but still with a different Pod IP.

**10. Why is CPU a weak HPA signal for this RAG workload?**
Much of the request latency can come from waiting on Ollama, database
I/O, retrieval, or other downstream work. The API Pod can be waiting
while its own CPU is relatively low, so CPU utilization can miss the
actual bottleneck. For a production RAG service I would consider a
metric closer to demand, such as in-flight requests, queue depth, model
concurrency, or latency, and validate it with load tests.

**11. What happens if a container exceeds its CPU limit versus its
memory limit?**
CPU and memory are different resources. If a container reaches its CPU
limit, the kernel throttles its CPU time, so the application becomes
slower. If it exceeds its memory limit, the process can be killed and
Kubernetes reports `OOMKilled`. Resource requests are also important
because the Scheduler uses them for placement, and CPU-based HPA
commonly calculates utilization relative to the CPU request.

**12. Why did our HPA show `cpu: <unknown>/65%`?**
The HPA object existed, but the cluster had no usable CPU metrics. In
our Minikube cluster, `kubectl top` returned `Metrics API not
available`, and there was no metrics-server Pod. So the HPA knew the
target was 65%, but had no current value to compare against it. I
would verify metrics-server first, then confirm the target Pods have
CPU requests configured.

**13. What is `CrashLoopBackOff`? Does it imply exit code 137?**
No. `CrashLoopBackOff` means Kubernetes has repeatedly restarted a
container and is increasing the delay between retries. The underlying
exit reason can be many things. In our frontend example the previous
container actually exited with code 0 after kubelet restarted it
because the liveness probe kept failing. Exit code 137 specifically
means termination by `SIGKILL`; it is often seen with OOM kills, but
you should inspect `kubectl describe pod`, the last state, events, and
`kubectl logs --previous` instead of guessing from the status label.

**14. What is the difference between `ImagePullBackOff`,
`CrashLoopBackOff`, and `Pending`?**
`ImagePullBackOff` means the node cannot obtain the image and is
backing off between pull attempts. `CrashLoopBackOff` means a container
starts and then repeatedly exits or is restarted. `Pending` means the
Pod has not become runnable yet, for example because it cannot be
scheduled, is waiting for storage, is pulling an image, or is still
waiting on initialization. `kubectl describe pod` and Events usually
tell you which case you actually have.

**15. What happens internally when I run `kubectl apply` for a
Deployment?**
`kubectl` sends the desired object to the API server. After
authentication, authorization, validation, and admission, the desired
state is persisted in etcd. The Deployment controller notices the new
desired state and manages a ReplicaSet. The ReplicaSet controller
creates the required Pods. The Scheduler assigns each unscheduled Pod
to a node. The kubelet on that node asks the container runtime to start
the containers and continuously reports status back through the API
server.

**16. Can one Deployment have more than one ReplicaSet?**
Yes. A Deployment creates a new ReplicaSet when its Pod template
changes and keeps older ReplicaSets, usually scaled to zero, as rollout
history. During a rolling update the old and new ReplicaSets can both
have running Pods at the same time. The Deployment coordinates the
rollout between ReplicaSets; each ReplicaSet's job is simply to
maintain its current desired number of Pods.

**17. Who chooses the node for a Pod, and who chooses the Pod for a
request?**
The **Scheduler** chooses a node for a new Pod. Request routing is
separate. A Service's ready EndpointSlices and the cluster's Service
networking implementation determine which backend Pod receives a
connection. This is an important distinction: Scheduler answers *where
does the Pod run?*; Service networking answers *which ready Pod
receives this request?*

**18. What are the main Kubernetes control-plane components?**
The core control plane is the API server, etcd, the Scheduler, and the
controller manager. The API server is the front door for cluster
operations; etcd stores cluster state; the Scheduler assigns Pods to
nodes; controllers continuously reconcile desired and actual state. On
each node, kubelet and the container runtime make assigned Pods
actually run. CoreDNS, the CNI, CSI drivers, ingress controllers, and
metrics-server are important supporting components, but they are not
all part of the core control plane.

**19. Who handles `volumeClaimTemplates` in a StatefulSet?**
The StatefulSet controller is responsible for the StatefulSet's desired
Pod identities and associated claims. A `volumeClaimTemplates` entry
results in a PVC per StatefulSet ordinal, such as `pgdata-postgres-0`.
Storage controllers and the CSI/storage provisioner then arrange the
backing volume, and kubelet mounts that volume into the Pod on the
selected node.

**20. If a ConfigMap changes, does a running application automatically
use the new value?**
Not necessarily. If the ConfigMap is consumed as environment
variables, those values are copied into the container only at process
start, so existing Pods do not change. If it is mounted as files,
kubelet can update the files, but the application still has to notice
and reload them. For this application, which reads configuration at
startup, I would normally roll the Deployment so new Pods start with
the new configuration.

**21. Why isn't "Postgres port 5432 is open" the same as "the
application is ready"?**
A TCP check only proves that something is listening on the port. It
does not prove the schema exists, the pgvector extension is installed,
migrations have completed, credentials work, or the database can
satisfy the application's actual queries. We saw this ourselves: raw
connectivity worked while the vector-store health check still failed
until the init-db Job created the required extension and schema.
Network reachability and application readiness are different layers.

**22. What happens if the Kubernetes control plane is temporarily
unavailable?**
Existing containers on healthy nodes can continue running because
kubelet and the container runtime are local to those nodes. But
operations that need control-plane decisions are affected: `kubectl`
calls fail, controllers cannot reconcile normally, and new unscheduled
Pods cannot be assigned by the Scheduler. So control-plane failure does
not automatically stop every application process, but it prevents the
cluster from adapting correctly while the control plane is unavailable.

**23. What breaks when an application that works with one replica is
scaled to several?**
Anything that assumes one process owns the world can become wrong:
in-memory mutable state, process-local locks, rate limiters, local
files, caches, scheduled work, and database connection pools. Replicas
also multiply per-process memory such as embedding/model weights.
Kubernetes only creates and routes between Pods; it does not
synchronize their hidden state. I would externalize correctness-critical
shared state, make background work idempotent/owned, size connection
pools with replica count, and load-test the real downstream bottleneck
rather than assuming more Pods always means more throughput.

**24. How would you debug a Pod that is not working?**
I start broad, then narrow down: `kubectl get pods -o wide`, then
`kubectl describe pod` for scheduling, image-pull, probe, mount, and
restart events. I check `kubectl logs` and `--previous` if it
restarted. If it is Running but unreachable, I inspect readiness, the
Service selector, EndpointSlices, and Ingress. For resource problems I
check requests/limits, `OOMKilled`, and `kubectl top` if metrics-server
is available. The key is to use the Pod state and Events to decide the
next command instead of randomly trying fixes.

**25. How do you roll back a bad deployment?**
First I confirm what failed using rollout status, the new ReplicaSet,
Pod Events, and logs if the container started. If the new version is
the cause, I use `kubectl rollout undo deployment/rag-api -n rag` or,
in a Helm/GitOps environment, revert the release or Git change through
the deployment mechanism used by the team. Our bad-image exercise
showed the useful behavior of `maxUnavailable: 0`: the old healthy
Pods stayed in service while the new ReplicaSet failed to pull its
image, so rollback could restore the previous desired state without
first taking the service down.

### Flashcard glossary

| Term | Meaning |
|---|---|
| **Reconciliation loop** | A controller's endless "compare desired vs. actual, correct the gap" cycle. The mechanism behind every self-healing behavior in Kubernetes. |
| **Label / Selector** | The only mechanism connecting objects to each other. No IDs, no foreign keys. Get one wrong and something silently selects zero (or the wrong) targets. |
| **Pod** | The smallest deployable unit; one or more containers sharing a network namespace, always co-scheduled on one node. |
| **ReplicaSet** | Keeps N copies of a pod template running. A Deployment manages ReplicaSets over time; you rarely touch one directly. |
| **EndpointSlice** | Kubernetes objects that represent the backend addresses for a Service. Readiness affects whether a Pod endpoint is considered ready for normal Service traffic. |
| **Headless Service** | A Service with `clusterIP: None`. It does not provide one shared virtual ClusterIP and is commonly used with StatefulSets for stable per-Pod network identity. |
| **ExternalName** | A Service that's pure DNS indirection to something outside the cluster. No pods, no ClusterIP, no selector. |
| **Init container** | Must exit successfully before the main container starts. Kubernetes' answer to Compose's `depends_on`, since a Deployment can't natively express "wait for that other workload." |
| **Requests vs. Limits** | Requests feed scheduling and HPA math; limits are a hard runtime ceiling. CPU over the limit throttles; memory over the limit OOMKills. |
| **maxSurge / maxUnavailable** | Controls how a rolling update trades "run extra pods briefly" against "let ready-pod count dip" during a spec change. |
| **NotReady / unreachable toleration** | Pods normally tolerate the node NotReady/unreachable taints for about 300 seconds by default before eviction behavior takes effect, avoiding overly eager duplicate recovery during brief partitions. |
| **NetworkPolicy** | Declarative pod-level firewall rules. Real only if the cluster's CNI plugin actually implements enforcement (kindnet doesn't; Calico does). |
| **HPA** | Polls metrics-server, compares against target utilization, and writes a new number into a Deployment's own `spec.replicas`. Not a separate execution path. |
| **Job** | Run-to-completion semantics with retry and backoff, then stop. Structurally different from a Deployment, which assumes forever-running. |
| **kustomize** | Composes plain YAML with no templating language. Its `configMapGenerator` enforces a security sandbox on where generator input files can live. |

## Additional questions, in plain English

You're prepping for an interview where you're claiming ownership of a
real system serving real customers, not a toy. So these answers lean
toward "what would actually be true in a real production setup," even
in places where this repo's own `k8s/` is just a local learning
exercise and isn't meant to match that yet. Where the two differ, that
is called out directly instead of blurred together.

**Q: Are these YAML files only Helm chart files, or are they different?**

Different. What's in `k8s/` are plain, hand-written Kubernetes files,
exactly what you'd type yourself and hand to `kubectl apply -f`. Helm
is a separate tool built on top of plain YAML files like these. It adds
three things this repo doesn't have: templating (so the same file can
have `{{ .Values.replicaCount }}` instead of a hardcoded `3`, filled in
differently per environment), packaging (many files bundled into one
versioned "chart" you install, upgrade, and roll back as a single
unit), and a release history (Helm remembers what version of the chart
is currently installed). Kustomize, which this repo does use for
`k8s/base`, is a lighter middle ground. It composes and patches plain
YAML files with no templating language at all, just overlays. A real
company, especially a large one, will almost always use Helm (or
Kustomize overlays, or both) instead of raw files exactly like these,
so the same app definition can be deployed to dev, staging, and prod
with different values, without copying the whole file three times.

**Q: Does Kubernetes run Compose files, or Dockerfiles?**

Neither, directly. Kubernetes runs a container **image**: the finished
thing a Dockerfile produces after you run `docker build`. The
Dockerfile itself is just the recipe. By the time Kubernetes is
involved, that recipe has already been baked into an image sitting
somewhere, either a registry or, in this local setup, loaded into
Minikube's node image store. Kubernetes has no idea what a Dockerfile is and
never reads one. It also doesn't read `docker-compose.yml`. Compose is
a completely separate tool with its own file format that only Docker's
own `docker compose` command understands. What replaces Compose's job
in Kubernetes is your own YAML manifests (Deployment, Service, and so
on): the same basic idea ("here's what to run, here's the config"),
with different, more detailed syntax spread across more object types.

**Q: What are HPA, Ingress, StatefulSet (is there any other "Set"?), and NetworkPolicy?**

- **HPA (HorizontalPodAutoscaler)**: automatically changes how many
  copies of your app are running, based on real usage such as CPU
  load. More copies when busy, fewer when quiet.
- **Ingress**: the rule that says "requests coming from outside the
  cluster, for this domain name, should go to that internal Service."
  It's how the outside world gets in at all.
- **StatefulSet**: like a Deployment, but for things that need a stable
  identity and their own permanent storage, mainly databases.
- **Is there another "Set"?** Yes, two more worth knowing.
  - **ReplicaSet**: the low-level thing that actually keeps N copies of
    a pod running. You almost never write one by hand; a Deployment
    creates and manages one for you underneath.
  - **DaemonSet**: runs exactly one copy of a pod on every node in the
    cluster, automatically, even as nodes are added or removed. Used
    for things that need to be everywhere, such as a log-collector
    agent, a monitoring agent, or a network plugin. Nothing in this
    repo uses one, but it's a real, commonly asked about object type.
- **NetworkPolicy**: a firewall rule between pods. "Only pods with this
  label are allowed to talk to pods with that label, on this port."

**Q: You want a container that crashes to come back automatically, without a human running docker start at 3am. Why didn't Docker itself handle this?**

Good catch. Plain Docker (and Compose) genuinely can do this much on
its own, on one machine. Your own `docker-compose.yml` already has
`restart: unless-stopped` on both services, and that alone restarts a
crashed container automatically, with no Kubernetes involved. The
honest line isn't "Docker can't restart a container." It's that Docker
only knows about **one machine**. It has no idea other machines exist,
can't move a container to a different machine, can't decide which of
ten machines has free capacity right now, and has no way to keep a
database's storage attached to whichever machine its container ends up
on next. The moment you need more than one machine, either for capacity
or so the whole app survives one machine dying and not just one
container, you need something whose job is to watch many machines as
one fleet and make placement decisions across all of them. That's the
gap Kubernetes fills. On a single machine, Compose's restart policy
already gets you most of the way there.

**Q: You want to update the app without a window where nothing is running. What does that actually mean?**

Picture the naive way to deploy an update: stop the old container, then
start the new one. Between those two steps, for however many seconds
it takes the new one to boot up, there are **zero** copies of your app
running. Any request that arrives in that gap just fails. That gap is
the "window where nothing is running." A rolling update avoids it by
doing the steps in the other order: start the new version first, wait
until it's confirmed healthy and actually taking traffic, then stop the
old version. At every single moment during the whole update, at least
one working copy is up and answering requests. That's what
`k8s/base/rag-api-deployment.yaml`'s `maxUnavailable: 0` setting
guarantees.

**Q: So can there be more than two namespaces for the same app? I'm assuming there can only be one namespace.**

Both assumptions are off; there's no cap at all. A cluster can have as
many namespaces as you want: one per environment (dev, staging, prod),
one per team, or one per customer if you're doing that kind of
isolation. Companies routinely run dozens. The same container image can
also run in several namespaces at once, completely independently, with
no awareness of each other. For example, a "rag-api" Deployment in
namespace `dev` and a separate, unrelated "rag-api" Deployment in
namespace `staging` can use the same image with different data and
different config, and never interact. This repo happens to use exactly
one namespace (`rag`) because it's a small learning setup, not because
one is any kind of limit.

**Q: Is a container just called a "Pod" in Kubernetes? Can a Pod hold more than one container? What's a "replica" then?**

A few things to untangle here, one at a time.

- A Pod is **not** just a renamed container. It's a wrapper around one
  or more containers. Most of the time, including every Pod in this
  repo, that wrapper holds exactly one container, so in practice "Pod"
  and "container" feel interchangeable, but they're not the same
  concept.
- A Pod can hold more than one container. A common pattern is a main
  app container plus a small helper "sidecar" container, for example
  something that ships logs elsewhere. When it does, **all the
  containers in that one Pod share a single IP address**, not different
  ones. That's exactly what "share a network address" means: they're
  bundled tightly enough that from the network's point of view they
  look like one thing with one address, and they can talk to each
  other over `localhost`.
- One namespace can hold many Pods, and each Pod holds one or more
  containers.
- A **replica** is a full separate copy of the same Pod: same container
  image, same config, running side by side for redundancy or to handle
  more load. `replicas: 3` on a Deployment means three entirely
  separate Pod objects exist, each with its own IP address. They are
  not one Pod with three containers in it; that's the multi-container
  case above, a different thing entirely. A Service then load-balances
  traffic across those three separate Pods.

**Q: Where do you define how much space or resources a Pod gets, or the max Pods or size a namespace can have?**

Two separate, unrelated settings.

- **Per-container**: the `resources.requests` and `resources.limits`
  block already in `k8s/base/rag-api-deployment.yaml`. It sets how much
  CPU and memory this one container is guaranteed, and the hard ceiling
  it can't exceed.
- **Per-namespace**: a `ResourceQuota` object (a total cap on CPU,
  memory, or Pod count across the whole namespace) and a `LimitRange`
  object (a default, minimum, or maximum applied automatically to any
  container in that namespace that doesn't specify its own). **Neither
  exists in this repo's `k8s/` right now.** A real multi-team
  production cluster, the kind you'd find at a company like SAP or
  Amadeus, almost always has these, specifically so one team's app
  can't accidentally, or maliciously, starve every other team sharing
  the same cluster.

**Q: Can you show me a diagram of what's under what, and where each thing lives?**

There are two separate, crossing ways to slice a cluster. People often
flatten these into one mental picture, which is where confusion usually
comes from. Keeping them apart is worth more than the diagram itself.

**1. The logical ownership chain**: what belongs to what, on paper.

```mermaid
flowchart TB
  NS["Namespace: rag"]
  NS --> DEP["Deployment: rag-api"]
  DEP --> RS["ReplicaSet\n(created automatically by the Deployment)"]
  RS --> P1["Pod rag-api-a1b2"]
  RS --> P2["Pod rag-api-c3d4"]
  P1 --> C1["Container: rag-api\n(one running process)"]
  P2 --> C2["Container: rag-api\n(one running process)"]
  NS --> SS["StatefulSet: postgres"]
  SS --> P3["Pod postgres-0"]
  P3 --> C3["Container: postgres"]
  P3 --> PVC["PersistentVolumeClaim: pgdata-postgres-0"]
  NS --> SVC["Service: rag-api\n(points at P1 and P2 by label, does not 'contain' them)"]
```

**2. The physical placement axis**: which real machine each Pod
actually runs on. This is completely independent of the chain above.

```mermaid
flowchart TB
  CLUSTER["Cluster: rag-learning"]
  CLUSTER --> CP["Control-plane node\n(the brain: API server, etcd, scheduler)"]
  CLUSTER --> W1["Worker node 1"]
  CLUSTER --> W2["Worker node 2"]
  W1 --> P1["Pod rag-api-a1b2"]
  W1 --> P3["Pod postgres-0"]
  W2 --> P2["Pod rag-api-c3d4"]
```

Notice that `rag-api-a1b2` and `postgres-0` both happen to land on
Worker 1 in that second diagram, even though they belong to completely
different apps in the first diagram, and different namespaces don't map
to different nodes either. Namespace is an organizational and
permissions concept; node is a physical placement concept. The
Scheduler is the only thing that connects the two, deciding on the fly
which node any given Pod lands on.

**Q: Is apiVersion: v1 the latest version of Kubernetes itself?**

No, and this trips a lot of people up. `apiVersion` has nothing to do
with the version number of your Kubernetes cluster (like "1.24" or
"1.30"). It versions **one specific object type's own schema**, and
every object type versions independently of every other one. `v1` for
things like Pod, Service, ConfigMap, and Namespace means "the first
fully stabilized, no-longer-changing schema for this object." It has
stayed `v1` for years, even as the cluster itself moved through many
version numbers. Other object types sit on different apiVersions for
the same reason: Deployments are `apps/v1`, HPAs are `autoscaling/v2`
(it used to be `v1` and got a real redesign), and Ingress is
`networking.k8s.io/v1`. None of these numbers track each other, or the
cluster version, at all.

**Q: Can metadata.name and the app label be different? What's the actual difference between them?**

Yes, they can be different, and they mean genuinely different things.

- `metadata.name` is the object's own unique identifier, like a
  variable name. No two objects of the same kind in the same namespace
  can share one.
- `app: something` is just a **label**: an arbitrary tag you invent,
  with no built-in meaning to Kubernetes at all. It only matters
  because a Service's `selector` field looks for pods carrying a
  specific label value.

They're usually set to the same string (`name: rag-api`, label `app:
rag-api`) purely as a human convention, for readability. Kubernetes
itself never requires them to match. A real, common case where they
don't match is a canary release, where you might name a Deployment
`rag-api-canary` but give its pods the same `app: rag-api` label as the
stable Deployment, specifically so one existing Service sends a small
slice of real traffic to both versions at once, under one name.

**Q: Can I start a Pod just by running kubectl apply -f x.yaml, or do I need to start the Docker containers myself first?**

Just `kubectl apply -f x.yaml`. You never run `docker run` yourself in
a Kubernetes world. The only prerequisite is that the container
**image** already exists somewhere reachable, built once via `docker
build`, then either pushed to a registry the cluster can pull from or,
in this local Minikube setup, loaded directly into the cluster with
`minikube image load -p rag-learning`. Once that image exists, `kubectl apply` is the only
command you run. Kubernetes itself, specifically the `kubelet` on
whichever node the Scheduler picked, pulls that image and starts the
actual container process for you, automatically, using its own
container runtime under the hood. Build the image once; from then on,
`kubectl apply` is the whole interaction.

**Q: How would the configs look different in production at a big organization like SAP or Amadeus? What values would actually change?**

Concretely, tied to what's already in this repo's `k8s/`:

- **Images**: `imagePullPolicy: IfNotPresent` against a locally loaded
  `local-rag-api:kind` tag becomes a real registry image with an
  immutable version tag (`myregistry.company.com/rag-api:1.4.2`, never
  `:latest`), pulled via an `imagePullSecrets` credential for a private
  registry.
- **Replicas**: `minReplicas: 1` becomes realistically 3 or more, often
  spread across multiple availability zones, so losing one zone doesn't
  take the app down.
- **Resource requests and limits**: set from real load-test numbers,
  not the reasonable guesses used here.
- **Secrets**: never a plain committed YAML file like
  `secret-app.example.yaml`. Pulled at deploy time from a real secret
  manager (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault),
  often via a controller that syncs them into Kubernetes Secrets
  automatically.
- **The database**: this is a big one. Most real companies do **not**
  run their own production Postgres inside Kubernetes as a StatefulSet
  at all. They use a managed cloud database (AWS RDS, or Azure Database
  for PostgreSQL) instead, specifically to avoid handling backups,
  failover, and patching themselves. The StatefulSet approach in this
  repo is the right teaching example, not necessarily what a real
  production system would actually run.
- **Storage**: if something does need a PVC, it's backed by a real
  cloud disk (AWS EBS, Azure Disk) via a real `StorageClass`, not
  kind's local-path-provisioner, which only exists for a laptop.
- **Networking**: a real domain name and TLS certificate (usually via
  `cert-manager` and Let's Encrypt, or a corporate certificate
  authority), NetworkPolicy that's actually enforced (a real CNI like
  Calico or Cilium, not kind's kindnet), and often a service mesh
  (Istio, Linkerd) for stricter pod-to-pod encryption and traffic
  control.
- **RBAC**: real access control on who and what can read Secrets or
  deploy changes. This repo has none configured at all.
- **Autoscaling**: usually on a business-relevant custom metric, such
  as queue depth or p95 latency, rather than raw CPU, exactly as
  flagged earlier in this guide.
- **How it gets deployed at all**: nobody runs `kubectl apply` by hand
  against production. A CI/CD pipeline, often GitOps-style via ArgoCD
  or Flux, applies changes automatically when code merges, usually
  through a Helm chart with different values per environment.
- **Multiple environments, multiple clusters**: dev, staging, and prod
  are normally separate namespaces at minimum, often entirely separate
  clusters, and sometimes separate cloud regions for disaster recovery.

**Q: When is minikube actually used, local only or real production too?**

Local only, always. Minikube, like kind, is explicitly a single-machine
learning and testing sandbox. It has real, known limitations that make
it unsuitable for production: no real multi-machine cluster, no real
cloud load balancer, and everything capped by your one laptop's
resources. Real production Kubernetes always runs on either a managed
cloud service (AWS EKS, Azure AKS, Google GKE) or a self-managed
cluster of real servers, never minikube or kind. Their entire value is
being cheap, fast, and disposable, and running on a laptop: exactly the
opposite of what production needs.

**Q: What is DNS? What are "kind" nodes? What's "the same daemon"? How many "kinds" are there?**

One at a time.

- **DNS** (Domain Name System) is the general system, used across the
  whole internet and separately, in a smaller form, inside a Kubernetes
  cluster, that translates a readable name (`google.com`, or inside the
  cluster, `rag-api`) into the actual numeric IP address computers use
  to route traffic. Any time you type a name instead of a raw IP
  address and it works, DNS is the thing doing that translation
  somewhere behind the scenes.
- **"kind nodes"**: kind fakes an entire multi-machine cluster by
  making each pretend "node" actually just be one ordinary Docker
  container running on your one real laptop. A 3-node kind cluster is
  really 3 Docker containers sitting next to each other, each one
  internally running the real Kubernetes node software (kubelet, a
  container runtime, and so on), each pretending to be a separate
  physical machine.
- **"the same daemon"**: Docker Desktop runs one single background
  engine process, the "Docker daemon," that manages every container on
  your machine, no matter what created it. Since kind's fake nodes are
  just regular Docker containers, and the special `host.docker.internal`
  hostname is also something Docker Desktop's own daemon sets up, both
  are managed by that same underlying engine. That's exactly why a DNS
  trick that already works for your Compose containers has a real
  chance of also working for kind's node containers.
- **How many "kinds" are there?** Just one. "kind" is the literal,
  proper name of one specific tool; it stands for "Kubernetes IN
  Docker." It's not a category with several variants. The likely
  source of confusion is that the YAML field `kind:` (as in `kind:
  Deployment`, `kind: Service`) is a completely unrelated use of the
  same English word, meaning "what type of object is this." Two
  unrelated things that happen to share a name.

**Q: Right now Docker Desktop and Ollama are both on my local machine. What would the real production version of this look like, and what if I swapped Ollama for OpenAI or Gemini?**

In real production there's no "Docker Desktop" at all; that's a
developer-laptop product. The cluster runs on real Linux machines,
either your company's own servers or a cloud provider's managed
Kubernetes.

Ollama running natively on the host, outside the cluster, was a
deliberate choice specifically for this local laptop setup, because
Docker Desktop can't easily pass a GPU through into a Linux container.
In a real cloud production setup you'd have two realistic paths.

1. **Self-host the model inside the cluster**, on real GPU-enabled
   nodes. Cloud providers sell GPU node pools specifically for this,
   for example EKS with GPU EC2 instances. Ollama, or something like
   vLLM, would then just be another containerized Deployment, with no
   `ExternalName` or `host.docker.internal` trick needed at all,
   because it's genuinely inside the cluster.
2. **Don't self-host at all**, and call a hosted API (OpenAI,
   Anthropic, Google Gemini) over the public internet instead.
   Operationally simpler, since there's no GPU infrastructure to run or
   patch, but you pay per token and your data leaves your own
   infrastructure.

If you swapped Ollama for a hosted provider, the actual code change is
small. This app's own config already treats the generation provider as
a swap point, and separately already supports OpenAI and Anthropic for
a different purpose, the RAGAS judge. Concretely, on the Kubernetes
side: add the provider's API key to the Secret, never the ConfigMap,
since it's sensitive, point the config at that provider, and delete the
`ollama-host` ExternalName Service entirely. You wouldn't need it,
since a Pod can reach the public internet by default with no special
Kubernetes networking object required, unless a NetworkPolicy egress
rule specifically blocks outbound traffic.

**Q: How does pod scheduling across machines actually work?**

There's a dedicated control-plane component called the **Scheduler**,
and its only job is this: whenever a new Pod exists with no node
assigned to it yet, look at every node's spare capacity (each node's
total capacity minus what's already claimed by `resources.requests` on
every other pod already there), check any placement rules (like the
`ingress-ready=true` node label used in this repo's kind config, or
affinity and taint-toleration rules in a real cluster), and pick
whichever node fits best. It then writes that decision down, "this Pod
belongs to this node," and its job for that Pod is done. From there,
that specific node's own local agent, `kubelet`, takes over and
actually pulls the image and starts the container. It's an automatic,
per-Pod bin-packing decision, made fresh every time, never something
you assign by hand.

**Q: What are an "Ingress controller" and the "control plane"?**

- **Control plane**: the small set of components that make every
  decision and hold all the cluster's desired-state data. This is the
  API server (the front door everything talks to), etcd (the actual
  database storing that state), the Scheduler, and the various
  controllers running their reconciliation loops. It typically runs on
  its own dedicated node or nodes, separate from the "worker" nodes
  that actually run your application containers.
- **Ingress controller**: an `Ingress` object, by itself, is just a
  rule written down. It does nothing on its own. Something has to
  actually read that rule and make it real, and that's the Ingress
  controller: an ordinary running program, a Deployment of pods like
  ingress-nginx, that watches for `Ingress` objects and does the actual
  HTTP routing they describe. Same pattern as NetworkPolicy: the object
  alone is inert data, and a separate, real, running piece of software
  has to exist and implement it.

**Q: A Kubernetes Secret is encoded, not encrypted. What's the actual difference?**

- **Encoding** (base64, here) is a reversible transformation that needs
  **no secret key at all** to undo. Anyone, with zero special access,
  can turn it back into the original text with one command or even a
  random website. Its only real purpose is letting arbitrary binary or
  special-character data be written safely as plain text inside a YAML
  file. It was never designed to hide anything from anyone.
- **Encryption** requires a secret key to reverse. Without that key,
  the scrambled result is meant to be practically unreadable no matter
  how much effort someone puts in. That's what actually keeps something
  hidden from someone who isn't supposed to see it.

So a Secret's real protection is limited. It's a distinct object type
that tooling and access rules can treat differently from a ConfigMap if
you set that up, and a real cluster typically adds genuine encryption
underneath it, encrypting the actual etcd database Secrets are stored
in, but that's a separate feature you have to explicitly turn on, not
something a Secret gives you automatically just by existing.

**Q: What do I need to start, to actually see and play with Kubernetes and its commands? How do I proceed?**

The working path on this machine is Minikube, not kind. The repo keeps
the kind 3-node topology as a reference, but Minikube is the cluster
that actually ran the exercises.

1. Start the local cluster: `minikube start -p rag-learning`.
2. Confirm it: `kubectl get nodes -o wide` and `kubectl get pods -A`.
3. Apply the Minikube/base manifests incrementally rather than
   everything at once, then inspect Pods, StatefulSets, Services, PVCs,
   and EndpointSlices after each stage.
4. Work through the failure drills: scale `rag-api`, delete a Pod, stop
   Postgres and watch readiness, perform a rolling restart, deploy a
   bad image and roll it back.
5. Use the retained kind material later for the multi-node-only
   exercises such as node placement and node failure when the host
   runtime supports it.

The important thing is not which local tool creates the cluster. The
Kubernetes objects and control-loop behavior you are practicing are the
same API concepts.

**Q: In configmap-app.yaml, shouldn't POSTGRES_DB and POSTGRES_USER be the same as what's in .env?**

They already are. `.env.example` has `POSTGRES_USER=rag` and
`POSTGRES_DB=ragdb`. `k8s/base/configmap-app.yaml` has `POSTGRES_DB:
"ragdb"` and `POSTGRES_USER: "rag"`: identical strings, on purpose (the
password, which is sensitive, lives in `secret-app.example.yaml`
instead, also set to `"rag"` to match). Two things from `.env.example`
are deliberately not copied over, and it's worth knowing why.

- **`POSTGRES_HOST_PORT=15987`** only exists so Postgres is reachable
  from your Windows host at `localhost:15987`, bypassing the container
  network entirely. Inside the cluster, `rag-api` never reaches
  Postgres through a host port at all. It talks straight to the
  `postgres` Service on its real port, `5432`. There's nothing to map
  to a different host port unless you specifically want to `kubectl
  port-forward` in yourself.
- **`DATABASE_URL`'s hostname** differs on purpose. `.env` says
  `localhost:15987`; the Secret says
  `postgres.rag.svc.cluster.local:5432`. Same credentials, same
  database name, just a different address, because "how do I reach
  Postgres" is a different question from inside the cluster (a Service
  DNS name) than it is from your own laptop (a mapped host port).

**Q: In frontend-deployment.yaml, why does httpGet look empty for both probes? Which value points at the frontend Docker image, and how does it actually know what to run? Why two separate files, Deployment and Service?**

It isn't empty. It's easy to misread because the value is a single
character.

```yaml
readinessProbe:
  httpGet:
    path: /
    port: 80
```

`path: /` means "just hit the root URL, whatever's there." That's the
right check for the frontend specifically because it's a plain nginx
server handing back `index.html`. There's no dedicated `/health`
endpoint on this side (that only exists on the API), so "did I get any
successful response from `/`" is the honest, simplest thing to check.

The image is named here.

```yaml
# base/reference manifest
image: local-rag-frontend:kind

# Minikube overlay resolves this to the locally loaded Minikube tag
```

That's the entire connection. It does **not** point at
`frontend/Dockerfile` or `docker-compose.frontend.yml` directly; neither
is ever read by Kubernetes. What actually happens, per `k8s/README.md`,
is that you build the frontend image locally, for example `docker build
-t local-rag-frontend:minikube ./frontend`. Then `minikube image load -p
rag-learning local-rag-frontend:minikube` makes that image available to
the Minikube node without using a remote registry. The Minikube overlay
points the Deployment at that tag. Kubernetes still only sees an image
reference; it never reads the Dockerfile itself. By the time Kubernetes
sees that line, the image already fully exists. The Dockerfile was just
the recipe; `docker build` is what actually followed it.

Two files exist because a Deployment and a Service do genuinely
different jobs, kept as separate object types on purpose.

- **`frontend-deployment.yaml`** answers "what should be running":
  which image, how many copies, what health checks, how much CPU and
  memory. This is the thing that actually creates real Pod processes.
- **`frontend-service.yaml`** answers "how does anything else find and
  reach it": a stable name (`frontend`) that stays constant no matter
  how many times the Deployment recreates pods underneath it.

If you deleted `frontend-service.yaml` right now, the frontend pod
would keep running fine. It would just become unreachable by name from
anything else in the cluster, including the Ingress, which routes to
`frontend` the Service, never to a pod directly.

**Q: So the pod's configuration lives in the Deployment file, and the pod's name lives in the Service file, which everything else uses to refer to that pod?**

Close, but one part needs correcting. **Pod names never live in the
Service file at all, and they're not something a human writes
anywhere.** Kubernetes generates them automatically, something like
`frontend-7d8f9c6b5-abcde`, and a new, differently named pod is created
every single time one is recreated for any reason (a crash, a rolling
update, a `kubectl delete pod`). Nothing in the cluster is supposed to
depend on that name staying the same.

What actually happens is closer to this.

- The **Deployment** configures the pods (image, resources, probes) and
  stamps a **label** on them (`app: frontend`), not a name, a label.
- The **Service** has its own stable name (`frontend`) and never refers
  to individual pods by name either. It keeps a live list of pod IPs
  that currently match its label selector, and other things reach it by
  the Service's own name (`http://frontend`, or `http://rag-api:8000`
  from inside nginx's config), never by any individual pod's name.

So: pod configuration is the Deployment's job. A stable name other
things can actually rely on is the Service's job. Individual pod names
are internal, auto-generated, and disposable; nothing is meant to
reference them directly.

**Q: Can the app label and the selector be different? Also, why type: ClusterIP, and what other types exist?**

Two different questions bundled together.

**Labels vs. selector.** Within one Deployment, its own
`spec.selector` and its pod template's `spec.template.metadata.labels`
are required to match. This isn't just convention; the API server
actively rejects a Deployment where they don't, with a real validation
error on `kubectl apply`. What can differ freely is which label key you
use at all. `app` isn't a reserved or special name; it's pure
convention. You could call it `tier`, `component`, or anything else, as
long as the same key and value are used consistently by whatever needs
to find those pods. A Service's selector also isn't tied to matching
exactly one Deployment's labels. The earlier canary example is real:
two separate Deployments can each stamp the same `app: rag-api` label
on their pods, and one single Service picks up both groups at once,
with no need to know that two different Deployments are involved.

**Why ClusterIP here specifically.** Nothing outside the cluster needs
to reach the `frontend` Service directly. The real front door for
outside traffic is the Ingress plus the ingress-nginx controller. The
`frontend` Service only ever needs to be reachable from inside the
cluster, by the Ingress controller's own pods, which is exactly what
`ClusterIP` provides: an address that only works inside the cluster's
own network.

The other `type` values, and when you'd actually reach for them:

| `type:` | What it does | When you'd use it |
|---|---|---|
| `ClusterIP` (default) | Internal-only virtual IP | Almost everything: anything only other in-cluster things need to reach |
| `NodePort` | Same, plus a fixed port opened on every node's real IP | Quick or manual external access without an Ingress, common on bare metal or local testing |
| `LoadBalancer` | Same, plus your cloud provider provisions a real external load balancer with a real public IP | The actual entry point a cloud Ingress controller usually sits behind |
| `ExternalName` | Pure DNS indirection to something outside the cluster entirely | Reaching something that isn't a pod at all, like `ollama-host` reaching your native Ollama process |

**Q: If, in a real setup, images are pushed by CI/CD to a registry like Harbor, what changes in the image reference?**

Yes, in two ways.

1. **The value itself** becomes the full registry path plus an
   immutable version tag, something like `image:
   harbor.mycompany.com/rag-project/rag-frontend:1.4.2`, or for maximum
   reproducibility, an exact content digest like
   `...@sha256:abc123...` instead of a tag at all. Never `:latest` in a
   real deployment. You want to know exactly which build is running,
   and be able to roll back to a specific one.
2. **A pull credential is usually needed too.** This Minikube setup
   needs no registry authentication because the image can be loaded
   directly into the node with `minikube image load`. A private
   registry like Harbor requires authentication to pull from, so the
   pod spec would also need an `imagePullSecrets` entry pointing at a
   Kubernetes Secret holding Harbor's login credentials.

The part that usually isn't a human hand-editing this YAML file every
release is normally automated. `k8s/base/kustomization.yaml` already
has the exact hook a real pipeline would use for this.

```yaml
images:
  - name: local-rag-frontend
    newTag: kind
```

A CI/CD pipeline would run something like `kustomize edit set image
local-rag-frontend=harbor.mycompany.com/rag-project/rag-frontend:1.4.2`
as one step of the deploy, rewriting just that tag, rather than anyone
manually editing the Deployment file itself on every release.

**Q: How does the pod know where DATABASE_URL or COHERE_API_KEY come from? Is there actually something called rag-secret defined somewhere? And what are SIGTERM and SIGKILL?**

There are two separate lookups chained together here, worth seeing as
two distinct steps.

**Lookup 1: Kubernetes puts a value into the container's environment**,
in `rag-api-deployment.yaml`.
```yaml
env:
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: rag-secret      # which Secret object to read
        key: DATABASE_URL     # which key inside that Secret
```
This says: before starting this container, set an environment variable
named `DATABASE_URL` inside it, using whatever value sits under the key
`DATABASE_URL` in the Secret object named `rag-secret`. By the time the
Python process actually starts, `DATABASE_URL` is just an ordinary
environment variable. Kubernetes' involvement is already finished.

**Lookup 2: the app decides which variable name to read**, and this has
nothing to do with Kubernetes. It's `config/default.yaml`'s existing
`*_env_var` convention (`vectorstore.connection_env_var: DATABASE_URL`,
`reranker.cohere.api_key_env_var: COHERE_API_KEY`). `config.py` reads
these fields and calls `os.environ.get("DATABASE_URL")` at runtime, the
same code path whether running natively, in Compose, or here.

The two sides meet only because the **names match**. Nothing enforces
that automatically. Rename the field in the manifest to `DB_URL` and
the app would silently fail to find its database, since `config.py`
would still be looking for `DATABASE_URL`.

And yes, `rag-secret` is a real object, defined in full in
`k8s/base/secret-app.example.yaml`.
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secret
  namespace: rag
type: Opaque
stringData:
  POSTGRES_PASSWORD: "rag"
  DATABASE_URL: "postgresql://rag:rag@postgres.rag.svc.cluster.local:5432/ragdb"
  JWT_HS256_SECRET: "dev-only-insecure-jwt-signing-secret-CHANGE-ME"
  COHERE_API_KEY: ""
```
`metadata.name: rag-secret` is the exact string every
`secretKeyRef.name` points at. It's listed in
`k8s/base/kustomization.yaml`'s resources, so `kubectl apply -k
k8s/base` creates it. If it hadn't been applied yet, the pod would fail
to start at all, with `CreateContainerConfigError` (Kubernetes can't
resolve a `secretKeyRef` pointing at a Secret that doesn't exist).

`SIGTERM` and `SIGKILL` are operating-system signals, the actual
low-level way any process gets told to stop, which Kubernetes uses
under the hood.

- **`SIGTERM`** means "please shut down." It's a polite request the
  process can catch and react to: finish the request it's mid-way
  through, close connections cleanly, then exit on its own terms. This
  is what Kubernetes sends first, and it's exactly what uvicorn is
  built to handle.
- **`SIGKILL`** means "die, right now, no say in it." The process
  cannot catch, ignore, or react to this at all; the operating system
  just ends it instantly, wherever it happened to be. No cleanup. A
  last resort, because it can leave things in a bad state.

The actual sequence in `rag-api-deployment.yaml`: Kubernetes decides to
remove the pod. The `preStop` hook runs (`sleep 5`, giving Service and
Endpoint removal time to propagate). Then **`SIGTERM`** is sent, and
uvicorn finishes any in-flight request before exiting. If the process is
still running once `terminationGracePeriodSeconds: 40` has fully
elapsed, Kubernetes gives up and sends **`SIGKILL`** unconditionally.
That's why the grace period has to comfortably exceed the `preStop`
delay plus real shutdown time. Set it too low and a well-behaved
graceful shutdown gets cut off by `SIGKILL` before it finishes,
defeating the entire point of catching `SIGTERM` in the first place.

**Q: How would this proceed if I need to pass different config values, say config/production.yaml instead of config/default.yaml?**

`config/production.yaml` doesn't need adding to the image separately.
The Dockerfile already copies the whole `config/` directory in
(`COPY --chown=rag:rag config ./config`), so `production.yaml` is
already sitting inside `local-rag-api:kind`, unused. The only thing
deciding which file loads is one environment variable, read in
`api/deps.py:get_config()`.

```python
def get_config() -> AppConfig:
    override_path = os.environ.get("RAG_CONFIG_PATH")
    if override_path:
        return load_config(override_path)
    return load_config()   # falls back to config/default.yaml
```

`k8s/base/configmap-app.yaml` currently sets `RAG_CONFIG_PATH: ""`.
Empty string is falsy in Python, so it always falls through to
`default.yaml`. Switching profiles means changing that one value to
`"/app/config/production.yaml"` (the in-container path, since `/app` is
the Dockerfile's `WORKDIR`).

**The part that's easy to miss: editing the ConfigMap alone doesn't
reach a running pod.** `envFrom` and `env` values are injected once, at
container start, not as a live-reloaded file. The real sequence is:
```bash
kubectl apply -f k8s/base/configmap-app.yaml
kubectl rollout restart deployment/rag-api -n rag   # forces new pods to pick it up
```
The second command matters, and forgetting it is a genuinely common
real-world mistake. A ConfigMap mounted as a volume, meaning a file on
disk, does get updated in a running container after a short delay, but
one consumed as an env var, like here, never does, no matter how it's
mounted. Env vars are copied in once, at process start.

**Switching also drags in real consequences**, from the actual diff
between the two files.

- `security.auth.enabled: true` means every request now needs a real
  `Authorization: Bearer <jwt>` header. The Secret already has a value
  for `JWT_HS256_SECRET`, so nothing new needs adding, but a plain
  no-auth-header `curl` (like the README's smoke test) would now get a
  401. Mint a token first via `scripts/issue_dev_token.py`.
- `security.rate_limit.enabled: true` means 60 requests per minute per
  tenant, actually enforced now.
- `observability.tracing.enabled: true` needs a real OTLP endpoint or
  spans vanish silently. The ConfigMap already points
  `OTEL_EXPORTER_OTLP_ENDPOINT` at Jaeger's in-cluster address, so this
  only really works once `kubectl apply -f k8s/observability/` has also
  been run. Production config turns an optional layer into a real
  dependency.
- `generation.model_name: qwen2.5:3b`, not `default.yaml`'s `1.5b`, is
  a native-Ollama-side requirement with nothing to do with Kubernetes.
  Run `ollama pull qwen2.5:3b` on the host first.
- `judge.provider: openai` needs no action for the running API itself.
  Production `answer()` never calls a hosted LLM at all; only the
  separate, optional `run_ragas_eval.py` tooling would ever touch this.

**The better long-term pattern**, and a real interview-worthy point, is
that hand-editing one ConfigMap value works for a one-off experiment,
but doesn't scale to "dev permanently points at `default.yaml`, prod
permanently points at `production.yaml`, without them fighting each
other." The standard answer is kustomize overlays, building on the
`k8s/base/` that already exists.

```
k8s/
  base/            (unchanged, the shared foundation)
  overlays/
    dev/
      kustomization.yaml   # bases: [../../base], patches RAG_CONFIG_PATH: ""
    prod/
      kustomization.yaml   # bases: [../../base], patches RAG_CONFIG_PATH: "/app/config/production.yaml"
```

Each overlay is a small `kustomization.yaml` saying "start from
`k8s/base`, then patch just this one field." Deploy with `kubectl apply
-k k8s/overlays/prod` instead of `k8s/base` directly, and the two
environments' manifests never duplicate or drift apart on everything
they do share, which is almost everything.

**Q: Why doesn't editing a ConfigMap update an already-running pod?**

It depends entirely on how that pod consumes the ConfigMap, and the two
paths behave completely differently. This is worth knowing as a
standalone fact, not just something that came up while switching config
profiles.

**As an environment variable** (`envFrom` or
`env.valueFrom.configMapKeyRef`, the way every value in `rag-config`
reaches `rag-api` today), the value is copied into the container's
environment exactly once, at process start. There is no mechanism
watching for a change afterward. Editing the ConfigMap object changes
what a future pod would receive, and does nothing at all to a pod
that's already running. This is fundamental to how Unix processes work,
not a Kubernetes limitation. An environment variable was never
something another process, or another version of `kubectl`, can reach
in and mutate after the fact.

**As a mounted volume** (a ConfigMap's key/value pairs written out as
actual files on disk inside the container, via `volumes` and
`volumeMounts` instead of `env`), this genuinely is different. The
kubelet does watch for ConfigMap changes and updates the mounted file's
content in the running container, typically within about a minute,
governed by kubelet's sync period, not instant. But that only gets you
a fresh file. Whether the application actually notices and reacts to
the file changing is entirely up to the application itself. If it read
the file once at startup and cached the value in memory, as most apps,
including this one, do for their config, an updated file sitting on
disk changes nothing until the process restarts and reads it again. An
app has to be specifically written to watch the file, for example with
`inotify`, or poll it periodically, for a mounted ConfigMap's
live-update behavior to actually reach the running code.

So the complete version of "does a ConfigMap change reach a running
pod" is: **never**, if consumed as an env var, and **only the file on
disk updates**, if mounted as a volume, and even then only if the app
is written to notice. Either way, for an app like this one that reads
config once at startup, the fix is the same regardless of which
mechanism was used: `kubectl rollout restart deployment/rag-api -n rag`
to force fresh pods that read the new value from scratch.

**Q: Do readinessProbe and livenessProbe use the same endpoint?**

Not anymore, and that change is deliberate. The original manifest
pointed both at `/health`, but the hands-on tests showed that the
endpoint's semantics were wrong for Kubernetes lifecycle decisions.

```text
startupProbe  -> /livez
livenessProbe -> /livez
readinessProbe -> /readyz
```

`/livez` answers only "is the application process alive?" and avoids
failing because Postgres or Ollama is temporarily unavailable. `/readyz`
answers "can this instance serve normal traffic correctly right now?"
and can return 503 on a critical dependency outage. They can use the
same port, `8000`, because they are simply different HTTP requests to
different paths on the same FastAPI server.

**Q: Whenever a service depends on another service being ready first, do you always need an initContainer plus busybox?**

No, that's one option, not a rule, and `busybox` specifically isn't
required either.

Kubernetes has no native concept of "wait for another Deployment or
Service to be ready," the way Compose's `depends_on: condition:
service_healthy` does. `initContainers` are the closest idiomatic tool
for expressing "block until X" inside one Pod, reached for here
specifically to mirror that existing Compose behavior as directly as
Kubernetes allows, not because it's the only way. A far more common
real-world alternative is to build retry logic into the app itself:
connect to the database, retry with backoff on failure, don't crash.
Plenty of production apps skip init containers entirely this way; the
app just logs some connection errors during its own retry window, and
the readinessProbe naturally keeps it out of traffic until it actually
succeeds. `busybox` isn't special either. It's used purely because it's
a tiny, generic Linux image that happens to bundle `nc` (netcat), which
is all this particular check needs. Any image with a way to test a TCP
connection would work.

**Q: If I change the app label in the Deployment file, does that change the running container's name? I assumed the container's name came from the label, not metadata.name.**

Neither, actually. There are four different names in this one file, and
none of them derive from each other.

```yaml
metadata:
  name: rag-api              # (1) the Deployment object's own name
...
  template:
    metadata:
      labels:
        app: rag-api          # (2) an arbitrary label, just a tag
    spec:
      containers:
        - name: rag-api        # (3) the container's actual name
```

Plus a fourth: Kubernetes auto-generates the running **Pod's** name,
something like `rag-api-7d8f9c6b5-abcde`, from a hash of the Deployment
name and the pod template. That's neither the label nor the container
name.

**Item (3), `containers[0].name`, is what actually determines "the
container's name"**: the string you'd pass to `kubectl logs <pod> -c
rag-api`, or see in `kubectl describe pod`. It happens to be the
literal string `"rag-api"` here, same as everything else, but that's
coincidence, not derivation. Renaming it to `containers[0].name:
api-process` tomorrow would change nothing else about the file.

Changing only the label without updating `spec.selector.matchLabels` to
match produces two different failure modes worth knowing apart. The
Deployment itself is likely **rejected outright** on `kubectl apply`,
because a Deployment's `spec.selector` is immutable after creation and
must match the pod template's labels: a loud, immediate error. But if
you changed the label and the selector together, and only forgot the
**Service**'s own selector (`rag-api-service.yaml`'s `selector: { app:
rag-api }`), that fails silently. The Service wouldn't error at all; it
would just quietly stop matching any pods, and `kubectl get endpoints
rag-api -n rag` would show an empty list. That loud-versus-silent
distinction is exactly why label mismatches are such a common
real-world debugging trap.

**Q: replicas: 1, does that mean one ReplicaSet with one Pod holding a variable number of containers? What commands show all of this?**

One ReplicaSet, managed automatically by the Deployment, maintains
exactly one Pod, which runs exactly one container (`containers` has one
entry, `rag-api`), plus the `wait-for-postgres` init container, which
already ran to completion and exited before the main container ever
started, so it doesn't count toward "currently running containers"
afterward. `replicas` controls how many full copies of the whole Pod
template exist. It says nothing about how many containers sit inside
one Pod; that's fixed separately by how many entries are in the
`containers:` list.

Commands that show this concretely:

```bash
kubectl get deployments -n rag
# NAME      READY   UP-TO-DATE   AVAILABLE
# rag-api   1/1     1            1

kubectl get statefulsets -n rag
# NAME       READY
# postgres   1/1

kubectl get pods -n rag
# NAME                       READY   STATUS
# rag-api-7d8f9c6b5-abcde    1/1     Running     # READY = ready containers / total containers in that pod
# postgres-0                 1/1     Running

kubectl get pods -n rag -o wide          # + node placement, pod IP

kubectl describe pod rag-api-7d8f9c6b5-abcde -n rag
# full detail: init containers (separately listed, with exit status),
# main containers, images, probe results, resource requests/limits, events

kubectl get all -n rag                   # one broad view: Deployments, StatefulSets,
                                          # ReplicaSets, Pods, Services together
```

**Q: Give me an example of something being live but not ready, and the reverse.**

**Live but not ready** is a normal and useful state. We demonstrated it
directly: scale Postgres to zero. FastAPI can still answer `/livez`
with 200, so the process is alive, while `/readyz` returns 503 because
a critical dependency is unavailable. Kubernetes keeps the container
running but removes that Pod from normal Service traffic.

**Ready but not live** should not be a stable intended state. Because
the probes run on independent schedules, there can be a brief stale
window where the last readiness result is still true just after the
process has become unhealthy and before the next liveness/readiness
checks update status. If liveness keeps failing past its threshold,
kubelet restarts the container.

**Q: What is kubelet?**

The agent process running on **every single node**, one per machine,
worker or control-plane. Its job is to watch the API server for any Pod
that's been assigned to its own node, a decision the Scheduler already
made, and actually make that real: pull the image, start the container
via the node's container runtime (containerd, here), and then keep
watching it. Crucially, **kubelet is the thing literally performing the
HTTP GET for both probes**. It's not some separate control-plane
service reaching out; it's the same local agent that started the
container in the first place, polling it. It also reports status back
up to the API server, which is how `kubectl get pods` knows anything at
all, and enforces the container's resource limits. If the Scheduler
decides which machine a Pod runs on, kubelet is the thing on that
machine that actually makes it run and keeps checking on it.

**Q: Can you actually see the control plane and the Scheduler while a cluster is running?**

It depends entirely on the kind of cluster. On a real managed cluster
(EKS, GKE, AKS), the control plane is hidden. The cloud provider runs
it as a managed service, and there's no way to `kubectl get` into its
internals at all.

On this local Minikube cluster, and on many self-managed kubeadm-style
clusters, the control-plane components are visible as static Pods in
`kube-system`.

```bash
kubectl get pods -n kube-system
```

In the current cluster you can see, among others: `kube-apiserver-rag-learning` (the API server itself),
`etcd-...` (the datastore), `kube-scheduler-...` (yes, the Scheduler,
as a real pod), `kube-controller-manager-...` (the Deployment,
ReplicaSet, and Node controllers, all bundled in here), `coredns-...`
(cluster DNS), `kube-proxy-...` (the per-node component writing the
iptables rules), and `kindnet-...` (the CNI plugin). You can even run
`kubectl logs kube-scheduler-<node> -n kube-system` and watch its real
scheduling decisions as they happen.

One accuracy note: these specific ones are technically "static pods."
kubelet starts them directly from manifest files on the control-plane
node's own disk, not via the normal API-server-driven scheduling path,
but they still show up and behave like ordinary pods for everything
you'd want to do with them.

**Q: If a file is uploaded via POST /ingest and the pod dies, is anything shared or recovered?**

No, and this is exactly the gap already flagged as a real limitation of
this specific deployment. `rag-api` is a **Deployment**, with no PVC
attached to `/app/data/uploads` at all. Any file written there via a
real ingest request lives only on that one specific pod's ephemeral,
container-local filesystem. The moment that pod is deleted or
recreated, for any reason at all, whether a rolling update, a node
failure, or even a plain `kubectl delete pod`, the replacement gets a
completely fresh, empty filesystem. The uploaded file isn't moved or
recovered anywhere; it's just gone. With more than one replica, it's
worse even while nothing dies: a file landing on pod A is invisible to
pods B and C the entire time, since there's no shared storage between
them at all. The fix, not implemented here, would be a
`PersistentVolumeClaim` mounted at that path: `ReadWriteOnce` if only
"survive this one pod's restarts" matters, or `ReadWriteMany` (or
object storage like S3) if it needs to work correctly across multiple
replicas.

**Q: Does a replacement pod have to land on a different node than the one it died on?**

No. The Scheduler makes a completely fresh placement decision every
time a new pod object is created, based on whatever capacity is
available across all nodes at that exact moment. It has no memory of
"this pod used to live here," and no rule against landing back on the
same node either. Whether it moves or not just depends on what's free
right now.

There's a related gap worth naming plainly here. Right now, with
`replicas: 3`, the Scheduler is allowed to put all three `rag-api` pods
on the same single node; nothing in this deployment prevents that. If
it does, and that one node crashes, all three pods go down together.
People often assume that running 3 replicas automatically protects them
if one machine fails, but that's only true if the 3 replicas actually
end up spread across different machines. If they all land on one
machine by chance, having 3 replicas gives zero extra protection over
having 1, for exactly the node-failure scenario replicas are usually
assumed to guard against. The real fix is a feature called Pod
Topology Spread Constraints, or Pod Anti-Affinity rules: a way to
explicitly tell the Scheduler "don't put more than one of these on the
same node." Neither is configured anywhere in this deployment.

**Q: Referring to init-db-job.yaml, what is a Job?**

A `Job` is a Kubernetes object for work that should run once and then
stop, not stay running forever. That's the key thing that makes it
different from a Deployment. A Deployment treats a container exiting as
a failure to be restarted forever; a Job treats a container exiting
successfully as the whole point: "run this, and once it finishes,
you're done."

Concretely, `init-db-job.yaml` runs `python /scripts/init_db.py`, the
one-time database schema setup, creating tables and the pgvector
extension. That's a perfect Job use case: it needs to happen exactly
once when setting things up, not continuously. `backoffLimit: 3` means
"if it fails, retry up to 3 times before giving up." `restartPolicy:
Never` at the pod level, a required setting for Jobs, means a failed
attempt gets a brand-new pod, not a restarted container in the same
pod.

**Q: What is a PVC?**

`PersistentVolumeClaim`: a **request** for durable disk storage that
outlives any single pod. Think of it this way: the pod says "I need 1Gi
of storage that will still be there if I get deleted and recreated,"
and the PVC is the object representing that request being granted. In
this repo, `postgres-statefulset.yaml`'s `volumeClaimTemplates` creates
one PVC per Postgres pod (`pgdata-postgres-0`), backed by an actual disk
that kind provisions automatically. When `postgres-0` gets deleted and
recreated, the replacement pod reattaches to that same PVC, and the
data survives, because the PVC and the real storage behind it are a
separate object from the pod itself, with their own independent
lifetime.

Contrast this with `rag-api`, which has **no** PVC at all. That's
exactly why uploaded files don't survive a pod being recreated; no PVC
means nothing does.

**Q: Is there anything like Docker's layer caching in Kubernetes?**

Two different things worth separating here, because the honest answer
is "not exactly, but something related and relevant does apply."

**Build-time layer caching**, what `docker build` does, doesn't apply
to Kubernetes at all, because Kubernetes never builds images. It only
ever runs an image that's already fully built. That build step
(`docker build -t local-rag-api:kind .`) happens completely separately,
before Kubernetes is even involved, using whatever caching Docker
itself does. Nothing about running the result in Kubernetes changes
that.

**What genuinely does carry over**: container images are stored as
layers, which is just the OCI image format, independent of Docker
specifically, and a node's container runtime (containerd, in this local
cluster) keeps its own local store of layers it's already pulled. If two
images happen to share a base layer, the second pull can reuse what's
already on disk instead of re-downloading it. The practical, everyday
version of this in this repo is `imagePullPolicy: IfNotPresent`, used
throughout every manifest here. If the exact image and tag are already
sitting in a node's local store, say from an earlier `minikube image
load`, Kubernetes skips pulling anything at all. That's the
closest thing to a "cache hit" you'll actually experience in this
setup: not build caching, but "don't even try to fetch something you
already have."

**Q: In postgres-statefulset.yaml I see kind: Service and then kind: StatefulSet. How do you decide which one a component should be? The ollama one also uses kind: Service, not a Deployment.**

`Service` and `StatefulSet`/`Deployment` are not alternatives you choose
between. They answer two completely different questions, and a real
component almost always needs one of each.

- **Deployment / StatefulSet / Job / DaemonSet** answers "what actually
  runs." It defines the container, image, command, and replicas: the
  thing that gets scheduled onto a node and actually runs a process.
- **Service** answers "how does anything reach it." It's a stable
  name/address that routes traffic to whichever Pods currently match a
  label selector, so nothing has to hardcode a Pod's IP, which changes
  every time a Pod is recreated.

So in `postgres-statefulset.yaml`, you're not choosing between `Service`
and `StatefulSet`. You're seeing two separate objects in the same file,
joined by a `---` separator, that together make up "Postgres." That's a
style choice, colocating tightly related objects in one file for
readability, not a rule; splitting them into two files would behave
identically. The `Service` gives Postgres a stable DNS name; the
`StatefulSet` is what actually runs the `postgres` container.

Ollama is the genuine exception worth understanding, not the norm. Its
Service (`ollama-external-service.yaml`) is `type: ExternalName`, pure
DNS aliasing: no selector, no Pods, no ClusterIP. Ollama runs natively
on the Windows host, completely outside Kubernetes, so there's nothing
to run and no Deployment or StatefulSet at all. The Service exists only
so things inside the cluster can say "reach `ollama-host`" using a
normal-looking Kubernetes name that resolves to `host.docker.internal`,
the real address of the process running outside the cluster.

The reverse case exists too: `k8s/jobs/init-db-job.yaml` is a workload
with no Service, since nothing needs to reach the schema-init script by
a stable name while it runs once and exits.

The decision, applied to every object in `k8s/base/`:

| Component | Needs a workload? | Which kind, and why | Needs a Service? | Why |
|---|---|---|---|---|
| `rag-api` | Yes | `Deployment`, interchangeable stateless replicas | Yes | Other pods reach it by name (`rag-api.rag.svc.cluster.local`) |
| `frontend` | Yes | `Deployment`, same reasoning | Yes | Ingress reaches it by name |
| `postgres` | Yes | `StatefulSet`, needs stable identity and its own durable disk per replica | Yes (headless) | Stable per-pod DNS (`postgres-0....`), not load-balanced |
| `init-db` | Yes | `Job`, runs once to completion, not kept alive | No | Nothing calls it by name; it's not a long-lived service |
| `ollama-host` | No, it's outside the cluster | (none) | Yes (`ExternalName`) | Just needs a name that resolves to `host.docker.internal` |

So the two questions for any new component are: does something need to
actually run as a process in the cluster, and if so what kind of
controller fits its lifecycle (stateless and replaceable means
Deployment, needs stable identity and storage means StatefulSet, runs
once means Job, one-per-node means DaemonSet), and does anything need to
reach it by a stable name, in which case add a Service regardless of
what answered the first question, or even with no answer to the first
question at all, as Ollama shows.

**Q: When people say "this cluster runs 3 controllers," what does that actually mean, and is total Pods in the cluster always equal to some single replicas number?**

"Controller" here just means a workload object that manages Pods. In
this repo that's the `frontend` Deployment, the `rag-api` Deployment,
and the `postgres` StatefulSet: two Deployments and one StatefulSet,
three separate long-running workloads. Each currently has its own
`replicas: 1`, so together they produce three application Pods right
now. Jobs like `init-db` can add more Pods temporarily, and if you also
apply `k8s/observability/`, Prometheus, Grafana, and Jaeger add three
more Deployments, each with their own replica count.

There's no single global "replicas" number for a cluster. Each workload
owns its own count.

```text
frontend Deployment      replicas: 1  -> 1 Pod
rag-api Deployment       replicas: 1  -> 1 Pod
postgres StatefulSet     replicas: 1  -> 1 Pod
                                        ------
                                        3 Pods (core app, right now)
```

And "number of Pods equals replicas" is only exactly true in a steady
state, not always. During a rolling update, `rag-api-deployment.yaml`'s
`maxSurge: 1` lets Kubernetes briefly run one Pod more than `replicas`,
the new version starting up before the old one is removed, so a
Deployment can genuinely have more Pods than its `replicas` value for a
few seconds mid-rollout. "Pods equal replicas" is the right mental
model for a resting cluster, not an absolute invariant at every instant.

**Q: Is HPA attached to a Node? How does it relate to the Scheduler deciding where Pods run?**

No. `k8s/base/rag-api-hpa.yaml` targets a Deployment
(`scaleTargetRef: kind: Deployment, name: rag-api`), never a Node. It
answers exactly one question: how many Pods should exist. It has no
concept of nodes at all.

HPA decides "scale rag-api to 4 replicas" and writes that number onto
the Deployment. The Deployment's ReplicaSet then creates or deletes
Pods to match. A completely separate component, the Scheduler, then
independently decides which node each of those Pods actually lands on,
based on capacity at that moment. HPA never tells the Scheduler where
to place anything, and the Scheduler has no opinion on how many Pods
should exist.

```text
HPA         -> "how many Pods should exist" (targets a Deployment/StatefulSet)
Scheduler   -> "which node does each Pod land on" (acts on every Pod, HPA-created or not)
```

If a cluster runs out of node capacity while HPA is asking for more
Pods, those extra Pods just sit `Pending` until capacity frees up or a
node is added. HPA itself never adds nodes; that's a separate mechanism
entirely, a cluster or node autoscaler, not something this repo
configures.

**Q: Isn't Ingress the right fit for reaching Ollama, since it's about routing traffic?**

No, because they route traffic in opposite directions. Ingress
(`k8s/base/ingress.yaml`) is for traffic coming into the cluster from
outside: a browser hits `rag.local`, Ingress reads the hostname and
path and forwards to the `frontend` Service inside the cluster.
Ollama's `ExternalName` Service is the mirror image: traffic already
inside the cluster, a `rag-api` Pod, needs to reach something outside
it, Ollama, running natively on the Windows host.

```text
Ingress:        outside -> Ingress -> Service -> Pod    (inbound)
ExternalName:    Pod -> Service -> outside hostname      (outbound)
```

Ingress has no mechanism for "let a Pod call out to something" at all;
that's not what it's built for. `ExternalName` is the right tool
specifically because it's pure DNS aliasing with no inbound routing
involved. It just makes `ollama-host` resolve to `host.docker.internal`
for anything inside the cluster that looks it up.

**Q: Can one Kubernetes YAML object have more than one kind, or is it one file, one kind?**

One object can only ever have exactly one `kind`. An object cannot be
both `kind: Deployment` and `kind: Service` at once; that isn't valid
in any Kubernetes sense.

What can hold more than one thing is a file. A single `.yaml` file can
contain several independent objects, each with its own
`apiVersion`/`kind`/`metadata`, separated by a bare `---` line.
`k8s/base/postgres-statefulset.yaml` is exactly this: one file, two
objects (a `Service` and a `StatefulSet`), stacked with `---` between
them. `kubectl apply -f` reads a `---`-separated file as multiple
independent objects to create, not one object with two kinds. This
repo's own `k8s/base/kustomization.yaml` lists that one filename once,
and both kustomize and `kubectl` apply everything inside it correctly,
because the split into two objects happens at parse time, before either
tool cares about file boundaries at all.

## Core clarifications worth memorizing

These are the distinctions that cause the most confusion in interviews.

**Dockerfile vs Docker Compose vs Kubernetes**

```text
Dockerfile
-> builds one container image

Docker Compose
-> runs several containers together, usually on one host

Kubernetes
-> schedules and operates container images across a cluster
```

Kubernetes does not execute a Dockerfile or a Compose file. It runs the images
that were already built.

**Service vs FastAPI service**

```text
FastAPI service
= your application code

Kubernetes Service
= a networking object that gives Pods a stable name/IP
```

**ClusterIP**

A ClusterIP belongs to one Kubernetes Service. It is not the IP of the cluster.

```text
Service frontend   -> ClusterIP A
Service rag-api    -> ClusterIP B
Service prometheus -> ClusterIP C
```

**Pod vs container vs replica**

```text
Pod
+-- one main container
+-- optional tightly coupled sidecar(s)

replicas: 3
=
three copies of that complete Pod template
```

**Node**

A node is the machine Kubernetes schedules Pods onto. Four GPUs do not
necessarily mean four nodes; one node can contain several GPUs.

**Managed Kubernetes**

EKS, AKS, and GKE operate much of the Kubernetes control plane for you. Cloud
is not mandatory: Kubernetes can also run on private cloud, OpenStack, VMs, or
bare metal.

**Persistent storage**

A PVC is a request for storage. The actual disk is supplied by a storage
backend through a StorageClass/CSI driver, for example cloud block storage or
an on-prem storage system.

**Base64 vs checksum vs encryption**

```text
Base64:
reversible encoding, no secret key, not security

Checksum/hash:
compact fingerprint used for integrity/change detection

Encryption:
protects confidentiality and requires cryptographic key material
```

The ingestion pipeline uses a checksum because it needs to answer, cheaply and
deterministically, "did the document content change?" It is not using the
checksum to hide the document.

**nginx in this repo**

The production frontend image uses nginx to serve the built React/Vite static
files and reverse-proxy API requests to `http://rag-api:8000`. nginx is not
Kubernetes; it is an application-level web server/proxy running inside the
frontend container.

## Interviewer pushback, and how I would answer it

> **Use a calibrated story.** Keep two things separate. At Elevait, you
> worked on application/ML/backend services that were containerized and
> deployed with Kubernetes and Helm through a CI/CD setup using Jenkins
> and Harbor. A colleague confirmed the cluster ran on C&H servers and
> was managed by StarOps. Your role was mainly application-side, not
> ownership of the Kubernetes control plane. Separately, this
> `local-rag-system` Kubernetes setup is your hands-on environment for
> going deeper into the platform itself. Do not merge those two stories
> into one.

**Q: "Did you personally administer the Kubernetes cluster in production?"**

No. My production experience was mainly from the application side:
deploying and troubleshooting services that ran on Kubernetes, working
with Helm-based configuration and the CI/CD path around Jenkins and
Harbor. The underlying cluster was managed by StarOps. I would not
claim control-plane administration, cluster upgrades, or platform
ownership because that was not my role.

**Q: "What infrastructure was the production cluster running on?"**

What I can state confidently is that the cluster ran on C&H servers and
was managed by StarOps, with a Kubernetes- and Helm-based setup. I do
not have enough evidence to call it EKS, AKS, GKE, OpenStack, VMware, or
bare metal, so I would not guess. If that distinction matters, I would
say exactly what I know and separate it from what the infrastructure
team owned.

**Q: "So what did you actually do with Kubernetes in production?"**

I worked on the application and backend side: containerized services,
deployment configuration, Helm values/manifests where relevant, CI/CD
integration, and troubleshooting when an application did not start or
behave correctly after deployment. My strength is reasoning from the
application's runtime contract into Kubernetes and debugging the
boundary between application, configuration, networking, dependencies,
and resource behavior.

**Q: "Is this local RAG Kubernetes setup itself a production deployment?"**

No. This is a deliberate hands-on learning and portfolio environment
built around a non-trivial application so I could practice the
mechanics directly: StatefulSet/PVC behavior, Services and
EndpointSlices, probes, HPA, rolling updates, bad-image rollback,
NetworkPolicy, Ingress, and failure scenarios. I use it to deepen
Kubernetes knowledge, not to claim that this exact local Minikube setup
is a customer production environment.

**Q: "Why should I value local Minikube practice if production is different?"**

Because the core Kubernetes control model is the same: API objects,
Deployments, ReplicaSets, StatefulSets, Services, probes, scheduling,
reconciliation, rollouts, and storage claims behave according to the
same APIs. What Minikube does not reproduce is the real production
infrastructure around them: multi-node failure domains, cloud or
on-prem load balancers, enterprise storage, real secret management,
GitOps, policy enforcement, and platform operations. I use Minikube to
learn the mechanics and I am explicit about where the production
differences start.

**Q: "Would you call yourself a Kubernetes expert?"**

No. I have solid application-side experience plus hands-on
understanding of the core workload and networking concepts. I can
reason through Pods, Deployments, ReplicaSets, StatefulSets, Services,
probes, resources, rollouts, HPA, Ingress, NetworkPolicy, PVCs, and
common failure modes. I would not claim deep cluster-administration
expertise in areas such as control-plane upgrades, multi-cluster
operations, CNI internals, or organization-wide RBAC design.

**Q: "Your local cluster had failures. Doesn't that mean the setup is unreliable?"**

The failures are actually useful because they exposed real assumptions.
We saw a slow model-loading startup get killed by liveness, which led
to a proper startup probe and separate `/livez` and `/readyz`. We also
found a frontend nginx resolver assumption that worked in Compose but
not cleanly across environments. I would rather show that I can
diagnose why a container is restarting, distinguish application from
platform issues, and fix the contract than present a demo that only
works because nothing difficult was exercised.

**Q: "Why did you add separate `/livez` and `/readyz` endpoints?"**

Because liveness and readiness answer different operational questions.
A temporary Postgres or Ollama problem should usually make the API
unready so it stops receiving normal traffic, but it should not cause
Kubernetes to restart an otherwise healthy FastAPI process. `/livez`
therefore avoids downstream dependency checks; `/readyz` can return a
real non-2xx response when the application cannot serve correctly. A
startup probe handles the separate problem of long cold starts.

**Q: "Why run Postgres in Kubernetes if many production systems use a managed database?"**

Those are separate questions. If I run a stateful workload in
Kubernetes, StatefulSet plus persistent storage is the appropriate
primitive to understand. Whether I should run the production database
in Kubernetes is an architecture and operations decision. If a managed
database is available and meets sovereignty, latency, cost, and
operational requirements, I would seriously consider it because
backups, failover, patching, and replication are significant
responsibilities.

**Q: "Your NetworkPolicy objects may not be enforced locally. Isn't that a security failure?"**

The important point is not to confuse "the API object exists" with
"traffic is being enforced." NetworkPolicy enforcement depends on the
CNI. In a local cluster I verify whether the chosen CNI actually
supports it. In production I would treat that as a platform requirement
and test the deny/allow behavior, not just check that `kubectl get
networkpolicy` returns objects.

**Q: "Most of your local workloads run with very few replicas. Isn't that unrealistic?"**

It is a resource-conscious local default, not a production sizing
recommendation. The important behavior is still testable: I can scale
the Deployment, watch EndpointSlices change, observe rolling updates
with `maxSurge`/`maxUnavailable`, and demonstrate what breaks when
process-local state meets multiple Pods. Production replica count would
come from availability requirements, load tests, failure domains, and
downstream capacity.

**Q: "Why not just use raw YAML in production?"**

I would normally expect a higher-level deployment workflow: Helm and/or
Kustomize for environment-specific configuration, plus CI/CD or GitOps
such as Argo CD or Flux to reconcile the desired version. Raw YAML is
useful because it makes the underlying Kubernetes objects explicit, but
production needs repeatable releases, versioned configuration, rollback,
secret handling, and controlled promotion across environments.

**Q: "What is still missing before this local setup would resemble production?"**

Several things: a real registry and immutable image versions,
production secret management, RBAC and service accounts, tested
NetworkPolicy enforcement, metrics-server/custom metrics if HPA is
required, production-grade storage and database backup/restore,
multi-node or multi-zone spreading, PDBs where appropriate, centralized
logs/metrics/traces, TLS and certificate management, GitOps/CI/CD
promotion, load testing, and a clear disaster-recovery story. I would
also validate that the application itself is safe under multiple
replicas, not just that Kubernetes can create them.

**Q: "Tell me about the most useful issue you found while doing this."**

The slow-start issue is a good example. The API loads heavy ML
libraries before it can serve HTTP, so the original liveness timing
killed the container while it was still legitimately starting. I
confirmed the process eventually started when left alone, then
separated startup from liveness: a startup probe gives cold start
enough time, liveness checks only the process, and readiness checks
whether the application can actually serve. That changed the design
based on observed behavior rather than just increasing a timeout
blindly.

**Q: "What would you do if you saw a Pod in `CrashLoopBackOff` during an incident?"**

I would not treat `CrashLoopBackOff` as the root cause. I would inspect
`kubectl describe pod` for last state and Events, then `kubectl logs
--previous` because the current container may have already restarted.
I would distinguish application exit, failed liveness, OOM kill, bad
configuration, missing dependency, or another cause before changing
anything. The status tells me the restart pattern; the Events and
previous logs tell me why.

**Q: "If I gave you a Kubernetes problem you had never seen before, how would you approach it?"**

I would reduce it by layer: desired state and controller, scheduling,
container startup, health, Service/EndpointSlice routing,
DNS/networking, storage, resources, and dependencies. I would use the
cluster's own status and Events to choose the next check. I do not need
to memorize every Kubernetes failure mode if I can identify which
component owns the behavior and test that layer systematically.

---

`README.md`, `EXERCISES.md`, and `LEARNING.md` in this directory are
the terse reference versions of this same material, for when you're
actually at the keyboard.
