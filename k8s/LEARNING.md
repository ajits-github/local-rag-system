# Kubernetes Concept and Interview Reference

This is the compact "why" guide for the Kubernetes experiment in this
repository.

Use:

- `README.md` to build and run the environment;
- `EXERCISES.md` to break things deliberately;
- this file to understand the architecture and prepare interview answers.

---

# 1. The entire system in one picture

```text
Windows host
|
|-- browser / curl
|-- Ollama (native process)
|
`-- Docker Desktop
    |
    `-- kind cluster: rag-learning
        |
        +-- control plane node
        |   +-- kube-apiserver
        |   +-- scheduler
        |   +-- controller manager
        |   `-- etcd
        |
        +-- worker node 1
        |
        +-- worker node 2
        |
        `-- namespace: rag
            |
            +-- Ingress
            |
            +-- Service: frontend (ClusterIP)
            |       |
            |       v
            |    frontend Pod
            |       |
            |       `-- nginx
            |            |
            |            | proxy /api
            |            v
            +-- Service: rag-api (ClusterIP)
            |       |
            |       +----> rag-api Pod A
            |       +----> rag-api Pod B
            |       `----> rag-api Pod C
            |
            +-- Service: postgres (headless)
            |       |
            |       v
            |    postgres-0
            |       |
            |       `-- PVC
            |
            +-- Service: ollama-host (ExternalName)
            |       |
            |       `----> host.docker.internal:11434
            |
            +-- Prometheus
            +-- Grafana
            `-- Jaeger
```

Two separate flows are worth remembering.

### Browser loads the frontend

```text
Browser
-> Ingress
-> frontend Service
-> frontend Pod
-> nginx serves static React/Vite files
```

### Browser asks the RAG system a question

```text
Browser
-> Ingress
-> frontend Service
-> frontend Pod / nginx
-> rag-api Service
-> one ready rag-api Pod
-> PostgreSQL / Ollama / other dependencies
```

Ingress does not have to know about `rag-api` in this design. nginx inside the
frontend container performs the internal API proxy hop.

---

# 2. The four layers people mix up

## Cluster

The whole Kubernetes environment.

```text
cluster
= control plane + nodes + Kubernetes objects/workloads
```

## Node

A machine Kubernetes can schedule Pods onto.

In production:

```text
VM or physical server
```

In this local `kind` cluster:

```text
Docker container pretending to be a Kubernetes node
```

## Pod

The smallest unit Kubernetes schedules.

A Pod contains one or more tightly coupled containers.

Containers in the same Pod:

- run on the same node;
- share the same Pod IP;
- share the network namespace;
- can communicate through `localhost`.

## Container

The running process environment created from a container image.

The relationship is:

```text
Cluster
`-- Node
    `-- Pod
        +-- container
        `-- optional sidecar container
```

---

# 3. Replica does not mean container

A replica is another copy of the complete Pod template.

Example:

```yaml
replicas: 3
```

means:

```text
Pod A
`-- rag-api container

Pod B
`-- rag-api container

Pod C
`-- rag-api container
```

If the Pod template had two containers, every replica would contain both.

A whole product normally has several independently scaled workloads:

```text
frontend Deployment: 3 replicas
rag-api Deployment: 8 replicas
worker Deployment: 4 replicas
model gateway: 3 replicas
postgres: separate stateful/data service
```

So "number of Pods" is not one number for the whole product.

---

# 4. Control plane and API server

The control plane manages cluster state.

Key components:

```text
kube-apiserver
    central API used by kubectl, kubelets, controllers, scheduler

etcd
    persistent store for Kubernetes cluster state

scheduler
    chooses a node for unscheduled Pods

controller manager
    runs reconciliation controllers
```

Almost everything talks through the API server:

```text
kubectl -----------+
scheduler ---------+
controllers -------+--> kube-apiserver --> etcd
kubelet -----------+
```

The kubelet on each node reports node/Pod status through the cluster API.

---

# 5. Reconciliation: the core Kubernetes idea

Kubernetes is declarative.

You say:

```text
desired replicas = 3
```

Kubernetes continuously compares:

```text
desired state
vs
actual state
```

If only two replicas exist:

```text
3 desired
2 actual
    |
    v
controller creates another Pod
```

This is why deleting a Deployment-managed Pod causes a replacement.

Kubernetes protects the desired workload state, not the identity of a
particular disposable Pod.

---

# 6. Deployment, ReplicaSet, StatefulSet, DaemonSet, Job

## Deployment

For stateless/interchangeable application replicas.

Adds:

- rolling updates;
- rollout history;
- rollback;
- ReplicaSet management.

Used here for:

```text
rag-api
frontend
```

## ReplicaSet

Keeps N matching Pods running.

Normally created and managed by a Deployment rather than written manually.

## StatefulSet

For workloads needing stable identity and/or stable per-Pod storage.

Used here for PostgreSQL.

Typical properties:

```text
stable ordinal Pod names
stable per-Pod storage claims
ordered identity
```

Important:

> StatefulSet does not automatically provide database replication or high
> availability.

## DaemonSet

Normally runs one matching Pod on every eligible node.

Common examples:

- node monitoring agent;
- log collector;
- CNI/network agent;
- security agent.

## Job

Runs to completion and stops.

Used here for database initialization.

## CronJob

Creates Jobs on a schedule.

Not used here, but worth knowing for interviews.

---

# 7. Service, ClusterIP, DNS, and EndpointSlice

This is one of the most important networking concepts.

## Kubernetes Service

A Service is a **Kubernetes networking object**.

It is not your FastAPI service.

The `rag-api` Service gives changing API Pods a stable network identity.

## ClusterIP

A normal Service gets an internal virtual IP.

Example:

```text
Service: rag-api
ClusterIP: 10.96.20.50
```

This is not "the IP of the Kubernetes cluster."

One cluster can contain many Services:

```text
frontend Service   -> 10.96.1.10
rag-api Service    -> 10.96.20.50
prometheus Service -> 10.96.40.12
```

Each may have its own ClusterIP.

## Pod IPs

The Service might currently point to:

```text
Pod A -> 10.244.1.20
Pod B -> 10.244.2.15
Pod C -> 10.244.3.8
```

Pod IPs are disposable.

## DNS

CoreDNS resolves:

```text
rag-api
    ->
Service ClusterIP
```

Why resolve the Service instead of a Pod?

Because the Service is stable and Pods are not.

## EndpointSlice

Kubernetes tracks the backend addresses belonging to a Service through
EndpointSlice resources.

Readiness influences whether a backend is ready for normal Service traffic.

## Final path

```text
client Pod
    |
    | http://rag-api:8000
    v
CoreDNS
    |
    | rag-api -> 10.96.20.50
    v
Service ClusterIP
    |
    | service forwarding
    v
one ready rag-api Pod
```

Traditionally, Service forwarding is implemented through kube-proxy with
iptables/IPVS. Some modern networking stacks implement equivalent behavior
through eBPF.

Interview-safe wording:

> DNS resolves the stable Service name to its ClusterIP. Kubernetes Service
> networking then forwards the connection to one currently usable backend Pod.

---

# 8. Service types

## ClusterIP

Internal-only virtual Service IP.

Default choice for internal services.

Used for:

```text
frontend
rag-api
```

## NodePort

Opens a port on every node.

Conceptually:

```text
node-ip:nodePort
    ->
Service
    ->
Pod
```

Useful for simple/direct exposure, especially in labs or some bare-metal
setups.

## LoadBalancer

Requests an external load balancer from the infrastructure integration.

Common in managed/cloud environments.

Example conceptually:

```text
public/private cloud LB
    ->
Kubernetes Service
    ->
Pods
```

## ExternalName

DNS alias to an external hostname.

Used here for:

```text
ollama-host
    ->
host.docker.internal
```

No selector, no Pod backend, no ClusterIP.

## Headless Service

```yaml
clusterIP: None
```

No shared virtual ClusterIP.

Commonly used with StatefulSets where clients may need stable per-Pod identities.

Used here for PostgreSQL.

---

# 9. Ingress and ingress controller

These are different things.

## Ingress

A Kubernetes resource containing HTTP routing rules.

Example:

```text
host: rag.local
/
  -> frontend Service
```

## Ingress controller

The actual running software that watches Ingress objects and implements the
routing.

Example:

```text
ingress-nginx
```

The Ingress object by itself does not forward packets.

Mental model:

```text
Ingress
    = desired routing rule

Ingress controller
    = software that makes the rule real
```

---

# 10. nginx in this repository

nginx is an application-level web server/reverse proxy running inside the
frontend container.

It serves the built React/Vite frontend files:

```text
index.html
JavaScript
CSS
```

and proxies API calls internally to:

```text
http://rag-api:8000
```

So there can be two HTTP routing layers:

```text
external:
Ingress controller
    -> frontend Service

inside frontend:
nginx
    -> rag-api Service
```

nginx is not Kubernetes.

---

# 11. Readiness, liveness, and health

Memorize these as different questions.

## `/health`

```text
"What is going on?"
```

Useful for detailed diagnostics.

## Readiness

```text
"Should this Pod receive user traffic right now?"
```

If readiness fails:

```text
container keeps running
Pod becomes NotReady
Service stops using it for normal traffic
Kubernetes keeps checking it
Pod can rejoin when it recovers
```

## Liveness

```text
"Is this container unhealthy enough that restarting it may help?"
```

If liveness repeatedly fails:

```text
kubelet restarts the failing container
```

A liveness failure does not inherently mean Kubernetes deletes the whole Pod
object and asks the Deployment to create a new one.

## Good production pattern

```text
/health
/livez
/readyz
```

The `z` is only a naming convention. Kubernetes does not require it.

Example:

```text
FastAPI alive
Postgres down

/livez  -> 200
/readyz -> 503
```

Why?

Restarting healthy API processes does not fix a database outage.

---

# 12. Requests and limits

Example:

```yaml
resources:
  requests:
    cpu: "250m"
    memory: "768Mi"

  limits:
    cpu: "1"
    memory: "1536Mi"
```

## Requests

Used for important scheduling decisions:

```text
Can this Pod fit on this node?
```

Also used in standard CPU-utilization HPA calculations.

## Limits

Runtime ceiling.

Typical effect:

```text
CPU above limit
    -> throttling

memory above limit
    -> OOM kill possible
```

The actual resources come from the underlying node machine.

Kubernetes allocates/schedules/enforces them; it does not manufacture CPU or
RAM.

---

# 13. Scheduling

Scheduler flow:

```text
new Pod has no node
    |
    v
scheduler evaluates candidate nodes
    |
    +-- enough requested CPU/memory?
    +-- node selectors?
    +-- affinity/anti-affinity?
    +-- taints/tolerations?
    +-- topology?
    +-- volumes?
    +-- GPU/resource requirements?
    |
    v
one node chosen
    |
    v
kubelet on that node starts the containers
```

Do not assume replicas will always spread evenly across nodes unless you define
placement constraints.

For high availability, real production often uses:

- topology spread constraints;
- pod anti-affinity;
- multiple availability zones;
- PodDisruptionBudgets.

---

# 14. HPA

HPA = Horizontal Pod Autoscaler.

It changes **Pod replica count** indirectly by updating the scale target, such
as a Deployment.

Conceptually:

```text
metrics
    |
    v
HPA
    |
    | desired replicas
    v
Deployment
    |
    v
ReplicaSet
    |
    v
Pods
```

Related but different:

```text
HPA
    -> number of Pods

VPA
    -> resource recommendations/requests for Pods

cluster/node autoscaling
    -> number/capacity of nodes
```

For this RAG workload, CPU may be a poor production scaling signal because user
latency can be dominated by model inference or external calls.

Potentially better signals:

- queue depth;
- in-flight requests;
- p95 latency;
- model serving concurrency;
- model accelerator saturation.

---

# 15. Persistent storage

A StatefulSet does not physically provide storage.

The chain is:

```text
StatefulSet
    ->
PersistentVolumeClaim (PVC)
    ->
StorageClass
    ->
CSI/storage provider
    ->
actual disk/storage system
```

Examples of actual backends:

```text
local kind
    -> local-path style storage

AWS
    -> EBS

Azure
    -> Azure Disk

GCP
    -> Persistent Disk

on-prem
    -> Ceph / SAN / NFS / other CSI backend
```

The PVC is a **request/claim**, not the physical storage itself.

---

# 16. ConfigMap and Secret

## ConfigMap

Non-secret configuration.

Examples:

- log level;
- service URL;
- feature/config selection.

## Secret

Sensitive configuration object.

Examples:

- database password;
- API key;
- JWT signing secret.

Important:

> Kubernetes Secret data is commonly represented as base64. Base64 is not
> encryption.

---

# 17. Base64, checksum, and encryption

These solve completely different problems.

## Base64

```text
input
    ->
reversible text encoding
```

No key.

Purpose:

- represent arbitrary bytes safely as text.

Not security.

## Checksum/hash

```text
file bytes
    ->
SHA-256-like fingerprint
```

Purpose:

- integrity;
- content identity;
- change detection.

This RAG ingestion pipeline uses checksums because it needs to answer:

```text
"Has this document content changed since the last ingestion?"
```

If:

```text
new checksum == stored checksum
```

the file can be treated as unchanged.

## Encryption

```text
plaintext + key
    ->
ciphertext
```

Purpose:

- confidentiality.

Requires cryptographic key material.

---

# 18. Why checksum updates must be atomic with ingestion

Suppose:

```text
old file checksum = AAA
database checksum = AAA
```

The file changes:

```text
new file checksum = BBB
```

A bad re-ingestion sequence would be:

```text
1. save database checksum BBB
2. delete old chunks
3. crash before inserting new chunks
```

Now:

```text
file checksum = BBB
database checksum = BBB
chunks = missing
```

On retry the application may conclude:

```text
BBB == BBB
therefore unchanged
skip ingestion
```

That is wrong because the chunks were never committed.

The safe idea is:

```text
BEGIN TRANSACTION

update checksum
replace chunks

COMMIT
```

If something fails before commit:

```text
ROLLBACK
```

so the database still reflects the old consistent state.

This is a good interview example of why idempotency and atomicity are related
but not the same thing.

---

# 19. ConfigMap update behavior

If a ConfigMap value reaches a container through:

```yaml
env:
envFrom:
```

it is copied into the process environment at container startup.

Changing the ConfigMap later does not mutate environment variables inside an
already-running process.

Typical fix:

```bash
kubectl apply ...
kubectl rollout restart deployment/rag-api -n rag
```

A ConfigMap mounted as files can be updated by kubelet, but the application
still has to be written to notice/reload those files.

---

# 20. Init container

An init container runs before the main application containers and must complete
successfully first.

In this repo it waits for PostgreSQL connectivity.

This is one possible dependency-startup strategy.

It is not always required.

Another common production pattern is:

```text
application starts
    ->
database connection fails
    ->
retry with backoff
    ->
readiness remains false
    ->
eventually succeeds
```

That can reduce tight startup coupling between services.

---

# 21. Rolling update

With:

```yaml
maxSurge: 1
maxUnavailable: 0
```

the Deployment can temporarily create one extra Pod while avoiding deliberate
loss of available desired capacity.

Simplified:

```text
old Pod Ready
    |
    v
create new Pod
    |
    v
new Pod Ready
    |
    v
remove old Pod
```

A bad new image can leave a rollout stuck while the old ReplicaSet remains
available.

Rollback:

```bash
kubectl rollout undo deployment/rag-api -n rag
```

---

# 22. Node failure

Node failure is not the same as Pod failure.

If the control plane stops receiving node/kubelet updates:

```text
node becomes unhealthy/unreachable
```

but Kubernetes must be careful before replacing workloads elsewhere.

Why?

The old node may be:

```text
network partitioned
but still physically running
```

That creates split-brain/state-ownership risks.

Modern Kubernetes uses node conditions, taints, tolerations, and controller
timers to decide when Pods should be evicted/replaced.

Interview-safe wording:

> Kubernetes is deliberately conservative around node failure because
> "unreachable from the control plane" is not the same as "definitely powered
> off."

---

# 23. NetworkPolicy

A NetworkPolicy describes which Pod traffic should be allowed or denied.

Example intent:

```text
frontend -> rag-api       allow
rag-api -> postgres       allow
random Pod -> postgres    deny
```

But the API object is not itself the firewall implementation.

Enforcement depends on the cluster networking/CNI data plane.

Therefore verify behavior, not only existence:

```bash
kubectl get networkpolicy
```

is not enough.

---

# 24. Kustomize vs Helm

## Plain manifests

Direct Kubernetes YAML.

## Kustomize

Composes and patches plain YAML.

Example:

```bash
kubectl apply -k k8s/base
```

No Helm-style template language required.

This repo uses it to group resources and centrally transform image tags.

## Helm

Packages and templates Kubernetes manifests.

Typical chart:

```text
Chart.yaml
values.yaml
templates/
```

Helm also tracks release history and supports upgrade/rollback workflows.

Mental model:

```text
Kustomize
    "start from YAML and patch it"

Helm
    "render YAML from a packaged chart + values"
```

---

# 25. Dockerfile, Compose, Kubernetes

Keep the jobs separate.

## Dockerfile

Build recipe for one image.

```text
source code
    ->
Dockerfile
    ->
container image
```

## Docker Compose

Runs multiple containers together, usually on one host.

Useful for:

- local development;
- integration testing;
- simple environments.

## Kubernetes

Runs/orchestrates container images as Pods across a cluster.

Useful when you need:

- multi-node scheduling;
- self-healing desired state;
- rolling updates;
- horizontal scaling;
- service discovery;
- policy;
- clustered operations.

Docker restart policies can restart a crashed container on one host. Kubernetes
adds orchestration across the cluster.

---

# 26. `kind`, minikube, EKS, AKS, GKE

## kind

Kubernetes IN Docker.

Good for:

- local learning;
- CI;
- disposable clusters;
- multi-node simulation.

## minikube

Another local Kubernetes development tool.

## EKS / AKS / GKE

Managed Kubernetes offerings:

```text
EKS -> AWS
AKS -> Azure
GKE -> Google Cloud
```

"Managed" mainly means the provider operates substantial parts of the
Kubernetes control-plane/platform lifecycle for you.

Cloud is not required for Kubernetes.

Self-managed clusters can run on:

- OpenStack;
- VMware;
- private cloud;
- physical servers;
- company data centers.

---

# 27. GPU nodes

A GPU is not automatically a node.

Example:

```text
Node A
+-- GPU 1
`-- GPU 2

Node B
+-- GPU 3
`-- GPU 4
```

That is:

```text
2 nodes
4 GPUs
```

Kubernetes runs on the machine and schedules workloads that request accelerator
resources.

Conceptually:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

assuming the cluster has the necessary drivers/device plugin/runtime setup.

For a production RAG system, model inference might run:

- inside Kubernetes on GPU node pools;
- in a separate model-serving platform;
- through a managed/hosted LLM API.

---

# 28. Process-local state: what breaks with replicas

Horizontal scaling is safe only when replicas agree on important shared state.

Known process-local areas in this codebase include:

1. synthetic MCP business state;
2. in-memory rate limiting;
3. Pod-local uploaded files;
4. per-Pod embedding model memory.

Example:

```text
Pod A:
case X = resolved

Pod B:
case X = open
```

A Kubernetes Service cannot repair this inconsistency. It only routes traffic.

The fix depends on the state:

```text
durable business state
    -> shared database/service

distributed rate limiting
    -> shared counter store, e.g. Redis

uploaded documents
    -> object storage/shared durable storage

expensive model memory
    -> possibly centralized model serving
```

This is the bridge between Kubernetes and distributed-systems design.

---

# 29. Observability: what each tool is doing

Current stack:

```text
Prometheus
    metrics

Grafana
    dashboards/visualization

Jaeger
    distributed trace backend/UI

OpenTelemetry
    instrumentation/export standard

structured logs
    application event records
```

Commercial tools such as Splunk or Dynatrace can overlap with several of these
capabilities, but they are not currently the observability stack described by
this repo.

A production troubleshooting mindset should connect:

```text
metrics
    "Is something wrong?"

logs
    "What happened?"

traces
    "Where did the request spend time?"
```

---

# 30. Kubernetes RBAC and Pod security

The current learning manifests do not represent a complete production RBAC
setup.

## RBAC

Controls who/what can perform Kubernetes API actions.

Core objects:

```text
ServiceAccount
Role
ClusterRole
RoleBinding
ClusterRoleBinding
```

Example:

```text
monitoring ServiceAccount
    can read Pod metadata
    cannot delete Deployments
```

## Pod security

Controls what containers may do on the host.

Typical hardening ideas:

```text
runAsNonRoot
allowPrivilegeEscalation: false
readOnlyRootFilesystem
drop Linux capabilities
avoid privileged containers
avoid hostPath/hostNetwork unless required
```

Do not confuse:

```text
RBAC
    -> Kubernetes API permissions

NetworkPolicy
    -> network communication

Pod security
    -> container privilege/isolation on node
```

---

# 31. Production architecture: local vs real

This repo's Kubernetes setup is a learning environment.

A production system for global users might add:

- several application replicas;
- topology spread across zones;
- PodDisruptionBudgets;
- multiple clusters/regions;
- WAF/API gateway/load balancers;
- TLS and corporate PKI;
- enforced NetworkPolicy;
- RBAC and workload identity;
- secret manager/KMS integration;
- immutable image digests;
- CI/CD or GitOps;
- managed or HA PostgreSQL;
- object storage for uploads;
- shared/distributed rate limiting;
- central metrics/logs/traces;
- GPU model-serving infrastructure or external model provider;
- backup/restore and disaster recovery;
- SLOs and alerting.

Do not claim every production system uses every item. The exact architecture
depends on traffic, compliance, platform standards, cost, and data residency.

---

# 32. Troubleshooting decision tree

## Pod does not start

Check:

```bash
kubectl describe pod ...
kubectl get events ...
```

Look for:

```text
Pending
ImagePullBackOff
CreateContainerConfigError
init container failure
PVC problem
resource scheduling problem
CrashLoopBackOff
```

## Pod runs but receives no traffic

Check:

```text
readiness
Service selector
EndpointSlice
ports / targetPort
Ingress
NetworkPolicy
application bind address
```

## Container keeps restarting

Check:

```text
liveness
exit code
OOMKilled
application crash
startup dependency failure
```

Use:

```bash
kubectl logs <pod> --previous
```

## Service name does not resolve

Check:

```text
Service exists?
correct namespace?
CoreDNS healthy?
correct Service name?
```

## HPA does nothing

Check:

```text
metrics-server available?
resource requests defined?
metric actually above target?
maxReplicas already reached?
```

---

# 33. Interview answers worth being able to say in 20 seconds

## Why Kubernetes?

> Docker can run and restart containers on one host. Kubernetes adds
> orchestration across a cluster: scheduling, replicas, service discovery,
> rolling updates, autoscaling, and reconciliation of desired state.

## Pod vs container

> A Pod is Kubernetes' scheduling unit and contains one or more tightly coupled
> containers. Containers in the same Pod share the Pod network namespace and
> IP.

## Service vs FastAPI service

> A Kubernetes Service is a stable networking abstraction in front of Pods. It
> is unrelated to the word "service" in application code.

## ClusterIP

> A ClusterIP is the stable internal virtual IP assigned to one Kubernetes
> Service. It is not the IP address of the whole cluster.

## Deployment vs StatefulSet

> Deployment is for interchangeable replicas. StatefulSet adds stable ordinal
> identity and stable per-Pod storage association for stateful workloads.

## readiness vs liveness

> Readiness controls whether a Pod receives traffic. Liveness controls whether
> kubelet should restart the container.

## HPA

> HPA changes replica count based on metrics. For RAG workloads I would check
> whether CPU represents the actual bottleneck before using it as the primary
> production scaling signal.

## Node failure

> Kubernetes must distinguish a dead node from an unreachable node. It therefore
> uses health status and eviction logic rather than immediately duplicating
> every workload elsewhere.

## NetworkPolicy

> The policy object expresses intent; actual enforcement comes from the
> networking/CNI implementation.

## Stateful data and replicas

> Scaling stateless API Pods is straightforward only if shared state is
> externalized. Process-local counters, locks, uploaded files, or mutable
> business state become distributed-systems problems as soon as replicas
> increase.

---

# 34. 25 interview questions to practice

1. What is the difference between a container, Pod, Node, and cluster?
2. What happens after `kubectl apply -f deployment.yaml`?
3. What does the Kubernetes API server do?
4. Why does deleting a Deployment-managed Pod create another Pod?
5. Deployment vs ReplicaSet: why do we need both?
6. Deployment vs StatefulSet: when would you choose each?
7. What does a PVC actually represent, and who supplies the real disk?
8. What is a Kubernetes Service?
9. What exactly is a ClusterIP?
10. How does DNS resolution for `rag-api` work inside a namespace?
11. What happens between Service ClusterIP and a real Pod IP?
12. What is the difference between ClusterIP, NodePort, LoadBalancer, and ExternalName?
13. Ingress vs Ingress controller: what is the difference?
14. What happens when readiness fails?
15. What happens when liveness fails?
16. Why should liveness normally avoid checking downstream dependencies?
17. What do resource requests and limits do?
18. How does HPA decide to scale?
19. Why can CPU-based autoscaling be misleading for a RAG/LLM API?
20. What happens during a rolling update?
21. What happens if a new image cannot be pulled?
22. How does Kubernetes react when a node becomes unreachable?
23. Why can process-local rate limiting fail after horizontal scaling?
24. What is the role of NetworkPolicy and the CNI?
25. What would you change before calling this Kubernetes setup production-ready?

If you can answer all 25 using the repo's concrete examples, you understand the
parts of Kubernetes that are most likely to matter in an infrastructure-heavy
interview.
