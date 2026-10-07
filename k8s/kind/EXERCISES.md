# Kubernetes Operational Exercises

These exercises are designed to make the Kubernetes concepts observable rather
than theoretical.

Run them against the local `rag-learning` cluster from `k8s/README.md`.

Keep two terminals open:

**Terminal A:**
```bash
kubectl get pods -n rag -o wide -w
```

**Terminal B:**
Use it for the exercise commands.

For each exercise, do four things:

1. predict what should happen;
2. run the command;
3. inspect what actually happened;
4. explain why in one or two sentences.

That last step is the interview preparation.

---

# 1. Delete a Pod and watch self-healing

## Goal

Understand the difference between:

```text
"restart this particular Pod"
```

and:

```text
"keep the desired number of replicas running"
```

## Run

```bash
kubectl get pods -n rag -l app=rag-api -o wide
kubectl delete pod -n rag -l app=rag-api --wait=false
kubectl get pods -n rag -l app=rag-api -o wide -w
```

## What should happen

The old Pod disappears.

A **new Pod object** appears with:

- a new Pod name;
- usually a new Pod IP;
- possibly a different node.

Why?

```text
Deployment says replicas = 1

actual Pods becomes 0
        |
        v
ReplicaSet/controller notices mismatch
        |
        v
new Pod created
```

The old Pod does not "come back."

## Inspect

```bash
kubectl describe deployment rag-api -n rag
kubectl get events -n rag --sort-by=.lastTimestamp
```

## Interview answer

> A Deployment preserves desired state, not Pod identity. If one replica
> disappears, the controller creates a replacement Pod so actual replicas again
> match `spec.replicas`.

---

# 2. Compare Deployment recovery with StatefulSet recovery

## Goal

See why PostgreSQL is treated differently from `rag-api`.

## Before

```bash
kubectl get pod postgres-0 -n rag -o wide
kubectl get pvc -n rag
```

## Delete the PostgreSQL Pod

```bash
kubectl delete pod postgres-0 -n rag
kubectl get pod postgres-0 -n rag -o wide -w
```

## What should happen

A Pod named:

```text
postgres-0
```

returns.

Its container is new, but its StatefulSet identity remains stable and the same
PVC remains associated with that ordinal.

Check:

```bash
kubectl get pvc -n rag
kubectl describe pod postgres-0 -n rag
```

## Interview answer

> Deployment replicas are intended to be interchangeable. StatefulSet Pods have
> stable ordinal identities and can keep stable per-Pod storage across
> recreation.

Important: a StatefulSet by itself does **not** make PostgreSQL highly
available. This setup still has one database replica.

---

# 3. Scale `rag-api` from 1 to 3 replicas

## Goal

See what a replica actually is and how a Service backend set changes.

## Run

```bash
kubectl scale deployment rag-api -n rag --replicas=3
kubectl get pods -n rag -l app=rag-api -o wide -w
```

Wait until all three are ready.

Then inspect:

```bash
kubectl get svc rag-api -n rag
kubectl get endpointslices -n rag
```

## What to notice

You now have:

```text
one Deployment
    |
    v
one ReplicaSet
    |
    +-- Pod A
    +-- Pod B
    `-- Pod C
```

Each Pod has its own IP.

The `rag-api` Kubernetes Service still has the same name and ClusterIP. Only the
backend Pod set changes.

## Important correction

Do not assume the scheduler will always spread replicas evenly across nodes
unless you explicitly configure topology spread constraints or pod
anti-affinity. The scheduler considers many factors, especially resource
requests and available capacity.

## Restore

```bash
kubectl scale deployment rag-api -n rag --replicas=1
```

## Interview answer

> Replicas are copies of the whole Pod template. Scaling a Deployment changes
> the number of Pods, while the Service identity remains stable.

---

# 4. Inspect Service name, ClusterIP, and Pod IPs

## Goal

Make the networking layers concrete.

## Run

```bash
kubectl get svc rag-api -n rag -o wide
kubectl get pods -n rag -l app=rag-api -o wide
kubectl get endpointslices -n rag
```

You should see three different concepts:

```text
Service name:
rag-api

Service ClusterIP:
10.96.x.x       (example)

Pod IP:
10.244.x.x      (example)
```

The ClusterIP belongs to the **Kubernetes Service**, not to the entire cluster.

## Test DNS from another Pod

```bash
kubectl run dns-test -n rag \
  --image=busybox:1.36 \
  --restart=Never \
  --rm -it \
  -- nslookup rag-api
```

## Interview answer

> CoreDNS resolves the stable Service name to the Service's ClusterIP. The
> Service networking layer then forwards the connection to one ready backend
> Pod. DNS does not need to change when replicas are added or removed.

---

# 5. Prove /livez and /readyz actually behave differently under a dependency outage

## Goal

Understand why one diagnostic `/health` endpoint is not automatically a
good Kubernetes liveness *or* readiness endpoint, and confirm the
application's three purpose-built endpoints
(`src/rag/api/routers/health.py`) actually behave the way their names
promise, not just in theory.

## What we discovered and fixed

An earlier pass through this exact exercise found a real gap: `/health`
reports `{"status": "degraded"}` correctly, but always with HTTP 200 --
readiness probes only ever look at the status code, never the body, so
wiring `readinessProbe` straight at `/health` meant a real Postgres
outage never removed the Pod from Service traffic at all. The fix (see
`ISSUES.md`) added two dedicated endpoints instead of reusing `/health`
for everything: `/livez` (no dependency checks, ever; a downstream
outage must never look like the process itself is broken) and `/readyz`
(the same dependency checks `/health` already made, but a real
`503` when something required is down). `startupProbe`/`livenessProbe`
now point at `/livez`, `readinessProbe` at `/readyz`; `/health` is
unchanged and still exists purely for human/dashboard diagnostics. The
rest of this exercise proves that fix actually holds, not just that the
old gap existed.

## Stop PostgreSQL

```bash
kubectl scale statefulset postgres -n rag --replicas=0
```

Port-forward the API if needed:

```bash
kubectl port-forward -n rag svc/rag-api 8000:8000
```

In another terminal:

```bash
curl -i http://localhost:8000/health
curl -i http://localhost:8000/livez
curl -i http://localhost:8000/readyz
kubectl get pods -n rag -l app=rag-api
kubectl get endpointslices -n rag
```

## What to notice

You should see all three diverge exactly as their names promise:

```text
/health  -> HTTP 200, body {"status": "degraded", ...}   (diagnostic only, never gates traffic)
/livez   -> HTTP 200, body {"status": "alive"}            (process is fine; no restart should happen)
/readyz  -> HTTP 503, body {"status": "not_ready", ...}   (this Pod should stop receiving traffic)

Pod Ready            = False
Service backend       = removed from endpointslices
Container restarts    = 0 (no liveness failure: /livez never touched Postgres)
```

That last line is the point: a downstream outage now correctly triggers
a *readiness* change, not a restart. Contrast with the old, fixed gap
above, where the Pod stayed `Ready = True` and kept receiving traffic
throughout the same outage.

## Restore PostgreSQL

```bash
kubectl scale statefulset postgres -n rag --replicas=1
kubectl wait --for=condition=ready pod/postgres-0 -n rag --timeout=120s
```

Confirm recovery:

```bash
curl -i http://localhost:8000/readyz
kubectl get endpointslices -n rag
```

`/readyz` should return to `200`/`{"status": "ready", ...}` and the Pod
should reappear in the Service's endpoints, with no restart having
occurred at any point.

## Interview answer

> Liveness answers whether restarting the container could help. Readiness
> answers whether the Pod should receive traffic. A downstream database outage
> should make the API unready, not trigger a restart storm. This requires
> liveness and readiness to check genuinely different things, not the same
> diagnostic endpoint reused twice.

---

# 6. Force a readiness failure and watch traffic removal

## Goal

Prove that the Kubernetes readiness mechanism itself works, independent
of exercise 5's dependency-outage scenario. This one breaks the probe
path directly, with no Postgres outage involved at all.

## Break the readiness path

`readinessProbe` points at `/readyz` (see exercise 5); point it
somewhere that doesn't exist instead:

```bash
kubectl patch deployment rag-api -n rag --type=json \
  -p='[{
    "op":"replace",
    "path":"/spec/template/spec/containers/0/readinessProbe/httpGet/path",
    "value":"/__does_not_exist__"
  }]'
```

Watch:

```bash
kubectl get pods -n rag -l app=rag-api -w
kubectl get endpointslices -n rag
kubectl describe pod -n rag -l app=rag-api
```

## Expected behavior

```text
container keeps running
        |
        v
readiness fails
        |
        v
Pod Ready = False
        |
        v
Pod stops being a normal Service backend
```

The container is not restarted merely because readiness failed.

## Restore

```bash
kubectl patch deployment rag-api -n rag --type=json \
  -p='[{
    "op":"replace",
    "path":"/spec/template/spec/containers/0/readinessProbe/httpGet/path",
    "value":"/readyz"
  }]'
```

---

# 7. Observe a rolling update

## Goal

See how Kubernetes replaces application versions without dropping every old
replica first.

## Run

```bash
kubectl rollout restart deployment/rag-api -n rag
kubectl get pods -n rag -l app=rag-api -w
```

Also inspect:

```bash
kubectl rollout status deployment/rag-api -n rag
kubectl rollout history deployment/rag-api -n rag
kubectl get rs -n rag -l app=rag-api
```

## What to notice

The Deployment creates a new ReplicaSet/pod template generation.

With:

```yaml
maxSurge: 1
maxUnavailable: 0
```

Kubernetes can temporarily run one extra Pod and should not deliberately reduce
the number of ready replicas below the desired count during the rollout.

## Interview answer

> `maxSurge` controls how many extra Pods can exist during rollout.
> `maxUnavailable` controls how many desired replicas may be unavailable.
> `maxUnavailable: 0` favors availability at the cost of temporary extra
> capacity.

---

# 8. Deploy a bad image and roll back

## Goal

Learn the difference between:

```text
bad new version
```

and:

```text
complete outage
```

## Deploy a non-existent image

```bash
kubectl set image deployment/rag-api -n rag \
  rag-api=local-rag-api:does-not-exist

kubectl rollout status deployment/rag-api -n rag --timeout=30s
kubectl get pods -n rag -l app=rag-api
```

Inspect:

```bash
kubectl describe pod -n rag -l app=rag-api
kubectl get events -n rag --sort-by=.lastTimestamp
```

You should see states such as:

```text
ErrImagePull
ImagePullBackOff
```

## Verify old capacity

If the rolling-update strategy can keep the old ReplicaSet available, check:

```bash
kubectl port-forward -n rag svc/rag-api 8000:8000
curl -i http://localhost:8000/health
```

## Roll back

```bash
kubectl rollout undo deployment/rag-api -n rag
kubectl rollout status deployment/rag-api -n rag
```

## Interview answer

> A safe rollout strategy can leave the old ReplicaSet serving while the new
> one fails to become ready. The rollout is stuck, but the previous version can
> remain available.

---

# 9. Compare liveness failure with readiness failure

## Goal

Make the distinction operationally obvious.

### Readiness failure

```text
Pod remains running
Pod becomes NotReady
normal Service traffic stops
Kubernetes keeps probing
Pod can recover without restart
```

### Liveness failure

Repeated liveness failure causes the **kubelet to restart the failing
container inside the Pod**.

The Pod object does not have to be deleted and recreated just because liveness
failed.

Inspect restart count with:

```bash
kubectl get pods -n rag
```

and previous-container logs with:

```bash
kubectl logs <pod-name> -n rag --previous
```

## Interview answer

> Readiness controls traffic eligibility. Liveness controls container restart.

---

# 10. Inspect resource requests and limits

## Goal

Understand what the scheduler uses and what the runtime enforces.

Inspect:

```bash
kubectl describe pod -n rag -l app=rag-api
kubectl top pods -n rag
kubectl top nodes
```

Look for:

```text
Requests:
CPU
Memory

Limits:
CPU
Memory
```

Mental model:

```text
requests
    -> used heavily for scheduling
    -> also used by CPU-utilization HPA math

limits
    -> runtime ceiling
```

Typical effects:

```text
CPU limit exceeded
    -> CPU throttling

memory limit exceeded
    -> container can be OOMKilled
```

## Interview answer

> Requests are primarily about scheduling/guaranteed capacity. Limits are about
> runtime enforcement.

---

# 11. Exercise HPA

## Goal

Observe that HPA changes the Deployment's replica count; it does not create a
separate execution path.

Verify metrics:

```bash
kubectl top pods -n rag
kubectl get hpa -n rag
```

Generate sustained load using your preferred tool.

Watch:

```bash
kubectl get hpa -n rag -w
kubectl get deployment rag-api -n rag -w
kubectl get pods -n rag -l app=rag-api -w
```

The chain is:

```text
metrics
    |
    v
HPA
    |
    | updates replicas
    v
Deployment
    |
    v
ReplicaSet
    |
    v
Pods
```

## Important RAG-specific lesson

CPU can be a weak signal here.

If requests spend most of their time waiting for Ollama or another model
provider, API CPU can remain low while user latency is high.

Better production signals could include:

- in-flight requests;
- queue depth;
- model concurrency;
- p95 request latency;
- model-serving saturation.

Do not add a custom autoscaler merely for the demo. The important part is being
able to explain the mismatch between the metric and the real bottleneck.

---

# 12. Kill a worker node

## Goal

Observe node failure separately from Pod failure.

List nodes:

```bash
kubectl get nodes -o wide
docker ps --filter "name=rag-learning-worker"
```

Stop a kind worker:

```bash
docker stop rag-learning-worker2
kubectl get nodes -w
```

## What to notice

The node eventually becomes `NotReady`/unreachable from the control plane.

Do not memorize one magic rescheduling timeout. Modern Kubernetes behavior is
driven by node status, taints, Pod tolerations, and controller timing.

The key idea:

> The control plane cannot immediately assume that an unreachable node is
> physically dead. It may be a network partition while the old workload is
> still running.

That is especially important for stateful systems.

## Restore

```bash
docker start rag-learning-worker2
kubectl get nodes -w
```

## Interview answer

> Kubernetes deliberately distinguishes "I cannot currently reach this node"
> from "I know every process on that node is dead." Failover therefore has to
> account for split-brain and state ownership, not only restart speed.

---

# 13. Verify NetworkPolicy instead of trusting the object

## Goal

Learn the difference between:

```text
policy object exists
```

and:

```text
packets are actually blocked
```

Apply:

```bash
kubectl apply -f k8s/base/networkpolicy.yaml
kubectl get networkpolicy -n rag
```

Test from an ad-hoc client:

```bash
kubectl run -n rag test-client \
  --image=busybox:1.36 \
  --rm -it \
  --restart=Never \
  -- wget -T 3 -O- http://rag-api:8000/health
```

Whether traffic is actually blocked depends on the CNI/data plane used by the
cluster.

The lesson is:

> Never stop at `kubectl get networkpolicy`. Verify the behavior.

If you create a separate kind cluster with a policy-aware CNI such as Calico or
Cilium, repeat the same traffic test and compare.

---

# 14. Prove PostgreSQL data survives Pod replacement

## Goal

Separate persistent storage from Pod lifetime.

Create or identify data in PostgreSQL.

Then:

```bash
kubectl get pvc -n rag
kubectl delete pod postgres-0 -n rag
kubectl wait --for=condition=ready pod/postgres-0 -n rag --timeout=120s
kubectl get pvc -n rag
```

Verify that the data still exists.

Mental model:

```text
Pod
    disposable

PVC
    separate durable storage claim
```

A StatefulSet reconnects the correct Pod ordinal to the same claim.

---

# 15. Observe process-local state problems after scaling

## Goal

Understand why "works with 1 replica" does not imply "works with 3 replicas."

Scale:

```bash
kubectl scale deployment rag-api -n rag --replicas=3
```

In this codebase, inspect the known process-local areas:

- synthetic MCP business state;
- in-memory rate limiter;
- Pod-local uploaded files;
- per-Pod embedding model memory.

The distributed-systems question is:

```text
Which state must every replica agree on?
```

If the answer is "all replicas need the same value," process-local Python
memory is usually not enough.

Potential production fixes depend on the type of state:

```text
business state
    -> database/shared service

rate limit counters
    -> shared store such as Redis

uploaded documents
    -> object storage/shared durable storage

large shared model serving
    -> separate model-serving service
```

Do not introduce these technologies merely because Kubernetes exists. Introduce
them when horizontal scaling creates a concrete requirement.

Restore:

```bash
kubectl scale deployment rag-api -n rag --replicas=1
```

---

# 16. Debugging drill: Pod is Pending

Run:

```bash
kubectl get pods -n rag
kubectl describe pod <pod-name> -n rag
kubectl get events -n rag --sort-by=.lastTimestamp
```

Ask:

- was it scheduled?
- are resource requests too high?
- is a PVC unavailable?
- is there an affinity/taint constraint?
- is an init container still running?

Interview pattern:

> Start with state and Events before guessing.

---

# 17. Debugging drill: Pod is Running but traffic fails

Run:

```bash
kubectl get pods -n rag -o wide
kubectl get svc -n rag
kubectl get endpointslices -n rag
kubectl describe svc rag-api -n rag
kubectl describe pod <pod-name> -n rag
```

Then check:

```text
ready?
Service selector correct?
targetPort correct?
EndpointSlice populated?
Ingress correct?
NetworkPolicy blocking?
application itself listening?
```

This is a much stronger debugging answer than saying "I would check the logs"
for every Kubernetes problem.

---

# 18. Inspection toolkit

Use these frequently:

```bash
kubectl get pods -n rag
kubectl get pods -n rag -o wide
kubectl get deployment -n rag
kubectl get rs -n rag
kubectl get statefulset -n rag
kubectl get svc -n rag
kubectl get endpointslices -n rag
kubectl get pvc -n rag
kubectl get ingress -n rag
kubectl get hpa -n rag
kubectl get events -n rag --sort-by=.lastTimestamp

kubectl describe pod <pod> -n rag
kubectl describe deployment rag-api -n rag

kubectl logs <pod> -n rag
kubectl logs <pod> -n rag --previous

kubectl exec -it <pod> -n rag -- sh

kubectl top pods -n rag
kubectl top nodes
```

---

# 19. Interview recap: what each failure teaches

| Exercise | What it proves |
|---|---|
| Delete API Pod | reconciliation/self-healing |
| Delete Postgres Pod | stable StatefulSet identity + PVC |
| Scale 1 -> 3 | replicas + Service backend changes |
| DNS/ClusterIP inspection | service discovery |
| Postgres outage | current readiness-contract gap |
| Broken readiness path | Service traffic removal |
| Rolling restart | ReplicaSets + rollout strategy |
| Bad image | rollout failure + rollback |
| Liveness failure | container restart semantics |
| HPA | controller changing Deployment replicas |
| Node failure | node health and failover uncertainty |
| NetworkPolicy | API object vs data-plane enforcement |
| PVC persistence | storage outlives Pod |
| Multi-replica state | distributed-state correctness |

If you can explain every row without memorized wording, you are in a strong
position for a Kubernetes-focused interview.
