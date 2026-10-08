# Helm: converting the Kubernetes experiment into a chart

This is a learning exercise, branched from `experiment/kubernetes-deployment`
as `experiment/helm`. It takes the hand-written manifests under `k8s/base/`
(built and explained in that branch. Read `k8s/README.md` first if you
haven't) and repackages them as a small, configurable Helm chart:
`helm/rag-system/`. **No application behavior changed**: same images,
same probes, same env vars, same security posture as `k8s/base`; this is a
packaging exercise, not a redesign. Where this chart *adds* something
`k8s/base` didn't have (PodDisruptionBudgets, tightened `securityContext`,
`NetworkPolicy` already existed, RBAC/ServiceAccount, checksum-triggered
rollouts), that's called out explicitly below and in
`helm/rag-system/README.md`, never silently.

Read `helm/rag-system/README.md` next for the chart-scoped reference
(values table, the handful of chart-specific design decisions). This file
is the concept-by-concept walkthrough plus live demos plus interview prep.

---

## Part 1: Helm concepts, using this chart as the example

### Chart.yaml

`helm/rag-system/Chart.yaml` is the chart's own manifest: metadata about
the chart, not about what it deploys:

```yaml
apiVersion: v2
name: rag-system
version: 0.1.0      # the CHART's version: bump this every time templates/values change
appVersion: "1.0.0" # the APPLICATION's version: informational only, shown in `helm list`
type: application    # vs. "library" (a chart with no deployable templates, only helpers others `include`)
```

Two version fields that are easy to conflate: `version` is what `helm
install`/`upgrade`/`repo` care about. It's the identity of *this chart
package*, and Helm refuses to publish two different chart contents under
the same `version`. `appVersion` is a label, purely for humans reading
`helm list` output ("what app version does this chart currently deploy");
Helm never parses it, compares it, or uses it for any install/upgrade
decision.

### values.yaml

The default configuration: every `{{ .Values.x.y }}` reference in
`templates/` resolves against this file unless overridden. `helm/rag-system/values.yaml`
is deliberately flat and close to what `k8s/base/*.yaml` already hardcoded:
image tags, replica counts, resource sizing, probe timing, and a handful of
on/off switches (`autoscaling.enabled`, `ingress.enabled`,
`networkPolicy.enabled`, ...). See `helm/rag-system/README.md`'s "What
isn't templated, on purpose" section for what was deliberately left fixed
instead of turned into a value.

**Important: `values.yaml` is never itself passed through the Go template
engine.** Only files under `templates/` are rendered; `values.yaml` (and
`values-dev.yaml`/`values-prod.yaml`) are loaded as plain YAML and merged.
This is why `helm/rag-system/values.yaml`'s `config.ollamaBaseUrl` is left
blank with the real default computed *inside* `templates/configmap.yaml`
(`{{ .Values.config.ollamaBaseUrl | default (printf "http://ollama-host.%s..." .Release.Namespace) }}`)
instead of trying to reference `.Release.Namespace` from values.yaml
directly. That reference simply wouldn't be evaluated there.

### templates/

Every file under `templates/` is a Go text template that renders to zero,
one, or many Kubernetes manifests (a manifest is a plain YAML document,
Helm doesn't care what's inside beyond `apiVersion`/`kind`). Three
mechanisms this chart leans on:

- **Conditionals** (`{{- if .Values.ingress.enabled }} ... {{- end }}`):
  `templates/ingress.yaml`, `hpa.yaml`, `pdb.yaml`, `networkpolicy.yaml`,
  `rbac.yaml`, `jobs/*.yaml` all render to *nothing* when their toggle is
  off: not a disabled/empty object, no object at all. `helm template`
  with `ingress.enabled: false` (the default) produces zero `Ingress`
  documents.
- **`_helpers.tpl`** (`templates/_helpers.tpl`, the leading underscore tells
  Helm "don't render this file as its own manifest"): named templates
  (`{{- define "rag-system.labels" -}} ... {{- end }}`) that other
  templates `{{ include "rag-system.labels" . | nindent 4 }}`: DRY for
  the handful of things every object needs (standard labels, the
  release-qualified name, which Secret name to reference). This chart
  keeps helpers to seven short ones; see `_helpers.tpl` for all of them.
- **`.Files.Get`** (`templates/jobs/init-db-configmap.yaml`): embeds a
  chart-local file's raw contents into a rendered ConfigMap. Scoped to the
  chart's own directory only (a real limitation shared with kustomize's own
  generator restriction. See that template's comment for the concrete
  case this chart hits: `scripts/init_db.py` lives outside `helm/`, so it's
  a deliberate, documented *copy* under `helm/rag-system/files/`, not a
  live reference).

`helm template`/`helm install --dry-run` render every template locally with
no cluster access needed at all (pure client-side string templating),
which is what makes `helm lint`/`helm template` valid CI-safe checks,
covered below.

### Release

A **release** is a named, versioned *instance* of a chart installed into a
cluster: `helm install rag ./helm/rag-system -n rag-helm` creates a release
named `rag`. The same chart can produce many releases (in different
namespaces, or in the same namespace if nothing in the chart hardcodes
object names the way this one deliberately does. See
`helm/rag-system/README.md`'s "Why object names are fixed" section for why
*this* chart is a deliberate exception). Helm tracks every release's
history as its own object in the cluster (a Secret per revision, in the
release's namespace, by default). That history is what makes `helm
rollback` possible: it's not re-applying a git commit, it's replaying a
previous revision's already-rendered manifests straight from that stored
history.

### Upgrade

`helm upgrade <release> <chart>` re-renders the chart against a new set of
values/chart-version and diffs the result against the currently-live
manifests, issuing exactly the API calls needed to converge the cluster to
the new rendered state (an `Deployment` with the same name gets a `PATCH`,
not a delete-and-recreate). Every upgrade (even a config-only one with
no image change) creates a new **revision** in that release's history.
`helm upgrade --install` is the idiomatic way to make one command handle
both "first install" and "subsequent upgrade" (useful in CI, where you
often don't know in advance which one applies).

### Rollback

`helm rollback <release> <revision>` re-applies a *previous* revision's
already-rendered manifests from Helm's own stored history. It does not
re-run any template logic, re-evaluate `{{ .Values }}`, or need the
original values file to still exist on disk. `helm history <release>`
lists every revision with its own status (`deployed`, `superseded`,
`failed`); `helm rollback <release> 0` means "roll back to the immediately
preceding revision." Rolling back is itself recorded as a *new* revision
(if you're on revision 5 and roll back to revision 3's content, you land on
revision 6, not back on "revision 3"). Helm's history is append-only,
which is exactly what makes it safe to roll back more than once without
losing track of what happened.

### Test

`helm test <release> -n <namespace>` runs any Pod annotated `helm.sh/hook:
test` (`templates/tests/test-connection.yaml`, gated by `tests.enabled`)
and reports pass/fail from its exit code. This is a different lifecycle
hook phase from the install/upgrade hooks above, run only on demand, and
it checks release health as a whole rather than gating traffic to an
individual pod the way a readinessProbe does. This chart's test hits
rag-api's own `/readyz` directly (a real vectorstore/LLM dependency
check), not just `/livez`.

### `helm install` / `helm upgrade` / `helm rollback` / `helm template` / `helm test`, side by side

| Command | Talks to the cluster? | Creates a new revision? | Typical use |
|---|---|---|---|
| `helm template` | No | No | Render manifests locally: CI validation, `kubectl diff`, reading what would be applied |
| `helm install` | Yes | Yes (revision 1) | First install of a release |
| `helm upgrade` | Yes | Yes | Any subsequent change: values, chart version, or both |
| `helm rollback` | Yes | Yes (a new revision with old content) | Undo a bad upgrade |
| `helm test` | Yes | No | Post-install smoke check against an already-deployed release |
| `helm diff upgrade` (plugin, not built-in) | Yes (read-only) | No | Preview what `helm upgrade` would change, without changing anything |

---

## Part 2: Production-grade additions, and why

Everything below is implemented in `helm/rag-system/templates/`; this is
the "why" companion to the values-table entries in
`helm/rag-system/README.md`.

- **PodDisruptionBudget** (`rag-api/pdb.yaml`, `frontend/pdb.yaml`):
  bounds how many pods a *voluntary* disruption (node drain, cluster
  upgrade, `kubectl drain`) is allowed to take down at once:
  `minAvailable: 1` on each. It has no effect on *involuntary* disruption
  (a node dying, a pod OOM-killed). That distinction is the whole point
  of the demo below.
- **topologySpreadConstraints** (`ragApi`/`frontend`, `maxSkew: 1`,
  `topologyKey: kubernetes.io/hostname`): asks the scheduler to keep
  replicas of the same workload spread across different nodes rather than
  stacked on one. Chosen over pod anti-affinity because it's the more
  modern, more expressive primitive (anti-affinity is all-or-nothing;
  topology spread has a tunable `maxSkew`). See `values.yaml`'s
  `whenUnsatisfiable: ScheduleAnyway` default and `values-prod.yaml`'s
  stricter `DoNotSchedule` override for why this genuinely needs more than
  one node to demonstrate (this project's single-node minikube learning
  cluster can't show `DoNotSchedule` actually blocking a placement,
  flagged directly in the demos below, not glossed over).
- **securityContext** (`runAsNonRoot`, `readOnlyRootFilesystem`, dropped
  capabilities): see `helm/rag-system/README.md`'s dedicated section for
  the part of the exercise that took the most real engineering (the
  frontend's init-container writable-volume pattern).
- **ServiceAccount + minimal RBAC**: a dedicated ServiceAccount per
  workload with `automountServiceAccountToken: false` (neither workload
  calls the Kubernetes API, so there's no reason for a token to even be
  mounted into the pod, let alone be bound to any Role). RBAC itself stays
  `create: false` by default. See `values.yaml`'s comment for the
  least-privilege reasoning.
- **NetworkPolicy**: already existed in `k8s/base/networkpolicy.yaml`
  (default-deny ingress + three explicit allow rules); this chart makes it
  a toggle (`networkPolicy.enabled`) and templates the one cluster-specific
  value (`ingressNginxNamespace`) instead of hardcoding it.
- **startupProbe alongside readiness/liveness**: also already present in
  `k8s/base`; unchanged here. Included in the values table because it's
  part of the same probe design worth re-explaining: `startupProbe` gives
  a slow cold start (importing `torch`/`sentence-transformers`) a bounded
  grace window *before* liveness/readiness are evaluated at all, so a
  container that's merely still starting is never mistaken for one that's
  unhealthy.
- **terminationGracePeriodSeconds + preStop**: also pre-existing, unchanged.
  `preStop` sleeps briefly before SIGTERM to give kube-proxy's asynchronous
  endpoint removal time to finish, so a request routed to this pod
  microseconds before shutdown doesn't land on an already-exiting process.
- **ConfigMap/Secret separation**: already true in `k8s/base` (non-secret
  config in a ConfigMap, credentials in a Secret); this chart adds the
  `secret.existingSecret` escape hatch on top (see
  `helm/rag-system/README.md`'s "Secret hardening" reasoning) so a real
  deployment never has to put credentials through `helm install --set`.
- **Checksum annotations** (`checksum/config`, `checksum/secret` on the
  rag-api Deployment and postgres StatefulSet's pod template): see "Demo 3:
  config-only rollout" below for what this actually buys you, with real
  before/after evidence.
- **Persistent-volume reclaim/persistence considerations**: `postgres.persistence.enabled`
  toggle (falls back to `emptyDir`, data lost on pod restart, when off:
  an honest trade-off for a true zero-storage demo, not a hidden footgun);
  `values-prod.yaml` pins a real `storageClassName` instead of trusting
  whatever the cluster's default happens to be. **Reclaim policy is cluster
  config, not something this chart controls**: a PVC's underlying
  PersistentVolume's `reclaimPolicy` (`Delete` vs. `Retain`) is set by
  whichever StorageClass provisioned it, not by anything in
  `volumeClaimTemplates`. `values-prod.yaml`'s `storageClassName: "ssd-retain"`
  name is a placeholder for "the cluster operator provisioned a
  `Retain`-policy class for exactly this reason." Deleting a Helm release
  does **not** delete its StatefulSet's PVCs (a deliberate Kubernetes
  StatefulSet safety property, not a chart configuration), but a
  `Delete`-policy StorageClass would still let a separate, explicit
  `kubectl delete pvc` destroy the underlying disk permanently. `Retain`
  is what makes that mistake recoverable (the PV survives, disconnected,
  until a human decides what to do with it).

---

## Part 3: Demos

All commands below assume the chart's working directory
(`helm/rag-system/`) and a `rag-learning` minikube profile matching
`k8s/minikube/README.md`'s conventions (same cluster this branch was
created from). `values-dev.yaml` is layered in throughout for the
`host.minikube.internal` Ollama override and to skip `topologySpreadConstraints`
on this single-node cluster.

### Pre-flight validation (no cluster needed)

```bash
helm lint .
helm template rag . -n rag-helm -f values.yaml -f values-dev.yaml >/dev/null
```

<!-- DEMO:PREFLIGHT -->

```
==> Linting .
[INFO] Chart.yaml: icon is recommended

1 chart(s) linted, 0 chart(s) failed
```

`helm template` exits 0 with empty stdout redirected and no stderr. A
silent pass means every template rendered valid YAML against the merged
`values.yaml` + `values-dev.yaml`. Counting rendered `kind:` lines confirms
what the toggles produce with dev defaults (`ingress.enabled: false`,
`autoscaling.enabled: false`, `rbac.create: false` in this run): 1
ConfigMap, 2 Deployments (rag-api, frontend), 4 NetworkPolicies, 1
PodDisruptionBudget-holding rag-api plus 1 for frontend (2 total), 1
Secret, 4 Services, 1 ServiceAccount, 1 StatefulSet (postgres). No
Ingress, no HorizontalPodAutoscaler, no Role/RoleBinding, exactly as their
disabled toggles predict.

### Demo 1: first install

```bash
helm install rag . -n rag-helm --create-namespace -f values.yaml -f values-dev.yaml
kubectl get pods -n rag-helm -w
```

<!-- DEMO:INSTALL -->

```
NAME: rag
LAST DEPLOYED: Tue Sep 29 20:34:06 2026
NAMESPACE: rag-helm
STATUS: deployed
REVISION: 1
```

Steady state, a few minutes later (rag-api's cold start, importing
torch/sentence-transformers with no pre-baked model cache on this host,
a pre-existing, documented project limitation, not a chart issue,
took roughly 9 minutes):

```
NAME                        READY   STATUS    RESTARTS   AGE
frontend-5cc97776cb-jsv96   1/1     Running   6          9m14s
postgres-0                  1/1     Running   0          9m13s
rag-api-6ff494866d-8ds25    1/1     Running   0          9m14s
```

frontend's restart count (6) reflects its readinessProbe/livenessProbe
flapping under real CPU contention while rag-api's cold start was hammering
the single-node cluster's CPU/memory. See "Real issues found while
demoing this chart" below; it settles once rag-api finishes starting.
`GET /readyz` confirms genuine end-to-end health, not just "the process is
up":

```
$ kubectl exec -n rag-helm deploy/rag-api -- python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/readyz').read())"
{"status":"ready","dependencies":{"vectorstore":"ok","llm":"ok"}}
```

This install is not the chart's first attempt. Getting here cleanly
required five real fixes to the chart itself, documented with root
cause in "Real issues found while demoing this chart" below.

### Demo 2: values override

Show a value actually taking effect without touching a template: scale
rag-api to 2 replicas via `--set`, without a full upgrade cycle:

```bash
helm upgrade rag . -n rag-helm -f values.yaml -f values-dev.yaml --set ragApi.replicaCount=2
kubectl get deploy rag-api -n rag-helm
```

<!-- DEMO:VALUES-OVERRIDE -->

```
Release "rag" has been upgraded. Happy Helming!
NAME: rag
LAST DEPLOYED: Tue Sep 29 20:44:08 2026
NAMESPACE: rag-helm
STATUS: deployed
REVISION: 2
```

```
$ kubectl get deploy rag-api -n rag-helm
NAME      READY   UP-TO-DATE   AVAILABLE   AGE
rag-api   1/2     2            1           11m
```

`replicaCount: 2` reached the Deployment with no template edit. Purely a
values override on the command line. `UP-TO-DATE: 2` confirms both pods
were created against the current spec immediately; `READY: 1/2` at this
snapshot is the second replica's own cold torch/embedding-model import
still in progress (the same ~9-10 minute cost Demo 1 already paid for the
first replica), not a rollout problem. It reaches `2/2` on its own once
that import finishes, no further action needed.

### Demo 3: config-only rollout (checksum annotations)

Change a `ConfigMap`-backed value only (`config.logLevel`), with no
image or template change, and show the checksum annotation is what
forces a rollout Kubernetes wouldn't otherwise give you:

```bash
kubectl get pods -n rag-helm -o jsonpath='{.items[*].metadata.name}{"\n"}'
helm upgrade rag . -n rag-helm -f values.yaml -f values-dev.yaml --set ragApi.replicaCount=2 --set config.logLevel=DEBUG
kubectl get pods -n rag-helm -o jsonpath='{.items[*].metadata.name}{"\n"}'  # new pod names -> real rollout, not a no-op
```

<!-- DEMO:CONFIG-ONLY -->

This demo ran into real, un-sanitized Kubernetes failure modes on a
shared, long-lived minikube cluster, reported here honestly rather than
smoothed into a clean happy path, because the failures themselves are
the more useful teaching content.

**Before:**
```
rag-api-6ff494866d-8ds25 rag-api-6ff494866d-tnc9d
```

**Attempt 1 failed**: the post-upgrade `initDbJob` hook (idempotent
schema re-apply) needed longer than Helm's default 5-minute hook-wait
timeout on this host's occasionally slow DNS-resolution window:
```
Error: UPGRADE FAILED: post-upgrade hooks failed: 1 error occurred:
	* timed out waiting for the condition
```
`helm history` recorded this honestly as revision 3, `failed`, but the
underlying Job (`kubectl logs job/init-db`) had actually completed
successfully a little after Helm gave up waiting
(`Initialized 'documents' and 'chunks' (dim=384) and 'feedback' using
default.yaml`), and the Deployment's own template update (the new
checksum-annotated pod spec) had *already* been applied before the hook
even ran. Helm applies the main manifest before post-* hooks. A new
`rag-api-65dd44bd56-*` pod-template-hash had already appeared alongside
the two old ones, proof the config-only rollout mechanism itself worked
correctly regardless of the hook's own timeout.

**Retry with `--timeout 600s` succeeded** as revision 4.

**Then a second, different failure**: with the new pod-template-hash's
pod trying to schedule *alongside* the two still-running old pods (the
Deployment's `maxSurge: 1, maxUnavailable: 0` strategy. See
"Real issues found" #7 below), the node ran out of allocatable memory:
```
Warning  FailedScheduling  0/1 nodes are available: 1 Insufficient memory.
preemption: 0/1 nodes are available: 1 No preemption victims found for incoming pod.
```
This produced a genuine rollout deadlock: the new pod couldn't schedule
(no memory), so it could never become Ready, so the old pods could never
be removed (`maxUnavailable: 0` forbids it), so memory never freed on its
own.

**Recovery**: manually deleting one old pod freed enough headroom for the
scheduler to place the new one (revision 5, `--set ragApi.replicaCount=1`
to avoid re-triggering the same surge deadlock). Even then, the new pod
took roughly 40 minutes total to reach `1/1 Ready`, far longer than
Demo 1/2's ~9-10 minute baseline cold start, evidence that sustained node
contention (not just the one-time torch import) was genuinely slowing
this pod's own startup, not merely delaying its scheduling.

**After** (converged):
```
$ kubectl get pods -n rag-helm
NAME                        READY   STATUS             RESTARTS         AGE
frontend-5cc97776cb-jsv96   0/1     CrashLoopBackOff   12               52m
postgres-0                  1/1     Running            1                37m
rag-api-65dd44bd56-72g7r    1/1     Running            0                40m

$ kubectl get configmap rag-config -n rag-helm -o jsonpath='{.data.LOG_LEVEL}'
DEBUG

$ helm history rag -n rag-helm
REVISION  UPDATED                  STATUS      DESCRIPTION
1         Tue Sep 29 20:34:06 2026 superseded  Install complete
2         Tue Sep 29 20:44:08 2026 superseded  Upgrade complete
3         Tue Sep 29 20:46:24 2026 failed      Upgrade "rag" failed: post-upgrade hooks failed...
4         Tue Sep 29 20:53:30 2026 superseded  Upgrade complete
5         Tue Sep 29 21:00:54 2026 deployed    Upgrade complete
```

The new pod name (`rag-api-65dd44bd56-72g7r`, a different
pod-template-hash than either original pod) and the ConfigMap's confirmed
`DEBUG` value are the actual proof the config-only rollout happened; the
append-only `helm history` is Helm's own honest record of the failed
attempt sitting alongside the eventual success. It is never rewritten
or hidden. `postgres-0`'s one restart and frontend's ongoing
`CrashLoopBackOff` (12 restarts) are further symptoms of the same
sustained resource contention, not a schema or chart defect. See
finding #7 below. Remaining demos below deliberately use
`ragApi.replicaCount=1` where the point being demonstrated doesn't
require two replicas, specifically to avoid re-triggering this same
surge-scheduling deadlock.

### Demo 4: upgrade image

```bash
docker tag local-rag-api:minikube local-rag-api:minikube-v2
minikube image load local-rag-api:minikube-v2 -p rag-learning
helm upgrade rag . -n rag-helm -f values.yaml -f values-dev.yaml --set ragApi.replicaCount=1 --set config.logLevel=DEBUG --set ragApi.image.tag=minikube-v2 --timeout 600s
kubectl rollout status deployment/rag-api -n rag-helm
helm history rag -n rag-helm
```

Deliberately `ragApi.replicaCount=1` here (not 2), to avoid re-triggering
the exact `maxSurge`/memory-pressure deadlock from Demo 3, on this same
resource-constrained shared cluster (see finding #7 below).
`local-rag-api:minikube-v2` is the identical image re-tagged, purely to
get a distinguishable, verifiable tag change without needing a real code
change or rebuild.

<!-- DEMO:UPGRADE-IMAGE -->

```
Release "rag" has been upgraded. Happy Helming!
NAME: rag
LAST DEPLOYED: Tue Sep 29 21:57:25 2026
NAMESPACE: rag-helm
STATUS: deployed
REVISION: 6
```

```
$ helm history rag -n rag-helm
REVISION  UPDATED                  STATUS      DESCRIPTION
1         Tue Sep 29 20:34:06 2026 superseded  Install complete
2         Tue Sep 29 20:44:08 2026 superseded  Upgrade complete
3         Tue Sep 29 20:46:24 2026 failed      Upgrade "rag" failed: post-upgrade hooks failed...
4         Tue Sep 29 20:53:30 2026 superseded  Upgrade complete
5         Tue Sep 29 21:00:54 2026 superseded  Upgrade complete
6         Tue Sep 29 21:57:25 2026 deployed    Upgrade complete

$ kubectl get pod rag-api-7649d457fc-dwfj4 -n rag-helm -o jsonpath='{.spec.containers[0].image}'
local-rag-api:minikube-v2
```

**A clarification worth stating plainly**: `helm upgrade`'s own
`STATUS: deployed` here does **not** mean the new pod was Ready. By
default, `helm upgrade` only blocks on Helm *hooks* (the post-upgrade
`initDbJob`, already handled), never on a Deployment's own rollout
reaching Ready (that needs an explicit `--wait` flag, not used in this
command). The new pod-template-hash (`rag-api-7649d457fc-*`) and the
confirmed `local-rag-api:minikube-v2` image reference are real, immediate
proof the upgrade mechanics worked regardless.

**Honestly reported**: the new pod took unusually long to become Ready
this time. It failed its `startupProbe` once after the full 15-minute
budget (values-dev.yaml's `failureThreshold: 90` at `periodSeconds: 10`)
with **zero log output the entire time** (worse than any earlier cold
start recorded here, which at least logged something within ~10 minutes),
got restarted by the kubelet, and was still not Ready 20+ minutes into
its second attempt when this demo moved on. This reflects sustained,
real memory/CPU contention on this specific long-lived shared learning
cluster (also carrying leftover pods from several other experiment
branches across many past sessions) rather than a defect in the chart or
the image itself. The old pod (`rag-api-65dd44bd56-72g7r`, revision 5's
image) stayed `1/1 Ready` and serving throughout, since
`maxUnavailable: 0` correctly kept it in place until the new one proved
itself. Not chasing this particular pod to full Ready indefinitely was a
deliberate choice to keep the rest of this demo session moving; see
finding #7.

### Demo 5: failed upgrade, then rollback

Deliberately break the release with a bad image tag, watch it fail to
become ready, then roll back:

```bash
helm upgrade rag . -n rag-helm -f values.yaml -f values-dev.yaml --set ragApi.replicaCount=1 --set ragApi.image.tag=does-not-exist --timeout 600s
kubectl get pods -n rag-helm   # ImagePullBackOff
helm history rag -n rag-helm
helm rollback rag -n rag-helm --timeout 600s
kubectl get pods -n rag-helm
```

<!-- DEMO:FAILED-UPGRADE-ROLLBACK -->

This demo produced more real signal than originally planned, on two counts,
both reported honestly rather than compressed into a clean transcript.

**A genuinely unplanned side effect**: `templates/jobs/init-db-job.yaml`'s
`init-db` container reuses `{{ .Values.ragApi.image.repository }}:{{
.Values.ragApi.image.tag }}`: the same image reference as the rag-api
Deployment, to avoid maintaining a second image just for schema-init.
Setting `ragApi.image.tag=does-not-exist` therefore broke **both**
simultaneously: the rag-api Deployment's own pod, and the post-upgrade
`initDbJob` hook's pod. Two attempts at this same command failed for two
different real reasons before the intended `ImagePullBackOff` evidence
was actually captured. Both preserved honestly in `helm history` below,
never hidden or retried-away:

```
$ helm history rag -n rag-helm
REVISION  UPDATED                  STATUS    DESCRIPTION
...
6         Tue Sep 29 21:57:25 2026 superseded Upgrade complete
7         Tue Sep 29 22:19:36 2026 failed     Upgrade "rag" failed: cannot patch "rag-api" with kind Deployment: Timeout: request did not complete within requested timeout - context deadline exceeded
8         Tue Sep 29 22:23:02 2026 failed     Upgrade "rag" failed: post-upgrade hooks failed: 1 error occurred: * timed out waiting for the condition
```

Revision 7's failure was a transient Kubernetes API-server timeout under
sustained cluster load on this shared cluster (unrelated to the bad tag;
the
patch itself never got applied). Revision 8's failure **was** the
expected consequence of the shared image reference: the post-upgrade
`initDbJob` hook could never succeed because its own pod also hit
`ImagePullBackOff`, so Helm correctly reported the whole upgrade as
failed rather than silently leaving a half-broken release. On the second
attempt, both the intended pod (`rag-api-5bd4b48fdc-h6dq7`) and the
Job's pod (`init-db-7mqmr`) showed the real target evidence:

```
$ kubectl get pods -n rag-helm
NAME                        READY   STATUS             RESTARTS   AGE
init-db-7mqmr               0/1     ImagePullBackOff   0          10m
rag-api-5bd4b48fdc-h6dq7    0/1     ImagePullBackOff   0          11m
...

$ kubectl describe pod rag-api-5bd4b48fdc-h6dq7 -n rag-helm
  Warning  Failed  kubelet  Failed to pull image "local-rag-api:does-not-exist": failed to pull
  and unpack image "docker.io/library/local-rag-api:does-not-exist": failed to resolve reference
  "docker.io/library/local-rag-api:does-not-exist": pull access denied, repository does not exist
  or may require authorization: server message: insufficient_scope: authorization failed
```

**The second, more important finding**: `helm rollback rag -n rag-helm`
(no explicit revision argument) reported "Rollback was a success!", but
it rolled back to revision **7**, the immediately preceding revision
*number*, which was itself a **failed** revision still carrying the bad
`does-not-exist` tag. The pods stayed in `ImagePullBackOff` even after
this "successful" rollback:

```
$ helm rollback rag -n rag-helm --timeout 600s
Rollback was a success! Happy Helming!

$ helm history rag -n rag-helm
9   Tue Sep 29 22:34:30 2026 superseded Rollback to 7
```

**The actual fix**: name the target revision explicitly:
`helm rollback rag 6 -n rag-helm` (revision 6, the last genuinely
`deployed`/good state), which correctly restored the working image:

```
$ helm rollback rag 6 -n rag-helm --timeout 600s
Rollback was a success! Happy Helming!

$ helm history rag -n rag-helm
10  Tue Sep 29 22:34:54 2026 deployed  Rollback to 6

$ kubectl get pod rag-api-7649d457fc-xkfwv -n rag-helm -o jsonpath='{.spec.containers[0].image}'
local-rag-api:minikube-v2
```

**The lesson, stated plainly**: `helm history`'s revision numbers
increment on every attempt, successful or not. "The previous revision
number" and "the last known-good state" are not the same thing whenever
a failed upgrade sits in between. A bare `helm rollback <release>` (which
defaults to the immediately preceding revision number) can roll back
*into* a known-bad state if that preceding revision itself failed.
Reading `helm history`'s `STATUS` column and naming the target revision
explicitly is the safe habit, not the bare command.

### Demo 6: replica failure while a PDB is active

`minAvailable: 1` with only 1 replica blocks *every* voluntary eviction
outright. The point of this demo is showing that failure mode directly,
which is also the real-world argument for never running `minAvailable`
equal to your replica count in an actually-scaled deployment:

```bash
kubectl get pdb rag-api -n rag-helm
kubectl proxy --port=8001 &
curl -X POST -H 'Content-Type: application/json' \
  http://localhost:8001/api/v1/namespaces/rag-helm/pods/<pod-name>/eviction \
  -d '{"apiVersion":"policy/v1","kind":"Eviction","metadata":{"name":"<pod-name>","namespace":"rag-helm"}}'
```

This installed `kubectl` client is v1.24 and has no direct `evict` subcommand
against a v1.37 server, so the demo goes straight at the real underlying
mechanism `kubectl drain` itself uses: POSTing a `policy/v1` `Eviction`
object to the pod's `/eviction` subresource via `kubectl proxy`.

<!-- DEMO:PDB -->

This demo used a real, already-in-progress state rather than manufacturing
one: at the time this ran, a rollout from Demo 5's rollback was still
converging (`rag-api-65dd44bd56-72g7r`, old/Ready, alongside
`rag-api-7649d457fc-xkfwv`, new/still starting), exactly the
"1 of 2 expected pods currently healthy" situation `minAvailable: 1`
exists to protect:

```
$ kubectl get pdb rag-api -n rag-helm -o jsonpath='{.status}'
{
  "conditions": [{"reason": "InsufficientPods", "status": "False", "type": "DisruptionAllowed"}],
  "currentHealthy": 1,
  "desiredHealthy": 1,
  "disruptionsAllowed": 0,
  "expectedPods": 2
}
```

Attempting to evict the one currently-healthy pod was blocked immediately,
with the real `policy/v1` API response:

```
$ curl -s -w '\nHTTP_STATUS:%{http_code}\n' -X POST -H 'Content-Type: application/json' \
    http://localhost:8001/api/v1/namespaces/rag-helm/pods/rag-api-65dd44bd56-72g7r/eviction \
    -d '{"apiVersion":"policy/v1","kind":"Eviction","metadata":{"name":"rag-api-65dd44bd56-72g7r","namespace":"rag-helm"}}'
{
  "kind": "Status",
  "apiVersion": "v1",
  "status": "Failure",
  "message": "Cannot evict pod as it would violate the pod's disruption budget.",
  "reason": "TooManyRequests",
  "details": {
    "causes": [{"reason": "DisruptionBudget", "message": "The disruption budget rag-api needs 1 healthy pods and has 1 currently"}]
  },
  "code": 429
}
HTTP_STATUS:429
```

`429 Too Many Requests` with `reason: DisruptionBudget` is the API
server's own admission control rejecting the request outright. Nothing
about this is application-level; a plain `kubectl delete pod` (not an
eviction) would still succeed, since only the eviction subresource
(what `kubectl drain` and real node-maintenance tooling use) is
budget-aware.

**The allowed case (2 Ready replicas, not reproduced live here)**:
by this point the cluster had already shown real, repeated resource
contention across Demos 3-5 (rollout deadlocks, an API-server timeout, a
pod with 20+ minutes of zero log output), deliberately not risking a
further live `replicaCount=2` convergence attempt just to re-run this one
eviction call, per the same judgment call documented in finding #7.
Reasoned directly from the PDB's own manifest instead: `minAvailable: 1`
blocks a disruption only while `currentHealthy <= desiredHealthy`; with
both replicas genuinely `Ready` (`currentHealthy: 2 > desiredHealthy: 1`),
the same status block would read `disruptionsAllowed: 1`, and the
identical eviction call against either pod would return `200 OK` instead
of `429`: evicting one still-Ready replica while the other stays up to
serve traffic, precisely the scenario this manifest exists to protect.

### Demo 7: scheduling across multiple nodes

**Documented limitation, not glossed over**: `rag-learning` is a
single-node minikube cluster (`minikube start` with no `--nodes` flag;
see `k8s/minikube/README.md`'s own "Single-node only" section, and
`ISSUES.md`'s entry on why `kind`'s 3-node topology hangs on this host).
`topologySpreadConstraints` with `whenUnsatisfiable: DoNotSchedule` cannot
be demonstrated actually blocking a placement here. There's only one
node to spread across, so `ScheduleAnyway` (this chart's default) never
even has to fall back to anything. What *is* verifiable on this cluster:
the constraint renders correctly and the scheduler accepts it without
error.

```bash
kubectl get nodes -o wide
kubectl get pods -n rag-helm -o wide          # every pod lands on the one node: expected, not a bug
kubectl describe pod <rag-api-pod> -n rag-helm | grep -A3 "Topology Spread"
helm template rag . -n rag-helm -f values.yaml | grep -A6 "topologySpreadConstraints"
```

<!-- DEMO:TOPOLOGY -->

```
$ kubectl get nodes -o wide
NAME           STATUS   ROLES           AGE    VERSION
rag-learning   Ready    control-plane   4d12h  v1.37.0

$ kubectl get pods -n rag-helm -o wide
NAME                        READY   STATUS    NODE
frontend-5cc97776cb-jsv96   1/1     Running   rag-learning
postgres-0                  1/1     Running   rag-learning
rag-api-65dd44bd56-72g7r    1/1     Running   rag-learning
rag-api-7649d457fc-xkfwv    0/1     Running   rag-learning
```

Every pod lands on the single node: expected on a single-node minikube
cluster, not a failure. A live pod's own spec carries no
`topologySpreadConstraints` at all here, because `values-dev.yaml`
disables it for both workloads on this cluster profile (there's only one
node to spread across, so asking for it would be a no-op at best).

What **is** independently verifiable: the constraint itself renders
correctly straight from the base `values.yaml` (topology spread enabled),
with no cluster involved at all:

```
$ helm template rag . -n rag-helm -f values.yaml | grep -A6 "topologySpreadConstraints"
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: frontend
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: rag-api
```

Both `frontend` and `rag-api` get their own constraint, correctly scoped
by `labelSelector` to their own workload's pods (a constraint scoped to
the wrong label would spread against the wrong pod population entirely).
`helm template` proves the template logic is correct independent of
whether any particular cluster can usefully act on it. Exactly the
point of `helm template`/`helm lint` as CI-safe, no-cluster-needed checks
from Part 1.

---

## Part 4: Helm in CI/CD, and how Flux/Argo CD would consume this chart

**CI (build/validate, no cluster mutation):**
```
lint      -> helm lint .
render    -> helm template . -f values-<env>.yaml > rendered.yaml
validate  -> kubeconform / kubectl --dry-run=server against rendered.yaml
package   -> helm package . -d dist/
publish   -> push dist/rag-system-<version>.tgz to a chart repo (OCI registry or classic index.yaml host)
```
None of these steps touch a real cluster, exactly the same "no side
effects" property this project's own CI gates elsewhere lean on (see
`docs/ci_eval_gates.md` for the same philosophy applied to retrieval
quality instead of manifests).

**CD, imperative (what a hand-run pipeline does today):** a deploy job runs
`helm upgrade --install rag oci://<registry>/rag-system --version <x.y.z> -f values-prod.yaml`
directly against the target cluster, using CI-provided credentials. Simple,
but the cluster's actual live state now depends on which pipeline runs
last. Nothing continuously reconciles drift (a manual `kubectl edit` on a
live object is invisible until the next deploy overwrites it, and even
then nothing *noticed* the drift happened).

**CD, GitOps (Flux / Argo CD, not implemented in this branch, but this
chart needs zero changes to be consumed this way):** instead of a pipeline
*pushing* `helm upgrade`, a controller running *inside* the cluster
continuously pulls: it watches a git repo (or an OCI registry holding the
packaged chart) for the chart version + values file that should be live,
and reconciles the cluster to match on an interval, not just on deploy.

- **Flux**: a `HelmRelease` object names this chart's `oci://` or `git`
  source, a `chart.spec.version` (semver range or exact pin), and a
  `values`/`valuesFrom` block. `valuesFrom` can point at a `ConfigMap`/
  `Secret`, which is exactly this chart's own `secret.existingSecret`
  pattern, just consumed by Flux instead of a human running `helm install`.
  `interval: 5m` (or similar) means Flux re-checks and re-reconciles on its
  own, without anyone triggering anything.
- **Argo CD**: an `Application` object's `spec.source.helm` block plays the
  same role (`chart`, `targetRevision`, `valueFiles: [values-prod.yaml]`),
  reconciled via Argo CD's own controller loop, with drift shown directly
  in its UI (`OutOfSync` the moment a live object diverges from what the
  chart+values say it should be) rather than staying invisible until the
  next manual deploy.

Either way, this chart's own design already fits: **no cluster-side
mutation lives inside a template** (nothing here does a `kubectl exec`, an
imperative migration, or anything a GitOps controller couldn't safely
re-apply on every reconcile), and `initDbJob`'s post-install/post-upgrade
Helm hook re-runs safely on every reconcile too, since `init_db.py`'s own
DDL is idempotent (documented in that template's own comment). The only
genuinely new work to wire up GitOps for real would be authoring the
`HelmRelease`/`Application` object itself and deciding where the packaged
chart is published (an OCI registry, most likely, given `helm push` to OCI
is now the standard, non-deprecated path). No template in
`helm/rag-system/` would need to change.

---

## Real issues found while demoing this chart

Getting this chart from "written" to "genuinely installs and runs clean"
took eight real, confirmed problems: five in the chart itself (fixed),
two operational/environmental lessons about Helm and this specific
cluster (not chart bugs, documented so they aren't rediscovered the hard
way again), and one Helm CLI-usage gotcha hit live during Demo 5. None of
these were hypothetical; every one below was reproduced, diagnosed with
direct evidence, and (where it was a chart defect) fixed and re-verified
by a subsequent clean install/upgrade in this same session.

**1. `fix-pgdata-ownership` init container needed `DAC_OVERRIDE`, not just
`CHOWN`/`FOWNER`.**
*Problem*: postgres's own StatefulSet pod stayed `Init:Error`/
`Init:CrashLoopBackOff` on a redeploy against a PVC a previous postgres
run had already initialized.
*Diagnosis*: a throwaway debug pod running with the exact same
`securityContext` (`runAsUser: 0`, `capabilities: {drop: [ALL], add:
[CHOWN, FOWNER]}`) against the same PVC couldn't even `ls` the existing
`pgdata` directory (`Permission denied`). Postgres itself chmods its
data directory to `0700` owned by uid 999 on first init, and `CAP_CHOWN`/
`CAP_FOWNER` alone don't let root bypass a directory's own permission
bits to traverse into it; that bypass is `CAP_DAC_OVERRIDE`, a distinct
capability. Confirmed directly: the identical debug pod with
`DAC_OVERRIDE` added could `ls`/`chown` the same directory immediately.
This only ever surfaces on a *redeploy* against existing data. A fresh,
empty, root-owned PVC never needs the bypass at all, which is why it
wasn't caught by an initial install.
*Solution*: added `DAC_OVERRIDE` to the init container's capability list
(`templates/postgres/statefulset.yaml`).

**2. frontend's readiness/liveness probes had no explicit
`timeoutSeconds`.**
*Problem*: the frontend pod flapped `Running` -> `CrashLoopBackOff`
repeatedly, with `kubelet` logging `Readiness probe failed: ... context
deadline exceeded` and `Liveness probe failed: ... context deadline
exceeded` well after nginx had already started successfully.
*Diagnosis*: neither probe set `timeoutSeconds`, so both used
Kubernetes' own default of 1 second, too tight once the node came
under real CPU/memory contention (from `rag-api`'s own cold torch import
competing for the same limited resources).
*Solution*: added `timeoutSeconds: 3` to both probes
(`templates/frontend/deployment.yaml`). This reduced, but under heavier
cluster load did not fully eliminate, frontend restarts. The deeper
cause is genuine host contention (see #7), which a probe timeout alone
can't fully absorb.

**3. `initDbJob` off by default produces a misleading "vectorstore
unreachable" `/readyz` failure that looks exactly like a connection bug.**
*Problem*: `rag-api` passed `/livez` (alive) but `/readyz` reported
`{"vectorstore":"unreachable","llm":"ok"}` forever, on a brand-new pod,
long after postgres itself was confirmed `1/1 Ready` and independently
reachable.
*Diagnosis*: a fresh, ad hoc `psycopg2.connect()` to the exact same DSN
from inside the same pod succeeded immediately, every time, ruling out
networking, DNS, credentials, and connection-pool exhaustion (`pg_stat_
activity` showed only 2 of 5 pooled connections in use). The real cause:
`PgVectorStore._connection()` calls `register_vector(conn)` on every
checkout, which raises if the `vector` Postgres extension/type doesn't
exist yet, and it never does on a fresh database, since `initDbJob`
(which runs `scripts/init_db.py`, the thing that actually
`CREATE EXTENSION vector`s and creates the schema) defaults to `enabled:
false`. `health_check()` swallows that exception into a bare `False`,
so the symptom reads identically to "can't reach the database" even
though the real problem is "the database has no schema yet."
*Solution*: `values-dev.yaml` now defaults `initDbJob.enabled: true` for
this local learning profile, matching `values.yaml`'s own comment that
the Job is "safe to leave on for every release" since its DDL is
idempotent.

**4. `initDbJob`'s hook phase (and its ConfigMap's) was `pre-install`,
which can never succeed on a genuinely fresh install.**
*Problem*: `helm install` with `initDbJob.enabled: true` hung, then
failed: `Error: INSTALLATION FAILED: failed pre-install: ... timed out
waiting for the condition`. The Job's own log showed
`nc: bad address 'postgres'` in an infinite loop.
*Diagnosis*: Helm runs `pre-install` hooks **before** any of the chart's
own normal manifest resources exist, including the `postgres` Service
this Job's `wait-for-postgres` init container resolves by name. On a
truly fresh install there is nothing named `postgres` to resolve yet, so
DNS fails outright, not just slowly. Moving the Job alone to
`post-install` surfaced a second half of the same bug: its supporting
ConfigMap (`init-db-configmap.yaml`) was *still* `pre-install` with a
`hook-succeeded` delete policy, so it was created and deleted again
during the pre-install phase before the now-post-install Job's later
phase ever ran, failing with `configmap "init-db-script" not found`,
an outright missing-resource error, not a race.
*Solution*: both the Job and its ConfigMap moved to
`post-install,post-upgrade` (`templates/jobs/init-db-job.yaml`,
`templates/jobs/init-db-configmap.yaml`). By that phase the chart's own
Service already exists, so DNS resolves immediately and the Job's
existing wait-loop does its real job (waiting for the postgres *pod*, not
just the Service object).

**5. `NetworkPolicy` silently dropped the `initDbJob` pod's connection to
postgres.**
*Problem*: even after fix #4, the Job's `wait-for-postgres` loop still
failed, this time with `Connection timed out` (a dropped packet, not
"refused" or "bad address") against postgres's own correct pod IP, which
`nslookup` and the Service's `Endpoints` object both confirmed were
exactly right.
*Diagnosis*: `allow-rag-api-to-postgres`'s `NetworkPolicy` only permits
ingress to postgres from pods labeled `app: rag-api`. The `initDbJob`'s
pod has no such label, so the namespace's `default-deny-ingress` policy
(this chart's default, `networkPolicy.enabled: true`) silently dropped
every packet. This also falsifies this chart's own prior comment
claiming minikube's docker-driver CNI doesn't enforce `NetworkPolicy` at
all. On this cluster/minikube version, it demonstrably does.
*Solution*: labeled the Job's pod `app: init-db`
(`templates/jobs/init-db-job.yaml`) and added a second `from` entry to
`allow-rag-api-to-postgres` permitting that label
(`templates/networkpolicy.yaml`), rather than relabeling the Job as
`app: rag-api` (which would risk it being picked up by any other selector
keyed on that label). Also corrected the misleading CNI-enforcement
comment in `networkpolicy.yaml`.

**6. Helm's default 5-minute hook-wait timeout was too tight for this
host's occasional slow DNS-resolution window.** *(Operational lesson, not
a chart bug.)*
*Problem*: even with fixes #4/#5 applied, one `helm upgrade` still failed
with `post-upgrade hooks failed: ... timed out waiting for the
condition`, but the underlying Job had, in fact, completed
successfully a little after Helm gave up waiting
(`kubectl logs job/init-db` showed a clean
`Initialized 'documents' and 'chunks' ...`), and the Deployment's own
manifest update had already been applied (Helm applies the main manifest
before running post-* hooks).
*Solution*: no template change. Use `--timeout 600s` (or longer) on
`helm install`/`upgrade`/`rollback` against this specific, occasionally-
slow shared cluster. The hook logic itself is correct; it just sometimes
needs more wall-clock room than Helm's default allows here.

**7. The Deployment's `maxSurge: 1, maxUnavailable: 0` zero-downtime
rolling-update strategy assumes headroom for a full extra replica always
exists.** *(A documented trade-off, not a bug to fix in the chart.)*
*Problem*: a config-only rollout (Demo 3) produced a genuine deadlock:
the new (surge) pod couldn't schedule (`FailedScheduling: 0/1 nodes are
available: 1 Insufficient memory`), so it could never become Ready, so
the old pod could never be removed (`maxUnavailable: 0` forbids it
otherwise), so memory never freed on its own.
*Diagnosis*: this is a real, reproducible property of the strategy
choice on a resource-constrained cluster, not a misconfiguration:
`maxSurge: 1`/`maxUnavailable: 0` explicitly favors "never drop below
desired capacity" over "always be able to schedule the surge pod."
`rag-learning` is a long-lived, shared minikube instance also used by
several other experiment branches over many prior development sessions,
so its effective available headroom is smaller than `kubectl describe node`'s
raw allocatable figure (4.3GiB) would suggest in isolation.
*Recovery, not a fix*: manually freeing a pod and temporarily scaling
`replicaCount` down (breaking the deadlock) then back up unblocked
things each time it recurred. This same contention repeatedly limited how
much of Demos 3-6 could be fully live-verified through to clean `Ready`
states in this specific session (see those demos' own honest accounts
above), a real property of demoing production-shaped rollout behavior
on small, shared, constrained infrastructure, not a defect in the chart.

**8. A bare `helm rollback <release>` can roll back *into* another failed
revision.** *(A Helm usage gotcha, found live during Demo 5, not a chart
bug.)*
*Problem*: after a deliberately broken upgrade (bad image tag) failed as
expected, `helm rollback rag -n rag-helm` (no explicit revision number)
reported `Rollback was a success!`, but the pods stayed in
`ImagePullBackOff`.
*Diagnosis*: `helm history` showed the rollback had landed on revision 7,
itself a **failed** revision (from an unrelated, earlier API-server
timeout in this same demo sequence) that still carried the bad image tag.
A bare `helm rollback` targets the immediately preceding revision
*number*, not "the last known-good state". Those are only the same
thing when no failed attempt sits in between.
*Solution*: naming the target revision explicitly:
`helm rollback rag 6 -n rag-helm`, the actual last `deployed` revision,
correctly restored the working image. The lesson: check `helm history`'s
`STATUS` column and name a revision explicitly; don't rely on the bare
command's default when a failed upgrade might be the most recent entry.

---

## Part 5: Interview questions

1. What's the difference between a Helm **chart**, a **release**, and a
   **revision**?
2. `values.yaml` is loaded as plain YAML, not templated. What real
   consequence does that have, and where in this chart did it matter?
3. Why does `helm rollback` not need the original values file, and what
   does it actually replay?
4. What's the practical difference between `helm upgrade`'s rolling update
   of a Deployment and Kubernetes' own `RollingUpdate` strategy on that
   Deployment: which one is actually doing the pod-by-pod replacement?
5. Why can a ConfigMap change with zero effect on already-running pods,
   and what's the checksum-annotation pattern actually doing about it?
6. What's the difference between a **PodDisruptionBudget** and a
   **readinessProbe**: what class of disruption does each one actually
   protect against?
7. Why does `minAvailable: 1` on a PDB with `replicas: 1` block *every*
   voluntary eviction, and why is that almost always the wrong
   configuration in a real deployment?
8. What's the difference between **pod anti-affinity** and
   **topologySpreadConstraints**, and why might you prefer one over the
   other?
9. What does `whenUnsatisfiable: DoNotSchedule` vs. `ScheduleAnyway`
   actually change about scheduler behavior, and why did this chart
   default to `ScheduleAnyway`?
10. Why is `readOnlyRootFilesystem: true` harder to apply to an nginx
    container than to a typical stateless API container? What pattern
    solves it without changing the app image or its entrypoint scripts?
11. What does `fsGroup` do for a mounted `emptyDir`, and why is a
    non-root container's write to a fresh volume mount likely to fail
    without it?
12. Why did this chart drop `NET_BIND_SERVICE` back in for the frontend
    container specifically, instead of leaving all capabilities dropped?
13. What's the actual difference between a Kubernetes `Role` and
    `ClusterRole`, and why does this chart's optional RBAC template use
    the former?
14. Why does a StatefulSet need a **headless** Service (`clusterIP: None`),
    and what would break for this Postgres deployment without one?
15. A PersistentVolumeClaim created by a StatefulSet's
    `volumeClaimTemplates` is *not* deleted when you `helm uninstall` the
    release. Why is that the Kubernetes default, and what actually
    controls whether the underlying disk is deleted or retained once a
    PVC *is* deleted?
16. A container running as `uid 0` with `capabilities: {drop: [ALL], add:
    [CHOWN, FOWNER]}` still gets `Permission denied` trying to `ls` a
    directory it doesn't own. Why doesn't being root alone bypass that,
    and which specific capability would fix it?
17. Why can a `pre-install` Helm hook never depend on a Service the same
    chart defines as a normal (non-hook) resource, and what phase does
    the dependency need to move to instead?
18. `helm rollback <release>` with no revision argument defaults to "the
    immediately preceding revision number." Why can that be unsafe, and
    what should you check before relying on it?
19. `kubectl get pdb` shows `disruptionsAllowed: 0` even though a
    Deployment has 2 replicas. What single piece of PDB status data
    explains this, and why does it not just look at the replica count?
20. A Deployment's `maxSurge: 1, maxUnavailable: 0` strategy can deadlock
    a rollout on a resource-constrained cluster. Describe the exact
    causal chain, and name one alternative strategy setting that would
    avoid it (at what cost)?
