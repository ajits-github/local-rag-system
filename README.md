# Helm chart experiment (branch: `experiment/helm`)

You're viewing the `experiment/helm` branch, a self-contained learning
exercise, not the production system. For the production
`local-rag-system` README, see
[`main`](https://github.com/ajits-github/local-rag-system/blob/main/README.md).

![Helm chart for the Kubernetes deployment experiment: chart anatomy
(Chart.yaml, values.yaml, templates/), the configurable values that
render into standard Kubernetes resources (Deployment, Service,
ConfigMap, Secret, HPA, Ingress, PVC), and the install/upgrade/rollback
lifecycle](helm/docs/assets/architecture-diagram.png)

A learning exercise, branched from `experiment/kubernetes-deployment`,
that repackages the hand-written manifests under `k8s/base/` as a small,
configurable Helm chart (`helm/rag-system/`). No application behavior
changes: same images, probes, env vars, and security posture as
`k8s/base`; this is a packaging exercise, not a redesign.

See [`helm/README.md`](helm/README.md) for the full writeup: Helm
concepts taught against this chart, live demos with real captured
output (install, values override, upgrade, broken upgrade + rollback,
PDB-blocked eviction), and an interview-question reference. See
[`helm/rag-system/README.md`](helm/rag-system/README.md) for the
chart-scoped values reference.
