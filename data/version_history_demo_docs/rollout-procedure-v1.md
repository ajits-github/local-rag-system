---
title: "Acme Rollout Procedure v1"
owner: "Acme Governance"
tenant_id: "acme"
allowed_roles:
  - "acme_viewer"
  - "acme_operator"
  - "acme_admin"
classification: "internal"
status: "superseded"
effective_from: "2026-02-01"
document_version: "1.0"
trust_level: "authoritative"
---
# Acme Rollout Procedure (v1.0)

Production rollouts use a direct rolling update that sends 100 percent of traffic to the new release, observed for 10 minutes.

| Setting | Value |
|---|---|
| Strategy | Rolling update |
| Initial traffic | 100 percent |
| Observation window | 10 minutes |
