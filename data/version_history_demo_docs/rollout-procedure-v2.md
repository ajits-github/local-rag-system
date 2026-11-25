---
title: "Acme Rollout Procedure v2"
owner: "Acme Governance"
tenant_id: "acme"
allowed_roles:
  - "acme_viewer"
  - "acme_operator"
  - "acme_admin"
classification: "internal"
status: "active"
effective_from: "2026-07-01"
document_version: "2.0"
supersedes: "rollout-procedure-v1.md"
trust_level: "authoritative"
---
# Acme Rollout Procedure (v2.0)

Production rollouts start with a 10 percent canary observed for 20 minutes, then a rolling promotion.

| Setting | Value |
|---|---|
| Strategy | Canary, then rolling promotion |
| Initial traffic | 10 percent |
| Observation window | 20 minutes |
