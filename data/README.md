# TechFusion GmbH Synthetic Knowledge Base

This repository contains an entirely fictional, publication-safe internal knowledge base for evaluating a local enterprise RAG system. It does not describe a real company or contain real credentials, domains, customer data, or proprietary material.

## A. Final file tree

```text
knowledge_base/
├── engineering/
│   ├── api-development-guidelines.md
│   ├── coding-standards.md
│   ├── deployment-process.md
│   └── database-migrations.md
├── architecture/
│   ├── system-overview.md
│   ├── document-processing-pipeline.md
│   ├── rag-service-architecture.md
│   └── data-storage.md
├── security/
│   ├── access-control-policy.md
│   ├── data-encryption.md
│   ├── incident-response-policy.md
│   └── gdpr-data-handling.md
├── operations/
│   ├── monitoring-and-alerting.md
│   └── backup-and-recovery.md
├── runbooks/
│   ├── document-service-outage.md
│   ├── postgres-recovery.md
│   └── failed-kubernetes-deployment.md
├── policies/
│   ├── data-retention-policy.md
│   └── remote-work-policy.md
└── hr/
    ├── employee-onboarding.md
    └── annual-leave-policy.md
evaluation/
└── evaluation_dataset.jsonl
```

## B. Company facts sheet

| Area | Canonical fact |
|---|---|
| Company | TechFusion GmbH, Berlin-based B2B SaaS company, approximately 180 employees |
| Product | DocuFlow: document classification, OCR, structured extraction, case routing, compliance evidence |
| Teams | Platform Engineering, Document Intelligence, Workflow, Trust and Compliance, Site Reliability Engineering (SRE), IT Operations, Customer Operations, People Operations |
| Core services | Gateway, Ingest API, Document Service, OCR Worker, Classifier, Extraction Worker, Workflow Engine, Compliance Ledger, Notification Service, RAG Service |
| Infrastructure | AWS eu-central-1; Kubernetes on EKS; PostgreSQL 16; S3-compatible object storage; Redis; OpenSearch; Kafka; Prometheus, Grafana, Loki and PagerDuty |
| Environments | local, development (dev), staging (stg), and production (prod) |
| Identity | Corporate SSO, phishing-resistant MFA, workload identities, group-based RBAC |
| Production region | AWS eu-central-1, three availability zones |
| API SLO | 99.9% monthly availability |
| Pipeline objective | 99% of eligible documents complete within 15 minutes |
| Disaster recovery | RPO 15 minutes; regional RTO 4 hours |
| Default document retention | 90 days after workflow completion; configurable 30 to 365 days |
| Severity model | SEV-1 to SEV-4; acknowledgments 5 min, 15 min, 4 business hours, 2 business days |

### Roles and escalation

- Incident Commander coordinates incidents and authorizes recovery or exceptional emergency action.
- Service teams own application behavior, validation, dashboards, and runbooks.
- SRE owns EKS, Argo CD, observability, deployment platform, and reliability coordination.
- Data Platform owns PostgreSQL, backups, migrations guidance, and database recovery.
- Security Engineering owns identity controls, encryption requirements, and security response.
- Trust and Compliance owns retention rules, compliance requirements, and legal-hold approval.
- Data Protection Officer advises independently on GDPR and breach assessment.
- People Operations owns employment lifecycle, leave, and remote-work administration.

### Naming conventions

- Environments: `local`, `dev`, `stg`, `prod`.
- External APIs: `/api/v1`; internal APIs: `/internal/v1`.
- Events: lower-case domain action plus version, for example `document.accepted.v1`.
- Public statuses: `UPLOADED`, `PROCESSING`, `REVIEW_REQUIRED`, `COMPLETED`, `FAILED`.
- Identifiers: opaque UUIDs; object keys never include personal data or filenames.
- Services use title case in prose and lowercase hyphenated names in deployment resources.

## C. Consistency and cross-reference plan

| Canonical subject | Primary document | Supporting references |
|---|---|---|
| Service topology and ownership | `architecture/system-overview.md` | pipeline, outage, storage, monitoring |
| Document stages and thresholds | `architecture/document-processing-pipeline.md` | system overview, outage, retention |
| RAG retrieval and evaluation | `architecture/rag-service-architecture.md` | access control, monitoring, retention |
| Deployment and rollback | `engineering/deployment-process.md` | failed deployment, migrations, monitoring |
| PostgreSQL change and recovery | `engineering/database-migrations.md` | backup, storage, Postgres runbook |
| Severity and response roles | `security/incident-response-policy.md` | monitoring and all runbooks |
| Retention and privacy | `policies/data-retention-policy.md` | GDPR, storage, backup, RAG architecture |
| Identity and production access | `security/access-control-policy.md` | onboarding, encryption, incident response |
| Employee procedures | HR pages | remote work, access control, incident response |

Deliberate retrieval difficulty comes from repeated but compatible facts. Retention periods appear in storage, GDPR, RAG, and the canonical retention policy. RPO and RTO appear in architecture, storage, backup, and recovery pages. Severity response times appear in the incident policy, monitoring guide, and runbooks. The deployment process and failed-deployment runbook share rollback thresholds but differ in purpose and procedural depth.

## Dataset usage

Markdown pages are under `knowledge_base/`. The evaluation set is JSONL under `evaluation/`. `relevant_documents` paths are relative to this repository root. Unanswerable examples intentionally ask for facts not present in the corpus and should trigger evidence-aware refusal.
