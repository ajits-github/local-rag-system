# Incident Response Process

**Owner:** Site Reliability Engineering
**Last Reviewed:** 2026-05-02
**Applies To:** All engineering staff participating in on-call rotations

## Severity Levels

Incidents are classified into four severity levels. **SEV1** indicates a
full outage or data loss affecting all customers. **SEV2** indicates a
major degradation affecting a significant subset of customers. **SEV3**
covers minor degradations with a workaround available. **SEV4** is used
for cosmetic or low-impact issues with no customer-facing effect.

## Escalation

For a **SEV1**, the on-call engineer must page a secondary responder and
declare an **Incident Commander within 15 minutes** of detection. SEV2
incidents allow up to 30 minutes before escalation is required. SEV3 and
SEV4 incidents are tracked but do not require immediate escalation.

## Communication

All active incidents are coordinated in the **`#incident-response`**
Slack channel, where the Incident Commander posts status updates at least
every 30 minutes for SEV1/SEV2. Customer-facing status is published via
**Statuspage.io**, updated by the Incident Commander or their delegate.

## Deployment Freeze

Once a SEV1 or SEV2 incident is declared, all non-emergency production
deployments are frozen until the incident is resolved, per the
deployment process's emergency hotfix exemption.

## Postmortems

A written postmortem is required for every SEV1 and SEV2 incident,
**due within 5 business days** of resolution. The postmortem must include
a timeline, root cause, customer impact, and at least one concrete action
item with an owner and due date. Postmortems are reviewed in the weekly
SRE sync.
