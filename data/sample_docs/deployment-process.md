# Deployment Process

**Owner:** Platform Engineering
**Last Reviewed:** 2026-06-15
**Applies To:** All production services in the `core-platform` and `billing` repositories

## Overview

This document describes how changes move from a merged pull request to
running in production. It applies to every engineer deploying to the
production environment, including on-call responders performing hotfixes.

## Deployment Windows

Standard deployments are only permitted on **Tuesdays and Thursdays,
between 10:00 and 14:00 UTC**. Deployments outside this window require
sign-off from the on-call engineering manager. Emergency hotfixes for
active incidents are exempt from the window restriction but must still
follow the approval process below.

## Approval Requirements

Every production deployment requires approval from **two engineers**: the
change author and one independent reviewer who did not write the code.
Reviewer approval must be recorded as a comment on the deployment ticket
before the pipeline is allowed to proceed past the staging gate.

## Pipeline and Rollout

Deployments run through **GitHub Actions**, which builds the artifact,
runs the test suite, and deploys first to the staging environment. After
staging verification passes, the pipeline performs a **canary rollout to
10% of production traffic for 30 minutes** before promoting to 100%.

## Rollback Procedure

If error rates exceed 2% during the canary window, the pipeline
automatically rolls back to the previous stable release. Manual rollback
can be triggered at any time by running `deploy rollback <service>` from
the platform CLI, which restores the last known-good artifact within
approximately five minutes.
