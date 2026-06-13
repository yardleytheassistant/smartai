---
name: ci-triage
description: Classify CI failures, draft fixes for easy ones, escalate the rest.
trigger: ci failure, workflow_run.failure, morning triage, build broke
---
# CI triage skill

Procedural memory for triaging continuous-integration failures. Compounds: every
confirmed failure mode below was added by the loop after a real incident.

## Classification rules
- env: missing secret or wrong env var. → escalate to a human, never auto-fix.
- flake: passes on retry with no code change. → retry once, then file an issue.
- bug: deterministic failure tied to a recent commit. → draft a fix.
- dependency: failure tied to a version bump. → draft a rollback.
- infra: timeout, OOM, runner issue. → escalate.

## Known failure modes
- (none yet — the eval loop and goal loop append confirmed modes here)

## Anti-patterns (do NOT do)
- Never disable a failing test to make CI green. File it instead.
- Never modify CI workflow files without human approval.

## State
After each run, update STATE.md with classifications, fixes drafted, escalations.
