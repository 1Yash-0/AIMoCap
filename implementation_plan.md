# Architecture Refactor & Metric Reconciliation

## Goal
Resolve contradictory metric diagnostics (10.6px vs 48.9px 2D error; 232.4mm vs 352.9mm MPJPE) by tracking variables strictly back to their data lineage before we make a final production architecture decision on the 2D rejection gates. 

## Status
- **Phase A (Metrics):** PENDING. Phase A remains open because historical counts, the 48.9px result, the 232.4mm baseline, and several requested validation checks have not been reproduced strictly.
- **Phase B (Manual Audit):** BLOCKED.
- **Phase C (Architecture Candidates):** BLOCKED.

## User Review Required
No architectural changes are proposed. A rigorous, JSON-driven audit is currently being executed to enforce exact metric reproduction.

## Open Questions
None currently. 
