# Decisions

What I cut, and why. Five days, one person.
A defensible cut is engineering; an undefended one is scope creep.

| Date | Decision | Reason | Cost accepted |
|---|---|---|---|
| Aug 30 | SQLite over PostgreSQL | Single-process batch analytical workload. Postgres would be resume decoration. | No concurrent writes |
| Aug 30 | Plain HTML over React | ~2h vs ~1 day for the same screen. Metrics don't depend on the UI. | Less polished demo |
| Aug 30 | No agent framework | Workflow is retrieve → classify → propose → validate → gate. A graph library adds indirection, not capability. | — |
