# Project Agent Override

## Git Autonomy

- When the user explicitly delegates commit and push decisions to the agent,
  treat that delegation as active for the rest of the session unless the user
  revokes it.
- After completing a coherent change set, commit and push without asking for
  confirmation.

## Documentation Cleanup

- Prefer reducing stale run notes over creating archive directories.
- Keep `tracks/leverage/runs/` for concrete run, batch, audit, probe, and
  example records.
- Keep cross-run strategy, current decisions, and policy changes under
  `tracks/leverage/docs/`.
