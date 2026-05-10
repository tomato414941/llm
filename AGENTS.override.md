# Project Agent Override

## Git Autonomy

- When the user explicitly delegates commit and push decisions to the agent,
  treat that delegation as active for the rest of the session unless the user
  revokes it.
- After completing a coherent change set, commit and push without asking for
  confirmation.
- Keep commits small and use the project format `type: description`.
- Before committing, check `git status --short --branch` and ensure the commit
  only includes changes relevant to the current task.
- After pushing, report the commit hash and confirm the working tree is clean.

## Documentation Cleanup

- Prefer reducing stale run notes over creating archive directories.
- Delete obsolete run notes only when their useful conclusions are already
  captured in current docs, current run notes, or git history.
- Keep `tracks/leverage/runs/` for concrete run, batch, audit, probe, and
  example records.
- Keep cross-run strategy, current decisions, and policy changes under
  `tracks/leverage/docs/`.
