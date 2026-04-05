# Project goal
Keep the existing challenge repository stable while adding RINE as a separate module under `RINE/`.

## Rules
- Preserve the existing baseline structure.
- Do not mix RINE files into the repo root; keep them under `RINE/`.
- Do not commit datasets, checkpoints, outputs, caches, or temporary files.
- Keep server usage in mind: the code should run after `git pull` on the workstation/server with server-local dataset paths.
- Before every final answer, update `PROJECT_LOG.md`.
- `PROJECT_LOG.md` must always use exactly these sections:
  - `[Current State]`
  - `[Previously Done]`
  - `[To Do Next]`
- Refresh those sections with the latest project summary before replying to the user.
