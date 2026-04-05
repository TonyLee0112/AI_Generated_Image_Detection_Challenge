[Current State]
- GitHub repo `TonyLee0112/AI_Generated_Image_Detection_Challenge` now contains a separate `RINE/` module.
- Baseline code remains untouched.
- `RINE/` currently contains:
  - `src/`
  - `train_rine.py`
  - `test.py`
  - `requirements_min.txt`
  - `README.md`
- `.gitignore` excludes datasets, outputs, checkpoints, caches, and temp files relevant to RINE usage.

[Previously Done]
- Cloned the existing GitHub repository with LFS smudge disabled because one tracked checkpoint object returned 404.
- Added `RINE/` and copied only required RINE code files.
- Updated `RINE/train_rine.py` so the default output path points to `RINE/outputs/...`.
- Rewrote `RINE/README.md` for server-side usage with local dataset paths.
- Updated `.gitignore` to exclude outputs/checkpoints/cache/data artifacts.
- Committed and pushed the RINE addition to `main`.
- Latest pushed commit for the RINE addition:
  - `1a69741` (`Add RINE module under RINE directory`)

[To Do Next]
- On the server, run `git pull origin main`.
- Install dependencies from `RINE/requirements_min.txt` if needed.
- Run `RINE/train_rine.py` using server-local dataset paths via `--data-root`.
- Before each final answer, refresh this file so it reflects the latest repository state.
