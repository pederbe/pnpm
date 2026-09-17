---
"pacquet": minor
---

Python environments now live in the store. Each project keeps only its `.venv` link, which points at the project's current environment generation under `python-envs` in the store. A repository with many Python projects no longer holds a `.pnpm/python-envs` directory in each of them. The next install relinks a `.venv` that an earlier release published, and the old `.pnpm/python-envs` directory can then be deleted [#15014](https://github.com/pnpm/pnpm/issues/15014).
