# Local SCC Result Archives

This directory is for local copies of SCC `runs/` and `logs/` artifacts.

These files are intentionally **not committed to Git** by default because the
full generation JSON outputs are large and change frequently.

Use the backup helper from the repo root:

```bash
bash scripts/pull_scc_results.sh
```

That script:

- finds the active SCC project directory under `cs505am/students/zfmsai`
- copies both `runs/` and `logs/` locally
- writes a timestamped snapshot under `local_scc_results/snapshots/<timestamp>/`
- refreshes `local_scc_results/latest/` as the newest local mirror

If you want a shareable summary in Git, commit the derived markdown tables or
analysis artifacts instead of the raw run dumps.
