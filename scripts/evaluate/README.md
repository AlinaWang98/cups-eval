# Evaluation Scripts

Use the package CLI for full evaluations:

```bash
uv run cups-eval run --config configs/mvp.medmcqa.yaml --sample-size 100
uv run cups-eval batch --matrix-config configs/batch.phase2.yaml --sample-size 100
```

For a no-provider smoke test:

```bash
uv run python scripts/evaluate/run_fake_smoke.py --sample-size 5
```

Generated runs are written to `runs/` and ignored by git.
