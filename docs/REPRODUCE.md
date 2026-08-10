# Reproducing The Public Artifacts

## 1. Install

```bash
uv sync --dev --extra reports
```

If `uv` is unavailable:

```bash
python3 -m pip install -e ".[reports]"
python3 -m pip install pytest
```

## 2. Run Tests

```bash
uv run pytest -q
```

## 3. Run A Fake Smoke Evaluation

This creates a tiny deterministic run without provider calls:

```bash
uv run python scripts/evaluate/run_fake_smoke.py --sample-size 5
```

Equivalent CLI form:

```bash
uv run cups-eval run \
  --config configs/mvp.medmcqa.yaml \
  --fake-llm \
  --sample-size 5
```

## 4. Run A Real Evaluation

Fill `.env` from `.env.example`, then run:

```bash
uv run cups-eval run \
  --config configs/mvp.medmcqa.yaml \
  --sample-size 100 \
  --output-root runs
```

To recompute summaries for a saved run:

```bash
uv run cups-eval analyze --run-dir runs/<timestamped-run-dir>
```

## 5. Batch Runs

```bash
uv run cups-eval batch \
  --matrix-config configs/batch.phase2.yaml \
  --sample-size 100 \
  --max-parallel-runs 1
```

Add `--fake-llm` for a no-provider dry smoke run, or `--dry-run` to print commands only.

## 6. Router, CUPS, And Mechanism Analysis

These commands operate on saved batch directories containing `router_features.csv`.

```bash
uv run cups-eval offline-router --batch-dir runs/<batch-dir>
uv run cups-eval cups --batch-dir runs/<batch-dir>
uv run cups-eval mechanistic --batch-dir runs/<batch-dir>
uv run cups-eval cot-router --batch-dir runs/<batch-dir>
```

Bundled paper-level compact artifacts are in `results/`. Full per-item outputs should be retrieved from the companion data repository once available.

## 7. Rebuild Figures And Audit Tables

```bash
uv run python scripts/figures/generate_academic_figures.py
uv run python scripts/analysis/audit_numeric_consistency.py
```

Table regeneration and release boundaries are documented in `scripts/tables/README.md`.
