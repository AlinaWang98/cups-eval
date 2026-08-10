# When Does Collaboration Pay? Code Release

This repository contains the public code and compact result artifacts for the paper experiments on cost-fair evaluation of collaborative LLM reasoning.

## What Is Included

- `cups_eval/`: evaluation, prompting, routing, CUPS, rescue/harm, and reporting code.
- `configs/`: model and dataset configuration templates.
- `scripts/evaluate/`: thin entry points for smoke and reproducibility runs.
- `scripts/analysis/`: public consistency and audit scripts over bundled result tables.
- `scripts/figures/`: figure generation from release-bundled tables.
- `scripts/tables/`: notes on table regeneration and release-bundled table boundaries.
- `results/`: compact paper tables, source tables, CUPS outputs, mechanism diagnostics, robustness summaries, and final figures.
- `data/`: metadata-only dataset inventory. Dataset payloads are not redistributed here.
- `tests/`: unit and smoke tests.

The repository intentionally excludes API logs, full per-item predictions, raw model responses, local run directories, and working drafts.

## Setup

```bash
uv sync --dev --extra reports
```

Alternatively, install with pip:

```bash
python3 -m pip install -e ".[reports]"
python3 -m pip install pytest
```

Copy `.env.example` to `.env` and fill only the provider keys you plan to use. The fake-LLM smoke test does not require any API key.

## Quick Check

```bash
uv run pytest -q
uv run python scripts/evaluate/run_fake_smoke.py --sample-size 5
uv run python scripts/analysis/audit_numeric_consistency.py
uv run python scripts/figures/generate_academic_figures.py
```

## Real Evaluation

```bash
uv run cups-eval run \
  --config configs/mvp.medmcqa.yaml \
  --sample-size 100 \
  --output-root runs
```

Batch configurations are under `configs/batch.*.yaml`. Generated runs are written to `runs/`, which is ignored by git.

Table regeneration guidance lives in `scripts/tables/README.md`.

## Availability Boundary

Use GitHub for this code release. Put full per-item prediction files, raw model outputs, and any large derived artifacts in a data repository such as Zenodo, Mendeley Data, Figshare, or OSF, then cite that DOI in the manuscript's Data Availability statement.

License selection is deferred to the authors before public upload.
