# Release Checklist

Use this checklist before publishing the repository for any journal or venue.

## Environment

- [ ] `uv sync --dev --extra reports`
- [ ] `uv run pytest -q`
- [ ] `uv run python scripts/evaluate/run_fake_smoke.py --sample-size 5`

## Public Artifacts

- [ ] `uv run python scripts/analysis/audit_numeric_consistency.py`
- [ ] `uv run python scripts/figures/generate_academic_figures.py`
- [ ] Confirm `results/` contains only compact public artifacts
- [ ] Confirm `data/` contains metadata only

## Reproducibility Boundary

- [ ] Confirm `configs/` contains stable templates for runs and batches
- [ ] Confirm `runs/` remains ignored by git
- [ ] Confirm full per-item outputs live in a companion data repository

## Publication Metadata

- [ ] Select a software license
- [ ] Add the archival DOI or URL for the companion data repository
- [ ] Update the manuscript code/data availability statement

