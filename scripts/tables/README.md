# Table Artifacts

This repository ships compact paper tables under `results/`.

## Bundled Tables

- `results/tables/`
- `results/source_tables/`
- `results/cups/`
- `results/mechanism/`
- `results/robustness/`
- `results/ci/`

## Regeneration Boundary

The bundled tables are the public release artifacts.

For fresh runs, use the existing `cups-eval` analysis commands:

```bash
uv run cups-eval offline-router --batch-dir runs/<batch-dir>
uv run cups-eval cups --batch-dir runs/<batch-dir>
uv run cups-eval mechanistic --batch-dir runs/<batch-dir>
uv run cups-eval cot-router --batch-dir runs/<batch-dir>
```

For figure rebuilds, use:

```bash
uv run python scripts/figures/generate_academic_figures.py
```

This directory is intentionally documentation-only.
