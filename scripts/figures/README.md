# Figure Scripts

Regenerate paper figures from the compact public tables:

```bash
uv run python scripts/figures/generate_academic_figures.py
```

Inputs are read from:

- `results/tables/`
- `results/source_tables/`
- `results/cups/`

Outputs are written to `results/figures/`.
