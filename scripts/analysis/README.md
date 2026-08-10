# Analysis Scripts

The main analysis entry points are available through the CLI:

```bash
uv run cups-eval offline-router --batch-dir runs/<batch-dir>
uv run cups-eval cups --batch-dir runs/<batch-dir>
uv run cups-eval mechanistic --batch-dir runs/<batch-dir>
uv run cups-eval cot-router --batch-dir runs/<batch-dir>
```

`audit_numeric_consistency.py` checks the bundled release tables, CUPS points, rescue/harm source tables, figure outputs, and path/token hygiene:

```bash
uv run python scripts/analysis/audit_numeric_consistency.py
```
