from __future__ import annotations

import argparse
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT))

from cups_eval.cli import main as cli_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny fake-LLM evaluation smoke test.")
    parser.add_argument("--config", type=Path, default=RELEASE_ROOT / "configs" / "mvp.medmcqa.yaml")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=RELEASE_ROOT / "runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return cli_main(
        [
            "run",
            "--config",
            str(args.config),
            "--fake-llm",
            "--sample-size",
            str(args.sample_size),
            "--output-root",
            str(args.output_root),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
