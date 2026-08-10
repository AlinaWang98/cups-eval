from __future__ import annotations

import argparse
from pathlib import Path

from cups_eval.batch import run_batch
from cups_eval.cot_router_analysis import write_cot_router_report
from cups_eval.cups_policy import (
    DEFAULT_LAMBDAS,
    cross_validate_cups,
    write_cups_report,
)
from cups_eval.mechanistic import read_csv_rows, write_mechanistic_report
from cups_eval.offline_router import (
    cross_validate_router,
    load_feature_rows,
    write_offline_router_report,
)
from cups_eval.runner import analyze_run, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cups-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the MVP experiment.")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--sample-size", type=int, default=None)
    run_parser.add_argument("--fake-llm", action="store_true")
    run_parser.add_argument("--output-root", type=Path, default=Path("runs"))
    run_parser.add_argument("--max-usd", type=float, default=None)
    run_parser.add_argument("--resume-run-dir", type=Path, default=None)
    run_parser.add_argument("--workers", type=int, default=1)

    analyze_parser = subparsers.add_parser("analyze", help="Recompute run summaries.")
    analyze_parser.add_argument("--run-dir", type=Path, required=True)

    batch_parser = subparsers.add_parser("batch", help="Run a dataset/model matrix.")
    batch_parser.add_argument("--matrix-config", type=Path, required=True)
    batch_parser.add_argument("--fake-llm", action="store_true")
    batch_parser.add_argument("--dry-run", action="store_true")
    batch_parser.add_argument("--sample-size", type=int, default=None)
    batch_parser.add_argument("--max-usd", type=float, default=None)
    batch_parser.add_argument("--workers", type=int, default=None)
    batch_parser.add_argument("--max-parallel-runs", type=int, default=None)
    batch_parser.add_argument("--resume-batch-dir", type=Path, default=None)

    offline_router_parser = subparsers.add_parser(
        "offline-router",
        help="Train/evaluate deployable router rules from saved router_features.csv files.",
    )
    offline_router_parser.add_argument("--batch-dir", type=Path, required=True)
    offline_router_parser.add_argument("--output-dir", type=Path, default=None)
    offline_router_parser.add_argument("--model-prefix", type=str, default=None)
    offline_router_parser.add_argument("--folds", type=int, default=5)
    offline_router_parser.add_argument("--budget-multiplier", type=float, default=1.5)
    offline_router_parser.add_argument("--max-rules", type=int, default=4)
    offline_router_parser.add_argument("--min-support", type=int, default=25)

    mechanistic_parser = subparsers.add_parser(
        "mechanistic",
        help="Build paper-ready mechanistic analysis tables from saved features and router outputs.",
    )
    mechanistic_parser.add_argument("--batch-dir", type=Path, required=True)
    mechanistic_parser.add_argument("--output-dir", type=Path, default=None)
    mechanistic_parser.add_argument("--model-prefix", type=str, default=None)
    mechanistic_parser.add_argument("--router-analysis-dir", type=Path, default=None)

    cot_router_parser = subparsers.add_parser(
        "cot-router",
        help="Build zero-cost CoT-first routing analysis from saved feature rows.",
    )
    cot_router_parser.add_argument("--batch-dir", type=Path, required=True)
    cot_router_parser.add_argument("--output-dir", type=Path, default=None)
    cot_router_parser.add_argument("--model-prefix", type=str, default=None)

    cups_parser = subparsers.add_parser(
        "cups",
        help="Train/evaluate cost-aware utility policy selection from saved features.",
    )
    cups_parser.add_argument("--batch-dir", type=Path, required=True)
    cups_parser.add_argument("--output-dir", type=Path, default=None)
    cups_parser.add_argument("--model-prefix", type=str, default=None)
    cups_parser.add_argument("--folds", type=int, default=5)
    cups_parser.add_argument(
        "--lambdas",
        type=str,
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
        help="Comma-separated utility cost penalties to sweep.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        run_dir = run_experiment(
            config_path=args.config,
            sample_size=args.sample_size,
            fake_llm=args.fake_llm,
            output_root=args.output_root,
            max_usd=args.max_usd,
            resume_run_dir=args.resume_run_dir,
            workers=args.workers,
        )
        print(run_dir)
        return 0
    if args.command == "analyze":
        analyze_run(args.run_dir)
        print(args.run_dir)
        return 0
    if args.command == "batch":
        result = run_batch(
            matrix_config_path=args.matrix_config,
            fake_llm=args.fake_llm,
            dry_run=args.dry_run,
            sample_size=args.sample_size,
            max_usd=args.max_usd,
            workers=args.workers,
            max_parallel_runs=args.max_parallel_runs,
            resume_batch_dir=args.resume_batch_dir,
        )
        print(result.batch_dir)
        for command in result.commands:
            print(" ".join(command))
        return 0
    if args.command == "offline-router":
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else args.batch_dir / "analysis_offline_router"
        )
        rows = load_feature_rows(args.batch_dir, model_prefix=args.model_prefix)
        result = cross_validate_router(
            rows,
            folds=args.folds,
            budget_multiplier=args.budget_multiplier,
            max_rules=args.max_rules,
            min_support=args.min_support,
        )
        write_offline_router_report(
            rows,
            result,
            output_dir,
            budget_multiplier=args.budget_multiplier,
        )
        print(output_dir)
        return 0
    if args.command == "mechanistic":
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else args.batch_dir / "analysis_mechanistic"
        )
        router_analysis_dir = args.router_analysis_dir or (
            args.batch_dir / "analysis_offline_router"
        )
        rows = load_feature_rows(args.batch_dir, model_prefix=args.model_prefix)
        predictions = read_csv_rows(router_analysis_dir / "offline_router_predictions.csv")
        routing_curve = read_csv_rows(router_analysis_dir / "routing_curve.csv")
        write_mechanistic_report(rows, predictions, routing_curve, output_dir)
        print(output_dir)
        return 0
    if args.command == "cot-router":
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else args.batch_dir / "analysis_cot_router"
        )
        rows = load_feature_rows(args.batch_dir, model_prefix=args.model_prefix)
        write_cot_router_report(rows, output_dir)
        print(output_dir)
        return 0
    if args.command == "cups":
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else args.batch_dir / "analysis_cups"
        )
        rows = load_feature_rows(args.batch_dir, model_prefix=args.model_prefix)
        result = cross_validate_cups(
            rows,
            folds=args.folds,
            lambdas=_parse_lambda_values(args.lambdas),
        )
        write_cups_report(rows, result, output_dir)
        print(output_dir)
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


def _parse_lambda_values(value: str) -> tuple[float, ...]:
    values = tuple(
        float(part.strip())
        for part in value.split(",")
        if part.strip()
    )
    return values or DEFAULT_LAMBDAS
