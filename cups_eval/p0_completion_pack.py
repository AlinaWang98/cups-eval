from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cups_eval.required_experiment_pack import paired_comparison_stats


STRATEGIES = ("direct", "cot", "self_consistency_3", "planner_verifier", "debate_2_judge")
COMPARISONS = ("cot", "self_consistency_3", "planner_verifier", "debate_2_judge")

DEFAULT_RUN_SPECS = [
    {
        "dataset": "math500",
        "model": "deepseek_v4_flash",
        "label": "DeepSeek v4 flash",
        "sample_label": "full500_anchor",
        "planned_n": 500,
        "run_dir": Path(
            "runs/phase3_math500_p0_full_anchor/20260803T071801Z/runs/math500/deepseek_v4_flash/20260803T071802Z"
        ),
    },
    {
        "dataset": "math500",
        "model": "glm_5_2",
        "label": "GLM-5.2",
        "sample_label": "clean100",
        "planned_n": 100,
        "run_dir": Path(
            "runs/phase3_math500_core_models/20260729T122614Z/runs/math500/glm_5_2/20260729T122615Z"
        ),
    },
    {
        "dataset": "math500",
        "model": "minimax_m3",
        "label": "MiniMax-M3",
        "sample_label": "clean100",
        "planned_n": 100,
        "run_dir": Path(
            "runs/phase3_minimax_math500/20260729T085816Z/runs/math500/minimax_m3/20260729T085817Z"
        ),
    },
    {
        "dataset": "math500",
        "model": "qwen_3_7_max_preview",
        "label": "Qwen3.7-Max-Preview",
        "sample_label": "clean100",
        "planned_n": 100,
        "run_dir": Path(
            "runs/phase3_qwen_math500_clean/20260730Tclean001/runs/math500/qwen_3_7_max_preview/20260730Tclean001"
        ),
    },
    {
        "dataset": "math500_frontier50",
        "model": "gpt_5_5",
        "label": "GPT-5.5",
        "sample_label": "frontier50_full5",
        "planned_n": 50,
        "run_dir": Path(
            "runs/phase3_math500_frontier50/20260805T034724Z/runs/math500_frontier50/gpt_5_5/20260805T034725Z"
        ),
    },
    {
        "dataset": "math500_frontier50",
        "model": "claude_sonnet_5",
        "label": "claude-sonnet-5",
        "sample_label": "frontier50_full5",
        "planned_n": 50,
        "run_dir": Path(
            "runs/phase3_math500_frontier50/20260805T034724Z/runs/math500_frontier50/claude_sonnet_5/20260805T034725Z"
        ),
    },
]


def build_p0_completion_pack(
    *,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    run_specs: list[dict[str, Any]] | None = None,
    canonical_dir: Path | None = None,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[1]
    destination = output_dir or root / "reports" / "cups_packet" / "p0_completion"
    canonical_path = canonical_dir or root / "reports" / "cups_packet" / "experiment_data_by_strategy"
    specs = [_normalize_spec(root, spec) for spec in (run_specs or DEFAULT_RUN_SPECS)]

    destination.mkdir(parents=True, exist_ok=True)
    run_inclusion = build_run_inclusion_rows(specs)
    strategy_metrics = build_strategy_metric_rows(specs)
    paired_rows = build_paired_rows(specs, bootstrap_iterations=bootstrap_iterations)
    audit_rows = build_math_audit_rows(specs)
    canonical_rows = build_canonical_coverage_rows(canonical_path)

    _write_csv(destination / "run_inclusion_manifest.csv", run_inclusion)
    _write_csv(destination / "strategy_metrics.csv", strategy_metrics)
    _write_csv(destination / "headline_paired_tests.csv", paired_rows)
    _write_csv(destination / "math500_final_label_audit.csv", audit_rows)
    _write_csv(destination / "canonical_math500_coverage.csv", canonical_rows)

    headline = {
        "complete_required_math_runs": sum(1 for row in run_inclusion if row["status"] == "included_complete"),
        "math_audited_runs": sum(1 for row in audit_rows if _to_int(row["audit_rows"]) > 0),
        "canonical_math500_strategy_rows": sum(_to_int(row["rows"]) for row in canonical_rows),
        "canonical_math500_full500_deepseek_rows": sum(
            _to_int(row["rows"])
            for row in canonical_rows
            if row["dataset"] == "math500"
            and row["model"] == "deepseek_v4_flash"
            and row["sample_label"] == "full500_anchor"
        ),
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(destination),
        "inputs": {
            "canonical_experiment_data": str(canonical_path),
            "run_dirs": [
                {
                    "dataset": spec["dataset"],
                    "model": spec["model"],
                    "sample_label": spec["sample_label"],
                    "run_dir": str(spec["run_dir"]),
                }
                for spec in specs
            ],
        },
        "outputs": {
            "run_inclusion": "run_inclusion_manifest.csv",
            "strategy_metrics": "strategy_metrics.csv",
            "headline_paired_tests": "headline_paired_tests.csv",
            "math500_final_label_audit": "math500_final_label_audit.csv",
            "canonical_math500_coverage": "canonical_math500_coverage.csv",
            "summary": "p0_completion.md",
        },
        "row_counts": {
            "run_inclusion": len(run_inclusion),
            "strategy_metrics": len(strategy_metrics),
            "headline_paired_tests": len(paired_rows),
            "math500_final_label_audit": len(audit_rows),
            "canonical_math500_coverage": len(canonical_rows),
        },
        "headline": headline,
        "notes": [
            "Generated offline from existing run artifacts; no provider calls are made.",
            "MATH correctness uses the evaluated item_metrics.csv and math_equivalence_audit.csv artifacts where present.",
            "MiniMax is included from its existing clean100 run; no MiniMax rerun is required when raw/evaluated artifacts are present.",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "p0_completion.md").write_text(
        render_p0_completion_markdown(manifest, run_inclusion, audit_rows, paired_rows, canonical_rows),
        encoding="utf-8",
    )
    return manifest


def build_run_inclusion_rows(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        raw_records = _read_jsonl(spec["run_dir"] / "raw_predictions.jsonl")
        item_rows = _read_csv(spec["run_dir"] / "item_metrics.csv")
        strategy_counts = Counter(str(row.get("strategy") or "") for row in raw_records)
        item_strategy_counts = Counter(str(row.get("strategy") or "") for row in item_rows)
        present_questions = {
            str(row.get("question_id") or "")
            for row in item_rows
            if row.get("question_id")
        }
        status = "included_complete" if item_rows else "missing_item_metrics"
        if item_rows and len(present_questions) < _to_int(spec["planned_n"]):
            status = "included_partial"
        rows.append(
            {
                "dataset": spec["dataset"],
                "model": spec["model"],
                "model_label": spec["label"],
                "sample_label": spec["sample_label"],
                "planned_n": spec["planned_n"],
                "unique_questions": len(present_questions),
                "raw_rows": len(raw_records),
                "item_metric_rows": len(item_rows),
                "raw_strategy_counts": json.dumps(dict(sorted(strategy_counts.items())), sort_keys=True),
                "item_strategy_counts": json.dumps(dict(sorted(item_strategy_counts.items())), sort_keys=True),
                "math_equivalence_audit_present": (spec["run_dir"] / "math_equivalence_audit.csv").exists(),
                "status": status,
                "run_dir": str(spec["run_dir"]),
            }
        )
    return rows


def build_strategy_metric_rows(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for spec in specs:
        rows = _read_csv(spec["run_dir"] / "item_metrics.csv")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("strategy") or "")].append(row)
        direct_cost = sum(_to_float(row.get("estimated_cost_usd")) for row in grouped.get("direct", []))
        direct_tokens = sum(_to_int(row.get("total_tokens")) for row in grouped.get("direct", []))
        for strategy in STRATEGIES:
            items = grouped.get(strategy, [])
            if not items:
                continue
            n = len(items)
            correct = sum(1 for row in items if _to_bool(row.get("correct")))
            parse_ok = sum(1 for row in items if _to_bool(row.get("parse_ok")))
            cost = sum(_to_float(row.get("estimated_cost_usd")) for row in items)
            tokens = sum(_to_int(row.get("total_tokens")) for row in items)
            output.append(
                {
                    "dataset": spec["dataset"],
                    "model": spec["model"],
                    "model_label": spec["label"],
                    "sample_label": spec["sample_label"],
                    "strategy": strategy,
                    "n": n,
                    "correct": correct,
                    "accuracy": correct / n if n else 0.0,
                    "parse_ok": parse_ok,
                    "parse_rate": parse_ok / n if n else 0.0,
                    "estimated_cost_usd": cost,
                    "total_tokens": tokens,
                    "cost_x_direct": cost / direct_cost if direct_cost else "",
                    "tokens_x_direct": tokens / direct_tokens if direct_tokens else "",
                    "run_dir": str(spec["run_dir"]),
                }
            )
    return output


def build_paired_rows(specs: list[dict[str, Any]], *, bootstrap_iterations: int) -> list[dict[str, Any]]:
    output = []
    for spec in specs:
        by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in _read_csv(spec["run_dir"] / "item_metrics.csv"):
            by_question[str(row.get("question_id") or "")][str(row.get("strategy") or "")] = row
        for comparison in COMPARISONS:
            selected = []
            baseline = []
            for strategies in by_question.values():
                if "direct" not in strategies or comparison not in strategies:
                    continue
                baseline.append(_to_bool(strategies["direct"].get("correct")))
                selected.append(_to_bool(strategies[comparison].get("correct")))
            if not selected:
                continue
            output.append(
                {
                    "dataset": spec["dataset"],
                    "model": spec["model"],
                    "model_label": spec["label"],
                    "sample_label": spec["sample_label"],
                    "comparison": comparison,
                    **paired_comparison_stats(
                        selected,
                        baseline,
                        bootstrap_iterations=bootstrap_iterations,
                    ),
                    "run_dir": str(spec["run_dir"]),
                }
            )
    return output


def build_math_audit_rows(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for spec in specs:
        audit_path = spec["run_dir"] / "math_equivalence_audit.csv"
        if not audit_path.exists():
            output.append(
                {
                    "dataset": spec["dataset"],
                    "model": spec["model"],
                    "model_label": spec["label"],
                    "sample_label": spec["sample_label"],
                    "audit_rows": 0,
                    "equivalent": 0,
                    "not_equivalent": 0,
                    "unresolved": 0,
                    "needs_blind_review": 0,
                    "audit_status": "missing",
                    "run_dir": str(spec["run_dir"]),
                }
            )
            continue
        rows = _read_csv(audit_path)
        verdicts = Counter(str(row.get("verdict") or "") for row in rows)
        needs_blind_review = sum(1 for row in rows if _to_bool(row.get("needs_blind_review")))
        output.append(
            {
                "dataset": spec["dataset"],
                "model": spec["model"],
                "model_label": spec["label"],
                "sample_label": spec["sample_label"],
                "audit_rows": len(rows),
                "equivalent": verdicts.get("equivalent", 0),
                "not_equivalent": verdicts.get("not_equivalent", 0),
                "unresolved": verdicts.get("unresolved", 0),
                "needs_blind_review": needs_blind_review,
                "audit_status": "present",
                "run_dir": str(spec["run_dir"]),
            }
        )
    return output


def build_canonical_coverage_rows(canonical_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(canonical_dir / "manifest.csv"):
        if str(row.get("dataset") or "") != "math500":
            continue
        raw_path = str(row.get("source_raw_predictions") or "")
        rows.append(
            {
                "source_batch": row.get("source_batch", ""),
                "dataset": row.get("dataset", ""),
                "model": row.get("model", ""),
                "run_id": row.get("run_id", ""),
                "sample_label": _math500_sample_label(raw_path),
                "strategy": row.get("strategy", ""),
                "rows": row.get("rows", ""),
                "unique_questions": row.get("unique_questions", ""),
                "source_raw_predictions": raw_path,
                "source_evaluated_artifact": row.get("source_evaluated_artifact", ""),
            }
        )
    return rows


def render_p0_completion_markdown(
    manifest: dict[str, Any],
    run_inclusion: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# P0 Completion Pack",
        "",
        "Generated offline from existing run artifacts. No provider calls were made.",
        "",
        "## Headline",
        "",
        f"- Complete/usable math runs: {manifest['headline']['complete_required_math_runs']}.",
        f"- Runs with math-equivalence audit artifacts: {manifest['headline']['math_audited_runs']}.",
        f"- Canonical MATH-500 strategy rows: {manifest['headline']['canonical_math500_strategy_rows']}.",
        f"- Canonical DeepSeek full500 strategy rows: {manifest['headline']['canonical_math500_full500_deepseek_rows']}.",
        "",
        "## Run Inclusion",
        "",
        "| model | sample | unique q | raw rows | item rows | audit | status |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in run_inclusion:
        lines.append(
            f"| {row['model_label']} | {row['sample_label']} | {row['unique_questions']} | "
            f"{row['raw_rows']} | {row['item_metric_rows']} | {row['math_equivalence_audit_present']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Math Final-Label Audit",
            "",
            "| model | sample | audit rows | equivalent | not equivalent | unresolved | blind queue |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit_rows:
        lines.append(
            f"| {row['model_label']} | {row['sample_label']} | {row['audit_rows']} | {row['equivalent']} | "
            f"{row['not_equivalent']} | {row['unresolved']} | {row['needs_blind_review']} |"
        )
    lines.extend(
        [
            "",
            "## Headline Paired Tests",
            "",
            "| model | sample | comparison | n | delta | 95% CI | p | rescues | harms |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| {model} | {sample} | `{comparison}` | {n} | {delta:+.3f} | [{low:+.3f}, {high:+.3f}] | {p:.3f} | {rescue} | {harm} |".format(
                model=row["model_label"],
                sample=row["sample_label"],
                comparison=row["comparison"],
                n=row["n"],
                delta=_to_float(row["delta_accuracy"]),
                low=_to_float(row["bootstrap_delta_ci_low"]),
                high=_to_float(row["bootstrap_delta_ci_high"]),
                p=_to_float(row["mcnemar_exact_p_two_sided"]),
                rescue=row["paired_rescues"],
                harm=row["paired_harms"],
            )
        )
    lines.extend(
        [
            "",
            "## Canonical MATH-500 Coverage",
            "",
            "| model | sample | strategy | rows | source |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in canonical_rows:
        lines.append(
            f"| {row['model']} | {row['sample_label']} | `{row['strategy']}` | {row['rows']} | `{row['source_batch']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _normalize_spec(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    output = dict(spec)
    run_dir = Path(output["run_dir"])
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    output["run_dir"] = run_dir
    return output


def _math500_sample_label(raw_path: str) -> str:
    if "phase3_math500_p0_full_anchor/20260803T071801Z" in raw_path:
        return "full500_anchor"
    if "phase3_math500_core_models" in raw_path:
        return "clean100_or_ceiling_control"
    if "phase3_minimax_math500" in raw_path:
        return "clean100"
    if "phase3_qwen_math500_clean" in raw_path:
        return "clean100"
    return "canonical"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build reports/cups_packet/p0_completion.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args(argv)
    manifest = build_p0_completion_pack(
        project_root=args.project_root,
        output_dir=args.output_dir,
        canonical_dir=args.canonical_dir,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps({"output_dir": manifest["output_dir"], "row_counts": manifest["row_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
