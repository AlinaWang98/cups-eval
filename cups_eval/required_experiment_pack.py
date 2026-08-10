from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from cups_eval.cups_policy import (
    best_cups_vs_cot,
    cross_validate_cups,
    cross_validate_cups_group_holdout,
    write_cups_report,
)
from cups_eval.offline_router import load_feature_rows


STRATEGIES = ("direct", "cot", "self_consistency_3", "planner_verifier", "debate_2_judge")
COLLAB_STRATEGIES = ("cot", "self_consistency_3", "planner_verifier", "debate_2_judge")
MATH500_MODEL_LABELS = {
    "DeepSeek v4 flash": "deepseek_v4_flash",
    "GLM-5.2": "glm_5_2",
    "MiniMax-M3": "minimax_m3",
    "Qwen3.7-Max-Preview": "qwen_3_7_max_preview",
    "Kimi-K3": "kimi_k3",
}


def build_canonical_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("strategy") or ""),
                str(row.get("dataset") or ""),
                str(row.get("model") or ""),
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for (strategy, dataset, model), items in sorted(grouped.items()):
        output.append(_metric_row(strategy, dataset, model, items))

    by_item = _canonical_item_matrix(rows)
    oracle_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (dataset, model, _qid), strategies in by_item.items():
        oracle_groups[(dataset, model)].append(
            {
                "question_id": _qid,
                "correct": any(_to_bool(strategies.get(strategy, {}).get("correct")) for strategy in STRATEGIES),
                "parse_ok": any(_to_bool(strategies.get(strategy, {}).get("parse_ok")) for strategy in STRATEGIES),
                "estimated_cost_usd": _oracle_min_cost(strategies),
                "total_tokens": _oracle_min_tokens(strategies),
                "source_batch": _join_unique(
                    item.get("source_batch")
                    for item in strategies.values()
                ),
                "run_id": _join_unique(item.get("run_id") for item in strategies.values()),
                "source_raw_predictions": _join_unique(
                    item.get("source_raw_predictions")
                    for item in strategies.values()
                ),
            }
        )
    for (dataset, model), items in sorted(oracle_groups.items()):
        output.append(_metric_row("oracle_any", dataset, model, items))

    return output


def build_rescue_harm_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_item = _canonical_item_matrix(rows)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (dataset, model, _qid), strategies in by_item.items():
        direct = strategies.get("direct")
        if direct is None:
            continue
        direct_correct = _to_bool(direct.get("correct"))
        for comparison in COLLAB_STRATEGIES:
            other = strategies.get(comparison)
            if other is None:
                continue
            key = (dataset, model, comparison)
            record = grouped.setdefault(
                key,
                {
                    "dataset": dataset,
                    "model": model,
                    "comparison": comparison,
                    "n_aligned": 0,
                    "direct_correct": 0,
                    "comparison_correct": 0,
                    "both_correct": 0,
                    "both_wrong": 0,
                    "rescue": 0,
                    "harm": 0,
                },
            )
            comparison_correct = _to_bool(other.get("correct"))
            record["n_aligned"] += 1
            record["direct_correct"] += int(direct_correct)
            record["comparison_correct"] += int(comparison_correct)
            if direct_correct and comparison_correct:
                record["both_correct"] += 1
            elif direct_correct and not comparison_correct:
                record["harm"] += 1
            elif (not direct_correct) and comparison_correct:
                record["rescue"] += 1
            else:
                record["both_wrong"] += 1

    output = []
    for record in sorted(grouped.values(), key=lambda row: (row["dataset"], row["model"], row["comparison"])):
        n = int(record["n_aligned"])
        direct_correct = int(record["direct_correct"])
        comparison_correct = int(record["comparison_correct"])
        rescue = int(record["rescue"])
        harm = int(record["harm"])
        record["rescue_rate"] = rescue / n if n else 0.0
        record["harm_rate"] = harm / n if n else 0.0
        record["net_gain"] = rescue - harm
        record["net_gain_rate"] = (rescue - harm) / n if n else 0.0
        record["direct_accuracy"] = direct_correct / n if n else 0.0
        record["comparison_accuracy"] = comparison_correct / n if n else 0.0
        output.append(record)
    return output


def build_oracle_opportunity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_item = _canonical_item_matrix(rows)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for (dataset, model, _qid), strategies in by_item.items():
        direct = strategies.get("direct")
        if direct is None:
            continue
        key = (dataset, model)
        record = grouped.setdefault(
            key,
            {
                "dataset": dataset,
                "model": model,
                "n_aligned": 0,
                "direct_correct": 0,
                "direct_error": 0,
                "oracle_rescue": 0,
                "unresolved": 0,
            },
        )
        direct_correct = _to_bool(direct.get("correct"))
        any_collab_correct = any(
            _to_bool(strategies.get(strategy, {}).get("correct"))
            for strategy in COLLAB_STRATEGIES
        )
        record["n_aligned"] += 1
        if direct_correct:
            record["direct_correct"] += 1
        elif any_collab_correct:
            record["direct_error"] += 1
            record["oracle_rescue"] += 1
        else:
            record["direct_error"] += 1
            record["unresolved"] += 1

    output = []
    for record in sorted(grouped.values(), key=lambda row: (row["dataset"], row["model"])):
        n = int(record["n_aligned"])
        direct_correct = int(record["direct_correct"])
        oracle_rescue = int(record["oracle_rescue"])
        direct_error = int(record["direct_error"])
        record["direct_accuracy"] = direct_correct / n if n else 0.0
        record["oracle_any_accuracy"] = (direct_correct + oracle_rescue) / n if n else 0.0
        record["oracle_gap"] = record["oracle_any_accuracy"] - record["direct_accuracy"]
        record["rescue_rate_among_direct_errors"] = oracle_rescue / direct_error if direct_error else 0.0
        return_fields = record
        output.append(return_fields)
    return output


def paired_comparison_stats(
    selected_correct: Iterable[Any],
    baseline_correct: Iterable[Any],
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 13,
) -> dict[str, Any]:
    selected = [_to_bool(value) for value in selected_correct]
    baseline = [_to_bool(value) for value in baseline_correct]
    if len(selected) != len(baseline):
        raise ValueError("selected_correct and baseline_correct must have the same length")
    n = len(selected)
    diffs = [int(s) - int(b) for s, b in zip(selected, baseline)]
    selected_total = sum(int(value) for value in selected)
    baseline_total = sum(int(value) for value in baseline)
    rescues = sum(1 for s, b in zip(selected, baseline) if s and not b)
    harms = sum(1 for s, b in zip(selected, baseline) if (not s) and b)
    both_correct = sum(1 for s, b in zip(selected, baseline) if s and b)
    both_wrong = sum(1 for s, b in zip(selected, baseline) if (not s) and (not b))
    if n:
        low, high = _bootstrap_mean_ci(diffs, bootstrap_iterations=bootstrap_iterations, seed=seed)
    else:
        low, high = (0.0, 0.0)
    return {
        "n": n,
        "selected_correct": selected_total,
        "baseline_correct": baseline_total,
        "selected_accuracy": selected_total / n if n else 0.0,
        "baseline_accuracy": baseline_total / n if n else 0.0,
        "delta_accuracy": (selected_total - baseline_total) / n if n else 0.0,
        "paired_rescues": rescues,
        "paired_harms": harms,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_p_two_sided": _mcnemar_exact_p(rescues, harms),
        "bootstrap_delta_ci_low": low,
        "bootstrap_delta_ci_high": high,
    }


def router_fp_fn_rows(
    scope_id: str,
    scope_label: str,
    policy: str,
    predictions: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    *,
    selection_type: str = "",
    evaluation_regime: str = "",
) -> list[dict[str, Any]]:
    feature_by_key = {_row_key(row): row for row in feature_rows}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for prediction in predictions:
        if prediction.get("policy") and str(prediction.get("policy")) != policy:
            continue
        feature = feature_by_key.get(_row_key(prediction))
        if feature is None:
            continue
        dataset = str(prediction.get("dataset") or feature.get("dataset") or "")
        model = str(prediction.get("model") or feature.get("model") or "")
        key = (dataset, model)
        record = grouped.setdefault(
            key,
            {
                "scope_id": scope_id,
                "scope_label": scope_label,
                "selection_type": selection_type,
                "dataset": dataset,
                "model": model,
                "policy": policy,
                "n": 0,
                "true_positive": 0,
                "false_negative": 0,
                "false_positive": 0,
                "true_negative": 0,
            },
        )
        if evaluation_regime:
            record["evaluation_regime"] = evaluation_regime
        oracle_needed = (not _to_bool(feature.get("direct_correct"))) and any(
            _to_bool(feature.get(f"{strategy}_correct")) for strategy in COLLAB_STRATEGIES
        )
        escalated = str(prediction.get("selected_strategy") or "direct") != "direct"
        record["n"] += 1
        if oracle_needed and escalated:
            record["true_positive"] += 1
        elif oracle_needed and not escalated:
            record["false_negative"] += 1
        elif (not oracle_needed) and escalated:
            record["false_positive"] += 1
        else:
            record["true_negative"] += 1

    output = []
    for record in sorted(grouped.values(), key=lambda row: (row["dataset"], row["model"])):
        tp = int(record["true_positive"])
        fn = int(record["false_negative"])
        fp = int(record["false_positive"])
        tn = int(record["true_negative"])
        n = int(record["n"])
        record["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
        record["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
        record["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else 0.0
        record["false_negative_rate"] = fn / (fn + tp) if (fn + tp) else 0.0
        record["escalation_rate"] = (tp + fp) / n if n else 0.0
        output.append(record)
    return output


def build_required_experiment_pack(
    *,
    project_root: Path | None = None,
    experiment_data_dir: Path | None = None,
    output_dir: Path | None = None,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[1]
    canonical_dir = experiment_data_dir or root / "reports" / "cups_packet" / "experiment_data_by_strategy"
    destination = output_dir or root / "reports" / "cups_packet" / "required_experiments_pack"
    _prepare_output_dir(destination)

    canonical_rows = read_canonical_rows(canonical_dir)
    provenance_dir = destination / "provenance_audit"
    cups_dir = destination / "cups_stats"
    mechanism_dir = destination / "mechanism_diagnostics"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    cups_dir.mkdir(parents=True, exist_ok=True)
    mechanism_dir.mkdir(parents=True, exist_ok=True)

    canonical_metrics = build_canonical_metrics(canonical_rows)
    _write_csv(provenance_dir / "canonical_metrics_by_strategy_dataset_model.csv", canonical_metrics)
    discrepancies = build_math500_provenance_discrepancies(root, canonical_metrics)
    _write_csv(provenance_dir / "math500_provenance_discrepancies.csv", discrepancies)
    inactive_sources = build_math500_inactive_provenance_sources(root, canonical_metrics)
    _write_csv(provenance_dir / "math500_inactive_provenance_sources.csv", inactive_sources)
    raw_corrections = _read_csv(canonical_dir / "evaluation_audit" / "raw_vs_evaluated_corrections.csv")
    _write_csv(provenance_dir / "raw_vs_evaluated_corrections.csv", raw_corrections)
    (provenance_dir / "provenance_audit.md").write_text(
        render_provenance_audit(canonical_metrics, discrepancies, inactive_sources, raw_corrections),
        encoding="utf-8",
    )

    rescue_harm_rows = build_rescue_harm_rows(canonical_rows)
    oracle_rows = build_oracle_opportunity_rows(canonical_rows)
    _write_csv(mechanism_dir / "rescue_harm_by_dataset_model.csv", rescue_harm_rows)
    _write_csv(mechanism_dir / "oracle_opportunity_by_dataset_model.csv", oracle_rows)

    cups_artifacts = build_cups_artifacts(
        root,
        cups_dir,
        bootstrap_iterations=bootstrap_iterations,
    )
    _write_csv(cups_dir / "cups_best_policy_tests.csv", cups_artifacts["best_policy_tests"])
    _write_csv(cups_dir / "cups_per_dataset_model_metrics.csv", cups_artifacts["per_dataset_model_metrics"])
    _write_csv(cups_dir / "cups_strategy_mix_by_scope.csv", cups_artifacts["strategy_mix"])
    _write_csv(cups_dir / "cups_curve_all_scopes.csv", cups_artifacts["curve_rows"])
    _write_csv(cups_dir / "missing_cups_scopes.csv", cups_artifacts["missing_scopes"])
    _write_csv(mechanism_dir / "cups_router_fp_fn_by_scope_dataset_model.csv", cups_artifacts["router_fp_fn"])
    (cups_dir / "cups_stats.md").write_text(
        render_cups_stats(cups_artifacts),
        encoding="utf-8",
    )
    (mechanism_dir / "mechanism_diagnostics.md").write_text(
        render_mechanism_diagnostics(rescue_harm_rows, oracle_rows, cups_artifacts["router_fp_fn"]),
        encoding="utf-8",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(destination),
        "inputs": {
            "canonical_experiment_data": str(canonical_dir),
            "router_feature_scopes": cups_artifacts["scope_summaries"],
        },
        "outputs": {
            "provenance_audit": [
                "provenance_audit/canonical_metrics_by_strategy_dataset_model.csv",
                "provenance_audit/math500_provenance_discrepancies.csv",
                "provenance_audit/math500_inactive_provenance_sources.csv",
                "provenance_audit/raw_vs_evaluated_corrections.csv",
                "provenance_audit/provenance_audit.md",
            ],
            "cups_stats": [
                "cups_stats/cups_best_policy_tests.csv",
                "cups_stats/cups_per_dataset_model_metrics.csv",
                "cups_stats/cups_strategy_mix_by_scope.csv",
                "cups_stats/cups_curve_all_scopes.csv",
                "cups_stats/missing_cups_scopes.csv",
                "cups_stats/cups_stats.md",
                "cups_stats/generated_cups/*",
            ],
            "mechanism_diagnostics": [
                "mechanism_diagnostics/rescue_harm_by_dataset_model.csv",
                "mechanism_diagnostics/oracle_opportunity_by_dataset_model.csv",
                "mechanism_diagnostics/cups_router_fp_fn_by_scope_dataset_model.csv",
                "mechanism_diagnostics/mechanism_diagnostics.md",
            ],
        },
        "row_counts": {
            "canonical_rows": len(canonical_rows),
            "canonical_metrics": len(canonical_metrics),
            "math500_discrepancies": len(discrepancies),
            "math500_inactive_sources": len(inactive_sources),
            "raw_vs_evaluated_corrections": len(raw_corrections),
            "cups_best_policy_tests": len(cups_artifacts["best_policy_tests"]),
            "cups_per_dataset_model_metrics": len(cups_artifacts["per_dataset_model_metrics"]),
            "cups_strategy_mix": len(cups_artifacts["strategy_mix"]),
            "router_fp_fn": len(cups_artifacts["router_fp_fn"]),
            "rescue_harm": len(rescue_harm_rows),
            "oracle_opportunity": len(oracle_rows),
        },
        "notes": [
            "CUPS reports are regenerated offline from existing router_features.csv files.",
            "Canonical strategy metrics are computed from reports/cups_packet/experiment_data_by_strategy using evaluated/reparsed fields when available.",
            "Inactive MATH-500 reruns are audited separately from active source comparisons.",
            "Raw-vs-evaluated corrections are reported instead of silently overwriting parser/source differences.",
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (destination / "README.md").write_text(render_pack_readme(manifest), encoding="utf-8")
    return manifest


def build_cups_artifacts(
    project_root: Path,
    output_dir: Path,
    *,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    generated_dir = output_dir / "generated_cups"
    generated_dir.mkdir(parents=True, exist_ok=True)
    best_policy_tests: list[dict[str, Any]] = []
    per_dataset_model_metrics: list[dict[str, Any]] = []
    strategy_mix: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    router_fp_fn: list[dict[str, Any]] = []
    missing_scopes: list[dict[str, Any]] = []
    scope_summaries: list[dict[str, Any]] = []

    for scope in _cups_feature_scopes(project_root):
        rows = load_feature_rows(scope["batch_dir"])
        if not rows:
            missing_scopes.append(
                {
                    "scope_id": scope["scope_id"],
                    "scope_label": scope["scope_label"],
                    "dataset": scope.get("dataset", ""),
                    "model": scope.get("model", ""),
                    "reason": "no router_features.csv rows found",
                    "source_path": str(scope["batch_dir"]),
                }
            )
            continue
        feature_by_key = {_row_key(row): row for row in rows}
        for regime, result, unsupported_reason in _cups_evaluation_results(rows):
            if result is None:
                missing_scopes.append(
                    {
                        "scope_id": scope["scope_id"],
                        "scope_label": scope["scope_label"],
                        "evaluation_regime": regime,
                        "dataset": _join_unique(row.get("dataset") for row in rows),
                        "model": _join_unique(row.get("model") for row in rows),
                        "reason": unsupported_reason,
                        "source_path": str(scope["batch_dir"]),
                    }
                )
                continue
            result = _with_cups_evaluation_regime(result, regime)
            scope_output = generated_dir / scope["scope_id"] / regime
            write_cups_report(rows, result, scope_output)
            scope_summary = {
                "scope_id": scope["scope_id"],
                "scope_label": scope["scope_label"],
                "evaluation_regime": regime,
                "source_path": str(scope["batch_dir"]),
                "rows": len(rows),
                "dataset_models": sorted({f"{row.get('dataset')}-{row.get('model')}" for row in rows}),
                "generated_report_dir": str(scope_output),
            }
            scope_summaries.append(scope_summary)

            for curve in result.get("curve", []):
                curve_rows.append(
                    {
                        "scope_id": scope["scope_id"],
                        "scope_label": scope["scope_label"],
                        **curve,
                    }
                )

            selected_policies = selected_cups_policies(result.get("curve", []))
            all_predictions = list(result.get("predictions", []))
            for selection_type, policy in selected_policies:
                selected_predictions = [
                    row for row in all_predictions
                    if str(row.get("policy")) == policy
                ]
                curve = _find_curve_row(result.get("curve", []), policy)
                if not selected_predictions:
                    continue
                for baseline in ("direct", "cot"):
                    stats = _paired_stats_for_predictions(
                        selected_predictions,
                        feature_by_key,
                        baseline,
                        bootstrap_iterations=bootstrap_iterations,
                    )
                    best_policy_tests.append(
                        _policy_test_row(
                            scope,
                            regime,
                            selection_type,
                            policy,
                            baseline,
                            stats,
                            curve,
                            scope_output,
                        )
                    )
                    per_dataset_model_metrics.extend(
                        _per_dataset_policy_test_rows(
                            scope,
                            regime,
                            selection_type,
                            policy,
                            baseline,
                            selected_predictions,
                            feature_by_key,
                            curve,
                            scope_output,
                            bootstrap_iterations=bootstrap_iterations,
                        )
                    )
                strategy_mix.extend(
                    _strategy_mix_rows(
                        scope,
                        regime,
                        selection_type,
                        policy,
                        selected_predictions,
                    )
                )
                router_fp_fn.extend(
                    router_fp_fn_rows(
                        scope["scope_id"],
                        scope["scope_label"],
                        policy,
                        selected_predictions,
                        rows,
                        selection_type=selection_type,
                        evaluation_regime=regime,
                    )
                )

    missing_scopes.extend(_known_missing_cups_scopes(project_root))
    return {
        "best_policy_tests": best_policy_tests,
        "per_dataset_model_metrics": per_dataset_model_metrics,
        "strategy_mix": strategy_mix,
        "curve_rows": curve_rows,
        "router_fp_fn": router_fp_fn,
        "missing_scopes": missing_scopes,
        "scope_summaries": scope_summaries,
    }


def _cups_evaluation_results(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any] | None, str]]:
    results: list[tuple[str, dict[str, Any] | None, str]] = [
        ("item_oof", cross_validate_cups(rows, folds=5), ""),
    ]
    for regime, holdout in (
        ("leave_one_dataset_model_out", "dataset_model"),
        ("leave_one_model_out", "model"),
    ):
        try:
            results.append((regime, cross_validate_cups_group_holdout(rows, holdout=holdout), ""))
        except ValueError as exc:
            results.append((regime, None, str(exc)))
    return [
        row
        for row in results
    ]


def _with_cups_evaluation_regime(result: dict[str, Any], regime: str) -> dict[str, Any]:
    output = dict(result)
    output["evaluation_regime"] = regime
    for key in ("curve", "pareto", "predictions", "probabilities", "folds"):
        output[key] = [
            {**row, "evaluation_regime": regime}
            for row in result.get(key, [])
        ]
    return output


def build_math500_provenance_discrepancies(
    project_root: Path,
    canonical_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical_by_key = _math500_canonical_by_key(canonical_metrics)
    active_dirs_by_model = _active_math500_run_dirs(project_root, canonical_metrics)
    references = [
        ref
        for ref in _math500_reference_metrics(project_root)
        if _is_active_math500_reference(ref, active_dirs_by_model)
    ]
    return _math500_reference_comparisons(references, canonical_by_key)


def build_math500_inactive_provenance_sources(
    project_root: Path,
    canonical_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical_by_key = _math500_canonical_by_key(canonical_metrics)
    active_dirs_by_model = _active_math500_run_dirs(project_root, canonical_metrics)
    inactive = [
        ref
        for ref in _math500_reference_metrics(project_root)
        if not _is_active_math500_reference(ref, active_dirs_by_model)
    ]
    rows = _math500_reference_comparisons(inactive, canonical_by_key)
    for row in rows:
        if row.get("source_kind") == "math500_capability_gradient":
            row["status"] = "aggregate_reference"
        elif row.get("source_kind") == "summary_json" and _summary_superseded_by_router_features(row):
            row["status"] = "superseded_evaluation_artifact"
        elif row.get("status") != "missing_canonical":
            row["status"] = "superseded_source"
    return rows


def _math500_canonical_by_key(canonical_metrics: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (row["dataset"], row["model"], row["strategy"]): row
        for row in canonical_metrics
        if row.get("dataset") == "math500"
    }


def _active_math500_run_dirs(
    project_root: Path,
    canonical_metrics: list[dict[str, Any]],
) -> dict[str, set[Path]]:
    active: dict[str, set[Path]] = defaultdict(set)
    for row in canonical_metrics:
        if row.get("dataset") != "math500":
            continue
        model = str(row.get("model") or "")
        for raw_path in str(row.get("source_raw_predictions") or "").split(";"):
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = project_root / path
            active[model].add(path.parent.resolve())
    return active


def _is_active_math500_reference(ref: dict[str, Any], active_dirs_by_model: dict[str, set[Path]]) -> bool:
    if ref.get("source_kind") == "math500_capability_gradient":
        return True
    active_dirs = active_dirs_by_model.get(str(ref.get("model") or ""), set())
    if not active_dirs:
        return True
    path = Path(str(ref.get("source_path") or ""))
    parent = path.parent.resolve()
    if parent not in active_dirs:
        return False
    if ref.get("source_kind") == "summary_json" and (parent / "router_features.csv").exists():
        return False
    return True


def _summary_superseded_by_router_features(row: dict[str, Any]) -> bool:
    path = Path(str(row.get("source_path") or ""))
    return (path.parent / "router_features.csv").exists()


def _math500_reference_metrics(project_root: Path) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    gradient_path = project_root / "reports" / "cups_packet" / "math500_capability_gradient.csv"
    if gradient_path.exists():
        for row in _read_csv(gradient_path):
            if _to_int(row.get("complete_q")) <= 0:
                continue
            model = MATH500_MODEL_LABELS.get(str(row.get("model") or ""), str(row.get("model") or ""))
            for strategy in (*STRATEGIES, "oracle_any"):
                source_column = "oracle" if strategy == "oracle_any" else strategy
                if source_column not in row:
                    continue
                references.append(
                    {
                        "source_kind": "math500_capability_gradient",
                        "source_path": str(gradient_path),
                        "model": model,
                        "strategy": strategy,
                        "source_n": _to_int(row.get("complete_q")),
                        "source_accuracy": _to_float(row.get(source_column)),
                    }
                )

    for path in sorted((project_root / "runs").glob("**/runs/math500/*/*/summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        model = path.parts[-3]
        for source_strategy, metrics in payload.items():
            if not isinstance(metrics, dict) or "accuracy" not in metrics:
                continue
            strategy = "oracle_any" if source_strategy == "oracle_router" else source_strategy
            if strategy not in (*STRATEGIES, "oracle_any"):
                continue
            if _to_int(metrics.get("n")) < 50:
                continue
            references.append(
                {
                    "source_kind": "summary_json",
                    "source_path": str(path),
                    "model": model,
                    "strategy": strategy,
                    "source_n": _to_int(metrics.get("n")),
                    "source_accuracy": _to_float(metrics.get("accuracy")),
                }
            )

    for path in sorted((project_root / "runs").glob("**/runs/math500/*/*/router_features.csv")):
        rows = _read_csv(path)
        if not rows or len(rows) < 50:
            continue
        model = path.parts[-3]
        for strategy in STRATEGIES:
            column = _router_feature_correct_column(strategy)
            if column not in rows[0]:
                continue
            references.append(
                {
                    "source_kind": "router_features",
                    "source_path": str(path),
                    "model": model,
                    "strategy": strategy,
                    "source_n": len(rows),
                    "source_accuracy": _rate(rows, lambda row, col=column: _to_bool(row.get(col))),
                }
            )
        references.append(
            {
                "source_kind": "router_features",
                "source_path": str(path),
                "model": model,
                "strategy": "oracle_any",
                "source_n": len(rows),
                "source_accuracy": _rate(rows, lambda row: _to_bool(row.get(_router_feature_correct_column("oracle_any")))),
            }
        )

    return references


def _math500_reference_comparisons(
    references: list[dict[str, Any]],
    canonical_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for ref in references:
        key = ("math500", ref["model"], ref["strategy"])
        canonical = canonical_by_key.get(key)
        canonical_n = canonical.get("n") if canonical else ""
        canonical_accuracy = canonical.get("accuracy") if canonical else ""
        if canonical is None:
            delta = ""
            status = "missing_canonical"
        else:
            delta_value = float(ref["source_accuracy"]) - float(canonical["accuracy"])
            delta = delta_value
            if _to_int(ref["source_n"]) != _to_int(canonical["n"]):
                status = "n_mismatch"
            elif abs(delta_value) < 1e-12:
                status = "match"
            elif abs(delta_value) <= 0.011:
                status = "minor_delta"
            else:
                status = "major_delta"
        output.append(
            {
                **ref,
                "canonical_n": canonical_n,
                "canonical_accuracy": canonical_accuracy,
                "delta_source_minus_canonical": delta,
                "status": status,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["model"],
            row["strategy"],
            row["source_kind"],
            row["source_path"],
        ),
    )


def read_canonical_rows(experiment_data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for path in sorted((experiment_data_dir / strategy).glob("*.csv")):
            for row in _read_csv(path):
                row["strategy"] = strategy
                rows.append(row)
    return rows


def render_provenance_audit(
    canonical_metrics: list[dict[str, Any]],
    discrepancies: list[dict[str, Any]],
    inactive_sources: list[dict[str, Any]] | None = None,
    raw_corrections: list[dict[str, Any]] | None = None,
) -> str:
    inactive_sources = inactive_sources or []
    raw_corrections = raw_corrections or []
    major = [row for row in discrepancies if row.get("status") == "major_delta"]
    n_mismatch = [row for row in discrepancies if row.get("status") == "n_mismatch"]
    lines = [
        "# Provenance Audit",
        "",
        "Canonical source: `reports/cups_packet/experiment_data_by_strategy`.",
        "",
        f"- Canonical metric rows: {len(canonical_metrics)}.",
        f"- Active MATH-500 reference comparisons: {len(discrepancies)}.",
        f"- Inactive/superseded MATH-500 reference rows: {len(inactive_sources)}.",
        f"- Raw-vs-evaluated correction rows: {len(raw_corrections)}.",
        f"- Major accuracy deltas: {len(major)}.",
        f"- Sample-size mismatches: {len(n_mismatch)}.",
        "",
        "## Important Flags",
        "",
    ]
    if not major and not n_mismatch:
        lines.append("- No major MATH-500 provenance discrepancies were detected.")
    else:
        for row in sorted(major, key=lambda item: -abs(_to_float(item.get("delta_source_minus_canonical"))))[:12]:
            lines.append(
                "- "
                f"{row['model']} `{row['strategy']}`: "
                f"{_fmt_rate(row.get('canonical_accuracy'))} canonical vs "
                f"{_fmt_rate(row.get('source_accuracy'))} in {row['source_kind']} "
                f"(delta {100 * _to_float(row.get('delta_source_minus_canonical')):+.1f} pp)."
            )
        for row in n_mismatch[:8]:
            lines.append(
                "- "
                f"{row['model']} `{row['strategy']}` has n={row['source_n']} in {row['source_kind']} "
                f"vs canonical n={row['canonical_n']}."
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `canonical_metrics_by_strategy_dataset_model.csv` is the frozen evaluated canonical metric table.",
            "- `math500_provenance_discrepancies.csv` compares canonical MATH-500 metrics with active evaluated sources and the capability-gradient table.",
            "- `math500_inactive_provenance_sources.csv` keeps superseded reruns and archival summaries out of active comparisons while preserving traceability.",
            "- `raw_vs_evaluated_corrections.csv` records item-level raw-to-evaluated answer/correctness changes.",
            "",
            "## Interpretation Boundary",
            "",
            "Canonical MATH-500 numbers should be quoted from the evaluated table. Raw/source disagreements are not silently hidden: active discrepancies remain flagged, and superseded or raw-vs-evaluated differences are retained as audit artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_cups_stats(artifacts: dict[str, Any]) -> str:
    lines = [
        "# CUPS Statistical Package",
        "",
        "CUPS was regenerated offline from existing `router_features.csv` files. No model/API calls are involved.",
        "",
        "## Scope Summary",
        "",
        "| scope | rows | dataset-models |",
        "|---|---:|---|",
    ]
    for scope in artifacts["scope_summaries"]:
        lines.append(
            f"| `{scope['scope_id']}` | {scope['rows']} | "
            f"{', '.join(scope['dataset_models'])} |"
        )
    lines.extend(
        [
            "",
            "## Best-Policy Tests",
            "",
            "| scope | selection | policy | baseline | delta | 95% CI | p |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in artifacts["best_policy_tests"]:
        lines.append(
            f"| `{row['scope_id']}` | {row['selection_type']} | `{row['policy']}` | "
            f"{row['baseline']} | {100 * _to_float(row['delta_accuracy']):+.1f} pp | "
            f"[{100 * _to_float(row['bootstrap_delta_ci_low']):+.1f}, "
            f"{100 * _to_float(row['bootstrap_delta_ci_high']):+.1f}] | "
            f"{_fmt_p(row['mcnemar_exact_p_two_sided'])} |"
        )
    if artifacts["missing_scopes"]:
        lines.extend(["", "## Missing/Excluded CUPS Scopes", ""])
        for row in artifacts["missing_scopes"]:
            lines.append(
                f"- `{row.get('dataset', '')}-{row.get('model', '')}` ({row['scope_id']}): {row['reason']}."
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `cups_best_policy_tests.csv`: scope-level CUPS vs direct/CoT paired tests.",
            "- `cups_per_dataset_model_metrics.csv`: dataset-model paired tests.",
            "- `cups_strategy_mix_by_scope.csv`: selected strategy distributions.",
            "- `cups_curve_all_scopes.csv`: all CUPS and baseline curve rows.",
            "- `generated_cups/*`: regenerated CUPS prediction/curve reports per scope.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_mechanism_diagnostics(
    rescue_harm: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    router_rows: list[dict[str, Any]],
) -> str:
    top_rescue = sorted(rescue_harm, key=lambda row: _to_float(row.get("net_gain_rate")), reverse=True)[:10]
    lines = [
        "# Mechanism Diagnostics",
        "",
        "Diagnostics are split into canonical rescue/harm decompositions and CUPS router FP/FN tables.",
        "",
        "## Largest Canonical Net Gains",
        "",
        "| dataset | model | comparison | direct | comparison | net |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in top_rescue:
        lines.append(
            f"| {row['dataset']} | {row['model']} | `{row['comparison']}` | "
            f"{_fmt_rate(row['direct_accuracy'])} | {_fmt_rate(row['comparison_accuracy'])} | "
            f"{100 * _to_float(row['net_gain_rate']):+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## Oracle Opportunity Snapshot",
            "",
            "| dataset | model | direct acc | oracle acc | gap | direct errors rescued |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(oracle_rows, key=lambda item: _to_float(item.get("oracle_gap")), reverse=True)[:12]:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {_fmt_rate(row['direct_accuracy'])} | "
            f"{_fmt_rate(row['oracle_any_accuracy'])} | {100 * _to_float(row['oracle_gap']):+.1f} pp | "
            f"{_fmt_rate(row['rescue_rate_among_direct_errors'])} |"
        )
    lines.extend(
        [
            "",
            "## Router FP/FN Snapshot",
            "",
            "| scope | dataset | model | policy | TP | FN | FP | TN | recall | FPR |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in router_rows[:18]:
        lines.append(
            f"| `{row['scope_id']}` | {row['dataset']} | {row['model']} | `{row['policy']}` | "
            f"{row['true_positive']} | {row['false_negative']} | {row['false_positive']} | "
            f"{row['true_negative']} | {_fmt_rate(row['recall'])} | {_fmt_rate(row['false_positive_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `rescue_harm_by_dataset_model.csv`: direct vs each collaboration strategy.",
            "- `oracle_opportunity_by_dataset_model.csv`: direct-first oracle rescue opportunity.",
            "- `cups_router_fp_fn_by_scope_dataset_model.csv`: CUPS escalation diagnosis.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_pack_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Required Experiments Pack",
        "",
        "This directory contains the required offline experiment supplements for the manuscript.",
        "",
        "## Contents",
        "",
        "- `provenance_audit/`: frozen canonical metrics and MATH-500 source discrepancy audit.",
        "- `cups_stats/`: regenerated CUPS outputs, paired tests, bootstrap CIs, and strategy mix.",
        "- `mechanism_diagnostics/`: rescue/harm decomposition, oracle opportunity, and router FP/FN diagnostics.",
        "",
        "## Row Counts",
        "",
    ]
    for key, value in manifest["row_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- All CUPS outputs are offline analyses over existing `router_features.csv` files.",
            "- Canonical correctness uses evaluated/reparsed artifact fields where available; raw responses remain linked as provenance.",
            "- MATH-500 parser/source disagreements are preserved in active, inactive, and raw-vs-evaluated audit files.",
            "- The Kimi-K3 MATH-500 ceiling-control raw run is included in canonical metrics but excluded from CUPS because no `router_features.csv` exists for that run.",
        ]
    )
    return "\n".join(lines) + "\n"


def _metric_row(strategy: str, dataset: str, model: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    correct = sum(1 for row in items if _to_bool(row.get("correct")))
    parse_ok = sum(1 for row in items if _to_bool(row.get("parse_ok")))
    return {
        "strategy": strategy,
        "dataset": dataset,
        "model": model,
        "n": n,
        "unique_questions": len({str(row.get("question_id") or "") for row in items}),
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "parse_ok": parse_ok,
        "parse_rate": parse_ok / n if n else 0.0,
        "total_tokens": sum(_to_float(row.get("total_tokens")) for row in items),
        "estimated_cost_usd": sum(_to_float(row.get("estimated_cost_usd")) for row in items),
        "source_batches": _join_unique(row.get("source_batch") for row in items),
        "run_ids": _join_unique(row.get("run_id") for row in items),
        "source_raw_predictions": _join_unique(row.get("source_raw_predictions") for row in items),
    }


def _canonical_item_matrix(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    by_item: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row.get("dataset") or ""),
            str(row.get("model") or ""),
            str(row.get("question_id") or ""),
        )
        by_item[key][str(row.get("strategy") or "")] = row
    return by_item


def _oracle_min_cost(strategies: dict[str, dict[str, Any]]) -> float:
    correct_costs = [
        _to_float(row.get("estimated_cost_usd"))
        for row in strategies.values()
        if _to_bool(row.get("correct")) and _to_float(row.get("estimated_cost_usd")) > 0
    ]
    if correct_costs:
        return min(correct_costs)
    costs = [
        _to_float(row.get("estimated_cost_usd"))
        for row in strategies.values()
        if _to_float(row.get("estimated_cost_usd")) > 0
    ]
    return min(costs) if costs else 0.0


def _oracle_min_tokens(strategies: dict[str, dict[str, Any]]) -> float:
    correct_tokens = [
        _to_float(row.get("total_tokens"))
        for row in strategies.values()
        if _to_bool(row.get("correct")) and _to_float(row.get("total_tokens")) > 0
    ]
    if correct_tokens:
        return min(correct_tokens)
    tokens = [_to_float(row.get("total_tokens")) for row in strategies.values() if _to_float(row.get("total_tokens")) > 0]
    return min(tokens) if tokens else 0.0


def _paired_stats_for_predictions(
    predictions: list[dict[str, Any]],
    feature_by_key: dict[tuple[str, str, str], dict[str, Any]],
    baseline: str,
    *,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    selected_values = []
    baseline_values = []
    for prediction in predictions:
        feature = feature_by_key.get(_row_key(prediction))
        if feature is None:
            continue
        selected_values.append(prediction.get("selected_correct"))
        baseline_values.append(feature.get("direct_correct" if baseline == "direct" else "cot_correct"))
    return paired_comparison_stats(
        selected_values,
        baseline_values,
        bootstrap_iterations=bootstrap_iterations,
    )


def _policy_test_row(
    scope: dict[str, Any],
    evaluation_regime: str,
    selection_type: str,
    policy: str,
    baseline: str,
    stats: dict[str, Any],
    curve: dict[str, Any],
    report_dir: Path,
) -> dict[str, Any]:
    return {
        "scope_id": scope["scope_id"],
        "scope_label": scope["scope_label"],
        "evaluation_regime": evaluation_regime,
        "selection_type": selection_type,
        "policy": policy,
        "baseline": baseline,
        **stats,
        "cups_cost_multiplier": curve.get("cost_multiplier", ""),
        "cups_relative_cost_per_correct": curve.get("relative_cost_per_correct", ""),
        "cups_estimated_cost_usd": curve.get("estimated_cost_usd", ""),
        "cups_escalation_rate": curve.get("escalation_rate", ""),
        "generated_cups_report_dir": str(report_dir),
    }


def _per_dataset_policy_test_rows(
    scope: dict[str, Any],
    evaluation_regime: str,
    selection_type: str,
    policy: str,
    baseline: str,
    predictions: list[dict[str, Any]],
    feature_by_key: dict[tuple[str, str, str], dict[str, Any]],
    curve: dict[str, Any],
    report_dir: Path,
    *,
    bootstrap_iterations: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[(str(prediction.get("dataset") or ""), str(prediction.get("model") or ""))].append(prediction)
    output = []
    for (dataset, model), items in sorted(grouped.items()):
        stats = _paired_stats_for_predictions(
            items,
            feature_by_key,
            baseline,
            bootstrap_iterations=bootstrap_iterations,
        )
        output.append(
            {
                "scope_id": scope["scope_id"],
                "scope_label": scope["scope_label"],
                "evaluation_regime": evaluation_regime,
                "dataset": dataset,
                "model": model,
                "selection_type": selection_type,
                "policy": policy,
                "baseline": baseline,
                **stats,
                "scope_cups_cost_multiplier": curve.get("cost_multiplier", ""),
                "scope_cups_relative_cost_per_correct": curve.get("relative_cost_per_correct", ""),
                "generated_cups_report_dir": str(report_dir),
            }
        )
    return output


def _strategy_mix_rows(
    scope: dict[str, Any],
    evaluation_regime: str,
    selection_type: str,
    policy: str,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for prediction in predictions:
        dataset = str(prediction.get("dataset") or "")
        model = str(prediction.get("model") or "")
        selected = str(prediction.get("selected_strategy") or "")
        grouped[(dataset, model, selected)] += 1
        totals[(dataset, model)] += 1
    output = []
    for (dataset, model, strategy), n in sorted(grouped.items()):
        total = totals[(dataset, model)]
        output.append(
            {
                "scope_id": scope["scope_id"],
                "scope_label": scope["scope_label"],
                "evaluation_regime": evaluation_regime,
                "dataset": dataset,
                "model": model,
                "selection_type": selection_type,
                "policy": policy,
                "selected_strategy": strategy,
                "n": n,
                "rate": n / total if total else 0.0,
            }
        )
    return output


def selected_cups_policies(curve: list[dict[str, Any]]) -> list[tuple[str, str]]:
    cups_rows = [row for row in curve if row.get("policy_family") == "cups"]
    if not cups_rows:
        return []
    best_accuracy = sorted(
        cups_rows,
        key=lambda row: (
            -_to_float(row.get("accuracy")),
            _to_float(row.get("relative_cost_per_correct")),
            _to_float(row.get("cost_multiplier")),
            str(row.get("policy")),
        ),
    )[0]
    best_vs_cot = best_cups_vs_cot(curve)
    return [
        ("best_accuracy", str(best_accuracy.get("policy"))),
        ("best_vs_cot_cpc", str(best_vs_cot.get("policy"))),
    ]


def _find_curve_row(curve: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    for row in curve:
        if str(row.get("policy")) == policy:
            return row
    return {}


def _cups_feature_scopes(project_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "scope_id": "medical_glm_kimi",
            "scope_label": "Medical GLM/Kimi",
            "batch_dir": project_root / "runs" / "phase2_balanced" / "20260728T153221Z",
        },
        {
            "scope_id": "nonmedical_glm_kimi",
            "scope_label": "Nonmedical GLM/Kimi",
            "batch_dir": project_root / "runs" / "phase2_nonmedical_sanity" / "20260729T033232Z",
        },
        {
            "scope_id": "nonmedical_deepseek_flash",
            "scope_label": "Nonmedical DeepSeek Flash",
            "batch_dir": project_root / "runs" / "phase2_nonmedical_sanity_deepseek_flash" / "20260729T020951Z",
        },
        {
            "scope_id": "math500_core_deepseek_glm",
            "scope_label": "MATH-500 DeepSeek/GLM",
            "batch_dir": project_root / "runs" / "phase3_math500_core_models" / "20260729T122614Z",
        },
        {
            "scope_id": "math500_minimax_m3",
            "scope_label": "MATH-500 MiniMax-M3",
            "batch_dir": project_root / "runs" / "phase3_minimax_math500" / "20260729T085816Z",
        },
        {
            "scope_id": "math500_qwen_clean",
            "scope_label": "MATH-500 Qwen clean",
            "batch_dir": project_root / "runs" / "phase3_qwen_math500_clean" / "20260730Tclean001",
        },
    ]


def _known_missing_cups_scopes(project_root: Path) -> list[dict[str, Any]]:
    kimi_raw = (
        project_root
        / "runs"
        / "phase3_math500_core_models"
        / "20260729T122614Z"
        / "runs"
        / "math500"
        / "kimi_k3"
        / "20260729T122615Z"
        / "raw_predictions.jsonl"
    )
    feature_path = kimi_raw.with_name("router_features.csv")
    if kimi_raw.exists() and not feature_path.exists():
        return [
            {
                "scope_id": "math500_kimi_k3",
                "scope_label": "MATH-500 Kimi-K3 ceiling-control",
                "dataset": "math500",
                "model": "kimi_k3",
                "reason": "raw ceiling-control exists, but router_features.csv is absent; CUPS requires feature rows",
                "source_path": str(kimi_raw),
            }
        ]
    return []


def _mcnemar_exact_p(rescues: int, harms: int) -> float:
    discordant = rescues + harms
    if discordant == 0:
        return 1.0
    tail = min(rescues, harms)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def _bootstrap_mean_ci(
    values: list[float],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(bootstrap_iterations):
        total = 0.0
        for _index in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return _percentile_sorted(means, 0.025), _percentile_sorted(means, 0.975)


def _percentile_sorted(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = percentile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower]) * (1 - weight) + float(values[upper]) * weight


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("model") or ""),
        str(row.get("question_id") or ""),
    )


def _rate(rows: list[dict[str, Any]], predicate: Any) -> float:
    return sum(1 for row in rows if predicate(row)) / len(rows) if rows else 0.0


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _join_unique(values: Iterable[Any]) -> str:
    unique = sorted({str(value) for value in values if value not in {None, ""}})
    return ";".join(unique)


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


def _prepare_output_dir(path: Path) -> None:
    if path.exists():
        if path.name != "required_experiments_pack":
            raise ValueError(f"Refusing to clear unexpected output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _fmt_rate(value: Any) -> str:
    return f"{100 * _to_float(value):.1f}%"


def _fmt_p(value: Any) -> str:
    p = _to_float(value)
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _router_feature_correct_column(strategy: str) -> str:
    if strategy == "direct":
        return "direct_correct"
    if strategy == "self_consistency_3":
        return "self_consistency_correct"
    if strategy == "oracle_any":
        return "oracle_correct"
    return f"{strategy}_correct"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the required offline experiment pack.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--experiment-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args(argv)

    manifest = build_required_experiment_pack(
        project_root=args.project_root,
        experiment_data_dir=args.experiment_data_dir,
        output_dir=args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps({"output_dir": manifest["output_dir"], "row_counts": manifest["row_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
