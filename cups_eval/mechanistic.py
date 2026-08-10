from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from cups_eval.offline_router import (
    available_collab_strategies,
    strategy_cost,
    strategy_prefix,
)


NUMERIC_DIAGNOSTIC_FEATURES = (
    "direct_confidence",
    "direct_rationale_word_count",
    "question_word_count",
    "question_char_count",
    "option_mean_word_count",
    "option_max_char_count",
    "option_max_similarity",
    "question_option_max_overlap",
    "question_option_mean_overlap",
    "self_consistency_entropy",
    "self_consistency_vote_margin",
    "answer_disagreement_count",
)

MODEL_INDEPENDENT_COLUMNS = (
    "dataset",
    "model",
    "question_id",
    "gold",
    "direct_answer",
    "direct_confidence",
    "direct_rationale_word_count",
    "question_word_count",
    "question_char_count",
    "option_mean_word_count",
    "option_max_char_count",
    "option_min_char_count",
    "option_max_similarity",
    "question_option_max_overlap",
    "question_option_mean_overlap",
    "has_negation_or_except",
    "subject_name",
    "topic_name",
    "self_consistency_entropy",
    "self_consistency_vote_margin",
    "self_consistency_stable_majority",
    "answer_disagreement_count",
    "direct_correct",
    "oracle_any_rescuable",
    "cot_correct",
    "self_consistency_correct",
    "planner_verifier_correct",
    "debate_2_judge_correct",
)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_case_type_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for comparison in (*available_collab_strategies(rows), "oracle_any"):
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        totals: dict[tuple[str, str], int] = defaultdict(int)
        for row in rows:
            dataset = str(row.get("dataset") or "unknown")
            model = str(row.get("model") or "unknown")
            case_type = _case_type(row, comparison)
            key = (dataset, model, case_type)
            totals[(dataset, model)] += 1
            if key not in grouped:
                grouped[key] = {
                    "dataset": dataset,
                    "model": model,
                    "comparison": comparison,
                    "case_type": case_type,
                    "n": 0,
                    "mean_direct_confidence": 0.0,
                    "mean_question_word_count": 0.0,
                }
            grouped[key]["n"] += 1
            grouped[key]["mean_direct_confidence"] += _to_float(row.get("direct_confidence"))
            grouped[key]["mean_question_word_count"] += _to_float(row.get("question_word_count"))
        for (dataset, model, _case), record in grouped.items():
            n = int(record["n"])
            total = totals[(dataset, model)]
            record["rate_within_dataset_model"] = n / total if total else 0.0
            record["mean_direct_confidence"] = record["mean_direct_confidence"] / n if n else 0.0
            record["mean_question_word_count"] = record["mean_question_word_count"] / n if n else 0.0
            records.append(record)
    return sorted(records, key=lambda row: (row["comparison"], row["dataset"], row["model"], row["case_type"]))


def build_confidence_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for comparison in (*available_collab_strategies(rows), "oracle_any"):
        for row in rows:
            key = (
                str(row.get("dataset") or "unknown"),
                str(row.get("model") or "unknown"),
                comparison,
                _case_type(row, comparison),
            )
            buckets[key].append(_to_float(row.get("direct_confidence")))
    output = []
    for (dataset, model, comparison, case_type), values in sorted(buckets.items()):
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "comparison": comparison,
                "case_type": case_type,
                "n": len(values),
                "confidence_mean": _mean(values),
                "confidence_p25": _percentile(values, 0.25),
                "confidence_median": _percentile(values, 0.50),
                "confidence_p75": _percentile(values, 0.75),
                "confidence_low_rate": _rate(values, lambda value: value < 0.70),
                "confidence_high_rate": _rate(values, lambda value: value >= 0.90),
            }
        )
    return output


def build_difficulty_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for comparison in (*available_collab_strategies(rows), "oracle_any"):
        for row in rows:
            grouped[
                (
                    str(row.get("dataset") or "unknown"),
                    str(row.get("model") or "unknown"),
                    comparison,
                    _case_type(row, comparison),
                )
            ].append(row)
    output = []
    for (dataset, model, comparison, case_type), items in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "comparison": comparison,
                "case_type": case_type,
                "n": len(items),
                "question_words_mean": _mean_feature(items, "question_word_count"),
                "question_chars_mean": _mean_feature(items, "question_char_count"),
                "option_similarity_mean": _mean_feature(items, "option_max_similarity"),
                "answer_disagreement_mean": _mean_feature(items, "answer_disagreement_count"),
                "self_consistency_entropy_mean": _mean_feature(items, "self_consistency_entropy"),
                "self_consistency_margin_mean": _mean_feature(items, "self_consistency_vote_margin"),
            }
        )
    return output


def build_router_diagnostic_rows(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prediction_by_key = {
        _row_key(prediction): prediction
        for prediction in predictions
    }
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        prediction = prediction_by_key.get(_row_key(row), {})
        diagnostic_type = _router_diagnostic_type(row, prediction)
        dataset = str(row.get("dataset") or "unknown")
        model = str(row.get("model") or "unknown")
        key = (dataset, model, diagnostic_type)
        totals[(dataset, model)] += 1
        if key not in grouped:
            grouped[key] = {
                "dataset": dataset,
                "model": model,
                "diagnostic_type": diagnostic_type,
                "n": 0,
                "mean_direct_confidence": 0.0,
                "mean_question_word_count": 0.0,
                "mean_answer_disagreement": 0.0,
            }
        grouped[key]["n"] += 1
        grouped[key]["mean_direct_confidence"] += _to_float(row.get("direct_confidence"))
        grouped[key]["mean_question_word_count"] += _to_float(row.get("question_word_count"))
        grouped[key]["mean_answer_disagreement"] += _to_float(row.get("answer_disagreement_count"))
    for (dataset, model, _diagnostic), record in grouped.items():
        n = int(record["n"])
        total = totals[(dataset, model)]
        record["rate_within_dataset_model"] = n / total if total else 0.0
        for field in (
            "mean_direct_confidence",
            "mean_question_word_count",
            "mean_answer_disagreement",
        ):
            record[field] = record[field] / n if n else 0.0
    return sorted(grouped.values(), key=lambda row: (row["dataset"], row["model"], row["diagnostic_type"]))


def build_router_feature_gap_rows(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prediction_by_key = {_row_key(prediction): prediction for prediction in predictions}
    typed_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        typed_rows[_router_diagnostic_type(row, prediction_by_key.get(_row_key(row), {}))].append(row)
    contrasts = [
        ("false_negative_vs_true_positive", "false_negative", "true_positive"),
        ("false_positive_vs_true_negative", "false_positive", "true_negative"),
    ]
    output = []
    for contrast, left_name, right_name in contrasts:
        left = typed_rows.get(left_name, [])
        right = typed_rows.get(right_name, [])
        for feature in NUMERIC_DIAGNOSTIC_FEATURES:
            left_values = [_to_float(row.get(feature)) for row in left]
            right_values = [_to_float(row.get(feature)) for row in right]
            left_mean = _mean(left_values)
            right_mean = _mean(right_values)
            pooled = _pooled_std(left_values, right_values)
            output.append(
                {
                    "contrast": contrast,
                    "feature": feature,
                    "left_group": left_name,
                    "right_group": right_name,
                    "left_n": len(left_values),
                    "right_n": len(right_values),
                    "left_mean": left_mean,
                    "right_mean": right_mean,
                    "mean_diff": left_mean - right_mean,
                    "standardized_diff": (left_mean - right_mean) / pooled if pooled else 0.0,
                }
            )
    return sorted(output, key=lambda row: (row["contrast"], -abs(float(row["standardized_diff"])), row["feature"]))


def build_model_independent_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = {}
        for column in MODEL_INDEPENDENT_COLUMNS:
            if column == "oracle_any_rescuable":
                item[column] = _oracle_any_rescuable(row)
            elif column in row:
                item[column] = row[column]
            else:
                item[column] = ""
        output.append(item)
    return output


def write_mechanistic_report(
    rows: list[dict[str, Any]],
    router_predictions: list[dict[str, Any]],
    routing_curve: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows = build_case_type_rows(rows)
    confidence_rows = build_confidence_case_rows(rows)
    difficulty_rows = build_difficulty_case_rows(rows)
    diagnostic_rows = build_router_diagnostic_rows(rows, router_predictions)
    feature_gap_rows = build_router_feature_gap_rows(rows, router_predictions)
    model_independent_rows = build_model_independent_feature_rows(rows)
    fig1_rows = build_fig1_cost_accuracy_rows(rows, router_predictions, routing_curve)
    fig3_rows = [
        row for row in case_rows
        if row["case_type"] in {"rescue", "harm"}
    ]

    _write_csv(output_dir / "case_type_by_dataset.csv", case_rows)
    _write_csv(output_dir / "confidence_by_case.csv", confidence_rows)
    _write_csv(output_dir / "difficulty_by_case.csv", difficulty_rows)
    _write_csv(output_dir / "router_diagnostics.csv", diagnostic_rows)
    _write_csv(output_dir / "router_feature_gaps.csv", feature_gap_rows)
    _write_csv(output_dir / "model_independent_features.csv", model_independent_rows)
    _write_csv(output_dir / "fig1_cost_accuracy_points.csv", fig1_rows)
    _write_csv(output_dir / "fig2_routing_curve.csv", routing_curve)
    _write_csv(output_dir / "fig3_rescue_harm_by_dataset.csv", fig3_rows)
    _write_csv(output_dir / "fig4_router_fp_fn_breakdown.csv", diagnostic_rows)
    (output_dir / "mechanistic_summary.json").write_text(
        json.dumps(
            {
                "n": len(rows),
                "fig1_points": fig1_rows,
                "top_router_feature_gaps": feature_gap_rows[:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "mechanistic_report.md").write_text(
        _render_report(rows, fig1_rows, case_rows, confidence_rows, diagnostic_rows, feature_gap_rows),
        encoding="utf-8",
    )


def build_fig1_cost_accuracy_rows(
    rows: list[dict[str, Any]],
    router_predictions: list[dict[str, Any]],
    routing_curve: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    direct_cost = sum(_to_float(row.get("direct_cost_usd")) for row in rows)
    available_strategies = available_collab_strategies(rows)
    output = []
    output.append(_policy_point(rows, "direct", "baseline", True, "solid"))
    for strategy in available_strategies:
        output.append(_policy_point(rows, f"always_{strategy}", strategy, True, "solid"))
    output.append(_direct_first_oracle_point(rows, direct_cost))
    if "debate_2_judge" in available_strategies:
        output.append(_posthoc_oracle_point(rows, "debate_2_judge", direct_cost))
    if router_predictions:
        output.append(_router_prediction_point(rows, router_predictions, "learned_rule_list"))
    if routing_curve:
        calibrated = [
            row for row in routing_curve
            if row.get("policy_family") == "calibrated_logistic"
        ]
        if calibrated:
            best_accuracy = sorted(
                calibrated,
                key=lambda row: (-_to_float(row.get("accuracy")), _to_float(row.get("cost_multiplier"))),
            )[0]
            best_low_cost_pool = [
                row for row in calibrated
                if _to_float(row.get("cost_multiplier")) <= 1.30
            ]
            output.append(_routing_curve_point(best_accuracy, "calibrated_best_accuracy"))
            if best_low_cost_pool:
                output.append(
                    _routing_curve_point(
                        sorted(
                            best_low_cost_pool,
                            key=lambda row: (-_to_float(row.get("accuracy")), _to_float(row.get("cost_multiplier"))),
                        )[0],
                        "calibrated_best_leq_1p3x",
                    )
                )
    return output


def _case_type(row: dict[str, Any], comparison: str) -> str:
    direct_correct = _to_bool(row.get("direct_correct"))
    if comparison == "oracle_any":
        if direct_correct:
            return "direct_correct"
        return "rescue" if _oracle_any_rescuable(row) else "unresolved"
    comparison_correct = _strategy_correct(row, comparison)
    if direct_correct and comparison_correct:
        return "both_correct"
    if direct_correct and not comparison_correct:
        return "harm"
    if not direct_correct and comparison_correct:
        return "rescue"
    return "both_wrong"


def _router_diagnostic_type(row: dict[str, Any], prediction: dict[str, Any]) -> str:
    oracle_needed = (not _to_bool(row.get("direct_correct"))) and _oracle_any_rescuable(row)
    selected_strategy = str(prediction.get("selected_strategy") or "direct")
    escalated = selected_strategy != "direct"
    if oracle_needed and escalated:
        return "true_positive"
    if oracle_needed and not escalated:
        return "false_negative"
    if (not oracle_needed) and escalated:
        return "false_positive"
    return "true_negative"


def _policy_point(
    rows: list[dict[str, Any]],
    policy: str,
    strategy: str,
    deployable: bool,
    line_style: str,
) -> dict[str, Any]:
    direct_cost = sum(_to_float(row.get("direct_cost_usd")) for row in rows)
    if strategy == "baseline":
        correct = sum(1 for row in rows if _to_bool(row.get("direct_correct")))
        cost = direct_cost
        escalation_rate = 0.0
    else:
        correct = sum(1 for row in rows if _strategy_correct(row, strategy))
        cost = sum(strategy_cost(row, strategy) for row in rows)
        escalation_rate = 1.0
    n = len(rows)
    return {
        "policy": policy,
        "point_type": "method",
        "deployable": deployable,
        "line_style": line_style,
        "accuracy": correct / n if n else 0.0,
        "cost_multiplier": cost / direct_cost if direct_cost else 0.0,
        "cost_per_correct": cost / correct if correct else "",
        "escalation_rate": escalation_rate,
    }


def _direct_first_oracle_point(rows: list[dict[str, Any]], direct_cost: float) -> dict[str, Any]:
    correct = 0
    cost = 0.0
    escalations = 0
    available_strategies = available_collab_strategies(rows)
    for row in rows:
        cost += _to_float(row.get("direct_cost_usd"))
        if _to_bool(row.get("direct_correct")):
            correct += 1
            continue
        rescuers = [
            strategy_cost(row, strategy)
            for strategy in available_strategies
            if _strategy_correct(row, strategy)
        ]
        if rescuers:
            cost += min(rescuers)
            correct += 1
            escalations += 1
    n = len(rows)
    return {
        "policy": "direct_first_oracle_any",
        "point_type": "upper_bound",
        "deployable": False,
        "line_style": "dashed",
        "accuracy": correct / n if n else 0.0,
        "cost_multiplier": cost / direct_cost if direct_cost else 0.0,
        "cost_per_correct": cost / correct if correct else "",
        "escalation_rate": escalations / n if n else 0.0,
    }


def _posthoc_oracle_point(
    rows: list[dict[str, Any]],
    strategy: str,
    direct_cost: float,
) -> dict[str, Any]:
    correct = sum(
        1
        for row in rows
        if _to_bool(row.get("direct_correct"))
        or ((not _to_bool(row.get("direct_correct"))) and _strategy_correct(row, strategy))
    )
    cost = direct_cost + sum(strategy_cost(row, strategy) for row in rows)
    n = len(rows)
    return {
        "policy": f"posthoc_oracle_direct_plus_{strategy}",
        "point_type": "posthoc_reference",
        "deployable": False,
        "line_style": "dotted",
        "accuracy": correct / n if n else 0.0,
        "cost_multiplier": cost / direct_cost if direct_cost else 0.0,
        "cost_per_correct": cost / correct if correct else "",
        "escalation_rate": 1.0,
    }


def _router_prediction_point(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    policy: str,
) -> dict[str, Any]:
    row_by_key = {_row_key(row): row for row in rows}
    correct = 0
    cost = 0.0
    direct_cost = 0.0
    escalations = 0
    for prediction in predictions:
        row = row_by_key.get(_row_key(prediction), {})
        selected_strategy = str(prediction.get("selected_strategy") or "direct")
        direct_item_cost = _to_float(row.get("direct_cost_usd"))
        direct_cost += direct_item_cost
        if selected_strategy == "direct":
            item_cost = direct_item_cost
        else:
            item_cost = direct_item_cost + strategy_cost(row, selected_strategy)
            escalations += 1
        cost += item_cost
        if _to_bool(prediction.get("selected_correct")):
            correct += 1
    n = len(predictions)
    return {
        "policy": policy,
        "point_type": "method",
        "deployable": True,
        "line_style": "solid",
        "accuracy": correct / n if n else 0.0,
        "cost_multiplier": cost / direct_cost if direct_cost else 0.0,
        "cost_per_correct": cost / correct if correct else "",
        "escalation_rate": escalations / n if n else 0.0,
    }


def _routing_curve_point(row: dict[str, Any], policy: str) -> dict[str, Any]:
    return {
        "policy": policy,
        "point_type": "method",
        "deployable": True,
        "line_style": "solid",
        "accuracy": _to_float(row.get("accuracy")),
        "cost_multiplier": _to_float(row.get("cost_multiplier")),
        "cost_per_correct": row.get("cost_per_correct", ""),
        "escalation_rate": _to_float(row.get("escalation_rate")),
    }


def _render_report(
    rows: list[dict[str, Any]],
    fig1_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    confidence_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    feature_gap_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Mechanistic Analysis",
        "",
        f"Scope: {len(rows)} feature rows.",
        "",
        "## Paper Figure Data",
        "",
        "- Fig 1: `fig1_cost_accuracy_points.csv` for cost-accuracy motivation points.",
        "- Fig 2: `fig2_routing_curve.csv` for threshold-swept routing curves.",
        "- Fig 3: `fig3_rescue_harm_by_dataset.csv` for rescue/harm decomposition.",
        "- Fig 4: `fig4_router_fp_fn_breakdown.csv` for router FP/FN diagnosis.",
        "- Transfer-ready features: `model_independent_features.csv`.",
        "",
        "## Fig 1 Points",
        "",
        "| policy | type | accuracy | cost x | cost/correct |",
        "|---|---|---:|---:|---:|",
    ]
    for row in fig1_rows:
        lines.append(
            f"| `{row['policy']}` | {row['point_type']} | "
            f"{_to_float(row['accuracy']):.3f} | {_to_float(row['cost_multiplier']):.2f} | "
            f"{_fmt_usd(row.get('cost_per_correct'))} |"
        )
    lines.extend(
        [
            "",
            "## Rescue/Harm Snapshot",
            "",
            "| comparison | dataset | model | rescue n | harm n | rescue-harm pp |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in _rescue_harm_snapshot(case_rows):
        lines.append(
            f"| `{row['comparison']}` | {row['dataset']} | {row['model']} | "
            f"{row['rescue_n']} | {row['harm_n']} | {100 * row['net_rate']:+.1f} |"
        )
    lines.extend(
        [
            "",
            "## Confidence Snapshot",
            "",
            "| comparison | case | n | mean confidence | high-confidence rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in _top_confidence_rows(confidence_rows):
        lines.append(
            f"| `{row['comparison']}` | {row['case_type']} | {row['n']} | "
            f"{_to_float(row['confidence_mean']):.3f} | "
            f"{100 * _to_float(row['confidence_high_rate']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Router Diagnosis",
            "",
            "| dataset | model | type | n | rate | mean confidence | mean question words |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostic_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['diagnostic_type']} | {row['n']} | "
            f"{100 * _to_float(row['rate_within_dataset_model']):.1f}% | "
            f"{_to_float(row['mean_direct_confidence']):.3f} | "
            f"{_to_float(row['mean_question_word_count']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Missing Feature Diagnostics",
            "",
            "| contrast | feature | standardized diff | left mean | right mean |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in feature_gap_rows[:12]:
        lines.append(
            f"| {row['contrast']} | `{row['feature']}` | "
            f"{_to_float(row['standardized_diff']):+.2f} | "
            f"{_to_float(row['left_mean']):.3f} | {_to_float(row['right_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `direct_first_oracle_any` is a non-deployable upper bound: it assumes perfect knowledge of when collaboration will rescue a direct error.",
            "- `posthoc_oracle_direct_plus_debate_2_judge` is a post-hoc reference that charges both direct and debate observations for every item.",
            "- False negatives are the most important missing-feature pool: oracle could rescue them, but the router did not escalate.",
        ]
    )
    return "\n".join(lines) + "\n"


def _rescue_harm_snapshot(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in case_rows:
        if row["case_type"] not in {"rescue", "harm"}:
            continue
        key = (row["comparison"], row["dataset"], row["model"])
        grouped.setdefault(
            key,
            {
                "comparison": row["comparison"],
                "dataset": row["dataset"],
                "model": row["model"],
                "rescue_n": 0,
                "harm_n": 0,
                "net_rate": 0.0,
            },
        )
        grouped[key][f"{row['case_type']}_n"] += int(row["n"])
        sign = 1 if row["case_type"] == "rescue" else -1
        grouped[key]["net_rate"] += sign * _to_float(row["rate_within_dataset_model"])
    return sorted(grouped.values(), key=lambda row: (row["comparison"], row["dataset"], row["model"]))


def _top_confidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interesting = [
        row for row in rows
        if row.get("case_type") in {"rescue", "harm", "unresolved"}
    ]
    return sorted(
        interesting,
        key=lambda row: (
            str(row.get("comparison")),
            str(row.get("case_type")),
            -int(row.get("n", 0) or 0),
        ),
    )[:16]


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("model") or ""),
        str(row.get("question_id") or ""),
    )


def _strategy_correct(row: dict[str, Any], strategy: str) -> bool:
    if strategy == "direct":
        return _to_bool(row.get("direct_correct"))
    return _to_bool(row.get(f"{strategy_prefix(strategy)}_correct"))


def _oracle_any_rescuable(row: dict[str, Any]) -> bool:
    return any(
        _strategy_correct(row, strategy)
        for strategy in available_collab_strategies([row])
    )


def _mean_feature(rows: list[dict[str, Any]], feature: str) -> float:
    return _mean([_to_float(row.get(feature)) for row in rows])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _rate(values: list[float], predicate: Any) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if predicate(value)) / len(values)


def _pooled_std(left: list[float], right: list[float]) -> float:
    values = left + right
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return number


def _fmt_usd(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"${_to_float(value):.6f}"
