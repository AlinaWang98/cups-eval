from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cups_eval.offline_router import strategy_cost, strategy_prefix


BASELINE_STRATEGIES = ("direct", "cot", "planner_verifier", "debate_2_judge")
ESCALATION_STRATEGIES = ("direct", "planner_verifier", "debate_2_judge")
CONFIDENCE_THRESHOLDS = (
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)


def build_cot_first_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    direct_predictions = _strategy_predictions(rows, "direct")
    cot_predictions = _strategy_predictions(rows, "cot")
    oracle_predictions = _cot_first_oracle_predictions(rows)
    direct_metrics = _aggregate(direct_predictions)
    cot_metrics = _aggregate(cot_predictions)
    oracle_metrics = _aggregate(oracle_predictions)
    curve = [
        _curve_row(
            "direct",
            "baseline",
            "direct",
            None,
            direct_predictions,
            direct_metrics,
            cot_metrics,
            oracle_metrics,
        ),
        _curve_row(
            "always_cot",
            "baseline",
            "cot",
            None,
            cot_predictions,
            direct_metrics,
            cot_metrics,
            oracle_metrics,
        ),
        _curve_row(
            "cot_first_oracle_direct_planner_debate",
            "oracle",
            "mixed",
            None,
            oracle_predictions,
            direct_metrics,
            cot_metrics,
            oracle_metrics,
        ),
    ]
    for strategy in ("planner_verifier", "debate_2_judge"):
        if _strategy_available(rows, strategy):
            curve.append(
                _curve_row(
                    f"always_{strategy}",
                    "always_collaborate",
                    strategy,
                    None,
                    _strategy_predictions(rows, strategy),
                    direct_metrics,
                    cot_metrics,
                    oracle_metrics,
                )
            )
    for target_strategy in ESCALATION_STRATEGIES:
        if target_strategy != "direct" and not _strategy_available(rows, target_strategy):
            continue
        for threshold in CONFIDENCE_THRESHOLDS:
            curve.append(
                _curve_row(
                    f"cot_conf_lt_{threshold:.2f}_to_{target_strategy}",
                    "cot_confidence",
                    target_strategy,
                    threshold,
                    _cot_threshold_predictions(rows, target_strategy, threshold),
                    direct_metrics,
                    cot_metrics,
                    oracle_metrics,
                )
            )
    return curve


def evaluate_cot_first_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _aggregate(_cot_first_oracle_predictions(rows))


def write_cot_router_report(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = build_cot_first_curve(rows)
    pareto = _pareto_rows([row for row in curve if row["policy_family"] != "oracle"])
    main_table = _main_table_rows(curve)
    by_dataset_model = _dataset_model_rows(rows)
    opportunity_rows = _cot_oracle_opportunity_rows(by_dataset_model)
    _write_csv(output_dir / "cot_first_routing_curve.csv", curve)
    _write_csv(output_dir / "cot_first_pareto.csv", pareto)
    _write_csv(output_dir / "paper_main_table.csv", main_table)
    _write_csv(output_dir / "cot_first_by_dataset_model.csv", by_dataset_model)
    _write_csv(output_dir / "cot_oracle_opportunity_by_dataset_model.csv", opportunity_rows)
    summary = {
        "n": len(rows),
        "best_deployable_vs_cot": _best_deployable_vs_cot(curve),
        "main_table": main_table,
        "pareto": pareto,
        "cot_oracle_opportunity_by_dataset_model": opportunity_rows,
    }
    (output_dir / "phase2_6_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "phase2_6_report.md").write_text(
        _render_report(curve, pareto, main_table),
        encoding="utf-8",
    )
    (output_dir / "claim_memo.md").write_text(
        _render_claim_memo(curve, opportunity_rows),
        encoding="utf-8",
    )


def _strategy_predictions(rows: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        cost = _strategy_total_cost(row, strategy)
        output.append(
            _prediction(
                row,
                selected_strategy=strategy,
                correct=_strategy_correct(row, strategy),
                estimated_cost_usd=cost,
                escalated=False,
            )
        )
    return output


def _cot_threshold_predictions(
    rows: list[dict[str, Any]],
    target_strategy: str,
    threshold: float,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        cot_cost = _strategy_total_cost(row, "cot")
        if _to_float(row.get("cot_confidence")) < threshold:
            selected = target_strategy
            cost = cot_cost + _strategy_total_cost(row, target_strategy)
            correct = _strategy_correct(row, target_strategy)
            escalated = True
        else:
            selected = "cot"
            cost = cot_cost
            correct = _strategy_correct(row, "cot")
            escalated = False
        output.append(
            _prediction(
                row,
                selected_strategy=selected,
                correct=correct,
                estimated_cost_usd=cost,
                escalated=escalated,
            )
        )
    return output


def _cot_first_oracle_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        cot_cost = _strategy_total_cost(row, "cot")
        selected = "cot"
        correct = _strategy_correct(row, "cot")
        cost = cot_cost
        escalated = False
        if not correct:
            rescuers = [
                (_strategy_total_cost(row, strategy), strategy)
                for strategy in ESCALATION_STRATEGIES
                if _strategy_available([row], strategy) and _strategy_correct(row, strategy)
            ]
            if rescuers:
                extra_cost, selected = min(rescuers)
                cost = cot_cost + extra_cost
                correct = True
                escalated = True
        output.append(
            _prediction(
                row,
                selected_strategy=selected,
                correct=correct,
                estimated_cost_usd=cost,
                escalated=escalated,
            )
        )
    return output


def _prediction(
    row: dict[str, Any],
    *,
    selected_strategy: str,
    correct: bool,
    estimated_cost_usd: float,
    escalated: bool,
) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "model": row.get("model", ""),
        "question_id": row.get("question_id", ""),
        "selected_strategy": selected_strategy,
        "selected_correct": bool(correct),
        "estimated_cost_usd": estimated_cost_usd,
        "direct_cost_usd": _strategy_total_cost(row, "direct"),
        "cot_cost_usd": _strategy_total_cost(row, "cot"),
        "escalated": bool(escalated),
    }


def _aggregate(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(predictions)
    correct = sum(1 for row in predictions if _to_bool(row.get("selected_correct")))
    cost = sum(_to_float(row.get("estimated_cost_usd")) for row in predictions)
    direct_cost = sum(_to_float(row.get("direct_cost_usd")) for row in predictions)
    escalations = sum(1 for row in predictions if _to_bool(row.get("escalated")))
    accuracy = correct / n if n else 0.0
    cost_multiplier = cost / direct_cost if direct_cost else 0.0
    return {
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "estimated_cost_usd": cost,
        "direct_cost_usd": direct_cost,
        "cost_multiplier": cost_multiplier,
        "relative_cost_per_correct": cost_multiplier / accuracy if accuracy else None,
        "cost_per_correct": cost / correct if correct else None,
        "escalation_rate": escalations / n if n else 0.0,
    }


def _curve_row(
    policy: str,
    policy_family: str,
    target_strategy: str,
    threshold: float | None,
    predictions: list[dict[str, Any]],
    direct_metrics: dict[str, Any],
    cot_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
) -> dict[str, Any]:
    metrics = _aggregate(predictions)
    accuracy = float(metrics["accuracy"])
    direct_accuracy = float(direct_metrics["accuracy"])
    cot_accuracy = float(cot_metrics["accuracy"])
    oracle_accuracy = float(oracle_metrics["accuracy"])
    direct_cpc = direct_metrics.get("relative_cost_per_correct")
    cot_cpc = cot_metrics.get("relative_cost_per_correct")
    cpc = metrics.get("relative_cost_per_correct")
    return {
        "policy": policy,
        "policy_family": policy_family,
        "target_strategy": target_strategy,
        "threshold": "" if threshold is None else threshold,
        "n": metrics["n"],
        "accuracy": accuracy,
        "delta_vs_direct": accuracy - direct_accuracy,
        "delta_vs_cot": accuracy - cot_accuracy,
        "retained_direct_oracle_gain": _retained_gain(direct_accuracy, oracle_accuracy, accuracy),
        "retained_cot_oracle_gain": _retained_gain(cot_accuracy, oracle_accuracy, accuracy),
        "estimated_cost_usd": metrics["estimated_cost_usd"],
        "cost_multiplier": metrics["cost_multiplier"],
        "relative_cost_per_correct": cpc,
        "cpc_delta_vs_direct": (
            cpc - direct_cpc if cpc is not None and direct_cpc is not None else None
        ),
        "cpc_delta_vs_cot": cpc - cot_cpc if cpc is not None and cot_cpc is not None else None,
        "escalation_rate": metrics["escalation_rate"],
        "dominates_direct_cpc": _dominates(metrics, direct_metrics),
        "dominates_cot_cpc": _dominates(metrics, cot_metrics),
    }


def _main_table_rows(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "direct",
        "always_cot",
        "always_planner_verifier",
        "always_debate_2_judge",
        "cot_first_oracle_direct_planner_debate",
    }
    rows = [row for row in curve if row["policy"] in wanted]
    best = _best_deployable_vs_cot(curve)
    if best:
        best_row = dict(best)
        best_row["policy"] = f"best_deployable::{best_row['policy']}"
        rows.append(best_row)
    return rows


def _dataset_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset") or ""), str(row.get("model") or ""))
        grouped.setdefault(key, []).append(row)
    output = []
    for (dataset, model), items in sorted(grouped.items()):
        curve = build_cot_first_curve(items)
        for row in _main_table_rows(curve):
            item = dict(row)
            item["dataset"] = dataset
            item["model"] = model
            output.append(item)
    return output


def _cot_oracle_opportunity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dataset") or ""), str(row.get("model") or ""))
        grouped.setdefault(key, {})[str(row.get("policy") or "")] = row
    output = []
    for (dataset, model), items in sorted(grouped.items()):
        cot = items.get("always_cot")
        oracle = items.get("cot_first_oracle_direct_planner_debate")
        if not cot or not oracle:
            continue
        cot_accuracy = _to_float(cot.get("accuracy"))
        oracle_accuracy = _to_float(oracle.get("accuracy"))
        cot_cpc = _to_float(cot.get("relative_cost_per_correct"))
        oracle_cpc = _to_float(oracle.get("relative_cost_per_correct"))
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "cot_accuracy": cot_accuracy,
                "cot_relative_cpc": cot_cpc,
                "cot_first_oracle_accuracy": oracle_accuracy,
                "cot_first_oracle_relative_cpc": oracle_cpc,
                "oracle_delta_accuracy_vs_cot": oracle_accuracy - cot_accuracy,
                "oracle_delta_relative_cpc_vs_cot": oracle_cpc - cot_cpc,
                "oracle_improves_accuracy_and_cpc": (
                    oracle_accuracy > cot_accuracy and oracle_cpc < cot_cpc
                ),
            }
        )
    return output


def _best_deployable_vs_cot(curve: list[dict[str, Any]]) -> dict[str, Any] | None:
    deployable = [
        row for row in curve
        if row["policy_family"] not in {"oracle", "baseline", "always_collaborate"}
    ]
    if not deployable:
        return None
    cot = next((row for row in curve if row["policy"] == "always_cot"), None)
    if not cot:
        return None
    cot_cpc = _optional_float(cot.get("relative_cost_per_correct"))
    cot_acc = _to_float(cot.get("accuracy"))
    dominating = [
        row for row in deployable
        if _to_float(row.get("accuracy")) > cot_acc
        and _optional_float(row.get("relative_cost_per_correct")) is not None
        and cot_cpc is not None
        and float(row["relative_cost_per_correct"]) <= cot_cpc
    ]
    candidates = dominating or [
        row for row in deployable
        if _to_float(row.get("accuracy")) > cot_acc
    ] or deployable
    return max(
        candidates,
        key=lambda row: (
            bool(row in dominating),
            _to_float(row.get("accuracy")),
            -_to_float(row.get("relative_cost_per_correct")),
            -_to_float(row.get("cost_multiplier")),
        ),
    )


def _pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    best_accuracy = -1.0
    for row in sorted(
        rows,
        key=lambda item: (
            _to_float(item.get("relative_cost_per_correct")),
            -_to_float(item.get("accuracy")),
            str(item.get("policy")),
        ),
    ):
        accuracy = _to_float(row.get("accuracy"))
        if accuracy > best_accuracy + 1e-12:
            item = dict(row)
            item["pareto_rank"] = len(frontier) + 1
            frontier.append(item)
            best_accuracy = accuracy
    return frontier


def _render_report(
    curve: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    main_table: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 2.6 CoT-First Router Analysis",
        "",
        "No provider calls were made; this report is computed from existing saved predictions.",
        "",
        "## Main Table",
        "",
        "| policy | accuracy | cost x | relative cost/correct | delta vs CoT | escalation | ROG vs CoT oracle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_table:
        lines.append(_policy_line(row))
    lines.extend(
        [
            "",
            "## Pareto Points",
            "",
            "| rank | policy | accuracy | cost x | relative cost/correct | escalation |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in pareto[:20]:
        lines.append(
            "| {rank} | `{policy}` | {acc:.3f} | {cost:.2f} | {cpc} | {esc:.1%} |".format(
                rank=row["pareto_rank"],
                policy=row["policy"],
                acc=_to_float(row.get("accuracy")),
                cost=_to_float(row.get("cost_multiplier")),
                cpc=_fmt_optional(row.get("relative_cost_per_correct")),
                esc=_to_float(row.get("escalation_rate")),
            )
        )
    return "\n".join(lines) + "\n"


def _render_claim_memo(
    curve: list[dict[str, Any]],
    opportunity_rows: list[dict[str, Any]],
) -> str:
    by_policy = {row["policy"]: row for row in curve}
    direct = by_policy.get("direct", {})
    cot = by_policy.get("always_cot", {})
    oracle = by_policy.get("cot_first_oracle_direct_planner_debate", {})
    best = _best_deployable_vs_cot(curve) or {}
    dominates_cot = [
        row for row in curve
        if row.get("dominates_cot_cpc") is True
        and row.get("policy_family") not in {"oracle", "baseline", "always_collaborate"}
    ]
    lines = [
        "# Claim Memo: CoT-first Selective Collaboration",
        "",
        "## Bottom Line",
        "",
        (
            f"- Direct baseline: accuracy={_to_float(direct.get('accuracy')):.3f}, "
            f"relative CPC={_fmt_optional(direct.get('relative_cost_per_correct'))}."
        ),
        (
            f"- CoT baseline: accuracy={_to_float(cot.get('accuracy')):.3f}, "
            f"relative CPC={_fmt_optional(cot.get('relative_cost_per_correct'))}."
        ),
        (
            f"- CoT-first oracle: accuracy={_to_float(oracle.get('accuracy')):.3f}, "
            f"cost={_to_float(oracle.get('cost_multiplier')):.2f}x, "
            f"relative CPC={_fmt_optional(oracle.get('relative_cost_per_correct'))}."
        ),
        (
            f"- Best deployable CoT-first threshold point: `{best.get('policy', 'n/a')}`, "
            f"accuracy={_to_float(best.get('accuracy')):.3f}, "
            f"cost={_to_float(best.get('cost_multiplier')):.2f}x, "
            f"relative CPC={_fmt_optional(best.get('relative_cost_per_correct'))}."
        ),
        "",
        "## Interpretation",
        "",
    ]
    if dominates_cot:
        best_dom = max(dominates_cot, key=lambda row: _to_float(row.get("accuracy")))
        lines.append(
            f"- At least one deployable CoT-first threshold dominates always-CoT on cost-normalized accuracy: `{best_dom['policy']}`."
        )
    else:
        lines.append(
            "- No deployable CoT-first threshold dominates always-CoT on relative cost per correct. The opportunity exists mainly as an oracle gap, not yet as a usable router."
        )
    lines.extend(
        [
            "- This supports a CUPS framing around cost-fair evaluation and uncaptured collaboration opportunity, with router method development still an open problem.",
            _opportunity_sentence(opportunity_rows),
            "- The next API spend should be gated on whether this same CoT-vs-collaboration pattern needs non-medical confirmation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _opportunity_sentence(rows: list[dict[str, Any]]) -> str:
    winners = [
        row for row in rows
        if row.get("oracle_improves_accuracy_and_cpc") is True
    ]
    if not winners:
        return "- Dataset-level opportunity is accuracy-focused only: no dataset/model slice improves both accuracy and relative cost per correct over always-CoT."
    labels = ", ".join(
        f"{row['dataset']}/{row['model']}"
        for row in winners
    )
    return (
        "- Dataset-level opportunity is heterogeneous: CoT-first oracle improves both "
        f"accuracy and relative cost per correct on {labels}."
    )


def _policy_line(row: dict[str, Any]) -> str:
    return "| `{policy}` | {acc:.3f} | {cost:.2f} | {cpc} | {delta:+.3f} | {esc:.1%} | {rog} |".format(
        policy=row["policy"],
        acc=_to_float(row.get("accuracy")),
        cost=_to_float(row.get("cost_multiplier")),
        cpc=_fmt_optional(row.get("relative_cost_per_correct")),
        delta=_to_float(row.get("delta_vs_cot")),
        esc=_to_float(row.get("escalation_rate")),
        rog=_fmt_optional(row.get("retained_cot_oracle_gain")),
    )


def _strategy_total_cost(row: dict[str, Any], strategy: str) -> float:
    if strategy == "direct":
        return _to_float(row.get("direct_cost_usd"))
    if strategy == "cot":
        return _to_float(row.get("cot_cost_usd"))
    return strategy_cost(row, strategy)


def _strategy_correct(row: dict[str, Any], strategy: str) -> bool:
    if strategy == "direct":
        return _to_bool(row.get("direct_correct"))
    if strategy == "cot":
        return _to_bool(row.get("cot_correct"))
    return _to_bool(row.get(f"{strategy_prefix(strategy)}_correct"))


def _strategy_available(rows: list[dict[str, Any]], strategy: str) -> bool:
    if strategy in {"direct", "cot"}:
        return any(_strategy_total_cost(row, strategy) > 0 for row in rows)
    prefix = strategy_prefix(strategy)
    return any(
        _to_float(row.get(f"{prefix}_cost_usd")) > 0.0
        or str(row.get(f"{prefix}_answer") or "").strip() != ""
        for row in rows
    )


def _dominates(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    metric_cpc = metrics.get("relative_cost_per_correct")
    baseline_cpc = baseline.get("relative_cost_per_correct")
    if metric_cpc is None or baseline_cpc is None:
        return False
    return (
        float(metrics["accuracy"]) > float(baseline["accuracy"])
        and float(metric_cpc) <= float(baseline_cpc)
    )


def _retained_gain(baseline: float, oracle: float, candidate: float) -> float | None:
    denominator = oracle - baseline
    if denominator <= 0:
        return None
    return (candidate - baseline) / denominator


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_optional(value: Any) -> str:
    number = _optional_float(value)
    return "NA" if number is None else f"{number:.3f}"


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
