from __future__ import annotations

from collections import defaultdict
from typing import Any

from cups_eval.schema import StrategyResult


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_summary(
    rows: list[StrategyResult],
    baseline_strategy: str = "direct",
    oracle_accuracy: float | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[StrategyResult]] = defaultdict(list)
    for row in rows:
        grouped[row.strategy].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for strategy, items in grouped.items():
        n = len(items)
        correct = sum(1 for item in items if is_correct_result(item))
        total_tokens = sum(item.usage.total_tokens for item in items)
        api_calls = sum(int(item.metadata.get("api_calls", len(item.calls))) for item in items)
        cost = sum(float(item.metadata.get("estimated_cost_usd", 0.0)) for item in items)
        parse_ok = sum(1 for item in items if item.parse_ok)
        accuracy = correct / n if n else 0.0
        summary[strategy] = {
            "n": n,
            "correct": correct,
            "accuracy": accuracy,
            "parse_success_rate": parse_ok / n if n else 0.0,
            "total_tokens": total_tokens,
            "api_calls": api_calls,
            "estimated_cost_usd": cost,
            "tokens_per_correct": _safe_div(total_tokens, correct),
            "api_calls_per_correct": _safe_div(api_calls, correct),
            "cost_per_correct": _safe_div(cost, correct),
        }

    baseline = summary.get(baseline_strategy)
    for strategy, item in summary.items():
        if not baseline or strategy == baseline_strategy:
            item["collaboration_utility"] = None
        else:
            delta_accuracy = item["accuracy"] - baseline["accuracy"]
            delta_cost = item["estimated_cost_usd"] - baseline["estimated_cost_usd"]
            item["collaboration_utility"] = _safe_div(delta_accuracy, delta_cost)
        if oracle_accuracy is None:
            item["oracle_router_gap"] = None
        else:
            item["oracle_router_gap"] = oracle_accuracy - item["accuracy"]
    return summary


def is_correct_result(item: StrategyResult) -> bool:
    if "correct" in item.metadata:
        return bool(item.metadata["correct"])
    return item.answer == item.metadata.get("gold")


def pareto_frontier(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    points = sorted(
        (
            {
                "strategy": strategy,
                "accuracy": float(item["accuracy"]),
                "estimated_cost_usd": float(item["estimated_cost_usd"]),
            }
            for strategy, item in summaries.items()
        ),
        key=lambda row: (row["estimated_cost_usd"], -row["accuracy"], row["strategy"]),
    )
    frontier: list[dict[str, Any]] = []
    best_accuracy = -1.0
    for point in points:
        if point["accuracy"] > best_accuracy:
            frontier.append(point)
            best_accuracy = point["accuracy"]
    return frontier
