from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MAIN_METHODS = (
    "direct",
    "cot",
    "planner_verifier",
    "debate_2_judge",
    "direct_first_oracle_any",
    "learned_rule_list",
)

OBSERVED_METHODS = (
    "direct",
    "cot",
    "planner_verifier",
    "debate_2_judge",
)


@dataclass(frozen=True)
class PolicyRecord:
    scope: str
    family: str
    dataset: str
    model: str
    question_id: str
    method: str
    correct: bool
    cost_usd: float

    @property
    def item_key(self) -> tuple[str, str, str]:
        return (self.dataset, self.model, self.question_id)


def load_policy_records(
    *,
    batch_dir: Path | str,
    scope: str,
    family: str,
    offline_predictions_path: Path | str | None = None,
    dataset_filter: str | None = None,
) -> list[PolicyRecord]:
    root = Path(batch_dir)
    records: dict[tuple[str, str, str, str], PolicyRecord] = {}
    for path in sorted(root.glob("runs/*/*/*/item_metrics.csv")):
        dataset = path.parts[-4]
        model = path.parts[-3]
        if dataset_filter and dataset != dataset_filter:
            continue
        for row in _read_csv(path):
            method = str(row.get("strategy") or "")
            if method not in OBSERVED_METHODS:
                continue
            question_id = str(row.get("question_id") or "")
            key = (dataset, model, question_id, method)
            records[key] = PolicyRecord(
                scope=scope,
                family=family,
                dataset=dataset,
                model=model,
                question_id=question_id,
                method=method,
                correct=_to_bool(row.get("correct")),
                cost_usd=_to_float(row.get("estimated_cost_usd")),
            )
    direct_cost_by_key = {
        (record.dataset, record.model, record.question_id): record.cost_usd
        for record in records.values()
        if record.method == "direct"
    }
    for path in sorted(root.glob("runs/*/*/*/router_features.csv")):
        dataset = path.parts[-4]
        model = path.parts[-3]
        if dataset_filter and dataset != dataset_filter:
            continue
        for row in _read_csv(path):
            question_id = str(row.get("question_id") or "")
            if not question_id or "oracle_correct" not in row:
                continue
            direct_cost = direct_cost_by_key.get((dataset, model, question_id), 0.0)
            oracle_cost = _direct_first_oracle_cost(row, direct_cost)
            key = (dataset, model, question_id, "direct_first_oracle_any")
            records[key] = PolicyRecord(
                scope=scope,
                family=family,
                dataset=dataset,
                model=model,
                question_id=question_id,
                method="direct_first_oracle_any",
                correct=_direct_first_oracle_correct(row),
                cost_usd=oracle_cost,
            )
    if offline_predictions_path is not None:
        path = Path(offline_predictions_path)
        if path.exists():
            for row in _read_csv(path):
                dataset = str(row.get("dataset") or "")
                if dataset_filter and dataset != dataset_filter:
                    continue
                model = str(row.get("model") or "")
                question_id = str(row.get("question_id") or "")
                key = (dataset, model, question_id, "learned_rule_list")
                records[key] = PolicyRecord(
                    scope=scope,
                    family=family,
                    dataset=dataset,
                    model=model,
                    question_id=question_id,
                    method="learned_rule_list",
                    correct=_to_bool(row.get("selected_correct")),
                    cost_usd=_to_float(row.get("estimated_cost_usd")),
                )
    return sorted(records.values(), key=lambda item: (item.scope, item.dataset, item.model, item.question_id, item.method))


def aggregate_policy_rows(records: list[PolicyRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scope, family), group in _group_by(records, lambda item: (item.scope, item.family)).items():
        by_method = _records_by_method(group)
        item_keys = sorted({record.item_key for record in by_method.get("direct", [])})
        direct_cost = sum(record.cost_usd for record in by_method.get("direct", []))
        for method in MAIN_METHODS:
            method_records = by_method.get(method, [])
            if not method_records:
                continue
            rows.append(
                {
                    "scope": scope,
                    "family": family,
                    "method": method,
                    **_metric_payload(method_records, direct_cost, len(item_keys)),
                }
            )
    return _sort_rows(rows)


def bootstrap_policy_rows(
    records: list[PolicyRecord],
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    actual_rows = aggregate_policy_rows(records)
    intervals: dict[tuple[str, str, str], dict[str, tuple[float, float, float]]] = {}
    rng = random.Random(seed)
    for (scope, family), group in _group_by(records, lambda item: (item.scope, item.family)).items():
        by_method_key = {(record.method, record.item_key): record for record in group}
        item_keys = sorted({record.item_key for record in group if record.method == "direct"})
        if not item_keys:
            continue
        samples: dict[str, dict[str, list[float]]] = {
            method: {"accuracy": [], "cost_x_direct": [], "relative_cpc": []}
            for method in MAIN_METHODS
        }
        for _ in range(max(1, iterations)):
            sampled_keys = [rng.choice(item_keys) for _ in item_keys]
            direct_records = [
                by_method_key[("direct", key)]
                for key in sampled_keys
                if ("direct", key) in by_method_key
            ]
            direct_cost = sum(record.cost_usd for record in direct_records)
            for method in MAIN_METHODS:
                method_records = [
                    by_method_key[(method, key)]
                    for key in sampled_keys
                    if (method, key) in by_method_key
                ]
                if not method_records:
                    continue
                metrics = _metric_payload(method_records, direct_cost, len(sampled_keys))
                samples[method]["accuracy"].append(float(metrics["accuracy"]))
                samples[method]["cost_x_direct"].append(float(metrics["cost_x_direct"]))
                samples[method]["relative_cpc"].append(float(metrics["relative_cost_per_correct"]))
        for method, metric_values in samples.items():
            if not metric_values["accuracy"]:
                continue
            intervals[(scope, family, method)] = {
                metric: _ci(values)
                for metric, values in metric_values.items()
            }
    rows: list[dict[str, Any]] = []
    for row in actual_rows:
        key = (str(row["scope"]), str(row["family"]), str(row["method"]))
        interval = intervals.get(key, {})
        enriched = dict(row)
        for metric, prefix in (
            ("accuracy", "accuracy"),
            ("cost_x_direct", "cost_x_direct"),
            ("relative_cpc", "relative_cpc"),
        ):
            low, median, high = interval.get(metric, (math.nan, math.nan, math.nan))
            enriched[f"{prefix}_ci_low"] = low
            enriched[f"{prefix}_ci_median"] = median
            enriched[f"{prefix}_ci_high"] = high
        rows.append(enriched)
    return _sort_rows(rows)


def write_cross_domain_ci_report(
    *,
    specs: list[dict[str, Any]],
    output_dir: Path | str,
    iterations: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    records: list[PolicyRecord] = []
    for spec in specs:
        records.extend(
            load_policy_records(
                batch_dir=Path(str(spec["batch_dir"])),
                scope=str(spec["scope"]),
                family=str(spec.get("family") or spec["scope"]),
                offline_predictions_path=(
                    Path(str(spec["offline_predictions_path"]))
                    if spec.get("offline_predictions_path")
                    else None
                ),
                dataset_filter=(
                    str(spec["dataset_filter"])
                    if spec.get("dataset_filter")
                    else None
                ),
            )
        )
    rows = bootstrap_policy_rows(records, iterations=iterations, seed=seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "cross_domain_ci_table.csv", rows)
    (out / "cross_domain_ci_table.md").write_text(
        _render_markdown(rows, iterations=iterations, seed=seed),
        encoding="utf-8",
    )
    return rows


def _metric_payload(
    records: list[PolicyRecord],
    direct_cost: float,
    n: int,
) -> dict[str, Any]:
    correct = sum(1 for record in records if record.correct)
    total_cost = sum(record.cost_usd for record in records)
    accuracy = correct / n if n else 0.0
    cost_x_direct = total_cost / direct_cost if direct_cost else 0.0
    relative_cpc = cost_x_direct / accuracy if accuracy else math.inf
    return {
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "estimated_cost_usd": total_cost,
        "cost_x_direct": cost_x_direct,
        "relative_cost_per_correct": relative_cpc,
        "cost_per_correct_usd": total_cost / correct if correct else math.inf,
    }


def _direct_first_oracle_correct(row: dict[str, Any]) -> bool:
    if _to_bool(row.get("direct_correct")):
        return True
    return any(
        _to_bool(row.get(f"{strategy}_correct"))
        for strategy in ("cot", "self_consistency_3", "planner_verifier", "debate_2_judge")
    )


def _direct_first_oracle_cost(row: dict[str, Any], direct_cost: float) -> float:
    if _to_bool(row.get("direct_correct")):
        return direct_cost or _to_float(row.get("direct_cost_usd"))
    rescuers = [
        _to_float(row.get(f"{strategy}_cost_usd"))
        for strategy in ("cot", "self_consistency_3", "planner_verifier", "debate_2_judge")
        if _to_bool(row.get(f"{strategy}_correct"))
    ]
    if rescuers:
        return (direct_cost or _to_float(row.get("direct_cost_usd"))) + min(rescuers)
    return direct_cost or _to_float(row.get("direct_cost_usd"))


def _render_markdown(rows: list[dict[str, Any]], *, iterations: int, seed: int) -> str:
    lines = [
        "# Cross-Domain Bootstrap CI Table",
        "",
        "Generated from completed local runs only. No provider calls were made.",
        "",
        f"Bootstrap iterations: {iterations}; seed: {seed}.",
        "",
        "| Scope | Method | Accuracy [95% CI] | Cost x Direct [95% CI] | Relative CPC [95% CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scope} | `{method}` | {acc:.3f} [{acc_l:.3f}, {acc_h:.3f}] | "
            "{cost:.2f} [{cost_l:.2f}, {cost_h:.2f}] | "
            "{cpc:.3f} [{cpc_l:.3f}, {cpc_h:.3f}] |".format(
                scope=row["scope"],
                method=row["method"],
                acc=_finite(row["accuracy"]),
                acc_l=_finite(row["accuracy_ci_low"]),
                acc_h=_finite(row["accuracy_ci_high"]),
                cost=_finite(row["cost_x_direct"]),
                cost_l=_finite(row["cost_x_direct_ci_low"]),
                cost_h=_finite(row["cost_x_direct_ci_high"]),
                cpc=_finite(row["relative_cost_per_correct"]),
                cpc_l=_finite(row["relative_cpc_ci_low"]),
                cpc_h=_finite(row["relative_cpc_ci_high"]),
            )
        )
    return "\n".join(lines) + "\n"


def _records_by_method(records: list[PolicyRecord]) -> dict[str, list[PolicyRecord]]:
    by_method: dict[str, list[PolicyRecord]] = {}
    for record in records:
        by_method.setdefault(record.method, []).append(record)
    return by_method


def _group_by(
    records: Iterable[PolicyRecord],
    key_fn: Any,
) -> dict[Any, list[PolicyRecord]]:
    groups: dict[Any, list[PolicyRecord]] = {}
    for record in records:
        groups.setdefault(key_fn(record), []).append(record)
    return groups


def _ci(values: list[float]) -> tuple[float, float, float]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return (math.nan, math.nan, math.nan)
    return (_percentile(finite, 0.025), _percentile(finite, 0.5), _percentile(finite, 0.975))


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    method_rank = {method: index for index, method in enumerate(MAIN_METHODS)}
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("scope") or ""),
            method_rank.get(str(row.get("method") or ""), 999),
            str(row.get("method") or ""),
        ),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _finite(value: Any) -> float:
    number = _to_float(value)
    return number if math.isfinite(number) else math.nan
