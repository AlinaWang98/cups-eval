from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


COLLAB_STRATEGIES = (
    "cot",
    "self_consistency_3",
    "planner_verifier",
    "debate_2_judge",
)

NUMERIC_FEATURES = (
    "question_word_count",
    "question_char_count",
    "option_mean_word_count",
    "option_max_char_count",
    "option_min_char_count",
    "option_max_similarity",
    "question_option_max_overlap",
    "question_option_mean_overlap",
    "direct_confidence",
    "direct_rationale_word_count",
)

BOOLEAN_FEATURES = ("has_negation_or_except",)

CATEGORICAL_FEATURES = (
    "dataset",
    "model",
    "direct_answer",
    "subject_name",
    "topic_name",
)


@dataclass(frozen=True)
class Predicate:
    label: str
    feature: str
    op: str
    value: str | float | bool

    def matches(self, row: dict[str, Any]) -> bool:
        if self.op == "<=":
            return _to_float(row.get(self.feature)) <= float(self.value)
        if self.op == ">":
            return _to_float(row.get(self.feature)) > float(self.value)
        if self.op == "==":
            return str(row.get(self.feature) or "") == str(self.value)
        if self.op == "is":
            return _to_bool(row.get(self.feature)) is bool(self.value)
        raise ValueError(f"Unknown predicate op: {self.op}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "feature": self.feature,
            "op": self.op,
            "value": self.value,
        }


@dataclass(frozen=True)
class RouterRule:
    predicate: Predicate
    target_strategy: str

    def as_dict(self) -> dict[str, Any]:
        payload = self.predicate.as_dict()
        payload["target_strategy"] = self.target_strategy
        return payload


@dataclass(frozen=True)
class FeatureSpec:
    numeric_means: dict[str, float]
    numeric_scales: dict[str, float]
    categories: dict[str, list[str]]


@dataclass(frozen=True)
class LogisticRescueModel:
    feature_spec: FeatureSpec
    weights: list[float]
    platt_a: float
    platt_b: float

    def score(self, row: dict[str, Any]) -> float:
        return _dot(self.weights, _feature_vector(row, self.feature_spec))

    def predict_probability(self, row: dict[str, Any]) -> float:
        return _sigmoid(self.platt_a * self.score(row) + self.platt_b)


def load_feature_rows(batch_dir: Path, model_prefix: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(batch_dir.glob("runs/*/*/*/router_features.csv")):
        dataset = path.parts[-4]
        model = path.parts[-3]
        if model_prefix and not model.startswith(model_prefix):
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["dataset"] = dataset
                row["model"] = model
                row["run_id"] = path.parts[-2]
                rows.append(_normalize_feature_row(row))
    return rows


def cross_validate_router(
    rows: list[dict[str, Any]],
    *,
    folds: int = 5,
    budget_multiplier: float = 1.5,
    max_rules: int = 4,
    min_support: int = 25,
) -> dict[str, Any]:
    if not rows:
        empty_metrics = evaluate_policy([], [])
        return {
            "folds": [],
            "aggregate": empty_metrics,
            "rules": [],
            "predictions": [],
            "routing_curve": [],
            "routing_pareto": [],
        }
    fold_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [row for row in rows if _fold_id(row, folds) != fold]
        test = [row for row in rows if _fold_id(row, folds) == fold]
        rules = train_greedy_router(
            train,
            budget_multiplier=budget_multiplier,
            max_rules=max_rules,
            min_support=min_support,
        )
        train_metrics = evaluate_policy(train, rules)
        test_metrics = evaluate_policy(test, rules)
        direct_metrics = evaluate_policy(test, [])
        oracle_metrics = evaluate_direct_first_oracle(test)
        fold_rows.append(
            {
                "fold": fold,
                "train_n": len(train),
                "test_n": len(test),
                "direct_accuracy": direct_metrics["accuracy"],
                "learned_accuracy": test_metrics["accuracy"],
                "oracle_after_direct_accuracy": oracle_metrics["accuracy"],
                "learned_gain": test_metrics["accuracy"] - direct_metrics["accuracy"],
                "oracle_gain": oracle_metrics["accuracy"] - direct_metrics["accuracy"],
                "learned_cost_multiplier": test_metrics["cost_multiplier"],
                "oracle_after_direct_cost_multiplier": oracle_metrics["cost_multiplier"],
                "learned_escalation_rate": test_metrics["escalation_rate"],
                "oracle_after_direct_escalation_rate": oracle_metrics["escalation_rate"],
                "retained_oracle_gain": _retained_gain(
                    direct_metrics["accuracy"],
                    oracle_metrics["accuracy"],
                    test_metrics["accuracy"],
                ),
                "train_accuracy": train_metrics["accuracy"],
                "train_cost_multiplier": train_metrics["cost_multiplier"],
            }
        )
        for index, rule in enumerate(rules):
            rule_rows.append(
                {
                    "fold": fold,
                    "position": index + 1,
                    **rule.as_dict(),
                }
            )
        predictions.extend(_prediction_rows(test, rules, fold))
    routing_curve = build_routing_curve(
        rows,
        learned_predictions=predictions,
        folds=folds,
    )
    return {
        "folds": fold_rows,
        "aggregate": _aggregate_predictions(predictions),
        "rules": rule_rows,
        "predictions": predictions,
        "routing_curve": routing_curve,
        "routing_pareto": pareto_curve_rows(routing_curve),
    }


def train_greedy_router(
    rows: list[dict[str, Any]],
    *,
    budget_multiplier: float = 1.5,
    max_rules: int = 4,
    min_support: int = 25,
) -> list[RouterRule]:
    rules: list[RouterRule] = []
    available_strategies = available_collab_strategies(rows)
    predicates = _candidate_predicates(rows, min_support=min_support)
    current = evaluate_policy(rows, rules)
    for _ in range(max_rules):
        best: tuple[tuple[float, float, int, str, str], RouterRule, dict[str, Any]] | None = None
        for predicate in predicates:
            support = sum(1 for row in rows if predicate.matches(row))
            if support < min_support:
                continue
            for strategy in available_strategies:
                candidate = RouterRule(predicate, strategy)
                metrics = evaluate_policy(rows, rules + [candidate])
                if metrics["cost_multiplier"] > budget_multiplier + 1e-12:
                    continue
                correct_gain = metrics["correct"] - current["correct"]
                if correct_gain <= 0:
                    continue
                extra_cost = metrics["estimated_cost_usd"] - current["estimated_cost_usd"]
                score = (
                    float(metrics["correct"]),
                    -float(extra_cost),
                    -support,
                    predicate.label,
                    strategy,
                )
                if best is None or score > best[0]:
                    best = (score, candidate, metrics)
        if best is None:
            break
        rules.append(best[1])
        current = best[2]
    return rules


def evaluate_policy(
    rows: list[dict[str, Any]],
    rules: list[RouterRule],
) -> dict[str, Any]:
    predictions = _prediction_rows(rows, rules)
    return _aggregate_predictions(predictions)


def evaluate_direct_first_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _aggregate_predictions(_direct_first_oracle_predictions(rows))


def build_routing_curve(
    rows: list[dict[str, Any]],
    *,
    learned_predictions: list[dict[str, Any]] | None = None,
    folds: int = 5,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    available_strategies = available_collab_strategies(rows)
    direct_predictions = _strategy_policy_predictions(rows, "direct")
    oracle_predictions = _direct_first_oracle_predictions(rows)
    direct_metrics = _aggregate_predictions(direct_predictions)
    oracle_metrics = _aggregate_predictions(oracle_predictions)
    curve = [
        _curve_row(
            policy="direct",
            policy_family="baseline",
            target_strategy="direct",
            threshold=None,
            predictions=direct_predictions,
            direct_metrics=direct_metrics,
            oracle_metrics=oracle_metrics,
        ),
        _curve_row(
            policy="direct_first_oracle",
            policy_family="oracle",
            target_strategy="oracle",
            threshold=None,
            predictions=oracle_predictions,
            direct_metrics=direct_metrics,
            oracle_metrics=oracle_metrics,
        ),
    ]
    for strategy in available_strategies:
        curve.append(
            _curve_row(
                policy=f"always_{strategy}",
                policy_family="always_collaborate",
                target_strategy=strategy,
                threshold=None,
                predictions=_strategy_policy_predictions(rows, strategy),
                direct_metrics=direct_metrics,
                oracle_metrics=oracle_metrics,
            )
        )
    if learned_predictions is not None:
        curve.append(
            _curve_row(
                policy="learned_rule_list",
                policy_family="rule_list",
                target_strategy="mixed",
                threshold=None,
                predictions=learned_predictions,
                direct_metrics=direct_metrics,
                oracle_metrics=oracle_metrics,
            )
        )
    confidence_thresholds = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    for strategy in available_strategies:
        for threshold in confidence_thresholds:
            curve.append(
                _curve_row(
                    policy=f"confidence_lt_{threshold:.2f}_to_{strategy}",
                    policy_family="confidence",
                    target_strategy=strategy,
                    threshold=threshold,
                    predictions=_threshold_router_predictions(
                        rows,
                        strategy,
                        scores=[
                            1.0 - _to_float(row.get("direct_confidence"))
                            for row in rows
                        ],
                        threshold=1.0 - threshold,
                    ),
                    direct_metrics=direct_metrics,
                    oracle_metrics=oracle_metrics,
                )
            )
    logistic_scores = _cross_validated_rescue_scores(
        rows,
        folds=folds,
        strategies=available_strategies,
    )
    probability_thresholds = (
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    )
    for strategy in available_strategies:
        scores = logistic_scores.get(strategy, [0.0 for _ in rows])
        for threshold in probability_thresholds:
            curve.append(
                _curve_row(
                    policy=f"calibrated_p_ge_{threshold:.2f}_to_{strategy}",
                    policy_family="calibrated_logistic",
                    target_strategy=strategy,
                    threshold=threshold,
                    predictions=_threshold_router_predictions(
                        rows,
                        strategy,
                        scores=scores,
                        threshold=threshold,
                    ),
                    direct_metrics=direct_metrics,
                    oracle_metrics=oracle_metrics,
                )
            )
    return curve


def pareto_curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if row.get("policy_family") != "oracle"
    ]
    frontier: list[dict[str, Any]] = []
    best_accuracy = -1.0
    for row in sorted(
        candidates,
        key=lambda item: (
            float(item["cost_multiplier"]),
            -float(item["accuracy"]),
            str(item["policy"]),
        ),
    ):
        accuracy = float(row["accuracy"])
        if accuracy > best_accuracy + 1e-12:
            item = dict(row)
            item["pareto_rank"] = len(frontier) + 1
            frontier.append(item)
            best_accuracy = accuracy
    return frontier


def _direct_first_oracle_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    available_strategies = available_collab_strategies(rows)
    for row in rows:
        selected = "direct"
        correct = _to_bool(row.get("direct_correct"))
        extra_cost = 0.0
        if not correct:
            rescuers = [
                (strategy_cost(row, strategy), strategy)
                for strategy in available_strategies
                if _to_bool(row.get(f"{strategy_prefix(strategy)}_correct"))
            ]
            if rescuers:
                extra_cost, selected = min(rescuers)
                correct = True
        predictions.append(
            _prediction_row(
                row=row,
                fold=None,
                selected_strategy=selected,
                correct=correct,
                extra_cost_usd=extra_cost,
            )
        )
    return predictions


def write_offline_router_report(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    output_dir: Path,
    *,
    budget_multiplier: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "offline_router_cv.csv", result["folds"])
    _write_csv(output_dir / "offline_router_predictions.csv", result["predictions"])
    _write_csv(output_dir / "offline_router_rules.csv", result["rules"])
    _write_csv(output_dir / "routing_curve.csv", result.get("routing_curve", []))
    _write_csv(output_dir / "routing_pareto.csv", result.get("routing_pareto", []))
    (output_dir / "offline_router_summary.json").write_text(
        json.dumps(
            {
                "n": len(rows),
                "budget_multiplier": budget_multiplier,
                "aggregate": result["aggregate"],
                "folds": result["folds"],
                "rules": result["rules"],
                "routing_curve": result.get("routing_curve", []),
                "routing_pareto": result.get("routing_pareto", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "offline_router_report.md").write_text(
        _render_report(rows, result, budget_multiplier),
        encoding="utf-8",
    )


def strategy_prefix(strategy: str) -> str:
    if strategy == "self_consistency_3":
        return "self_consistency"
    return strategy


def strategy_cost(row: dict[str, Any], strategy: str) -> float:
    return _to_float(row.get(f"{strategy_prefix(strategy)}_cost_usd"))


def available_collab_strategies(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        strategy
        for strategy in COLLAB_STRATEGIES
        if any(_strategy_observed(row, strategy) for row in rows)
    )


def _strategy_observed(row: dict[str, Any], strategy: str) -> bool:
    prefix = strategy_prefix(strategy)
    return (
        _to_float(row.get(f"{prefix}_cost_usd")) > 0.0
        or str(row.get(f"{prefix}_answer") or "").strip() != ""
    )


def _prediction_rows(
    rows: list[dict[str, Any]],
    rules: list[RouterRule],
    fold: int | None = None,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        selected = "direct"
        matched_rule = ""
        extra_cost = 0.0
        for rule in rules:
            if rule.predicate.matches(row):
                selected = rule.target_strategy
                matched_rule = rule.predicate.label
                extra_cost = strategy_cost(row, selected)
                break
        correct = _strategy_correct(row, selected)
        output.append(
            _prediction_row(
                row=row,
                fold=fold,
                selected_strategy=selected,
                correct=correct,
                extra_cost_usd=extra_cost,
                matched_rule=matched_rule,
            )
        )
    return output


def _strategy_policy_predictions(
    rows: list[dict[str, Any]],
    strategy: str,
) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        direct_cost = _to_float(row.get("direct_cost_usd"))
        if strategy == "direct":
            estimated_cost = direct_cost
        else:
            estimated_cost = strategy_cost(row, strategy)
        predictions.append(
            {
                "fold": "",
                "dataset": row.get("dataset", ""),
                "model": row.get("model", ""),
                "question_id": row.get("question_id", ""),
                "gold": row.get("gold", ""),
                "direct_answer": row.get("direct_answer", ""),
                "selected_strategy": strategy,
                "matched_rule": "",
                "direct_correct": _to_bool(row.get("direct_correct")),
                "selected_correct": _strategy_correct(row, strategy),
                "direct_cost_usd": direct_cost,
                "extra_cost_usd": 0.0 if strategy == "direct" else estimated_cost,
                "estimated_cost_usd": estimated_cost,
                "oracle_after_direct_correct": _oracle_after_direct_correct(row),
            }
        )
    return predictions


def _threshold_router_predictions(
    rows: list[dict[str, Any]],
    target_strategy: str,
    *,
    scores: list[float],
    threshold: float,
) -> list[dict[str, Any]]:
    predictions = []
    for row, score in zip(rows, scores):
        selected = target_strategy if score >= threshold else "direct"
        direct_cost = _to_float(row.get("direct_cost_usd"))
        if selected == "direct":
            estimated_cost = direct_cost
            extra_cost = 0.0
        else:
            extra_cost = strategy_cost(row, selected)
            estimated_cost = direct_cost + extra_cost
        predictions.append(
            {
                "fold": "",
                "dataset": row.get("dataset", ""),
                "model": row.get("model", ""),
                "question_id": row.get("question_id", ""),
                "gold": row.get("gold", ""),
                "direct_answer": row.get("direct_answer", ""),
                "selected_strategy": selected,
                "matched_rule": f"score>={threshold:.3f}",
                "direct_correct": _to_bool(row.get("direct_correct")),
                "selected_correct": _strategy_correct(row, selected),
                "direct_cost_usd": direct_cost,
                "extra_cost_usd": extra_cost,
                "estimated_cost_usd": estimated_cost,
                "oracle_after_direct_correct": _oracle_after_direct_correct(row),
            }
        )
    return predictions


def _prediction_row(
    *,
    row: dict[str, Any],
    fold: int | None,
    selected_strategy: str,
    correct: bool,
    extra_cost_usd: float,
    matched_rule: str = "",
) -> dict[str, Any]:
    direct_cost = _to_float(row.get("direct_cost_usd"))
    return {
        "fold": "" if fold is None else fold,
        "dataset": row.get("dataset", ""),
        "model": row.get("model", ""),
        "question_id": row.get("question_id", ""),
        "gold": row.get("gold", ""),
        "direct_answer": row.get("direct_answer", ""),
        "selected_strategy": selected_strategy,
        "matched_rule": matched_rule,
        "direct_correct": _to_bool(row.get("direct_correct")),
        "selected_correct": bool(correct),
        "direct_cost_usd": direct_cost,
        "extra_cost_usd": float(extra_cost_usd),
        "estimated_cost_usd": direct_cost + float(extra_cost_usd),
        "oracle_after_direct_correct": _oracle_after_direct_correct(row),
    }


def _aggregate_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(predictions)
    if n == 0:
        return {
            "n": 0,
            "correct": 0,
            "accuracy": 0.0,
            "estimated_cost_usd": 0.0,
            "direct_cost_usd": 0.0,
            "cost_multiplier": 0.0,
            "escalation_rate": 0.0,
        }
    correct = sum(1 for row in predictions if _to_bool(row["selected_correct"]))
    cost = sum(_to_float(row["estimated_cost_usd"]) for row in predictions)
    direct_cost = sum(_to_float(row["direct_cost_usd"]) for row in predictions)
    escalations = sum(1 for row in predictions if row["selected_strategy"] != "direct")
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "estimated_cost_usd": cost,
        "direct_cost_usd": direct_cost,
        "cost_multiplier": cost / direct_cost if direct_cost else 0.0,
        "escalation_rate": escalations / n,
        "cost_per_correct": cost / correct if correct else None,
    }


def _curve_row(
    *,
    policy: str,
    policy_family: str,
    target_strategy: str,
    threshold: float | None,
    predictions: list[dict[str, Any]],
    direct_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
) -> dict[str, Any]:
    metrics = _aggregate_predictions(predictions)
    direct_accuracy = float(direct_metrics["accuracy"])
    oracle_accuracy = float(oracle_metrics["accuracy"])
    direct_cost_multiplier = float(direct_metrics["cost_multiplier"])
    oracle_cost_multiplier = float(oracle_metrics["cost_multiplier"])
    retained = _retained_gain(
        direct_accuracy,
        oracle_accuracy,
        float(metrics["accuracy"]),
    )
    item_multipliers = [
        _to_float(row["estimated_cost_usd"]) / _to_float(row["direct_cost_usd"])
        for row in predictions
        if _to_float(row["direct_cost_usd"]) > 0
    ]
    return {
        "policy": policy,
        "policy_family": policy_family,
        "target_strategy": target_strategy,
        "threshold": "" if threshold is None else threshold,
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "delta_vs_direct": float(metrics["accuracy"]) - direct_accuracy,
        "oracle_gap_remaining": oracle_accuracy - float(metrics["accuracy"]),
        "retained_oracle_gain": retained,
        "estimated_cost_usd": metrics["estimated_cost_usd"],
        "cost_multiplier": metrics["cost_multiplier"],
        "cost_overhead_vs_oracle": (
            float(metrics["cost_multiplier"]) / oracle_cost_multiplier
            if oracle_cost_multiplier
            else None
        ),
        "extra_cost_vs_oracle_extra_cost": _safe_div(
            float(metrics["cost_multiplier"]) - direct_cost_multiplier,
            oracle_cost_multiplier - direct_cost_multiplier,
        ),
        "escalation_rate": metrics["escalation_rate"],
        "cost_per_correct": metrics.get("cost_per_correct"),
        "item_cost_multiplier_p50": _percentile(item_multipliers, 0.50),
        "item_cost_multiplier_p90": _percentile(item_multipliers, 0.90),
        "item_cost_multiplier_max": max(item_multipliers) if item_multipliers else 0.0,
    }


def _candidate_predicates(rows: list[dict[str, Any]], min_support: int) -> list[Predicate]:
    predicates: list[Predicate] = []
    for feature in NUMERIC_FEATURES:
        values = sorted(_to_float(row.get(feature)) for row in rows)
        thresholds = _quantiles(values, (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95))
        for threshold in thresholds:
            predicates.append(
                Predicate(f"{feature}<={threshold:.4g}", feature, "<=", threshold)
            )
            predicates.append(
                Predicate(f"{feature}>{threshold:.4g}", feature, ">", threshold)
            )
    for feature in BOOLEAN_FEATURES:
        predicates.append(Predicate(f"{feature}=true", feature, "is", True))
        predicates.append(Predicate(f"{feature}=false", feature, "is", False))
    for feature in CATEGORICAL_FEATURES:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(feature) or "")
            counts[value] = counts.get(value, 0) + 1
        for value, count in sorted(counts.items()):
            if value and count >= min_support:
                predicates.append(Predicate(f"{feature}=={value}", feature, "==", value))
    return predicates


def _cross_validated_rescue_scores(
    rows: list[dict[str, Any]],
    *,
    folds: int,
    strategies: tuple[str, ...] | None = None,
) -> dict[str, list[float]]:
    selected_strategies = strategies or available_collab_strategies(rows)
    output = {
        strategy: [0.0 for _ in rows]
        for strategy in selected_strategies
    }
    if not rows:
        return output
    effective_folds = max(2, min(folds, len(rows)))
    indexed_rows = list(enumerate(rows))
    for strategy in selected_strategies:
        for fold in range(effective_folds):
            train = [
                row for index, row in indexed_rows
                if _fold_id(row, effective_folds) != fold
            ]
            test = [
                (index, row) for index, row in indexed_rows
                if _fold_id(row, effective_folds) == fold
            ]
            if not test:
                continue
            model = _fit_logistic_rescue_model(train, strategy)
            for index, row in test:
                output[strategy][index] = model.predict_probability(row)
    return output


def _fit_logistic_rescue_model(
    rows: list[dict[str, Any]],
    target_strategy: str,
) -> LogisticRescueModel:
    feature_spec = _fit_feature_spec(rows)
    x_rows = [_feature_vector(row, feature_spec) for row in rows]
    y_rows = [
        1.0
        if (
            not _to_bool(row.get("direct_correct"))
            and _strategy_correct(row, target_strategy)
        )
        else 0.0
        for row in rows
    ]
    if not x_rows:
        return LogisticRescueModel(feature_spec, [0.0], 1.0, 0.0)
    weights = _fit_logistic_weights(x_rows, y_rows, epochs=350, learning_rate=0.12, l2=0.01)
    scores = [_dot(weights, x_row) for x_row in x_rows]
    platt_a, platt_b = _fit_platt_scaler(scores, y_rows)
    return LogisticRescueModel(feature_spec, weights, platt_a, platt_b)


def _fit_feature_spec(rows: list[dict[str, Any]]) -> FeatureSpec:
    numeric_means = {}
    numeric_scales = {}
    for feature in NUMERIC_FEATURES:
        values = [_to_float(row.get(feature)) for row in rows]
        mean = sum(values) / len(values) if values else 0.0
        variance = (
            sum((value - mean) ** 2 for value in values) / len(values)
            if values
            else 0.0
        )
        scale = math.sqrt(variance) if variance > 0 else 1.0
        numeric_means[feature] = mean
        numeric_scales[feature] = scale
    category_support = max(2, len(rows) // 100)
    categories: dict[str, list[str]] = {}
    for feature in CATEGORICAL_FEATURES:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(feature) or "")
            if value:
                counts[value] = counts.get(value, 0) + 1
        categories[feature] = [
            value for value, count in sorted(counts.items())
            if count >= category_support
        ]
    return FeatureSpec(numeric_means, numeric_scales, categories)


def _feature_vector(row: dict[str, Any], feature_spec: FeatureSpec) -> list[float]:
    values = [1.0]
    for feature in NUMERIC_FEATURES:
        values.append(
            (
                _to_float(row.get(feature))
                - feature_spec.numeric_means.get(feature, 0.0)
            )
            / feature_spec.numeric_scales.get(feature, 1.0)
        )
    for feature in BOOLEAN_FEATURES:
        values.append(1.0 if _to_bool(row.get(feature)) else 0.0)
    for feature in CATEGORICAL_FEATURES:
        row_value = str(row.get(feature) or "")
        for value in feature_spec.categories.get(feature, []):
            values.append(1.0 if row_value == value else 0.0)
    return values


def _fit_logistic_weights(
    x_rows: list[list[float]],
    y_rows: list[float],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> list[float]:
    width = len(x_rows[0])
    weights = [0.0 for _ in range(width)]
    n = len(x_rows)
    for epoch in range(epochs):
        gradients = [0.0 for _ in range(width)]
        for x_row, label in zip(x_rows, y_rows):
            error = _sigmoid(_dot(weights, x_row)) - label
            for index, value in enumerate(x_row):
                gradients[index] += error * value
        step = learning_rate / (1.0 + epoch / 150.0)
        for index in range(width):
            penalty = 0.0 if index == 0 else l2 * weights[index]
            weights[index] -= step * (gradients[index] / n + penalty)
    return weights


def _fit_platt_scaler(scores: list[float], labels: list[float]) -> tuple[float, float]:
    if not scores or len(set(labels)) < 2:
        return 1.0, 0.0
    a = 1.0
    b = 0.0
    n = len(scores)
    for epoch in range(250):
        grad_a = 0.0
        grad_b = 0.0
        for score, label in zip(scores, labels):
            error = _sigmoid(a * score + b) - label
            grad_a += error * score
            grad_b += error
        step = 0.08 / (1.0 + epoch / 120.0)
        a -= step * grad_a / n
        b -= step * grad_b / n
    return a, b


def _quantiles(values: list[float], quantiles: Iterable[float]) -> list[float]:
    if not values:
        return []
    output = []
    for quantile in quantiles:
        index = min(len(values) - 1, max(0, int(round((len(values) - 1) * quantile))))
        output.append(values[index])
    return sorted(set(output))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * quantile))),
    )
    return ordered[index]


def _strategy_correct(row: dict[str, Any], strategy: str) -> bool:
    if strategy == "direct":
        return _to_bool(row.get("direct_correct"))
    return _to_bool(row.get(f"{strategy_prefix(strategy)}_correct"))


def _oracle_after_direct_correct(row: dict[str, Any]) -> bool:
    if _to_bool(row.get("direct_correct")):
        return True
    return any(
        _to_bool(row.get(f"{strategy_prefix(strategy)}_correct"))
        for strategy in COLLAB_STRATEGIES
        if _strategy_observed(row, strategy)
    )


def _fold_id(row: dict[str, Any], folds: int) -> int:
    key = f"{row.get('dataset')}|{row.get('model')}|{row.get('question_id')}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def _retained_gain(direct: float, oracle: float, learned: float) -> float | None:
    denominator = oracle - direct
    if denominator <= 0:
        return None
    return (learned - direct) / denominator


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _render_report(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    budget_multiplier: float,
) -> str:
    aggregate = result["aggregate"]
    routing_curve = result.get("routing_curve", [])
    routing_pareto = result.get("routing_pareto", [])
    direct = evaluate_policy(rows, [])
    oracle = evaluate_direct_first_oracle(rows)
    retained = _retained_gain(
        direct["accuracy"],
        oracle["accuracy"],
        aggregate["accuracy"],
    )
    lines = [
        "# Offline Router Cross-Validation",
        "",
        f"Scope: {len(rows)} feature rows; out-of-fold greedy rule-list evaluation under a {budget_multiplier:.2f}x direct-cost budget.",
        "",
        "## Opportunity Gap",
        "",
        (
            f"- Oracle opportunity: {direct['accuracy']:.3f} -> "
            f"{oracle['accuracy']:.3f} "
            f"({oracle['accuracy'] - direct['accuracy']:+.3f}) @ "
            f"{oracle['cost_multiplier']:.2f}x direct cost."
        ),
        (
            f"- Learned rule-list captured: {direct['accuracy']:.3f} -> "
            f"{aggregate['accuracy']:.3f} "
            f"({aggregate['accuracy'] - direct['accuracy']:+.3f}) @ "
            f"{aggregate['cost_multiplier']:.2f}x direct cost."
        ),
        f"- Retained oracle gain (ROG): {_fmt_pct(retained) if retained is not None else 'n/a'}.",
        (
            f"- Cost overhead vs direct-first oracle: "
            f"{_fmt_optional_multiplier(_safe_div(float(aggregate['cost_multiplier']), float(oracle['cost_multiplier'])))}."
        ),
        "",
        "## Aggregate",
        "",
        "| policy | accuracy | gain vs direct | cost x | escalation | cost/correct |",
        "|---|---:|---:|---:|---:|---:|",
        _policy_line("direct", direct, direct),
        _policy_line("direct-first oracle", oracle, direct),
        _policy_line("learned rule-list", aggregate, direct),
        "",
        f"Retained oracle gain: {_fmt_pct(retained) if retained is not None else 'n/a'}.",
        "",
        "## Folds",
        "",
        "| fold | n | direct | learned | oracle | learned cost x | retained oracle gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["folds"]:
        retained_fold = row["retained_oracle_gain"]
        lines.append(
            "| {fold} | {test_n} | {direct_accuracy:.3f} | {learned_accuracy:.3f} | "
            "{oracle_after_direct_accuracy:.3f} | {learned_cost_multiplier:.2f} | "
            "{retained} |".format(
                retained=_fmt_pct(retained_fold) if retained_fold is not None else "n/a",
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Learned Rules",
            "",
            "| fold | position | predicate | target |",
            "|---|---:|---|---|",
        ]
    )
    for row in result["rules"]:
        lines.append(
            f"| {row['fold']} | {row['position']} | `{row['label']}` | `{row['target_strategy']}` |"
        )
    if not result["rules"]:
        lines.append("| n/a | n/a | no positive budgeted rule learned | n/a |")
    lines.extend(
        [
            "",
            "## Routing Curve",
            "",
            "| policy | family | target | threshold | accuracy | ROG | cost x | cost overhead vs oracle | escalation |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _best_curve_rows(routing_curve, limit=12):
        lines.append(_curve_line(row))
    if not routing_curve:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Pareto Frontier",
            "",
            "| rank | policy | family | accuracy | cost x | ROG |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in routing_pareto[:12]:
        lines.append(
            "| {rank} | `{policy}` | {family} | {accuracy:.3f} | {cost:.2f} | {rog} |".format(
                rank=row.get("pareto_rank", ""),
                policy=row["policy"],
                family=row["policy_family"],
                accuracy=float(row["accuracy"]),
                cost=float(row["cost_multiplier"]),
                rog=_fmt_pct(row.get("retained_oracle_gain")),
            )
        )
    if not routing_pareto:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `direct-first oracle` charges the direct call for every question, then charges the cheapest successful collaboration only when direct is wrong.",
            "- The learned router is deployable in the narrow MVP sense: it uses question/direct-output features and pays for any escalation it chooses.",
            "- Treat this as model/dataset-internal evidence; publication claims still need stronger cross-model validation and confidence intervals.",
        ]
    )
    return "\n".join(lines) + "\n"


def _best_curve_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deployable = [
        row for row in rows
        if row.get("policy_family") not in {"oracle", "baseline"}
    ]
    return sorted(
        deployable,
        key=lambda row: (
            -float(row["accuracy"]),
            float(row["cost_multiplier"]),
            str(row["policy"]),
        ),
    )[:limit]


def _curve_line(row: dict[str, Any]) -> str:
    threshold = row.get("threshold", "")
    threshold_text = "n/a" if threshold == "" else f"{float(threshold):.2f}"
    return (
        f"| `{row['policy']}` | {row['policy_family']} | `{row['target_strategy']}` | "
        f"{threshold_text} | {float(row['accuracy']):.3f} | "
        f"{_fmt_pct(row.get('retained_oracle_gain'))} | "
        f"{float(row['cost_multiplier']):.2f} | "
        f"{_fmt_optional_multiplier(row.get('cost_overhead_vs_oracle'))} | "
        f"{float(row['escalation_rate']):.1%} |"
    )


def _policy_line(name: str, metrics: dict[str, Any], direct: dict[str, Any]) -> str:
    gain = metrics["accuracy"] - direct["accuracy"]
    return (
        f"| {name} | {metrics['accuracy']:.3f} | {gain:+.3f} | "
        f"{metrics['cost_multiplier']:.2f} | {metrics['escalation_rate']:.1%} | "
        f"{_fmt_usd(metrics.get('cost_per_correct'))} |"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for feature in NUMERIC_FEATURES:
        normalized[feature] = _to_float(normalized.get(feature))
    for feature in BOOLEAN_FEATURES:
        normalized[feature] = _to_bool(normalized.get(feature))
    for strategy in ("direct", *COLLAB_STRATEGIES):
        prefix = strategy_prefix(strategy)
        if f"{prefix}_correct" in normalized:
            normalized[f"{prefix}_correct"] = _to_bool(normalized.get(f"{prefix}_correct"))
        if f"{prefix}_cost_usd" in normalized:
            normalized[f"{prefix}_cost_usd"] = _to_float(normalized.get(f"{prefix}_cost_usd"))
    return normalized


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


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _fmt_optional_multiplier(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.2f}x"


def _fmt_usd(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):.6f}"
