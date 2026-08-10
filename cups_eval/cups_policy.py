from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cups_eval.offline_router import (
    _dot,
    _feature_vector,
    _fit_feature_spec,
    _fit_logistic_weights,
    _fit_platt_scaler,
    _fold_id,
    _sigmoid,
    _to_bool,
    _to_float,
    strategy_cost,
    strategy_prefix,
)


POLICY_STRATEGIES = (
    "direct",
    "cot",
    "planner_verifier",
    "debate_2_judge",
)

DEFAULT_LAMBDAS = (
    0.0,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
)


@dataclass(frozen=True)
class StrategyCorrectnessModel:
    feature_spec: Any
    weights: list[float]
    platt_a: float
    platt_b: float
    constant_probability: float | None = None

    def predict_probability(self, row: dict[str, Any]) -> float:
        if self.constant_probability is not None:
            return self.constant_probability
        score = _dot(self.weights, _feature_vector(row, self.feature_spec))
        return _sigmoid(self.platt_a * score + self.platt_b)


def fit_strategy_models(
    rows: list[dict[str, Any]],
    strategies: tuple[str, ...] | None = None,
) -> dict[str, StrategyCorrectnessModel]:
    selected_strategies = strategies or available_policy_strategies(rows)
    feature_spec = _fit_feature_spec(rows)
    x_rows = [_feature_vector(row, feature_spec) for row in rows]
    models: dict[str, StrategyCorrectnessModel] = {}
    for strategy in selected_strategies:
        labels = [1.0 if strategy_correct(row, strategy) else 0.0 for row in rows]
        if not x_rows:
            models[strategy] = StrategyCorrectnessModel(feature_spec, [0.0], 1.0, 0.0, 0.0)
            continue
        if len(set(labels)) < 2:
            probability = (sum(labels) + 1.0) / (len(labels) + 2.0)
            models[strategy] = StrategyCorrectnessModel(
                feature_spec,
                [0.0 for _ in x_rows[0]],
                1.0,
                0.0,
                probability,
            )
            continue
        weights = _fit_logistic_weights(
            x_rows,
            labels,
            epochs=400,
            learning_rate=0.12,
            l2=0.01,
        )
        scores = [_dot(weights, x_row) for x_row in x_rows]
        platt_a, platt_b = _fit_platt_scaler(scores, labels)
        models[strategy] = StrategyCorrectnessModel(feature_spec, weights, platt_a, platt_b)
    return models


def predict_strategy_probabilities(
    row: dict[str, Any],
    models: dict[str, StrategyCorrectnessModel],
) -> dict[str, float]:
    return {
        strategy: model.predict_probability(row)
        for strategy, model in models.items()
        if strategy_total_cost(row, strategy) > 0.0
    }


def select_strategy(
    row: dict[str, Any],
    probabilities: dict[str, float],
    *,
    lambda_value: float,
    estimated_cost_ratios: dict[str, float] | None = None,
) -> str:
    if not probabilities:
        return "direct"
    priority = {strategy: index for index, strategy in enumerate(POLICY_STRATEGIES)}
    best: tuple[float, float, int, str] | None = None
    best_strategy = "direct"
    for strategy, probability in probabilities.items():
        cost_ratio = (
            float(estimated_cost_ratios[strategy])
            if estimated_cost_ratios is not None and strategy in estimated_cost_ratios
            else strategy_cost_ratio(row, strategy)
        )
        utility = float(probability) - float(lambda_value) * cost_ratio
        candidate = (
            utility,
            -cost_ratio,
            -priority.get(strategy, len(priority)),
            strategy,
        )
        if best is None or candidate > best:
            best = candidate
            best_strategy = strategy
    return best_strategy


def build_cups_predictions(
    rows: list[dict[str, Any]],
    probabilities_by_key: dict[Any, dict[str, float]],
    *,
    lambda_value: float,
    fold: int | None = None,
    cost_ratios_by_key: dict[Any, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        probabilities = _lookup_probabilities(probabilities_by_key, row)
        estimated_cost_ratios = (
            _lookup_probabilities(cost_ratios_by_key, row)
            if cost_ratios_by_key is not None
            else None
        )
        selected = select_strategy(
            row,
            probabilities,
            lambda_value=lambda_value,
            estimated_cost_ratios=estimated_cost_ratios,
        )
        predictions.append(
            prediction_row(
                row,
                selected_strategy=selected,
                selected_probability=probabilities.get(selected, 0.0),
                lambda_value=lambda_value,
                fold=fold,
                policy=f"cups_lambda_{format_lambda(lambda_value)}",
                selection_cost_ratio=(
                    estimated_cost_ratios.get(selected)
                    if estimated_cost_ratios
                    else strategy_cost_ratio(row, selected)
                ),
            )
        )
    return predictions


def aggregate_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(predictions)
    if not n:
        return {
            "n": 0,
            "correct": 0,
            "accuracy": 0.0,
            "estimated_cost_usd": 0.0,
            "direct_cost_usd": 0.0,
            "cost_multiplier": 0.0,
            "relative_cost_per_correct": None,
            "cost_per_correct": None,
            "escalation_rate": 0.0,
        }
    correct = sum(1 for row in predictions if _to_bool(row.get("selected_correct")))
    cost = sum(_to_float(row.get("estimated_cost_usd")) for row in predictions)
    direct_cost = sum(_to_float(row.get("direct_cost_usd")) for row in predictions)
    non_direct = sum(1 for row in predictions if row.get("selected_strategy") != "direct")
    accuracy = correct / n
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
        "escalation_rate": non_direct / n,
    }


def cross_validate_cups(
    rows: list[dict[str, Any]],
    *,
    folds: int = 5,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
) -> dict[str, Any]:
    if not rows:
        curve_result = build_cups_curve([], {}, lambdas=lambdas)
        return {
            "folds": [],
            "probabilities": [],
            "predictions": [],
            "curve": curve_result["curve"],
            "pareto": curve_result["pareto"],
        }
    effective_folds = max(2, min(folds, len(rows)))
    strategies = available_policy_strategies(rows)
    probabilities_by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    cost_ratios_by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    probability_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in range(effective_folds):
        train = [row for row in rows if _fold_id(row, effective_folds) != fold]
        test = [row for row in rows if _fold_id(row, effective_folds) == fold]
        models = fit_strategy_models(train, strategies)
        cost_ratios = fit_strategy_cost_ratios(train, strategies)
        fold_probability_map: dict[Any, dict[str, float]] = {}
        fold_cost_ratio_map: dict[Any, dict[str, float]] = {}
        for row in test:
            probabilities = predict_strategy_probabilities(row, models)
            key = row_key(row)
            probabilities_by_key[key] = probabilities
            cost_ratios_by_key[key] = cost_ratios
            fold_probability_map[key] = probabilities
            fold_cost_ratio_map[key] = cost_ratios
            probability_rows.append(
                {
                    "fold": fold,
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "question_id": row.get("question_id", ""),
                    **{f"p_{strategy}": probabilities.get(strategy, "") for strategy in POLICY_STRATEGIES},
                    **{f"cost_ratio_{strategy}": cost_ratios.get(strategy, "") for strategy in POLICY_STRATEGIES},
                }
            )
        fold_curve = build_cups_curve(
            test,
            fold_probability_map,
            lambdas=lambdas,
            cost_ratios_by_key=fold_cost_ratio_map,
        )["curve"]
        best_cups = best_cups_curve_row(fold_curve)
        direct = _curve_by_policy(fold_curve, "direct")
        cot = _curve_by_policy(fold_curve, "always_cot")
        oracle = _curve_by_policy(fold_curve, "strategy_selection_oracle")
        fold_rows.append(
            {
                "fold": fold,
                "train_n": len(train),
                "test_n": len(test),
                "direct_accuracy": _to_float(direct.get("accuracy")),
                "cot_accuracy": _to_float(cot.get("accuracy")),
                "oracle_accuracy": _to_float(oracle.get("accuracy")),
                "best_cups_policy": best_cups.get("policy", ""),
                "best_cups_accuracy": _to_float(best_cups.get("accuracy")),
                "best_cups_cost_multiplier": _to_float(best_cups.get("cost_multiplier")),
                "best_cups_relative_cpc": best_cups.get("relative_cost_per_correct"),
            }
        )
    curve_result = build_cups_curve(
        rows,
        probabilities_by_key,
        lambdas=lambdas,
        cost_ratios_by_key=cost_ratios_by_key,
    )
    predictions: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        predictions.extend(
            build_cups_predictions(
                rows,
                probabilities_by_key,
                lambda_value=lambda_value,
                cost_ratios_by_key=cost_ratios_by_key,
            )
        )
    return {
        "folds": fold_rows,
        "probabilities": probability_rows,
        "predictions": predictions,
        "curve": curve_result["curve"],
        "pareto": curve_result["pareto"],
    }


def cross_validate_cups_group_holdout(
    rows: list[dict[str, Any]],
    *,
    holdout: str,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
) -> dict[str, Any]:
    groups = _group_rows_for_holdout(rows, holdout)
    regime = _holdout_regime_name(holdout)
    if len(groups) < 2:
        raise ValueError(f"CUPS {regime} requires at least two groups; found {len(groups)}")

    strategies = available_policy_strategies(rows)
    probabilities_by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    cost_ratios_by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    probability_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for fold, (group, test) in enumerate(sorted(groups.items())):
        train = [
            row
            for other_group, group_rows in groups.items()
            if other_group != group
            for row in group_rows
        ]
        models = fit_strategy_models(train, strategies)
        cost_ratios = fit_strategy_cost_ratios(train, strategies)
        fold_probability_map: dict[Any, dict[str, float]] = {}
        fold_cost_ratio_map: dict[Any, dict[str, float]] = {}
        for row in test:
            probabilities = predict_strategy_probabilities(row, models)
            key = row_key(row)
            probabilities_by_key[key] = probabilities
            cost_ratios_by_key[key] = cost_ratios
            fold_probability_map[key] = probabilities
            fold_cost_ratio_map[key] = cost_ratios
            probability_rows.append(
                {
                    "fold": fold,
                    "evaluation_regime": regime,
                    "heldout_group": group,
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "question_id": row.get("question_id", ""),
                    **{f"p_{strategy}": probabilities.get(strategy, "") for strategy in POLICY_STRATEGIES},
                    **{f"cost_ratio_{strategy}": cost_ratios.get(strategy, "") for strategy in POLICY_STRATEGIES},
                }
            )
        fold_curve = build_cups_curve(
            test,
            fold_probability_map,
            lambdas=lambdas,
            cost_ratios_by_key=fold_cost_ratio_map,
        )["curve"]
        best_cups = best_cups_curve_row(fold_curve)
        direct = _curve_by_policy(fold_curve, "direct")
        cot = _curve_by_policy(fold_curve, "always_cot")
        oracle = _curve_by_policy(fold_curve, "strategy_selection_oracle")
        fold_rows.append(
            {
                "fold": fold,
                "evaluation_regime": regime,
                "heldout_group": group,
                "train_n": len(train),
                "test_n": len(test),
                "direct_accuracy": _to_float(direct.get("accuracy")),
                "cot_accuracy": _to_float(cot.get("accuracy")),
                "oracle_accuracy": _to_float(oracle.get("accuracy")),
                "best_cups_policy": best_cups.get("policy", ""),
                "best_cups_accuracy": _to_float(best_cups.get("accuracy")),
                "best_cups_cost_multiplier": _to_float(best_cups.get("cost_multiplier")),
                "best_cups_relative_cpc": best_cups.get("relative_cost_per_correct"),
            }
        )

    curve_result = build_cups_curve(
        rows,
        probabilities_by_key,
        lambdas=lambdas,
        cost_ratios_by_key=cost_ratios_by_key,
    )
    predictions: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        predictions.extend(
            build_cups_predictions(
                rows,
                probabilities_by_key,
                lambda_value=lambda_value,
                cost_ratios_by_key=cost_ratios_by_key,
            )
        )
    return {
        "folds": fold_rows,
        "probabilities": probability_rows,
        "predictions": predictions,
        "curve": curve_result["curve"],
        "pareto": curve_result["pareto"],
    }


def build_cups_curve(
    rows: list[dict[str, Any]],
    probabilities_by_key: dict[Any, dict[str, float]],
    *,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    cost_ratios_by_key: dict[Any, dict[str, float]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {"curve": [], "pareto": []}
    available = available_policy_strategies(rows)
    direct_predictions = strategy_predictions(rows, "direct")
    cot_predictions = (
        strategy_predictions(rows, "cot")
        if "cot" in available
        else direct_predictions
    )
    oracle_predictions = strategy_selection_oracle_predictions(rows, available)
    direct_metrics = aggregate_predictions(direct_predictions)
    cot_metrics = aggregate_predictions(cot_predictions)
    oracle_metrics = aggregate_predictions(oracle_predictions)
    curve = [
        curve_row(
            "direct",
            "baseline",
            "direct",
            None,
            direct_predictions,
            direct_metrics,
            cot_metrics,
            oracle_metrics,
        )
    ]
    if "cot" in available:
        curve.append(
            curve_row(
                "always_cot",
                "baseline",
                "cot",
                None,
                cot_predictions,
                direct_metrics,
                cot_metrics,
                oracle_metrics,
            )
        )
    for strategy in available:
        if strategy in {"direct", "cot"}:
            continue
        curve.append(
            curve_row(
                f"always_{strategy}",
                "always_collaborate",
                strategy,
                None,
                strategy_predictions(rows, strategy),
                direct_metrics,
                cot_metrics,
                oracle_metrics,
            )
        )
    curve.append(
        curve_row(
            "strategy_selection_oracle",
            "oracle",
            "mixed",
            None,
            oracle_predictions,
            direct_metrics,
            cot_metrics,
            oracle_metrics,
        )
    )
    for lambda_value in lambdas:
        curve.append(
            curve_row(
                f"cups_lambda_{format_lambda(lambda_value)}",
                "cups",
                "mixed",
                lambda_value,
                build_cups_predictions(
                    rows,
                    probabilities_by_key,
                    lambda_value=lambda_value,
                    cost_ratios_by_key=cost_ratios_by_key,
                ),
                direct_metrics,
                cot_metrics,
                oracle_metrics,
            )
        )
    return {"curve": curve, "pareto": pareto_rows(curve)}


def write_cups_report(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "cups_curve.csv", result.get("curve", []))
    _write_csv(output_dir / "cups_pareto.csv", result.get("pareto", []))
    _write_csv(output_dir / "cups_predictions.csv", result.get("predictions", []))
    _write_csv(output_dir / "cups_probabilities.csv", result.get("probabilities", []))
    _write_csv(output_dir / "cups_folds.csv", result.get("folds", []))
    summary = {
        "n": len(rows),
        "best_cups_vs_cot": best_cups_vs_cot(result.get("curve", [])),
        "best_cups": best_cups_curve_row(result.get("curve", [])),
        "curve": result.get("curve", []),
        "pareto": result.get("pareto", []),
        "folds": result.get("folds", []),
    }
    (output_dir / "cups_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "cups_policy_report.md").write_text(
        render_report(rows, result),
        encoding="utf-8",
    )


def available_policy_strategies(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    output = []
    for strategy in POLICY_STRATEGIES:
        if any(strategy_total_cost(row, strategy) > 0.0 for row in rows):
            output.append(strategy)
    return tuple(output or ("direct",))


def fit_strategy_cost_ratios(
    rows: list[dict[str, Any]],
    strategies: tuple[str, ...],
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for strategy in strategies:
        values = [
            strategy_cost_ratio(row, strategy)
            for row in rows
            if strategy_total_cost(row, strategy) > 0.0
            and strategy_total_cost(row, "direct") > 0.0
        ]
        ratios[strategy] = sum(values) / len(values) if values else 1.0
    ratios.setdefault("direct", 1.0)
    return ratios


def strategy_predictions(rows: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    return [
        prediction_row(
            row,
            selected_strategy=strategy,
            selected_probability=1.0,
            lambda_value=None,
            fold=None,
            policy=f"always_{strategy}" if strategy != "direct" else "direct",
        )
        for row in rows
    ]


def strategy_selection_oracle_predictions(
    rows: list[dict[str, Any]],
    strategies: tuple[str, ...],
) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        correct_strategies = [
            (strategy_total_cost(row, strategy), strategy)
            for strategy in strategies
            if strategy_correct(row, strategy)
            and strategy_total_cost(row, strategy) > 0.0
        ]
        if correct_strategies:
            _, selected = min(
                correct_strategies,
                key=lambda item: (item[0], POLICY_STRATEGIES.index(item[1])),
            )
        else:
            selected = min(
                strategies,
                key=lambda strategy: (
                    strategy_total_cost(row, strategy) or math.inf,
                    POLICY_STRATEGIES.index(strategy),
                ),
            )
        predictions.append(
            prediction_row(
                row,
                selected_strategy=selected,
                selected_probability=1.0,
                lambda_value=None,
                fold=None,
                policy="strategy_selection_oracle",
            )
        )
    return predictions


def prediction_row(
    row: dict[str, Any],
    *,
    selected_strategy: str,
    selected_probability: float,
    lambda_value: float | None,
    fold: int | None,
    policy: str,
    selection_cost_ratio: float | None = None,
) -> dict[str, Any]:
    return {
        "fold": "" if fold is None else fold,
        "policy": policy,
        "lambda": "" if lambda_value is None else lambda_value,
        "dataset": row.get("dataset", ""),
        "model": row.get("model", ""),
        "question_id": row.get("question_id", ""),
        "gold": row.get("gold", ""),
        "selected_strategy": selected_strategy,
        "selected_probability": selected_probability,
        "selected_correct": strategy_correct(row, selected_strategy),
        "direct_correct": strategy_correct(row, "direct"),
        "direct_cost_usd": strategy_total_cost(row, "direct"),
        "estimated_cost_usd": strategy_total_cost(row, selected_strategy),
        "item_cost_multiplier": strategy_cost_ratio(row, selected_strategy),
        "selection_cost_ratio": (
            strategy_cost_ratio(row, selected_strategy)
            if selection_cost_ratio is None
            else selection_cost_ratio
        ),
    }


def curve_row(
    policy: str,
    policy_family: str,
    target_strategy: str,
    lambda_value: float | None,
    predictions: list[dict[str, Any]],
    direct_metrics: dict[str, Any],
    cot_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
) -> dict[str, Any]:
    metrics = aggregate_predictions(predictions)
    accuracy = float(metrics["accuracy"])
    direct_accuracy = float(direct_metrics["accuracy"])
    cot_accuracy = float(cot_metrics["accuracy"])
    oracle_accuracy = float(oracle_metrics["accuracy"])
    cpc = metrics.get("relative_cost_per_correct")
    direct_cpc = direct_metrics.get("relative_cost_per_correct")
    cot_cpc = cot_metrics.get("relative_cost_per_correct")
    return {
        "policy": policy,
        "policy_family": policy_family,
        "target_strategy": target_strategy,
        "lambda": "" if lambda_value is None else lambda_value,
        "n": metrics["n"],
        "accuracy": accuracy,
        "delta_vs_direct": accuracy - direct_accuracy,
        "delta_vs_cot": accuracy - cot_accuracy,
        "retained_direct_oracle_gain": retained_gain(direct_accuracy, oracle_accuracy, accuracy),
        "retained_cot_oracle_gain": retained_gain(cot_accuracy, oracle_accuracy, accuracy),
        "estimated_cost_usd": metrics["estimated_cost_usd"],
        "cost_multiplier": metrics["cost_multiplier"],
        "relative_cost_per_correct": cpc,
        "cpc_delta_vs_direct": (
            float(cpc) - float(direct_cpc)
            if cpc is not None and direct_cpc is not None
            else None
        ),
        "cpc_delta_vs_cot": (
            float(cpc) - float(cot_cpc)
            if cpc is not None and cot_cpc is not None
            else None
        ),
        "escalation_rate": metrics["escalation_rate"],
        "dominates_direct_cpc": dominates(metrics, direct_metrics),
        "dominates_cot_cpc": dominates(metrics, cot_metrics),
    }


def pareto_rows(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row for row in curve
        if row.get("policy_family") != "oracle"
        and row.get("relative_cost_per_correct") is not None
    ]
    frontier = []
    best_accuracy = -1.0
    for row in sorted(
        candidates,
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


def strategy_correct(row: dict[str, Any], strategy: str) -> bool:
    if strategy == "direct":
        return _to_bool(row.get("direct_correct"))
    if strategy == "cot":
        return _to_bool(row.get("cot_correct"))
    return _to_bool(row.get(f"{strategy_prefix(strategy)}_correct"))


def strategy_total_cost(row: dict[str, Any], strategy: str) -> float:
    if strategy == "direct":
        return _to_float(row.get("direct_cost_usd"))
    if strategy == "cot":
        return _to_float(row.get("cot_cost_usd"))
    return strategy_cost(row, strategy)


def strategy_cost_ratio(row: dict[str, Any], strategy: str) -> float:
    direct_cost = strategy_total_cost(row, "direct")
    strategy_cost_value = strategy_total_cost(row, strategy)
    if direct_cost <= 0:
        return strategy_cost_value
    return strategy_cost_value / direct_cost


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("model") or ""),
        str(row.get("question_id") or ""),
    )


def format_lambda(lambda_value: float) -> str:
    text = f"{lambda_value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _group_rows_for_holdout(rows: list[dict[str, Any]], holdout: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = _holdout_group(row, holdout)
        groups.setdefault(group, []).append(row)
    return groups


def _holdout_group(row: dict[str, Any], holdout: str) -> str:
    dataset = str(row.get("dataset") or "")
    model = str(row.get("model") or "")
    if holdout == "dataset_model":
        return f"{dataset}-{model}"
    if holdout == "model":
        return model
    raise ValueError(f"Unknown CUPS holdout: {holdout}")


def _holdout_regime_name(holdout: str) -> str:
    if holdout == "dataset_model":
        return "leave_one_dataset_model_out"
    if holdout == "model":
        return "leave_one_model_out"
    raise ValueError(f"Unknown CUPS holdout: {holdout}")


def retained_gain(baseline: float, oracle: float, candidate: float) -> float | None:
    denominator = oracle - baseline
    if denominator <= 0:
        return None
    return (candidate - baseline) / denominator


def dominates(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    metric_cpc = metrics.get("relative_cost_per_correct")
    baseline_cpc = baseline.get("relative_cost_per_correct")
    if metric_cpc is None or baseline_cpc is None:
        return False
    return (
        float(metrics["accuracy"]) > float(baseline["accuracy"])
        and float(metric_cpc) <= float(baseline_cpc)
    )


def best_cups_curve_row(curve: list[dict[str, Any]]) -> dict[str, Any]:
    cups_rows = [row for row in curve if row.get("policy_family") == "cups"]
    if not cups_rows:
        return {}
    return max(
        cups_rows,
        key=lambda row: (
            _to_float(row.get("accuracy")),
            -_to_float(row.get("relative_cost_per_correct")),
            -_to_float(row.get("cost_multiplier")),
        ),
    )


def best_cups_vs_cot(curve: list[dict[str, Any]]) -> dict[str, Any]:
    cot = _curve_by_policy(curve, "always_cot")
    if not cot:
        return best_cups_curve_row(curve)
    cot_acc = _to_float(cot.get("accuracy"))
    cot_cpc = cot.get("relative_cost_per_correct")
    candidates = [row for row in curve if row.get("policy_family") == "cups"]
    dominating = [
        row for row in candidates
        if _to_float(row.get("accuracy")) > cot_acc
        and row.get("relative_cost_per_correct") is not None
        and cot_cpc is not None
        and float(row["relative_cost_per_correct"]) <= float(cot_cpc)
    ]
    pool = dominating or [
        row for row in candidates
        if _to_float(row.get("accuracy")) > cot_acc
    ] or candidates
    if not pool:
        return {}
    return max(
        pool,
        key=lambda row: (
            bool(row in dominating),
            _to_float(row.get("accuracy")),
            -_to_float(row.get("relative_cost_per_correct")),
        ),
    )


def render_report(rows: list[dict[str, Any]], result: dict[str, Any]) -> str:
    curve = result.get("curve", [])
    pareto = result.get("pareto", [])
    by_policy = {row.get("policy"): row for row in curve}
    best = best_cups_vs_cot(curve)
    lines = [
        "# Cost-aware Utility Policy Selection (CUPS)",
        "",
        "No provider calls were made; this report is computed from existing saved predictions.",
        "",
        "CUPS trains one out-of-fold correctness model per strategy and selects the strategy with the highest probability-minus-cost utility.",
        "Selection uses train-fold mean cost ratios; evaluation then charges the realized cost of the selected strategy.",
        "",
        "## Main Policies",
        "",
        "| policy | family | accuracy | cost x | relative CPC | delta vs always-CoT | dominates always-CoT CPC |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for policy in (
        "direct",
        "always_cot",
        "always_planner_verifier",
        "always_debate_2_judge",
        "strategy_selection_oracle",
    ):
        row = by_policy.get(policy)
        if row:
            lines.append(_policy_line(row))
    if best:
        labeled = dict(best)
        labeled["policy"] = f"best_cups::{labeled['policy']}"
        lines.append(_policy_line(labeled))
    lines.extend(
        [
            "",
            "## CUPS Lambda Sweep",
            "",
            "| policy | accuracy | cost x | relative CPC | delta vs always-CoT | non-direct rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [row for row in curve if row.get("policy_family") == "cups"]:
        lines.append(
            "| `{policy}` | {acc:.3f} | {cost:.2f} | {cpc} | {delta:+.3f} | {rate:.1%} |".format(
                policy=row["policy"],
                acc=_to_float(row.get("accuracy")),
                cost=_to_float(row.get("cost_multiplier")),
                cpc=_fmt_optional(row.get("relative_cost_per_correct")),
                delta=_to_float(row.get("delta_vs_cot")),
                rate=_to_float(row.get("escalation_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## Pareto Frontier",
            "",
            "| rank | policy | family | accuracy | relative CPC | cost x |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in pareto[:20]:
        lines.append(
            "| {rank} | `{policy}` | {family} | {acc:.3f} | {cpc} | {cost:.2f} |".format(
                rank=row.get("pareto_rank", ""),
                policy=row.get("policy", ""),
                family=row.get("policy_family", ""),
                acc=_to_float(row.get("accuracy")),
                cpc=_fmt_optional(row.get("relative_cost_per_correct")),
                cost=_to_float(row.get("cost_multiplier")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- Treat CUPS as a deployable policy selector only if at least one lambda point improves over always-CoT at equal or lower relative cost per correct.",
            "- If it fails that bar, the result is still useful evidence that the current cheap observables do not capture the oracle opportunity.",
        ]
    )
    return "\n".join(lines) + "\n"


def _lookup_probabilities(
    probabilities_by_key: dict[Any, dict[str, float]],
    row: dict[str, Any],
) -> dict[str, float]:
    return (
        probabilities_by_key.get(row_key(row))
        or probabilities_by_key.get(str(row.get("question_id") or ""))
        or probabilities_by_key.get(row.get("question_id"))
        or {}
    )


def _curve_by_policy(curve: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    return next((row for row in curve if row.get("policy") == policy), {})


def _policy_line(row: dict[str, Any]) -> str:
    return "| `{policy}` | {family} | {acc:.3f} | {cost:.2f} | {cpc} | {delta:+.3f} | {dom} |".format(
        policy=row.get("policy", ""),
        family=row.get("policy_family", ""),
        acc=_to_float(row.get("accuracy")),
        cost=_to_float(row.get("cost_multiplier")),
        cpc=_fmt_optional(row.get("relative_cost_per_correct")),
        delta=_to_float(row.get("delta_vs_cot")),
        dom=row.get("dominates_cot_cpc"),
    )


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
    if value in (None, ""):
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number) or math.isinf(number):
        return "NA"
    return f"{number:.3f}"
