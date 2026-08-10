from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable


BASELINE_STRATEGIES = (
    "direct",
    "cot",
    "self_consistency_3",
    "planner_verifier",
    "debate_2_judge",
)


@dataclass(frozen=True)
class RuleSpec:
    name: str
    fn: Callable[[dict[str, Any]], dict[str, Any]]
    deployable: bool = True
    observation_mode: str = "predecision_or_charged_probe"
    cost_note: str = ""


def build_feature_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_question[str(record["question_id"])][str(record["strategy"])] = record

    rows = []
    for question_id, strategies in sorted(by_question.items()):
        if "direct" not in strategies:
            continue
        direct = strategies["direct"]
        cot = strategies.get("cot")
        sc = strategies.get("self_consistency_3")
        planner = strategies.get("planner_verifier")
        debate = strategies.get("debate_2_judge")
        oracle = _oracle_select(strategies)
        direct_answer = _answer(direct)
        cot_answer = _answer(cot)
        planner_answer = _answer(planner)
        question = str(direct.get("question", ""))
        options = {
            str(key): str(value)
            for key, value in (direct.get("options") or {}).items()
        }
        question_metadata = direct.get("question_metadata") or {}
        baseline_answers = [
            _answer(strategies.get(strategy))
            for strategy in BASELINE_STRATEGIES
        ]
        unique_answers = {answer for answer in baseline_answers if answer}
        rows.append(
            {
                "question_id": question_id,
                "gold": direct.get("gold"),
                "question_word_count": len(_tokens(question)),
                "question_char_count": len(question),
                "option_mean_word_count": _mean([len(_tokens(value)) for value in options.values()]),
                "option_max_char_count": max([len(value) for value in options.values()] or [0]),
                "option_min_char_count": min([len(value) for value in options.values()] or [0]),
                "option_max_similarity": _max_option_similarity(options),
                "question_option_max_overlap": _question_option_overlap(question, options, "max"),
                "question_option_mean_overlap": _question_option_overlap(question, options, "mean"),
                "has_negation_or_except": _has_negation_or_except(question),
                "subject_name": question_metadata.get("subject_name", ""),
                "topic_name": question_metadata.get("topic_name", ""),
                "direct_answer": direct_answer,
                "direct_correct": _correct(direct),
                "direct_confidence": _confidence(direct),
                "direct_cost_usd": _cost(direct),
                "direct_rationale_word_count": len(_tokens(str(direct.get("rationale", "")))),
                "cot_answer": cot_answer,
                "cot_correct": _correct(cot),
                "cot_confidence": _confidence(cot),
                "cot_cost_usd": _cost(cot),
                "direct_cot_agree": _agree(direct_answer, cot_answer),
                "self_consistency_answer": _answer(sc),
                "self_consistency_correct": _correct(sc),
                "self_consistency_cost_usd": _cost(sc),
                "self_consistency_entropy": _self_consistency_entropy(sc),
                "self_consistency_vote_margin": _self_consistency_vote_margin(sc),
                "self_consistency_stable_majority": bool(
                    (sc or {}).get("metadata", {}).get("stable_majority", False)
                ),
                "planner_verifier_answer": planner_answer,
                "planner_verifier_correct": _correct(planner),
                "planner_verifier_confidence": _confidence(planner),
                "planner_verifier_cost_usd": _cost(planner),
                "planner_changes_direct": _disagree(direct_answer, planner_answer),
                "debate_2_judge_answer": _answer(debate),
                "debate_2_judge_correct": _correct(debate),
                "debate_2_judge_confidence": _confidence(debate),
                "debate_2_judge_cost_usd": _cost(debate),
                "unique_answer_count": len(unique_answers),
                "answer_disagreement_count": max(0, len(unique_answers) - 1),
                "oracle_selected_strategy": str(oracle.get("strategy", "none")),
                "oracle_correct": _correct(oracle),
                "oracle_cost_usd": _cost(oracle),
                "should_collaborate": str(oracle.get("strategy", "none")) != "direct",
            }
        )
    return rows


def evaluate_candidate_rules(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _evaluate_rules(features, include_oracle=True)


def evaluate_train_test_rules(
    features: list[dict[str, Any]],
    train_fraction: float = 0.7,
    seed: int = 42,
    cost_budget_multiplier: float = 1.5,
) -> list[dict[str, Any]]:
    if not features:
        return []
    shuffled = list(features)
    random.Random(seed).shuffle(shuffled)
    split_at = int(len(shuffled) * train_fraction)
    split_at = min(max(split_at, 1), len(shuffled) - 1) if len(shuffled) > 1 else 1
    train = shuffled[:split_at]
    test = shuffled[split_at:]

    train_rows = _evaluate_rules(train, include_oracle=False)
    test_rows = {
        row["rule"]: row for row in _evaluate_rules(test, include_oracle=False)
    }
    train_direct = _rule_by_name(train, "always_direct")
    test_direct = _rule_by_name(test, "always_direct")
    train_oracle = _rule_by_name(train, "oracle_upper_bound", include_oracle=True)
    test_oracle = _rule_by_name(test, "oracle_upper_bound", include_oracle=True)
    selected_rule = _select_budgeted_rule(
        train_rows,
        direct_cost=float(train_direct["estimated_cost_usd"]),
        cost_budget_multiplier=cost_budget_multiplier,
    )
    output = []
    for train_row in train_rows:
        test_row = test_rows[train_row["rule"]]
        train_cost_multiplier = _safe_div(
            float(train_row["estimated_cost_usd"]),
            float(train_direct["estimated_cost_usd"]),
        )
        test_cost_multiplier = _safe_div(
            float(test_row["estimated_cost_usd"]),
            float(test_direct["estimated_cost_usd"]),
        )
        output.append(
            {
                "rule": train_row["rule"],
                "selected": train_row["rule"] == selected_rule,
                "deployable": train_row.get("deployable", True),
                "observation_mode": train_row.get("observation_mode", ""),
                "cost_note": train_row.get("cost_note", ""),
                "train_n": train_row["n"],
                "train_accuracy": train_row["accuracy"],
                "train_direct_accuracy": train_direct["accuracy"],
                "train_oracle_accuracy": train_oracle["accuracy"],
                "train_cost_usd": train_row["estimated_cost_usd"],
                "train_direct_cost_usd": train_direct["estimated_cost_usd"],
                "train_cost_multiplier": train_cost_multiplier,
                "train_escalation_rate": train_row["escalation_rate"],
                "test_n": test_row["n"],
                "test_accuracy": test_row["accuracy"],
                "test_direct_accuracy": test_direct["accuracy"],
                "test_oracle_accuracy": test_oracle["accuracy"],
                "test_cost_usd": test_row["estimated_cost_usd"],
                "test_direct_cost_usd": test_direct["estimated_cost_usd"],
                "test_cost_multiplier": test_cost_multiplier,
                "test_escalation_rate": test_row["escalation_rate"],
                "test_cost_per_correct": test_row["cost_per_correct"],
                "test_oracle_regret": float(test_oracle["accuracy"]) - float(test_row["accuracy"]),
                "test_retained_oracle_gain": _retained_gain(
                    direct_accuracy=float(test_direct["accuracy"]),
                    oracle_accuracy=float(test_oracle["accuracy"]),
                    policy_accuracy=float(test_row["accuracy"]),
                ),
            }
        )
    return output


def build_strata_rows(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not features:
        return []
    strata_specs = [
        ("direct_confidence", _confidence_bucket),
        ("direct_cot_agreement", lambda row: "agree" if row.get("direct_cot_agree") else "disagree"),
        ("self_consistency_entropy", _entropy_bucket),
        ("question_length", _question_length_bucket),
        ("negation_or_except", lambda row: "yes" if row.get("has_negation_or_except") else "no"),
        ("subject_name", lambda row: str(row.get("subject_name") or "unknown")),
    ]
    rows: list[dict[str, Any]] = []
    for feature_name, bucket_fn in strata_specs:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in features:
            buckets[bucket_fn(row)].append(row)
        for stratum, items in sorted(buckets.items()):
            rows.append(_summarize_stratum(feature_name, stratum, items))
    return rows


def _evaluate_rules(
    features: list[dict[str, Any]],
    include_oracle: bool,
) -> list[dict[str, Any]]:
    rules = [
        RuleSpec("always_direct", _always_direct),
        RuleSpec("always_planner_verifier", _always_planner_verifier),
        RuleSpec("always_debate_2_judge", _always_debate),
        RuleSpec("cot_disagree_to_planner", _cot_disagree_to_planner),
        RuleSpec(
            "planner_changes_direct",
            _planner_changes_direct,
            observation_mode="charged_planner_probe",
        ),
        RuleSpec("has_negation_to_planner", _flag_to_planner("has_negation_or_except")),
        RuleSpec(
            "answer_disagreement_to_planner",
            _numeric_at_least_to_planner(
                "unique_answer_count",
                2,
                executed_if_true=list(BASELINE_STRATEGIES),
                executed_if_false=list(BASELINE_STRATEGIES),
            ),
            deployable=False,
            observation_mode="posthoc_full_strategy",
            cost_note=(
                "requires observing all baseline strategy answers; retained for "
                "diagnostic analysis and excluded from learned-router selection"
            ),
        ),
    ]
    if include_oracle:
        rules.insert(
            0,
            RuleSpec(
                "oracle_upper_bound",
                _oracle_upper_bound,
                deployable=False,
                observation_mode="oracle_gold_labels",
                cost_note="uses gold correctness labels and is an upper bound only",
            ),
        )
    for threshold in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        for target in ("cot", "planner_verifier", "self_consistency_3", "debate_2_judge"):
            rules.append(
                RuleSpec(
                    f"direct_conf_lt_{threshold:.2f}_to_{target}",
                    _direct_conf_to_strategy(threshold, target),
                )
            )
    for threshold in (0.35, 0.50, 0.65):
        rules.append(
            RuleSpec(
                f"option_similarity_gt_{threshold:.2f}_to_planner",
                _numeric_greater_to_planner("option_max_similarity", threshold),
            )
        )
    for threshold in (12, 20, 30):
        rules.append(
            RuleSpec(
                f"question_words_gt_{threshold}_to_planner",
                _numeric_greater_to_planner("question_word_count", threshold),
            )
        )

    rows = []
    for spec in rules:
        decisions = [spec.fn(row) for row in features]
        n = len(decisions)
        correct = sum(1 for item in decisions if item["correct"])
        total_cost = sum(float(item["cost"]) for item in decisions)
        total_calls = sum(int(item["calls"]) for item in decisions)
        escalations = sum(1 for item in decisions if item["selected_strategy"] != "direct")
        rows.append(
            {
                "rule": spec.name,
                "deployable": spec.deployable,
                "observation_mode": spec.observation_mode,
                "cost_note": spec.cost_note,
                "n": n,
                "correct": correct,
                "accuracy": correct / n if n else 0.0,
                "estimated_cost_usd": total_cost,
                "api_calls": total_calls,
                "cost_per_correct": total_cost / correct if correct else None,
                "api_calls_per_correct": total_calls / correct if correct else None,
                "escalation_rate": escalations / n if n else 0.0,
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            -float(item["accuracy"]),
            float(item["estimated_cost_usd"]),
            str(item["rule"]),
        ),
    )


def _oracle_select(strategies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        record for name, record in strategies.items() if name in BASELINE_STRATEGIES
    ]
    correct = [record for record in candidates if _correct(record)]
    pool = correct or candidates
    if not pool:
        return {"strategy": "none", "correct": False, "estimated_cost_usd": 0.0}
    return min(
        pool,
        key=lambda record: (
            _cost(record),
            int(record.get("api_calls", 1) or 1),
            str(record.get("strategy", "")),
        ),
    )


def _always_direct(row: dict[str, Any]) -> dict[str, Any]:
    return _decision(row, "direct", ["direct"])


def _oracle_upper_bound(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_strategy": row["oracle_selected_strategy"],
        "executed_strategies": [row["oracle_selected_strategy"]],
        "correct": bool(row["oracle_correct"]),
        "cost": float(row["oracle_cost_usd"]),
        "calls": _strategy_calls([row["oracle_selected_strategy"]]),
    }


def _always_planner_verifier(row: dict[str, Any]) -> dict[str, Any]:
    return _decision(row, "planner_verifier", ["planner_verifier"])


def _always_debate(row: dict[str, Any]) -> dict[str, Any]:
    return _decision(row, "debate_2_judge", ["debate_2_judge"])


def _cot_disagree_to_planner(row: dict[str, Any]) -> dict[str, Any]:
    if row["direct_cot_agree"]:
        return _decision(row, "direct", ["direct", "cot"])
    return _decision(row, "planner_verifier", ["direct", "cot", "planner_verifier"])


def _planner_changes_direct(row: dict[str, Any]) -> dict[str, Any]:
    if row["planner_changes_direct"]:
        return _decision(row, "planner_verifier", ["direct", "planner_verifier"])
    return _decision(row, "direct", ["direct", "planner_verifier"])


def _direct_conf_to_planner(threshold: float):
    return _direct_conf_to_strategy(threshold, "planner_verifier")


def _direct_conf_to_strategy(threshold: float, target: str):
    def rule(row: dict[str, Any]) -> dict[str, Any]:
        if row["direct_confidence"] < threshold:
            return _decision(row, target, ["direct", target])
        return _decision(row, "direct", ["direct"])

    return rule


def _flag_to_planner(field: str):
    def rule(row: dict[str, Any]) -> dict[str, Any]:
        if row.get(field):
            return _decision(row, "planner_verifier", ["direct", "planner_verifier"])
        return _decision(row, "direct", ["direct"])

    return rule


def _numeric_at_least_to_planner(
    field: str,
    threshold: float,
    executed_if_true: list[str] | None = None,
    executed_if_false: list[str] | None = None,
):
    def rule(row: dict[str, Any]) -> dict[str, Any]:
        if float(row.get(field, 0.0) or 0.0) >= threshold:
            return _decision(
                row,
                "planner_verifier",
                executed_if_true or ["direct", "planner_verifier"],
            )
        return _decision(row, "direct", executed_if_false or ["direct"])

    return rule


def _numeric_greater_to_planner(field: str, threshold: float):
    def rule(row: dict[str, Any]) -> dict[str, Any]:
        if float(row.get(field, 0.0) or 0.0) > threshold:
            return _decision(row, "planner_verifier", ["direct", "planner_verifier"])
        return _decision(row, "direct", ["direct"])

    return rule


def _decision(
    row: dict[str, Any],
    selected_strategy: str,
    executed_strategies: list[str],
) -> dict[str, Any]:
    answer = _strategy_answer(row, selected_strategy)
    cost = sum(_strategy_cost(row, strategy) for strategy in executed_strategies)
    return {
        "selected_strategy": selected_strategy,
        "executed_strategies": executed_strategies,
        "correct": answer == row["gold"],
        "cost": cost,
        "calls": _strategy_calls(executed_strategies),
    }


def _strategy_answer(row: dict[str, Any], strategy: str) -> str | None:
    if strategy == "direct":
        return row["direct_answer"]
    return row.get(f"{_strategy_field_prefix(strategy)}_answer")


def _strategy_cost(row: dict[str, Any], strategy: str) -> float:
    return float(row.get(f"{_strategy_field_prefix(strategy)}_cost_usd", 0.0) or 0.0)


def _strategy_field_prefix(strategy: str) -> str:
    return {
        "self_consistency_3": "self_consistency",
    }.get(strategy, strategy)


def _strategy_calls(strategies: list[str]) -> int:
    calls_by_strategy = {
        "direct": 1,
        "cot": 1,
        "planner_verifier": 2,
        "debate_2_judge": 3,
        "self_consistency_3": 3,
    }
    return sum(calls_by_strategy.get(strategy, 1) for strategy in strategies)


def _rule_by_name(
    features: list[dict[str, Any]],
    rule_name: str,
    include_oracle: bool = False,
) -> dict[str, Any]:
    rows = _evaluate_rules(features, include_oracle=include_oracle)
    for row in rows:
        if row["rule"] == rule_name:
            return row
    raise KeyError(rule_name)


def _select_budgeted_rule(
    rows: list[dict[str, Any]],
    direct_cost: float,
    cost_budget_multiplier: float,
) -> str:
    budget = direct_cost * cost_budget_multiplier
    candidates = [
        row
        for row in rows
        if _is_deployable(row) and float(row["estimated_cost_usd"]) <= budget
    ]
    if not candidates:
        return "always_direct"
    selected = sorted(
        candidates,
        key=lambda item: (
            -float(item["accuracy"]),
            float(item["estimated_cost_usd"]),
            str(item["rule"]),
        ),
    )[0]
    return str(selected["rule"])


def _is_deployable(row: dict[str, Any]) -> bool:
    value = row.get("deployable", True)
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _retained_gain(
    direct_accuracy: float,
    oracle_accuracy: float,
    policy_accuracy: float,
) -> float | None:
    denominator = oracle_accuracy - direct_accuracy
    if denominator == 0:
        return None
    return (policy_accuracy - direct_accuracy) / denominator


def _answer(record: dict[str, Any] | None) -> str | None:
    if not record:
        return None
    answer = record.get("answer")
    return str(answer) if answer else None


def _correct(record: dict[str, Any] | None) -> bool:
    return bool(record and record.get("correct"))


def _confidence(record: dict[str, Any] | None) -> float:
    return _float((record or {}).get("confidence"))


def _cost(record: dict[str, Any] | None) -> float:
    return _float((record or {}).get("estimated_cost_usd"))


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _agree(left: str | None, right: str | None) -> bool:
    return bool(left and right and left == right)


def _disagree(left: str | None, right: str | None) -> bool:
    return bool(left and right and left != right)


def _self_consistency_entropy(record: dict[str, Any] | None) -> float:
    metadata = (record or {}).get("metadata", {})
    vote_counts = metadata.get("vote_counts", {})
    total = sum(int(count) for count in vote_counts.values())
    if total <= 0:
        return 0.0
    classes = len(vote_counts)
    if classes <= 1:
        return 0.0
    entropy = 0.0
    for count in vote_counts.values():
        probability = int(count) / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(classes)


def _self_consistency_vote_margin(record: dict[str, Any] | None) -> float:
    metadata = (record or {}).get("metadata", {})
    vote_counts = sorted(
        (int(count) for count in metadata.get("vote_counts", {}).values()),
        reverse=True,
    )
    total = sum(vote_counts)
    if total <= 0 or not vote_counts:
        return 0.0
    top = vote_counts[0]
    runner_up = vote_counts[1] if len(vote_counts) > 1 else 0
    return (top - runner_up) / total


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _max_option_similarity(options: dict[str, str]) -> float:
    token_sets = [set(_tokens(value)) for value in options.values()]
    if len(token_sets) < 2:
        return 0.0
    return max((_jaccard(left, right) for left, right in combinations(token_sets, 2)), default=0.0)


def _question_option_overlap(
    question: str,
    options: dict[str, str],
    mode: str,
) -> float:
    question_tokens = set(_tokens(question))
    overlaps = [
        _jaccard(question_tokens, set(_tokens(option)))
        for option in options.values()
    ]
    if not overlaps:
        return 0.0
    if mode == "max":
        return max(overlaps)
    return _mean(overlaps)


def _has_negation_or_except(question: str) -> bool:
    return bool(
        re.search(
            r"\b(not|except|least|false|incorrect|contraindicated|avoid|unlikely)\b",
            question,
            flags=re.IGNORECASE,
        )
    )


def _confidence_bucket(row: dict[str, Any]) -> str:
    confidence = float(row.get("direct_confidence", 0.0) or 0.0)
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.75:
        return "medium"
    return "low"


def _entropy_bucket(row: dict[str, Any]) -> str:
    entropy = float(row.get("self_consistency_entropy", 0.0) or 0.0)
    if entropy >= 0.67:
        return "high"
    if entropy >= 0.34:
        return "medium"
    return "low"


def _question_length_bucket(row: dict[str, Any]) -> str:
    word_count = float(row.get("question_word_count", 0.0) or 0.0)
    if word_count >= 30:
        return "long"
    if word_count >= 15:
        return "medium"
    return "short"


def _summarize_stratum(
    feature_name: str,
    stratum: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(items)
    direct_correct = sum(1 for row in items if row.get("direct_correct"))
    oracle_correct = sum(1 for row in items if row.get("oracle_correct"))
    best_collab_correct = 0
    for row in items:
        best_collab_correct += int(
            any(
                row.get(f"{strategy}_correct")
                for strategy in (
                    "cot",
                    "self_consistency",
                    "planner_verifier",
                    "debate_2_judge",
                )
            )
        )
    direct_accuracy = direct_correct / n if n else 0.0
    oracle_accuracy = oracle_correct / n if n else 0.0
    best_collab_accuracy = best_collab_correct / n if n else 0.0
    direct_cost = sum(float(row.get("direct_cost_usd", 0.0) or 0.0) for row in items)
    oracle_cost = sum(float(row.get("oracle_cost_usd", 0.0) or 0.0) for row in items)
    return {
        "feature": feature_name,
        "stratum": stratum,
        "n": n,
        "direct_accuracy": direct_accuracy,
        "best_collaboration_accuracy": best_collab_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "oracle_gain_over_direct": oracle_accuracy - direct_accuracy,
        "direct_cost_usd": direct_cost,
        "oracle_cost_usd": oracle_cost,
    }


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
