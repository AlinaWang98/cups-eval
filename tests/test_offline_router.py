from cups_eval.offline_router import (
    Predicate,
    RouterRule,
    available_collab_strategies,
    build_routing_curve,
    cross_validate_router,
    evaluate_direct_first_oracle,
    evaluate_policy,
    train_greedy_router,
    write_offline_router_report,
)


def feature_row(
    qid,
    *,
    direct_correct,
    cot_correct=False,
    planner_correct=False,
    debate_correct=False,
    confidence=0.9,
    rationale_words=10,
    dataset="medmcqa",
    model="deepseek_v4_flash",
):
    return {
        "question_id": qid,
        "dataset": dataset,
        "model": model,
        "gold": "B",
        "direct_answer": "A" if not direct_correct else "B",
        "direct_correct": direct_correct,
        "direct_confidence": confidence,
        "direct_rationale_word_count": rationale_words,
        "direct_cost_usd": 1.0,
        "cot_correct": cot_correct,
        "cot_cost_usd": 0.2,
        "self_consistency_correct": False,
        "self_consistency_cost_usd": 0.6,
        "planner_verifier_correct": planner_correct,
        "planner_verifier_cost_usd": 0.4,
        "debate_2_judge_correct": debate_correct,
        "debate_2_judge_cost_usd": 0.7,
        "question_word_count": 10,
        "question_char_count": 50,
        "option_mean_word_count": 2,
        "option_max_char_count": 10,
        "option_min_char_count": 4,
        "option_max_similarity": 0.1,
        "question_option_max_overlap": 0.1,
        "question_option_mean_overlap": 0.05,
        "has_negation_or_except": False,
        "subject_name": "Medicine",
        "topic_name": "Basics",
    }


def test_direct_first_oracle_charges_direct_plus_successful_escalation():
    rows = [
        feature_row("q1", direct_correct=False, cot_correct=True),
        feature_row("q2", direct_correct=True, debate_correct=True),
        feature_row("q3", direct_correct=False),
    ]

    metrics = evaluate_direct_first_oracle(rows)

    assert metrics["accuracy"] == 2 / 3
    assert metrics["estimated_cost_usd"] == 3.2
    assert metrics["cost_multiplier"] == 3.2 / 3.0
    assert metrics["escalation_rate"] == 1 / 3


def test_evaluate_policy_applies_first_matching_rule_and_counts_cost():
    rows = [
        feature_row("q1", direct_correct=False, cot_correct=True, confidence=0.4),
        feature_row("q2", direct_correct=True, cot_correct=False, confidence=0.95),
    ]
    rules = [
        RouterRule(Predicate("low_conf", "direct_confidence", "<=", 0.5), "cot")
    ]

    metrics = evaluate_policy(rows, rules)

    assert metrics["accuracy"] == 1.0
    assert metrics["estimated_cost_usd"] == 2.2
    assert metrics["cost_multiplier"] == 1.1
    assert metrics["escalation_rate"] == 0.5


def test_train_greedy_router_learns_budgeted_low_confidence_rescue():
    rows = []
    for index in range(20):
        rows.append(
            feature_row(
                f"low-{index}",
                direct_correct=False,
                cot_correct=True,
                confidence=0.4,
            )
        )
        rows.append(
            feature_row(
                f"high-{index}",
                direct_correct=True,
                cot_correct=False,
                confidence=0.95,
            )
        )

    rules = train_greedy_router(
        rows,
        budget_multiplier=1.2,
        max_rules=2,
        min_support=5,
    )

    assert rules
    assert rules[0].predicate.feature == "direct_confidence"
    assert rules[0].target_strategy == "cot"
    metrics = evaluate_policy(rows, rules)
    assert metrics["accuracy"] == 1.0
    assert metrics["cost_multiplier"] <= 1.2


def test_cross_validate_router_returns_out_of_fold_predictions():
    rows = [
        feature_row(
            f"q{index}",
            direct_correct=index % 3 != 0,
            cot_correct=index % 3 == 0,
            confidence=0.4 if index % 3 == 0 else 0.95,
        )
        for index in range(30)
    ]

    result = cross_validate_router(
        rows,
        folds=3,
        budget_multiplier=1.3,
        max_rules=2,
        min_support=3,
    )

    assert len(result["folds"]) == 3
    assert len(result["predictions"]) == len(rows)
    assert result["aggregate"]["accuracy"] > evaluate_policy(rows, [])["accuracy"]


def test_empty_cross_validation_result_can_be_reported(tmp_path):
    result = cross_validate_router([], folds=3)

    write_offline_router_report([], result, tmp_path, budget_multiplier=1.5)

    report = (tmp_path / "offline_router_report.md").read_text(encoding="utf-8")
    assert "Scope: 0 feature rows" in report
    assert "learned rule-list" in report


def test_report_writes_routing_curve_outputs(tmp_path):
    rows = [
        feature_row("q1", direct_correct=False, cot_correct=True, confidence=0.3),
        feature_row("q2", direct_correct=True, cot_correct=False, confidence=0.95),
        feature_row("q3", direct_correct=False, cot_correct=False, confidence=0.95),
        feature_row("q4", direct_correct=True, cot_correct=True, confidence=0.3),
    ]
    result = cross_validate_router(rows, folds=2, min_support=1)

    write_offline_router_report(rows, result, tmp_path, budget_multiplier=1.5)

    assert (tmp_path / "routing_curve.csv").exists()
    assert (tmp_path / "routing_pareto.csv").exists()
    report = (tmp_path / "offline_router_report.md").read_text(encoding="utf-8")
    assert "Retained oracle gain" in report
    assert "Routing Curve" in report


def test_routing_curve_reports_rog_and_oracle_cost_overhead_for_confidence_baseline():
    rows = [
        feature_row("q1", direct_correct=False, cot_correct=True, confidence=0.3),
        feature_row("q2", direct_correct=True, cot_correct=True, confidence=0.3),
        feature_row("q3", direct_correct=False, cot_correct=False, confidence=0.95),
    ]

    curve = build_routing_curve(rows, folds=2)
    by_policy = {row["policy"]: row for row in curve}

    direct = by_policy["direct"]
    oracle = by_policy["direct_first_oracle"]
    confidence = by_policy["confidence_lt_0.50_to_cot"]

    assert direct["accuracy"] == 1 / 3
    assert oracle["accuracy"] == 2 / 3
    assert confidence["accuracy"] == 2 / 3
    assert confidence["retained_oracle_gain"] == 1.0
    assert confidence["cost_overhead_vs_oracle"] > 1.0


def test_routing_curve_omits_collaboration_strategies_missing_from_rows():
    rows = [
        feature_row("q1", direct_correct=False, cot_correct=True, planner_correct=True),
        feature_row("q2", direct_correct=True, cot_correct=False, planner_correct=False),
    ]
    for row in rows:
        row["self_consistency_cost_usd"] = 0.0
        row["self_consistency_answer"] = ""

    curve = build_routing_curve(rows)

    policies = {row["policy"] for row in curve}
    targets = {row["target_strategy"] for row in curve}
    assert "self_consistency_3" not in available_collab_strategies(rows)
    assert "always_self_consistency_3" not in policies
    assert "self_consistency_3" not in targets


def test_calibrated_logistic_curve_uses_out_of_fold_rescue_signal():
    rows = []
    for index in range(30):
        rows.append(
            feature_row(
                f"rescue-{index}",
                direct_correct=False,
                cot_correct=True,
                confidence=0.95,
                rationale_words=80,
            )
        )
        rows.append(
            feature_row(
                f"safe-{index}",
                direct_correct=True,
                cot_correct=False,
                confidence=0.95,
                rationale_words=5,
            )
        )

    curve = build_routing_curve(rows, folds=3)
    calibrated = [
        row for row in curve
        if row["policy_family"] == "calibrated_logistic"
        and row["target_strategy"] == "cot"
    ]

    assert calibrated
    assert max(row["accuracy"] for row in calibrated) > 0.5
    assert any(row["retained_oracle_gain"] > 0 for row in calibrated)
