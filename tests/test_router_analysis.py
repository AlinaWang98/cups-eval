import pytest

from cups_eval.router_analysis import (
    build_feature_rows,
    build_strata_rows,
    evaluate_candidate_rules,
    evaluate_train_test_rules,
)


def row(
    qid,
    strategy,
    answer,
    gold,
    cost,
    confidence=0.9,
    metadata=None,
    question="Which option is least likely?",
):
    return {
        "question_id": qid,
        "strategy": strategy,
        "answer": answer,
        "gold": gold,
        "correct": answer == gold,
        "confidence": confidence,
        "question": question,
        "options": {"A": "alpha", "B": "bravo", "C": "charlie", "D": "delta"},
        "question_metadata": {"subject_name": "Medicine", "topic_name": "Basics"},
        "parse_ok": True,
        "api_calls": 1,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "estimated_cost_usd": cost,
        "metadata": metadata or {},
    }


def test_build_feature_rows_extracts_oracle_labels_and_disagreement_features():
    records = [
        row("q1", "direct", "A", "B", 0.01, confidence=0.95),
        row("q1", "cot", "B", "B", 0.02),
        row(
            "q1",
            "self_consistency_3",
            "B",
            "B",
            0.06,
            metadata={"vote_counts": {"A": 1, "B": 2}, "stable_majority": True},
        ),
        row(
            "q1",
            "planner_verifier",
            "B",
            "B",
            0.03,
            metadata={"planner_answer": "A", "verifier_answer": "B"},
        ),
        row("q1", "debate_2_judge", "B", "B", 0.05),
        row("q2", "direct", "C", "C", 0.01),
        row("q2", "cot", "C", "C", 0.02),
        row(
            "q2",
            "self_consistency_3",
            "C",
            "C",
            0.06,
            metadata={"vote_counts": {"C": 3}, "stable_majority": True},
        ),
        row("q2", "planner_verifier", "C", "C", 0.03),
        row("q2", "debate_2_judge", "C", "C", 0.05),
    ]

    features = build_feature_rows(records)

    q1 = next(item for item in features if item["question_id"] == "q1")
    q2 = next(item for item in features if item["question_id"] == "q2")
    assert q1["oracle_selected_strategy"] == "cot"
    assert q1["should_collaborate"] is True
    assert q1["direct_cot_agree"] is False
    assert q1["planner_changes_direct"] is True
    assert 0.0 < q1["self_consistency_entropy"] < 1.0
    assert q1["question_word_count"] > 0
    assert q1["has_negation_or_except"] is True
    assert q1["unique_answer_count"] == 2
    assert q1["self_consistency_vote_margin"] == 1 / 3
    assert q1["subject_name"] == "Medicine"
    assert q2["oracle_selected_strategy"] == "direct"
    assert q2["should_collaborate"] is False


def test_evaluate_candidate_rules_scores_cost_aware_policies():
    records = [
        row("q1", "direct", "A", "B", 0.01, confidence=0.95),
        row("q1", "cot", "B", "B", 0.02),
        row("q1", "planner_verifier", "B", "B", 0.03),
        row("q1", "debate_2_judge", "B", "B", 0.05),
        row("q2", "direct", "C", "C", 0.01),
        row("q2", "cot", "C", "C", 0.02),
        row("q2", "planner_verifier", "C", "C", 0.03),
        row("q2", "debate_2_judge", "C", "C", 0.05),
    ]
    features = build_feature_rows(records)

    results = evaluate_candidate_rules(features)
    by_name = {item["rule"]: item for item in results}

    assert by_name["always_direct"]["accuracy"] == 0.5
    assert by_name["oracle_upper_bound"]["accuracy"] == 1.0
    assert by_name["oracle_upper_bound"]["estimated_cost_usd"] == 0.03
    assert by_name["cot_disagree_to_planner"]["accuracy"] == 1.0
    assert by_name["cot_disagree_to_planner"]["estimated_cost_usd"] == 0.09
    assert by_name["cot_disagree_to_planner"]["escalation_rate"] == 0.5


def test_self_consistency_rule_uses_self_consistency_answer_and_cost():
    records = [
        row("q1", "direct", "A", "B", 0.01, confidence=0.40),
        row("q1", "self_consistency_3", "B", "B", 0.06),
    ]
    features = build_feature_rows(records)

    results = evaluate_candidate_rules(features)
    by_name = {item["rule"]: item for item in results}
    low_conf_sc = by_name["direct_conf_lt_0.95_to_self_consistency_3"]

    assert low_conf_sc["accuracy"] == 1.0
    assert low_conf_sc["estimated_cost_usd"] == pytest.approx(0.07)
    assert low_conf_sc["api_calls"] == 4


def test_posthoc_answer_disagreement_rule_is_diagnostic_and_fully_costed():
    records = [
        row("q1", "direct", "A", "B", 0.01),
        row("q1", "cot", "A", "B", 0.02),
        row("q1", "self_consistency_3", "A", "B", 0.03),
        row("q1", "planner_verifier", "B", "B", 0.04),
        row("q1", "debate_2_judge", "C", "B", 0.05),
    ]
    features = build_feature_rows(records)

    results = evaluate_candidate_rules(features)
    by_name = {item["rule"]: item for item in results}
    disagreement = by_name["answer_disagreement_to_planner"]

    assert disagreement["deployable"] is False
    assert disagreement["observation_mode"] == "posthoc_full_strategy"
    assert disagreement["estimated_cost_usd"] == pytest.approx(0.15)
    assert disagreement["api_calls"] == 10


def test_evaluate_train_test_rules_selects_deployable_rule_not_oracle():
    records = [
        row("q1", "direct", "A", "B", 0.01),
        row("q1", "cot", "B", "B", 0.02),
        row("q1", "planner_verifier", "B", "B", 0.03),
        row("q1", "debate_2_judge", "B", "B", 0.05),
        row("q2", "direct", "C", "C", 0.01),
        row("q2", "cot", "C", "C", 0.02),
        row("q2", "planner_verifier", "C", "C", 0.03),
        row("q2", "debate_2_judge", "C", "C", 0.05),
        row("q3", "direct", "A", "B", 0.01),
        row("q3", "cot", "B", "B", 0.02),
        row("q3", "planner_verifier", "B", "B", 0.03),
        row("q3", "debate_2_judge", "B", "B", 0.05),
        row("q4", "direct", "D", "D", 0.01),
        row("q4", "cot", "D", "D", 0.02),
        row("q4", "planner_verifier", "D", "D", 0.03),
        row("q4", "debate_2_judge", "D", "D", 0.05),
    ]
    features = build_feature_rows(records)

    results = evaluate_train_test_rules(features, train_fraction=0.5, seed=7)
    selected = [item for item in results if item["selected"]]

    assert len(selected) == 1
    assert selected[0]["rule"] != "oracle_upper_bound"
    assert selected[0]["deployable"] is True
    assert selected[0]["train_n"] == 2
    assert selected[0]["test_n"] == 2
    assert "test_accuracy" in selected[0]
    assert "test_retained_oracle_gain" in selected[0]
    assert "test_cost_multiplier" in selected[0]


def test_evaluate_train_test_rules_selects_budgeted_gain_rule():
    records = []
    for index in range(8):
        qid = f"q{index}"
        direct_is_wrong = qid in {"q0", "q6"}
        direct_answer = "A" if direct_is_wrong else "B"
        confidence = 0.40 if direct_is_wrong else 0.95
        records.extend(
            [
                row(qid, "direct", direct_answer, "B", 1.0, confidence=confidence),
                row(qid, "cot", "B", "B", 0.1),
                row(qid, "planner_verifier", "B", "B", 1.8),
                row(qid, "debate_2_judge", "B", "B", 2.5),
            ]
        )
    features = build_feature_rows(records)

    results = evaluate_train_test_rules(
        features,
        train_fraction=0.5,
        seed=1,
        cost_budget_multiplier=1.5,
    )
    selected = next(item for item in results if item["selected"])

    assert selected["rule"].startswith("direct_conf_lt_")
    assert selected["test_accuracy"] > selected["test_direct_accuracy"]
    assert selected["test_cost_multiplier"] <= 1.5
    assert selected["test_retained_oracle_gain"] == 1.0


def test_evaluate_train_test_rules_does_not_select_diagnostic_rule():
    records = []
    for index in range(8):
        qid = f"q{index}"
        direct_answer = "A"
        planner_answer = "B"
        debate_answer = "C" if index % 2 == 0 else "A"
        records.extend(
            [
                row(qid, "direct", direct_answer, "B", 1.0),
                row(qid, "cot", direct_answer, "B", 0.1),
                row(qid, "self_consistency_3", direct_answer, "B", 0.2),
                row(qid, "planner_verifier", planner_answer, "B", 0.3),
                row(qid, "debate_2_judge", debate_answer, "B", 0.4),
            ]
        )
    features = build_feature_rows(records)

    results = evaluate_train_test_rules(
        features,
        train_fraction=0.5,
        seed=1,
        cost_budget_multiplier=10.0,
    )
    selected = next(item for item in results if item["selected"])

    assert selected["rule"] != "answer_disagreement_to_planner"
    assert selected["deployable"] is True


def test_build_strata_rows_handles_empty_and_missing_fields():
    assert build_strata_rows([]) == []

    features = build_feature_rows(
        [
            row("q1", "direct", "A", "B", 1.0, confidence=0.4),
            row("q1", "planner_verifier", "B", "B", 0.2),
            row("q2", "direct", "C", "C", 1.0, confidence=0.95),
            row("q2", "planner_verifier", "C", "C", 0.2),
        ]
    )

    strata = build_strata_rows(features)

    assert strata
    assert all("oracle_gain_over_direct" in item for item in strata)
    assert any(item["stratum"] == "low" for item in strata)
