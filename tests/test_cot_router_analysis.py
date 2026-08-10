from cups_eval.cot_router_analysis import (
    build_cot_first_curve,
    evaluate_cot_first_oracle,
    write_cot_router_report,
)


def feature_row(
    qid,
    *,
    direct_correct=False,
    cot_correct=False,
    planner_correct=False,
    debate_correct=False,
    cot_confidence=0.9,
):
    return {
        "question_id": qid,
        "dataset": "medmcqa",
        "model": "glm_5_2",
        "gold": "B",
        "direct_answer": "A",
        "direct_correct": direct_correct,
        "direct_confidence": 0.8,
        "direct_cost_usd": 1.0,
        "cot_answer": "B" if cot_correct else "A",
        "cot_correct": cot_correct,
        "cot_confidence": cot_confidence,
        "cot_cost_usd": 1.0,
        "planner_verifier_correct": planner_correct,
        "planner_verifier_cost_usd": 2.0,
        "debate_2_judge_correct": debate_correct,
        "debate_2_judge_cost_usd": 3.0,
    }


def test_cot_first_oracle_charges_cot_plus_cheapest_successful_rescue():
    rows = [
        feature_row("q1", cot_correct=True, planner_correct=True),
        feature_row("q2", cot_correct=False, direct_correct=True, debate_correct=True),
        feature_row("q3", cot_correct=False, planner_correct=True, debate_correct=True),
        feature_row("q4", cot_correct=False),
    ]

    metrics = evaluate_cot_first_oracle(rows)

    assert metrics["accuracy"] == 3 / 4
    assert metrics["estimated_cost_usd"] == 7.0
    assert metrics["cost_multiplier"] == 7.0 / 4.0
    assert metrics["escalation_rate"] == 0.5


def test_cot_confidence_threshold_routes_from_cot_baseline():
    rows = [
        feature_row("q1", cot_correct=False, debate_correct=True, cot_confidence=0.4),
        feature_row("q2", cot_correct=True, debate_correct=False, cot_confidence=0.95),
    ]

    curve = build_cot_first_curve(rows)
    by_policy = {row["policy"]: row for row in curve}
    policy = by_policy["cot_conf_lt_0.50_to_debate_2_judge"]

    assert by_policy["always_cot"]["accuracy"] == 0.5
    assert policy["accuracy"] == 1.0
    assert policy["cost_multiplier"] == 5.0 / 2.0
    assert policy["escalation_rate"] == 0.5


def test_cot_router_report_writes_claim_memo_and_curve(tmp_path):
    rows = [
        feature_row("q1", cot_correct=False, debate_correct=True, cot_confidence=0.4),
        feature_row("q2", cot_correct=True, debate_correct=False, cot_confidence=0.95),
    ]

    write_cot_router_report(rows, tmp_path)

    assert (tmp_path / "cot_first_routing_curve.csv").exists()
    assert (tmp_path / "cot_oracle_opportunity_by_dataset_model.csv").exists()
    assert (tmp_path / "paper_main_table.csv").exists()
    memo = (tmp_path / "claim_memo.md").read_text(encoding="utf-8")
    assert "CoT-first" in memo
