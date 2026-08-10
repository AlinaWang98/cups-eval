from pathlib import Path

from cups_eval.mechanistic import (
    build_case_type_rows,
    build_model_independent_feature_rows,
    build_router_diagnostic_rows,
    write_mechanistic_report,
)


def feature_row(
    qid,
    *,
    dataset="medmcqa",
    model="deepseek_v4_flash",
    direct_correct,
    cot_correct=False,
    planner_correct=False,
    debate_correct=False,
    confidence=0.9,
    words=20,
):
    return {
        "question_id": qid,
        "dataset": dataset,
        "model": model,
        "gold": "B",
        "direct_answer": "A" if not direct_correct else "B",
        "direct_correct": direct_correct,
        "direct_confidence": confidence,
        "direct_rationale_word_count": 12,
        "direct_cost_usd": 1.0,
        "cot_correct": cot_correct,
        "cot_cost_usd": 0.2,
        "self_consistency_correct": False,
        "self_consistency_cost_usd": 0.6,
        "planner_verifier_correct": planner_correct,
        "planner_verifier_cost_usd": 0.4,
        "debate_2_judge_correct": debate_correct,
        "debate_2_judge_cost_usd": 0.7,
        "question_word_count": words,
        "question_char_count": words * 5,
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


def prediction_row(qid, *, selected_strategy, selected_correct, dataset="medmcqa", model="deepseek_v4_flash"):
    return {
        "dataset": dataset,
        "model": model,
        "question_id": qid,
        "selected_strategy": selected_strategy,
        "selected_correct": selected_correct,
    }


def test_case_type_rows_count_rescue_and_harm_by_dataset():
    rows = [
        feature_row("q1", direct_correct=False, debate_correct=True),
        feature_row("q2", direct_correct=True, debate_correct=False),
        feature_row("q3", dataset="medqa", direct_correct=True, debate_correct=True),
    ]

    cases = build_case_type_rows(rows)
    lookup = {
        (row["dataset"], row["comparison"], row["case_type"]): row
        for row in cases
    }

    assert lookup[("medmcqa", "debate_2_judge", "rescue")]["n"] == 1
    assert lookup[("medmcqa", "debate_2_judge", "harm")]["n"] == 1
    assert lookup[("medqa", "debate_2_judge", "both_correct")]["n"] == 1
    assert lookup[("medmcqa", "oracle_any", "rescue")]["n"] == 1


def test_case_type_rows_omit_missing_collaboration_strategies():
    rows = [
        feature_row("q1", direct_correct=False, cot_correct=True),
        feature_row("q2", direct_correct=True, cot_correct=False),
    ]
    for row in rows:
        row["self_consistency_cost_usd"] = 0.0
        row["self_consistency_answer"] = ""

    cases = build_case_type_rows(rows)

    comparisons = {row["comparison"] for row in cases}
    assert "cot" in comparisons
    assert "self_consistency_3" not in comparisons


def test_router_diagnostics_labels_false_positive_and_false_negative():
    rows = [
        feature_row("q1", direct_correct=False, cot_correct=True),
        feature_row("q2", direct_correct=True, cot_correct=False),
        feature_row("q3", direct_correct=False, cot_correct=False),
        feature_row("q4", direct_correct=True, cot_correct=False),
    ]
    predictions = [
        prediction_row("q1", selected_strategy="direct", selected_correct=False),
        prediction_row("q2", selected_strategy="cot", selected_correct=False),
        prediction_row("q3", selected_strategy="cot", selected_correct=False),
        prediction_row("q4", selected_strategy="direct", selected_correct=True),
    ]

    diagnostics = build_router_diagnostic_rows(rows, predictions)
    lookup = {row["diagnostic_type"]: row for row in diagnostics}

    assert lookup["false_negative"]["n"] == 1
    assert lookup["false_positive"]["n"] == 2
    assert lookup["true_negative"]["n"] == 1


def test_model_independent_features_exclude_provider_specific_fields():
    rows = [
        feature_row("q1", direct_correct=False, cot_correct=True),
    ]

    features = build_model_independent_feature_rows(rows)

    assert features[0]["question_id"] == "q1"
    assert "direct_confidence" in features[0]
    assert "direct_cost_usd" not in features[0]
    assert "logprob" not in features[0]


def test_write_mechanistic_report_outputs_paper_figure_csvs(tmp_path):
    rows = [
        feature_row("q1", direct_correct=False, cot_correct=True, confidence=0.3),
        feature_row("q2", direct_correct=True, cot_correct=False, confidence=0.95),
    ]
    predictions = [
        prediction_row("q1", selected_strategy="cot", selected_correct=True),
        prediction_row("q2", selected_strategy="direct", selected_correct=True),
    ]

    write_mechanistic_report(rows, predictions, [], tmp_path)

    assert (tmp_path / "case_type_by_dataset.csv").exists()
    assert (tmp_path / "router_diagnostics.csv").exists()
    assert (tmp_path / "model_independent_features.csv").exists()
    assert (tmp_path / "fig1_cost_accuracy_points.csv").exists()
    assert (tmp_path / "fig3_rescue_harm_by_dataset.csv").exists()
    report = (tmp_path / "mechanistic_report.md").read_text(encoding="utf-8")
    assert "Mechanistic Analysis" in report
    assert "Missing Feature Diagnostics" in report
