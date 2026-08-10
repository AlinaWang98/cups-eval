from pathlib import Path

from cups_eval.cli import main
from cups_eval.cups_policy import (
    aggregate_predictions,
    build_cups_predictions,
    build_cups_curve,
    cross_validate_cups,
    select_strategy,
    write_cups_report,
)


def feature_row(
    qid,
    *,
    direct_correct=False,
    cot_correct=False,
    planner_correct=False,
    debate_correct=False,
    confidence=0.8,
    rationale_words=10,
    dataset="gsm8k",
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
        "cot_answer": "B" if cot_correct else "A",
        "cot_correct": cot_correct,
        "cot_confidence": confidence,
        "cot_cost_usd": 0.9,
        "planner_verifier_answer": "B" if planner_correct else "A",
        "planner_verifier_correct": planner_correct,
        "planner_verifier_cost_usd": 2.0,
        "debate_2_judge_answer": "B" if debate_correct else "A",
        "debate_2_judge_correct": debate_correct,
        "debate_2_judge_cost_usd": 3.0,
        "question_word_count": 12,
        "question_char_count": 70,
        "option_mean_word_count": 2,
        "option_max_char_count": 10,
        "option_min_char_count": 4,
        "option_max_similarity": 0.1,
        "question_option_max_overlap": 0.1,
        "question_option_mean_overlap": 0.05,
        "has_negation_or_except": False,
        "subject_name": "math",
        "topic_name": "arithmetic",
        "self_consistency_entropy": 0.0,
        "self_consistency_vote_margin": 0.0,
        "answer_disagreement_count": 0,
    }


def test_select_strategy_penalizes_cost_and_uses_deterministic_tie_break():
    row = feature_row("q1")

    tied = {
        "direct": 0.80,
        "cot": 0.80,
        "planner_verifier": 0.80,
        "debate_2_judge": 0.80,
    }
    assert select_strategy(row, tied, lambda_value=0.0) == "cot"

    planner_wins_when_lambda_is_small = {
        "direct": 0.70,
        "cot": 0.72,
        "planner_verifier": 0.90,
        "debate_2_judge": 0.86,
    }
    assert select_strategy(row, planner_wins_when_lambda_is_small, lambda_value=0.02) == "planner_verifier"
    assert select_strategy(row, planner_wins_when_lambda_is_small, lambda_value=0.30) == "cot"


def test_select_strategy_can_use_train_fold_cost_ratios_to_avoid_cost_leakage():
    row = feature_row("q1")
    row["cot_cost_usd"] = 0.05
    probabilities = {"direct": 0.80, "cot": 0.81}

    selected = select_strategy(
        row,
        probabilities,
        lambda_value=0.10,
        estimated_cost_ratios={"direct": 1.0, "cot": 10.0},
    )

    assert selected == "direct"


def test_build_cups_predictions_aggregates_mixed_strategy_costs():
    rows = [
        feature_row("q1", direct_correct=True, cot_correct=True, planner_correct=False),
        feature_row("q2", direct_correct=False, cot_correct=False, planner_correct=True),
    ]
    probabilities = {
        "q1": {"direct": 0.9, "cot": 0.95, "planner_verifier": 0.2},
        "q2": {"direct": 0.2, "cot": 0.3, "planner_verifier": 0.9},
    }

    predictions = build_cups_predictions(rows, probabilities, lambda_value=0.02)
    metrics = aggregate_predictions(predictions)

    assert [row["selected_strategy"] for row in predictions] == ["cot", "planner_verifier"]
    assert metrics["accuracy"] == 1.0
    assert metrics["estimated_cost_usd"] == 2.9
    assert metrics["cost_multiplier"] == 1.45
    assert metrics["relative_cost_per_correct"] == 1.45


def test_cross_validate_cups_returns_out_of_fold_predictions_for_each_lambda():
    rows = []
    for index in range(30):
        rows.append(
            feature_row(
                f"easy-{index}",
                direct_correct=True,
                cot_correct=True,
                planner_correct=False,
                confidence=0.95,
                rationale_words=5,
            )
        )
        rows.append(
            feature_row(
                f"hard-{index}",
                direct_correct=False,
                cot_correct=False,
                planner_correct=True,
                confidence=0.35,
                rationale_words=80,
            )
        )

    result = cross_validate_cups(rows, folds=3, lambdas=(0.0, 0.10, 0.50))

    assert len(result["folds"]) == 3
    assert len(result["predictions"]) == len(rows) * 3
    assert {row["lambda"] for row in result["curve"] if row["policy_family"] == "cups"} == {0.0, 0.1, 0.5}
    assert max(row["accuracy"] for row in result["curve"] if row["policy_family"] == "cups") > 0.5


def test_build_cups_curve_includes_baselines_oracle_and_pareto_rows():
    rows = [
        feature_row("q1", direct_correct=True, cot_correct=True),
        feature_row("q2", direct_correct=False, cot_correct=True),
        feature_row("q3", direct_correct=False, cot_correct=False, planner_correct=True),
    ]
    probabilities = {
        "q1": {"direct": 0.8, "cot": 0.9, "planner_verifier": 0.2},
        "q2": {"direct": 0.3, "cot": 0.8, "planner_verifier": 0.2},
        "q3": {"direct": 0.2, "cot": 0.3, "planner_verifier": 0.9},
    }

    result = build_cups_curve(rows, probabilities, lambdas=(0.0, 0.2))
    policies = {row["policy"] for row in result["curve"]}

    assert {"direct", "always_cot", "always_planner_verifier", "cups_lambda_0", "cups_lambda_0.2"} <= policies
    assert result["pareto"]
    assert any(row["policy_family"] == "oracle" for row in result["curve"])


def test_write_cups_report_outputs_method_files(tmp_path: Path):
    rows = [
        feature_row("q1", direct_correct=True, cot_correct=True),
        feature_row("q2", direct_correct=False, cot_correct=False, planner_correct=True),
    ]
    result = cross_validate_cups(rows, folds=2, lambdas=(0.0, 0.2))

    write_cups_report(rows, result, tmp_path)

    assert (tmp_path / "cups_curve.csv").exists()
    assert (tmp_path / "cups_pareto.csv").exists()
    assert (tmp_path / "cups_predictions.csv").exists()
    assert (tmp_path / "cups_summary.json").exists()
    report = (tmp_path / "cups_policy_report.md").read_text(encoding="utf-8")
    assert "Cost-aware Utility Policy Selection" in report
    assert "always-CoT" in report


def test_cups_cli_writes_report_from_batch_features(tmp_path: Path):
    batch_dir = tmp_path / "batch"
    run_dir = batch_dir / "runs" / "gsm8k" / "model_a" / "run1"
    run_dir.mkdir(parents=True)
    rows = [
        feature_row("q1", direct_correct=True, cot_correct=True),
        feature_row("q2", direct_correct=False, planner_correct=True, confidence=0.3),
    ]
    _write_csv(run_dir / "router_features.csv", rows)
    output_dir = tmp_path / "cups"

    exit_code = main(
        [
            "cups",
            "--batch-dir",
            str(batch_dir),
            "--output-dir",
            str(output_dir),
            "--folds",
            "2",
            "--lambdas",
            "0,0.2",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "cups_policy_report.md").exists()
    assert (output_dir / "cups_curve.csv").exists()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
