from __future__ import annotations

import csv
from pathlib import Path

from cups_eval.cross_domain_ci import (
    aggregate_policy_rows,
    load_policy_records,
    write_cross_domain_ci_report,
)


def test_load_policy_records_includes_methods_oracle_and_learned(tmp_path: Path) -> None:
    batch_dir = _make_batch_fixture(tmp_path)
    records = load_policy_records(
        batch_dir=batch_dir,
        scope="toy",
        family="toy_family",
        offline_predictions_path=batch_dir / "analysis" / "offline_router_predictions.csv",
    )

    by_method = {record.method for record in records}

    assert {
        "direct",
        "cot",
        "planner_verifier",
        "debate_2_judge",
        "direct_first_oracle_any",
        "learned_rule_list",
    } <= by_method
    assert sum(1 for record in records if record.method == "direct") == 2
    assert sum(1 for record in records if record.method == "direct_first_oracle_any") == 2
    assert sum(1 for record in records if record.method == "learned_rule_list") == 2


def test_load_policy_records_can_filter_to_one_dataset(tmp_path: Path) -> None:
    batch_dir = _make_batch_fixture(tmp_path)
    extra_run = batch_dir / "runs" / "dataset_b" / "model_a" / "run1"
    extra_run.mkdir(parents=True)
    _write_csv(
        extra_run / "item_metrics.csv",
        [
            {
                "question_id": "q3",
                "strategy": "direct",
                "correct": "True",
                "estimated_cost_usd": "1.0",
            }
        ],
    )

    records = load_policy_records(
        batch_dir=batch_dir,
        scope="toy",
        family="toy_family",
        dataset_filter="dataset_a",
    )

    assert {record.dataset for record in records} == {"dataset_a"}


def test_aggregate_policy_rows_reports_relative_cost_per_correct(tmp_path: Path) -> None:
    batch_dir = _make_batch_fixture(tmp_path)
    records = load_policy_records(
        batch_dir=batch_dir,
        scope="toy",
        family="toy_family",
        offline_predictions_path=batch_dir / "analysis" / "offline_router_predictions.csv",
    )

    rows = aggregate_policy_rows(records)
    by_method = {row["method"]: row for row in rows}

    assert by_method["direct"]["accuracy"] == 0.5
    assert by_method["direct"]["cost_x_direct"] == 1.0
    assert by_method["direct"]["relative_cost_per_correct"] == 2.0
    assert by_method["cot"]["accuracy"] == 1.0
    assert by_method["cot"]["cost_x_direct"] == 1.0
    assert by_method["cot"]["relative_cost_per_correct"] == 1.0
    assert by_method["direct_first_oracle_any"]["accuracy"] == 1.0
    assert by_method["direct_first_oracle_any"]["cost_x_direct"] == 1.5


def test_write_cross_domain_ci_report_outputs_markdown_and_csv(tmp_path: Path) -> None:
    batch_dir = _make_batch_fixture(tmp_path)
    output_dir = tmp_path / "reports"

    write_cross_domain_ci_report(
        specs=[
            {
                "scope": "toy",
                "family": "toy_family",
                "batch_dir": str(batch_dir),
                "offline_predictions_path": str(
                    batch_dir / "analysis" / "offline_router_predictions.csv"
                ),
            }
        ],
        output_dir=output_dir,
        iterations=50,
        seed=7,
    )

    csv_path = output_dir / "cross_domain_ci_table.csv"
    md_path = output_dir / "cross_domain_ci_table.md"
    assert csv_path.exists()
    assert md_path.exists()

    rows = list(csv.DictReader(csv_path.open()))
    direct = next(row for row in rows if row["method"] == "direct")
    assert direct["accuracy_ci_low"] != ""
    assert direct["relative_cpc_ci_high"] != ""
    assert "# Cross-Domain Bootstrap CI Table" in md_path.read_text(encoding="utf-8")


def _make_batch_fixture(tmp_path: Path) -> Path:
    batch_dir = tmp_path / "batch"
    run_dir = batch_dir / "runs" / "dataset_a" / "model_a" / "run1"
    run_dir.mkdir(parents=True)
    _write_csv(
        run_dir / "item_metrics.csv",
        [
            {
                "question_id": "q1",
                "strategy": "direct",
                "correct": "True",
                "estimated_cost_usd": "1.0",
            },
            {
                "question_id": "q1",
                "strategy": "cot",
                "correct": "True",
                "estimated_cost_usd": "1.0",
            },
            {
                "question_id": "q1",
                "strategy": "planner_verifier",
                "correct": "True",
                "estimated_cost_usd": "2.0",
            },
            {
                "question_id": "q1",
                "strategy": "debate_2_judge",
                "correct": "True",
                "estimated_cost_usd": "3.0",
            },
            {
                "question_id": "q2",
                "strategy": "direct",
                "correct": "False",
                "estimated_cost_usd": "1.0",
            },
            {
                "question_id": "q2",
                "strategy": "cot",
                "correct": "True",
                "estimated_cost_usd": "1.0",
            },
            {
                "question_id": "q2",
                "strategy": "planner_verifier",
                "correct": "True",
                "estimated_cost_usd": "2.0",
            },
            {
                "question_id": "q2",
                "strategy": "debate_2_judge",
                "correct": "False",
                "estimated_cost_usd": "3.0",
            },
        ],
    )
    _write_csv(
        run_dir / "router_features.csv",
        [
            {
                "question_id": "q1",
                "oracle_correct": "True",
                "oracle_selected_strategy": "direct",
                "oracle_cost_usd": "1.0",
                "direct_correct": "True",
                "direct_cost_usd": "1.0",
                "cot_correct": "True",
                "cot_cost_usd": "1.0",
                "planner_verifier_correct": "True",
                "planner_verifier_cost_usd": "2.0",
                "debate_2_judge_correct": "True",
                "debate_2_judge_cost_usd": "3.0",
            },
            {
                "question_id": "q2",
                "oracle_correct": "True",
                "oracle_selected_strategy": "cot",
                "oracle_cost_usd": "1.0",
                "direct_correct": "False",
                "direct_cost_usd": "1.0",
                "cot_correct": "True",
                "cot_cost_usd": "1.0",
                "planner_verifier_correct": "True",
                "planner_verifier_cost_usd": "2.0",
                "debate_2_judge_correct": "False",
                "debate_2_judge_cost_usd": "3.0",
            },
        ],
    )
    analysis_dir = batch_dir / "analysis"
    analysis_dir.mkdir()
    _write_csv(
        analysis_dir / "offline_router_predictions.csv",
        [
            {
                "dataset": "dataset_a",
                "model": "model_a",
                "question_id": "q1",
                "selected_correct": "True",
                "estimated_cost_usd": "1.0",
            },
            {
                "dataset": "dataset_a",
                "model": "model_a",
                "question_id": "q2",
                "selected_correct": "True",
                "estimated_cost_usd": "2.0",
            },
        ],
    )
    return batch_dir


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
