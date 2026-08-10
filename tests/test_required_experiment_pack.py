import csv
import json

from cups_eval.experiment_data_by_strategy import build_experiment_data_by_strategy
from cups_eval.required_experiment_pack import (
    build_cups_artifacts,
    build_math500_inactive_provenance_sources,
    build_math500_provenance_discrepancies,
    build_canonical_metrics,
    build_oracle_opportunity_rows,
    build_rescue_harm_rows,
    paired_comparison_stats,
    router_fp_fn_rows,
    selected_cups_policies,
)


def canonical_row(strategy, qid, *, correct, dataset="toy", model="model_a", cost=0.1):
    return {
        "strategy": strategy,
        "dataset": dataset,
        "model": model,
        "question_id": qid,
        "correct": str(correct),
        "parse_ok": "True",
        "total_tokens": "10",
        "estimated_cost_usd": str(cost),
        "source_batch": "fixture_batch",
        "run_id": "fixture_run",
        "source_raw_predictions": "/tmp/raw_predictions.jsonl",
    }


def feature_row(qid, *, direct, cot=False, planner=False, debate=False, dataset="toy", model="model_a"):
    return {
        "dataset": dataset,
        "model": model,
        "question_id": qid,
        "direct_correct": str(direct),
        "cot_correct": str(cot),
        "planner_verifier_correct": str(planner),
        "debate_2_judge_correct": str(debate),
    }


def prediction_row(qid, *, selected_strategy, selected_correct, dataset="toy", model="model_a"):
    return {
        "dataset": dataset,
        "model": model,
        "question_id": qid,
        "selected_strategy": selected_strategy,
        "selected_correct": str(selected_correct),
        "estimated_cost_usd": "0.2",
        "direct_cost_usd": "0.1",
    }


def test_build_canonical_metrics_includes_strategy_and_oracle_accuracy():
    rows = [
        canonical_row("direct", "q1", correct=True),
        canonical_row("direct", "q2", correct=False),
        canonical_row("cot", "q1", correct=True),
        canonical_row("cot", "q2", correct=True),
        canonical_row("planner_verifier", "q1", correct=False),
        canonical_row("planner_verifier", "q2", correct=False),
    ]

    metrics = build_canonical_metrics(rows)
    lookup = {(row["strategy"], row["dataset"], row["model"]): row for row in metrics}

    assert lookup[("direct", "toy", "model_a")]["accuracy"] == 0.5
    assert lookup[("cot", "toy", "model_a")]["accuracy"] == 1.0
    assert lookup[("oracle_any", "toy", "model_a")]["correct"] == 2
    assert lookup[("oracle_any", "toy", "model_a")]["accuracy"] == 1.0


def test_rescue_harm_rows_align_direct_against_each_strategy():
    rows = [
        canonical_row("direct", "q1", correct=False),
        canonical_row("direct", "q2", correct=True),
        canonical_row("direct", "q3", correct=False),
        canonical_row("cot", "q1", correct=True),
        canonical_row("cot", "q2", correct=False),
        canonical_row("cot", "q3", correct=False),
    ]

    diagnostics = build_rescue_harm_rows(rows)

    assert diagnostics == [
        {
            "dataset": "toy",
            "model": "model_a",
            "comparison": "cot",
            "n_aligned": 3,
            "direct_correct": 1,
            "comparison_correct": 1,
            "both_correct": 0,
            "both_wrong": 1,
            "rescue": 1,
            "harm": 1,
            "rescue_rate": 1 / 3,
            "harm_rate": 1 / 3,
            "net_gain": 0,
            "net_gain_rate": 0.0,
            "direct_accuracy": 1 / 3,
            "comparison_accuracy": 1 / 3,
        }
    ]


def test_oracle_opportunity_counts_direct_errors_rescued_by_any_strategy():
    rows = [
        canonical_row("direct", "q1", correct=False),
        canonical_row("direct", "q2", correct=False),
        canonical_row("direct", "q3", correct=True),
        canonical_row("cot", "q1", correct=True),
        canonical_row("planner_verifier", "q2", correct=False),
        canonical_row("debate_2_judge", "q2", correct=True),
    ]

    opportunities = build_oracle_opportunity_rows(rows)

    assert opportunities[0]["direct_error"] == 2
    assert opportunities[0]["oracle_rescue"] == 2
    assert opportunities[0]["unresolved"] == 0
    assert opportunities[0]["oracle_any_accuracy"] == 1.0


def test_self_consistency_participates_in_canonical_oracle_and_rescue_harm():
    rows = [
        canonical_row("direct", "q1", correct=False),
        canonical_row("self_consistency_3", "q1", correct=True),
    ]

    metrics = build_canonical_metrics(rows)
    rescue_harm = build_rescue_harm_rows(rows)

    oracle = [
        row for row in metrics
        if row["strategy"] == "oracle_any" and row["dataset"] == "toy"
    ][0]
    assert oracle["correct"] == 1
    assert rescue_harm[0]["comparison"] == "self_consistency_3"
    assert rescue_harm[0]["rescue"] == 1


def test_paired_comparison_stats_reports_mcnemar_and_bootstrap_delta():
    stats = paired_comparison_stats(
        [True, True, True, True],
        [False, False, False, True],
        bootstrap_iterations=200,
        seed=7,
    )

    assert stats["n"] == 4
    assert stats["paired_rescues"] == 3
    assert stats["paired_harms"] == 0
    assert stats["delta_accuracy"] == 0.75
    assert stats["mcnemar_exact_p_two_sided"] == 0.25
    assert stats["bootstrap_delta_ci_low"] <= stats["delta_accuracy"]
    assert stats["bootstrap_delta_ci_high"] >= stats["delta_accuracy"]


def test_router_fp_fn_rows_group_selected_cups_policy_by_dataset_model():
    features = [
        feature_row("q1", direct=False, cot=True),
        feature_row("q2", direct=False, cot=True),
        feature_row("q3", direct=True, cot=False),
        feature_row("q4", direct=True, cot=False),
    ]
    predictions = [
        prediction_row("q1", selected_strategy="cot", selected_correct=True),
        prediction_row("q2", selected_strategy="direct", selected_correct=False),
        prediction_row("q3", selected_strategy="cot", selected_correct=False),
        prediction_row("q4", selected_strategy="direct", selected_correct=True),
    ]

    rows = router_fp_fn_rows(
        "fixture_scope",
        "Fixture Scope",
        "cups_lambda_0",
        predictions,
        features,
    )

    assert rows == [
        {
            "scope_id": "fixture_scope",
            "scope_label": "Fixture Scope",
            "selection_type": "",
            "dataset": "toy",
            "model": "model_a",
            "policy": "cups_lambda_0",
            "n": 4,
            "true_positive": 1,
            "false_negative": 1,
            "false_positive": 1,
            "true_negative": 1,
            "precision": 0.5,
            "recall": 0.5,
            "false_positive_rate": 0.5,
            "false_negative_rate": 0.5,
            "escalation_rate": 0.5,
        }
    ]


def test_selected_cups_policies_match_best_accuracy_and_best_vs_cot_semantics():
    curve = [
        {"policy": "always_cot", "policy_family": "baseline", "accuracy": 0.80, "relative_cost_per_correct": 1.1},
        {
            "policy": "cups_lambda_0.03",
            "policy_family": "cups",
            "accuracy": 0.90,
            "relative_cost_per_correct": 1.5,
        },
        {
            "policy": "cups_lambda_0.3",
            "policy_family": "cups",
            "accuracy": 0.86,
            "relative_cost_per_correct": 1.0,
        },
    ]

    assert selected_cups_policies(curve) == [
        ("best_accuracy", "cups_lambda_0.03"),
        ("best_vs_cot_cpc", "cups_lambda_0.3"),
    ]


def test_build_cups_artifacts_writes_all_evaluation_regimes(tmp_path, monkeypatch):
    batch_dir = tmp_path / "runs" / "fixture_batch"
    for model in ("model_a", "model_b"):
        run_dir = batch_dir / "runs" / "toy" / model / "run1"
        run_dir.mkdir(parents=True)
        rows = [
            {
                "dataset": "toy",
                "model": model,
                "question_id": "q1",
                "gold": "B",
                "direct_correct": "True",
                "direct_cost_usd": "1.0",
                "cot_correct": "True",
                "cot_cost_usd": "0.9",
                "planner_verifier_correct": "False",
                "planner_verifier_cost_usd": "2.0",
                "debate_2_judge_correct": "False",
                "debate_2_judge_cost_usd": "3.0",
                "direct_confidence": "0.9",
                "direct_rationale_word_count": "5",
            },
            {
                "dataset": "toy",
                "model": model,
                "question_id": "q2",
                "gold": "B",
                "direct_correct": "False",
                "direct_cost_usd": "1.0",
                "cot_correct": "True",
                "cot_cost_usd": "0.9",
                "planner_verifier_correct": "True",
                "planner_verifier_cost_usd": "2.0",
                "debate_2_judge_correct": "False",
                "debate_2_judge_cost_usd": "3.0",
                "direct_confidence": "0.2",
                "direct_rationale_word_count": "80",
            },
        ]
        with (run_dir / "router_features.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    monkeypatch.setattr(
        "cups_eval.required_experiment_pack._cups_feature_scopes",
        lambda project_root: [
            {
                "scope_id": "fixture_scope",
                "scope_label": "Fixture Scope",
                "batch_dir": batch_dir,
            }
        ],
    )
    monkeypatch.setattr("cups_eval.required_experiment_pack._known_missing_cups_scopes", lambda project_root: [])

    artifacts = build_cups_artifacts(tmp_path, tmp_path / "cups", bootstrap_iterations=20)

    regimes = {row["evaluation_regime"] for row in artifacts["curve_rows"]}
    assert regimes == {"item_oof", "leave_one_dataset_model_out", "leave_one_model_out"}
    assert {
        row["evaluation_regime"]
        for row in artifacts["best_policy_tests"]
    } == regimes
    assert (tmp_path / "cups" / "generated_cups" / "fixture_scope" / "item_oof" / "cups_curve.csv").exists()
    assert (
        tmp_path
        / "cups"
        / "generated_cups"
        / "fixture_scope"
        / "leave_one_dataset_model_out"
        / "cups_curve.csv"
    ).exists()
    assert (
        tmp_path
        / "cups"
        / "generated_cups"
        / "fixture_scope"
        / "leave_one_model_out"
        / "cups_curve.csv"
    ).exists()


def test_build_cups_artifacts_marks_single_group_holdouts_unsupported(tmp_path, monkeypatch):
    batch_dir = tmp_path / "runs" / "single_model_batch"
    run_dir = batch_dir / "runs" / "math500" / "minimax_m3" / "run1"
    run_dir.mkdir(parents=True)
    rows = [
        {
            "dataset": "math500",
            "model": "minimax_m3",
            "question_id": "q1",
            "gold": "B",
            "direct_correct": "True",
            "direct_cost_usd": "1.0",
            "cot_correct": "True",
            "cot_cost_usd": "0.9",
            "planner_verifier_correct": "False",
            "planner_verifier_cost_usd": "2.0",
            "debate_2_judge_correct": "False",
            "debate_2_judge_cost_usd": "3.0",
            "direct_confidence": "0.9",
            "direct_rationale_word_count": "5",
        },
        {
            "dataset": "math500",
            "model": "minimax_m3",
            "question_id": "q2",
            "gold": "B",
            "direct_correct": "False",
            "direct_cost_usd": "1.0",
            "cot_correct": "True",
            "cot_cost_usd": "0.9",
            "planner_verifier_correct": "True",
            "planner_verifier_cost_usd": "2.0",
            "debate_2_judge_correct": "False",
            "debate_2_judge_cost_usd": "3.0",
            "direct_confidence": "0.2",
            "direct_rationale_word_count": "80",
        },
    ]
    with (run_dir / "router_features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(
        "cups_eval.required_experiment_pack._cups_feature_scopes",
        lambda project_root: [
            {
                "scope_id": "single_scope",
                "scope_label": "Single Scope",
                "batch_dir": batch_dir,
            }
        ],
    )
    monkeypatch.setattr("cups_eval.required_experiment_pack._known_missing_cups_scopes", lambda project_root: [])

    artifacts = build_cups_artifacts(tmp_path, tmp_path / "cups", bootstrap_iterations=20)

    assert {row["evaluation_regime"] for row in artifacts["curve_rows"]} == {"item_oof"}
    missing = {(row["scope_id"], row.get("evaluation_regime")): row for row in artifacts["missing_scopes"]}
    assert ("single_scope", "leave_one_dataset_model_out") in missing
    assert ("single_scope", "leave_one_model_out") in missing
    assert (tmp_path / "cups" / "generated_cups" / "single_scope" / "item_oof" / "cups_curve.csv").exists()
    assert not (
        tmp_path
        / "cups"
        / "generated_cups"
        / "single_scope"
        / "leave_one_dataset_model_out"
        / "cups_curve.csv"
    ).exists()


def test_math500_provenance_excludes_placeholders_and_short_preruns(tmp_path):
    gradient = tmp_path / "reports" / "cups_packet" / "math500_capability_gradient.csv"
    gradient.parent.mkdir(parents=True)
    gradient.write_text(
        "\n".join(
            [
                "model,complete_q,direct,cot,planner_verifier,debate_2_judge,oracle,status",
                "MiniMax-M3,100,0.89,0.88,0.93,0.94,0.98,complete",
                "GPT frontier placeholder,,TBD,TBD,TBD,TBD,TBD,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    short_summary = (
        tmp_path
        / "runs"
        / "phase3_minimax_math500"
        / "short"
        / "runs"
        / "math500"
        / "minimax_m3"
        / "run5"
        / "summary.json"
    )
    short_summary.parent.mkdir(parents=True)
    short_summary.write_text(
        json.dumps({"direct": {"n": 5, "accuracy": 1.0}}),
        encoding="utf-8",
    )
    full_summary = (
        tmp_path
        / "runs"
        / "phase3_minimax_math500"
        / "full"
        / "runs"
        / "math500"
        / "minimax_m3"
        / "run100"
        / "summary.json"
    )
    full_summary.parent.mkdir(parents=True)
    full_summary.write_text(
        json.dumps({"direct": {"n": 100, "accuracy": 0.78}}),
        encoding="utf-8",
    )
    canonical = [
        {
            "dataset": "math500",
            "model": "minimax_m3",
            "strategy": "direct",
            "n": 100,
            "accuracy": 0.76,
        }
    ]

    rows = build_math500_provenance_discrepancies(tmp_path, canonical)

    assert not any("placeholder" in row["model"] for row in rows)
    assert not any(row["source_n"] == 5 for row in rows)
    assert any(row["source_kind"] == "summary_json" and row["source_n"] == 100 for row in rows)


def test_experiment_data_uses_evaluated_item_metrics_and_audits_raw_corrections(tmp_path):
    run_dir = tmp_path / "runs" / "batch" / "run" / "runs" / "math500" / "minimax_m3" / "eval_run"
    run_dir.mkdir(parents=True)
    raw_record = {
        "question_id": "test/algebra/101.json",
        "question": "For what values of x?",
        "options": {},
        "question_metadata": {"subject_name": "Algebra", "topic_name": "Intervals"},
        "gold": "x\\in[-2,7]",
        "strategy": "direct",
        "answer": "[-2,7]",
        "correct": False,
        "confidence": 0.99,
        "rationale": "Solve the quadratic inequality.",
        "parse_ok": True,
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        "api_calls": 1,
        "estimated_cost_usd": 0.01,
        "calls": [{"role": "direct"}],
    }
    (run_dir / "raw_predictions.jsonl").write_text(json.dumps(raw_record) + "\n", encoding="utf-8")
    (run_dir / "item_metrics.csv").write_text(
        "\n".join(
            [
                "question_id,strategy,gold,answer,correct,confidence,parse_ok,api_calls,prompt_tokens,completion_tokens,total_tokens,estimated_cost_usd",
                "test/algebra/101.json,direct,\"x\\in[-2,7]\",\"[-2,7]\",True,0.99,True,1,3,4,7,0.01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports" / "cups_packet" / "experiment_data_by_strategy"
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.csv").write_text(
        "\n".join(
            [
                "source_batch,dataset,model,run_id,strategy,rows,unique_questions,source_raw_predictions,output_file",
                "phase3_math500_minimax,math500,minimax_m3,eval_run,direct,1,1,runs/batch/run/runs/math500/minimax_m3/eval_run/raw_predictions.jsonl,reports/cups_packet/experiment_data_by_strategy/direct/math500-minimax_m3.csv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_experiment_data_by_strategy(project_root=tmp_path, output_dir=output_dir)

    canonical_rows = list(csv.DictReader((output_dir / "direct" / "math500-minimax_m3.csv").open()))
    assert canonical_rows[0]["correct"] == "True"
    assert canonical_rows[0]["answer"] == "[-2,7]"
    assert canonical_rows[0]["source_raw_predictions"].endswith("raw_predictions.jsonl")
    assert canonical_rows[0]["source_evaluated_artifact"].endswith("item_metrics.csv")
    audit_rows = list(csv.DictReader((output_dir / "evaluation_audit" / "raw_vs_evaluated_corrections.csv").open()))
    assert audit_rows == [
        {
            "dataset": "math500",
            "model": "minimax_m3",
            "strategy": "direct",
            "question_id": "test/algebra/101.json",
            "correction_category": "interval_notation_equivalence",
            "raw_answer": "[-2,7]",
            "raw_correct": "False",
            "raw_parse_ok": "True",
            "evaluated_answer": "[-2,7]",
            "evaluated_correct": "True",
            "evaluated_parse_ok": "True",
            "source_raw_predictions": "runs/batch/run/runs/math500/minimax_m3/eval_run/raw_predictions.jsonl",
            "source_evaluated_artifact": "runs/batch/run/runs/math500/minimax_m3/eval_run/item_metrics.csv",
        }
    ]
    assert manifest["row_counts"]["raw_vs_evaluated_corrections"] == 1


def test_math500_provenance_keeps_superseded_qwen_out_of_active_discrepancies(tmp_path):
    clean_dir = (
        tmp_path
        / "runs"
        / "phase3_qwen_math500_clean"
        / "clean"
        / "runs"
        / "math500"
        / "qwen_3_7_max_preview"
        / "clean_run"
    )
    old_dir = (
        tmp_path
        / "runs"
        / "phase3_qwen_math500"
        / "old"
        / "runs"
        / "math500"
        / "qwen_3_7_max_preview"
        / "old_run"
    )
    clean_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    for path, accuracy in ((clean_dir / "router_features.csv", "True"), (old_dir / "router_features.csv", "False")):
        body = "\n".join(
            f"q{i},{accuracy},{accuracy},{accuracy},{accuracy},{accuracy}"
            for i in range(50)
        )
        path.write_text(
            "\n".join(
                [
                    "question_id,direct_correct,cot_correct,planner_verifier_correct,debate_2_judge_correct,oracle_correct",
                    body,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    canonical = [
        {
            "dataset": "math500",
            "model": "qwen_3_7_max_preview",
            "strategy": strategy,
            "n": 50,
            "accuracy": 1.0,
            "source_raw_predictions": "runs/phase3_qwen_math500_clean/clean/runs/math500/qwen_3_7_max_preview/clean_run/raw_predictions.jsonl",
        }
        for strategy in ("direct", "cot", "planner_verifier", "debate_2_judge", "oracle_any")
    ]

    active_rows = build_math500_provenance_discrepancies(tmp_path, canonical)
    inactive_rows = build_math500_inactive_provenance_sources(tmp_path, canonical)

    assert {row["status"] for row in active_rows} == {"match"}
    assert not any("phase3_qwen_math500/old" in row["source_path"] for row in active_rows)
    assert any(
        row["status"] == "superseded_source"
        and "phase3_qwen_math500/old" in row["source_path"]
        for row in inactive_rows
    )


def test_math500_provenance_treats_stale_summary_as_inactive_when_evaluated_rows_exist(tmp_path):
    run_dir = (
        tmp_path
        / "runs"
        / "phase3_minimax_math500"
        / "batch"
        / "runs"
        / "math500"
        / "minimax_m3"
        / "run"
    )
    run_dir.mkdir(parents=True)
    body = "\n".join(f"q{i},True,True,True,True,True" for i in range(50))
    (run_dir / "router_features.csv").write_text(
        "\n".join(
            [
                "question_id,direct_correct,cot_correct,planner_verifier_correct,debate_2_judge_correct,oracle_correct",
                body,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"direct": {"n": 50, "accuracy": 0.0}}),
        encoding="utf-8",
    )
    canonical = [
        {
            "dataset": "math500",
            "model": "minimax_m3",
            "strategy": strategy,
            "n": 50,
            "accuracy": 1.0,
            "source_raw_predictions": "runs/phase3_minimax_math500/batch/runs/math500/minimax_m3/run/raw_predictions.jsonl",
        }
        for strategy in ("direct", "cot", "planner_verifier", "debate_2_judge", "oracle_any")
    ]

    active_rows = build_math500_provenance_discrepancies(tmp_path, canonical)
    inactive_rows = build_math500_inactive_provenance_sources(tmp_path, canonical)

    assert {row["source_kind"] for row in active_rows} == {"router_features"}
    assert {row["status"] for row in active_rows} == {"match"}
    assert any(
        row["source_kind"] == "summary_json"
        and row["status"] == "superseded_evaluation_artifact"
        for row in inactive_rows
    )


def test_math500_router_feature_provenance_maps_self_consistency_alias_into_oracle(tmp_path):
    run_dir = (
        tmp_path
        / "runs"
        / "phase3_math500_p0_full_anchor"
        / "batch"
        / "runs"
        / "math500"
        / "deepseek_v4_flash"
        / "run"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "router_features.csv").write_text(
        "\n".join(
            [
                "question_id,direct_correct,cot_correct,self_consistency_correct,planner_verifier_correct,debate_2_judge_correct,oracle_correct",
                "q1,True,True,True,True,True,True",
            ]
            + [f"q{i},False,False,True,False,False,True" for i in range(2, 51)]
        )
        + "\n",
        encoding="utf-8",
    )
    canonical = [
        {
            "dataset": "math500",
            "model": "deepseek_v4_flash",
            "strategy": "direct",
            "n": 50,
            "accuracy": 0.02,
            "source_raw_predictions": "runs/phase3_math500_p0_full_anchor/batch/runs/math500/deepseek_v4_flash/run/raw_predictions.jsonl",
        },
        {
            "dataset": "math500",
            "model": "deepseek_v4_flash",
            "strategy": "cot",
            "n": 50,
            "accuracy": 0.02,
            "source_raw_predictions": "runs/phase3_math500_p0_full_anchor/batch/runs/math500/deepseek_v4_flash/run/raw_predictions.jsonl",
        },
        {
            "dataset": "math500",
            "model": "deepseek_v4_flash",
            "strategy": "self_consistency_3",
            "n": 50,
            "accuracy": 1.0,
            "source_raw_predictions": "runs/phase3_math500_p0_full_anchor/batch/runs/math500/deepseek_v4_flash/run/raw_predictions.jsonl",
        },
        {
            "dataset": "math500",
            "model": "deepseek_v4_flash",
            "strategy": "planner_verifier",
            "n": 50,
            "accuracy": 0.02,
            "source_raw_predictions": "runs/phase3_math500_p0_full_anchor/batch/runs/math500/deepseek_v4_flash/run/raw_predictions.jsonl",
        },
        {
            "dataset": "math500",
            "model": "deepseek_v4_flash",
            "strategy": "debate_2_judge",
            "n": 50,
            "accuracy": 0.02,
            "source_raw_predictions": "runs/phase3_math500_p0_full_anchor/batch/runs/math500/deepseek_v4_flash/run/raw_predictions.jsonl",
        },
        {
            "dataset": "math500",
            "model": "deepseek_v4_flash",
            "strategy": "oracle_any",
            "n": 50,
            "accuracy": 1.0,
            "source_raw_predictions": "runs/phase3_math500_p0_full_anchor/batch/runs/math500/deepseek_v4_flash/run/raw_predictions.jsonl",
        },
    ]

    rows = build_math500_provenance_discrepancies(tmp_path, canonical)
    router_rows = [
        row
        for row in rows
        if row["source_kind"] == "router_features"
    ]

    assert {
        (row["strategy"], row["status"])
        for row in router_rows
    } == {
        ("direct", "match"),
        ("cot", "match"),
        ("self_consistency_3", "match"),
        ("planner_verifier", "match"),
        ("debate_2_judge", "match"),
        ("oracle_any", "match"),
    }
