import csv
import json

from cups_eval.p0_completion_pack import build_p0_completion_pack


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_p0_completion_pack_summarizes_math_audit_and_canonical_coverage(tmp_path):
    run_dir = tmp_path / "runs" / "phase" / "batch" / "runs" / "math500" / "deepseek_v4_flash" / "run1"
    run_dir.mkdir(parents=True)
    raw_rows = [
        {"question_id": "q1", "strategy": "direct"},
        {"question_id": "q1", "strategy": "cot"},
        {"question_id": "q2", "strategy": "direct"},
        {"question_id": "q2", "strategy": "cot"},
    ]
    (run_dir / "raw_predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_rows) + "\n",
        encoding="utf-8",
    )
    write_csv(
        run_dir / "item_metrics.csv",
        [
            {"question_id": "q1", "strategy": "direct", "correct": "True", "parse_ok": "True"},
            {"question_id": "q1", "strategy": "cot", "correct": "True", "parse_ok": "True"},
            {"question_id": "q2", "strategy": "direct", "correct": "False", "parse_ok": "True"},
            {"question_id": "q2", "strategy": "cot", "correct": "True", "parse_ok": "True"},
        ],
    )
    write_csv(
        run_dir / "math_equivalence_audit.csv",
        [
            {"verdict": "equivalent", "needs_blind_review": "False"},
            {"verdict": "unresolved", "needs_blind_review": "True"},
        ],
    )
    canonical_dir = tmp_path / "reports" / "cups_packet" / "experiment_data_by_strategy"
    write_csv(
        canonical_dir / "manifest.csv",
        [
            {
                "source_batch": "p0",
                "dataset": "math500",
                "model": "deepseek_v4_flash",
                "run_id": "run1",
                "strategy": "direct",
                "rows": "2",
                "unique_questions": "2",
                "source_raw_predictions": "runs/phase/batch/runs/math500/deepseek_v4_flash/run1/raw_predictions.jsonl",
                "source_evaluated_artifact": "runs/phase/batch/runs/math500/deepseek_v4_flash/run1/item_metrics.csv",
                "output_file": "reports/cups_packet/experiment_data_by_strategy/direct/math500-deepseek_v4_flash.csv",
            },
            {
                "source_batch": "p0",
                "dataset": "math500",
                "model": "deepseek_v4_flash",
                "run_id": "run1",
                "strategy": "cot",
                "rows": "2",
                "unique_questions": "2",
                "source_raw_predictions": "runs/phase/batch/runs/math500/deepseek_v4_flash/run1/raw_predictions.jsonl",
                "source_evaluated_artifact": "runs/phase/batch/runs/math500/deepseek_v4_flash/run1/item_metrics.csv",
                "output_file": "reports/cups_packet/experiment_data_by_strategy/cot/math500-deepseek_v4_flash.csv",
            },
        ],
    )

    manifest = build_p0_completion_pack(
        project_root=tmp_path,
        output_dir=tmp_path / "reports" / "cups_packet" / "p0_completion",
        run_specs=[
            {
                "dataset": "math500",
                "model": "deepseek_v4_flash",
                "label": "DeepSeek v4 flash",
                "sample_label": "fixture",
                "planned_n": 2,
                "run_dir": run_dir,
            }
        ],
        canonical_dir=canonical_dir,
    )

    assert manifest["row_counts"]["run_inclusion"] == 1
    assert manifest["row_counts"]["math500_final_label_audit"] == 1
    assert manifest["row_counts"]["canonical_math500_coverage"] == 2
    assert manifest["headline"]["complete_required_math_runs"] == 1
    assert (tmp_path / "reports" / "cups_packet" / "p0_completion" / "p0_completion.md").exists()

    audit_rows = list(
        csv.DictReader(
            (tmp_path / "reports" / "cups_packet" / "p0_completion" / "math500_final_label_audit.csv").open()
        )
    )
    assert audit_rows[0]["audit_rows"] == "2"
    assert audit_rows[0]["unresolved"] == "1"
    assert audit_rows[0]["needs_blind_review"] == "1"
