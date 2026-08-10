from pathlib import Path
import csv
import json
import threading
import time

from cups_eval.llm import FakeLLMClient
from cups_eval.runner import run_experiment


def test_fake_llm_run_writes_expected_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("CUPS_MODEL_PROFILE", "deepseek_v4_pro")
    run_dir = run_experiment(
        config_path=Path("configs/mvp.medmcqa.yaml"),
        sample_size=5,
        fake_llm=True,
        output_root=tmp_path,
        dataset_records=[
            {
                "id": f"q{i}",
                "question": f"Question {i}?",
                "opa": "alpha",
                "opb": "bravo",
                "opc": "charlie",
                "opd": "delta",
                "cop": i % 4,
                "choice_type": "single",
            }
            for i in range(5)
        ],
    )

    assert (run_dir / "raw_predictions.jsonl").exists()
    assert (run_dir / "item_metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "pareto.csv").exists()
    assert (run_dir / "router_features.csv").exists()
    assert (run_dir / "rule_search.csv").exists()
    assert (run_dir / "learned_rule_eval.csv").exists()
    assert (run_dir / "strata_summary.csv").exists()
    assert (run_dir / "routing_decisions.csv").exists()
    assert (run_dir / "cost_audit.csv").exists()
    assert (run_dir / "paired_stats.csv").exists()
    pareto_rows = list(csv.DictReader((run_dir / "pareto.csv").open()))
    assert all(row["strategy"] != "oracle_router" for row in pareto_rows)
    assert (run_dir / "run_metadata.json").exists()
    assert "## Verdict" in (run_dir / "summary.md").read_text()
    assert "| probe_router |" in (run_dir / "summary.md").read_text()
    assert "router" in (run_dir / "summary.md").read_text()
    assert "| oracle_router |" in (run_dir / "summary.md").read_text()

    routing_rows = list(csv.DictReader((run_dir / "routing_decisions.csv").open()))
    assert len(routing_rows) == 5
    assert {
        "question_id",
        "oracle_choice",
        "oracle_escalated",
        "best_strategy_per_item",
        "router_choice",
        "router_correct",
        "probe_router_choice",
        "probe_router_correct",
        "oracle_correct",
    }.issubset(routing_rows[0])

    audit_rows = list(csv.DictReader((run_dir / "cost_audit.csv").open()))
    assert len(audit_rows) == 35
    assert all(row["api_calls_match"] == "True" for row in audit_rows)
    assert all(row["usage_matches_call_sum"] == "True" for row in audit_rows)

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert "code_fingerprint_sha256" in metadata
    assert "git" in metadata
    assert metadata["llm"]["profile"] == "deepseek_v4_pro"
    assert metadata["llm"]["model"] == "deepseek-v4-pro"
    assert metadata["pricing"]["input_per_million_usd"] == 0.44


def test_fake_llm_parallel_run_writes_unique_strategy_rows(tmp_path):
    run_dir = run_experiment(
        config_path=Path("configs/mvp.medmcqa.yaml"),
        sample_size=5,
        fake_llm=True,
        output_root=tmp_path,
        workers=3,
        dataset_records=[
            {
                "id": f"parallel-q{i}",
                "question": f"Parallel question {i}?",
                "opa": "alpha",
                "opb": "bravo",
                "opc": "charlie",
                "opd": "delta",
                "cop": i % 4,
                "choice_type": "single",
            }
            for i in range(5)
        ],
    )

    raw_records = [
        json.loads(line)
        for line in (run_dir / "raw_predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    keys = {(record["question_id"], record["strategy"]) for record in raw_records}

    assert len(raw_records) == 35
    assert len(keys) == 35
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "cost_audit.csv").exists()


def test_parallel_provider_clients_are_created_inside_workers(tmp_path, monkeypatch):
    created_threads: list[str] = []

    class RecordingLLM(FakeLLMClient):
        def call(self, question, role, **kwargs):
            time.sleep(0.001)
            return super().call(question, role, **kwargs)

    def from_config(_config):
        created_threads.append(threading.current_thread().name)
        return RecordingLLM(mode="always_correct")

    monkeypatch.setattr(
        "cups_eval.runner.OpenAICompatibleLLMClient.from_config",
        from_config,
    )

    run_experiment(
        config_path=Path("configs/mvp.medmcqa.yaml"),
        sample_size=4,
        fake_llm=False,
        output_root=tmp_path,
        workers=3,
        dataset_records=[
            {
                "id": f"provider-q{i}",
                "question": f"Provider question {i}?",
                "opa": "alpha",
                "opb": "bravo",
                "opc": "charlie",
                "opd": "delta",
                "cop": i % 4,
                "choice_type": "single",
            }
            for i in range(4)
        ],
    )

    assert created_threads
    assert "MainThread" not in set(created_threads)
