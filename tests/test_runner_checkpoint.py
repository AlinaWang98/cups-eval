import json
import time
from pathlib import Path

import pytest

from cups_eval.llm import FakeLLMClient
from cups_eval.runner import (
    _reparse_records_from_calls,
    _run_strategy_record_with_timeout,
    run_experiment,
)
from cups_eval.schema import Question


class HangingStrategy:
    name = "hang"

    def run(self, question, llm):
        time.sleep(5)


def test_run_experiment_keeps_partial_raw_predictions_when_strategy_fails(
    tmp_path,
    monkeypatch,
):
    def fail_cot(self, question, llm):
        raise RuntimeError("forced cot failure")

    monkeypatch.setattr("cups_eval.runner.CotStrategy.run", fail_cot)

    with pytest.raises(RuntimeError, match="forced cot failure"):
        run_experiment(
            config_path=Path("configs/mvp.medmcqa.yaml"),
            sample_size=1,
            fake_llm=True,
            output_root=tmp_path,
            dataset_records=[
                {
                    "id": "q1",
                    "question": "Question?",
                    "opa": "alpha",
                    "opb": "bravo",
                    "opc": "charlie",
                    "opd": "delta",
                    "cop": 0,
                    "choice_type": "single",
                }
            ],
        )

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    raw_path = run_dirs[0] / "raw_predictions.jsonl"
    assert raw_path.exists()
    lines = raw_path.read_text().splitlines()
    assert len(lines) == 1
    assert '"strategy": "direct"' in lines[0]


def test_run_experiment_resume_skips_completed_strategy_rows(tmp_path, monkeypatch):
    def fail_cot_once(self, question, llm):
        if not hasattr(fail_cot_once, "already_failed"):
            fail_cot_once.already_failed = True
            raise RuntimeError("forced cot failure")
        return original_cot_run(self, question, llm)

    from cups_eval.strategies import CotStrategy

    original_cot_run = CotStrategy.run
    monkeypatch.setattr("cups_eval.runner.CotStrategy.run", fail_cot_once)
    dataset_records = [
        {
            "id": "q1",
            "question": "Question?",
            "opa": "alpha",
            "opb": "bravo",
            "opc": "charlie",
            "opd": "delta",
            "cop": 0,
            "choice_type": "single",
        }
    ]

    with pytest.raises(RuntimeError, match="forced cot failure"):
        run_experiment(
            config_path=Path("configs/mvp.medmcqa.yaml"),
            sample_size=1,
            fake_llm=True,
            output_root=tmp_path,
            dataset_records=dataset_records,
        )

    run_dir = next(tmp_path.iterdir())
    resumed = run_experiment(
        config_path=Path("configs/mvp.medmcqa.yaml"),
        sample_size=1,
        fake_llm=True,
        output_root=tmp_path,
        dataset_records=dataset_records,
        resume_run_dir=run_dir,
    )

    assert resumed == run_dir
    rows = [
        json.loads(line)
        for line in (run_dir / "raw_predictions.jsonl").read_text().splitlines()
    ]
    strategies = [row["strategy"] for row in rows]
    assert strategies.count("direct") == 1
    assert set(strategies) == {
        "direct",
        "cot",
        "self_consistency_3",
        "debate_2_judge",
        "planner_verifier",
        "router",
        "probe_router",
    }
    assert (run_dir / "run_metadata.json").exists()
    assert json.loads((run_dir / "run_metadata.json").read_text())["status"] == "complete"


def test_run_experiment_resume_rejects_incompatible_question_set(tmp_path):
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    existing = {
        "question_id": "q-does-not-belong",
        "question": "Old?",
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "gold": "A",
        "strategy": "direct",
        "answer": "A",
        "correct": True,
        "confidence": 0.9,
        "rationale": "",
        "parse_ok": True,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "api_calls": 1,
        "calls": [],
        "metadata": {},
        "estimated_cost_usd": 0.0,
    }
    (run_dir / "raw_predictions.jsonl").write_text(json.dumps(existing) + "\n")

    with pytest.raises(ValueError, match="not part of the requested sample"):
        run_experiment(
            config_path=Path("configs/mvp.medmcqa.yaml"),
            sample_size=1,
            fake_llm=True,
            output_root=tmp_path,
            dataset_records=[
                {
                    "id": "q1",
                    "question": "Question?",
                    "opa": "alpha",
                    "opb": "bravo",
                    "opc": "charlie",
                    "opd": "delta",
                    "cop": 0,
                    "choice_type": "single",
                }
            ],
            resume_run_dir=run_dir,
        )


def test_run_experiment_respects_configured_strategy_subset(tmp_path):
    config_path = tmp_path / "subset.yaml"
    config_path.write_text(
        """
dataset:
  sample_size: 1
  seed: 42
pricing: {}
run:
  max_usd: 1.0
  strategies:
    - direct
    - cot
    - debate_2_judge
""",
        encoding="utf-8",
    )

    run_dir = run_experiment(
        config_path=config_path,
        sample_size=1,
        fake_llm=True,
        output_root=tmp_path / "runs",
        dataset_records=[
            {
                "id": "q1",
                "question": "Question?",
                "opa": "alpha",
                "opb": "bravo",
                "opc": "charlie",
                "opd": "delta",
                "cop": 0,
                "choice_type": "single",
            }
        ],
    )

    rows = [
        json.loads(line)
        for line in (run_dir / "raw_predictions.jsonl").read_text().splitlines()
    ]
    assert [row["strategy"] for row in rows] == [
        "direct",
        "cot",
        "debate_2_judge",
    ]
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["strategies"] == ["direct", "cot", "debate_2_judge"]


def test_strategy_timeout_records_failure_row_instead_of_hanging():
    question = Question(
        id="q1",
        question="Question?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        answer="A",
    )

    record, cost = _run_strategy_record_with_timeout(
        question=question,
        strategy=HangingStrategy(),
        config={
            "pricing": {},
            "run": {
                "strategy_timeout_seconds": 0.1,
                "force_strategy_timeout_for_fake": True,
            },
        },
        fake_llm=True,
        llm=FakeLLMClient(),
    )

    assert cost == 0.0
    assert record["strategy"] == "hang"
    assert record["answer"] is None
    assert record["correct"] is False
    assert record["parse_ok"] is False
    assert record["api_calls"] == 0
    assert record["metadata"]["error_type"] == "strategy_timeout"


def test_reparse_records_from_calls_repairs_math_answer_and_correctness():
    record = {
        "question_id": "math-1",
        "question": "Compute.",
        "options": {},
        "question_metadata": {"answer_type": "math_expression"},
        "gold": "13535",
        "strategy": "direct",
        "answer": None,
        "correct": False,
        "confidence": 0.0,
        "rationale": "",
        "parse_ok": False,
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "api_calls": 1,
        "estimated_cost_usd": 0.01,
        "calls": [
            {
                "role": "direct",
                "answer": None,
                "confidence": 0.0,
                "rationale": "",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                "parse_ok": False,
                "raw_response": "<think>work. The answer is $13535$.</think>",
            }
        ],
        "metadata": {},
    }

    repaired = _reparse_records_from_calls([record])[0]

    assert repaired["answer"] == "13535"
    assert repaired["parse_ok"] is True
    assert repaired["correct"] is True
    assert repaired["estimated_cost_usd"] == 0.01
    assert repaired["calls"][0]["answer"] == "13535"
