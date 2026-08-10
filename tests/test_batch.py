from pathlib import Path

import yaml

from cups_eval.batch import expand_jobs, run_batch


def write_yaml(path: Path, payload: dict):
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_expand_jobs_builds_dataset_model_matrix(tmp_path):
    base = tmp_path / "base.yaml"
    medqa = tmp_path / "medqa.yaml"
    matrix = tmp_path / "matrix.yaml"
    write_yaml(
        base,
        {
            "dataset": {"type": "medmcqa", "sample_size": 30, "seed": 42},
            "llm": {"active_profile": "deepseek_v4_pro", "profiles": {"deepseek_v4_pro": {}}},
        },
    )
    write_yaml(
        medqa,
        {
            "dataset_name": "medqa",
            "dataset": {
                "type": "medqa",
                "name": "GBaker/MedQA-USMLE-4-options",
                "split": "test",
                "sample_size": 300,
                "seed": 42,
            },
        },
    )
    write_yaml(
        matrix,
        {
            "base_config": str(base),
            "model_profiles": ["deepseek_v4_pro", "glm_5_2"],
            "dataset_configs": [str(medqa)],
        },
    )

    jobs = expand_jobs(matrix)

    assert [job.dataset_name for job in jobs] == ["medqa", "medqa"]
    assert [job.model_profile for job in jobs] == ["deepseek_v4_pro", "glm_5_2"]
    assert jobs[0].config["dataset"]["type"] == "medqa"
    assert jobs[1].config["llm"]["active_profile"] == "glm_5_2"
    assert "deepseek_v4_pro" in jobs[1].config["llm"]["profiles"]


def test_run_batch_dry_run_writes_job_configs_and_commands(tmp_path):
    base = tmp_path / "base.yaml"
    medmcqa = tmp_path / "medmcqa.yaml"
    matrix = tmp_path / "matrix.yaml"
    batch_root = tmp_path / "runs"
    write_yaml(
        base,
        {
            "dataset": {"type": "medmcqa", "sample_size": 5, "seed": 42},
            "llm": {
                "active_profile": "deepseek_v4_pro",
                "profiles": {"deepseek_v4_pro": {}},
            },
            "run": {"max_usd": 1.0},
        },
    )
    write_yaml(
        medmcqa,
        {
            "dataset_name": "medmcqa",
            "dataset": {
                "type": "medmcqa",
                "name": "openlifescienceai/medmcqa",
                "split": "validation",
                "sample_size": 5,
                "seed": 42,
            },
        },
    )
    write_yaml(
        matrix,
        {
            "base_config": str(base),
            "model_profiles": ["deepseek_v4_pro"],
            "dataset_configs": [str(medmcqa)],
            "output_root": str(batch_root),
            "workers_per_run": 2,
            "max_parallel_runs": 1,
        },
    )

    result = run_batch(matrix, fake_llm=True, dry_run=True)

    assert result.batch_dir.exists()
    assert len(result.commands) == 1
    assert "--workers" in result.commands[0]
    assert "2" in result.commands[0]
    assert (result.batch_dir / "configs" / "medmcqa.deepseek_v4_pro.yaml").exists()
    assert (result.batch_dir / "batch_jobs.json").exists()


def test_run_batch_uses_model_specific_worker_override(tmp_path):
    base = tmp_path / "base.yaml"
    medmcqa = tmp_path / "medmcqa.yaml"
    matrix = tmp_path / "matrix.yaml"
    batch_root = tmp_path / "runs"
    write_yaml(
        base,
        {
            "dataset": {"type": "medmcqa", "sample_size": 5, "seed": 42},
            "llm": {
                "active_profile": "deepseek_v4_pro",
                "profiles": {"deepseek_v4_pro": {}, "kimi_k3": {}},
            },
        },
    )
    write_yaml(
        medmcqa,
        {
            "dataset_name": "medmcqa",
            "dataset": {
                "type": "medmcqa",
                "name": "openlifescienceai/medmcqa",
                "split": "validation",
                "sample_size": 5,
                "seed": 42,
            },
        },
    )
    write_yaml(
        matrix,
        {
            "base_config": str(base),
            "model_profiles": ["deepseek_v4_pro", "kimi_k3"],
            "dataset_configs": [str(medmcqa)],
            "output_root": str(batch_root),
            "workers_per_run": 4,
            "model_workers": {"kimi_k3": 1},
        },
    )

    result = run_batch(matrix, fake_llm=True, dry_run=True)

    deepseek_workers = result.commands[0][result.commands[0].index("--workers") + 1]
    kimi_workers = result.commands[1][result.commands[1].index("--workers") + 1]
    assert deepseek_workers == "4"
    assert kimi_workers == "1"


def test_run_batch_resume_adds_existing_run_dir_to_commands(tmp_path):
    base = tmp_path / "base.yaml"
    medmcqa = tmp_path / "medmcqa.yaml"
    matrix = tmp_path / "matrix.yaml"
    batch_dir = tmp_path / "existing-batch"
    previous_run = batch_dir / "runs" / "medmcqa" / "deepseek_v4_pro" / "20260101T000000Z"
    previous_run.mkdir(parents=True)
    write_yaml(
        base,
        {
            "dataset": {"type": "medmcqa", "sample_size": 5, "seed": 42},
            "llm": {
                "active_profile": "deepseek_v4_pro",
                "profiles": {"deepseek_v4_pro": {}},
            },
        },
    )
    write_yaml(
        medmcqa,
        {
            "dataset_name": "medmcqa",
            "dataset": {
                "type": "medmcqa",
                "name": "openlifescienceai/medmcqa",
                "split": "validation",
                "sample_size": 5,
                "seed": 42,
            },
        },
    )
    write_yaml(
        matrix,
        {
            "base_config": str(base),
            "model_profiles": ["deepseek_v4_pro"],
            "dataset_configs": [str(medmcqa)],
            "output_root": str(tmp_path / "runs"),
        },
    )

    result = run_batch(
        matrix,
        fake_llm=True,
        dry_run=True,
        resume_batch_dir=batch_dir,
    )

    assert result.batch_dir == batch_dir
    assert "--resume-run-dir" in result.commands[0]
    assert str(previous_run) in result.commands[0]
