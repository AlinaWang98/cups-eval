from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

@dataclass(frozen=True)
class BatchJob:
    dataset_name: str
    model_profile: str
    config: dict[str, Any]


@dataclass(frozen=True)
class BatchResult:
    batch_dir: Path
    commands: list[list[str]]
    return_codes: list[int]


def expand_jobs(matrix_config_path: Path | str) -> list[BatchJob]:
    matrix_path = Path(matrix_config_path)
    matrix = _read_yaml(matrix_path)
    base_path = _resolve_path(matrix_path, str(matrix["base_config"]))
    base_config = _read_yaml(base_path)
    model_profiles = [str(item) for item in matrix["model_profiles"]]
    dataset_paths = [
        _resolve_path(matrix_path, str(item))
        for item in matrix["dataset_configs"]
    ]

    jobs: list[BatchJob] = []
    for dataset_path in dataset_paths:
        dataset_payload = _read_yaml(dataset_path)
        dataset_name = str(dataset_payload.get("dataset_name") or dataset_path.stem)
        dataset_override = {
            key: value
            for key, value in dataset_payload.items()
            if key != "dataset_name"
        }
        for profile in model_profiles:
            config = _deep_merge(base_config, dataset_override)
            llm = dict(config.get("llm", {}))
            llm["active_profile"] = profile
            config["llm"] = llm
            jobs.append(
                BatchJob(
                    dataset_name=dataset_name,
                    model_profile=profile,
                    config=config,
                )
            )
    return jobs


def run_batch(
    matrix_config_path: Path | str,
    fake_llm: bool = False,
    dry_run: bool = False,
    sample_size: int | None = None,
    max_usd: float | None = None,
    workers: int | None = None,
    max_parallel_runs: int | None = None,
    resume_batch_dir: Path | str | None = None,
) -> BatchResult:
    matrix_path = Path(matrix_config_path)
    matrix = _read_yaml(matrix_path)
    jobs = expand_jobs(matrix_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(str(matrix.get("output_root", "runs/batches")))
    batch_dir = Path(resume_batch_dir) if resume_batch_dir is not None else output_root / timestamp
    config_dir = batch_dir / "configs"
    log_dir = batch_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    workers_per_run = int(workers if workers is not None else matrix.get("workers_per_run", 1))
    model_workers = matrix.get("model_workers", {})
    if model_workers is None:
        model_workers = {}
    if not isinstance(model_workers, dict):
        raise ValueError("model_workers must be a mapping when provided")
    parallel_runs = int(
        max_parallel_runs
        if max_parallel_runs is not None
        else matrix.get("max_parallel_runs", 1)
    )
    commands: list[list[str]] = []
    job_records: list[dict[str, Any]] = []
    for job in jobs:
        config_path = config_dir / f"{job.dataset_name}.{job.model_profile}.yaml"
        _write_yaml(config_path, job.config)
        run_output_root = batch_dir / "runs" / job.dataset_name / job.model_profile
        job_workers = int(model_workers.get(job.model_profile, workers_per_run))
        command = [
            sys.executable,
            "-m",
            "cups_eval",
            "run",
            "--config",
            str(config_path),
            "--output-root",
            str(run_output_root),
            "--workers",
            str(job_workers),
        ]
        if fake_llm:
            command.append("--fake-llm")
        if sample_size is not None:
            command.extend(["--sample-size", str(sample_size)])
        if max_usd is not None:
            command.extend(["--max-usd", str(max_usd)])
        resume_run_dir = _latest_run_dir(run_output_root)
        if resume_run_dir is not None:
            command.extend(["--resume-run-dir", str(resume_run_dir)])
        commands.append(command)
        job_records.append(
            {
                "dataset_name": job.dataset_name,
                "model_profile": job.model_profile,
                "config_path": str(config_path),
                "output_root": str(run_output_root),
                "log_path": str(log_dir / f"{job.dataset_name}.{job.model_profile}.log"),
                "command": command,
            }
        )

    (batch_dir / "batch_jobs.json").write_text(
        json.dumps(job_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if dry_run:
        return BatchResult(batch_dir=batch_dir, commands=commands, return_codes=[])

    return_codes = _run_commands(
        commands=commands,
        job_records=job_records,
        max_parallel_runs=parallel_runs,
    )
    return BatchResult(batch_dir=batch_dir, commands=commands, return_codes=return_codes)


def _latest_run_dir(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _run_commands(
    commands: list[list[str]],
    job_records: list[dict[str, Any]],
    max_parallel_runs: int,
) -> list[int]:
    def run_one(index: int) -> int:
        record = job_records[index]
        log_path = Path(str(record["log_path"]))
        env = dict(os.environ)
        env["CUPS_MODEL_PROFILE"] = str(record["model_profile"])
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(
                commands[index],
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        return process.returncode

    return_codes = [0 for _ in commands]
    with ThreadPoolExecutor(max_workers=max(1, max_parallel_runs)) as executor:
        futures = {executor.submit(run_one, index): index for index in range(len(commands))}
        for future in as_completed(futures):
            index = futures[future]
            return_codes[index] = future.result()
    if any(code != 0 for code in return_codes):
        failed = [
            f"{record['dataset_name']}/{record['model_profile']}={code}"
            for record, code in zip(job_records, return_codes)
            if code != 0
        ]
        raise RuntimeError("Batch jobs failed: " + ", ".join(failed))
    return return_codes


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _resolve_path(base_file: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_file.parent / path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
