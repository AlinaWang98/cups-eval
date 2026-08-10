from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cups_eval.config import estimate_cost_usd, load_config
from cups_eval.datasets import load_question_sample, sample_records
from cups_eval.answer_normalization import normalize_answer_for_question
from cups_eval.llm import FakeLLMClient, OpenAICompatibleLLMClient, parse_model_response
from cups_eval.metrics import compute_summary, is_correct_result, pareto_frontier
from cups_eval.router import BudgetAwareRouter, ProbeRouter
from cups_eval.router_analysis import (
    build_feature_rows,
    build_strata_rows,
    evaluate_candidate_rules,
    evaluate_train_test_rules,
)
from cups_eval.schema import Question, StrategyResult, Usage
from cups_eval.strategies import (
    CotStrategy,
    Debate2JudgeStrategy,
    DirectStrategy,
    PlannerVerifierStrategy,
    SelfConsistencyStrategy,
)


def run_experiment(
    config_path: Path,
    sample_size: int | None = None,
    fake_llm: bool = False,
    output_root: Path = Path("runs"),
    max_usd: float | None = None,
    dataset_records: list[dict[str, Any]] | None = None,
    resume_run_dir: Path | None = None,
    workers: int = 1,
) -> Path:
    config = load_config(config_path)
    dataset_config = config["dataset"]
    selected_size = sample_size or int(dataset_config["sample_size"])
    if dataset_records is None:
        questions = load_question_sample(dataset_config, sample_size=selected_size)
    else:
        questions = sample_records(
            dataset_records,
            sample_size=selected_size,
            seed=int(dataset_config["seed"]),
            choice_type=str(dataset_config.get("choice_type", "single")),
            cop_base=int(dataset_config.get("cop_base", 0)),
            dataset_type=str(dataset_config.get("type", "medmcqa")),
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = resume_run_dir if resume_run_dir is not None else output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, int(workers))
    llm = _make_llm(config, fake_llm) if worker_count == 1 else None
    strategies = _build_strategies(config)
    strategy_names = [strategy.name for strategy in strategies]
    existing_records = _read_raw_records(run_dir / "raw_predictions.jsonl")
    _validate_resume_records(existing_records, questions, strategy_names)
    completed = {
        (str(record["question_id"]), str(record["strategy"]))
        for record in existing_records
    }
    max_budget = max_usd if max_usd is not None else float(config.get("run", {}).get("max_usd", 0.0))
    total_cost = sum(float(record.get("estimated_cost_usd", 0.0)) for record in existing_records)
    if max_budget and total_cost > max_budget:
        raise RuntimeError(
            f"Cost budget exceeded: estimated ${total_cost:.4f} > ${max_budget:.4f}"
        )

    _write_run_metadata(
        run_dir=run_dir,
        config_path=config_path,
        config=config,
        questions=questions,
        strategy_names=strategy_names,
        fake_llm=fake_llm,
        max_usd=max_budget,
        status="running",
        resumed=resume_run_dir is not None,
        completed_records=len(existing_records),
        workers=worker_count,
    )

    raw_path = run_dir / "raw_predictions.jsonl"
    raw_mode = "a" if existing_records else "w"
    with raw_path.open(raw_mode, encoding="utf-8") as raw_handle:
        pending = [
            (index, question, strategy)
            for index, question in enumerate(questions, start=1)
            for strategy in strategies
            if (question.id, strategy.name) not in completed
        ]
        if worker_count == 1:
            total_cost = _run_pending_serial(
                pending=pending,
                raw_handle=raw_handle,
                llm=llm,
                config=config,
                fake_llm=fake_llm,
                completed=completed,
                total_cost=total_cost,
                max_budget=max_budget,
                question_count=len(questions),
            )
        else:
            total_cost = _run_pending_parallel(
                pending=pending,
                raw_handle=raw_handle,
                config=config,
                fake_llm=fake_llm,
                completed=completed,
                total_cost=total_cost,
                max_budget=max_budget,
                question_count=len(questions),
                workers=worker_count,
            )

    analyze_run(run_dir)
    _write_run_metadata(
        run_dir=run_dir,
        config_path=config_path,
        config=config,
        questions=questions,
        strategy_names=strategy_names,
        fake_llm=fake_llm,
        max_usd=max_budget,
        status="complete",
        resumed=resume_run_dir is not None,
        completed_records=len(completed),
        workers=worker_count,
    )
    return run_dir


def _make_llm(config: dict[str, Any], fake_llm: bool) -> Any:
    return (
        FakeLLMClient(mode="always_correct")
        if fake_llm
        else OpenAICompatibleLLMClient.from_config(config)
    )


def _build_strategies(config: dict[str, Any]) -> list[Any]:
    available = {
        "direct": DirectStrategy(),
        "cot": CotStrategy(),
        "self_consistency_3": SelfConsistencyStrategy(samples=3),
        "debate_2_judge": Debate2JudgeStrategy(),
        "planner_verifier": PlannerVerifierStrategy(),
        "router": BudgetAwareRouter(
            confidence_threshold=float(config.get("router", {}).get("confidence_threshold", 0.70))
        ),
        "probe_router": ProbeRouter(
            confidence_threshold=float(
                config.get("probe_router", {}).get("confidence_threshold", 0.70)
            ),
            verifier_threshold=float(
                config.get("probe_router", {}).get("verifier_threshold", 0.65)
            ),
        ),
    }
    requested = (config.get("run", {}) or {}).get("strategies")
    if requested is None:
        return list(available.values())
    names = [str(name) for name in requested]
    unknown = [name for name in names if name not in available]
    if unknown:
        known = ", ".join(sorted(available))
        raise ValueError(
            f"Unknown run.strategies entries {unknown!r}; known strategies: {known}"
        )
    return [available[name] for name in names]


def _run_pending_serial(
    pending: list[tuple[int, Question, Any]],
    raw_handle: Any,
    llm: Any,
    config: dict[str, Any],
    fake_llm: bool,
    completed: set[tuple[str, str]],
    total_cost: float,
    max_budget: float,
    question_count: int,
) -> float:
    last_index: int | None = None
    for index, question, strategy in pending:
        if index != last_index:
            print(f"[{index}/{question_count}] {question.id}", flush=True)
            last_index = index
        record, cost = _run_strategy_record_with_timeout(
            question=question,
            strategy=strategy,
            llm=llm,
            config=config,
            fake_llm=fake_llm,
        )
        total_cost = _append_completed_record(
            raw_handle=raw_handle,
            completed=completed,
            record=record,
            cost=cost,
            total_cost=total_cost,
            max_budget=max_budget,
        )
    return total_cost


def _run_pending_parallel(
    pending: list[tuple[int, Question, Any]],
    raw_handle: Any,
    config: dict[str, Any],
    fake_llm: bool,
    completed: set[tuple[str, str]],
    total_cost: float,
    max_budget: float,
    question_count: int,
    workers: int,
) -> float:
    if not pending:
        return total_cost

    print(
        f"Running {len(pending)} pending strategy rows with {workers} workers.",
        flush=True,
    )
    thread_local = threading.local()

    def worker_llm() -> Any:
        if not hasattr(thread_local, "llm"):
            thread_local.llm = _make_llm(config, fake_llm)
        return thread_local.llm

    def run_in_worker(question: Question, strategy: Any) -> tuple[dict[str, Any], float]:
        return _run_strategy_record_with_timeout(
            question=question,
            strategy=strategy,
            llm=worker_llm(),
            config=config,
            fake_llm=fake_llm,
        )

    completed_rows = 0
    last_printed_index = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_in_worker, question, strategy): (
                index,
                question.id,
                strategy.name,
            )
            for index, question, strategy in pending
        }
        for future in as_completed(futures):
            index, question_id, strategy_name = futures[future]
            record, cost = future.result()
            total_cost = _append_completed_record(
                raw_handle=raw_handle,
                completed=completed,
                record=record,
                cost=cost,
                total_cost=total_cost,
                max_budget=max_budget,
            )
            completed_rows += 1
            if index > last_printed_index:
                print(
                    f"[{index}/{question_count}] {question_id} "
                    f"({completed_rows}/{len(pending)} rows)",
                    flush=True,
                )
                last_printed_index = index
            elif completed_rows % 50 == 0:
                print(
                    f"{completed_rows}/{len(pending)} pending rows complete.",
                    flush=True,
                )
    return total_cost


def _run_strategy_record(
    question: Question,
    strategy: Any,
    llm: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    result = strategy.run(question, llm)
    cost = estimate_cost_usd(
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
        config.get("pricing", {}),
    )
    result.metadata.update(
        {
            "gold": question.answer,
            "api_calls": len(result.calls),
            "estimated_cost_usd": cost,
            "question_id": question.id,
        }
    )
    record = result.to_record(question)
    record["estimated_cost_usd"] = result.metadata["estimated_cost_usd"]
    return record, cost


def _run_strategy_record_with_timeout(
    question: Question,
    strategy: Any,
    llm: Any,
    config: dict[str, Any],
    fake_llm: bool,
) -> tuple[dict[str, Any], float]:
    timeout_seconds = float(config.get("run", {}).get("strategy_timeout_seconds") or 0.0)
    force_fake_timeout = bool(
        config.get("run", {}).get("force_strategy_timeout_for_fake", False)
    )
    if (
        timeout_seconds <= 0
        or (fake_llm and not force_fake_timeout)
        or (
            not fake_llm
            and not isinstance(llm, OpenAICompatibleLLMClient)
        )
    ):
        return _run_strategy_record(question, strategy, llm, config)

    context = mp.get_context("spawn")
    queue: Any = context.Queue(maxsize=1)
    process = context.Process(
        target=_run_strategy_record_child,
        args=(queue, question, strategy, config, fake_llm),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        return _timeout_record(question, strategy, timeout_seconds), 0.0

    if process.exitcode != 0 and queue.empty():
        raise RuntimeError(
            f"Strategy worker exited with code {process.exitcode}: "
            f"{question.id}/{strategy.name}"
        )
    try:
        status, payload = queue.get_nowait()
    except Exception as exc:
        raise RuntimeError(
            f"Strategy worker produced no result: {question.id}/{strategy.name}"
        ) from exc
    if status == "ok":
        return payload
    raise RuntimeError(str(payload))


def _run_strategy_record_child(
    queue: Any,
    question: Question,
    strategy: Any,
    config: dict[str, Any],
    fake_llm: bool,
) -> None:
    try:
        llm = _make_llm(config, fake_llm)
        queue.put(("ok", _run_strategy_record(question, strategy, llm, config)))
    except Exception as exc:
        queue.put(("error", f"{exc.__class__.__name__}: {exc}"))


def _timeout_record(
    question: Question,
    strategy: Any,
    timeout_seconds: float,
) -> dict[str, Any]:
    result = StrategyResult(
        strategy=strategy.name,
        answer=None,
        confidence=0.0,
        rationale=f"strategy timed out after {timeout_seconds:.1f}s",
        calls=[],
        usage=Usage(),
        parse_ok=False,
        metadata={
            "gold": question.answer,
            "api_calls": 0,
            "estimated_cost_usd": 0.0,
            "question_id": question.id,
            "error_type": "strategy_timeout",
            "timeout_seconds": timeout_seconds,
        },
    )
    record = result.to_record(question)
    record["estimated_cost_usd"] = 0.0
    return record


def _append_completed_record(
    raw_handle: Any,
    completed: set[tuple[str, str]],
    record: dict[str, Any],
    cost: float,
    total_cost: float,
    max_budget: float,
) -> float:
    total_cost += cost
    if max_budget and total_cost > max_budget:
        raise RuntimeError(
            f"Cost budget exceeded: estimated ${total_cost:.4f} > ${max_budget:.4f}"
        )
    raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    raw_handle.flush()
    completed.add((str(record["question_id"]), str(record["strategy"])))
    return total_cost


def analyze_run(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _reparse_records_from_calls(_read_raw_records(run_dir / "raw_predictions.jsonl"))
    summary = _summary_from_records(rows)
    _write_item_metrics(run_dir / "item_metrics.csv", rows)
    _write_router_analysis_files(run_dir, rows)
    _write_dict_csv(run_dir / "routing_decisions.csv", _build_routing_decision_rows(rows))
    _write_dict_csv(run_dir / "cost_audit.csv", _build_cost_audit_rows(rows))
    _write_dict_csv(run_dir / "paired_stats.csv", _build_paired_stats_rows(rows))
    _write_summary_files(run_dir, summary)
    return summary


def _reparse_records_from_calls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_reparse_record_from_calls(record) for record in records]


def _reparse_record_from_calls(record: dict[str, Any]) -> dict[str, Any]:
    calls = record.get("calls") or []
    if not calls:
        return dict(record)

    answer_type = str(
        (record.get("question_metadata") or {}).get("answer_type", "choice")
    )
    valid_options = tuple((record.get("options") or {}).keys())
    repaired_calls = [
        _reparse_call_dict(call, valid_options=valid_options, answer_type=answer_type)
        for call in calls
    ]
    answer, confidence, rationale, parse_ok, metadata_updates = _select_reparsed_answer(
        str(record.get("strategy", "")),
        repaired_calls,
    )

    repaired = dict(record)
    repaired["calls"] = repaired_calls
    repaired["answer"] = answer
    repaired["confidence"] = confidence
    repaired["rationale"] = rationale
    repaired["parse_ok"] = parse_ok
    repaired["correct"] = (
        normalize_answer_for_question(answer, answer_type)
        == normalize_answer_for_question(str(record.get("gold", "")), answer_type)
    )
    if metadata_updates:
        metadata = dict(repaired.get("metadata") or {})
        metadata.update(metadata_updates)
        repaired["metadata"] = metadata
    return repaired


def _reparse_call_dict(
    call: dict[str, Any],
    valid_options: tuple[str, ...],
    answer_type: str,
) -> dict[str, Any]:
    repaired = dict(call)
    answer, confidence, rationale, parse_ok = parse_model_response(
        str(call.get("raw_response") or ""),
        valid_options=valid_options,
        answer_type=answer_type,
    )
    repaired["answer"] = answer
    repaired["confidence"] = confidence
    repaired["rationale"] = rationale
    repaired["parse_ok"] = parse_ok
    return repaired


def _select_reparsed_answer(
    strategy: str,
    calls: list[dict[str, Any]],
) -> tuple[str | None, float, str, bool, dict[str, Any]]:
    if strategy in {"direct", "cot"}:
        return _call_answer_tuple(calls[0]) if calls else (None, 0.0, "", False, {})

    if strategy == "planner_verifier":
        planner = calls[0] if len(calls) > 0 else {}
        verifier = calls[1] if len(calls) > 1 else {}
        selected = verifier if verifier.get("answer") else planner
        answer, confidence, rationale, parse_ok, _ = _call_answer_tuple(selected)
        return (
            answer,
            confidence,
            rationale,
            parse_ok,
            {
                "planner_answer": planner.get("answer"),
                "verifier_answer": verifier.get("answer"),
            },
        )

    if strategy == "debate_2_judge":
        agent_a = calls[0] if len(calls) > 0 else {}
        agent_b = calls[1] if len(calls) > 1 else {}
        judge = calls[2] if len(calls) > 2 else {}
        selected = judge if judge.get("answer") else agent_a if agent_a.get("answer") else agent_b
        answer, confidence, rationale, parse_ok, _ = _call_answer_tuple(selected)
        if not judge.get("answer") and answer:
            confidence = max(
                float(agent_a.get("confidence", 0.0) or 0.0),
                float(agent_b.get("confidence", 0.0) or 0.0),
            )
            rationale = rationale or "judge parse failed; used first agent fallback"
        return (
            answer,
            confidence,
            rationale,
            parse_ok,
            {"agent_answers": [agent_a.get("answer"), agent_b.get("answer")]},
        )

    if len(calls) == 1:
        return _call_answer_tuple(calls[0])
    return (None, 0.0, "", False, {})


def _call_answer_tuple(
    call: dict[str, Any],
) -> tuple[str | None, float, str, bool, dict[str, Any]]:
    answer = call.get("answer")
    return (
        str(answer) if answer is not None else None,
        float(call.get("confidence", 0.0) or 0.0),
        str(call.get("rationale", "") or ""),
        bool(call.get("parse_ok")),
        {},
    )


def _read_raw_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _validate_resume_records(
    records: list[dict[str, Any]],
    questions: list[Question],
    strategy_names: list[str],
) -> None:
    question_ids = {question.id for question in questions}
    valid_strategies = set(strategy_names)
    for record in records:
        question_id = str(record.get("question_id"))
        strategy = str(record.get("strategy"))
        if question_id not in question_ids:
            raise ValueError(
                f"Resume row question_id={question_id!r} is not part of the requested sample."
            )
        if strategy not in valid_strategies:
            raise ValueError(
                f"Resume row strategy={strategy!r} is not part of this run's strategy set."
            )


def _write_run_metadata(
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    questions: list[Question],
    strategy_names: list[str],
    fake_llm: bool,
    max_usd: float,
    status: str,
    resumed: bool,
    completed_records: int,
    workers: int,
) -> None:
    config_bytes = config_path.read_bytes()
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset": {
            "name": config.get("dataset", {}).get("name"),
            "split": config.get("dataset", {}).get("split"),
            "seed": config.get("dataset", {}).get("seed"),
            "sample_size": len(questions),
            "question_ids": [question.id for question in questions],
        },
        "strategies": strategy_names,
        "llm": _freeze_llm_config(config.get("llm", {})),
        "pricing": config.get("pricing", {}),
        "code_fingerprint_sha256": _code_fingerprint(),
        "git": _git_metadata(),
        "fake_llm": fake_llm,
        "max_usd": max_usd,
        "resumed": resumed,
        "completed_records": completed_records,
        "workers": workers,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _freeze_llm_config(llm_config: dict[str, Any]) -> dict[str, Any]:
    model_env = str(llm_config.get("model_env", "OPENAI_MODEL"))
    base_url_env = llm_config.get("base_url_env")
    base_url = (
        _env_or_default(str(base_url_env), str(llm_config.get("base_url", "")))
        if base_url_env
        else llm_config.get("base_url")
    )
    return {
        "profile": llm_config.get("profile"),
        "provider": llm_config.get("provider"),
        "base_url": base_url,
        "base_url_env": base_url_env,
        "api_key_env": llm_config.get("api_key_env"),
        "model_env": model_env,
        "model": _env_or_default(model_env, str(llm_config.get("model", ""))),
        "temperature": llm_config.get("temperature"),
        "force_temperature": llm_config.get("force_temperature"),
        "max_tokens": llm_config.get("max_tokens"),
        "timeout_seconds": llm_config.get("timeout_seconds"),
        "retry_attempts": llm_config.get("retry_attempts"),
        "retry_backoff_seconds": llm_config.get("retry_backoff_seconds"),
        "thinking": llm_config.get("thinking"),
    }


def _env_or_default(env_name: str, default: str) -> str:
    value = os.environ.get(env_name)
    return value if value else default


def _code_fingerprint() -> str:
    hasher = hashlib.sha256()
    paths = [Path("pyproject.toml"), Path("uv.lock")]
    paths.extend(sorted(Path("cups_eval").glob("*.py")))
    paths.extend(sorted(Path("configs").glob("*.yaml")))
    for path in paths:
        if not path.exists():
            continue
        hasher.update(str(path).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=Path.cwd(),
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
        return {"available": True, "commit": commit, "status_short": status}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _write_outputs(
    run_dir: Path,
    records: list[dict[str, Any]],
    results: list[StrategyResult],
) -> None:
    _write_item_metrics(run_dir / "item_metrics.csv", records)
    _write_router_analysis_files(run_dir, records)
    _write_dict_csv(run_dir / "routing_decisions.csv", _build_routing_decision_rows(records))
    _write_dict_csv(run_dir / "cost_audit.csv", _build_cost_audit_rows(records))
    _write_dict_csv(run_dir / "paired_stats.csv", _build_paired_stats_rows(records))
    summary = _summary_from_results(results)
    _write_summary_files(run_dir, summary)


def _write_router_analysis_files(
    run_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    features = build_feature_rows(records)
    _write_dict_csv(run_dir / "router_features.csv", features)
    _write_dict_csv(run_dir / "rule_search.csv", evaluate_candidate_rules(features))
    _write_dict_csv(
        run_dir / "learned_rule_eval.csv",
        evaluate_train_test_rules(features),
    )
    _write_dict_csv(run_dir / "strata_summary.csv", build_strata_rows(features))


def _build_routing_decision_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_question[str(record["question_id"])][str(record["strategy"])] = record

    rows: list[dict[str, Any]] = []
    for question_id, strategies in sorted(by_question.items()):
        if "direct" not in strategies:
            continue
        oracle = _oracle_select_record(strategies)
        router = strategies.get("router", {})
        probe_router = strategies.get("probe_router", {})
        direct = strategies["direct"]
        rows.append(
            {
                "question_id": question_id,
                "gold": direct.get("gold"),
                "direct_answer": direct.get("answer"),
                "direct_correct": bool(direct.get("correct")),
                "direct_cost_usd": float(direct.get("estimated_cost_usd", 0.0) or 0.0),
                "direct_api_calls": int(direct.get("api_calls", 0) or 0),
                "oracle_choice": oracle.get("strategy"),
                "oracle_escalated": oracle.get("strategy") != "direct",
                "best_strategy_per_item": oracle.get("strategy"),
                "oracle_answer": oracle.get("answer"),
                "oracle_correct": bool(oracle.get("correct")),
                "oracle_cost_usd": float(oracle.get("estimated_cost_usd", 0.0) or 0.0),
                "oracle_api_calls": int(oracle.get("api_calls", 0) or 0),
                "router_choice": _final_strategy(router),
                "router_answer": router.get("answer"),
                "router_correct": bool(router.get("correct")),
                "router_cost_usd": float(router.get("estimated_cost_usd", 0.0) or 0.0),
                "router_api_calls": int(router.get("api_calls", 0) or 0),
                "probe_router_choice": _final_strategy(probe_router),
                "probe_router_answer": probe_router.get("answer"),
                "probe_router_correct": bool(probe_router.get("correct")),
                "probe_router_cost_usd": float(
                    probe_router.get("estimated_cost_usd", 0.0) or 0.0
                ),
                "probe_router_api_calls": int(probe_router.get("api_calls", 0) or 0),
            }
        )
    return rows


def _oracle_select_record(strategies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = {"direct", "cot", "self_consistency_3", "planner_verifier", "debate_2_judge"}
    candidates = [
        record
        for name, record in strategies.items()
        if name in baseline
    ]
    correct = [record for record in candidates if record.get("correct")]
    pool = correct or candidates
    if not pool:
        return {"strategy": "", "correct": False, "estimated_cost_usd": 0.0, "api_calls": 0}
    return min(
        pool,
        key=lambda record: (
            float(record.get("estimated_cost_usd", 0.0) or 0.0),
            int(record.get("api_calls", 0) or 0),
            str(record.get("strategy", "")),
        ),
    )


def _final_strategy(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(metadata.get("final_strategy") or record.get("strategy") or "")


def _build_cost_audit_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        calls = record.get("calls") or []
        usage = record.get("usage") or {}
        call_prompt = sum(
            int((call.get("usage") or {}).get("prompt_tokens", 0) or 0)
            for call in calls
        )
        call_completion = sum(
            int((call.get("usage") or {}).get("completion_tokens", 0) or 0)
            for call in calls
        )
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        api_calls_reported = int(record.get("api_calls", 0) or 0)
        call_count = len(calls)
        rows.append(
            {
                "question_id": record.get("question_id"),
                "strategy": record.get("strategy"),
                "api_calls_reported": api_calls_reported,
                "call_count": call_count,
                "api_calls_match": api_calls_reported == call_count,
                "prompt_tokens_reported": prompt_tokens,
                "completion_tokens_reported": completion_tokens,
                "calls_prompt_tokens_sum": call_prompt,
                "calls_completion_tokens_sum": call_completion,
                "usage_matches_call_sum": (
                    prompt_tokens == call_prompt
                    and completion_tokens == call_completion
                ),
                "estimated_cost_usd": float(record.get("estimated_cost_usd", 0.0) or 0.0),
                "route": ",".join((record.get("metadata") or {}).get("route", [])),
                "roles": ",".join(str(call.get("role", "")) for call in calls),
            }
        )
    return rows


def _build_paired_stats_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_question[str(record["question_id"])][str(record["strategy"])] = record

    strategies = sorted(
        {
            str(record["strategy"])
            for record in records
            if str(record["strategy"]) not in {"direct", "oracle_router"}
        }
    )
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        direct_flags: list[bool] = []
        strategy_flags: list[bool] = []
        direct_cost = 0.0
        strategy_cost = 0.0
        for item in by_question.values():
            if "direct" not in item or strategy not in item:
                continue
            direct = item["direct"]
            candidate = item[strategy]
            direct_correct = bool(direct.get("correct"))
            candidate_correct = bool(candidate.get("correct"))
            direct_flags.append(direct_correct)
            strategy_flags.append(candidate_correct)
            direct_cost += float(direct.get("estimated_cost_usd", 0.0) or 0.0)
            strategy_cost += float(candidate.get("estimated_cost_usd", 0.0) or 0.0)

        n = len(direct_flags)
        gains = sum(
            1
            for direct_correct, strategy_correct in zip(direct_flags, strategy_flags)
            if not direct_correct and strategy_correct
        )
        harms = sum(
            1
            for direct_correct, strategy_correct in zip(direct_flags, strategy_flags)
            if direct_correct and not strategy_correct
        )
        both_correct = sum(
            1
            for direct_correct, strategy_correct in zip(direct_flags, strategy_flags)
            if direct_correct and strategy_correct
        )
        both_wrong = sum(
            1
            for direct_correct, strategy_correct in zip(direct_flags, strategy_flags)
            if not direct_correct and not strategy_correct
        )
        direct_accuracy = sum(direct_flags) / n if n else 0.0
        strategy_accuracy = sum(strategy_flags) / n if n else 0.0
        ci_low, ci_high = _bootstrap_accuracy_diff_ci(
            direct_flags,
            strategy_flags,
        )
        p_value = _exact_paired_binomial_p(gains, harms)
        rows.append(
            {
                "strategy": strategy,
                "n": n,
                "direct_accuracy": direct_accuracy,
                "strategy_accuracy": strategy_accuracy,
                "accuracy_delta": strategy_accuracy - direct_accuracy,
                "gains": gains,
                "harms": harms,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "discordant": gains + harms,
                "exact_p_value": p_value,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "direct_cost_usd": direct_cost,
                "strategy_cost_usd": strategy_cost,
                "cost_multiplier": _safe_div(strategy_cost, direct_cost),
                "significant_at_0_05": p_value <= 0.05,
            }
        )
    return rows


def _exact_paired_binomial_p(gains: int, harms: int) -> float:
    discordant = gains + harms
    if discordant == 0:
        return 1.0
    tail = min(gains, harms)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2)
        for k in range(tail + 1)
    ]
    max_log = max(log_terms)
    probability = math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)
    return min(1.0, 2.0 * probability)


def _bootstrap_accuracy_diff_ci(
    direct_flags: list[bool],
    strategy_flags: list[bool],
    resamples: int = 2000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    n = len(direct_flags)
    if n == 0:
        return None, None
    deltas = [
        int(strategy_correct) - int(direct_correct)
        for direct_correct, strategy_correct in zip(direct_flags, strategy_flags)
    ]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        total = 0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    low_index = int(0.025 * (resamples - 1))
    high_index = int(0.975 * (resamples - 1))
    return samples[low_index], samples[high_index]


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_from_results(results: list[StrategyResult]) -> dict[str, dict[str, Any]]:
    oracle_accuracy = _oracle_accuracy(results)
    with_oracle = results + _oracle_router_results(results)
    return compute_summary(with_oracle, baseline_strategy="direct", oracle_accuracy=oracle_accuracy)


def _summary_from_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from cups_eval.schema import Usage

    results = []
    for record in records:
        usage_payload = record["usage"]
        result = StrategyResult(
            strategy=record["strategy"],
            answer=record["answer"],
            confidence=float(record["confidence"]),
            rationale=record.get("rationale", ""),
            calls=[],
            usage=Usage(
                prompt_tokens=int(usage_payload["prompt_tokens"]),
                completion_tokens=int(usage_payload["completion_tokens"]),
            ),
            parse_ok=bool(record["parse_ok"]),
            metadata={
                "gold": record["gold"],
                "correct": bool(record["correct"]),
                "api_calls": int(record["api_calls"]),
                "estimated_cost_usd": float(record["estimated_cost_usd"]),
                "question_id": record["question_id"],
            },
        )
        results.append(result)
    return _summary_from_results(results)


def _oracle_accuracy(results: list[StrategyResult]) -> float:
    by_question: dict[str, list[StrategyResult]] = defaultdict(list)
    for result in results:
        by_question[str(result.metadata["question_id"])].append(result)
    if not by_question:
        return 0.0
    oracle_correct = 0
    for items in by_question.values():
        if any(
            item.strategy not in {"router", "probe_router", "oracle_router"}
            and is_correct_result(item)
            for item in items
        ):
            oracle_correct += 1
    return oracle_correct / len(by_question)


def _oracle_router_results(results: list[StrategyResult]) -> list[StrategyResult]:
    by_question: dict[str, list[StrategyResult]] = defaultdict(list)
    for result in results:
        by_question[str(result.metadata["question_id"])].append(result)

    oracle_rows: list[StrategyResult] = []
    for items in by_question.values():
        candidates = [
            item
            for item in items
            if item.strategy not in {"router", "probe_router", "oracle_router"}
        ]
        if not candidates:
            continue
        correct = [item for item in candidates if is_correct_result(item)]
        pool = correct or candidates
        selected = min(
            pool,
            key=lambda item: (
                float(item.metadata.get("estimated_cost_usd", 0.0)),
                int(item.metadata.get("api_calls", len(item.calls))),
                item.strategy,
            ),
        )
        metadata = dict(selected.metadata)
        metadata["selected_strategy"] = selected.strategy
        oracle_rows.append(
            StrategyResult(
                strategy="oracle_router",
                answer=selected.answer,
                confidence=selected.confidence,
                rationale=f"oracle selected {selected.strategy}",
                calls=selected.calls,
                usage=selected.usage,
                parse_ok=selected.parse_ok,
                metadata=metadata,
            )
        )
    return oracle_rows


def _write_item_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "strategy",
        "gold",
        "answer",
        "correct",
        "confidence",
        "parse_ok",
        "api_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            usage = record["usage"]
            writer.writerow(
                {
                    "question_id": record["question_id"],
                    "strategy": record["strategy"],
                    "gold": record["gold"],
                    "answer": record["answer"],
                    "correct": record["correct"],
                    "confidence": record["confidence"],
                    "parse_ok": record["parse_ok"],
                    "api_calls": record["api_calls"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "estimated_cost_usd": record["estimated_cost_usd"],
                }
            )


def _write_summary_files(run_dir: Path, summary: dict[str, dict[str, Any]]) -> None:
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    deployable_summary = {
        strategy: item
        for strategy, item in summary.items()
        if strategy != "oracle_router"
    }
    frontier = pareto_frontier(deployable_summary)
    with (run_dir / "pareto.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["strategy", "accuracy", "estimated_cost_usd"],
        )
        writer.writeheader()
        writer.writerows(frontier)

    learned_eval = _read_csv_dicts(run_dir / "learned_rule_eval.csv")
    paired_stats = _read_csv_dicts(run_dir / "paired_stats.csv")
    (run_dir / "summary.md").write_text(
        _render_summary_markdown(summary, frontier, learned_eval, paired_stats),
        encoding="utf-8",
    )


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _render_summary_markdown(
    summary: dict[str, dict[str, Any]],
    frontier: list[dict[str, Any]],
    learned_eval: list[dict[str, Any]] | None = None,
    paired_stats: list[dict[str, Any]] | None = None,
) -> str:
    verdict = _verdict_block(summary, learned_eval or [], paired_stats or [])
    lines = [
        "# CUPS Evaluation Summary",
        "",
        "This is a feasibility smoke test, not a paper-level result.",
        "",
        "## Verdict",
        "",
        *verdict,
        "",
        "| Strategy | Accuracy | Cost USD | Tokens/Correct | Calls/Correct | Utility | Oracle Gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, item in summary.items():
        lines.append(
            "| {strategy} | {accuracy:.3f} | {cost:.6f} | {tokens} | {calls} | {utility} | {gap} |".format(
                strategy=strategy,
                accuracy=float(item["accuracy"]),
                cost=float(item["estimated_cost_usd"]),
                tokens=_fmt_optional(item["tokens_per_correct"]),
                calls=_fmt_optional(item["api_calls_per_correct"]),
                utility=_fmt_optional(item["collaboration_utility"]),
                gap=_fmt_optional(item["oracle_router_gap"]),
            )
        )
    lines.extend(["", "## Pareto Frontier", ""])
    for item in frontier:
        lines.append(
            f"- {item['strategy']}: accuracy={item['accuracy']:.3f}, "
            f"cost=${item['estimated_cost_usd']:.6f}"
        )
    if paired_stats:
        lines.extend(["", "## Paired Direct Comparisons", ""])
        lines.append(
            "| Strategy | Delta Accuracy | Gains | Harms | Exact p | 95% Bootstrap CI | Cost x Direct |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in paired_stats:
            lines.append(
                "| {strategy} | {delta} | {gains} | {harms} | {p_value} | [{ci_low}, {ci_high}] | {cost_multiplier} |".format(
                    strategy=row.get("strategy", ""),
                    delta=_fmt_optional(_float_from_row(row, "accuracy_delta")),
                    gains=row.get("gains", ""),
                    harms=row.get("harms", ""),
                    p_value=_fmt_optional(_float_from_row(row, "exact_p_value")),
                    ci_low=_fmt_optional(_float_from_row(row, "bootstrap_ci_low")),
                    ci_high=_fmt_optional(_float_from_row(row, "bootstrap_ci_high")),
                    cost_multiplier=_fmt_optional(_float_from_row(row, "cost_multiplier")),
                )
            )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `collaboration_utility` is accuracy gain over `direct` divided by extra estimated cost.",
            "- `oracle_router_gap` is oracle strategy accuracy minus the reported strategy accuracy.",
            "- The Pareto frontier excludes `oracle_router` because oracle uses post-hoc correctness labels.",
            "- Rules marked diagnostic-only are useful for boundary analysis but are excluded from deployable learned-router selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def _verdict_block(
    summary: dict[str, dict[str, Any]],
    learned_eval: list[dict[str, Any]],
    paired_stats: list[dict[str, Any]],
) -> list[str]:
    direct = summary.get("direct", {})
    oracle = summary.get("oracle_router", {})
    direct_accuracy = float(direct.get("accuracy", 0.0) or 0.0)
    direct_cost = float(direct.get("estimated_cost_usd", 0.0) or 0.0)
    oracle_accuracy = float(oracle.get("accuracy", direct_accuracy) or 0.0)
    oracle_cost = float(oracle.get("estimated_cost_usd", direct_cost) or 0.0)
    parse_success = _overall_parse_success(summary)
    best_collab_name, best_collab = _best_collaboration(summary)
    selected_learned = _selected_learned_row(learned_eval)
    learned_accuracy = _float_from_row(selected_learned, "test_accuracy")
    learned_cost = _float_from_row(selected_learned, "test_cost_usd")
    learned_multiplier = _float_from_row(selected_learned, "test_cost_multiplier")
    retained_gain = _float_from_row(selected_learned, "test_retained_oracle_gain")
    learned_deployable = _bool_from_row(selected_learned, "deployable", default=True)

    retained_enough = (
        learned_deployable
        and retained_gain is not None
        and retained_gain >= 0.40
        and learned_multiplier is not None
        and learned_multiplier <= 1.50
    )
    oracle_gap = oracle_accuracy - direct_accuracy
    router_signal = _best_router_signal(
        summary=summary,
        paired_stats=paired_stats,
        direct_accuracy=direct_accuracy,
        direct_cost=direct_cost,
        oracle_gap=oracle_gap,
    )
    router_supported = bool(router_signal and router_signal["supported"])
    if parse_success < 0.95:
        verdict = "not feasible"
        reason = "parse success is below the 95% MVP threshold"
    elif router_supported or retained_enough:
        verdict = "feasible"
        reason = "a deployable router is validated within the 1.5x direct-cost budget"
    elif oracle_gap >= 0.02:
        verdict = "conditionally feasible"
        reason = "oracle routing has meaningful headroom, but the current deployable router does not capture enough of it"
    else:
        verdict = "not feasible"
        reason = "oracle headroom over direct is too small under this cost-fair setup"

    lines = [
        f"- Final feasibility judgment: **{verdict}** ({reason}).",
        f"- Direct: accuracy={direct_accuracy:.3f}, cost=${direct_cost:.6f}.",
        f"- Oracle router: accuracy={oracle_accuracy:.3f}, cost=${oracle_cost:.6f}, gap={oracle_gap:.3f}.",
        f"- Parse success: {parse_success:.3f}.",
    ]
    if best_collab_name and best_collab:
        best_paired = _paired_row(paired_stats, best_collab_name)
        lines.append(
            "- Best non-oracle strategy: "
            f"{best_collab_name}, accuracy={float(best_collab['accuracy']):.3f}, "
            f"cost=${float(best_collab['estimated_cost_usd']):.6f}, "
            f"cost_multiplier={_fmt_optional(_safe_div(float(best_collab['estimated_cost_usd']), direct_cost))}, "
            f"paired_p={_fmt_optional(_float_from_row(best_paired, 'exact_p_value'))}."
        )
    if router_signal:
        lines.append(
            "- Current deployable router signal: "
            f"{router_signal['strategy']}, accuracy={router_signal['accuracy']:.3f}, "
            f"cost_multiplier={_fmt_optional(router_signal['cost_multiplier'])}, "
            f"retained_oracle_gain={_fmt_optional(router_signal['retained_oracle_gain'])}, "
            f"paired_p={_fmt_optional(router_signal['paired_p_value'])}."
        )
    if selected_learned:
        lines.append(
            "- Learned offline rule: "
            f"{selected_learned.get('rule')}, test_accuracy={_fmt_optional(learned_accuracy)}, "
            f"test_cost=${_fmt_optional(learned_cost)}, "
            f"cost_multiplier={_fmt_optional(learned_multiplier)}, "
            f"retained_oracle_gain={_fmt_optional(retained_gain)}, "
            f"deployable={learned_deployable}."
        )
    return lines


def _best_router_signal(
    summary: dict[str, dict[str, Any]],
    paired_stats: list[dict[str, Any]],
    direct_accuracy: float,
    direct_cost: float,
    oracle_gap: float,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for strategy in ("router", "probe_router"):
        item = summary.get(strategy)
        if not item:
            continue
        accuracy = float(item.get("accuracy", 0.0) or 0.0)
        cost = float(item.get("estimated_cost_usd", 0.0) or 0.0)
        gain = accuracy - direct_accuracy
        retained = _safe_div(gain, oracle_gap) if oracle_gap > 0 else None
        paired = _paired_row(paired_stats, strategy)
        p_value = _float_from_row(paired, "exact_p_value")
        cost_multiplier = _safe_div(cost, direct_cost)
        supported = (
            gain > 0
            and cost_multiplier is not None
            and cost_multiplier <= 1.50
            and (
                (p_value is not None and p_value <= 0.05)
                or (retained is not None and retained >= 0.40)
            )
        )
        candidates.append(
            {
                "strategy": strategy,
                "accuracy": accuracy,
                "cost_multiplier": cost_multiplier,
                "retained_oracle_gain": retained,
                "paired_p_value": p_value,
                "supported": supported,
            }
        )
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            not bool(item["supported"]),
            -float(item["accuracy"]),
            float(item["cost_multiplier"] or 999999.0),
            str(item["strategy"]),
        ),
    )[0]


def _paired_row(
    paired_stats: list[dict[str, Any]],
    strategy: str,
) -> dict[str, Any] | None:
    for row in paired_stats:
        if row.get("strategy") == strategy:
            return row
    return None


def _overall_parse_success(summary: dict[str, dict[str, Any]]) -> float:
    total = 0
    parsed = 0.0
    for strategy, item in summary.items():
        if strategy == "oracle_router":
            continue
        n = int(item.get("n", 0) or 0)
        total += n
        parsed += float(item.get("parse_success_rate", 0.0) or 0.0) * n
    return parsed / total if total else 0.0


def _best_collaboration(
    summary: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    candidates = {
        strategy: item
        for strategy, item in summary.items()
        if strategy not in {"direct", "oracle_router"}
    }
    if not candidates:
        return None, None
    name, item = sorted(
        candidates.items(),
        key=lambda pair: (
            -float(pair[1].get("accuracy", 0.0) or 0.0),
            float(pair[1].get("estimated_cost_usd", 0.0) or 0.0),
            pair[0],
        ),
    )[0]
    return name, item


def _selected_learned_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("selected", "")).lower() == "true":
            return row
    return None


def _float_from_row(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _bool_from_row(
    row: dict[str, Any] | None,
    key: str,
    default: bool = False,
) -> bool:
    if not row:
        return default
    value = row.get(key)
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"
