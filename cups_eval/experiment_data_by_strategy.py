from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "source_batch",
    "dataset",
    "model",
    "run_id",
    "strategy",
    "question_id",
    "question",
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
    "rationale",
    "subject_name",
    "topic_name",
    "source_raw_predictions",
    "source_evaluated_artifact",
]

AUDIT_FIELDS = [
    "dataset",
    "model",
    "strategy",
    "question_id",
    "correction_category",
    "raw_answer",
    "raw_correct",
    "raw_parse_ok",
    "evaluated_answer",
    "evaluated_correct",
    "evaluated_parse_ok",
    "source_raw_predictions",
    "source_evaluated_artifact",
]


def build_experiment_data_by_strategy(
    *,
    project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.csv"
    manifest_rows = _read_csv(manifest_path)
    row_counts: dict[str, int] = {}
    audit_rows: list[dict[str, Any]] = []

    for manifest_row in manifest_rows:
        raw_rel = str(manifest_row.get("source_raw_predictions") or "")
        raw_path = project_root / raw_rel
        evaluated_path = raw_path.with_name("item_metrics.csv")
        evaluated_rel = (
            evaluated_path.relative_to(project_root).as_posix()
            if evaluated_path.exists()
            else ""
        )
        raw_rows = _read_jsonl(raw_path)
        evaluated_by_key = {
            _metric_key(row): row
            for row in _read_csv(evaluated_path)
        } if evaluated_path.exists() else {}

        output_rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            key = _metric_key(raw)
            evaluated = evaluated_by_key.get(key, {})
            output_rows.append(
                _canonical_row(
                    manifest_row,
                    raw,
                    evaluated,
                    raw_rel=raw_rel,
                    evaluated_rel=evaluated_rel,
                )
            )
            audit = _audit_row(
                manifest_row,
                raw,
                evaluated,
                raw_rel=raw_rel,
                evaluated_rel=evaluated_rel,
            )
            if audit:
                audit_rows.append(audit)

        output_path = project_root / str(manifest_row.get("output_file") or "")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(output_path, output_rows, OUTPUT_FIELDS)
        row_counts[str(output_path.relative_to(output_dir))] = len(output_rows)

    audit_dir = output_dir / "evaluation_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(audit_dir / "raw_vs_evaluated_corrections.csv", audit_rows, AUDIT_FIELDS)
    row_counts["raw_vs_evaluated_corrections"] = len(audit_rows)

    manifest = {
        "output_dir": output_dir.relative_to(project_root).as_posix()
        if _is_relative_to(output_dir, project_root)
        else str(output_dir),
        "row_counts": row_counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _canonical_row(
    manifest_row: dict[str, Any],
    raw: dict[str, Any],
    evaluated: dict[str, Any],
    *,
    raw_rel: str,
    evaluated_rel: str,
) -> dict[str, Any]:
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    metadata = raw.get("question_metadata") if isinstance(raw.get("question_metadata"), dict) else {}
    return {
        "source_batch": manifest_row.get("source_batch", ""),
        "dataset": manifest_row.get("dataset", ""),
        "model": manifest_row.get("model", ""),
        "run_id": manifest_row.get("run_id", ""),
        "strategy": str(evaluated.get("strategy") or raw.get("strategy") or manifest_row.get("strategy") or ""),
        "question_id": str(evaluated.get("question_id") or raw.get("question_id") or ""),
        "question": raw.get("question", ""),
        "gold": str(evaluated.get("gold") or raw.get("gold") or ""),
        "answer": str(evaluated.get("answer") or raw.get("answer") or ""),
        "correct": _bool_text(evaluated.get("correct", raw.get("correct", ""))),
        "confidence": str(evaluated.get("confidence", raw.get("confidence", ""))),
        "parse_ok": _bool_text(evaluated.get("parse_ok", raw.get("parse_ok", ""))),
        "api_calls": str(evaluated.get("api_calls", raw.get("api_calls", ""))),
        "prompt_tokens": str(evaluated.get("prompt_tokens", usage.get("prompt_tokens", ""))),
        "completion_tokens": str(evaluated.get("completion_tokens", usage.get("completion_tokens", ""))),
        "total_tokens": str(evaluated.get("total_tokens", usage.get("total_tokens", ""))),
        "estimated_cost_usd": str(evaluated.get("estimated_cost_usd", raw.get("estimated_cost_usd", ""))),
        "rationale": raw.get("rationale", ""),
        "subject_name": metadata.get("subject_name", raw.get("subject_name", "")),
        "topic_name": metadata.get("topic_name", raw.get("topic_name", "")),
        "source_raw_predictions": raw_rel,
        "source_evaluated_artifact": evaluated_rel,
    }


def _audit_row(
    manifest_row: dict[str, Any],
    raw: dict[str, Any],
    evaluated: dict[str, Any],
    *,
    raw_rel: str,
    evaluated_rel: str,
) -> dict[str, Any] | None:
    if not evaluated:
        return None
    raw_correct = _bool_text(raw.get("correct", ""))
    evaluated_correct = _bool_text(evaluated.get("correct", ""))
    raw_answer = str(raw.get("answer") or "")
    evaluated_answer = str(evaluated.get("answer") or "")
    raw_parse_ok = _bool_text(raw.get("parse_ok", ""))
    evaluated_parse_ok = _bool_text(evaluated.get("parse_ok", ""))
    if (
        raw_correct == evaluated_correct
        and raw_answer == evaluated_answer
        and raw_parse_ok == evaluated_parse_ok
    ):
        return None
    return {
        "dataset": manifest_row.get("dataset", ""),
        "model": manifest_row.get("model", ""),
        "strategy": str(evaluated.get("strategy") or raw.get("strategy") or manifest_row.get("strategy") or ""),
        "question_id": str(evaluated.get("question_id") or raw.get("question_id") or ""),
        "correction_category": _correction_category(raw, evaluated),
        "raw_answer": raw_answer,
        "raw_correct": raw_correct,
        "raw_parse_ok": raw_parse_ok,
        "evaluated_answer": evaluated_answer,
        "evaluated_correct": evaluated_correct,
        "evaluated_parse_ok": evaluated_parse_ok,
        "source_raw_predictions": raw_rel,
        "source_evaluated_artifact": evaluated_rel,
    }


def _correction_category(raw: dict[str, Any], evaluated: dict[str, Any]) -> str:
    gold = str(evaluated.get("gold") or raw.get("gold") or "")
    answer = str(evaluated.get("answer") or raw.get("answer") or "")
    if "\\in" in gold or "\\in" in answer or "[" in answer:
        return "interval_notation_equivalence"
    return "evaluated_metric_override"


def _metric_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("question_id") or ""), str(row.get("strategy") or ""))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value)
    if text.lower() == "true":
        return "True"
    if text.lower() == "false":
        return "False"
    return text


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
