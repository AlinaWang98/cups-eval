from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cups_eval.answer_normalization import (
    normalize_math_expression_answer,
    normalize_numeric_answer,
)
from cups_eval.schema import OPTION_LETTERS, Question


def normalize_medmcqa_record(record: dict[str, Any], cop_base: int = 0) -> Question:
    cop = int(record["cop"])
    if cop_base == 0:
        if cop < 0 or cop > 3:
            raise ValueError(f"MedMCQA cop must be in 0..3, got {cop!r}")
        answer_index = cop
    elif cop_base == 1:
        if cop < 1 or cop > 4:
            raise ValueError(f"MedMCQA cop must be in 1..4, got {cop!r}")
        answer_index = cop - 1
    else:
        raise ValueError(f"Unsupported cop_base: {cop_base!r}")

    options = {
        "A": str(record["opa"]),
        "B": str(record["opb"]),
        "C": str(record["opc"]),
        "D": str(record["opd"]),
    }
    metadata = {
        key: record[key]
        for key in ("choice_type", "subject_name", "topic_name")
        if key in record
    }
    return Question(
        id=str(record.get("id", record.get("qid", ""))),
        question=str(record["question"]),
        options=options,
        answer=OPTION_LETTERS[answer_index],
        metadata=metadata,
    )


def normalize_medqa_record(
    record: dict[str, Any],
    fallback_id: str,
) -> Question:
    answer = str(record["answer_idx"]).strip().upper()
    options = _normalize_options(record["options"])
    _validate_answer(answer, options)
    metadata = {
        key: record[key]
        for key in ("meta_info", "metamap_phrases")
        if key in record
    }
    return Question(
        id=str(record.get("id") or record.get("qid") or fallback_id),
        question=str(record["question"]),
        options=options,
        answer=answer,
        metadata=metadata,
    )


def normalize_pubmedqa_record(record: dict[str, Any]) -> Question:
    decision = str(record["final_decision"]).strip().lower()
    answer_map = {"yes": "A", "no": "B", "maybe": "C"}
    if decision not in answer_map:
        raise ValueError(f"Unsupported PubMedQA final_decision: {decision!r}")
    context = record.get("context") or {}
    contexts = context.get("contexts") or []
    context_text = "\n".join(str(item) for item in contexts)
    question_text = str(record["question"])
    if context_text:
        question_text = f"{question_text}\n\nAbstract context:\n{context_text}"
    return Question(
        id=str(record.get("pubid", "")),
        question=question_text,
        options={"A": "yes", "B": "no", "C": "maybe"},
        answer=answer_map[decision],
        metadata={
            "pubid": record.get("pubid"),
            "dataset_task": "pubmedqa",
        },
    )


def normalize_medxpertqa_record(record: dict[str, Any]) -> Question:
    answer = str(record["label"]).strip().upper()
    options = _normalize_options(record["options"])
    _validate_answer(answer, options)
    metadata = {
        key: record[key]
        for key in ("medical_task", "body_system", "question_type")
        if key in record
    }
    return Question(
        id=str(record.get("id", "")),
        question=str(record["question"]),
        options=options,
        answer=answer,
        metadata=metadata,
    )


def normalize_mmlu_pro_record(
    record: dict[str, Any],
    fallback_id: str,
) -> Question:
    raw_options = list(record["options"])
    if len(raw_options) > len(OPTION_LETTERS):
        raise ValueError(f"MMLU-Pro supports at most {len(OPTION_LETTERS)} options")
    options = {
        OPTION_LETTERS[index]: str(option)
        for index, option in enumerate(raw_options)
    }
    answer = str(record.get("answer", "")).strip().upper()
    if not answer and "answer_index" in record:
        answer_index = int(record["answer_index"])
        answer = OPTION_LETTERS[answer_index]
    _validate_answer(answer, options)
    metadata = {
        key: record[key]
        for key in ("answer_index", "category", "src")
        if key in record
    }
    return Question(
        id=str(record.get("question_id") or record.get("id") or fallback_id),
        question=str(record["question"]),
        options=options,
        answer=answer,
        metadata=metadata,
    )


def normalize_gsm8k_answer(text: str) -> str:
    return normalize_numeric_answer(text)


def normalize_gsm8k_record(record: dict[str, Any], fallback_id: str) -> Question:
    return Question(
        id=str(record.get("id") or fallback_id),
        question=str(record["question"]),
        options={},
        answer=normalize_gsm8k_answer(str(record["answer"])),
        metadata={
            "answer_type": "numeric",
            "dataset_task": "gsm8k",
        },
    )


def normalize_math500_answer(text: str) -> str:
    return normalize_math_expression_answer(text)


def normalize_math500_record(record: dict[str, Any], fallback_id: str) -> Question:
    metadata = {
        "answer_type": "math_expression",
        "dataset_task": "math500",
    }
    if "subject" in record:
        metadata["subject_name"] = record["subject"]
    if "level" in record:
        metadata["level"] = record["level"]
    return Question(
        id=str(record.get("unique_id") or record.get("id") or fallback_id),
        question=str(record["problem"]),
        options={},
        answer=normalize_math500_answer(str(record["answer"])),
        metadata=metadata,
    )


def normalize_record(
    record: dict[str, Any],
    dataset_type: str,
    index: int,
    cop_base: int = 0,
) -> Question:
    if dataset_type == "medmcqa":
        return normalize_medmcqa_record(record, cop_base=cop_base)
    if dataset_type == "medqa":
        return normalize_medqa_record(record, fallback_id=f"medqa-{index}")
    if dataset_type == "pubmedqa":
        return normalize_pubmedqa_record(record)
    if dataset_type == "medxpertqa":
        return normalize_medxpertqa_record(record)
    if dataset_type == "mmlu_pro":
        return normalize_mmlu_pro_record(record, fallback_id=f"mmlu-pro-{index}")
    if dataset_type == "gsm8k":
        return normalize_gsm8k_record(record, fallback_id=f"gsm8k-{index}")
    if dataset_type == "math500":
        return normalize_math500_record(record, fallback_id=f"math500-{index}")
    raise ValueError(f"Unsupported dataset type: {dataset_type!r}")


def sample_records(
    records: Iterable[dict[str, Any]],
    sample_size: int,
    seed: int,
    choice_type: str = "single",
    cop_base: int = 0,
    dataset_type: str = "medmcqa",
) -> list[Question]:
    normalized = []
    for index, record in enumerate(records):
        if dataset_type == "medmcqa" and record.get("choice_type") != choice_type:
            continue
        normalized.append(
            normalize_record(
                record,
                dataset_type=dataset_type,
                index=index,
                cop_base=cop_base,
            )
        )
    if sample_size > len(normalized):
        raise ValueError(
            f"Requested {sample_size} records, but only {len(normalized)} records "
            f"are available for dataset_type={dataset_type!r}."
        )
    rng = random.Random(seed)
    shuffled = list(normalized)
    rng.shuffle(shuffled)
    return shuffled[:sample_size]


def load_medmcqa_sample(
    dataset_name: str,
    split: str,
    sample_size: int,
    seed: int,
    choice_type: str = "single",
    cop_base: int = 0,
) -> list[Question]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    return sample_records(dataset, sample_size, seed, choice_type, cop_base)


def load_question_sample(dataset_config: dict[str, Any], sample_size: int) -> list[Question]:
    records = _load_records(dataset_config)
    return sample_records(
        records,
        sample_size=sample_size,
        seed=int(dataset_config["seed"]),
        choice_type=str(dataset_config.get("choice_type", "single")),
        cop_base=int(dataset_config.get("cop_base", 0)),
        dataset_type=str(dataset_config.get("type", "medmcqa")),
    )


def _load_records(dataset_config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    split = str(dataset_config["split"])
    local_path = dataset_config.get("local_path")
    if local_path and _has_local_dataset_payload(Path(local_path)):
        from datasets import load_from_disk

        dataset = load_from_disk(str(Path(local_path)))
        return dataset[split] if hasattr(dataset, "keys") else dataset

    from datasets import load_dataset

    name = str(dataset_config["name"])
    config_name = dataset_config.get("config_name")
    if config_name:
        return load_dataset(name, str(config_name), split=split)
    return load_dataset(name, split=split)


def _has_local_dataset_payload(path: Path) -> bool:
    if not path.exists():
        return False
    return any(path.rglob("*.arrow")) or any(path.rglob("*.parquet"))


def _normalize_options(options: dict[str, Any]) -> dict[str, str]:
    normalized = {
        str(letter).upper(): str(text)
        for letter, text in options.items()
        if str(letter).upper() in OPTION_LETTERS
    }
    return dict(sorted(normalized.items()))


def _validate_answer(answer: str, options: dict[str, str]) -> None:
    if answer not in options:
        raise ValueError(
            f"Answer {answer!r} is not present in options {sorted(options)}"
        )
