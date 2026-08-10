from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cups_eval.answer_normalization import normalize_answer_for_question


OPTION_LETTERS = tuple("ABCDEFGHIJ")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    options: dict[str, str]
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def answer_type(self) -> str:
        return str(self.metadata.get("answer_type", "choice"))


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ModelCall:
    role: str
    prompt: str
    raw_response: str
    answer: str | None
    confidence: float
    rationale: str
    usage: Usage
    parse_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "answer": self.answer,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "usage": self.usage.to_dict(),
            "parse_ok": self.parse_ok,
            "raw_response": self.raw_response,
        }


@dataclass
class StrategyResult:
    strategy: str
    answer: str | None
    confidence: float
    rationale: str
    calls: list[ModelCall]
    usage: Usage
    parse_ok: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self, question: Question) -> dict[str, Any]:
        correct = normalize_answer_for_question(
            self.answer,
            question.answer_type,
        ) == normalize_answer_for_question(question.answer, question.answer_type)
        return {
            "question_id": question.id,
            "question": question.question,
            "options": question.options,
            "question_metadata": question.metadata,
            "gold": question.answer,
            "strategy": self.strategy,
            "answer": self.answer,
            "correct": correct,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "parse_ok": self.parse_ok,
            "usage": self.usage.to_dict(),
            "api_calls": len(self.calls),
            "calls": [call.to_dict() for call in self.calls],
            "metadata": self.metadata,
        }
