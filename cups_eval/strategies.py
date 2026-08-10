from __future__ import annotations

from collections import Counter
from typing import Protocol

from cups_eval.schema import ModelCall, Question, StrategyResult, Usage


class LLMCallable(Protocol):
    def call(
        self,
        question: Question,
        role: str,
        context: str | None = None,
        temperature: float | None = None,
    ) -> ModelCall:
        ...


def sum_usage(calls: list[ModelCall]) -> Usage:
    usage = Usage()
    for call in calls:
        usage = usage.add(call.usage)
    return usage


def call_context(calls: list[ModelCall]) -> str:
    return "\n".join(
        f"{call.role}: answer={call.answer}, confidence={call.confidence:.2f}, "
        f"rationale={call.rationale}"
        for call in calls
    )


class DirectStrategy:
    name = "direct"

    def run(self, question: Question, llm: LLMCallable) -> StrategyResult:
        call = llm.call(question, "direct")
        return StrategyResult(
            strategy=self.name,
            answer=call.answer,
            confidence=call.confidence,
            rationale=call.rationale,
            calls=[call],
            usage=call.usage,
            parse_ok=call.parse_ok,
        )


class CotStrategy:
    name = "cot"

    def run(self, question: Question, llm: LLMCallable) -> StrategyResult:
        call = llm.call(question, "cot")
        return StrategyResult(
            strategy=self.name,
            answer=call.answer,
            confidence=call.confidence,
            rationale=call.rationale,
            calls=[call],
            usage=call.usage,
            parse_ok=call.parse_ok,
        )


class SelfConsistencyStrategy:
    def __init__(self, samples: int = 3) -> None:
        self.samples = samples
        self.name = f"self_consistency_{samples}"

    def run(self, question: Question, llm: LLMCallable) -> StrategyResult:
        calls = [
            llm.call(question, "self_consistency", temperature=0.7)
            for _ in range(self.samples)
        ]
        answers = [call.answer for call in calls if call.answer]
        vote_counts = Counter(answers)
        if not answers:
            answer = None
            confidence = 0.0
            stable_majority = False
        else:
            max_votes = max(vote_counts.values())
            tied = {answer for answer, count in vote_counts.items() if count == max_votes}
            answer = next(item for item in answers if item in tied)
            confidence = max_votes / self.samples
            stable_majority = max_votes > self.samples / 2
        return StrategyResult(
            strategy=self.name,
            answer=answer,
            confidence=confidence,
            rationale="majority vote over independent samples",
            calls=calls,
            usage=sum_usage(calls),
            parse_ok=bool(answer),
            metadata={
                "vote_counts": dict(vote_counts),
                "stable_majority": stable_majority,
            },
        )


class Debate2JudgeStrategy:
    name = "debate_2_judge"

    def run(self, question: Question, llm: LLMCallable) -> StrategyResult:
        agent_a = llm.call(question, "debater_a", temperature=0.4)
        agent_b = llm.call(question, "debater_b", temperature=0.4)
        judge = llm.call(
            question,
            "judge",
            context=call_context([agent_a, agent_b]),
            temperature=0.0,
        )
        calls = [agent_a, agent_b, judge]
        fallback = agent_a.answer or agent_b.answer
        answer = judge.answer or fallback
        return StrategyResult(
            strategy=self.name,
            answer=answer,
            confidence=judge.confidence if judge.answer else max(
                agent_a.confidence, agent_b.confidence
            ),
            rationale=judge.rationale or "judge parse failed; used first agent fallback",
            calls=calls,
            usage=sum_usage(calls),
            parse_ok=bool(answer),
            metadata={"agent_answers": [agent_a.answer, agent_b.answer]},
        )


class PlannerVerifierStrategy:
    name = "planner_verifier"

    def run(self, question: Question, llm: LLMCallable) -> StrategyResult:
        planner = llm.call(question, "planner", temperature=0.2)
        verifier = llm.call(
            question,
            "verifier",
            context=call_context([planner]),
            temperature=0.0,
        )
        calls = [planner, verifier]
        answer = verifier.answer or planner.answer
        confidence = verifier.confidence if verifier.answer else planner.confidence
        return StrategyResult(
            strategy=self.name,
            answer=answer,
            confidence=confidence,
            rationale=verifier.rationale or planner.rationale,
            calls=calls,
            usage=sum_usage(calls),
            parse_ok=bool(answer),
            metadata={"planner_answer": planner.answer, "verifier_answer": verifier.answer},
        )
