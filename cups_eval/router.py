from __future__ import annotations

from cups_eval.schema import Question, StrategyResult
from cups_eval.strategies import (
    Debate2JudgeStrategy,
    DirectStrategy,
    LLMCallable,
    PlannerVerifierStrategy,
    SelfConsistencyStrategy,
    call_context,
    sum_usage,
)


class BudgetAwareRouter:
    name = "router"

    def __init__(self, confidence_threshold: float = 0.70) -> None:
        self.confidence_threshold = confidence_threshold

    def run(self, question: Question, llm: LLMCallable) -> StrategyResult:
        route: list[str] = []
        calls = []

        direct = DirectStrategy().run(question, llm)
        route.append(direct.strategy)
        calls.extend(direct.calls)
        final = direct
        if direct.parse_ok and direct.confidence >= self.confidence_threshold:
            return self._wrap(final, route, calls)

        sc = SelfConsistencyStrategy(samples=3).run(question, llm)
        route.append(sc.strategy)
        calls.extend(sc.calls)
        final = sc
        stable = bool(sc.metadata.get("stable_majority"))
        if stable and sc.confidence >= self.confidence_threshold:
            return self._wrap(final, route, calls)

        debate = Debate2JudgeStrategy().run(question, llm)
        route.append(debate.strategy)
        calls.extend(debate.calls)
        final = debate
        return self._wrap(final, route, calls)

    def _wrap(
        self,
        final: StrategyResult,
        route: list[str],
        calls,
    ) -> StrategyResult:
        return StrategyResult(
            strategy=self.name,
            answer=final.answer,
            confidence=final.confidence,
            rationale=final.rationale,
            calls=list(calls),
            usage=sum_usage(list(calls)),
            parse_ok=final.parse_ok,
            metadata={"route": route, "final_strategy": final.strategy},
        )


class ProbeRouter:
    name = "probe_router"

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        verifier_threshold: float = 0.65,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.verifier_threshold = verifier_threshold

    def run(self, question: Question, llm: LLMCallable) -> StrategyResult:
        route: list[str] = []
        calls = []

        direct = DirectStrategy().run(question, llm)
        route.append(direct.strategy)
        calls.extend(direct.calls)

        verifier = llm.call(
            question,
            "answer_verifier",
            context=None,
            temperature=0.0,
        )
        route.append("answer_verifier")
        calls.append(verifier)

        disagree = bool(
            direct.answer
            and verifier.answer
            and direct.answer != verifier.answer
        )
        direct_good_enough = (
            direct.parse_ok
            and verifier.parse_ok
            and not disagree
            and direct.confidence >= self.confidence_threshold
            and verifier.confidence >= self.verifier_threshold
        )
        if direct_good_enough:
            return self._wrap(
                final=direct,
                route=route,
                calls=calls,
                direct=direct,
                verifier=verifier,
                disagree=disagree,
            )

        planner_verifier = PlannerVerifierStrategy().run(question, llm)
        route.append(planner_verifier.strategy)
        calls.extend(planner_verifier.calls)
        final = planner_verifier
        if (
            planner_verifier.parse_ok
            and planner_verifier.confidence >= self.confidence_threshold
        ):
            return self._wrap(final, route, calls, direct, verifier, disagree)

        debate = Debate2JudgeStrategy().run(question, llm)
        route.append(debate.strategy)
        calls.extend(debate.calls)
        return self._wrap(debate, route, calls, direct, verifier, disagree)

    def _wrap(
        self,
        final: StrategyResult,
        route: list[str],
        calls,
        direct: StrategyResult,
        verifier,
        disagree: bool,
    ) -> StrategyResult:
        return StrategyResult(
            strategy=self.name,
            answer=final.answer,
            confidence=final.confidence,
            rationale=final.rationale,
            calls=list(calls),
            usage=sum_usage(list(calls)),
            parse_ok=final.parse_ok,
            metadata={
                "route": route,
                "final_strategy": final.strategy,
                "direct_answer": direct.answer,
                "direct_confidence": direct.confidence,
                "verifier_answer": verifier.answer,
                "verifier_confidence": verifier.confidence,
                "direct_verifier_disagree": disagree,
            },
        )
