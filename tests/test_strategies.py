from cups_eval.llm import FakeLLMClient
from cups_eval.schema import Question
from cups_eval.strategies import (
    CotStrategy,
    Debate2JudgeStrategy,
    DirectStrategy,
    PlannerVerifierStrategy,
    SelfConsistencyStrategy,
)


def make_question(answer="B"):
    return Question(
        id="q1",
        question="Which answer should be selected?",
        options={"A": "alpha", "B": "bravo", "C": "charlie", "D": "delta"},
        answer=answer,
        metadata={},
    )


def test_each_strategy_returns_structured_result_with_usage_and_calls():
    question = make_question()
    strategies = [
        DirectStrategy(),
        CotStrategy(),
        SelfConsistencyStrategy(samples=3),
        Debate2JudgeStrategy(),
        PlannerVerifierStrategy(),
    ]

    for strategy in strategies:
        result = strategy.run(question, FakeLLMClient(mode="always_correct"))

        assert result.strategy == strategy.name
        assert result.answer in {"A", "B", "C", "D"}
        assert 0.0 <= result.confidence <= 1.0
        assert result.calls
        assert result.usage.total_tokens > 0
        assert result.parse_ok is True


def test_self_consistency_tie_uses_deterministic_first_valid_answer():
    question = make_question(answer="D")
    llm = FakeLLMClient(scripted_answers=["C", "D", "B"])

    result = SelfConsistencyStrategy(samples=3).run(question, llm)

    assert result.answer == "C"
    assert result.metadata["vote_counts"] == {"C": 1, "D": 1, "B": 1}
    assert result.metadata["stable_majority"] is False
