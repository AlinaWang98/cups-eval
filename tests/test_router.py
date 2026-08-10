from cups_eval.llm import FakeLLMClient
from cups_eval.router import BudgetAwareRouter, ProbeRouter
from cups_eval.schema import Question


def make_question(answer="A"):
    return Question(
        id="router-q",
        question="Pick the answer.",
        options={"A": "alpha", "B": "bravo", "C": "charlie", "D": "delta"},
        answer=answer,
        metadata={},
    )


def test_router_stops_after_high_confidence_direct_answer():
    router = BudgetAwareRouter(confidence_threshold=0.70)
    llm = FakeLLMClient(scripted_answers=["A"], scripted_confidences=[0.95])

    result = router.run(make_question("A"), llm)

    assert result.strategy == "router"
    assert result.answer == "A"
    assert result.metadata["route"] == ["direct"]
    assert len(result.calls) == 1


def test_router_escalates_low_confidence_direct_to_collaboration():
    router = BudgetAwareRouter(confidence_threshold=0.70)
    llm = FakeLLMClient(
        scripted_answers=["B", "C", "D", "B", "A", "A", "A"],
        scripted_confidences=[0.40, 0.45, 0.50, 0.45, 0.82, 0.80, 0.86],
    )

    result = router.run(make_question("A"), llm)

    assert result.strategy == "router"
    assert result.answer == "A"
    assert result.metadata["route"] == ["direct", "self_consistency_3", "debate_2_judge"]
    assert len(result.calls) == 7


def test_probe_router_stops_when_verifier_agrees_with_direct():
    router = ProbeRouter(confidence_threshold=0.70, verifier_threshold=0.65)
    llm = FakeLLMClient(
        scripted_answers=["A", "A"],
        scripted_confidences=[0.95, 0.80],
    )

    result = router.run(make_question("A"), llm)

    assert result.strategy == "probe_router"
    assert result.answer == "A"
    assert result.metadata["route"] == ["direct", "answer_verifier"]
    assert result.metadata["direct_verifier_disagree"] is False
    assert len(result.calls) == 2
    assert "Context from prior calls" not in result.calls[1].prompt


def test_probe_router_escalates_when_verifier_disagrees_with_high_confidence_direct():
    router = ProbeRouter(confidence_threshold=0.70, verifier_threshold=0.65)
    llm = FakeLLMClient(
        scripted_answers=["B", "A", "A", "A"],
        scripted_confidences=[0.95, 0.82, 0.78, 0.84],
    )

    result = router.run(make_question("A"), llm)

    assert result.strategy == "probe_router"
    assert result.answer == "A"
    assert result.metadata["route"] == [
        "direct",
        "answer_verifier",
        "planner_verifier",
    ]
    assert result.metadata["direct_answer"] == "B"
    assert result.metadata["verifier_answer"] == "A"
    assert result.metadata["direct_verifier_disagree"] is True
    assert len(result.calls) == 4
