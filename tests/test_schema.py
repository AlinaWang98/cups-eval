from cups_eval.schema import Question, StrategyResult, Usage


def test_strategy_result_normalizes_numeric_answer_when_marking_correct():
    question = Question(
        id="gsm8k-1",
        question="What is 40 + 2?",
        options={},
        answer="42",
        metadata={"answer_type": "numeric"},
    )
    result = StrategyResult(
        strategy="direct",
        answer="42.0",
        confidence=0.8,
        rationale="computed",
        calls=[],
        usage=Usage(),
        parse_ok=True,
    )

    record = result.to_record(question)

    assert record["correct"] is True
