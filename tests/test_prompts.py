from cups_eval.prompts import SYSTEM_PROMPT, build_prompt
from cups_eval.schema import Question


def test_prompts_forbid_chain_of_thought_and_require_short_json_rationale():
    question = Question(
        id="q1",
        question="A patient has a thyroid nodule. What is the best answer?",
        options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        answer="A",
    )

    prompt = build_prompt(question, "cot")
    combined = f"{SYSTEM_PROMPT}\n{prompt}".lower()

    assert "do not write chain-of-thought" in combined
    assert "rationale must be 25 words or fewer" in combined
    assert "return only compact json" in combined


def test_numeric_prompt_requests_numeric_json_without_option_letters():
    question = Question(
        id="gsm8k-1",
        question="Natalia sold 48 clips in April and half as many in May. Total?",
        options={},
        answer="72",
        metadata={"answer_type": "numeric"},
    )

    prompt = build_prompt(question, "direct")

    assert "Options:" not in prompt
    assert "Valid answer letters" not in prompt
    assert '"answer": "42"' in prompt
    assert "numeric answer" in prompt.lower()
