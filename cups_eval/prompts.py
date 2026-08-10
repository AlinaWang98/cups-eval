from __future__ import annotations

import json

from cups_eval.schema import Question


SYSTEM_PROMPT = (
    "You answer research benchmark questions. "
    "Return only compact JSON with keys answer, confidence, and rationale. "
    "confidence must be a number from 0 to 1. "
    "Do not write chain-of-thought, scratch work, markdown, or text outside JSON. "
    "rationale must be 25 words or fewer."
)


def format_question(question: Question) -> str:
    if not question.options:
        return f"Question:\n{question.question}"
    options = "\n".join(
        f"{letter}. {text}" for letter, text in sorted(question.options.items())
    )
    return f"Question:\n{question.question}\n\nOptions:\n{options}"


def build_prompt(question: Question, role: str, context: str | None = None) -> str:
    task = {
        "direct": "Choose the best answer directly.",
        "cot": "Use brief private reasoning, then choose the best answer.",
        "self_consistency": "Solve independently; do not rely on other attempts.",
        "debater_a": "You are Agent A. Solve independently and note key evidence.",
        "debater_b": "You are Agent B. Solve independently and challenge likely mistakes.",
        "judge": "You are the judge. Use the agent answers to choose the final answer.",
        "planner": "Plan the reasoning path and choose the best answer.",
        "verifier": "Verify the planner answer. Keep it or correct it.",
        "answer_verifier": "Independently verify the proposed answer. Return the answer you believe is correct.",
    }.get(role, "Choose the best answer.")
    parts = [task, format_question(question)]
    if context:
        parts.append("Context from prior calls:\n" + context)
    if question.answer_type == "numeric":
        parts.append("Return the final numeric answer in the answer field.")
        example_answer = "42"
    elif question.answer_type == "math_expression":
        parts.append(
            "Return only the final mathematical expression in the answer field; "
            "do not include derivation text."
        )
        example_answer = "\\frac{1}{2}"
    else:
        parts.append("Valid answer letters: " + ", ".join(sorted(question.options)))
        example_answer = "A"
    parts.append(
        "Return JSON exactly like "
        + json.dumps(
            {
                "answer": example_answer,
                "confidence": 0.75,
                "rationale": "brief reason",
            }
        )
        + ". Return only compact JSON. "
        "Do not write chain-of-thought. "
        "The rationale must be 25 words or fewer."
    )
    return "\n\n".join(parts)
