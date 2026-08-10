from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def normalize_numeric_answer(text: str) -> str:
    answer_text = str(text).split("####")[-1]
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer_text)
    if not matches:
        return ""
    value = matches[-1].replace(",", "")
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    if number == number.to_integral_value():
        return str(number.to_integral_value())
    return format(number.normalize(), "f")


def normalize_math_expression_answer(text: str) -> str:
    value = str(text).strip()
    value = value.strip("$")
    value = _strip_boxed(value)
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\!", "").replace("\\,", "")
    value = re.sub(r"([A-Za-z])\\in(?=[\\[\\(\\{])", "", value)
    value = value.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    value = re.sub(r"\s+", "", value)
    value = value.rstrip("?")
    value = re.sub(r"(?:(?<=^)|(?<=,))[A-Za-z]=", "", value)
    if re.fullmatch(r"\([A-Za-z]\)", value):
        value = value[1:-1]
    value = _normalize_integer_separators(value)
    value = _expand_simple_pm_pair(value)
    value = _sort_top_level_pair(value)
    return value


def normalize_answer_for_question(answer: str | None, answer_type: str) -> str | None:
    if answer is None:
        return None
    if answer_type == "numeric":
        normalized = normalize_numeric_answer(answer)
        return normalized or None
    if answer_type == "math_expression":
        normalized = normalize_math_expression_answer(answer)
        return normalized or None
    return str(answer).strip().upper()


def _strip_boxed(text: str) -> str:
    text = text.strip()
    for command in ("\\boxed", "\\fbox"):
        prefix = command + "{"
        if text.startswith(prefix) and text.endswith("}"):
            return text[len(prefix):-1].strip()
    return text


def _normalize_integer_separators(text: str) -> str:
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+", text):
        return text.replace(",", "")
    return text


def _expand_simple_pm_pair(text: str) -> str:
    match = re.fullmatch(r"(.+)\\pm(.+)", text)
    if not match:
        return text
    center, offset = match.groups()
    return f"{center}-{offset},{center}+{offset}"


def _sort_top_level_pair(text: str) -> str:
    parts = _split_top_level_commas(text)
    if len(parts) != 2:
        return text
    return ",".join(sorted(parts, key=lambda part: part.replace("+", "~")))


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "({[":
            depth += 1
        elif char in ")}]" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    if parts:
        parts.append(text[start:])
    return parts
