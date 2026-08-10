from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from typing import Any

from cups_eval.answer_normalization import (
    normalize_math_expression_answer,
    normalize_numeric_answer,
)
from cups_eval.prompts import SYSTEM_PROMPT, build_prompt
from cups_eval.schema import OPTION_LETTERS, ModelCall, Question, Usage


def parse_model_response(
    raw_response: str,
    valid_options: Sequence[str] | None = None,
    answer_type: str = "choice",
) -> tuple[str | None, float, str, bool]:
    text = raw_response.strip()
    payload = _extract_json_payload(text)

    if answer_type == "numeric":
        answer = normalize_numeric_answer(str(payload.get("answer", "")))
        if not answer:
            answer = normalize_numeric_answer(text)
    elif answer_type == "math_expression":
        answer = normalize_math_expression_answer(str(payload.get("answer", "")))
        if not answer:
            answer = _extract_marked_math_answer(text)
    else:
        allowed = tuple(
            str(option).upper()
            for option in (valid_options if valid_options is not None else OPTION_LETTERS)
        )
        answer = str(payload.get("answer", "")).strip().upper()
        if answer not in allowed and allowed:
            alternatives = "".join(re.escape(letter) for letter in allowed)
            letter_match = re.search(rf"\b([{alternatives}])\b", text.upper())
            answer = letter_match.group(1) if letter_match else ""

    confidence_value = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    rationale = str(payload.get("rationale", "")).strip()
    if answer_type in {"numeric", "math_expression"}:
        parse_ok = bool(answer)
    else:
        parse_ok = answer in allowed
    return (answer if parse_ok else None, confidence, rationale, parse_ok)


def _extract_json_payload(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _find_embedded_json_payload(text)
    if isinstance(payload, list):
        payload = next((item for item in payload if isinstance(item, dict)), {})
    if not isinstance(payload, dict):
        payload = {}
    return payload


def _find_embedded_json_payload(text: str) -> Any:
    decoder = json.JSONDecoder()
    first_payload: Any = {}
    for match in re.finditer(r"[\[{]", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if "answer" in payload:
                return payload
            if not first_payload:
                first_payload = payload
        elif isinstance(payload, list):
            dict_payload = next((item for item in payload if isinstance(item, dict)), None)
            if dict_payload and "answer" in dict_payload:
                return payload
            if dict_payload and not first_payload:
                first_payload = payload
    return first_payload


def _extract_marked_math_answer(text: str) -> str:
    patterns = (
        r"(?:final\s+answer|the\s+answer|answer)\s*(?:is|:)\s*\$([^$]{1,240})\$",
        r"(?:final\s+answer|the\s+answer|answer)\s*(?:is|:)\s*(\\boxed\{[^{}]{1,240}\})",
        r"(?:final\s+answer|the\s+answer|answer)\s*(?:is|:)\s*([^\n<.]{1,160})",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        for match in reversed(matches):
            candidate = match.group(1).strip()
            candidate = candidate.rstrip(".,;: ")
            normalized = normalize_math_expression_answer(candidate)
            if normalized:
                return normalized
    return ""


def approximate_usage(prompt: str, raw_response: str) -> Usage:
    prompt_tokens = max(1, len(prompt.split()))
    completion_tokens = max(1, len(raw_response.split()))
    return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def build_extra_body(llm_config: dict[str, Any]) -> dict[str, Any] | None:
    thinking = llm_config.get("thinking")
    if thinking is None:
        return None
    thinking_value = str(thinking).strip().lower()
    if thinking_value not in {"enabled", "disabled"}:
        raise ValueError("llm.thinking must be either 'enabled' or 'disabled'")
    return {"thinking": {"type": thinking_value}}


class FakeLLMClient:
    def __init__(
        self,
        mode: str = "always_correct",
        scripted_answers: Sequence[str] | None = None,
        scripted_confidences: Sequence[float] | None = None,
    ) -> None:
        self.mode = mode
        self.scripted_answers = list(scripted_answers or [])
        self.scripted_confidences = list(scripted_confidences or [])
        self._index = 0

    def call(
        self,
        question: Question,
        role: str,
        context: str | None = None,
        temperature: float | None = None,
    ) -> ModelCall:
        prompt = build_prompt(question, role, context)
        if self._index < len(self.scripted_answers):
            answer = self.scripted_answers[self._index]
            if question.answer_type == "choice":
                answer = answer.upper()
        elif self.mode == "cycle":
            answer = OPTION_LETTERS[self._index % len(OPTION_LETTERS)]
        else:
            answer = question.answer

        if self._index < len(self.scripted_confidences):
            confidence = float(self.scripted_confidences[self._index])
        else:
            confidence = 0.86 if answer == question.answer else 0.42

        self._index += 1
        raw = json.dumps(
            {
                "answer": answer,
                "confidence": confidence,
                "rationale": f"fake {role} response",
            }
        )
        parsed_answer, parsed_confidence, rationale, parse_ok = parse_model_response(
            raw,
            valid_options=tuple(question.options),
            answer_type=question.answer_type,
        )
        return ModelCall(
            role=role,
            prompt=prompt,
            raw_response=raw,
            answer=parsed_answer,
            confidence=parsed_confidence,
            rationale=rationale,
            usage=approximate_usage(prompt, raw),
            parse_ok=parse_ok,
        )


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: int = 60,
        extra_body: dict[str, Any] | None = None,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        force_temperature: float | None = None,
    ) -> None:
        import httpx
        from openai import OpenAI

        request_timeout = httpx.Timeout(
            float(timeout_seconds),
            connect=min(10.0, float(timeout_seconds)),
            read=float(timeout_seconds),
            write=min(10.0, float(timeout_seconds)),
            pool=min(10.0, float(timeout_seconds)),
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.force_temperature = force_temperature
        self.request_timeout = request_timeout

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleLLMClient":
        llm_config = config["llm"]
        env_name = str(llm_config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(env_name)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {env_name}")
        model_env = str(llm_config.get("model_env", "OPENAI_MODEL"))
        model = _env_or_default(model_env, str(llm_config["model"]))
        base_url_env = llm_config.get("base_url_env")
        base_url = (
            _env_or_default(str(base_url_env), str(llm_config["base_url"]))
            if base_url_env
            else str(llm_config["base_url"])
        )
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=float(llm_config.get("temperature", 0.2)),
            max_tokens=int(llm_config.get("max_tokens", 512)),
            timeout_seconds=int(llm_config.get("timeout_seconds", 60)),
            extra_body=build_extra_body(llm_config),
            retry_attempts=int(llm_config.get("retry_attempts", 3)),
            retry_backoff_seconds=float(llm_config.get("retry_backoff_seconds", 1.0)),
            force_temperature=_optional_float(llm_config.get("force_temperature")),
        )

    def call(
        self,
        question: Question,
        role: str,
        context: str | None = None,
        temperature: float | None = None,
    ) -> ModelCall:
        prompt = build_prompt(question, role, context)
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": (
                getattr(self, "force_temperature", None)
                if getattr(self, "force_temperature", None) is not None
                else self.temperature if temperature is None else temperature
            ),
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "extra_body": self.extra_body,
            "stream": False,
            "timeout": getattr(self, "request_timeout", None),
        }
        response = self._create_with_retries(request)
        raw = _message_text(response.choices[0].message)
        answer, confidence, rationale, parse_ok = parse_model_response(
            raw,
            valid_options=tuple(question.options),
            answer_type=question.answer_type,
        )
        usage_payload = getattr(response, "usage", None)
        if usage_payload:
            usage = Usage(
                prompt_tokens=int(getattr(usage_payload, "prompt_tokens", 0) or 0),
                completion_tokens=int(
                    getattr(usage_payload, "completion_tokens", 0) or 0
                ),
            )
        else:
            usage = approximate_usage(prompt, raw)
        return ModelCall(
            role=role,
            prompt=prompt,
            raw_response=raw,
            answer=answer,
            confidence=confidence,
            rationale=rationale,
            usage=usage,
            parse_ok=parse_ok,
        )

    def _create_with_retries(self, request: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return self.client.chat.completions.create(**request)
            except Exception as exc:
                if not _is_retryable_provider_error(exc) or attempt >= self.retry_attempts:
                    raise
                last_error = exc
                delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM request failed before any provider attempt")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content:
        return str(content)
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        return str(reasoning_content)
    return ""


def _is_retryable_provider_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = exc.__class__.__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
    }


def _env_or_default(env_name: str, default: str) -> str:
    value = os.environ.get(env_name)
    return value if value else default


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
