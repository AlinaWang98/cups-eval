import json

from cups_eval.llm import OpenAICompatibleLLMClient, build_extra_body, parse_model_response
from cups_eval.schema import Question


def test_build_extra_body_disables_deepseek_thinking_when_configured():
    assert build_extra_body({"thinking": "disabled"}) == {
        "thinking": {"type": "disabled"}
    }


def test_build_extra_body_omits_thinking_when_not_configured():
    assert build_extra_body({}) is None


class _FakeUsage:
    prompt_tokens = 7
    completion_tokens = 3


class _FakeMessage:
    content = json.dumps({"answer": "A", "confidence": 0.8, "rationale": "ok"})


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = _FakeUsage()


class _FlakyCompletions:
    def __init__(self):
        self.attempts = 0

    def create(self, **kwargs):
        self.attempts += 1
        if self.attempts == 1:
            raise TimeoutError("temporary timeout")
        return _FakeResponse()


class _FakeChat:
    def __init__(self):
        self.completions = _FlakyCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def test_openai_compatible_client_retries_transient_provider_errors(monkeypatch):
    monkeypatch.setattr("cups_eval.llm.time.sleep", lambda delay: None)
    client = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
    client.client = _FakeClient()
    client.model = "fake-model"
    client.temperature = 0.2
    client.max_tokens = 128
    client.extra_body = None
    client.retry_attempts = 2
    client.retry_backoff_seconds = 0.0

    result = client.call(
        Question(
            id="q1",
            question="Question?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            answer="A",
        ),
        role="direct",
    )

    assert result.answer == "A"
    assert client.client.chat.completions.attempts == 2


def test_openai_compatible_client_can_force_provider_temperature(monkeypatch):
    captured = {}

    class CapturingCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class CapturingChat:
        def __init__(self):
            self.completions = CapturingCompletions()

    class CapturingClient:
        def __init__(self):
            self.chat = CapturingChat()

    client = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
    client.client = CapturingClient()
    client.model = "fake-model"
    client.temperature = 0.2
    client.force_temperature = 1.0
    client.max_tokens = 128
    client.extra_body = None
    client.retry_attempts = 1
    client.retry_backoff_seconds = 0.0

    client.call(
        Question(
            id="q1",
            question="Question?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            answer="A",
        ),
        role="answer_verifier",
        temperature=0.0,
    )

    assert captured["temperature"] == 1.0


def test_openai_compatible_client_passes_request_level_timeout():
    captured = {}

    class CapturingCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class CapturingChat:
        def __init__(self):
            self.completions = CapturingCompletions()

    class CapturingClient:
        def __init__(self):
            self.chat = CapturingChat()

    client = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
    client.client = CapturingClient()
    client.model = "fake-model"
    client.temperature = 0.2
    client.force_temperature = None
    client.max_tokens = 128
    client.extra_body = None
    client.retry_attempts = 1
    client.retry_backoff_seconds = 0.0
    client.request_timeout = object()

    client.call(
        Question(
            id="q1",
            question="Question?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            answer="A",
        ),
        role="direct",
    )

    assert captured["timeout"] is client.request_timeout


def test_openai_compatible_client_falls_back_to_reasoning_content_when_content_empty():
    class ReasoningOnlyMessage:
        content = ""
        reasoning_content = json.dumps(
            {"answer": "B", "confidence": 0.7, "rationale": "reasoned json"}
        )

    class ReasoningOnlyChoice:
        message = ReasoningOnlyMessage()

    class ReasoningOnlyResponse:
        choices = [ReasoningOnlyChoice()]
        usage = _FakeUsage()

    class ReasoningOnlyCompletions:
        def create(self, **kwargs):
            return ReasoningOnlyResponse()

    class ReasoningOnlyChat:
        def __init__(self):
            self.completions = ReasoningOnlyCompletions()

    class ReasoningOnlyClient:
        def __init__(self):
            self.chat = ReasoningOnlyChat()

    client = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
    client.client = ReasoningOnlyClient()
    client.model = "fake-model"
    client.temperature = 0.2
    client.force_temperature = None
    client.max_tokens = 128
    client.extra_body = None
    client.retry_attempts = 1
    client.retry_backoff_seconds = 0.0
    client.request_timeout = object()

    result = client.call(
        Question(
            id="q1",
            question="Question?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            answer="B",
        ),
        role="direct",
    )

    assert result.answer == "B"
    assert result.parse_ok is True
    assert result.raw_response == ReasoningOnlyMessage.reasoning_content


def test_parse_model_response_respects_dynamic_option_letters():
    answer, confidence, _, parse_ok = parse_model_response(
        '{"answer": "I", "confidence": 0.7, "rationale": "ok"}',
        valid_options=("A", "B", "I"),
    )

    assert answer == "I"
    assert confidence == 0.7


def test_parse_model_response_supports_math_expression_answers():
    answer, confidence, _, parse_ok = parse_model_response(
        '{"answer": "\\\\boxed{ \\\\frac{1}{2} }", "confidence": 0.9, "rationale": "ok"}',
        valid_options=(),
        answer_type="math_expression",
    )

    assert answer == "\\frac{1}{2}"
    assert confidence == 0.9
    assert parse_ok is True
    assert parse_ok is True


def test_parse_model_response_extracts_json_after_reasoning_with_latex_braces():
    answer, confidence, rationale, parse_ok = parse_model_response(
        '<think>scratch with {braces} and $x^2$</think>{"answer":"-\\\\sqrt{3}","confidence":0.99,"rationale":"verified"}',
        valid_options=(),
        answer_type="math_expression",
    )

    assert answer == "-\\sqrt{3}"
    assert confidence == 0.99
    assert rationale == "verified"
    assert parse_ok is True


def test_parse_model_response_rejects_unstructured_math_reasoning_as_answer():
    answer, confidence, rationale, parse_ok = parse_model_response(
        "<think>I computed many steps and got $-\\sqrt{3}$ but did not return JSON.</think>",
        valid_options=(),
        answer_type="math_expression",
    )

    assert answer is None
    assert confidence == 0.0
    assert rationale == ""
    assert parse_ok is False


def test_parse_model_response_extracts_explicit_math_final_answer_marker():
    answer, confidence, rationale, parse_ok = parse_model_response(
        "<think>long derivation. So yes, the answer is $13535$.</think>",
        valid_options=(),
        answer_type="math_expression",
    )

    assert answer == "13535"
    assert confidence == 0.0
    assert rationale == ""
    assert parse_ok is True


def test_parse_model_response_rejects_letters_outside_question_options():
    answer, _, _, parse_ok = parse_model_response(
        '{"answer": "D", "confidence": 0.7, "rationale": "ok"}',
        valid_options=("A", "B", "C"),
    )

    assert answer is None
    assert parse_ok is False


def test_parse_model_response_accepts_single_object_wrapped_in_json_array():
    answer, confidence, rationale, parse_ok = parse_model_response(
        '[{"answer": "C", "confidence": 0.81, "rationale": "array wrapper"}]',
        valid_options=("A", "B", "C", "D"),
    )

    assert answer == "C"
    assert confidence == 0.81
    assert rationale == "array wrapper"
    assert parse_ok is True


def test_parse_model_response_normalizes_numeric_json_answer():
    answer, confidence, rationale, parse_ok = parse_model_response(
        '{"answer": "1,200.0", "confidence": 0.72, "rationale": "computed total"}',
        answer_type="numeric",
    )

    assert answer == "1200"
    assert confidence == 0.72
    assert rationale == "computed total"
    assert parse_ok is True


def test_parse_model_response_extracts_numeric_answer_from_plain_text():
    answer, _, _, parse_ok = parse_model_response(
        "The final answer is 42.0.",
        answer_type="numeric",
    )

    assert answer == "42"
    assert parse_ok is True
