from cups_eval.datasets import (
    _load_records,
    normalize_gsm8k_answer,
    normalize_gsm8k_record,
    normalize_math500_answer,
    normalize_math500_record,
    normalize_medmcqa_record,
    normalize_medqa_record,
    normalize_medxpertqa_record,
    normalize_mmlu_pro_record,
    normalize_pubmedqa_record,
    sample_records,
)


def test_load_records_falls_back_to_huggingface_when_local_path_is_missing(monkeypatch):
    calls = []

    def fake_load_dataset(name, config_name=None, *, split):
        calls.append((name, config_name, split))
        return [{"id": "remote-1"}]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    records = _load_records(
        {
            "name": "owner/dataset",
            "config_name": "default",
            "local_path": "data/datasets/missing/default",
            "split": "test",
        }
    )

    assert list(records) == [{"id": "remote-1"}]
    assert calls == [("owner/dataset", "default", "test")]


def test_load_records_falls_back_when_local_path_has_only_metadata(monkeypatch, tmp_path):
    metadata_only = tmp_path / "dataset"
    metadata_only.mkdir()
    (metadata_only / "dataset_dict.json").write_text('{"splits": ["test"]}')
    calls = []

    def fake_load_dataset(name, config_name=None, *, split):
        calls.append((name, config_name, split))
        return [{"id": "remote-1"}]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    records = _load_records(
        {
            "name": "owner/dataset",
            "config_name": "default",
            "local_path": str(metadata_only),
            "split": "test",
        }
    )

    assert list(records) == [{"id": "remote-1"}]
    assert calls == [("owner/dataset", "default", "test")]


def test_normalize_medmcqa_record_maps_zero_based_cop_to_letter_by_default():
    record = {
        "id": "q1",
        "question": "Which option is correct?",
        "opa": "alpha",
        "opb": "bravo",
        "opc": "charlie",
        "opd": "delta",
        "cop": 2,
        "choice_type": "single",
        "subject_name": "Medicine",
        "topic_name": "Basics",
    }

    question = normalize_medmcqa_record(record)

    assert question.id == "q1"
    assert question.options == {
        "A": "alpha",
        "B": "bravo",
        "C": "charlie",
        "D": "delta",
    }
    assert question.answer == "C"
    assert question.metadata["subject_name"] == "Medicine"


def test_normalize_medmcqa_record_can_map_one_based_cop_when_requested():
    record = {
        "id": "q1",
        "question": "Which option is correct?",
        "opa": "alpha",
        "opb": "bravo",
        "opc": "charlie",
        "opd": "delta",
        "cop": 2,
        "choice_type": "single",
    }

    question = normalize_medmcqa_record(record, cop_base=1)

    assert question.answer == "B"


def test_sample_records_is_deterministic_and_filters_non_single_choice():
    records = [
        {
            "id": f"q{i}",
            "question": f"Question {i}",
            "opa": "a",
            "opb": "b",
            "opc": "c",
            "opd": "d",
            "cop": 1,
            "choice_type": "single" if i != 3 else "multi",
        }
        for i in range(8)
    ]

    first = sample_records(records, sample_size=4, seed=42)
    second = sample_records(records, sample_size=4, seed=42)

    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) == 4
    assert "q3" not in {item.id for item in first}


def test_sample_records_uses_stable_shuffle_prefix_for_nested_pilots():
    records = [
        {
            "id": f"q{i}",
            "question": f"Question {i}",
            "opa": "a",
            "opb": "b",
            "opc": "c",
            "opd": "d",
            "cop": 1,
            "choice_type": "single",
        }
        for i in range(20)
    ]

    small = sample_records(records, sample_size=5, seed=42)
    large = sample_records(records, sample_size=12, seed=42)

    assert [item.id for item in small] == [item.id for item in large[:5]]


def test_normalize_medqa_record_maps_options_and_answer_idx():
    question = normalize_medqa_record(
        {
            "question": "Best treatment?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer_idx": "D",
            "meta_info": "step2",
        },
        fallback_id="medqa-1",
    )

    assert question.id == "medqa-1"
    assert question.answer == "D"
    assert question.options["D"] == "d"
    assert question.metadata["meta_info"] == "step2"


def test_normalize_pubmedqa_record_maps_yes_no_maybe_to_three_options():
    question = normalize_pubmedqa_record(
        {
            "pubid": 123,
            "question": "Does X help?",
            "context": {"contexts": ["Background text.", "Result text."]},
            "final_decision": "maybe",
        }
    )

    assert question.id == "123"
    assert question.answer == "C"
    assert question.options == {"A": "yes", "B": "no", "C": "maybe"}
    assert "Background text." in question.question


def test_normalize_medxpertqa_record_preserves_a_to_j_options():
    question = normalize_medxpertqa_record(
        {
            "id": "mx-1",
            "question": "Hard medical question?",
            "options": {letter: f"option {letter}" for letter in "ABCDEFGHIJ"},
            "label": "I",
            "medical_task": "Treatment",
            "body_system": "Skeletal",
        }
    )

    assert question.id == "mx-1"
    assert question.answer == "I"
    assert len(question.options) == 10
    assert question.metadata["body_system"] == "Skeletal"


def test_normalize_mmlu_pro_record_maps_options_and_answer_index():
    question = normalize_mmlu_pro_record(
        {
            "question_id": 17,
            "question": "Which statement is true?",
            "options": ["alpha", "bravo", "charlie", "delta"],
            "answer": "C",
            "category": "biology",
        },
        fallback_id="mmlu-pro-17",
    )

    assert question.id == "17"
    assert question.options == {
        "A": "alpha",
        "B": "bravo",
        "C": "charlie",
        "D": "delta",
    }
    assert question.answer == "C"
    assert question.metadata["category"] == "biology"


def test_normalize_gsm8k_record_extracts_numeric_gold_answer():
    question = normalize_gsm8k_record(
        {
            "question": "Jan has 1,200 marbles and gives away 1/4 of them. How many remain?",
            "answer": "Jan gives away 300. #### 900",
        },
        fallback_id="gsm8k-1",
    )

    assert question.id == "gsm8k-1"
    assert question.options == {}
    assert question.answer == "900"
    assert question.metadata["answer_type"] == "numeric"


def test_normalize_gsm8k_answer_handles_commas_and_equivalent_decimals():
    assert normalize_gsm8k_answer("#### 1,200") == "1200"
    assert normalize_gsm8k_answer("Final answer: 42.0") == "42"
    assert normalize_gsm8k_answer("x = -7") == "-7"


def test_normalize_math500_record_preserves_expression_answer():
    question = normalize_math500_record(
        {
            "problem": "Convert $(0,3)$ to polar coordinates.",
            "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
            "subject": "Precalculus",
            "level": 2,
            "unique_id": "test/precalculus/807.json",
        },
        fallback_id="math500-1",
    )

    assert question.id == "test/precalculus/807.json"
    assert question.options == {}
    assert question.answer == "(3,\\frac{\\pi}{2})"
    assert question.metadata["answer_type"] == "math_expression"
    assert question.metadata["subject_name"] == "Precalculus"


def test_normalize_math500_answer_strips_boxed_and_spacing():
    assert normalize_math500_answer("$\\boxed{ 42 }$") == "42"
    assert normalize_math500_answer("\\left\\{ 1, 2 \\right\\}") == "\\{1,2\\}"


def test_normalize_math500_answer_handles_latex_surface_forms():
    assert normalize_math500_answer("\\dfrac{17}{50}") == "\\frac{17}{50}"
    assert normalize_math500_answer("\\text{Evelyn}") == "Evelyn"
    assert normalize_math500_answer("30^\\circ") == "30"
    assert normalize_math500_answer("11,\\!111,\\!111,\\!100") == "11111111100"
    assert normalize_math500_answer("x\\in[-2,7]") == "[-2,7]"


def test_normalize_math500_answer_expands_simple_pm_pair():
    assert normalize_math500_answer("1\\pm\\sqrt{19}") == "1-\\sqrt{19},1+\\sqrt{19}"
    assert (
        normalize_math500_answer("1+\\sqrt{19},1-\\sqrt{19}")
        == "1-\\sqrt{19},1+\\sqrt{19}"
    )


def test_normalize_math500_answer_handles_assignments_and_wrapped_letters():
    assert normalize_math500_answer("a=2") == "2"
    assert (
        normalize_math500_answer("x=1-\\sqrt{19},x=1+\\sqrt{19}")
        == "1-\\sqrt{19},1+\\sqrt{19}"
    )
    assert normalize_math500_answer("\\text{(C)}") == "C"
    assert normalize_math500_answer("144?") == "144"
