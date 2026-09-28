import pytest

from icr.benchmarks import answer_options, benchmark_metadata, benchmark_spec, select_item_ids
from icr.protocol import (
    answer_is_correct,
    independent_solver_prompt,
    parse_task_answer,
    revision_prompt,
)


def test_gpqa_nested_options_use_final_submission_layer():
    question = """Which object is densest?
a) First scientific candidate.
b) Second scientific candidate.
c) Third scientific candidate.
d) Fourth scientific candidate.

A. d
B. a
C. b
D. c"""
    assert answer_options(question) == {
        "a": "d",
        "b": "a",
        "c": "b",
        "d": "c",
    }


def test_arc_colon_options_are_supported():
    question = "Question?\na: Alpha\nb: Beta\nc: Gamma\nd: Delta"
    metadata = benchmark_metadata(
        "arc_challenge", [{"question": question, "gold": "c"}]
    )
    assert metadata[0]["benchmark"] == "arc_challenge"
    assert metadata[0]["answer_options"]["c"] == "Gamma"


def test_arc_variable_choice_counts_are_supported():
    three = answer_options("Question?\na: Alpha\nb: Beta\nc: Gamma")
    five = answer_options("Question?\na: A\nb: B\nc: C\nd: D\ne: E")
    assert set(three) == set("abc")
    assert set(five) == set("abcde")


def test_seeded_subset_is_reproducible_sorted_and_not_prefix():
    first = select_item_ids(1172, sample_size=300, selection_seed=42)
    second = select_item_ids(1172, sample_size=300, selection_seed=42)
    assert first == second
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 300
    assert first != list(range(300))


def test_selection_modes_are_mutually_exclusive():
    with pytest.raises(ValueError):
        select_item_ids(10, limit=2, sample_size=2)


def test_non_choice_metadata_does_not_require_options():
    metadata = benchmark_metadata("gsm8k", [{"question": "2 + 3?", "gold": "5"}])
    assert metadata == [
        {"benchmark": "gsm8k_test", "answer_type": "number", "answer_options": {}}
    ]


def test_code_contract_extracts_and_executes_last_python_block():
    response = """Draft:
```python
def add(a, b):
    return 0
```
Final:
```python
def add(a, b):
    return a + b
```"""
    code = parse_task_answer("mbppplus", response)
    assert code == "def add(a, b):\n    return a + b"
    assert answer_is_correct("mbppplus", code, "assert add(2, 3) == 5")
    assert not answer_is_correct(
        "mbppplus", "def add(a, b):\n    return 0", "assert add(2, 3) == 5"
    )
    assert "markdown Python code block" in independent_solver_prompt(
        "humanevalplus", "Implement add."
    )




def test_gsm8k_uses_numeric_output_contract_and_parser():
    assert benchmark_spec("gsm8k").default_max_new_tokens == 4096
    # Phase 1 uses the numeric solver prompt; phase 2 the per-task output contract.
    assert "\\boxed{NUMBER}" in independent_solver_prompt("gsm8k", "2 + 3?")
    assert parse_task_answer("gsm8k", "Therefore \\boxed{5}.") == "5"
    assert answer_is_correct("gsm8k", "5", "5")
    assert not answer_is_correct("gsm8k", None, "5")
    prompt = revision_prompt(
        "gsm8k",
        question="2 + 3?",
        receiver_prior_reasoning="It is 4.",
        receiver_prior_answer="4",
        external_block="External message: 2 + 3 = 5.",
    )
    assert "no thousands separators" in prompt
    assert "A, B, C, or D" not in prompt
