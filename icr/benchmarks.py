"""Benchmark adapters for the ICR evaluation."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from methods.state_bridge import load_dataset_by_name

from .parsing_v3 import extract_numeric_gold


@dataclass(frozen=True)
class BenchmarkSpec:
    task: str
    label: str
    domain: str
    default_max_new_tokens: int
    answer_type: str


# Token budgets. A generation that runs out of budget before emitting its final
# answer parses as None and is then scored wrong, so the correction subset fills
# up with truncations instead of reasoning errors. At the original budgets that
# was 28.0% of GPQA prebeliefs and 20.7% of HumanEval+ ones, against 0.0% for
# MedQA and 0.2% for GSM8K, and on HumanEval+ it left the follow-rate metrics
# with an empty denominator. Both budgets are doubled.
SPECS = {
    "medqa": BenchmarkSpec("medqa", "medqa300", "medical", 8192, "choice"),
    "gpqa": BenchmarkSpec("gpqa", "gpqa_diamond", "general", 16384, "choice"),
    "arc_challenge": BenchmarkSpec(
        "arc_challenge", "arc_challenge", "general", 4096, "choice"
    ),
    "gsm8k": BenchmarkSpec("gsm8k", "gsm8k_test", "math", 4096, "number"),
    "mbppplus": BenchmarkSpec("mbppplus", "mbppplus_test", "code", 8192, "code"),
    "humanevalplus": BenchmarkSpec(
        "humanevalplus", "humanevalplus_test", "code", 8192, "code"
    ),
}


def benchmark_spec(task: str) -> BenchmarkSpec:
    try:
        return SPECS[task]
    except KeyError as error:
        raise ValueError(
            f"Unsupported ICR benchmark {task!r}; choose from {sorted(SPECS)}"
        ) from error


def load_benchmark(task: str) -> list[dict[str, Any]]:
    benchmark_spec(task)
    rows = [dict(row) for row in load_dataset_by_name(task)]
    if task == "gsm8k":
        # Fix S1: the upstream `#### 2,125` gold is truncated to `2` by
        # utils.extract_gold. Re-derive it from the preserved solution text
        # instead of modifying that upstream helper.
        for row in rows:
            gold = extract_numeric_gold(str(row.get("solution") or ""))
            if gold is not None:
                row["gold"] = gold
    return rows


_TRAILING_OPTION_RE = re.compile(r"^\s*([A-Ea-e])\s*[.):]\s*\S")


def trailing_option_labels(question: str) -> list[str]:
    """Labels of the trailing option block, in order, or [] when absent."""
    labels: list[str] = []
    for line in reversed(question.splitlines()):
        if not line.strip():
            continue
        match = _TRAILING_OPTION_RE.match(line)
        if not match:
            break
        labels.append(match.group(1).lower())
    return list(reversed(labels))


def structural_exclusions(task: str, data: Sequence[Mapping[str, Any]]) -> list[int]:
    """Label-free exclusions based only on question structure.

    ARC-Challenge ships a handful of three- and five-option items whose answer
    space cannot be expressed in the frozen A-D output contract. They are
    excluded by option count alone, never by gold, prediction, or correctness.
    Item IDs keep their original dataset indices so exclusions stay traceable.
    """
    if task != "arc_challenge":
        return []
    return [
        index
        for index, row in enumerate(data)
        if len(trailing_option_labels(str(row["question"]))) != 4
    ]


def select_item_ids(
    row_count: int,
    *,
    item_ids: Sequence[int] | None = None,
    limit: int | None = None,
    sample_size: int | None = None,
    selection_seed: int = 42,
) -> list[int]:
    """Choose a recorded, deterministic benchmark subset without changing row IDs."""
    modes = sum(value is not None for value in (item_ids, limit, sample_size))
    if modes > 1:
        raise ValueError("Use only one of item_ids, limit, or sample_size")
    if item_ids is not None:
        selected = [int(value) for value in item_ids]
    elif sample_size is not None:
        if not 0 < sample_size <= row_count:
            raise ValueError("sample_size must be between 1 and the dataset size")
        selected = sorted(random.Random(selection_seed).sample(range(row_count), sample_size))
    else:
        end = row_count if limit is None else limit
        if not 0 < end <= row_count:
            raise ValueError("limit must be between 1 and the dataset size")
        selected = list(range(end))
    if len(selected) != len(set(selected)):
        raise ValueError("Selected item IDs must be unique")
    if not selected or any(not 0 <= value < row_count for value in selected):
        raise ValueError("Selected item IDs are empty or out of range")
    return selected


_OPTION_RE = re.compile(r"^\s*([A-Ea-e])\s*[.):]\s*(.+?)\s*$")


def answer_options(question: str) -> dict[str, str]:
    """Extract the final option layer, including GPQA's nested choices."""
    options: dict[str, str] = {}
    for line in question.splitlines():
        match = _OPTION_RE.match(line)
        if match:
            options[match.group(1).lower()] = match.group(2).strip()
    if len(options) < 2:
        raise ValueError("Question does not contain at least two labeled options")
    return options


def benchmark_metadata(
    task: str, data: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    spec = benchmark_spec(task)
    if spec.answer_type != "choice":
        return [
            {
                "benchmark": spec.label,
                "answer_type": spec.answer_type,
                "answer_options": {},
            }
            for _ in data
        ]
    result = []
    for row in data:
        options = answer_options(str(row["question"]))
        gold = str(row.get("gold", "")).lower()
        if gold and gold not in options:
            raise ValueError(f"Gold option {gold!r} is absent from question options")
        result.append(
            {
                "benchmark": spec.label,
                "answer_type": spec.answer_type,
                "answer_options": options,
            }
        )
    return result
