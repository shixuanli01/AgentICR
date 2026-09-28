"""Frozen prompts, seeds, parsing, and atomic artifact helpers for ICR."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from utils import (
    extract_markdown_python_block,
    run_with_timeout,
    set_seed,
)

from . import AGENTS, CONDITIONS, DIRECTIONS
from .parsing_v3 import (
    CHOICE_LABELS,
    normalize_numeric,
    numeric_equal,
    parse_choice_answer,
    parse_numeric_answer,
)
from .prompts_v3 import PROMPT_VERSION
from .prompts_v3 import revision_prompt as _v3_revision_prompt


SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

INDEPENDENT_SOLVER_PROMPT = """You are an independent problem-solving agent.

Solve the medical multiple-choice question carefully and independently.

Reason from the evidence in the question.
Do not assume another agent will review your answer.

At the end, return exactly one final option in benchmark-compatible form: \\boxed{{A}}, replacing A with one of A, B, C, or D.

Your response should contain:
1. your reasoning
2. your final answer

Medical multiple-choice question:
{question}"""

GENERAL_INDEPENDENT_SOLVER_PROMPT = """You are an independent problem-solving agent.

Solve the multiple-choice question carefully and independently.

Reason from the evidence in the question.
Do not assume another agent will review your answer.

At the end, return exactly one final option in benchmark-compatible form: \\boxed{{A}}, replacing A with one of A, B, C, or D.

Your response should contain:
1. your reasoning
2. your final answer

Multiple-choice question:
{question}"""

NUMERIC_INDEPENDENT_SOLVER_PROMPT = """You are an independent problem-solving agent.

Solve the math word problem carefully and independently.

Show the reasoning needed to verify the calculation.
Do not assume another agent will review your answer.

At the end, return exactly one final numeric answer in the form \\boxed{{NUMBER}}.

Math word problem:
{question}"""

CODE_INDEPENDENT_SOLVER_PROMPT = """You are an independent programming agent.

Solve the programming problem carefully and independently.
Check the function signature, edge cases, and examples in the problem.
Do not assume another agent will review your answer.

Return the complete implementation in exactly one markdown Python code block.

Programming problem:
{question}"""

def stable_seed(global_seed: int, item_id: int, *parts: str) -> int:
    payload = "\0".join((str(global_seed), str(item_id), *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def replicated_stable_seed(
    global_seed: int, replication_id: str, item_id: int, *parts: str
) -> int:
    """Stable seed with replication identity included in a fixed field order."""
    payload = "\0".join((str(global_seed), replication_id, str(item_id), *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def prebelief_seed(
    global_seed: int,
    item_id: int,
    agent_id: str,
    replication_id: Optional[str] = None,
) -> int:
    if agent_id not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_id}")
    if replication_id is None:
        return stable_seed(global_seed, item_id, f"agent_{agent_id}_pre")
    return replicated_stable_seed(
        global_seed, replication_id, item_id, f"agent_{agent_id}_pre"
    )


def revision_seed(
    global_seed: int,
    item_id: int,
    direction: str,
    replication_id: Optional[str] = None,
) -> int:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction: {direction}")
    if replication_id is None:
        return stable_seed(global_seed, item_id, direction, "revision")
    return replicated_stable_seed(
        global_seed, replication_id, item_id, direction, "revision"
    )


NUMERIC_TASKS = frozenset({"gsm8k"})


def reset_rng(seed: int) -> None:
    set_seed(seed)


def parse_task_answer(task: str, text: str) -> Optional[str]:
    """ICR-V3 prediction parsing (see ``icr.parsing_v3`` for the fixed defects)."""
    if task in NUMERIC_TASKS:
        return parse_numeric_answer(text)
    if task in {"mbppplus", "humanevalplus"}:
        return extract_markdown_python_block(text)
    return parse_choice_answer(text, CHOICE_LABELS)


def canonical_gold(task: str, value: Any) -> Optional[str]:
    if task in {"mbppplus", "humanevalplus"}:
        return None if value is None else str(value)
    if task in NUMERIC_TASKS:
        return normalize_numeric(None if value is None else str(value))
    return canonical_answer(value)


def answer_is_correct(task: str, prediction: Any, gold: Any) -> bool:
    if task in {"mbppplus", "humanevalplus"}:
        if prediction is None or gold is None:
            return False
        passed, _ = run_with_timeout(f"{prediction}\n{gold}", timeout=10)
        return passed
    if task in NUMERIC_TASKS:
        if prediction is None or gold is None:
            return False
        return numeric_equal(str(prediction), str(gold))
    normalized_prediction = canonical_gold(task, prediction)
    normalized_gold = canonical_gold(task, gold)
    return (
        normalized_prediction is not None
        and normalized_gold is not None
        and normalized_prediction == normalized_gold
    )


GPQA_LAYER_DISAMBIGUATION = (
    "\n\nAnswer with a label from the final A-D list, not with the lowercase "
    "a)-d) items quoted inside those options."
)


def independent_solver_prompt(task: str, question: str) -> str:
    """Phase-1 prompt, byte-identical to V2 so prebeliefs stay comparable.

    V3 rewrote this template and asked for *concise* reasoning, which V2 never
    did. Independent-solve accuracy on MedQA fell from 72.17% to 67.67% and the
    A/B correctness-disagreement subset -- the only place CR and PR can be
    measured -- shrank from 70 to 56 directional cases. Phase 1 is therefore
    restored verbatim; only phase 2 carries the V3 alignment changes.

    GPQA is the single deviation. Its question text carries two option layers,
    lowercase a)-d) content quoted inside an uppercase A-D submission list, and
    the gold label refers to the outer list. V2 never ran GPQA under ICR, so
    there is no baseline to match, and the ambiguity would silently misgrade.
    """
    if task == "medqa":
        template = INDEPENDENT_SOLVER_PROMPT
    elif task in NUMERIC_TASKS:
        template = NUMERIC_INDEPENDENT_SOLVER_PROMPT
    elif task in {"mbppplus", "humanevalplus"}:
        template = CODE_INDEPENDENT_SOLVER_PROMPT
    else:
        template = GENERAL_INDEPENDENT_SOLVER_PROMPT
    prompt = template.format(question=question)
    if task == "gpqa":
        marker = "replacing A with one of A, B, C, or D."
        prompt = prompt.replace(
            marker, marker + GPQA_LAYER_DISAMBIGUATION, 1
        )
    return prompt


def revision_prompt(
    task: str,
    *,
    question: str,
    receiver_prior_reasoning: str,
    receiver_prior_answer: Optional[str],
    external_block: str,
) -> str:
    """ICR-V3 revision prompt.

    ``external_block`` is the only condition-varying block and sits between the
    receiver's prior and the integration rules.
    """
    return _v3_revision_prompt(
        task,
        question=question,
        receiver_prior_reasoning=receiver_prior_reasoning,
        receiver_prior_answer=receiver_prior_answer,
        external_block_text=external_block,
    )


def canonical_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in set("abcd") else None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256_text(payload)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_conditions(value: str) -> tuple[str, ...]:
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(requested) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    if not requested:
        raise ValueError("At least one condition is required")
    if len(requested) != len(set(requested)):
        raise ValueError("Conditions must not be repeated")
    return requested


def other_item_id(item_id: int, selected_ids: Sequence[int], offset: int = 137) -> int:
    ordered = list(selected_ids)
    if len(ordered) < 2:
        raise ValueError("Other-message controls require at least two selected items")
    try:
        position = ordered.index(item_id)
    except ValueError as error:
        raise ValueError(f"Item {item_id} is not in the selected set") from error
    other = ordered[(position + offset) % len(ordered)]
    if other == item_id:
        other = ordered[(position + 1) % len(ordered)]
    return int(other)


def classify_pair(sender_correct: bool, receiver_correct: bool) -> str:
    if sender_correct and not receiver_correct:
        return "correction_opportunity"
    if not sender_correct and receiver_correct:
        return "destruction_risk"
    if not sender_correct and not receiver_correct:
        return "both_wrong"
    return "both_correct"
