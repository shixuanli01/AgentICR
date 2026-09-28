"""ICR-V3 revision prompt: one template, one condition-varying slot.

Phase 1 is not here. V3 rewrote it too and asked for concise reasoning, which
cost independent-solve accuracy and shrank the only subset where CR and PR can
be measured, so it was restored to V2's templates in icr.protocol. Only the
revision prompt carries V3's alignment changes.

The revision prompt is fixed to five blocks in this order:

    [Task / Question] -> [Receiver's own prior] -> [External information]
    -> [How to integrate] -> [Output format]

Only the external-information block varies across communication conditions. The
visible text of ``true_statebridge`` and ``true_latentmas`` is byte-identical,
and ``true_text`` differs from them only in the message body itself.
"""

from __future__ import annotations

from typing import Optional

from prompts import EMBEDDING_HINT_MARKER


PROMPT_VERSION = "icr_v3_mid_injection"

CHOICE_TASKS = ("medqa", "gpqa", "arc_challenge")
CODE_TASKS = ("mbppplus", "humanevalplus")

_MCQ_UPPER = (
    "Return your concise reasoning, then exactly one final answer as \\boxed{X}, "
    "where X is one of A, B, C, or D."
)

ANSWER_FORMAT = {
    "medqa": _MCQ_UPPER,
    "arc_challenge": (
        "Return your concise reasoning, then exactly one final answer as \\boxed{X}, "
        "where X is one of the option labels shown above: a, b, c, or d."
    ),
    "gpqa": (
        "Return your concise reasoning, then exactly one final answer as \\boxed{X}, "
        "where X is one of A, B, C, or D from the final labeled list above. "
        "Do not answer with the lowercase a)-d) items quoted inside those options."
    ),
    "gsm8k": (
        "Return your concise reasoning, then exactly one final answer as \\boxed{N}, "
        "where N is a single number written in plain digits, with no thousands "
        "separators, no units, and no other symbols."
    ),
    "mbppplus": (
        "Return the complete final implementation in exactly one markdown Python code "
        "block. Do not put tests or explanatory prose inside that code block."
    ),
}
ANSWER_FORMAT["humanevalplus"] = ANSWER_FORMAT["mbppplus"]


REVISION_TEMPLATE = """You previously solved this problem independently.

Original problem:
{question}

Your previous reasoning:
{receiver_prior_reasoning}

Your previous answer:
{receiver_prior_answer}

{external_block}

Your task is to REVISE your belief, not to restart from scratch.

Evaluate your previous reasoning and any external message critically.

* Change your answer only if you find a concrete error in your previous reasoning, or evidence that is better supported than it.
* Do not change your answer merely because an external message is present.
* Resolve any disagreement using the evidence in the original problem.

{answer_format}"""


CODE_REVISION_TEMPLATE = """You previously solved this problem independently.

Original problem:
{question}

Your previous reasoning and implementation:
{receiver_prior_reasoning}

{external_block}

Your task is to REVISE your implementation, not to restart from scratch and not to copy
an external message.

Evaluate your previous implementation and any external message critically.

* Change your implementation only if you find a concrete defect in it, or an approach that is better supported than it.
* Do not change your implementation merely because an external message is present.
* Check the required function signature, imports, examples, and edge cases against the original problem.

{answer_format}"""


NO_MESSAGE_BLOCK = "No external message is available."
MESSAGE_LEAD_IN = (
    "An external message from another reasoning process is available.\n\n"
    "External message:"
)
LATENT_BLOCK = f"{MESSAGE_LEAD_IN}\n{EMBEDDING_HINT_MARKER}"


def answer_format(task: str) -> str:
    try:
        return ANSWER_FORMAT[task]
    except KeyError as error:
        raise ValueError(f"No V3 answer format registered for task {task!r}") from error


def external_block(condition: str, *, sender_reasoning: Optional[str] = None) -> str:
    """Render the single condition-varying block of the V3 revision prompt."""
    if condition == "none":
        return NO_MESSAGE_BLOCK
    if (
        condition.endswith("_text")
        or condition.endswith("_answer")
    ):
        if sender_reasoning is None:
            raise ValueError("Textual conditions require a message body")
        return f"{MESSAGE_LEAD_IN}\n{sender_reasoning}"
    if (
        condition.endswith("_statebridge")
        or condition.endswith("_latentmas")
    ):
        return LATENT_BLOCK
    raise ValueError(f"Unknown condition: {condition}")


def revision_prompt(
    task: str,
    *,
    question: str,
    receiver_prior_reasoning: str,
    receiver_prior_answer: Optional[str],
    external_block_text: str,
) -> str:
    template = CODE_REVISION_TEMPLATE if task in CODE_TASKS else REVISION_TEMPLATE
    return template.format(
        question=question,
        receiver_prior_reasoning=receiver_prior_reasoning,
        receiver_prior_answer=receiver_prior_answer or "UNPARSEABLE",
        external_block=external_block_text,
        answer_format=answer_format(task),
    )
