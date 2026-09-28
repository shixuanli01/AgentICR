"""ICR-V4 receiver policy: verify against the problem, then decide.

Under V3 every channel lands on one correction/preservation trade-off, and the
change rate parameterises it with R^2 = 0.951 -- channels differ in how loudly
they speak, not in how well the receiver uses them. The gate is on the receiver:
its own answer margin predicts whether it will change its answer at AUC 0.26 to
0.43 across seven conditions, all at z < -3 on 718 records, while the same
margin predicts whether it is right at AUC 0.490. The receiver opens up when it
feels unsure, and feeling unsure carries no information about being wrong.

The V3 instruction is a plausible cause:

    Change your answer only if you find a concrete error in YOUR PREVIOUS
    REASONING, or evidence that is better supported than it.

The burden is to find fault in oneself, so the less sure the receiver is of
itself, the more it concedes. V4 changes what is checked and against what:

  * the external message is verified against the ORIGINAL PROBLEM, not against
    the receiver's answer, so openness depends on whether the message survives
    checking rather than on how the receiver feels about its own answer;
  * the receiver's own reasoning is checked the same way, so neither side is
    privileged;
  * the answer follows from the claims that survive, rather than from whether a
    reason to defect was found.

This is a protocol version, not a prompt tweak: V3 results are not comparable
across it, and every channel must be rerun under it for the comparison to mean
anything.
"""

from __future__ import annotations

from typing import Optional

from icr.prompts_v3 import ANSWER_FORMAT, CODE_TASKS, answer_format

PROMPT_VERSION = "icr_v4_verify_then_decide"

VERIFY_TEMPLATE = """You previously solved this problem independently.

Original problem:
{question}

Your previous reasoning:
{receiver_prior_reasoning}

Your previous answer:
{receiver_prior_answer}

{external_block}

Work through two steps, in this order.

Step 1 - Check the external message against the original problem.
Take each claim it makes and test it against the facts stated in the problem.
Do not compare it to your previous answer while doing this. Say which of its
claims hold and which do not. If no external message is available, say so and
go to Step 2.

Step 2 - Check your previous reasoning the same way, against the problem.
Say which of its claims hold and which do not.

Then give the answer supported by the claims that survived both steps. If the
surviving claims point to the answer you already gave, keep it. If they point
elsewhere, change it.

{answer_format}"""

CODE_VERIFY_TEMPLATE = """You previously solved this problem independently.

Original problem:
{question}

Your previous implementation:
{receiver_prior_reasoning}

{external_block}

Work through two steps, in this order.

Step 1 - Check the external message against the original problem.
Test what it does against the stated requirements and against the examples in
the problem. Do not compare it to your previous implementation while doing
this. Say where it is correct and where it fails. If no external message is
available, say so and go to Step 2.

Step 2 - Check your previous implementation the same way, against the problem.
Say where it is correct and where it fails.

Then give the implementation supported by what survived both steps.

{answer_format}"""


def revision_prompt_v4(
    task: str,
    *,
    question: str,
    receiver_prior_reasoning: str,
    receiver_prior_answer: Optional[str],
    external_block: str,
) -> str:
    """Keyword name matches ``icr.protocol.revision_prompt`` so the runtime can
    swap one builder for the other without a shim."""
    template = CODE_VERIFY_TEMPLATE if task in CODE_TASKS else VERIFY_TEMPLATE
    return template.format(
        question=question,
        receiver_prior_reasoning=receiver_prior_reasoning,
        receiver_prior_answer=receiver_prior_answer or "UNPARSEABLE",
        external_block=external_block,
        answer_format=answer_format(task),
    )
