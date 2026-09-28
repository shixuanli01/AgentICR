"""ICR-V3 answer parsing.

Fixes four scoring defects inherited from ``utils.extract_gsm8k_answer`` and
``utils.extract_gold`` without modifying those upstream-derived helpers, so the
frozen StateBridge control and every existing analysis stay reproducible:

S1  ``#### 2,125`` truncated the *gold* itself to ``2``.
S2  ``\\boxed{2,125}`` parsed as ``2``, so a correctly written ``\\boxed{2125}``
    was graded wrong while the comma-formatted answer was graded right.
S3  ``\\boxed{C. Colorectal cancer}`` parsed as unparseable.
S4  ``\\boxed{\\text{C}}`` parsed as unparseable, because the legacy ``[^}]*``
    pattern stops at the first closing brace.
S5  ``\\boxed{18.0}`` compared unequal to gold ``18`` under string equality.

"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional, Sequence


CHOICE_LABELS = ("a", "b", "c", "d")

_BOXED_TOKEN = "\\boxed"
_MATH_WRAPPER_RE = re.compile(
    r"\\(?:text|textbf|textit|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}"
)
_MATH_SPACING_RE = re.compile(r"\\[!,;:> ]")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_GOLD_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_BARE_LABEL_RE = re.compile(r"^([A-Za-z])$")
_LEADING_LABEL_RE = re.compile(r"^\(?\s*([A-Za-z])\s*[)\].:,\-–]")


def iter_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """Every ``\\boxed{...}`` as ``(start, end, payload)`` over ``text``.

    ``start`` indexes the backslash and ``end`` is one past the closing brace,
    so ``text[start:end]`` is the whole macro.
    """
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    while True:
        start = text.find(_BOXED_TOKEN, cursor)
        if start < 0:
            return spans
        index = start + len(_BOXED_TOKEN)
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != "{":
            cursor = start + len(_BOXED_TOKEN)
            continue
        depth = 0
        end = index
        while end < len(text):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        if depth == 0 and end < len(text):
            spans.append((start, end + 1, text[index + 1 : end]))
            cursor = end + 1
        else:
            # Unbalanced: the payload runs to the end of the text.
            spans.append((start, len(text), text[index + 1 :]))
            return spans


def iter_boxed(text: str) -> list[str]:
    """Return every ``\\boxed{...}`` payload, honouring nested braces."""
    return [payload for _, _, payload in iter_boxed_spans(text)]


def clean_boxed(content: str) -> str:
    """Strip LaTeX wrappers and spacing macros from a boxed payload."""
    previous = None
    while previous != content:
        previous = content
        content = _MATH_WRAPPER_RE.sub(r"\1", content)
    content = _MATH_SPACING_RE.sub(" ", content)
    content = content.replace("$", "").replace("{", "").replace("}", "")
    return content.strip()


def _first_number(value: str) -> Optional[str]:
    match = _NUMBER_RE.search(value)
    return match.group(0).replace(",", "") if match else None


def _last_number(value: str) -> Optional[str]:
    matches = _NUMBER_RE.findall(value)
    return matches[-1].replace(",", "") if matches else None


def parse_numeric_answer(text: str) -> Optional[str]:
    """Numeric prediction, comma- and LaTeX-tolerant (fixes S2 and S4)."""
    payloads = iter_boxed(text)
    if payloads:
        content = clean_boxed(payloads[-1])
        number = _first_number(content)
        return number if number is not None else (content or None)
    return _last_number(text)


def extract_numeric_gold(solution: str) -> Optional[str]:
    """Gold from a ``#### value`` line without comma truncation (fixes S1)."""
    match = _GOLD_RE.search(solution)
    return match.group(1).replace(",", "") if match else None


def parse_choice_answer(
    text: str, labels: Sequence[str] = CHOICE_LABELS
) -> Optional[str]:
    """Option label prediction tolerant of ``C. text`` and ``\\text{C}``.

    Fixes S3 and S4. When no ``\\boxed`` payload is present the answer stays
    unparseable, matching the legacy contract.
    """
    payloads = iter_boxed(text)
    if not payloads:
        return None
    content = clean_boxed(payloads[-1])
    match = _BARE_LABEL_RE.match(content) or _LEADING_LABEL_RE.match(content)
    if not match:
        return None
    label = match.group(1).lower()
    return label if label in tuple(labels) else None


def normalize_numeric(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.strip().lower().replace(",", "")


def _decimal(value: str) -> Optional[Decimal]:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def numeric_equal(prediction: Optional[str], gold: Optional[str]) -> bool:
    """Compare numerically when both sides are numbers, else exactly (S5)."""
    left = normalize_numeric(prediction)
    right = normalize_numeric(gold)
    if left is None or right is None:
        return False
    left_decimal, right_decimal = _decimal(left), _decimal(right)
    if left_decimal is not None and right_decimal is not None:
        return left_decimal == right_decimal
    return left == right
