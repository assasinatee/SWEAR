#!/usr/bin/env python3
"""Conservative multiple-choice scoring with duplicate-option equivalence.

Prediction parsing and correctness are deliberately separate.  A parser returns
the label selected by the model.  The scorer then accepts that label when its
option text is identical to the reference option text after deterministic text
normalisation.  This fixes duplicated answer choices without doing fuzzy or
model-based semantic judging.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Sequence


_CHOICE_PREFIX_RE = re.compile(r"^\s*[\(（\[]\s*[A-H]\s*[\)）\]]\s*", re.IGNORECASE)


def normalize_choice_text(value: str) -> str:
    """Return a deterministic comparison key for an option's answer content.

    Labels, Unicode presentation differences, case, punctuation, and whitespace
    are ignored.  Word/number content is otherwise required to match exactly.
    """

    text = unicodedata.normalize("NFKC", str(value))
    text = _CHOICE_PREFIX_RE.sub("", text).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def label_to_index(label: str | None, choice_count: int) -> int | None:
    if not isinstance(label, str) or len(label.strip()) != 1:
        return None
    index = ord(label.strip().upper()) - ord("A")
    return index if 0 <= index < choice_count else None


def equivalent_reference_labels(gt: str | None, choices: Sequence[str]) -> tuple[str, ...]:
    """Return every label whose option content is equivalent to the GT option."""

    gt_index = label_to_index(gt, len(choices))
    if gt_index is None:
        return ()
    reference_key = normalize_choice_text(choices[gt_index])
    if not reference_key:
        return (chr(ord("A") + gt_index),)
    return tuple(
        chr(ord("A") + index)
        for index, choice in enumerate(choices)
        if normalize_choice_text(choice) == reference_key
    )


@dataclass(frozen=True)
class ChoiceScore:
    match: bool
    strict_match: bool
    duplicate_equivalent_match: bool
    accepted_labels: tuple[str, ...]
    reason: str


def score_choice(
    pred: str | None, gt: str | None, choices: Sequence[str]
) -> ChoiceScore:
    """Score a parsed label, accepting exact duplicate answer choices.

    No fuzzy matching is performed.  In particular, a prose answer that sounds
    similar to the reference cannot override the label explicitly emitted by the
    model.
    """

    normalized_pred = pred.strip().upper() if isinstance(pred, str) else None
    normalized_gt = gt.strip().upper() if isinstance(gt, str) else None
    accepted = equivalent_reference_labels(normalized_gt, choices)
    strict = normalized_pred is not None and normalized_pred == normalized_gt
    duplicate_equivalent = (
        not strict
        and normalized_pred is not None
        and len(accepted) > 1
        and normalized_pred in accepted
    )
    match = strict or duplicate_equivalent
    if strict:
        reason = "strict_label"
    elif duplicate_equivalent:
        reason = "duplicate_option_equivalent"
    elif normalized_pred is None:
        reason = "no_prediction"
    elif not accepted:
        reason = "invalid_reference"
    else:
        reason = "different_option"
    return ChoiceScore(match, strict, duplicate_equivalent, accepted, reason)
