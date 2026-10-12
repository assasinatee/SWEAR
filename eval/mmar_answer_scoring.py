#!/usr/bin/env python3
"""Reference resolution and conservative option scoring for MMAR.

MMAR has 995 references that exactly equal one answer choice and five malformed
or multi-select references.  Reference resolution depends only on the benchmark
answer and choices; model output is never used to infer the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Sequence


MMAR_SCORER_VERSION = "mmar-reference-option-v1"


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def token_set(value: str) -> set[str]:
    return set(normalize_text(value).split())


@dataclass(frozen=True)
class ReferenceResolution:
    labels: tuple[str, ...]
    mode: str
    multi_select: bool


def _label(index: int) -> str:
    return chr(ord("A") + index)


def resolve_reference(answer: str, choices: Sequence[str]) -> ReferenceResolution:
    """Map an MMAR reference to one option or an explicit set of options."""

    answer_key = normalize_text(answer)
    choice_keys = [normalize_text(choice) for choice in choices]

    exact = tuple(_label(i) for i, key in enumerate(choice_keys) if key == answer_key)
    if exact:
        return ReferenceResolution(exact, "exact_option", len(exact) > 1)

    # Some MMAR references omit framing words from a single choice, e.g.
    # "Verbal slip ..." versus "Because verbal slip ...".
    answer_in_choice = tuple(
        _label(i)
        for i, key in enumerate(choice_keys)
        if answer_key and re.search(rf"(?<!\w){re.escape(answer_key)}(?!\w)", key)
    )
    if len(answer_in_choice) == 1:
        return ReferenceResolution(answer_in_choice, "answer_span_in_option", False)

    # A small number of references concatenate multiple correct choices.
    choices_in_answer = tuple(
        _label(i)
        for i, key in enumerate(choice_keys)
        if key and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", answer_key)
    )
    if choices_in_answer:
        covered = set().union(
            *(token_set(choices[i]) for i in range(len(choices)) if _label(i) in choices_in_answer)
        )
        answer_tokens = token_set(answer)
        coverage = len(covered & answer_tokens) / len(answer_tokens) if answer_tokens else 0.0
        if coverage >= 0.9:
            return ReferenceResolution(choices_in_answer, "concatenated_options", True)

    # Final deterministic repair for a paraphrased single reference.  Require a
    # strong unique token-F1 winner; otherwise leave it unresolved.
    answer_tokens = token_set(answer)
    similarities = []
    for index, choice in enumerate(choices):
        choice_tokens = token_set(choice)
        denom = len(answer_tokens) + len(choice_tokens)
        score = 2 * len(answer_tokens & choice_tokens) / denom if denom else 0.0
        similarities.append((score, index))
    similarities.sort(reverse=True)
    if similarities:
        best_score, best_index = similarities[0]
        runner_up = similarities[1][0] if len(similarities) > 1 else 0.0
        if best_score >= 0.5 and best_score - runner_up >= 0.1:
            return ReferenceResolution((_label(best_index),), "unique_token_f1", False)

    return ReferenceResolution((), "unresolved", False)


def _multi_select_content_match(
    answer: str, response_block: str, choices: Sequence[str], required_labels: tuple[str, ...]
) -> bool:
    response_tokens = token_set(response_block)
    answer_tokens = token_set(answer)
    if not answer_tokens or not answer_tokens.issubset(response_tokens):
        return False
    required = set(required_labels)
    forbidden = set()
    for index, choice in enumerate(choices):
        if _label(index) not in required:
            forbidden.update(token_set(choice) - answer_tokens)
    return response_tokens.isdisjoint(forbidden)


@dataclass(frozen=True)
class MMARScore:
    match: bool
    reference_labels: tuple[str, ...]
    reference_mode: str
    reason: str


def score_mmar_answer(
    pred: str | None,
    answer: str,
    choices: Sequence[str],
    response_block: str = "",
) -> MMARScore:
    reference = resolve_reference(answer, choices)
    normalized_pred = pred.strip().upper() if isinstance(pred, str) else None
    if not reference.labels:
        return MMARScore(False, (), reference.mode, "unresolved_reference")
    if reference.multi_select:
        match = _multi_select_content_match(
            answer, response_block, choices, reference.labels
        )
        return MMARScore(
            match,
            reference.labels,
            reference.mode,
            "multi_select_content" if match else "incomplete_multi_select",
        )
    match = normalized_pred == reference.labels[0]
    return MMARScore(
        match,
        reference.labels,
        reference.mode,
        "selected_reference_option" if match else "different_option",
    )
