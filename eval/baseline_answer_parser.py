#!/usr/bin/env python3
"""Model-independent multiple-choice answer parsing for public baselines.

The parser is deliberately conservative: it prefers an explicit final-answer
block or answer cue, and never treats an arbitrary standalone A--H in free-form
reasoning as the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


PARSER_VERSION = "baseline-choice-v4"

_RESPONSE_TAG_RE = re.compile(r"<RESPONSE>(.*?)</RESPONSE>", re.IGNORECASE | re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_CHOICE_PREFIX_RE = re.compile(r"^\s*\([A-H]\)\s*", re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z<])")
_COMPACT_LETTER_RE = re.compile(
    r"^\s*[\(\[]?\s*([A-H])\s*[\)\]]?[.。,，;；:：!！]?\s*$", re.IGNORECASE
)
_TRAILING_LETTER_RE = re.compile(
    r"(?:^|\s)[\(\[]?\s*([A-H])\s*[\)\]]?[.。,，;；:：!！]?\s*$", re.IGNORECASE
)
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:[\(（\[]\s*([A-H])\s*[\)）\]]|([A-H])\s*[\)）])(?:\s|$)",
    re.IGNORECASE,
)
_ABSTENTION_RE = re.compile(
    r"(?i)\b(?:"
    r"i(?:'m|\s+am)\s+not\s+sure|"
    r"cannot\s+hear|can't\s+hear|"
    r"without\s+(?:actually\s+)?hearing|"
    r"none\s+of\s+the\s+(?:given\s+)?options|"
    r"not\s+(?:listed|provided|available)(?:\s+(?:among|in)\s+(?:the\s+)?options)?|"
    r"answer\s+(?:is|remains)\s+(?:unknown|none|unavailable)"
    r")\b"
)
_ROLE_LEAK_RE = re.compile(r"\n\s*(?:human|user)\s*:", re.IGNORECASE)

# These cues place the answer text after the captured option letter.
_TAIL_CUE_RES = (
    re.compile(
        r"(?i)\b(?:(?:the\s+)?(?:correct|final)\s+answer|"
        r"(?:the\s+)?correct\s+one|(?:my\s+)?answer|"
        r"(?:correct\s+)?(?:option|choice))\s*"
        r"(?:is|would\s+be|=|:)?\s*[,，:：\-]*\s*"
        r"[\(\[]?\s*([A-H])\s*[\)\]]?"
        r"(?=\s*[,，.;；:：!?]|\s|$)"
    ),
    re.compile(
        r"(?i)\b(?:it'?s|it\s+is)\s*[,，:：\-]+\s*"
        r"[\(\[]?\s*([A-H])\s*[\)\]]?"
        r"(?=\s*[,，.;；:：!?]|\s|$)"
    ),
    re.compile(
        r"(?i)\bthe\s+most\s+likely(?:\s+[a-z]+){0,6}\s+is\s*"
        r"[,，:：\-]*\s*[\(\[]?\s*([A-H])\s*[\)\]]?"
        r"(?=\s*[,，.;；:：!?]|\s|$)"
    ),
    re.compile(
        r"(?i)\b(?:the\s+)?(?:scenario|description)\b[^.!?\n]{0,160}?\b"
        r"(?:is|would\s+be)\s*[,，:：\-]*\s*"
        r"[\(\[]?\s*([A-H])\s*[\)\]]?"
        r"(?=\s*[,，.;；:：!?]|\s|$)"
    ),
    re.compile(
        r"(?i)\b(?:the\s+)?(?:dominant\s+sound|primary\s+event)\b[^.!?\n]{0,80}?\b"
        r"(?:is|would\s+be)\s*[,，:：\-]*\s*"
        r"[\(\[]?\s*([A-H])\s*[\)\]]?"
        r"(?=\s*[,，.;；:：!?]|\s|$)"
    ),
    re.compile(
        r"(?i)\b(?:i|we)(?:'d|\s+would)?\s+(?:go|lean)\s+with\s*"
        r"[,，:：\-]*\s*[\(\[]?\s*([A-H])\s*[\)\]]?"
        r"(?=\s*[,，.;；:：!?]|\s|$)"
    ),
    re.compile(
        r"(?i)\b(?:i|we)(?:'d|\s+would)\s+say\s*[,，:：\-]*\s*"
        r"[\(\[]?\s*([A-H])\s*[\)\]]?"
        r"(?=\s*[,，.;；:：!?]|\s|$)"
    ),
)

# These cues make a selection but do not place useful answer content after it.
_LETTER_ONLY_CUE_RES = (
    re.compile(
        r"(?i)(?:^|\b)([A-H])\s*,?\s+(?:is|would\s+be)\s+"
        r"(?:the\s+)?(?:correct|final|best)\s+(?:answer|option|choice)\b"
    ),
    re.compile(
        r"(?i)(?:^|\b)([A-H])\s*,?\s+seems\s+(?:to\s+be\s+)?"
        r"(?:the\s+)?(?:most\s+likely|best\s+fit|best|correct)\b"
    ),
    re.compile(
        r"(?i)(?<!\w)([A-H])\s*,\s+[^.!?\n]{0,100}?\s+"
        r"(?:is|seems)\s+(?:the\s+)?most\s+likely\b"
    ),
)


@dataclass(frozen=True)
class ParseResult:
    label: str | None
    method: str
    final_block_method: str


@dataclass(frozen=True)
class _Cue:
    start: int
    end: int
    label: str
    has_answer_tail: bool


def extract_final_block(text: str | None) -> tuple[str, str]:
    """Return a model's explicit final-answer span, if one exists."""
    value = text or ""
    match = _RESPONSE_TAG_RE.search(value)
    if match:
        return match.group(1).strip(), "response_tag"
    match = _ANSWER_TAG_RE.search(value)
    if match:
        return match.group(1).strip(), "answer_tag"
    if re.search(r"</think>", value, re.IGNORECASE):
        tail = re.split(r"</think>", value, flags=re.IGNORECASE)[-1].strip()
        if tail:
            return tail, "post_think"
    return value, "full_text"


def _strip_role_leakage(text: str) -> str:
    """Discard a hallucinated follow-up turn after the model's first answer."""
    return _ROLE_LEAK_RE.split(text, maxsplit=1)[0].strip()


def _normalise(value: str) -> str:
    value = _CHOICE_PREFIX_RE.sub("", str(value))
    return " ".join(re.findall(r"\w+", value.lower()))


def _match_choice_content(text: str, choices: Sequence[str]) -> str | None:
    """Map a span to one unique choice, preferring exact normalized phrases."""
    normalized_text = _normalise(text)
    phrase_hits = []
    for index, choice in enumerate(choices):
        normalized_choice = _normalise(choice)
        if normalized_choice and re.search(
            rf"(?<!\w){re.escape(normalized_choice)}(?!\w)", normalized_text
        ):
            phrase_hits.append(index)
    if len(phrase_hits) == 1:
        return chr(ord("A") + phrase_hits[0])

    text_tokens = set(normalized_text.split())
    token_hits = []
    for index, choice in enumerate(choices):
        choice_tokens = set(_normalise(choice).split())
        if choice_tokens and choice_tokens.issubset(text_tokens):
            token_hits.append(index)
    if len(token_hits) == 1:
        return chr(ord("A") + token_hits[0])
    return None


def _find_last_explicit_cue(text: str) -> _Cue | None:
    cues = []
    for pattern in _TAIL_CUE_RES:
        for match in pattern.finditer(text):
            cues.append(_Cue(match.start(), match.end(), match.group(1).upper(), True))
    for pattern in _LETTER_ONLY_CUE_RES:
        for match in pattern.finditer(text):
            cues.append(_Cue(match.start(), match.end(), match.group(1).upper(), False))
    return max(cues, key=lambda cue: cue.start) if cues else None


def _answer_clause_after(text: str, offset: int) -> str:
    tail = text[offset:].lstrip(" ,，:：-")
    return _SENTENCE_BOUNDARY_RE.split(tail, maxsplit=1)[0][:512]


def _resolve_explicit(
    text: str, choices: Sequence[str], prefix: str, block_method: str
) -> ParseResult | None:
    cue = _find_last_explicit_cue(text)
    if cue is None:
        return None
    if cue.has_answer_tail:
        content_label = _match_choice_content(_answer_clause_after(text, cue.end), choices)
        if content_label is not None:
            suffix = "content" if content_label == cue.label else "content_over_letter"
            return ParseResult(content_label, f"{prefix}cue_{suffix}", block_method)
    return ParseResult(cue.label, f"{prefix}cue_letter", block_method)


def resolve_choice(response: str | None, choices: Sequence[str]) -> ParseResult:
    """Resolve a free-form response to an option label A--H.

    Precedence is: explicit final block, explicit answer cue, unique choice
    content, then a strict trailing-letter fallback.  Arbitrary letters in the
    reasoning body are never accepted.
    """
    text = _strip_role_leakage(response or "")
    final_block, block_method = extract_final_block(text)
    final_block = _strip_role_leakage(final_block)

    if block_method != "full_text":
        # A deliberate final-answer cue is stronger evidence than incidental
        # option words elsewhere in the final block.  Content immediately after
        # the cue may still repair a conflicting label (handled by
        # _resolve_explicit), but whole-block bag-of-words matching must not
        # override an explicit selection.
        explicit = _resolve_explicit(final_block, choices, f"{block_method}_", block_method)
        if explicit is not None:
            return ParseResult(explicit.label, explicit.method, block_method)
        leading = _LEADING_LABEL_RE.search(final_block)
        if leading:
            label = leading.group(1) or leading.group(2)
            return ParseResult(
                label.upper(), f"{block_method}_leading_label", block_method
            )
        compact = _COMPACT_LETTER_RE.match(final_block)
        if compact:
            return ParseResult(compact.group(1).upper(), f"{block_method}_letter", block_method)
        if _ABSTENTION_RE.search(final_block):
            return ParseResult(None, f"{block_method}_abstention", block_method)
        content_label = _match_choice_content(final_block, choices)
        if content_label is not None:
            return ParseResult(content_label, f"{block_method}_content", block_method)
        # An explicit final block is authoritative.  Falling back to the hidden
        # reasoning body can turn a distractor mentioned during elimination into
        # a prediction (for example, RESPONSE says "none of the options" while
        # REASONING mentions "Option (b)").
        return ParseResult(None, f"{block_method}_no_match", block_method)

    explicit = _resolve_explicit(text, choices, "", block_method)
    if explicit is not None:
        return ParseResult(explicit.label, explicit.method, block_method)

    # Do not turn an explicit abstention into a correct answer merely because
    # the response later repeats one choice while discussing all possibilities.
    if not _ABSTENTION_RE.search(text):
        content_label = _match_choice_content(text, choices)
        if content_label is not None:
            return ParseResult(content_label, "full_content", block_method)

    leading = _LEADING_LABEL_RE.search(text)
    if leading:
        label = leading.group(1) or leading.group(2)
        return ParseResult(label.upper(), "leading_label", block_method)

    trailing = _TRAILING_LETTER_RE.search(text.strip())
    if trailing:
        return ParseResult(trailing.group(1).upper(), "trailing_letter", block_method)
    return ParseResult(None, "no_match", block_method)
