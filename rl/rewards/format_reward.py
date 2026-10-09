"""Format reward — checks proper tag closure and <R1>-<R5> rating tokens.

Expected format:

    <THINK>
      <PLANNING><R3>...content...</PLANNING>
      <CAPTION><R4>...content...</CAPTION>
      <REASONING><R2>...content...</REASONING>
      <SUMMARY>...</SUMMARY>
    </THINK>
    <RESPONSE>...</RESPONSE>

Scoring:
  - THINK 缺失或未闭合 → 整题 0 分（一票否决）
  - PLANNING + <R> + 闭合 → +1
  - CAPTION  + <R> + 闭合 → +1
  - REASONING + <R> + 闭合 → +1
  - SUMMARY 闭合          → +1
  - RESPONSE 闭合/boxed   → +2
"""

import re
from typing import List, Tuple
from .base import BaseReward


RATING_TAGS = ["PLANNING", "CAPTION", "REASONING"]
SECTION_TAGS = ["PLANNING", "CAPTION", "REASONING", "SUMMARY"]
BOXED_PATTERN = re.compile(r'\\boxed\{[^}]*\}')


def _tag_is_properly_closed(text: str, open_tag: str, close_tag: str) -> bool:
    open_pos = text.find(open_tag)
    if open_pos == -1:
        return False
    close_pos = text.find(close_tag)
    if close_pos == -1 or close_pos < open_pos:
        return False
    n_open = text.count(open_tag)
    n_close = text.count(close_tag)
    return n_close >= n_open


def _section_has_rating(text: str, tag: str) -> bool:
    """Check if <tag> is immediately followed by <R1>-<R5> (no whitespace)."""
    pattern = re.compile(rf'<{tag}>\s*<R[1-5]>')
    return bool(pattern.search(text))


class FormatReward(BaseReward):
    name = "format"

    def __call__(self, responses: List[str], ground_truths: List[str]) -> List[float]:
        scores = []
        for resp in responses:
            # ── Gate: THINK must be present and properly closed ──
            if not _tag_is_properly_closed(resp, "<THINK>", "</THINK>"):
                scores.append(0.0)
                continue

            raw = 0

            # ── Rated sections: PLANNING, CAPTION, REASONING ──
            for tag in RATING_TAGS:
                if _tag_is_properly_closed(resp, f"<{tag}>", f"</{tag}>") and \
                   _section_has_rating(resp, tag):
                    raw += 1

            # ── SUMMARY (no rating required) ──
            if _tag_is_properly_closed(resp, "<SUMMARY>", "</SUMMARY>"):
                raw += 1

            # ── Final answer ──
            if _tag_is_properly_closed(resp, "<RESPONSE>", "</RESPONSE>"):
                raw += 2
            elif BOXED_PATTERN.search(resp):
                raw += 2

            scores.append(min(raw / 6.0, 1.0))

        return scores


class StrictFormatReward(BaseReward):
    name = "format_strict"

    def __call__(self, responses: List[str], ground_truths: List[str]) -> List[float]:
        scores = []
        for resp in responses:
            if not _tag_is_properly_closed(resp, "<THINK>", "</THINK>"):
                scores.append(0.0)
                continue
            if not _tag_is_properly_closed(resp, "<RESPONSE>", "</RESPONSE>"):
                scores.append(0.0)
                continue

            all_ok = True
            for tag in RATING_TAGS:
                open_tag = f"<{tag}>"
                close_tag = f"</{tag}>"
                if open_tag in resp or close_tag in resp:
                    if not _tag_is_properly_closed(resp, open_tag, close_tag) or \
                       not _section_has_rating(resp, tag):
                        all_ok = False
                        break

            # SUMMARY if present must close
            if "<SUMMARY>" in resp or "</SUMMARY>" in resp:
                if not _tag_is_properly_closed(resp, "<SUMMARY>", "</SUMMARY>"):
                    all_ok = False

            scores.append(1.0 if all_ok else 0.0)

        return scores
