#!/usr/bin/env python3
"""
Quality / Accuracy reward with three evaluation paths:

  S2TT subset   ──→ BLEU score (response vs reference transcript)
  Choice type   ──�� Exact-match accuracy (option letter)
  Open type     -> neutral default score

Each path outputs a score in [0, 1] so every question contributes equally
to the curriculum scheduler's competence estimate.
"""

import re
import math
import os
from typing import List, Optional
from collections import Counter


from .base import BaseReward, extract_answer_from_response


# ============================================================================
# Part A — BLEU score (pure Python, no external deps)
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """Split text into tokens useful for BLEU scoring.

    Rules (one token per returned element):
      • CJK character (U+4E00..U+9FFF, U+3000..U+303F) → each char is a token
      • Everything else → split by whitespace (English words, numbers, etc.)
      • Empty / whitespace-only segments are dropped

    Examples:
        "hello world"            → ["hello", "world"]
        "hello world"            -> ["hello", "world"]
        CJK text is split per character.
    """
    CJK_START_1 = 0x4E00   # CJK Unified Ideographs
    CJK_END_1   = 0x9FFF
    CJK_START_2 = 0x3000   # CJK Symbols and Punctuation (includes fullwidth punctuation)
    CJK_END_2   = 0x303F

    tokens = []
    i = 0
    while i < len(text):
        cp = ord(text[i])
        # CJK character → single token
        if (CJK_START_1 <= cp <= CJK_END_1) or (CJK_START_2 <= cp <= CJK_END_2):
            tokens.append(text[i])
            i += 1
        else:
            # Non-CJK run → collect consecutive non-CJK chars, then split by whitespace
            j = i
            while j < len(text):
                cp2 = ord(text[j])
                if (CJK_START_1 <= cp2 <= CJK_END_1) or (CJK_START_2 <= cp2 <= CJK_END_2):
                    break
                j += 1
            segment = text[i:j].strip()
            if segment:
                tokens.extend(segment.split())
            i = j
    return tokens


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-gram counts from token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    candidate: str,
    reference: str,
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Compute BLEU score between candidate and reference.

    Args:
        candidate: Model-generated text.
        reference: Ground-truth reference text.
        max_n: Maximum n-gram order (default 4, standard BLEU).
        smooth: Apply add-1 smoothing to avoid zero precisions.

    Returns:
        BLEU score in [0, 1].
    """
    c_tokens = _tokenize(candidate)
    r_tokens = _tokenize(reference)

    # Brevity penalty
    c_len = len(c_tokens)
    r_len = len(r_tokens)

    if c_len == 0 or r_len == 0:
        return 0.0

    bp = math.exp(1.0 - r_len / c_len) if c_len < r_len else 1.0

    # n-gram precision — only compute over n-grams that exist
    precisions = []
    for n in range(1, max_n + 1):
        c_ngrams = _get_ngrams(c_tokens, n)
        r_ngrams = _get_ngrams(r_tokens, n)

        if len(c_ngrams) == 0 or len(r_ngrams) == 0:
            continue  # skip empty n-gram levels (sentence too short)

        # Count overlapping n-grams
        overlap = sum(min(c_ngrams[ng], r_ngrams.get(ng, 0)) for ng in c_ngrams)
        total = sum(c_ngrams.values())

        if total == 0:
            continue

        if smooth:
            precisions.append((overlap + 1.0) / (total + 1.0))
        else:
            precisions.append(overlap / total)

    # If no valid precisions found, return 0
    if not precisions:
        return 0.0

    # Geometric mean over valid precision levels
    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    return bp * math.exp(log_avg)


def compute_s2tt_reward(response: str, ground_truth: str) -> float:
    """S2TT reward: extract transcript from both and compute BLEU.

    The S2TT ground truth contains the full QA format with <RESPONSE> tags.
    We extract the actual transcript from both response and ground truth.
    """
    # Extract the response content (after final <RESPONSE> tag)
    resp_text = _extract_final_response(response)
    gt_text = _extract_final_response(ground_truth)

    if not resp_text or not gt_text:
        return 0.0

    return compute_bleu(resp_text, gt_text)


def _extract_final_response(text: str) -> str:
    """Extract content after the last <RESPONSE> tag."""
    m = re.search(r'<RESPONSE>(.*?)(?:</RESPONSE>|$)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: take text after last </SUMMARY> or <SUMMARY>
    m = re.search(r'</SUMMARY>\s*(.*?)$', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# ============================================================================
# Part B — Choice accuracy
# ============================================================================

def compute_choice_reward(response: str, ground_truth: str) -> float:
    """Choice question reward: extract option letter, compare with GT.

    Ground truth answer ends with "THE ANSWER IS X" or "The answer is X".
    """
    # Extract predicted answer from response
    pred = extract_answer_from_response(response)

    # Extract expected answer from ground truth
    expected = _extract_choice_from_gt(ground_truth)

    if pred is None or expected is None:
        return 0.0

    return 1.0 if pred.upper() == expected.upper() else 0.0


def _extract_choice_from_gt(gt: str) -> Optional[str]:
    """Extract the correct choice letter from ground truth answer.

    CRITICAL: the final answer lives in the <RESPONSE> section (or the very END
    of the text). The <THINK> reasoning mentions options like "Option A
    describes..." which are NOT the answer — so never take the FIRST "answer is
    X" over the whole text. Search the <RESPONSE> section first; fall back to
    the LAST explicit answer mention anywhere.
    """
    def _last_explicit(text: str) -> Optional[str]:
        matches = list(re.finditer(
            r'(?:the\s+)?(?:correct\s+)?(?:answer|choice|option|letter|selection|cause|reason|result)\s*(?:is|was|be|:|=)\s*\(?([A-Da-d])\)?',
            text, re.IGNORECASE))
        return matches[-1].group(1).upper() if matches else None

    # 1) <RESPONSE> section (the final answer)
    m = re.search(r'<RESPONSE>(.*?)(?:</RESPONSE>|$)', gt, re.DOTALL)
    if m:
        resp_sec = m.group(1).strip()
        let = _last_explicit(resp_sec)
        if let:
            return let
        par = re.findall(r'\(([A-Da-d])\)', resp_sec)
        if par:
            return par[-1].upper()
        c_style = re.findall(r'\b([A-Da-d])\s*\)', resp_sec)
        if c_style:
            return c_style[-1].upper()
        tail_letters = re.findall(r'\b([A-Da-d])\b', resp_sec[-60:])
        if tail_letters:
            return tail_letters[-1].upper()

    # 2) no <RESPONSE> section — LAST explicit answer anywhere (final answer)
    return _last_explicit(gt)


# ============================================================================
# Part D — Unified QualityReward
# ============================================================================

class QualityReward(BaseReward):
    """Unified accuracy/quality reward with three evaluation paths.

    Routing logic (evaluated in order):
      1. subset == "S2TT"       → BLEU score (transcript comparison)
      2. type == "choice"       → Exact-match accuracy
      3. type == "open"         → neutral default score

    Each path outputs [0, 1] so all questions contribute equally.
    """

    name = "quality"

    def __init__(self, default_open_score: float = 0.5):
        """
        Args:
            default_open_score: Neutral score for open-ended questions.
                (This release trains on choice tasks only; open items are not
                expected in the curriculum.)
        """
        self.default_open_score = default_open_score

    def __call__(
        self,
        responses: List[str],
        ground_truths: List[str],
        metadata: Optional[List[dict]] = None,
    ) -> List[float]:
        """Compute quality reward for each response.

        Args:
            responses: Model-generated responses.
            ground_truths: Ground-truth answers from dataset.
            metadata: Per-item metadata dicts with keys:
                - subset: str (e.g. "S2TT", "clotho", "musicbench")
                - type: str ("choice" | "open")
                - question: str (original prompt)
                - (optional) source_file, source_idx

        Returns:
            Rewards in [0, 1], same length as responses.
        """
        n = len(responses)
        if n == 0:
            return []

        rewards = [0.0] * n

        # If no metadata, try heuristic classification
        if metadata is None:
            metadata = [self._classify_by_gt(gt) for gt in ground_truths]

        # Phase 1: S2TT and choice (fast, no model needed)
        s2tt_indices = []
        choice_indices = []
        open_indices = []

        for i in range(n):
            meta = metadata[i] if i < len(metadata) else {}
            subset = str(meta.get("subset", "")).lower()
            qtype = str(meta.get("type", "")).lower()
            # Trust the dataset's subset/type when provided; only fall back to
            # ground-truth heuristics when metadata is empty (e.g. standalone use).
            has_meta = bool(subset or qtype)

            if subset == "s2tt" or (not has_meta and _is_s2tt_question(ground_truths[i])):
                # S2TT: BLEU score
                rewards[i] = compute_s2tt_reward(responses[i], ground_truths[i])
                s2tt_indices.append(i)

            elif qtype == "choice" or (not has_meta and _is_choice_question(ground_truths[i])):
                # Choice: accuracy
                rewards[i] = compute_choice_reward(responses[i], ground_truths[i])
                choice_indices.append(i)

            else:
                # Open: neutral default score
                open_indices.append(i)

        # Phase 2: Open-ended — neutral default (no judge in this release)
        for idx in open_indices:
            rewards[idx] = self.default_open_score

        return rewards

    def _classify_by_gt(self, ground_truth: str) -> dict:
        """Heuristic classification when metadata is not available."""
        gt = ground_truth or ""
        if _is_s2tt_question(gt):
            return {"subset": "S2TT", "type": "s2tt"}
        elif _is_choice_question(gt):
            return {"subset": "unknown", "type": "choice"}
        else:
            return {"subset": "unknown", "type": "open"}


def _is_s2tt_question(ground_truth: str) -> bool:
    """Check if a question is S2TT by examining the ground truth."""
    gt = (ground_truth or "").strip()
    # S2TT ground truth has Chinese transcript in <RESPONSE>
    if not gt:
        return False
    resp = _extract_final_response(gt)
    if not resp:
        return False
    # S2TT outputs Chinese text (contains CJK characters in RESPONSE, not just choice)
    has_chinese = bool(re.search(r'[一-鿿]', resp))
    has_choice = bool(re.search(r'(?:the|The)\s+answer\s+is\s+[A-Da-d]', gt))
    return has_chinese and not has_choice


def _is_choice_question(ground_truth: str) -> bool:
    """Check if a question has a multiple-choice answer."""
    gt = (ground_truth or "").strip()
    return bool(re.search(r'(?:the|The|THE)\s+(?:answer|Answer|ANSWER)\s+(?:is|IS)\s+[A-Da-d]', gt))


# ============================================================================
# Part E — Convenience wrapper for the trainer
# ============================================================================

class QualityRewardPipeline:
    """Composite reward: QualityReward + toggleable FormatReward.

    This is the reward pipeline used by the GRPO trainer.
    Before T_f steps: quality + format
    After T_f steps: quality only
    """

    def __init__(
        self,
        quality_kwargs: Optional[dict] = None,
        format_weight: float = 1.0,
        quality_weight: float = 1.0,
    ):
        from .format_reward import FormatReward

        kwargs = quality_kwargs or {}
        self.quality = QualityReward(**kwargs)
        self.format = FormatReward()
        self.format_weight = format_weight
        self.quality_weight = quality_weight

    def __call__(
        self,
        responses: List[str],
        ground_truths: List[str],
        metadata: Optional[List[dict]] = None,
        use_format: bool = True,
    ) -> List[float]:
        """Compute total reward.

        Args:
            responses: Model responses.
            ground_truths: Reference answers.
            metadata: Per-item metadata (subset, type, question).
            use_format: If True, include format reward.
        """
        quality_scores = self.quality(responses, ground_truths, metadata)

        if use_format:
            format_scores = self.format(responses, ground_truths)
            return [
                self.quality_weight * q + self.format_weight * f
                for q, f in zip(quality_scores, format_scores)
            ]

        return quality_scores
