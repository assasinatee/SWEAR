"""Reward pipeline — base class and composite pipeline."""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseReward(ABC):
    name: str = "base"

    @abstractmethod
    def __call__(self, responses: List[str], ground_truths: List[str]) -> List[float]:
        ...


class RewardPipeline:
    """Composable reward pipeline with per-step configuration."""

    def __init__(self, rewards: List[Tuple[BaseReward, float]]):
        self.rewards = rewards

    def __call__(self, responses: List[str], ground_truths: List[str]) -> List[float]:
        n = len(responses)
        if n == 0:
            return []
        total = [0.0] * n
        for fn, w in self.rewards:
            scores = fn(responses, ground_truths)
            for i in range(n):
                total[i] += w * scores[i]
        return total

    @property
    def reward_names(self) -> List[str]:
        return [r.name for r, _ in self.rewards]


# --- Answer extraction helpers ---

def extract_answer_from_response(response: str) -> Optional[str]:
    """Try multiple strategies to extract answer from model output.

    Ordered by reliability: explicit answer-verb patterns first, then
    parenthesized / "c)" style letters, then a last-standalone-letter fallback
    limited to the tail of the <RESPONSE> section (answers usually sit at the
    end), and only as a last resort a bare letter anywhere (avoids matching
    the article "a" mid-sentence).
    """
    # \boxed{...}
    m = re.search(r'\\boxed\{([^}]*)\}', response)
    if m:
        return m.group(1).strip()
    # <ANSWER>...</ANSWER>
    m = re.search(r'<ANSWER>(.*?)</ANSWER>', response, re.DOTALL)
    if m:
        return m.group(1).strip()

    def _last_explicit(text: str) -> Optional[str]:
        matches = list(re.finditer(
            r'(?:the\s+)?(?:answer|choice|option|letter|selection|cause|reason|result|correct\s+(?:answer|choice|option|letter))'
            r'\s*(?:is|was|be|:|=)\s*\(?([A-Da-d])\)?',
            text, re.IGNORECASE))
        return matches[-1].group(1).upper() if matches else None

    # CRITICAL: the <THINK> reasoning mentions options ("Option A describes...")
    # that are NOT the answer. Search the <RESPONSE> section FIRST (final answer),
    # then fall back to the LAST explicit answer anywhere.
    m = re.search(r'<RESPONSE>(.*?)(?:</RESPONSE>|$)', response, re.DOTALL)
    if m:
        resp_sec = m.group(1).strip()
        let = _last_explicit(resp_sec)
        if let:
            return let
        # parenthesized: (a), (b) — take the last
        choice_match = re.findall(r'\(([A-Da-d])\)', resp_sec)
        if choice_match:
            return choice_match[-1].upper()
        # "c)" / "c )" style (e.g. "the answer is c) 3.")
        c_match = re.findall(r'\b([A-Da-d])\s*\)', resp_sec)
        if c_match:
            return c_match[-1].upper()
        # last standalone letter inside the tail — answers sit at the end
        tail = resp_sec[-60:]
        tail_letters = re.findall(r'\b([A-Da-d])\b', tail)
        if tail_letters:
            return tail_letters[-1].upper()
        # weak fallback: first standalone letter anywhere
        letter_match = re.search(r'\b([A-Da-d])\b', resp_sec)
        if letter_match:
            return letter_match.group(1).upper()
    # No <RESPONSE> section — LAST explicit "answer is X" (final answer)
    return _last_explicit(response)

