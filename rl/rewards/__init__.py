"""Reward functions for RL training."""
from .base import BaseReward, RewardPipeline, extract_answer_from_response
from .format_reward import FormatReward, StrictFormatReward
from .quality_reward import (
    QualityReward,
    QualityRewardPipeline,
    compute_bleu,
    compute_s2tt_reward,
    compute_choice_reward,
)
