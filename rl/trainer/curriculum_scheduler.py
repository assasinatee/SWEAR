"""Curriculum scheduler and deterministic global quota planner.

Legacy Algorithm 1 core logic:
  - D_c starts as B0 (easiest bucket)
  - cs tracks performance on latest bucket
  - When cs >= k/K → merge Bk
  - On merge: new bucket gets full weight, old buckets decay (historical revisiting)
The redesigned B+ path adds two contracts:

* ``bounded_mastery`` progression: a 512-prompt recent window is kept for
  responsiveness, but a frontier must receive a minimum number of prompts and
  both closed reward families must pass for consecutive complete windows.
* a global residual quota planner whose target distribution factorizes into
  curriculum bucket weight x natural data-stratum share.  Bucket size is not
  multiplied into the curriculum weight.
"""

import hashlib
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple
import random


class CellQueueSampler:
    """Deterministic per-rank, per-cell sampling without replacement.

    Rank shards are disjoint, so independent local queues cannot emit the same
    dataset row on two ranks.  A cell is reshuffled only after every local item
    has been consumed once.
    """

    def __init__(
        self,
        cell_indices: Mapping[Tuple[int, str], Sequence[int]],
        seed: int,
        rank: int,
    ):
        self.seed = int(seed)
        self.rank = int(rank)
        self.cell_indices = {
            (int(bucket), str(stratum)): tuple(int(index) for index in indices)
            for (bucket, stratum), indices in cell_indices.items()
            if indices
        }
        self.epochs = {cell: 0 for cell in self.cell_indices}
        self.cursors = {cell: 0 for cell in self.cell_indices}
        self.sampled_counts = {cell: 0 for cell in self.cell_indices}
        self._orders: Dict[Tuple[int, str], List[int]] = {}

    def _order(self, cell: Tuple[int, str]) -> List[int]:
        if cell not in self.cell_indices:
            raise KeyError(f"rank {self.rank} has no samples for cell {cell!r}")
        if cell not in self._orders:
            bucket, stratum = cell
            epoch = self.epochs[cell]
            material = (
                f"bplus-cell:{self.seed}:{self.rank}:{bucket}:{stratum}:{epoch}"
            ).encode("utf-8")
            derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            order = list(self.cell_indices[cell])
            random.Random(derived_seed).shuffle(order)
            self._orders[cell] = order
        return self._orders[cell]

    def sample_cells(self, cells: Sequence[Tuple[int, str]]) -> List[int]:
        indices: List[int] = []
        for bucket_raw, stratum_raw in cells:
            cell = (int(bucket_raw), str(stratum_raw))
            order = self._order(cell)
            cursor = self.cursors[cell]
            if cursor >= len(order):
                self.epochs[cell] += 1
                self.cursors[cell] = 0
                self._orders.pop(cell, None)
                order = self._order(cell)
                cursor = 0
            indices.append(order[cursor])
            self.cursors[cell] = cursor + 1
            self.sampled_counts[cell] += 1
        return indices

    @staticmethod
    def _state_key(cell: Tuple[int, str]) -> str:
        return f"{cell[0]}\x1f{cell[1]}"

    def state_dict(self) -> dict:
        return {
            "seed": self.seed,
            "rank": self.rank,
            "epochs": {
                self._state_key(cell): value for cell, value in self.epochs.items()
            },
            "cursors": {
                self._state_key(cell): value for cell, value in self.cursors.items()
            },
            "sampled_counts": {
                self._state_key(cell): value
                for cell, value in self.sampled_counts.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("cell sampler seed mismatch")
        if int(state.get("rank", self.rank)) != self.rank:
            raise ValueError("cell sampler rank mismatch")

        def decode(mapping_name: str) -> Dict[Tuple[int, str], int]:
            result: Dict[Tuple[int, str], int] = {}
            raw_mapping = state.get(mapping_name, {})
            if not isinstance(raw_mapping, Mapping):
                raise ValueError(f"invalid {mapping_name} in cell sampler state")
            for key, value in raw_mapping.items():
                bucket, stratum = str(key).split("\x1f", 1)
                cell = (int(bucket), stratum)
                if cell not in self.cell_indices:
                    raise ValueError(f"unknown cell in sampler state: {cell!r}")
                result[cell] = int(value)
            return result

        for cell, value in decode("epochs").items():
            self.epochs[cell] = value
        for cell, value in decode("cursors").items():
            if value < 0 or value > len(self.cell_indices[cell]):
                raise ValueError(f"invalid cursor {value} for cell {cell!r}")
            self.cursors[cell] = value
        for cell, value in decode("sampled_counts").items():
            self.sampled_counts[cell] = value
        self._orders = {}


class CurriculumScheduler:
    def __init__(
        self,
        n_buckets: int,
        M: int = 512,
        gamma: float = 0.5,
        revisit_decay: float = 0.5,
        min_weight: float = 0.1,
        progression_mode: str = "legacy",
        min_frontier_prompts: int = 0,
        consecutive_mastery_windows: int = 1,
        mastery_by_family: Optional[Mapping[str, float]] = None,
        quota_seed: int = 42,
    ):
        if n_buckets < 1:
            raise ValueError("n_buckets must be >= 1")
        if M < 1:
            raise ValueError("M must be >= 1")
        if progression_mode not in {"legacy", "bounded_mastery", "fixed"}:
            raise ValueError(
                "progression_mode must be legacy, bounded_mastery, or fixed"
            )
        if min_frontier_prompts < 0:
            raise ValueError("min_frontier_prompts must be >= 0")
        if consecutive_mastery_windows < 1:
            raise ValueError("consecutive_mastery_windows must be >= 1")

        self.K = n_buckets
        self.M = M
        self.gamma = gamma
        self.revisit_decay = revisit_decay
        self.min_weight = min_weight
        self.progression_mode = progression_mode
        self.min_frontier_prompts = int(min_frontier_prompts)
        self.consecutive_mastery_windows = int(consecutive_mastery_windows)
        self.mastery_by_family = {
            str(key).casefold(): float(value)
            for key, value in (
                mastery_by_family or {"s2tt": 0.2, "choice": 0.5}
            ).items()
        }
        if self.progression_mode == "bounded_mastery" and not self.mastery_by_family:
            raise ValueError("bounded_mastery needs mastery_by_family thresholds")

        self.active_bucket_ids: Set[int] = {0}
        # Sampling weight per bucket: new buckets get 1.0, old ones decay
        self.bucket_weights: Dict[int, float] = {0: 1.0}
        self.next_bucket_idx = 1
        self.cs = 0.0
        self.buffer = []
        self.n_merges = 0

        # Exposure-gated progression state.  A reward is one prompt-level mean
        # over G completions, not one completion.
        self.frontier_prompts_seen = 0
        self.stage_windows = 0
        self.total_windows = 0
        self.mastery_streak = 0
        self.last_family_means: Dict[str, float] = {}
        self.last_family_counts: Dict[str, int] = {}
        self.last_window_pass: Optional[bool] = None

        # Global residual-quota state.  The planner is invoked on rank 0 and its
        # 28 global slots are broadcast before each distributed training step.
        self.quota_seed = int(quota_seed)
        self.quota_step = 0
        self.quota_stratum_counts: Dict[str, int] = {}
        self.quota_deficits: Dict[str, float] = {}
        self.quota_planned_counts: Dict[str, int] = {}
        self.quota_expected_counts: Dict[str, float] = {}

        print(
            f"[Curriculum] K={self.K}, M={self.M}, mode={self.progression_mode}, "
            f"min_frontier={self.min_frontier_prompts}, "
            f"consecutive={self.consecutive_mastery_windows}, "
            f"revisit_decay={self.revisit_decay}"
        )

    @property
    def active_buckets(self) -> List[int]:
        return sorted(self.active_bucket_ids)

    @property
    def is_full(self) -> bool:
        return len(self.active_bucket_ids) >= self.K

    def add_rewards(
        self, rewards: List[float], bucket_id: int,
        baselines: Optional[List[float]] = None,
        families: Optional[Sequence[str]] = None,
    ) -> Optional[bool]:
        """Feed rewards from the latest merged bucket.

        baselines: per-reward competence baseline (the "making progress" bar,
            NOT necessarily 0.5). avg_r is measured against the mean baseline:
            update = (avg_r - avg_baseline) * scale. This matters because task
            types have very different reward scales — S2TT BLEU starts near 0,
            choice accuracy starts near random, open scores near 0.5. A fixed
            0.5 baseline would keep the curriculum stuck on low-scale tasks.

        Returns:
            None = not enough samples yet
            True = merge triggered
            False = competence updated, no merge
        """
        if self.progression_mode == "fixed" or self.is_full:
            return None
        if bucket_id != self.next_bucket_idx - 1:
            return None

        if self.progression_mode == "bounded_mastery":
            return self._add_rewards_bounded(rewards, families)

        n = len(rewards)
        if baselines is None:
            baselines = [0.5] * n
        if n == 1:
            self.buffer.append((rewards[0], baselines[0]))
        else:
            self.buffer.extend(zip(rewards, baselines))
        if len(self.buffer) < self.M:
            return None

        avg_r = sum(r for r, _ in self.buffer) / len(self.buffer)
        avg_bl = sum(b for _, b in self.buffer) / len(self.buffer)
        update = (avg_r - avg_bl) * max(1.0 - self.cs, self.gamma)
        self.cs += update
        # Competence is a [0, 1] quantity: without a floor a mediocre model
        # drifts arbitrarily negative (e.g. avg_r=0.35 → cs=-0.6), and then has
        # to burn many buffers climbing back to the merge threshold.
        self.cs = max(0.0, min(1.0, self.cs))
        self.buffer = []

        threshold = self.next_bucket_idx / self.K
        if self.cs >= threshold and self.next_bucket_idx < self.K:
            self._merge()
            return True
        return False

    def _add_rewards_bounded(
        self,
        rewards: Sequence[float],
        families: Optional[Sequence[str]],
    ) -> Optional[bool]:
        """Update the exposure-gated, family-safe B+ mastery state."""

        if families is None or len(families) != len(rewards):
            raise ValueError(
                "bounded_mastery requires one family label per prompt reward"
            )

        completed_window = False
        for reward, family_raw in zip(rewards, families):
            family = str(family_raw).casefold()
            if family not in self.mastery_by_family:
                raise ValueError(
                    f"unknown mastery family {family!r}; expected one of "
                    f"{sorted(self.mastery_by_family)}"
                )
            self.buffer.append((float(reward), family))
            self.frontier_prompts_seen += 1

            if len(self.buffer) < self.M:
                continue

            # Consume exactly M entries.  Keeping any remainder prevents a
            # variable-sized final window if callers add a list at once.
            window = self.buffer[: self.M]
            self.buffer = self.buffer[self.M :]
            by_family: Dict[str, List[float]] = {
                family_name: [] for family_name in self.mastery_by_family
            }
            for value, family_name in window:
                by_family[family_name].append(float(value))

            means = {
                family_name: (
                    sum(values) / len(values) if values else -1.0
                )
                for family_name, values in by_family.items()
            }
            counts = {
                family_name: len(values)
                for family_name, values in by_family.items()
            }
            passed = all(
                counts[family_name] > 0
                and means[family_name] >= threshold
                for family_name, threshold in self.mastery_by_family.items()
            )
            self.mastery_streak = self.mastery_streak + 1 if passed else 0
            self.stage_windows += 1
            self.total_windows += 1
            self.last_family_means = means
            self.last_family_counts = counts
            self.last_window_pass = passed
            completed_window = True

            ratios = []
            for family_name, threshold in self.mastery_by_family.items():
                mean_value = means[family_name]
                if counts[family_name] == 0:
                    ratios.append(0.0)
                elif threshold > 0:
                    ratios.append(mean_value / threshold)
                else:
                    ratios.append(1.0 if mean_value >= threshold else 0.0)
            stage_fraction = max(0.0, min(1.0, min(ratios)))
            stage_floor = (self.next_bucket_idx - 1) / self.K
            self.cs = stage_floor + stage_fraction / self.K

            exposure_ready = (
                self.frontier_prompts_seen >= self.min_frontier_prompts
            )
            streak_ready = (
                self.mastery_streak >= self.consecutive_mastery_windows
            )
            if exposure_ready and streak_ready and self.next_bucket_idx < self.K:
                self._merge()
                return True

        return False if completed_window else None

    def _merge(self):
        k = self.next_bucket_idx
        self.active_bucket_ids.add(k)
        # New bucket gets full weight
        self.bucket_weights[k] = 1.0

        # Historical revisiting: decay weights of all previous buckets
        for bid in list(self.active_bucket_ids):
            if bid != k:
                self.bucket_weights[bid] *= self.revisit_decay
                self.bucket_weights[bid] = max(self.bucket_weights[bid], self.min_weight)

        self.next_bucket_idx += 1
        self.cs = (self.next_bucket_idx - 1) / self.K
        self.buffer = []
        self.n_merges += 1
        self.frontier_prompts_seen = 0
        self.stage_windows = 0
        self.mastery_streak = 0
        self.last_family_means = {}
        self.last_family_counts = {}
        self.last_window_pass = None

        weight_str = ", ".join(
            f"B{b}:{self.bucket_weights.get(b,0):.2f}" for b in self.active_buckets
        )
        print(f"  [Curriculum] *** MERGED B{k} ***")
        print(f"  [Curriculum] Active: {self.active_buckets}, weights: {{{weight_str}}}")

    # ------------------------------------------------------------------
    # Global residual quota
    # ------------------------------------------------------------------

    @staticmethod
    def _cell_key(bucket_id: int, stratum: str) -> str:
        return f"{int(bucket_id)}\x1f{str(stratum)}"

    @staticmethod
    def _decode_cell_key(key: str) -> Tuple[int, str]:
        bucket, stratum = key.split("\x1f", 1)
        return int(bucket), stratum

    def configure_quota(self, stratum_counts: Mapping[str, int]) -> None:
        """Set the immutable natural B+ stratum distribution."""

        clean = {
            str(key): int(value)
            for key, value in stratum_counts.items()
            if int(value) > 0
        }
        if not clean:
            raise ValueError("quota needs at least one positive stratum count")
        if self.quota_stratum_counts and clean != self.quota_stratum_counts:
            raise ValueError("cannot change quota stratum counts after configuration")
        self.quota_stratum_counts = dict(sorted(clean.items()))

    def plan_quota_cells(self, n_slots: int) -> List[Tuple[int, str]]:
        """Plan a deterministic global batch using deficit residual carry.

        For every slot, each active cell accrues its target probability and the
        cell with the largest residual deficit is selected.  The resulting
        integer allocation differs from its cumulative target by less than a
        small rounding residue, without multiplying bucket weights by sizes.
        """

        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if not self.quota_stratum_counts:
            raise RuntimeError("configure_quota must be called before planning")

        active_weights = {
            bid: float(self.bucket_weights.get(bid, 0.0))
            for bid in self.active_buckets
            if float(self.bucket_weights.get(bid, 0.0)) > 0
        }
        weight_total = sum(active_weights.values())
        stratum_total = sum(self.quota_stratum_counts.values())
        if weight_total <= 0 or stratum_total <= 0:
            raise RuntimeError("quota target distribution is empty")

        probabilities: Dict[str, float] = {}
        for bid, weight in active_weights.items():
            bucket_p = weight / weight_total
            for stratum, count in self.quota_stratum_counts.items():
                key = self._cell_key(bid, stratum)
                probabilities[key] = bucket_p * count / stratum_total
                self.quota_deficits.setdefault(key, 0.0)
                self.quota_planned_counts.setdefault(key, 0)
                self.quota_expected_counts.setdefault(key, 0.0)

        planned_keys: List[str] = []
        stable_keys = sorted(probabilities)
        for _ in range(n_slots):
            for key, probability in probabilities.items():
                self.quota_deficits[key] += probability
                self.quota_expected_counts[key] += probability
            chosen = max(
                stable_keys,
                key=lambda key: (self.quota_deficits[key], key),
            )
            self.quota_deficits[chosen] -= 1.0
            self.quota_planned_counts[chosen] += 1
            planned_keys.append(chosen)

        # Randomize placement across ranks without changing the exact global
        # cell counts.  The seed is derived from saved planner state.
        shuffle_rng = random.Random(
            f"bplus-quota:{self.quota_seed}:{self.quota_step}:{self.n_merges}"
        )
        shuffle_rng.shuffle(planned_keys)
        self.quota_step += 1
        return [self._decode_cell_key(key) for key in planned_keys]

    def quota_max_abs_error(self) -> float:
        keys = set(self.quota_planned_counts) | set(self.quota_expected_counts)
        if not keys:
            return 0.0
        return max(
            abs(
                self.quota_planned_counts.get(key, 0)
                - self.quota_expected_counts.get(key, 0.0)
            )
            for key in keys
        )

    def get_effective_sample_sizes(
        self, bucket_sizes: Dict[int, int]
    ) -> Dict[int, float]:
        """Return effective sample size per bucket after applying weights.

        Args:
            bucket_sizes: {bucket_id: total_samples_in_bucket}

        Returns:
            {bucket_id: effective_count}
        """
        return {
            bid: bucket_sizes.get(bid, 0) * self.bucket_weights.get(bid, 0.0)
            for bid in self.active_bucket_ids
        }

    def sample_indices(
        self,
        bucket_indices: Dict[int, List[int]],
        batch_size: int,
        rng: Optional[random.Random] = None,
    ) -> List[int]:
        """Sample indices from active buckets according to their weights.

        Args:
            bucket_indices: {bucket_id: list_of_indices_in_dataset}
            batch_size: Number of indices to sample.

        Returns:
            Sampled indices (may include repeats if not enough samples).
        """
        rng = rng or random.Random()

        # Build weighted pool
        pool = []
        for bid in self.active_bucket_ids:
            indices = bucket_indices.get(bid, [])
            weight = self.bucket_weights.get(bid, 0.0)
            if weight <= 0 or not indices:
                continue
            # Number of samples to take from this bucket
            n_from_bucket = max(1, int(len(indices) * weight))
            n_from_bucket = min(n_from_bucket, len(indices))
            sampled = rng.sample(indices, n_from_bucket)
            pool.extend(sampled)

        if len(pool) < batch_size:
            return rng.choices(pool, k=batch_size)
        return rng.sample(pool, batch_size)

    def state_dict(self) -> dict:
        return {
            "K": self.K, "M": self.M, "gamma": self.gamma,
            "revisit_decay": self.revisit_decay,
            "min_weight": self.min_weight,
            "progression_mode": self.progression_mode,
            "min_frontier_prompts": self.min_frontier_prompts,
            "consecutive_mastery_windows": self.consecutive_mastery_windows,
            "mastery_by_family": dict(self.mastery_by_family),
            "active_bucket_ids": sorted(self.active_bucket_ids),
            "bucket_weights": {k: v for k, v in self.bucket_weights.items()},
            "next_bucket_idx": self.next_bucket_idx,
            "cs": self.cs, "buffer": self.buffer, "n_merges": self.n_merges,
            "frontier_prompts_seen": self.frontier_prompts_seen,
            "stage_windows": self.stage_windows,
            "total_windows": self.total_windows,
            "mastery_streak": self.mastery_streak,
            "last_family_means": dict(self.last_family_means),
            "last_family_counts": dict(self.last_family_counts),
            "last_window_pass": self.last_window_pass,
            "quota_seed": self.quota_seed,
            "quota_step": self.quota_step,
            "quota_stratum_counts": dict(self.quota_stratum_counts),
            "quota_deficits": dict(self.quota_deficits),
            "quota_planned_counts": dict(self.quota_planned_counts),
            "quota_expected_counts": dict(self.quota_expected_counts),
        }

    def load_state_dict(self, state: dict):
        self.K = state["K"]
        self.M = state.get("M", 512)
        self.gamma = state.get("gamma", 0.5)
        self.revisit_decay = state.get("revisit_decay", 0.5)
        self.min_weight = state.get("min_weight", 0.1)
        self.progression_mode = state.get("progression_mode", "legacy")
        self.min_frontier_prompts = int(state.get("min_frontier_prompts", 0))
        self.consecutive_mastery_windows = int(
            state.get("consecutive_mastery_windows", 1)
        )
        self.mastery_by_family = {
            str(key).casefold(): float(value)
            for key, value in state.get(
                "mastery_by_family", {"s2tt": 0.2, "choice": 0.5}
            ).items()
        }
        self.active_bucket_ids = set(state["active_bucket_ids"])
        self.bucket_weights = {
            int(k): v for k, v in state.get("bucket_weights", {}).items()
        }
        self.next_bucket_idx = state["next_bucket_idx"]
        self.cs = state["cs"]
        self.buffer = state.get("buffer", [])
        self.n_merges = state.get("n_merges", 0)
        self.frontier_prompts_seen = int(state.get("frontier_prompts_seen", 0))
        self.stage_windows = int(state.get("stage_windows", 0))
        self.total_windows = int(state.get("total_windows", 0))
        self.mastery_streak = int(state.get("mastery_streak", 0))
        self.last_family_means = {
            str(key): float(value)
            for key, value in state.get("last_family_means", {}).items()
        }
        self.last_family_counts = {
            str(key): int(value)
            for key, value in state.get("last_family_counts", {}).items()
        }
        self.last_window_pass = state.get("last_window_pass")
        self.quota_seed = int(state.get("quota_seed", 42))
        self.quota_step = int(state.get("quota_step", 0))
        self.quota_stratum_counts = {
            str(key): int(value)
            for key, value in state.get("quota_stratum_counts", {}).items()
        }
        self.quota_deficits = {
            str(key): float(value)
            for key, value in state.get("quota_deficits", {}).items()
        }
        self.quota_planned_counts = {
            str(key): int(value)
            for key, value in state.get("quota_planned_counts", {}).items()
        }
        self.quota_expected_counts = {
            str(key): float(value)
            for key, value in state.get("quota_expected_counts", {}).items()
        }

    def get_progress(self) -> dict:
        return {
            "active_buckets": self.active_buckets,
            "bucket_weights": dict(self.bucket_weights),
            "competence": round(self.cs, 4),
            "next_threshold": round(self.next_bucket_idx / self.K, 4),
            "buffer": len(self.buffer),
            "is_full": self.is_full,
            "merges": self.n_merges,
            "progression_mode": self.progression_mode,
            "frontier_prompts": self.frontier_prompts_seen,
            "min_frontier_prompts": self.min_frontier_prompts,
            "stage_windows": self.stage_windows,
            "mastery_streak": self.mastery_streak,
            "mastery_needed": self.consecutive_mastery_windows,
            "last_family_means": dict(self.last_family_means),
            "last_family_counts": dict(self.last_family_counts),
            "last_window_pass": self.last_window_pass,
            "quota_steps": self.quota_step,
            "quota_max_abs_error": round(self.quota_max_abs_error(), 6),
        }
