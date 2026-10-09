"""GRPO trainer with dynamic KL, compression, and rating aux loss.

Three new mechanisms (Tasks A/B/C):
  A. Dynamic KL per sample based on the normalized norm of the three
     section ratings (PLANNING/CAPTION/REASONING)
  B. Self-difficulty-conditioned reasoning-token compression reward
  C. Auxiliary loss aligning all three section-rating logits with teacher labels
"""

import gc
import hashlib
import json
import os
import re
import time
import math
from bisect import bisect_right
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Callable, Any

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import GenerationConfig, PreTrainedModel

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )
except ImportError:  # pragma: no cover - obsolete torch images only
    FSDP = None
    FullStateDictConfig = None
    StateDictType = None


R_DIGIT_IDS = [16, 17, 18, 19, 20]  # 1, 2, 3, 4, 5
RATING_SECTIONS = ("PLANNING", "CAPTION", "REASONING")


class GRPOTrainer:
    """GRPO training engine with curriculum GRPO + auxiliary mechanisms."""

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer,
        processor,
        accelerator: Accelerator,
        curriculum_scheduler,
        reward_pipeline,
        G: int = 6,
        beta: float = 0.04,
        clip_eps: float = 0.2,
        T_f: int = 64,
        max_grad_norm: float = 1.0,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens_by_type: Optional[dict] = None,
        baseline_by_type: Optional[dict] = None,
        # Task A: dynamic KL
        use_dynamic_kl: bool = True,
        reset_reference_on_merge: bool = True,
        # Task B: self-difficulty conditioned compression reward.
        # Output length must fit the cap set by the model's OWN self-rated <R>.
        comp_caps: Optional[List[float]] = None,     # [R1..R5] target length caps (tokens)
        comp_caps_by_family: Optional[Dict[str, List[float]]] = None,
        comp_edges_by_family: Optional[Dict[str, List[float]]] = None,
        comp_mode: str = "global_rms",
        comp_section_caps_by_family: Optional[
            Dict[str, Dict[str, List[float]]]
        ] = None,
        comp_summary_caps_by_family: Optional[Dict[str, float]] = None,
        comp_invalid_reward: float = -0.5,
        comp_rating_levels: int = 5,
        comp_coeff: float = 0.15,                     # over-budget penalty strength
        comp_under_coeff: float = 0.1,                # under-budget penalty (anti "all-R5")
        comp_min_frac: float = 0.3,                   # cap_min = min_frac * cap
        # Task C: auxiliary value loss
        aux_loss_coeff: float = 0.1,
        soft_variance: float = 0.5,
        num_ratings: int = 5,
        streaming_backward: bool = False,
        micro_slots_per_rank: int = 4,
        global_prompts_per_step: Optional[int] = None,
        sampling_state_getter: Optional[Callable[[], dict]] = None,
        sampling_state_loader: Optional[Callable[[dict], None]] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.accelerator = accelerator
        self.scheduler = curriculum_scheduler
        self.reward_pipeline = reward_pipeline
        self.G = G
        self.beta = beta
        self.clip_eps = clip_eps
        self.T_f = T_f
        self.max_grad_norm = max_grad_norm
        self.gen_config = generation_config or GenerationConfig(
            max_new_tokens=64, do_sample=True, temperature=1.0, top_p=1.0,
            top_k=0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # Per-task-type max_new_tokens (used in train_step to pick a length per
        # prompt). Values beyond the global generation_config.max_new_tokens
        # would be pointless; per-type entries just override it downward/upward
        # within reason.
        self.max_new_tokens_by_type = max_new_tokens_by_type or {
            "s2tt": 512, "choice": 512, "open": 1024,
        }

        # Pending per-group curriculum rewards, gathered + applied across ranks
        # in _sync_curriculum after each step. Each entry is
        # (bucket_id, avg_quality, legacy_baseline, reward_family).
        self._pending_curriculum: List[tuple] = []

        # Competence baseline per task type (see CurriculumScheduler.add_rewards):
        # task-specific "making progress" bar, NOT necessarily 0.5.
        self.baseline_by_type = baseline_by_type or {
            "s2tt": 0.2, "choice": 0.5, "open": 0.5,
        }

        # Task A
        self.use_dynamic_kl = bool(use_dynamic_kl)
        self.reset_reference_on_merge = bool(reset_reference_on_merge)
        self.streaming_backward = bool(streaming_backward)
        self.micro_slots_per_rank = int(micro_slots_per_rank)
        self.global_prompts_per_step = int(
            global_prompts_per_step
            if global_prompts_per_step is not None
            else self.accelerator.num_processes * self.micro_slots_per_rank
        )
        self.sampling_state_getter = sampling_state_getter
        self.sampling_state_loader = sampling_state_loader

        # Task B
        # Kept at zero only for backwards compatibility with the removed
        # `_compute_length_penalty` helper. Training uses compression below.
        self.lp_coeff = 0.0
        self.base_lens = [50.0, 120.0, 80.0]
        self.comp_caps = comp_caps or [200.0, 350.0, 500.0, 700.0, 900.0]
        self.comp_caps_by_family = {
            str(k).lower(): [float(x) for x in v]
            for k, v in (comp_caps_by_family or {}).items()
        }
        self.comp_edges_by_family = {
            str(k).lower(): [float(x) for x in v]
            for k, v in (comp_edges_by_family or {}).items()
        }
        self.comp_mode = str(comp_mode).strip().lower()
        if self.comp_mode not in {
            "global_rms", "section_wise", "sft_trace_interval",
        }:
            raise ValueError(
                "compression mode must be 'global_rms', 'section_wise', or "
                "'sft_trace_interval', "
                f"got {comp_mode!r}"
            )
        self.comp_section_caps_by_family = {
            str(family).lower(): {
                str(section).upper(): [float(x) for x in caps]
                for section, caps in section_caps.items()
            }
            for family, section_caps in (
                comp_section_caps_by_family or {}
            ).items()
        }
        self.comp_summary_caps_by_family = {
            str(family).lower(): float(cap)
            for family, cap in (comp_summary_caps_by_family or {}).items()
        }
        self.comp_invalid_reward = float(comp_invalid_reward)
        if not -0.5 <= self.comp_invalid_reward <= 0.0:
            raise ValueError(
                "section-wise invalid_reward must be in [-0.5, 0], got "
                f"{self.comp_invalid_reward}"
            )
        self.comp_rating_levels = int(comp_rating_levels)
        if self.comp_rating_levels < 1:
            raise ValueError(
                "compression rating_levels must be positive, got "
                f"{self.comp_rating_levels}"
            )
        if (
            self.comp_mode == "sft_trace_interval"
            and self.comp_rating_levels != 5
        ):
            raise ValueError(
                "sft_trace_interval currently requires the emitted R1--R5 "
                f"scale (rating_levels=5), got {self.comp_rating_levels}"
            )
        for family, caps in self.comp_caps_by_family.items():
            edges = self.comp_edges_by_family.get(family)
            if edges is None or len(caps) != len(edges) + 1:
                raise ValueError(
                    f"compression family {family!r} needs len(caps)="
                    f"len(edges)+1, got {len(caps)} caps and "
                    f"{0 if edges is None else len(edges)} edges"
                )
            if edges != sorted(edges):
                raise ValueError(
                    f"compression edges for {family!r} must be sorted"
                )
        if self.comp_mode == "section_wise":
            if not self.comp_section_caps_by_family:
                raise ValueError(
                    "section_wise compression requires section_caps_by_family"
                )
            for family, section_caps in self.comp_section_caps_by_family.items():
                missing = set(RATING_SECTIONS) - set(section_caps)
                if missing:
                    raise ValueError(
                        f"section_wise compression family {family!r} is missing "
                        f"caps for {sorted(missing)}"
                    )
                for section in RATING_SECTIONS:
                    caps = section_caps[section]
                    if len(caps) != 5 or any(cap <= 0 for cap in caps):
                        raise ValueError(
                            f"section_wise {family}/{section} needs five "
                            f"positive caps, got {caps}"
                        )
                summary_cap = self.comp_summary_caps_by_family.get(family)
                if summary_cap is None or summary_cap <= 0:
                    raise ValueError(
                        f"section_wise compression family {family!r} needs a "
                        "positive summary cap"
                    )
        self.comp_coeff = comp_coeff
        self.comp_under_coeff = comp_under_coeff
        self.comp_min_frac = comp_min_frac
        if self.accelerator.is_main_process:
            print(
                f"[Compression] mode={self.comp_mode} "
                f"coeff={self.comp_coeff} under_coeff={self.comp_under_coeff}",
                flush=True,
            )

        # Task C
        self.aux_coeff = aux_loss_coeff
        self.soft_var = soft_variance
        self.num_ratings = num_ratings

        self.global_step = 0
        self._metrics_history: List[dict] = []

        self.ref_model = ref_model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    # ======================================================================
    # Generation
    # ======================================================================

    @torch.no_grad()
    def generate_responses(
        self, prompts: List[str], audios: List[Optional[np.ndarray]],
        max_new_tokens_list: Optional[List[int]] = None,
    ) -> dict:
        """Generate G responses per prompt.

        max_new_tokens_list: optional per-prompt max_new_tokens override
            (e.g. shorter for S2TT/choice, longer for open-ended), aligned with
            `prompts`. Uses generation by TYPE so open questions don't get
            truncated while transcription tasks don't waste tokens.
        """
        real_prompt_count = len(prompts)
        if real_prompt_count < 1:
            raise ValueError("generate_responses requires at least one prompt")

        # The strict global batch is 28 prompts over 8 ranks, hence four ranks
        # own 4 real prompts and four own 3.  FSDP.summon_full_params performs
        # collectives when ranks leave the generation context.  If 3-prompt
        # ranks arrive there while 4-prompt ranks are still doing a long,
        # cacheless rollout, the NCCL watchdog can expire.  Give every rank the
        # same four generation calls; dummy outputs are discarded before any
        # reward, curriculum, metric, or backward computation.
        generation_slots = (
            self.micro_slots_per_rank if self.streaming_backward
            else real_prompt_count
        )
        if real_prompt_count > generation_slots:
            raise ValueError(
                f"rank received {real_prompt_count} prompts but only "
                f"{generation_slots} synchronized generation slots"
            )
        padded_prompts = list(prompts)
        padded_audios = list(audios)
        padded_max_tokens = (
            list(max_new_tokens_list)
            if max_new_tokens_list is not None else None
        )
        while len(padded_prompts) < generation_slots:
            padded_prompts.append(prompts[0])
            padded_audios.append(audios[0])
            if padded_max_tokens is not None:
                padded_max_tokens.append(max_new_tokens_list[0])

        all_responses = []
        all_input_ids = []
        all_attn_masks = []
        all_completion_masks = []
        all_seq_starts = []
        prompt_indices = []

        self.model.eval()
        is_fsdp = FSDP is not None and isinstance(self.model, FSDP)
        generation_context = (
            FSDP.summon_full_params(
                self.model, recurse=True, writeback=False, rank0_only=False,
            )
            if is_fsdp else nullcontext()
        )

        # NOTE (2026-08-15): DO NOT toggle config.use_cache=True here for KV-cache
        # generation. Gradient checkpointing is on (memory-safe training forward,
        # batch4×G6 OOMs without it), and transformers FORCES use_cache=False
        # whenever gradient checkpointing is enabled ("use_cache=True is
        # incompatible with gradient checkpointing"). Toggling it back on is
        # silently defeated — generation runs cacheless anyway — and the
        # Qwen2.5-Omni dynamic-cache path is a native-crash/hang risk on
        # multi-node DDP (main-0815065117 crashed in the first step's generation
        # with no Python traceback, NCCL cascade). Leave generation cacheless.
        # Per-prompt audio features, reused by the log-prob forwards (the
        # response tokens must be scored under the SAME audio context that
        # generated them).
        audio_features: List = []
        audio_masks: List = []
        with generation_context:
            # HF generate calls the raw module repeatedly. FSDP parameters are
            # full only inside summon_full_params; outside this context every
            # policy/reference forward must go through its FSDP wrapper.
            gen_model = (
                self.model.module
                if is_fsdp else self.accelerator.unwrap_model(self.model)
            )
            self._gen_loop(gen_model, padded_prompts, padded_audios,
                           padded_max_tokens,
                           all_responses, all_input_ids, all_attn_masks,
                           all_completion_masks,
                           all_seq_starts, prompt_indices,
                           audio_features, audio_masks,
                           real_prompt_count=real_prompt_count)

        # Make the synchronization boundary explicit.  The process-group
        # timeout is raised to two hours by train_grpo.py for long rollouts.
        if is_fsdp and torch.distributed.is_initialized():
            torch.distributed.barrier()

        real_generation_count = real_prompt_count * self.G
        all_responses = all_responses[:real_generation_count]
        all_input_ids = all_input_ids[:real_generation_count]
        all_attn_masks = all_attn_masks[:real_generation_count]
        all_completion_masks = all_completion_masks[:real_generation_count]
        all_seq_starts = all_seq_starts[:real_generation_count]
        prompt_indices = prompt_indices[:real_generation_count]
        audio_features = audio_features[:real_prompt_count]
        audio_masks = audio_masks[:real_prompt_count]
        self.model.train()

        return {
            "responses": all_responses,
            "input_ids": all_input_ids,
            "attention_masks": all_attn_masks,
            "completion_masks": all_completion_masks,
            "seq_starts": all_seq_starts,
            "prompt_indices": prompt_indices,
            "audio_features": audio_features,
            "audio_masks": audio_masks,
        }

    @torch.no_grad()
    def _gen_loop(self, gen_model, prompts, audios, max_new_tokens_list,
                  all_responses, all_input_ids, all_attn_masks,
                  all_completion_masks,
                  all_seq_starts, prompt_indices, audio_features, audio_masks,
                  real_prompt_count=None):
        for i, (prompt, wav) in enumerate(zip(prompts, audios)):
            if getattr(self, "streaming_backward", False):
                rank = getattr(self.accelerator, "process_index", 0)
                kind = "real" if i < (real_prompt_count or len(prompts)) else "dummy"
                print(
                    f"[Rollout] rank={rank} slot={i + 1}/{len(prompts)} "
                    f"kind={kind} begin",
                    flush=True,
                )
            if max_new_tokens_list is not None and i < len(max_new_tokens_list):
                mt = int(max_new_tokens_list[i])
                if mt != self.gen_config.max_new_tokens:
                    self.gen_config.max_new_tokens = mt
            if self.processor is not None and wav is not None:
                # CRITICAL: use the chat template with a structured audio
                # content item so the correct audio token (151646) is inserted
                # into input_ids. Passing the raw "<audio>" text marker does NOT
                # get converted — input_ids ends up without the audio token and
                # the audio features are silently dropped (model hallucinates).
                messages = [{"role": "user", "content": [
                    {"type": "audio"},
                    {"type": "text", "text": prompt.replace("<audio>", "").strip()},
                ]}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                inputs = self.processor(
                    text=text, audio=wav,
                    sampling_rate=16000, return_tensors="pt", padding=True,
                )
            else:
                # Apply chat template for text-only generation
                messages = [{"role": "user", "content": prompt.replace("<audio>", "").strip()}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                inputs = self.tokenizer(
                    [text], return_tensors="pt", padding=True,
                )
            # store this prompt's audio features (used for log-prob scoring)
            af = inputs.get("input_features")
            am = inputs.get("feature_attention_mask")
            audio_features.append(af.cpu() if af is not None else None)
            audio_masks.append(am.cpu() if am is not None else None)
            inputs = {k: v.to(self.accelerator.device) if hasattr(v, 'to') else v
                      for k, v in inputs.items()}

            seq_len = inputs["input_ids"].shape[1]
            # ONE generate call for the whole group of G: repeat the prompt G
            # times along the batch dim so the prefill is shared and the G
            # decode streams run in parallel (do_sample samples independently
            # per row). This replaces G sequential generate calls — the actual
            # bottleneck of the step. (num_return_sequences>1 is unsupported on
            # this model, so we batch by repetition instead.)
            rep = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    rep[k] = v.repeat(self.G, *([1] * (v.dim() - 1)))
                else:
                    rep[k] = v

            gen_out = gen_model.generate(
                **rep,
                generation_config=self.gen_config,
            )
            if getattr(self, "streaming_backward", False):
                print(
                    f"[Rollout] rank={rank} slot={i + 1}/{len(prompts)} "
                    f"kind={kind} done",
                    flush=True,
                )
            if isinstance(gen_out, (tuple, list)):
                out_ids = gen_out[0]
            else:
                out_ids = gen_out

            for g in range(self.G):
                completion_ids = out_ids[g, seq_len:]
                completion_mask = self._build_completion_mask(
                    completion_ids, self.gen_config.eos_token_id,
                )

                # Decode only the real completion. HF generate right-pads rows
                # that hit EOS early so a batched output has a rectangular
                # shape; those post-EOS tokens are not policy actions.
                valid_completion = completion_ids[completion_mask.bool()]
                all_responses.append(self.tokenizer.decode(
                    valid_completion.tolist(), skip_special_tokens=True,
                ))
                prompt_indices.append(i)
                all_seq_starts.append(seq_len)
                all_input_ids.append(out_ids[g].cpu())

                prompt_mask = rep.get("attention_mask")
                if isinstance(prompt_mask, torch.Tensor):
                    prefix_mask = prompt_mask[g, :seq_len].to(dtype=torch.long)
                else:
                    prefix_mask = torch.ones(
                        seq_len, dtype=torch.long, device=out_ids.device,
                    )
                all_attn_masks.append(torch.cat(
                    (prefix_mask, completion_mask), dim=0,
                ).cpu())
                all_completion_masks.append(completion_mask.cpu())

    @staticmethod
    def _build_completion_mask(
        completion_ids: torch.Tensor, eos_token_id,
    ) -> torch.Tensor:
        """Return 1 through the first EOS (inclusive), then 0 for padding."""
        mask = torch.ones_like(completion_ids, dtype=torch.long)
        if eos_token_id is None or completion_ids.numel() == 0:
            return mask

        if isinstance(eos_token_id, torch.Tensor):
            eos_ids = eos_token_id.to(
                device=completion_ids.device, dtype=completion_ids.dtype,
            ).flatten()
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_ids = torch.tensor(
                list(eos_token_id), device=completion_ids.device,
                dtype=completion_ids.dtype,
            )
        else:
            eos_ids = torch.tensor(
                [eos_token_id], device=completion_ids.device,
                dtype=completion_ids.dtype,
            )

        if eos_ids.numel() == 0:
            return mask
        eos_positions = torch.isin(completion_ids, eos_ids).nonzero(
            as_tuple=False,
        )
        if eos_positions.numel() > 0:
            first_eos = int(eos_positions[0].item())
            mask[first_eos + 1:] = 0
        return mask

    # ======================================================================
    # Log-prob & hidden state computation
    # ======================================================================

    @staticmethod
    def _model_device(model, fallback):
        """Return a parameter device without assuming wrappers expose `.device`."""
        first_parameter = next(model.parameters(), None)
        return first_parameter.device if first_parameter is not None else fallback

    def _log_probs_and_hidden(
        self, model, input_ids, attn_mask, start_idx,
        with_grad=False, return_hidden=False
    ):
        """Compute log-probs for response tokens; optionally return hidden states.

        Qwen2.5-Omni's forward is on the thinker submodel, not the main model.
        Returns: (log_probs_tensor, optional_hidden_states)
        """
        # train_grpo always supplies a ThinkerForConditionalGeneration (possibly
        # PEFT/DDP/FSDP wrapped), so call the wrapper itself. Attribute probing
        # such as model.thinker can bypass distributed pre-forward hooks.
        forward_model = model
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            outputs = forward_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=return_hidden,
            )
            logits = outputs.logits  # [1, seq_len, vocab_size]

        log_probs = F.log_softmax(logits, dim=-1)
        shift_lp = log_probs[:, :-1, :]
        shift_ids = input_ids[:, 1:]
        token_lp = shift_lp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
        resp_lp = token_lp[:, start_idx - 1:].squeeze(0)

        if return_hidden:
            hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
            return resp_lp, hidden
        return resp_lp, None

    def compute_log_probs(
        self, input_ids_list, attn_masks_list, seq_starts_list,
        model=None, compute_policy_grad=False,
    ):
        """Compute log-probs under given model.

        Args:
            model: model to use. If None, uses policy model.
            compute_policy_grad: if True, keep gradients.
        """
        model = model or self.model
        target_device = self._model_device(model, self.accelerator.device)
        lps = []
        for i in range(len(input_ids_list)):
            ids = input_ids_list[i].unsqueeze(0).to(target_device)
            mask = attn_masks_list[i].unsqueeze(0).to(target_device)
            lp, _ = self._log_probs_and_hidden(
                model, ids, mask, seq_starts_list[i],
                with_grad=compute_policy_grad, return_hidden=False,
            )
            # Always return on accelerator device for consistent cross-device ops
            lps.append(lp.to(self.accelerator.device))
        return lps

    def compute_log_probs_batched(
        self, input_ids_list, attn_masks_list, seq_starts_list,
        model=None, compute_policy_grad=False,
    ):
        """Batched log-probs: pad all responses to max_len and do ONE forward.

        Results are identical to per-response forwards (padded positions are
        masked out by attention_mask and never extracted), but runs ~Nx fewer
        forwards. Used for old_lps / ref_lps (no_grad). Policy forward with
        hidden states stays per-response.

        Returns: list of per-response log-prob tensors, same as compute_log_probs.
        """
        model = model or self.model
        target_device = self._model_device(model, self.accelerator.device)
        n = len(input_ids_list)
        if n == 0:
            return []

        max_len = max(ids.shape[0] for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        batch_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=target_device)
        batch_mask = torch.zeros((n, max_len), dtype=torch.long, device=target_device)
        real_lens = []
        for i, ids in enumerate(input_ids_list):
            ln = ids.shape[0]
            real_lens.append(ln)
            batch_ids[i, :ln] = ids.to(target_device)
            batch_mask[i, :ln] = attn_masks_list[i].to(target_device)

        if hasattr(model, "thinker"):
            forward_model = model.thinker
        else:
            forward_model = self.accelerator.unwrap_model(model)
        ctx = torch.enable_grad() if compute_policy_grad else torch.no_grad()
        with ctx:
            outputs = forward_model(
                input_ids=batch_ids, attention_mask=batch_mask,
            )
            logits = outputs.logits  # [n, max_len, vocab]

        log_probs = F.log_softmax(logits, dim=-1)
        shift_lp = log_probs[:, :-1, :]
        shift_ids = batch_ids[:, 1:]
        token_lp = shift_lp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)  # [n, max_len-1]

        lps = []
        for i, start in enumerate(seq_starts_list):
            resp_lp = token_lp[i, start - 1: real_lens[i] - 1]
            lps.append(resp_lp.to(self.accelerator.device))
        return lps

    def compute_policy_lps_and_hidden(
        self, input_ids_list, attn_masks_list, seq_starts_list,
    ):
        """Compute policy log-probs AND return hidden states for aux loss.

        Returns:
            lps: list of log-prob tensors (with grad flowing)
            hidden_list: list of all hidden states from policy forward
        """
        lps = []
        hidden_list = []
        for i in range(len(input_ids_list)):
            ids = input_ids_list[i].unsqueeze(0).to(self.accelerator.device)
            mask = attn_masks_list[i].unsqueeze(0).to(self.accelerator.device)
            lp, hidden = self._log_probs_and_hidden(
                self.model, ids, mask, seq_starts_list[i],
                with_grad=True, return_hidden=True,
            )
            lps.append(lp)
            hidden_list.append(hidden)  # [1, seq_len, hidden_dim]
        return lps, hidden_list

    def _compute_log_probs_grouped(
        self, input_ids_list, attn_masks_list, seq_starts_list, prompt_indices,
        audio_features, audio_masks, model=None, compute_policy_grad=False,
        return_hidden=False, return_rating_logits=False,
    ):
        """Per-prompt grouped log-probs, WITH audio context.

        Responses of the same prompt share one audio clip, so they are batched
        together (G rows) with the audio features repeated — this keeps the
        model's masked_scatter alignment (each row's audio tokens match its
        features). Different prompts have different audio durations, so they
        run as separate forwards. Without audio (audio_features None), this
        reduces to the old padded-batch behaviour.

        Returns:
            Normally ``(lps, hidden_list)``. With return_rating_logits, also
            returns attached five-way digit logits as ``(sample, dim, logits)``.
        """
        model = model or self.model
        target_device = self._model_device(model, self.accelerator.device)
        forward_model = model

        n = len(input_ids_list)
        if n == 0:
            return [], []
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        prompt_to_idx: Dict[int, List[int]] = {}
        for i, pi in enumerate(prompt_indices):
            prompt_to_idx.setdefault(pi, []).append(i)

        lps: List = [None] * n
        hidden_list: List = [None] * n
        rating_logits: List[Tuple[int, int, torch.Tensor]] = []
        ctx = torch.enable_grad() if compute_policy_grad else torch.no_grad()
        with ctx:
            for pi, idxs in sorted(prompt_to_idx.items()):
                G = len(idxs)
                max_len = max(input_ids_list[i].shape[0] for i in idxs)
                batch_ids = torch.full((G, max_len), pad_id, dtype=torch.long, device=target_device)
                batch_mask = torch.zeros((G, max_len), dtype=torch.long, device=target_device)
                real_lens = []
                for r, i in enumerate(idxs):
                    ln = input_ids_list[i].shape[0]
                    real_lens.append(ln)
                    batch_ids[r, :ln] = input_ids_list[i].to(target_device)
                    batch_mask[r, :ln] = attn_masks_list[i].to(target_device)

                fwd_kwargs = {"input_ids": batch_ids, "attention_mask": batch_mask}
                af = audio_features[pi] if pi < len(audio_features) else None
                am = audio_masks[pi] if pi < len(audio_masks) else None
                if af is not None and am is not None:
                    fwd_kwargs["input_features"] = af.to(target_device).repeat(
                        G, *([1] * (af.dim() - 1)))
                    fwd_kwargs["feature_attention_mask"] = am.to(target_device).repeat(
                        G, *([1] * (am.dim() - 1)))
                if return_hidden:
                    fwd_kwargs["output_hidden_states"] = True

                outputs = forward_model(**fwd_kwargs)
                logits = outputs.logits  # [G, max_len, vocab]
                log_probs = F.log_softmax(logits, dim=-1)
                shift_lp = log_probs[:, :-1, :]
                shift_ids = batch_ids[:, 1:]
                token_lp = shift_lp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)

                for r, i in enumerate(idxs):
                    start = seq_starts_list[i]
                    rl = real_lens[r]
                    lps[i] = token_lp[r, start - 1: rl - 1].to(self.accelerator.device)
                    if return_hidden:
                        hidden_list[i] = outputs.hidden_states[-1][r:r + 1]
                    if return_rating_logits:
                        positions = self._find_rating_logit_positions(
                            batch_ids[r, :rl], start,
                            attn_masks_list[i][start:].to(batch_ids.device),
                        )
                        for dim, position in positions:
                            rating_logits.append((
                                i,
                                dim,
                                outputs.logits[r, position, R_DIGIT_IDS],
                            ))

        if return_rating_logits:
            return lps, hidden_list, rating_logits
        return lps, hidden_list

    def _find_rating_logit_positions(
        self, ids: torch.Tensor, seq_start: int,
        completion_mask: Optional[torch.Tensor] = None,
    ) -> List[Tuple[int, int]]:
        """Locate the LM-logit rows predicting the three emitted R digits."""
        scan_end = ids.shape[0]
        if completion_mask is not None:
            scan_end = min(scan_end, seq_start + int(completion_mask.sum().item()))
        digit_tensor = torch.tensor(R_DIGIT_IDS, device=ids.device)
        candidates = torch.isin(
            ids[seq_start:scan_end], digit_tensor,
        ).nonzero(as_tuple=False).flatten() + seq_start
        by_dim: Dict[int, List[int]] = {
            dim: [] for dim in range(len(RATING_SECTIONS))
        }
        for candidate in candidates.tolist():
            window = self.tokenizer.decode(
                ids[max(seq_start, candidate - 24):candidate + 1].tolist()
            )
            matches = list(re.finditer(
                r"<(PLANNING|CAPTION|REASONING)>\s*<R\s*([1-5])\s*$",
                window, flags=re.IGNORECASE,
            ))
            if matches:
                dim = RATING_SECTIONS.index(matches[-1].group(1).upper())
                by_dim[dim].append(candidate - 1)
        return [
            (dim, positions[0])
            for dim, positions in by_dim.items()
            if len(positions) == 1 and positions[0] >= 0
        ]

    def _rating_count_for_group(
        self, input_ids_list, seq_starts_list, completion_masks,
    ) -> int:
        return sum(
            len(self._find_rating_logit_positions(ids, start, mask))
            for ids, start, mask in zip(
                input_ids_list, seq_starts_list, completion_masks,
            )
        )

    def _aux_sum_from_rating_logits(
        self,
        rating_logits: List[Tuple[int, int, torch.Tensor]],
        teacher_scores_list: List[dict],
    ) -> Tuple[Optional[torch.Tensor], int]:
        if not rating_logits:
            return None, 0
        teacher = torch.tensor([
            [float(score.get("D_P", 0.5)),
             float(score.get("D_C", 0.5)),
             float(score.get("D_R", 0.5))]
            for score in teacher_scores_list
        ], device=self.accelerator.device, dtype=torch.float)
        labels = self._build_soft_label(teacher.reshape(-1, 1)).reshape(
            len(teacher_scores_list), len(RATING_SECTIONS), self.num_ratings,
        )
        logits = torch.stack([entry[2] for entry in rating_logits])
        selected_labels = torch.stack([
            labels[sample_idx, dim]
            for sample_idx, dim, _ in rating_logits
        ])
        loss_sum = F.kl_div(
            F.log_softmax(logits.float(), dim=-1),
            selected_labels,
            reduction="sum",
        )
        return loss_sum, len(rating_logits)

    # ======================================================================
    # Task A: Extract rating norm from generated tokens
    # ======================================================================

    @staticmethod
    def _extract_section_ratings(response: str) -> Tuple[List[float], int]:
        """Return normalized P/C/R ratings and the number of valid coordinates.

        Each section rating is mapped with ``u=(R-1)/4``. Requiring all three
        coordinates avoids silently treating whichever rating happens to appear
        first as a global difficulty value. A missing, duplicated, or malformed
        coordinate contributes 0 while the norm denominator stays fixed at 3;
        omitting a rating can therefore never increase the cap or reduce KL.
        """
        values = []
        valid_count = 0
        for section in RATING_SECTIONS:
            open_pattern = rf"<\s*{section}\s*>"
            close_pattern = rf"</\s*{section}\s*>"
            opens = re.findall(open_pattern, response, flags=re.IGNORECASE)
            closes = re.findall(close_pattern, response, flags=re.IGNORECASE)
            blocks = re.findall(
                open_pattern + r"(.*?)" + close_pattern,
                response, flags=re.IGNORECASE | re.DOTALL,
            )
            if len(opens) != 1 or len(closes) != 1 or len(blocks) != 1:
                values.append(0.0)
                continue

            body = blocks[0]
            prefix = re.match(r"\s*<\s*R([1-5])\s*>", body, re.IGNORECASE)
            all_ratings = re.findall(
                r"<\s*R([1-5])\s*>", body, flags=re.IGNORECASE,
            )
            if prefix is not None and len(all_ratings) == 1:
                values.append((int(prefix.group(1)) - 1) / 4.0)
                valid_count += 1
            else:
                values.append(0.0)
        return values, valid_count

    def _extract_r_norm(self, responses: List[str]) -> torch.Tensor:
        """Extract the normalized Euclidean norm of all three section ratings.

        IMPORTANT: token-based detection is unreliable for Qwen2.5-Omni —
        `<R3>` tokenizes as token 23370 only in isolation; inside a section
        tag (`<PLANNING><R3>`) the BPE merges `<` into `><` so token 23370
        never appears. We therefore scan decoded response text with regexes.

        For u_P/u_C/u_R in [0, 1], the scalar used by dynamic KL and the
        compression level is ``sqrt((u_P²+u_C²+u_R²)/3)``. This is the
        Euclidean norm divided by sqrt(3), so it remains in [0, 1]. Missing or
        malformed coordinates contribute zero with a fixed denominator; format
        reward and the explicit triplet-coverage metric expose those cases.

        Returns:
            r_norm: [n_responses] float in [0, 1]
        """
        device = self.accelerator.device
        n = len(responses)
        r_norm = torch.zeros(n, device=device)

        for i, response in enumerate(responses):
            values, _ = self._extract_section_ratings(response)
            r_norm[i] = math.sqrt(sum(v * v for v in values) / 3.0)

        return r_norm

    # ======================================================================
    # Legacy word-count length helper (inactive; retained for old callers)
    # ======================================================================

    @staticmethod
    def _extract_section_lengths(text: str) -> List[float]:
        """Count approximate tokens (words) in each section from decoded text."""
        lens = []
        for tag in ["PLANNING", "CAPTION", "REASONING"]:
            m = re.search(
                rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE
            )
            if m:
                content = m.group(1).strip()
                # Rough token estimate: whitespace-split tokens for English,
                # each CJK char counts as a token
                tokens = 0
                for seg in re.findall(
                    r'[一-鿿　-〿]|[^一-鿿　-〿]+',
                    content
                ):
                    if re.match(r'[一-鿿　-〿]', seg):
                        tokens += len(seg)
                    else:
                        tokens += len(seg.split())
                lens.append(float(tokens))
            else:
                # Section missing — assume baseline length
                lens.append(float(0))
        return lens

    def _compute_length_penalty(
        self, responses: List[str], r_norm: torch.Tensor
    ) -> torch.Tensor:
        """Penalize length mismatch with predicted difficulty.

        penalty = -coeff * Σ max(0, -sign(r_norm-0.5) * delta)²

        Only penalizes when direction is wrong:
          R5(难) 但写得短 → 惩罚
          R1(易) 但写得长 �� 惩罚
          R5 写长 / R1 写短 → 不惩罚
        """
        device = self.accelerator.device
        n = len(responses)
        penalties = torch.zeros(n, device=device)

        for i in range(n):
            sections = self._extract_section_lengths(responses[i])
            rn = r_norm[i].item()
            sign_r = 1.0 if rn > 0.5 else (-1.0 if rn < 0.5 else 0.0)

            total_penalty = 0.0
            for sec_len, base_len in zip(sections, self.base_lens):
                if base_len <= 0 or sec_len <= 0:
                    continue
                delta = (sec_len - base_len) / max(base_len, 1.0)
                mismatch = -sign_r * delta
                if mismatch > 0:
                    total_penalty += mismatch ** 2

            penalties[i] = -self.lp_coeff * total_penalty

        return penalties

    # ======================================================================
    # Task B: self-difficulty conditioned output compression
    # ======================================================================

    def _count_compression_tokens_by_section(
        self, response: str,
    ) -> Tuple[Dict[str, int], bool]:
        """Return per-section token counts and whether structure is canonical."""
        special_ids = set(getattr(self.tokenizer, "all_special_ids", None) or [])

        def _count_clean(text: str) -> int:
            content = re.sub(r"<[^>\n]+>", " ", text).strip()
            if not content:
                return 0
            token_ids = self.tokenizer.encode(content, add_special_tokens=False)
            return sum(token_id not in special_ids for token_id in token_ids)

        lengths: Dict[str, int] = {}
        all_sections_valid = True
        for section in (*RATING_SECTIONS, "SUMMARY"):
            open_pattern = rf"<\s*{section}\s*>"
            close_pattern = rf"</\s*{section}\s*>"
            opens = re.findall(open_pattern, response, flags=re.IGNORECASE)
            closes = re.findall(close_pattern, response, flags=re.IGNORECASE)
            matches = re.findall(
                open_pattern + r"(.*?)" + close_pattern, response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if len(opens) != 1 or len(closes) != 1 or len(matches) != 1:
                all_sections_valid = False
            # Duplicate closed bodies are all charged; invalid structure will
            # additionally use the conservative global fallback below.
            lengths[section] = sum(_count_clean(body) for body in matches)
        return lengths, all_sections_valid

    @staticmethod
    def _first_think_block(text: str) -> str:
        """Return the first complete THINK body, or the original text.

        A handful of distilled records contain a valid answer followed by a
        second answer pasted after the first RESPONSE block. The compressed
        SFT trace is the first complete THINK block; counting the pasted suffix
        would silently turn a local reference length into a multi-answer cap.
        """
        match = re.search(
            r"<\s*THINK(?:\s+[^>]*)?\s*>(.*?)</\s*THINK\s*>",
            text, flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1) if match else text

    def _count_rated_section_tokens(
        self, text: str, *, first_think_only: bool = False,
    ) -> Tuple[Dict[str, int], bool]:
        """Count P/C/R bodies and validate exactly one of each section."""
        if first_think_only:
            text = self._first_think_block(text)
        all_lengths, _ = self._count_compression_tokens_by_section(text)
        valid = True
        for section in RATING_SECTIONS:
            open_pattern = rf"<\s*{section}\s*>"
            close_pattern = rf"</\s*{section}\s*>"
            matches = re.findall(
                open_pattern + r"(.*?)" + close_pattern, text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if (
                len(re.findall(open_pattern, text, flags=re.IGNORECASE)) != 1
                or len(re.findall(close_pattern, text, flags=re.IGNORECASE)) != 1
                or len(matches) != 1
            ):
                valid = False
        lengths = {section: all_lengths[section] for section in RATING_SECTIONS}
        return lengths, valid

    def _count_compression_tokens(self, response: str) -> int:
        """Count model tokens only inside the compressible reasoning sections.

        This deliberately excludes section tags, `<RESPONSE>`, EOS, and batch
        padding. It matches the scope used to create the compressed SFT target,
        so a required final transcript/answer is never charged to the reasoning
        budget.
        """
        special_ids = set(getattr(self.tokenizer, "all_special_ids", None) or [])

        def _count_clean(text: str) -> int:
            content = re.sub(r"<[^>\n]+>", " ", text).strip()
            if not content:
                return 0
            token_ids = self.tokenizer.encode(
                content, add_special_tokens=False,
            )
            return sum(token_id not in special_ids for token_id in token_ids)

        section_lengths, all_sections_valid = (
            self._count_compression_tokens_by_section(response)
        )
        total = sum(section_lengths.values())

        if all_sections_valid:
            return total

        # A malformed/unclosed section must not make a long rationale invisible.
        # Fall back to the whole THINK prefix, stripping all markup and excluding
        # RESPONSE. This is conservative by design: formatting failures may pay
        # more, but can never obtain a cheaper length than their visible text.
        before_response = re.split(
            r"<\s*RESPONSE(?:\s+[^>]*)?\s*>", response,
            maxsplit=1, flags=re.IGNORECASE,
        )[0]
        think_match = re.search(
            r"<\s*THINK(?:\s+[^>]*)?\s*>(.*?)(?:</\s*THINK\s*>|$)",
            before_response, flags=re.IGNORECASE | re.DOTALL,
        )
        fallback = think_match.group(1) if think_match else before_response
        return _count_clean(fallback)

    def _compute_section_compression_reward(
        self,
        responses: List[str],
        resp_lengths: List[int],
        r_norm: torch.Tensor,
        families: Optional[List[str]] = None,
    ) -> Tuple[
        torch.Tensor, List[Dict[str, int]], List[bool], List[bool], List[bool]
    ]:
        """Score each P/C/R length against its own rating-conditioned cap.

        SUMMARY uses a family-specific fixed cap and no under-budget term.
        Invalid responses receive a fixed conservative reward. They must never
        fall back to the global objective: that creates a discontinuity where
        deleting one rating can be cheaper than obeying all section-wise caps.
        Section terms are averaged to preserve the approximate scale of the
        existing sequence-level shaping reward.
        """
        rewards = torch.zeros(len(responses), device=self.accelerator.device)
        all_lengths: List[Dict[str, int]] = []
        validity: List[bool] = []
        over_events: List[bool] = []
        under_events: List[bool] = []
        for i, response in enumerate(responses):
            family = (
                str(families[i]).lower()
                if families is not None and i < len(families) else ""
            )
            lengths, sections_valid = self._count_compression_tokens_by_section(
                response
            )
            ratings, rating_count = self._extract_section_ratings(response)
            valid = (
                sections_valid
                and rating_count == len(RATING_SECTIONS)
                and family in self.comp_section_caps_by_family
                and family in self.comp_summary_caps_by_family
            )
            all_lengths.append(lengths)
            validity.append(valid)
            if not valid:
                # Hard floor within the existing compression clamp.  A missing,
                # duplicated, or malformed P/C/R coordinate cannot improve the
                # reward by switching to the looser global cap contract.
                rewards[i] = self.comp_invalid_reward
                over_events.append(False)
                under_events.append(False)
                continue

            over_terms = []
            under_terms = []
            for section, rating in zip(RATING_SECTIONS, ratings):
                level = max(0, min(4, int(round(rating * 4.0))))
                cap = self.comp_section_caps_by_family[family][section][level]
                length = float(lengths[section])
                cap_min = self.comp_min_frac * cap
                over_terms.append(max(0.0, length - cap) / cap)
                under_terms.append(
                    max(0.0, cap_min - length) / max(cap_min, 1.0)
                )

            summary_cap = self.comp_summary_caps_by_family[family]
            summary_length = float(lengths["SUMMARY"])
            over_terms.append(
                max(0.0, summary_length - summary_cap) / summary_cap
            )
            rewards[i] = (
                -self.comp_coeff * sum(over_terms) / len(over_terms)
                -self.comp_under_coeff * sum(under_terms) / len(under_terms)
            )
            over_events.append(any(term > 0.0 for term in over_terms))
            under_events.append(any(term > 0.0 for term in under_terms))
        return rewards, all_lengths, validity, over_events, under_events

    def _compute_sft_trace_interval_reward(
        self,
        responses: List[str],
        compressed_answers: List[str],
    ) -> Tuple[
        torch.Tensor,
        List[Dict[str, int]],
        List[Dict[str, int]],
        List[bool],
        List[bool],
        List[bool],
    ]:
        """Regularize each P/C/R length around its compressed SFT trace.

        For rating ``b`` in ``{1, ..., K}`` and trace length ``C``, the
        zero-penalty interval is

            C * [1 - (K + 1 - b) delta, 1 + b delta],
            delta = 1 / (K (K + 1)).

        Its width is always C/K and it shifts monotonically to the right as the
        emitted rating increases. The raw distance sum is multiplied by the
        configured compression coefficient before the common reward clamp.
        SUMMARY and RESPONSE are deliberately outside this objective.
        """
        device = self.accelerator.device
        rewards = torch.zeros(len(responses), device=device)
        generated_lengths: List[Dict[str, int]] = []
        trace_lengths: List[Dict[str, int]] = []
        validity: List[bool] = []
        over_events: List[bool] = []
        under_events: List[bool] = []
        K = self.comp_rating_levels
        delta = 1.0 / (K * (K + 1))

        for i, response in enumerate(responses):
            generated, generated_valid = self._count_rated_section_tokens(
                response,
            )
            trace_text = compressed_answers[i] if i < len(compressed_answers) else ""
            trace, trace_valid = self._count_rated_section_tokens(
                str(trace_text or ""), first_think_only=True,
            )
            ratings, rating_count = self._extract_section_ratings(response)
            valid = (
                generated_valid
                and trace_valid
                and rating_count == len(RATING_SECTIONS)
                and all(trace[section] > 0 for section in RATING_SECTIONS)
            )
            generated_lengths.append(generated)
            trace_lengths.append(trace)
            validity.append(valid)
            if not valid:
                rewards[i] = self.comp_invalid_reward
                over_events.append(False)
                under_events.append(False)
                continue

            raw_distance = 0.0
            any_over = False
            any_under = False
            for section, normalized_rating in zip(RATING_SECTIONS, ratings):
                b = max(
                    1,
                    min(K, int(round(normalized_rating * (K - 1))) + 1),
                )
                C = float(trace[section])
                lower = C * (1.0 - (K + 1 - b) * delta)
                upper = C * (1.0 + b * delta)
                L = float(generated[section])
                distance = max(lower - L, 0.0, L - upper)
                raw_distance += distance / C
                any_under = any_under or L < lower
                any_over = any_over or L > upper

            rewards[i] = -self.comp_coeff * raw_distance
            over_events.append(any_over)
            under_events.append(any_under)

        return (
            rewards,
            generated_lengths,
            trace_lengths,
            validity,
            over_events,
            under_events,
        )

    def _compute_compression_reward(
        self, resp_lengths: List[int], r_norm: torch.Tensor,
        families: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compression reward: output length must fit the cap set by the model's
        own normalized P/C/R rating norm.

            bin = family-specific RMS quantile; cap = caps_by_family[family][bin]
            reward = -comp_coeff * max(0, L - cap) / cap     (over-budget only)

        Low rating norm → small cap → strong penalty for long outputs ⇒ the model
        learns to compress easy questions. High rating norm → large cap → free to be
        long; the quality reward is what prevents over-compression there.
        The aux loss anchors the self-rating to the teacher difficulty, so the model
        cannot simply always rate R5 to dodge the cap.
        ADDITIONAL anti-gaming term: under-budget (self-rates HIGH but writes SHORT)
        is also mildly penalized — otherwise the model could escape compression by
        always rating R5 (large cap) while keeping answers short. Under-budget is
        normalized by cap_min so a high rating demands a substantial output.
        """
        device = self.accelerator.device
        n = len(resp_lengths)
        rewards = torch.zeros(n, device=device)
        for i in range(n):
            family = families[i] if families is not None and i < len(families) else ""
            _, cap = self._compression_bin(r_norm[i].item(), family)
            if cap <= 0:
                continue
            cap_min = self.comp_min_frac * cap
            L = float(resp_lengths[i])
            over = max(0.0, L - cap) / cap
            under = max(0.0, cap_min - L) / max(cap_min, 1.0)
            rewards[i] = -self.comp_coeff * over - self.comp_under_coeff * under
        return rewards

    def _compression_bin(self, r_norm: float, family: str = "") -> Tuple[int, float]:
        """Map normalized difficulty to a bin and token cap.

        New configs provide task-family-specific continuous-RMS quantile edges.
        The legacy five evenly spaced levels remain a compatibility fallback for
        old checkpoints/configs only.
        """
        family = str(family).lower()
        if family in self.comp_caps_by_family:
            edges = self.comp_edges_by_family[family]
            caps = self.comp_caps_by_family[family]
            bin_idx = bisect_right(edges, float(r_norm))
            bin_idx = max(0, min(len(caps) - 1, bin_idx))
            return bin_idx, caps[bin_idx]

        level = int(round(float(r_norm) * 4.0))
        level = max(0, min(len(self.comp_caps) - 1, level))
        return level, float(self.comp_caps[level])

    # ======================================================================
    # Task C: Auxiliary value loss (soft logits alignment)
    # ======================================================================

    def _build_soft_label(self, teacher_scores: torch.Tensor) -> torch.Tensor:
        """Build a 5-way soft rating label from scalar teacher scores.

        ``teacher_scores`` is [N, 1] on the same [0, 1] D_P/D_C/D_R axes used
        to create the SFT section tags. SFT bins ``score * 5`` at half-integer
        boundaries, so the Gaussian center is ``clamp(score * 5, 1, 5)``.

        Returns:
            soft_label: [batch_size, 5] probability distribution
        """
        if teacher_scores.ndim == 1:
            teacher_scores = teacher_scores.unsqueeze(-1)
        if teacher_scores.ndim != 2 or teacher_scores.shape[-1] != 1:
            raise ValueError(
                "teacher_scores must have shape [N, 1], got "
                f"{tuple(teacher_scores.shape)}"
            )

        eps = 1e-8
        center = (teacher_scores.clamp(0.0, 1.0) * 5.0).clamp(1.0, 5.0)

        # indices: [0, 1, 2, 3, 4] → R1..R5
        idx = torch.arange(1, self.num_ratings + 1, device=teacher_scores.device,
                           dtype=torch.float).unsqueeze(0)  # [1, 5]
        dist = (idx - center) ** 2  # [N, 5]
        weights = torch.exp(-0.5 * dist / self.soft_var)
        soft_label = weights / (weights.sum(dim=-1, keepdim=True) + eps)
        return soft_label

    def _compute_aux_loss(
        self,
        hidden_states_list: List[torch.Tensor],
        input_ids_list: List[torch.Tensor],
        teacher_scores_list: List[dict],
        seq_starts_list: Optional[List[int]] = None,
        completion_masks: Optional[List[torch.Tensor]] = None,
    ):
        """Auxiliary rating loss with gradients flowing into the backbone.

        Uses pre-computed hidden states (from the policy forward pass)
        so no extra forward pass is needed.

        The <R> token logits come from lm_head on attached hidden states. In
        LoRA training lm_head stays frozen, but its input gradient still carries
        this auxiliary signal back into the trainable adapters.

        Each generated PLANNING/CAPTION/REASONING rating is supervised against
        its matching continuous teacher D_P/D_C/D_R score. The scalar norm used
        elsewhere is then derived from those three emitted ratings.

        NOTE on detection (2026-08-13 FIXED): Qwen2.5-Omni BPE merges the rating's
        `<` into the previous section tag — `<PLANNING><R3>` → '<','PL','ANN','ING',
        '><','R','3','>' — so token 23370 (`<R`) never appears and the OLD
        `ids == R_TOKEN_ID` detection found nothing (loss was dead even when
        enabled). The rating DIGIT (16-20) is always a standalone token; the fix
        locates digit tokens and confirms via a decoded window that they form
        `<R[1-5]>`, then reads the LM-head logits at the preceding 'R' token.
        `aux/r_token_samples` still reports how many responses actually carried a
        rating, so a constant ~KL(soft_label || uniform) is never mistaken for a
        learning mechanism.

        Returns:
            (aux_loss_or_None, n_r_samples)
        """
        if not teacher_scores_list or not hidden_states_list:
            return None, 0

        device = self.accelerator.device
        n = len(input_ids_list)

        # Teacher tensor: [N, 3], one target for each emitted section rating.
        teacher_tensor = torch.tensor([
            [float(ts.get("D_P", 0.5)),
             float(ts.get("D_C", 0.5)),
             float(ts.get("D_R", 0.5))]
            for ts in teacher_scores_list
        ], device=device, dtype=torch.float)

        # Soft labels: [N, 3, 5], preserving the P/C/R axes independently.
        soft_label = self._build_soft_label(
            teacher_tensor.reshape(-1, 1),
        ).reshape(n, len(RATING_SECTIONS), self.num_ratings)

        # Get lm_head (Qwen2.5-Omni uses thinker submodel's lm_head).
        # Use the UNWRAPPED model for attribute access; the hidden state still
        # belongs to the policy/DDP forward graph, so its input gradient flows
        # back through the trainable adapters.
        base = self.accelerator.unwrap_model(self.model)
        if hasattr(base, "thinker") and hasattr(base.thinker, "lm_head"):
            lm_head = base.thinker.lm_head
        elif hasattr(base, "lm_head"):
            lm_head = base.lm_head
        else:
            lm_head = base.get_output_embeddings()
        if lm_head is None:
            return None, 0

        all_r_hidden = []
        r_teacher_indices = []  # (sample index, P/C/R dimension)
        n_r_samples = 0

        digit_tensor = torch.tensor(R_DIGIT_IDS, device=device, dtype=torch.long)
        for i in range(n):
            ids = input_ids_list[i].to(device)
            hidden = hidden_states_list[i]  # [1, seq_len, hidden_dim]

            # Search generated policy actions only. A question may itself
            # contain literal rating-like text, and rectangular generation can
            # retain tokens after EOS; neither is an auxiliary prediction.
            scan_start = (
                int(seq_starts_list[i])
                if seq_starts_list is not None and i < len(seq_starts_list)
                else 0
            )
            scan_end = ids.shape[0]
            if completion_masks is not None and i < len(completion_masks):
                valid_completion = int(completion_masks[i].sum().item())
                scan_end = min(scan_end, scan_start + valid_completion)
            if scan_start >= scan_end:
                continue

            # Find <R1>..<R5> rating positions.
            # IMPORTANT: Qwen2.5-Omni BPE merges the rating's `<` into the previous
            # section tag — <PLANNING><R3> tokenizes to '<','PL','ANN','ING','><',
            # 'R','3','>' — so token 23370 (<R) NEVER appears and the old
            # `ids == R_TOKEN_ID` detection found nothing (aux loss was dead).
            # The rating digit (16-20) IS a standalone token; locate digit tokens
            # and associate each one with its immediately preceding section tag.
            is_digit = torch.isin(ids[scan_start:scan_end], digit_tensor)
            digit_positions = is_digit.nonzero(as_tuple=False) + scan_start
            positions_by_dim: Dict[int, List[int]] = {
                dim: [] for dim in range(len(RATING_SECTIONS))
            }
            for dp in digit_positions:
                pos = dp.item()
                # Decode only through the candidate digit and anchor the match
                # there. Looking past it can falsely classify an ordinary digit
                # in the body when a real <Rk> appears earlier nearby.
                win = self.tokenizer.decode(
                    ids[max(scan_start, pos - 24):pos + 1].tolist()
                )
                matches = list(re.finditer(
                    r"<(PLANNING|CAPTION|REASONING)>\s*<R\s*([1-5])\s*$",
                    win, flags=re.IGNORECASE,
                ))
                if not matches:
                    continue
                section = matches[-1].group(1).upper()
                dim = RATING_SECTIONS.index(section)
                positions_by_dim[dim].append(pos)

            for dim, positions in positions_by_dim.items():
                # Missing/duplicated coordinates are structurally invalid and
                # receive no aux target; the format reward handles the error.
                if len(positions) != 1:
                    continue
                # The LM head at (digit_pos - 1), the 'R' token, predicts digit.
                p = positions[0] - 1
                if p < 0:
                    continue
                # Keep hidden attached. A frozen lm_head is still differentiable
                # with respect to its input, which is how aux trains the LoRA path.
                h = hidden[:, p, :]  # [1, hidden_dim]
                all_r_hidden.append(h)
                r_teacher_indices.append((i, dim))
                n_r_samples += 1

        if not all_r_hidden:
            return None, n_r_samples

        r_hidden = torch.cat(all_r_hidden, dim=0)  # [n_r, hidden_dim]
        # Computing the full ~152k-vocabulary lm_head for only five rating
        # digits is wasteful, especially now that all three section ratings are
        # supervised. Selecting the exact five output rows is algebraically
        # identical and preserves gradients to r_hidden.
        if hasattr(lm_head, "weight"):
            digit_ids = torch.tensor(
                R_DIGIT_IDS, device=lm_head.weight.device, dtype=torch.long,
            )
            digit_weight = lm_head.weight.index_select(0, digit_ids)
            digit_bias = None
            if getattr(lm_head, "bias", None) is not None:
                digit_bias = lm_head.bias.index_select(0, digit_ids)
            r_logits = F.linear(r_hidden, digit_weight, digit_bias)
        else:
            r_logits = lm_head(r_hidden)[:, R_DIGIT_IDS]
        r_log_probs = F.log_softmax(r_logits, dim=-1)

        # Pair every emitted section rating with its matching teacher dimension.
        soft_label_r = torch.stack([
            soft_label[sample_idx, dim]
            for sample_idx, dim in r_teacher_indices
        ])
        aux_loss = F.kl_div(r_log_probs, soft_label_r, reduction="batchmean")

        return aux_loss, n_r_samples

    # ======================================================================
    # GRPO Loss (with Task A: dynamic KL)
    # ======================================================================

    def grpo_loss(
        self, log_probs, old_log_probs, advantages, ref_log_probs,
        beta_dynamic=None, lengths=None, completion_mask=None,
        kl_invalid=False,
    ):
        """Compute GRPO loss with per-sample dynamic KL.

        Args:
            log_probs: [G, resp_len] — current policy, with grad
            old_log_probs: [G, resp_len] — old/behavioral, detached
            advantages: [G] — normalized within-group
            ref_log_probs: [G, resp_len] — reference, detached
            beta_dynamic: [G] or scalar — RELATIVE KL modulation in [0.5, 1.0]
                (maximum three-rating norm halves KL; zero norm keeps it full).
                The absolute KL
                strength is self.beta only — beta_dynamic must NOT contain an
                extra coefficient, otherwise the KL term is double-scaled into
                insignificance.
            lengths: legacy [G] length tensor used to construct a prefix mask
                when completion_mask is not provided.
            completion_mask: [G, resp_len], 1 through the first EOS inclusive
                and 0 for all post-EOS/batch padding. Ratio and KL are averaged
                only over these real completion tokens.
            kl_invalid: whether this prompt group has no quality-reward
                variation. Sparse-KL skipping is controlled by this explicit
                quality-only signal, not by advantages that also contain
                format/compression rewards.
        """
        G = advantages.shape[0]

        # Masked means: EOS-following generation padding and group padding are
        # not policy actions and must not affect either the PPO ratio or KL.
        if completion_mask is not None:
            if completion_mask.shape != log_probs.shape:
                raise ValueError(
                    "completion_mask must have the same shape as log_probs: "
                    f"{tuple(completion_mask.shape)} != {tuple(log_probs.shape)}"
                )
            mask = completion_mask.to(
                device=log_probs.device, dtype=log_probs.dtype,
            )
        elif lengths is not None:
            seq_len = log_probs.shape[1]
            mask = (
                torch.arange(seq_len, device=log_probs.device)[None, :]
                < lengths[:, None]
            ).to(dtype=log_probs.dtype)
        else:
            mask = torch.ones_like(log_probs)

        denom = mask.sum(dim=-1).clamp(min=1.0)
        ratio_log = ((log_probs - old_log_probs) * mask).sum(dim=-1) / denom

        # Schulman K3 estimator, token-wise. Unlike mean(log pi - log pi_ref),
        # this is non-negative and has the correct KL-penalty gradient.
        ref_policy_log_ratio = (ref_log_probs - log_probs).float()
        ref_policy_log_ratio = torch.where(
            mask.bool(), ref_policy_log_ratio,
            torch.zeros_like(ref_policy_log_ratio),
        )
        per_token_kl = (
            torch.exp(ref_policy_log_ratio) - ref_policy_log_ratio - 1.0
        )
        per_sample_kl = (
            (per_token_kl * mask.float()).sum(dim=-1)
            / denom.float()
        )

        rho = torch.exp(ratio_log)

        surr1 = rho * advantages
        surr2 = torch.clamp(rho, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Sparse KL: only a quality-constant group is invalid. Total advantages
        # may still vary because of format/compression shaping and must not
        # silently turn the reference constraint on or off.
        if isinstance(kl_invalid, torch.Tensor):
            is_invalid = bool(kl_invalid.detach().all().item())
        else:
            is_invalid = bool(kl_invalid)

        if is_invalid:
            kl_loss = torch.tensor(0.0, device=advantages.device)
            kl_skipped = 1.0
        else:
            if beta_dynamic is not None and beta_dynamic.numel() == G:
                # Element-wise multiply by per-sample RELATIVE weight
                kl_loss = (per_sample_kl * beta_dynamic).mean()
            else:
                kl_loss = per_sample_kl.mean()

            kl_skipped = 0.0

        total = policy_loss + self.beta * kl_loss

        return total, {
            "loss/policy": policy_loss.item(),
            "loss/kl": kl_loss.item(),
            "loss/total": total.item(),
            "loss/kl_skipped": kl_skipped,
            "loss/rho_mean": rho.mean().item(),
        }

    @staticmethod
    def normalize_advantages(group_rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if group_rewards.max() == group_rewards.min():
            return torch.zeros_like(group_rewards)
        return (group_rewards - group_rewards.mean()) / (group_rewards.std() + eps)

    @staticmethod
    def combine_objectives(
        group_losses: List[torch.Tensor],
        aux_loss: Optional[torch.Tensor],
        aux_coeff: float,
    ) -> torch.Tensor:
        """Average prompt groups, then add a batch-size-invariant aux term."""
        if not group_losses:
            raise ValueError("group_losses must not be empty")
        objective = torch.stack(group_losses).mean()
        if aux_loss is not None and aux_coeff > 0:
            objective = objective + aux_coeff * aux_loss
        return objective

    @staticmethod
    def _empty_step_metrics() -> Dict[str, float]:
        return {
            "loss/total": 0.0, "loss/policy": 0.0, "loss/kl": 0.0,
            "loss/aux": 0.0, "loss/r_norm_mean": 0.0,
            "loss/kl_skipped": 0.0, "loss/rho_mean": 0.0,
            "reward/base_mean": 0.0, "reward/compression_mean": 0.0,
            "reward/quality_mean": 0.0, "reward/format_mean": 0.0,
            "reward/format_contribution_mean": 0.0,
            "reward/mean": 0.0, "reward/std": 0.0,
            "reward/s2tt": 0.0, "reward/choice": 0.0, "reward/open": 0.0,
            "adv/mean": 0.0, "adv/std": 0.0,
            "gen/resp_len_mean": 0.0, "gen/pct_has_response_tag": 0.0,
            "gen/r_token_frac": 0.0, "gen/r_triplet_frac": 0.0,
            "aux/r_token_samples": 0,
        }

    def _forward_one_prompt_group(
        self, gen: dict, prompt_index: int, model,
        compute_policy_grad: bool, return_rating_logits: bool = False,
    ):
        start = prompt_index * self.G
        end = start + self.G
        local_prompt_indices = [0] * self.G
        kwargs = dict(
            input_ids_list=gen["input_ids"][start:end],
            attn_masks_list=gen["attention_masks"][start:end],
            seq_starts_list=gen["seq_starts"][start:end],
            prompt_indices=local_prompt_indices,
            audio_features=[gen["audio_features"][prompt_index]],
            audio_masks=[gen["audio_masks"][prompt_index]],
            model=model,
            compute_policy_grad=compute_policy_grad,
            return_hidden=False,
            return_rating_logits=return_rating_logits,
        )
        return self._compute_log_probs_grouped(**kwargs)

    def _pad_group_for_loss(self, lps, completion_masks):
        max_len = max(lp.shape[0] for lp in lps)
        padded_lps = torch.stack([
            F.pad(lp, (0, max_len - lp.shape[0]), "constant", 0.0)
            if lp.shape[0] < max_len else lp[:max_len]
            for lp in lps
        ])
        padded_masks = []
        for mask in completion_masks:
            mask = mask.to(
                device=self.accelerator.device, dtype=padded_lps.dtype,
            )
            padded_masks.append(
                F.pad(mask, (0, max_len - mask.shape[0]), "constant", 0.0)
                if mask.shape[0] < max_len else mask[:max_len]
            )
        return padded_lps, torch.stack(padded_masks)

    def _backward_streaming_groups(
        self, *, gen, rewards, quality_scores, format_scores, base_rewards,
        comp_rewards, r_norm_all, teacher_scores_list,
    ) -> Dict[str, float]:
        """Backprop one prompt graph at a time while preserving the global mean."""
        import torch.distributed as dist

        metrics = self._empty_step_metrics()
        n_prompts = len(gen["audio_features"])
        if n_prompts < 1 or n_prompts > self.micro_slots_per_rank:
            raise RuntimeError(
                f"rank has {n_prompts} real prompts for "
                f"{self.micro_slots_per_rank} synchronized slots"
            )

        local_aux_count = self._rating_count_for_group(
            gen["input_ids"], gen["seq_starts"], gen["completion_masks"],
        ) if self.aux_coeff > 0 else 0
        global_aux_count_tensor = torch.tensor(
            local_aux_count, device=self.accelerator.device, dtype=torch.long,
        )
        if self.accelerator.num_processes > 1:
            dist.all_reduce(global_aux_count_tensor, op=dist.ReduceOp.SUM)
        global_aux_count = int(global_aux_count_tensor.item())
        world = self.accelerator.num_processes
        policy_scale = world / float(self.global_prompts_per_step)
        aux_scale = (
            self.aux_coeff * world / float(global_aux_count)
            if global_aux_count > 0 else 0.0
        )
        local_aux_sum_value = 0.0

        for slot in range(self.micro_slots_per_rank):
            is_real = slot < n_prompts
            group_index = slot if is_real else 0
            ref_lps, _ = self._forward_one_prompt_group(
                gen, group_index, self.ref_model, False,
            )
            policy_forward = self._forward_one_prompt_group(
                gen, group_index, self.model, True,
                return_rating_logits=(self.aux_coeff > 0 and is_real),
            )
            if self.aux_coeff > 0 and is_real:
                policy_lps, _, rating_logits = policy_forward
            else:
                policy_lps, _ = policy_forward
                rating_logits = []
            s, e = group_index * self.G, (group_index + 1) * self.G
            p_lps, completion_mask = self._pad_group_for_loss(
                policy_lps, gen["completion_masks"][s:e],
            )
            o_lps, _ = self._pad_group_for_loss(
                [lp.detach() for lp in policy_lps],
                gen["completion_masks"][s:e],
            )
            r_lps, _ = self._pad_group_for_loss(
                ref_lps, gen["completion_masks"][s:e],
            )

            if is_real:
                group_rn = r_norm_all[s:e]
                beta_dynamic = (
                    1.0 - 0.5 * group_rn
                    if self.use_dynamic_kl else torch.ones_like(group_rn)
                )
                grp_r = torch.tensor(
                    rewards[s:e], dtype=torch.float,
                    device=self.accelerator.device,
                )
                advantages = self.normalize_advantages(grp_r)
                grp_quality = [float(value) for value in quality_scores[s:e]]
                loss, group_metrics = self.grpo_loss(
                    p_lps, o_lps, advantages, r_lps,
                    beta_dynamic=beta_dynamic,
                    completion_mask=completion_mask,
                    kl_invalid=(
                        not grp_quality
                        or max(grp_quality) - min(grp_quality) <= 1e-8
                    ),
                )
                objective = policy_scale * loss
                aux_sum, aux_count = self._aux_sum_from_rating_logits(
                    rating_logits, teacher_scores_list[s:e],
                )
                if aux_sum is not None:
                    objective = objective + aux_scale * aux_sum
                    local_aux_sum_value += float(aux_sum.detach().item())
                    metrics["aux/r_token_samples"] += aux_count

                for key, value in group_metrics.items():
                    metrics[key] = metrics.get(key, 0.0) + value / n_prompts
                metrics["reward/quality_mean"] += (
                    sum(quality_scores[s:e]) / self.G / n_prompts
                )
                metrics["reward/format_mean"] += (
                    sum(format_scores[s:e]) / self.G / n_prompts
                )
                metrics["reward/format_contribution_mean"] += (
                    self.reward_pipeline.format_weight
                    * sum(format_scores[s:e]) / self.G / n_prompts
                )
                metrics["reward/base_mean"] += (
                    sum(base_rewards[s:e]) / self.G / n_prompts
                )
                metrics["reward/compression_mean"] += (
                    comp_rewards[s:e].mean().item() / n_prompts
                )
                metrics["reward/mean"] += grp_r.mean().item() / n_prompts
                metrics["reward/std"] += grp_r.std().item() / n_prompts
                metrics["adv/mean"] += advantages.mean().item() / n_prompts
                metrics["adv/std"] += advantages.std().item() / n_prompts
            else:
                # The zero-loss graph makes every rank execute the same number
                # of FSDP forward/backward collectives without changing data,
                # rewards, gradients, or curriculum counts.
                objective = p_lps.sum() * 0.0

            self.accelerator.backward(objective)
            del ref_lps, policy_lps, rating_logits, p_lps, o_lps, r_lps

        if local_aux_count > 0:
            if metrics["aux/r_token_samples"] != local_aux_count:
                raise RuntimeError(
                    "rating-logit extraction count changed between preflight "
                    f"and forward: {metrics['aux/r_token_samples']} != "
                    f"{local_aux_count}"
                )
            metrics["loss/aux"] = local_aux_sum_value / local_aux_count
            metrics["loss/total"] += self.aux_coeff * metrics["loss/aux"]
        return metrics

    # ======================================================================
    # Training step
    # ======================================================================

    def train_step(
        self, batch: dict, optimizer, lr_scheduler
    ) -> Dict[str, float]:
        """One GRPO training step with all three auxiliary mechanisms."""
        self.global_step += 1
        step_start = time.time()

        prompts = batch["question"]
        ground_truths = batch["ground_truth"]
        bucket_ids = batch.get("bucket_id", [0] * len(prompts))
        audios = batch.get("audio_waveform", [None] * len(prompts))

        # Teacher section scores (D_P, D_C, D_R) from curriculum dataset.
        # Expand to match responses (each prompt has G responses)
        per_prompt_scores = []
        for i in range(len(prompts)):
            ts = {}
            for key in ["D_P", "D_C", "D_R"]:
                val = batch.get(key, [0.5] * len(prompts))
                ts[key] = val[i] if isinstance(val, list) else float(val)
            per_prompt_scores.append(ts)
        teacher_scores_list = []
        for ts in per_prompt_scores:
            for _ in range(self.G):
                teacher_scores_list.append(ts.copy())

        # --- 1. Generate ---
        # Per-prompt generation length by task type: S2TT/choice stay short,
        # open-ended (long CoT) get a bigger budget.
        prompt_cat = []
        for i in range(len(prompts)):
            s = str((batch.get("subset") or [""] * len(prompts))[i]).lower()
            t = str((batch.get("type") or [""] * len(prompts))[i]).lower()
            prompt_cat.append("s2tt" if s == "s2tt" else ("choice" if t == "choice" else "open"))
        mtn_list = [
            self.max_new_tokens_by_type.get(c, self.gen_config.max_new_tokens)
            for c in prompt_cat
        ]
        gen = self.generate_responses(prompts, audios, max_new_tokens_list=mtn_list)
        if self.accelerator.is_main_process:
            print(f"  [step {self.global_step}] generation done ({len(gen['responses'])} responses)", flush=True)

        # --- 2. Extract normalized P/C/R rating norms (Task A) ---
        r_norm_all = self._extract_r_norm(gen["responses"])  # [n_gen]

        # --- 3. Compute base rewards ---
        gt_repeated = [ground_truths[pi] for pi in gen["prompt_indices"]]
        use_format = self.global_step < self.T_f

        # Pass the dataset's real subset/type as metadata so the reward router
        # does not have to re-guess from the ground-truth text, and so we can
        # log rewards per task type below.
        comp_answers = batch.get("compressed_answer") or batch.get("ground_truth")
        meta = []
        for pi in gen["prompt_indices"]:
            meta.append({
                "subset": (batch.get("subset") or [""] * len(prompts))[pi],
                "type": (batch.get("type") or [""] * len(prompts))[pi],
                "question": prompts[pi],
                # Teacher-distilled canonical answer — the reference to compare
                # the response against THIS (cleaner than the raw verbose GT).
                "compressed_answer": comp_answers[pi] if pi < len(comp_answers) else None,
            })

        # Quality and format are kept separate: quality is the only signal that
        # should drive competence/curriculum, and we want to log them apart.
        quality_scores = self.reward_pipeline.quality(
            gen["responses"], gt_repeated, meta,
        )
        format_scores = (
            self.reward_pipeline.format(gen["responses"], gt_repeated)
            if use_format else [0.0] * len(gen["responses"])
        )
        base_rewards = [
            quality_scores[i]
            + self.reward_pipeline.format_weight * format_scores[i]
            for i in range(len(quality_scores))
        ]

        # --- 4. Compression reward (Task B): output length must fit the cap set
        # by the model's normalized P/C/R rating norm. ---
        resp_lengths = [
            self._count_compression_tokens(response)
            for response in gen["responses"]
        ]
        comp_families = [
            prompt_cat[pi] for pi in gen["prompt_indices"]
        ]
        section_lengths_all: List[Dict[str, int]] = []
        section_validity: List[bool] = []
        section_over_flags: List[bool] = []
        section_under_flags: List[bool] = []
        trace_lengths_all: List[Dict[str, int]] = []
        if self.comp_mode == "section_wise":
            (
                comp_rewards,
                section_lengths_all,
                section_validity,
                section_over_flags,
                section_under_flags,
            ) = (
                self._compute_section_compression_reward(
                    gen["responses"], resp_lengths, r_norm_all, comp_families,
                )
            )
        elif self.comp_mode == "sft_trace_interval":
            trace_answers = [
                comp_answers[pi] if pi < len(comp_answers) else ""
                for pi in gen["prompt_indices"]
            ]
            (
                comp_rewards,
                section_lengths_all,
                trace_lengths_all,
                section_validity,
                section_over_flags,
                section_under_flags,
            ) = self._compute_sft_trace_interval_reward(
                gen["responses"], trace_answers,
            )
        else:
            comp_rewards = self._compute_compression_reward(
                resp_lengths, r_norm_all, comp_families,
            )
        comp_rewards = torch.clamp(comp_rewards, -0.5, 0.0)

        rewards = [
            base_rewards[i] + comp_rewards[i].item()
            for i in range(len(base_rewards))
        ]

        n_prompts = len(prompts)
        if self.streaming_backward:
            metrics = self._backward_streaming_groups(
                gen=gen,
                rewards=rewards,
                quality_scores=quality_scores,
                format_scores=format_scores,
                base_rewards=base_rewards,
                comp_rewards=comp_rewards,
                r_norm_all=r_norm_all,
                teacher_scores_list=teacher_scores_list,
            )
        else:
            # Legacy LoRA/DDP path: retain the original batchmean objective.
            ref_lps, _ = self._compute_log_probs_grouped(
                gen["input_ids"], gen["attention_masks"], gen["seq_starts"],
                gen["prompt_indices"], gen["audio_features"], gen["audio_masks"],
                model=self.ref_model, compute_policy_grad=False,
            )
            policy_lps, policy_hidden = self._compute_log_probs_grouped(
                gen["input_ids"], gen["attention_masks"], gen["seq_starts"],
                gen["prompt_indices"], gen["audio_features"], gen["audio_masks"],
                model=self.model, compute_policy_grad=True,
                return_hidden=(self.aux_coeff > 0),
            )
            old_lps = [lp.detach() for lp in policy_lps]
            metrics = self._empty_step_metrics()
            group_losses = []
            for g in range(n_prompts):
                s, e = g * self.G, (g + 1) * self.G
                group_rn = r_norm_all[s:e]
                beta_dynamic = (
                    1.0 - 0.5 * group_rn
                    if self.use_dynamic_kl else torch.ones_like(group_rn)
                )
                grp_r = torch.tensor(
                    rewards[s:e], dtype=torch.float,
                    device=self.accelerator.device,
                )
                adv = self.normalize_advantages(grp_r)
                p_lps, completion_mask = self._pad_group_for_loss(
                    policy_lps[s:e], gen["completion_masks"][s:e],
                )
                o_lps, _ = self._pad_group_for_loss(
                    old_lps[s:e], gen["completion_masks"][s:e],
                )
                r_lps, _ = self._pad_group_for_loss(
                    ref_lps[s:e], gen["completion_masks"][s:e],
                )
                grp_quality = [float(value) for value in quality_scores[s:e]]
                loss, group_metrics = self.grpo_loss(
                    p_lps, o_lps, adv, r_lps,
                    beta_dynamic=beta_dynamic,
                    completion_mask=completion_mask,
                    kl_invalid=(
                        not grp_quality
                        or max(grp_quality) - min(grp_quality) <= 1e-8
                    ),
                )
                group_losses.append(loss)
                for key, value in group_metrics.items():
                    metrics[key] = metrics.get(key, 0.0) + value / n_prompts
                metrics["reward/quality_mean"] += sum(quality_scores[s:e]) / self.G / n_prompts
                metrics["reward/format_mean"] += sum(format_scores[s:e]) / self.G / n_prompts
                metrics["reward/format_contribution_mean"] += self.reward_pipeline.format_weight * sum(format_scores[s:e]) / self.G / n_prompts
                metrics["reward/base_mean"] += sum(base_rewards[s:e]) / self.G / n_prompts
                metrics["reward/compression_mean"] += comp_rewards[s:e].mean().item() / n_prompts
                metrics["reward/mean"] += grp_r.mean().item() / n_prompts
                metrics["reward/std"] += grp_r.std().item() / n_prompts
                metrics["adv/mean"] += adv.mean().item() / n_prompts
                metrics["adv/std"] += adv.std().item() / n_prompts

            aux_loss_val = None
            n_r_samples = 0
            if self.aux_coeff > 0:
                aux_loss_val, n_r_samples = self._compute_aux_loss(
                    policy_hidden, gen["input_ids"], teacher_scores_list,
                    seq_starts_list=gen["seq_starts"],
                    completion_masks=gen["completion_masks"],
                )
            metrics["aux/r_token_samples"] = n_r_samples
            metrics["loss/aux"] = (
                aux_loss_val.item() if aux_loss_val is not None else 0.0
            )
            total_loss = self.combine_objectives(
                group_losses, aux_loss_val, self.aux_coeff,
            )
            metrics["loss/total"] += (
                self.aux_coeff * aux_loss_val.item()
                if aux_loss_val is not None else 0.0
            )
            self.accelerator.backward(total_loss)

        if self.accelerator.is_main_process:
            print(f"  [step {self.global_step}] log-probs done", flush=True)

        if FSDP is not None and isinstance(self.model, FSDP):
            grad_norm = self.model.clip_grad_norm_(self.max_grad_norm)
        else:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
        metrics["grad_norm"] = (
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        )
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        metrics["optim/lr"] = float(optimizer.param_groups[0]["lr"])
        metrics["optim/scheduler_step"] = int(
            lr_scheduler.state_dict().get("last_epoch", -1)
        )
        if self.accelerator.is_main_process:
            print(
                f"  [step {self.global_step}] backward+step done "
                f"lr={metrics['optim/lr']:.3e} "
                f"scheduler_step={metrics['optim/scheduler_step']}",
                flush=True,
            )

        # --- 9. Curriculum update ---
        # Feed the scheduler ONLY the QUALITY reward (format/length bonuses must
        # not inflate the competence estimate), and attribute each group to ITS
        # OWN bucket. Rewards are accumulated and applied in _sync_curriculum
        # (after the step), so all ranks contribute to one shared scheduler.
        if bucket_ids is not None and len(bucket_ids) > 0:
            for g in range(n_prompts):
                s, e = g * self.G, (g + 1) * self.G
                avg_quality = sum(quality_scores[s:e]) / self.G
                g_bid = int(bucket_ids[g]) if g < len(bucket_ids) else int(bucket_ids[0])
                baseline = self.baseline_by_type.get(prompt_cat[g], 0.5)
                self._pending_curriculum.append(
                    (g_bid, avg_quality, baseline, prompt_cat[g])
                )

        # --- 10. Metrics ---
        metrics["loss/r_norm_mean"] = r_norm_all.mean().item()
        # Per-task-type quality breakdown (helps spot constant/dead rewards).
        cat_scores: Dict[str, List[float]] = {}
        for i in range(len(quality_scores)):
            s = str(meta[i]["subset"]).lower()
            t = str(meta[i]["type"]).lower()
            cat = "s2tt" if s == "s2tt" else ("choice" if t == "choice" else "open")
            cat_scores.setdefault(cat, []).append(quality_scores[i])
        for cat, vals in cat_scores.items():
            metrics[f"reward/{cat}"] = sum(vals) / len(vals)

        # Per-dimension diagnostic: group samples by the DOMINANT difficulty axis
        # (argmax D_P/D_C/D_R) and report avg quality + group-dead rate per axis.
        # Shows which cognitive dimension the policy fails on (改进2 Phase 1).
        _dp = batch.get("D_P") or [0.5] * n_prompts
        _dc = batch.get("D_C") or [0.5] * n_prompts
        _dr = batch.get("D_R") or [0.5] * n_prompts
        dim_accum: Dict[str, List[float]] = {}
        dim_dead: Dict[str, List[int]] = {}
        for g in range(n_prompts):
            dim = ("P" if _dp[g] >= _dc[g] and _dp[g] >= _dr[g]
                   else ("C" if _dc[g] >= _dr[g] else "R"))
            s, e = g * self.G, (g + 1) * self.G
            gv = quality_scores[s:e]
            dim_accum.setdefault(dim, []).extend(gv)
            dim_dead.setdefault(dim, []).append(
                1 if (max(gv) - min(gv)) < 1e-8 else 0)
        for dim in ["P", "C", "R"]:
            qs, ds = dim_accum.get(dim, []), dim_dead.get(dim, [])
            if qs:
                metrics[f"dim/{dim}/quality"] = sum(qs) / len(qs)
                metrics[f"dim/{dim}/n"] = len(qs)
            if ds:
                metrics[f"dim/{dim}/dead"] = sum(ds) / len(ds)

        # Compression diagnostics: separate over- and under-budget events.
        # A negative reward can come from either side, so `(reward < 0)` is not
        # an over-budget indicator.
        if self.comp_mode in {"section_wise", "sft_trace_interval"}:
            over_flags = [float(value) for value in section_over_flags]
            under_flags = [float(value) for value in section_under_flags]
        else:
            over_flags = []
            under_flags = []
            for i, length in enumerate(resp_lengths):
                _, cap = self._compression_bin(
                    r_norm_all[i].item(), comp_families[i],
                )
                over_flags.append(1.0 if cap > 0 and length > cap else 0.0)
                under_flags.append(
                    1.0
                    if cap > 0 and length < self.comp_min_frac * cap else 0.0
                )
        metrics["comp/over_ratio"] = sum(over_flags) / max(len(over_flags), 1)
        metrics["comp/under_ratio"] = (
            sum(under_flags) / max(len(under_flags), 1)
        )
        metrics["comp/len_mean"] = float(sum(resp_lengths) / max(1, len(resp_lengths)))
        metrics["comp/high_norm_frac"] = (
            (r_norm_all > 0.85).float().mean().item()
        )
        metrics["comp/section_mode"] = float(
            self.comp_mode in {"section_wise", "sft_trace_interval"}
        )
        metrics["comp/trace_interval_mode"] = float(
            self.comp_mode == "sft_trace_interval"
        )
        if (
            self.comp_mode in {"section_wise", "sft_trace_interval"}
            and section_lengths_all
        ):
            metrics["comp/section_invalid_ratio"] = (
                1.0 - sum(float(x) for x in section_validity)
                / len(section_validity)
            )
            metrics["comp/section_valid_ratio"] = (
                sum(float(x) for x in section_validity)
                / len(section_validity)
            )
            sections_to_log = list(RATING_SECTIONS)
            if all("SUMMARY" in lengths for lengths in section_lengths_all):
                sections_to_log.append("SUMMARY")
            for section in sections_to_log:
                values = [lengths[section] for lengths in section_lengths_all]
                metrics[f"comp/{section.lower()}_len_mean"] = (
                    sum(values) / len(values)
                )
            if trace_lengths_all:
                for section in RATING_SECTIONS:
                    values = [lengths[section] for lengths in trace_lengths_all]
                    metrics[f"comp/{section.lower()}_trace_len_mean"] = (
                        sum(values) / len(values)
                    )
        len_by_level: Dict[int, List[int]] = {}
        for i in range(len(resp_lengths)):
            bin_idx, _ = self._compression_bin(
                r_norm_all[i].item(), comp_families[i],
            )
            lv = bin_idx + 1
            len_by_level.setdefault(lv, []).append(resp_lengths[i])
        for lv in range(1, 6):
            vals = len_by_level.get(lv, [])
            if vals:
                metrics[f"comp/len_r{lv}"] = sum(vals) / len(vals)

        # Diagnostics: keep a couple of (response, quality) pairs for main-rank
        # logging in train(), so low reward can be diagnosed (extraction bug vs
        # genuinely weak model). Also extract the RESPONSE-section content the
        # reward actually compares, so the log shows exactly what BLEU sees.
        try:
            from rewards.quality_reward import _extract_final_response
        except Exception:
            _extract_final_response = lambda t: t
        self._last_samples = []
        for i in range(min(2, len(gen["responses"]))):
            self._last_samples.append((
                gen["responses"][i], quality_scores[i],
                _extract_final_response(gen["responses"][i]),
                _extract_final_response(gt_repeated[i]),
                meta[i].get("type", ""),
            ))

        # Generation stats: response lengths, tag coverage, R-token usage.
        resp_lens = [
            int(mask.sum().item()) for mask in gen["completion_masks"]
        ]
        metrics["gen/resp_len_mean"] = float(sum(resp_lens)) / max(len(resp_lens), 1)
        has_resp = [
            1.0 if ("<RESPONSE>" in r or "</RESPONSE>" in r) else 0.0
            for r in gen["responses"]
        ]
        metrics["gen/pct_has_response_tag"] = (
            sum(has_resp) / max(len(has_resp), 1)
        )
        r_token_frac = [
            1.0 if re.search(r"<R([1-5])>", r) else 0.0
            for r in gen["responses"]
        ]
        metrics["gen/r_token_frac"] = (
            sum(r_token_frac) / max(len(r_token_frac), 1)
        )
        r_triplet_frac = [
            1.0 if self._extract_section_ratings(r)[1] == 3 else 0.0
            for r in gen["responses"]
        ]
        metrics["gen/r_triplet_frac"] = (
            sum(r_triplet_frac) / max(len(r_triplet_frac), 1)
        )

        # Cleanup
        for v in gen.values():
            del v
        if not self.streaming_backward:
            del old_lps, ref_lps, policy_lps, group_losses
        gc.collect()
        torch.cuda.empty_cache()

        metrics["step_time_ms"] = (time.time() - step_start) * 1000
        metrics["step"] = self.global_step
        self._metrics_history.append(metrics)
        return metrics

    # ======================================================================
    # Training loop
    # ======================================================================

    def train(
        self, sample_batch, optimizer, lr_scheduler,
        max_steps: int, per_device_bs: int = 4,
        log_steps: int = 10, save_steps: int = 200,
        output_dir: str = "./output", resume: Optional[str] = None,
    ):
        """GRPO training loop.

        Args:
            sample_batch: callable(per_device_bs) -> collated batch dict. Samples
                a fresh batch from the CURRENT curriculum weights every step, so
                bucket merges actually change what the model trains on.
        """
        is_main = self.accelerator.is_main_process

        if resume:
            self._load_checkpoint(resume, optimizer, lr_scheduler)

        if is_main:
            print(f"\n{'='*60}")
            print("Curriculum GRPO Training")
            print(f"  Steps: {max_steps} | G={self.G} | beta={self.beta}")
            print(
                f"  Task A: beta={self.beta}, dynamic={self.use_dynamic_kl}, "
                f"reset_ref={self.reset_reference_on_merge}"
            )
            print(
                f"  Task B: compression over={self.comp_coeff}, "
                f"under={self.comp_under_coeff}, min_frac={self.comp_min_frac}"
            )
            print(f"  Task C: aux_coeff={self.aux_coeff}, soft_var={self.soft_var}")
            print(f"  K={self.scheduler.K} | Device: {self.accelerator.device} | World: {self.accelerator.num_processes}")
            print(f"{'='*60}\n")

        step = self.global_step

        while step < max_steps:
            if step == 0:
                if is_main:
                    print(f"\n[Curriculum] Active buckets: {self.scheduler.active_buckets}")

            # Sample a fresh batch per step from the live curriculum weights.
            batch = sample_batch(per_device_bs)

            try:
                metrics = self.train_step(batch, optimizer, lr_scheduler)
            except Exception as e:
                import traceback
                print(f"\n<<<RANK{self.accelerator.process_index} TRAIN_STEP_ERROR: {type(e).__name__}: {e}>>>", flush=True)
                traceback.print_exc()
                raise

            # Apply curriculum rewards from ALL ranks on the main scheduler,
            # then broadcast the updated state so every rank samples identically.
            self._sync_curriculum()
            self.accelerator.wait_for_everyone()

            step = self.global_step

            if step % log_steps == 0 and is_main:
                prog = self.scheduler.get_progress()
                metric_record = {
                    "schema_version": "rl_training_metrics.v1",
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "step": int(step),
                    "max_steps": int(max_steps),
                    "active_buckets": list(prog["active_buckets"]),
                    "curriculum": prog,
                    "metrics": {
                        key: (
                            value.item()
                            if isinstance(value, (torch.Tensor, np.generic))
                            else value
                        )
                        for key, value in metrics.items()
                    },
                }
                metrics_path = os.path.join(output_dir, "metrics.jsonl")
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(metric_record, ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )
                    handle.flush()
                print(
                    f"Step {step}/{max_steps} | "
                    f"B={prog['active_buckets']} | "
                    f"cs={prog['competence']:.3f} | "
                    f"front={prog.get('frontier_prompts', 0)}/{prog.get('min_frontier_prompts', 0)} | "
                    f"streak={prog.get('mastery_streak', 0)}/{prog.get('mastery_needed', 1)} | "
                    f"qErr={prog.get('quota_max_abs_error', 0):.2f} | "
                    f"Loss={metrics['loss/total']:.4f} | "
                    f"R={metrics['reward/mean']:.3f} | "
                    f"q={metrics['reward/quality_mean']:.3f} | "
                    f"Rb={metrics['reward/base_mean']:.3f} | "
                    f"Rf={metrics['reward/format_contribution_mean']:.3f} | "
                    f"Rc={metrics['reward/compression_mean']:.3f} | "
                    f"Rstd={metrics['reward/std']:.3f} | "
                    f"qSC={metrics.get('reward/s2tt', 0):.2f}/{metrics.get('reward/choice', 0):.2f} | "
                    f"mSC={prog.get('last_family_means', {}).get('s2tt', 0):.2f}/{prog.get('last_family_means', {}).get('choice', 0):.2f} | "
                    f"f={metrics['reward/format_mean']:.2f} | "
                    f"rN={metrics['loss/r_norm_mean']:.3f} | "
                    f"aux={metrics['loss/aux']:.4f} | "
                    f"auxN={metrics.get('aux/r_token_samples', 0)} | "
                    f"lr={metrics.get('optim/lr', 0.0):.3e} | "
                    f"sched={metrics.get('optim/scheduler_step', -1)} | "
                    f"len={metrics['gen/resp_len_mean']:.0f} | "
                    f"rTrip={metrics.get('gen/r_triplet_frac', 0):.2f} | "
                    f"GN={metrics['grad_norm']:.2f} | "
                    f"dead={metrics.get('loss/kl_skipped', 0):.2f} | "
                    f"advStd={metrics.get('adv/std', 0):.3f} | "
                    f"cLen={metrics.get('comp/len_mean', 0):.0f} | "
                    f"cOver={metrics.get('comp/over_ratio', 0):.2f} | "
                    f"cUnder={metrics.get('comp/under_ratio', 0):.2f} | "
                    f"highNorm={metrics.get('comp/high_norm_frac', 0):.2f} | "
                    f"lenByBin={metrics.get('comp/len_r1', 0):.0f}/{metrics.get('comp/len_r3', 0):.0f}/{metrics.get('comp/len_r5', 0):.0f} | "
                    f"dim(P/C/R)={metrics.get('dim/P/quality', 0):.2f}/{metrics.get('dim/C/quality', 0):.2f}/{metrics.get('dim/R/quality', 0):.2f} | "
                    f"{metrics['step_time_ms']:.0f}ms"
                )
                for s_idx, (resp, q, resp_ext, gt_ext, typ) in enumerate(getattr(self, "_last_samples", [])):
                    print(f"  [sample{s_idx}] type={typ} q={q:.3f}")
                    print(f"    resp[:600]: {resp[:600]!r}")
                    print(f"    resp<RESPONSE>: {resp_ext[:200]!r}")
                    print(f"    gt<RESPONSE>  : {gt_ext[:200]!r}")

            # The terminal model is written once as step_final below.  Do not
            # duplicate it as step_<max_steps>; with save_steps=1000 and a
            # 2000-step run this yields exactly step_1000 + step_final.
            if step < max_steps and step % save_steps == 0:
                self._save_checkpoint(output_dir, step, optimizer, lr_scheduler)

        self._save_checkpoint(output_dir, "final", optimizer, lr_scheduler)
        if is_main:
            print(f"\nTraining complete. Final curriculum: {self.scheduler.get_progress()}")

    def _sync_curriculum(self):
        """Gather every rank's (bucket_id, avg_quality) groups, apply them to
        the MAIN rank's scheduler, then broadcast the updated state so all ranks
        share the same active buckets / weights / competence.

        Adaptive reference: after a bucket MERGE, the reference
        model is reset to the CURRENT policy — so the KL no longer drags the
        improved policy back to the original SFT reference.

        Uses torch.distributed directly (not accelerator.gather_object /
        broadcast_object) so it works on older accelerate versions in the sif.
        """
        import torch.distributed as dist

        prev_merges = self.scheduler.n_merges
        pending = self._pending_curriculum
        self._pending_curriculum = []
        world = self.accelerator.num_processes

        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, pending)
        else:
            gathered = [pending]

        if self.accelerator.is_main_process:
            for item in [entry for rank_items in gathered for entry in rank_items]:
                # Four fields are the redesigned contract.  Accept historical
                # three-field pending entries for legacy checkpoints/tests.
                if len(item) == 4:
                    g_bid, avg_q, baseline, family = item
                    families = [family]
                else:
                    g_bid, avg_q, baseline = item
                    families = None
                self.scheduler.add_rewards(
                    [avg_q], g_bid, baselines=[baseline], families=families,
                )
            new_state = self.scheduler.state_dict()
        else:
            new_state = None

        if world > 1:
            obj_list = [new_state]
            dist.broadcast_object_list(obj_list, src=0)
            new_state = obj_list[0]

        if new_state is not None:
            self.scheduler.load_state_dict(new_state)
            if (
                self.reset_reference_on_merge
                and self.scheduler.n_merges > prev_merges
            ):
                self._reset_reference()

    def _reset_reference(self):
        """Copy the policy's CURRENT weights (base + trained LoRA) into the
        reference model. Both are PeftModels over the same base, so the state
        dicts match; only the adapter weights actually change."""
        if self.accelerator.is_main_process:
            print(f"  [Curriculum] *** reset reference to current policy (merge #{self.scheduler.n_merges}) ***", flush=True)
        if FSDP is not None and isinstance(self.model, FSDP):
            if not isinstance(self.ref_model, FSDP):
                raise RuntimeError("FSDP policy/reference wrapping mismatch")
            policy_parameters = list(self.model.named_parameters())
            reference_parameters = list(self.ref_model.named_parameters())
            if len(policy_parameters) != len(reference_parameters):
                raise RuntimeError("FSDP policy/reference parameter counts differ")
            with torch.no_grad():
                for (policy_name, policy_param), (ref_name, ref_param) in zip(
                    policy_parameters, reference_parameters,
                ):
                    if policy_name != ref_name or policy_param.shape != ref_param.shape:
                        raise RuntimeError(
                            "FSDP policy/reference shard layout differs: "
                            f"{policy_name}{tuple(policy_param.shape)} != "
                            f"{ref_name}{tuple(ref_param.shape)}"
                        )
                    ref_param.copy_(policy_param)
                policy_buffers = list(self.model.named_buffers())
                reference_buffers = list(self.ref_model.named_buffers())
                if len(policy_buffers) != len(reference_buffers):
                    raise RuntimeError("FSDP policy/reference buffer counts differ")
                for (policy_name, policy_buffer), (ref_name, ref_buffer) in zip(
                    policy_buffers, reference_buffers,
                ):
                    if policy_name != ref_name or policy_buffer.shape != ref_buffer.shape:
                        raise RuntimeError("FSDP policy/reference buffer layout differs")
                    ref_buffer.copy_(policy_buffer)
        else:
            policy = self.accelerator.unwrap_model(self.model)
            # A failed adaptive-reference reset changes the objective, so it is
            # a fatal training error rather than a warning.
            self.ref_model.load_state_dict(policy.state_dict())
        for p in self.ref_model.parameters():
            p.requires_grad = False

    # ======================================================================
    # Checkpoint
    # ======================================================================

    def _save_checkpoint(self, output_dir: str, tag, optimizer, lr_scheduler):
        ckpt_dir = os.path.join(output_dir, "checkpoints", f"step_{tag}")
        is_fsdp = FSDP is not None and isinstance(self.model, FSDP)
        if self.accelerator.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        if is_fsdp:
            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, full_cfg,
            ):
                full_state = self.model.state_dict()
            if self.accelerator.is_main_process:
                self.model.module.save_pretrained(
                    ckpt_dir,
                    state_dict=full_state,
                    safe_serialization=True,
                )
                self.tokenizer.save_pretrained(ckpt_dir)
            del full_state

            # Same-topology resume payload. Each file contains that rank's
            # original-parameter shards and sharded AdamW state; unlike the HF
            # export above, this preserves optimizer continuity.
            distributed_dir = os.path.join(ckpt_dir, "distributed")
            if self.accelerator.is_main_process:
                os.makedirs(distributed_dir, exist_ok=True)
            self.accelerator.wait_for_everyone()
            rank = self.accelerator.process_index
            torch.save(
                {
                    "model_local": {
                        name: parameter.detach().cpu()
                        for name, parameter in self.model.named_parameters()
                    },
                    "reference_local": {
                        name: parameter.detach().cpu()
                        for name, parameter in self.ref_model.named_parameters()
                    },
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "global_step": self.global_step,
                    "sampling_state": (
                        self.sampling_state_getter()
                        if self.sampling_state_getter is not None else None
                    ),
                },
                os.path.join(distributed_dir, f"rank_{rank:02d}.pt"),
            )
        elif self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).save_pretrained(
                ckpt_dir, safe_serialization=True
            )
            self.tokenizer.save_pretrained(ckpt_dir)
            torch.save(
                optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt")
            )
            torch.save(
                lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt")
            )

        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
        if os.path.isfile(adapter_path):
            digest = hashlib.sha256()
            with open(adapter_path, "rb") as adapter_file:
                for chunk in iter(lambda: adapter_file.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            adapter_sha256 = digest.hexdigest()
        else:
            adapter_sha256 = "missing"
        scheduler_step = lr_scheduler.state_dict().get("last_epoch", -1)
        current_lr = float(optimizer.param_groups[0]["lr"])

        import json
        with open(os.path.join(ckpt_dir, "curriculum.json"), "w") as f:
            json.dump(self.scheduler.state_dict(), f)

        print(
            f"  [Save] {ckpt_dir} | "
            f"lr={current_lr:.3e} scheduler_step={scheduler_step} | "
            f"adapter_sha256={adapter_sha256}",
            flush=True,
        )
        self.accelerator.wait_for_everyone()

    def _load_checkpoint(self, checkpoint_dir, optimizer, lr_scheduler):
        """Resume the strict FSDP8 topology from per-rank shard payloads."""
        if FSDP is None or not isinstance(self.model, FSDP):
            raise NotImplementedError(
                "resume_from_checkpoint is implemented for FSDP8 full-RL only"
            )
        rank = self.accelerator.process_index
        shard_path = os.path.join(
            checkpoint_dir, "distributed", f"rank_{rank:02d}.pt",
        )
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"missing FSDP resume shard: {shard_path}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)

        def restore_local(named_parameters, saved, label):
            current = dict(named_parameters)
            if set(current) != set(saved):
                raise RuntimeError(f"{label} resume parameter names differ")
            with torch.no_grad():
                for name, parameter in current.items():
                    tensor = saved[name]
                    if parameter.shape != tensor.shape:
                        raise RuntimeError(
                            f"{label} resume shard shape differs for {name}: "
                            f"{tuple(parameter.shape)} != {tuple(tensor.shape)}"
                        )
                    parameter.copy_(tensor.to(parameter.device))

        restore_local(
            self.model.named_parameters(), payload["model_local"], "policy",
        )
        restore_local(
            self.ref_model.named_parameters(),
            payload["reference_local"],
            "reference",
        )
        optimizer.load_state_dict(payload["optimizer"])
        lr_scheduler.load_state_dict(payload["scheduler"])
        self.global_step = int(payload["global_step"])
        sampling_state = payload.get("sampling_state")
        if sampling_state is not None:
            if self.sampling_state_loader is None:
                raise RuntimeError(
                    "checkpoint contains sampling state but no loader is configured"
                )
            self.sampling_state_loader(sampling_state)
        curriculum_path = os.path.join(checkpoint_dir, "curriculum.json")
        with open(curriculum_path, encoding="utf-8") as handle:
            self.scheduler.load_state_dict(json.load(handle))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print(
                f"[Resume] restored strict FSDP8 checkpoint at step "
                f"{self.global_step}: {checkpoint_dir}",
                flush=True,
            )
