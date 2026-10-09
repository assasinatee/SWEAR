#!/usr/bin/env python3
"""Curriculum GRPO training initialized from LoRA or full SFT checkpoints."""

import argparse, hashlib, json, os, sys, random, yaml, gc, re
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    set_seed,
)
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers import Qwen2_5OmniProcessor
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from peft import PeftModel, LoraConfig

from model_checkpoint import resolve_sft_checkpoint, zero_checkpoint_tag

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision as FSDPMixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
except ImportError:  # very old torch images
    FSDP = None
    FSDPMixedPrecision = None
    ShardingStrategy = None
    transformer_auto_wrap_policy = None

# 8-bit AdamW is required for FULL-param RL (7B bf16 + vanilla fp32 AdamW states
# won't fit in 80GB per GPU). LoRA runs do not require bitsandbytes, but a full
# run must fail before allocating optimizer state if the image lacks it.
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except Exception:  # ImportError or missing .so on host
    bnb = None
    HAS_BNB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _make_run_manifest(
    *,
    config_path: Path,
    curriculum_file: Path,
    curriculum_sha256: str,
    sft_checkpoint: Path,
    sft_format: str,
    model_path: str,
    seed: int,
    world_size: int,
    per_device_batch: int,
    group_size: int,
    max_steps: int,
    global_prompt_batch: int | None = None,
    micro_slots_per_rank: int | None = None,
    distributed_strategy: str = "ddp",
) -> dict:
    """Collect non-secret, immutable provenance for one formal run."""

    code_relpaths = (
        "train_grpo.py",
        "model_checkpoint.py",
        "trainer/grpo_trainer.py",
        "trainer/curriculum_scheduler.py",
        "data/curriculum_dataset.py",
        "data/choice_permutation.py",
        "rewards/quality_reward.py",
        "rewards/format_reward.py",
        "scripts/build_bplus_curriculum.py",
    )
    code_sha256 = {}
    for relative in code_relpaths:
        code_path = Path(BASE_DIR) / relative
        if code_path.is_file():
            code_sha256[relative] = _sha256_file(code_path)

    curriculum_manifest_path = curriculum_file.with_name("manifest.json")
    curriculum_manifest = None
    if curriculum_manifest_path.is_file():
        curriculum_manifest = json.loads(
            curriculum_manifest_path.read_text(encoding="utf-8")
        )

    adapter_config_path = sft_checkpoint / "adapter_config.json"
    adapter_config = None
    if adapter_config_path.is_file():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))

    return {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_tag": os.environ.get("RL_JOB_TAG"),
        "training_sif": os.environ.get("TRAIN_SIF"),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256_file(config_path),
        },
        "code_sha256": code_sha256,
        "model": {
            "base": str(model_path),
            "sft_checkpoint": str(sft_checkpoint),
            "sft_checkpoint_format": str(sft_format),
            "sft_adapter_config": adapter_config,
        },
        "curriculum": {
            "path": str(curriculum_file),
            "size_bytes": curriculum_file.stat().st_size,
            "observed_sha256": str(curriculum_sha256),
            "manifest_path": (
                str(curriculum_manifest_path)
                if curriculum_manifest_path.is_file() else None
            ),
            "manifest": curriculum_manifest,
        },
        "runtime_contract": {
            "seed": int(seed),
            "world_size": int(world_size),
            "per_device_prompt_batch": int(per_device_batch),
            "global_prompt_batch": int(
                global_prompt_batch
                if global_prompt_batch is not None
                else world_size * per_device_batch
            ),
            "completions_per_prompt": int(group_size),
            "global_completions_per_step": int(
                (
                    global_prompt_batch
                    if global_prompt_batch is not None
                    else world_size * per_device_batch
                ) * group_size
            ),
            "micro_slots_per_rank": int(
                micro_slots_per_rank
                if micro_slots_per_rank is not None
                else per_device_batch
            ),
            "distributed_strategy": str(distributed_strategy),
            "max_steps": int(max_steps),
        },
    }


def _fsdp_auto_wrap_policy(model):
    """Wrap Qwen transformer blocks without hard-coding one transformers version."""
    layer_classes = {
        type(module)
        for module in model.modules()
        if type(module).__name__.endswith("DecoderLayer")
    }
    if not layer_classes:
        raise RuntimeError(
            "FSDP could not find any *DecoderLayer modules in the Qwen thinker"
        )
    print(
        "[FSDP] transformer layer classes: "
        + ", ".join(sorted(cls.__name__ for cls in layer_classes)),
        flush=True,
    )
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_classes,
    )


def _wrap_full_model_fsdp(model, device, auto_wrap_policy):
    if FSDP is None:
        raise RuntimeError("This torch build does not provide FSDP")
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=FSDPMixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    return parser.parse_args()


def _load_full_sft_model(sft_ckpt: str, checkpoint_format: str):
    """Load either an ordinary HF full model or raw DeepSpeed ZeRO-3 shards."""

    if checkpoint_format == "hf_full":
        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            sft_ckpt,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=True,
        )

    if checkpoint_format != "zero_full":
        raise ValueError(f"Not a full SFT checkpoint: {checkpoint_format}")

    try:
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Loading the raw full-SFT ZeRO checkpoint requires DeepSpeed in "
            "the training SIF. Alternatively merge it to Hugging Face format "
            "first with SFT/llama_factory/scripts/submit_merge_zero.sh."
        ) from exc

    tag = zero_checkpoint_tag(sft_ckpt)
    print(f"[Model] Reconstructing DeepSpeed ZeRO checkpoint tag={tag}...", flush=True)
    state_dict = get_fp32_state_dict_from_zero_checkpoint(sft_ckpt, tag=tag)
    config = AutoConfig.from_pretrained(
        sft_ckpt, trust_remote_code=True, local_files_only=True,
    )
    # Build the destination directly in bf16. Meta initialization is tempting,
    # but Qwen has non-persistent buffers that are intentionally absent from a
    # state_dict and would consequently remain on meta after assign=True.
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen2_5OmniThinkerForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(previous_default_dtype)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "ZeRO checkpoint does not match its Qwen2.5-Omni config: "
            f"missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    model = model.to(dtype=torch.bfloat16)
    del state_dict
    gc.collect()
    return model


def _expand_env(obj):
    """Recursively expand ${VAR} in config strings (falls back to "")."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = _expand_env(cfg)
    rl_cfg = cfg.pop("rl", {})
    return cfg, rl_cfg


def build_or_load_curriculum(rl_cfg):
    paths = rl_cfg.get("paths", {})
    curriculum_data = paths.get(
        "curriculum_data", os.path.join(BASE_DIR, "data", "curriculum"),
    )
    # Accept a curriculum directory or a direct .jsonl file path.
    if str(curriculum_data).endswith(".jsonl"):
        cf = curriculum_data
        curriculum_dir = os.path.dirname(cf)
        explicit_file = True
    else:
        curriculum_dir = curriculum_data
        cf = os.path.join(curriculum_dir, "curriculum_data.jsonl")
        explicit_file = False
    if not os.path.exists(cf):
        if explicit_file:
            raise FileNotFoundError(f"Curriculum file not found: {cf}")
        from data.build_curriculum import build_curriculum
        compress_root = paths.get(
            "compress_root",
            os.environ.get("COMPRESS_ROOT", ""),
        )
        curriculum = rl_cfg.get("curriculum", {})
        print(f"[Data] Building curriculum...")
        build_curriculum(
            compress_root=compress_root, output_dir=curriculum_dir,
            K=curriculum.get("K", 4),
            filter_low=curriculum.get("filter_low_ratio", 0.05),
            filter_high=curriculum.get("filter_high_ratio", 0.95),
        )
    return cf


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        vals = [item[k] for item in batch]
        if k == "difficulty_norm" or k == "rd_norm":
            out[k] = torch.tensor(vals, dtype=torch.float)
        elif k == "bucket_id":
            out[k] = torch.tensor(vals, dtype=torch.long)
        elif k == "audio_waveform":
            out[k] = vals
        else:
            out[k] = vals
    return out


def task_family(item: dict) -> str:
    """Match the reward router's precedence exactly."""

    if str(item.get("subset", "")).casefold() == "s2tt":
        return "s2tt"
    if str(item.get("type", "")).casefold() == "choice":
        return "choice"
    return "open"


def bplus_stratum(item: dict) -> str:
    """Return the stored B+ stratum, with a deterministic fallback."""

    stored = item.get("bplus_stratum")
    if stored not in (None, ""):
        return str(stored)
    family = task_family(item)
    if family == "s2tt":
        return "s2tt"
    return (
        f"{family}|subset={item.get('subset', '<missing>')}|"
        f"type={item.get('type', '<missing>')}"
    )


def _sanitize_dist_env():
    """Remove empty-string dist env vars that break Accelerate's int parsing.

    clusterx PYTORCH_DDP framework may set e.g. WORLD_SIZE/PMI_SIZE to ''
    which makes int(os.environ.get(...,-1)) raise ValueError in accelerate.
    """
    for e in ["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE",
              "WORLD_SIZE", "RANK", "LOCAL_RANK"]:
        v = os.environ.get(e)
        if v is not None and str(v).strip() == "":
            print(f"[Dist] Removing empty env {e}")
            del os.environ[e]


def main():
    import faulthandler
    faulthandler.enable()  # dump Python traceback on SIGABRT/SIGSEGV (native crash debug)
    _sanitize_dist_env()
    args = parse_args()
    llamacfg, rl_cfg = load_config(args.config)
    print("[Config] Loaded")
    if os.environ.get("RL_MAX_STEPS_OVERRIDE"):
        llamacfg["max_steps"] = int(os.environ["RL_MAX_STEPS_OVERRIDE"])
        print(f"[Config] max_steps override={llamacfg['max_steps']}", flush=True)
    if os.environ.get("RL_OUTPUT_DIR_OVERRIDE"):
        llamacfg["output_dir"] = os.environ["RL_OUTPUT_DIR_OVERRIDE"]
        print(f"[Config] output_dir override={llamacfg['output_dir']}", flush=True)
    if os.environ.get("RL_RESUME_OVERRIDE"):
        llamacfg["resume_from_checkpoint"] = os.environ["RL_RESUME_OVERRIDE"]
        print(
            f"[Config] resume override={llamacfg['resume_from_checkpoint']}",
            flush=True,
        )
    distributed_cfg = llamacfg.get("distributed", {}) or {}
    distributed_strategy = str(
        distributed_cfg.get("strategy", "ddp")
    ).strip().lower()
    use_fsdp = distributed_strategy == "fsdp"
    if distributed_strategy not in {"ddp", "fsdp"}:
        raise ValueError(
            f"unsupported distributed.strategy={distributed_strategy!r}"
        )
    seed = int(llamacfg.get("seed", 42))
    # Establish the base RNG state before any optional fresh adapter/model
    # initialization. A device-specific rank offset is applied after Accelerate
    # knows the distributed process index.
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"[Seed] base={seed}")

    curriculum_cfg = rl_cfg.get("curriculum", {})
    grpo_cfg = rl_cfg.get("grpo", {})

    # Data
    curriculum_file = build_or_load_curriculum(rl_cfg)
    print(f"[Data] {curriculum_file}")

    # `model_name_or_path` is the base model (tokenizer/processor source);
    # `sft_checkpoint` may be a LoRA adapter, an HF full model, or ZeRO-3
    # shards. Full-model SFT starts a fresh LoRA when lora_rank > 0.
    model_path = llamacfg.get(
        "model_name_or_path",
        os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Omni-7B"))
    requested_sft_ckpt = os.environ.get("SFT_CKPT") or llamacfg.get("sft_checkpoint")
    if not requested_sft_ckpt:
        raise ValueError(
            "An explicit SFT checkpoint is required: set sft_checkpoint in "
            "the YAML or export SFT_CKPT."
        )
    checkpoint = resolve_sft_checkpoint(requested_sft_ckpt)
    sft_ckpt = str(checkpoint.path)
    sft_format = checkpoint.format
    sft_is_lora = sft_format == "lora"
    # RL optimizer mode: "lora" (default, lightweight) | "full" (all params, needs
    # 8-bit AdamW or ZeRO). When sft_ckpt is itself a full model AND lora_rank==0, "full" is
    # implied; when sft_ckpt is full and lora_rank>0, "lora-on-full".
    sft_tok_src = sft_ckpt
    if not sft_is_lora and not os.path.exists(os.path.join(sft_ckpt, "tokenizer.json")):
        # Merged full ckpt produced by merge_zero_shards.py doesn't always carry
        # tokenizer files (we copy them, but be defensive) — fall back to base.
        sft_tok_src = model_path
    print(f"[Model] Base: {model_path}")
    if str(requested_sft_ckpt) != sft_ckpt:
        print(f"[Model] Resolved SFT container: {requested_sft_ckpt} -> {sft_ckpt}")
    print(f"[Model] SFT ckpt: {sft_ckpt}")
    print(f"[Model] SFT format: {sft_format}")

    tokenizer = AutoTokenizer.from_pretrained(
        sft_tok_src, trust_remote_code=True, local_files_only=True)

    # Load processor for audio + text processing (always from base — the SFT dirs
    # don't ship a full Qwen2.5-Omni processor, only preprocessor_config.json).
    print(f"[Model] Loading processor from {model_path}...")
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)

    # ===== Load the model skeleton =====
    if sft_is_lora:
        # Legacy: base = Qwen2.5-Omni base, attach the SFT LoRA as the trainable adapter.
        print(f"[Model] Loading thinker BASE model on CPU...")
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="cpu", local_files_only=True,
        )
        print(f"[Model] Loading SFT LoRA via PEFT...")
        model = PeftModel.from_pretrained(model, sft_ckpt, is_trainable=True)
    else:
        print(f"[Model] Loading SFT FULL model on CPU from {sft_ckpt}...")
        model = _load_full_sft_model(sft_ckpt, sft_format)
        # Check whether RL should attach a fresh LoRA or fine-tune all params.
        lora_rank = int(llamacfg.get("lora_rank", 0) or 0)
        if lora_rank > 0:
            print(f"[Model] RL mode: LoRA-on-SFT-full (rank={lora_rank}, "
                  f"alpha={llamacfg.get('lora_alpha', 2*lora_rank)})")
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=int(llamacfg.get("lora_alpha", 2 * lora_rank)),
                lora_dropout=float(llamacfg.get("lora_dropout", 0.0)),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
                bias="none",
            )
            model = PeftModel(model, lora_cfg)
        else:
            print(f"[Model] RL mode: FULL-param (no LoRA). "
                  f"Requires 8-bit AdamW or ZeRO — optimizer built later.")

    _global_trainable_params = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        trainable_params = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        total_params = sum(parameter.numel() for parameter in model.parameters())
        ratio = 100.0 * trainable_params / max(total_params, 1)
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {total_params:,} || trainable%: {ratio:.4f}"
        )
    model.train()

    # Gradient checkpointing is on by default; generation must stay cacheless
    # (KV cache hangs on this model + PEFT). The input-require-grads hook is
    # needed because the frozen base leaves checkpointed layers without
    # grad-requiring inputs.
    if llamacfg.get("gradient_checkpointing", True):
        model.config.use_cache = False
        checkpoint_kwargs = {}
        if use_fsdp:
            # Non-reentrant checkpointing composes with FSDP and avoids the
            # repeated-ready reducer failures seen in the old full-DDP run.
            checkpoint_kwargs["gradient_checkpointing_kwargs"] = {
                "use_reentrant": False,
            }
        model.gradient_checkpointing_enable(**checkpoint_kwargs)
        def _make_inputs_require_grad(module, inputs, outputs):
            try:
                outputs.requires_grad_(True)
            except Exception:
                pass
        _hooked = False
        base = getattr(model, "get_base_model", lambda: model)()
        for emb in [
            getattr(base, "get_input_embeddings", lambda: None)(),
            getattr(base, "embed_tokens", None),
            getattr(getattr(base, "model", None), "embed_tokens", None),
        ]:
            if emb is not None:
                try:
                    emb.register_forward_hook(_make_inputs_require_grad)
                    _hooked = True
                    break
                except Exception:
                    continue
        print(f"[Model] Gradient checkpointing enabled (input_require_grads hook: {_hooked})")
    else:
        model.config.use_cache = True
        print("[Model] Gradient checkpointing OFF (fast KV-cached generation)")

    # Accelerate
    print("[Dist] Initializing Accelerate...")
    # NOTE:
    #  - gradient_accumulation_steps is FIXED at 1: train_step performs
    #    generate → forward → optimizer.step() in one call, so true gradient
    #    accumulation is not supported. Keep the yaml value at 1.
    #  - mixed_precision stays "no": the model is loaded in bf16, which needs no
    #    autocast/AMP; the yaml "fp16" flag is ignored (and misleading) here.
    accelerator_kwargs = {
        "gradient_accumulation_steps": 1,
        "mixed_precision": "no",
        # NOTE: no split_batches — each rank samples its own disjoint shard.
        # This is a custom loop without an Accelerate DataLoader.  train_step
        # calls optimizer.step() and lr_scheduler.step() exactly once per
        # logical global step, so do not let Accelerate multiply scheduler
        # stepping by the world size.
        "step_scheduler_with_optimizer": False,
        # One cacheless 512-token Qwen-Omni rollout can take much longer on a
        # rank with long audio/output than on its peers.  The default 10-minute
        # NCCL watchdog can therefore kill a healthy rank while it waits at the
        # FSDP summon/unshard boundary.  Keep failures finite, but allow a full
        # production-shaped rollout to converge again before collectives.
        "kwargs_handlers": [
            InitProcessGroupKwargs(timeout=timedelta(hours=2))
        ],
    }
    if use_fsdp and (
        llamacfg.get("ddp_static_graph", False)
        or llamacfg.get("ddp_find_unused_parameters", False)
    ):
        raise ValueError("FSDP configs must not set DDP compatibility options")
    if not use_fsdp and llamacfg.get("ddp_static_graph", False):
        # Full-parameter Qwen2.5-Omni with gradient checkpointing re-enters
        # checkpointed decoder blocks during the GRPO policy backward.  DDP's
        # default reducer rejects the resulting repeated-ready hooks; the
        # graph is fixed for this training loop, so static_graph is valid.
        accelerator_kwargs["kwargs_handlers"].append(
            DistributedDataParallelKwargs(static_graph=True)
        )
        print("[Dist] DDP static_graph enabled", flush=True)
    elif llamacfg.get("ddp_find_unused_parameters", False):
        # Qwen2.5-Omni has branches that are not touched by the closed-only
        # GRPO loss.  Full-param DDP must account for those parameters when
        # reducing the policy backward, otherwise the next forward fails with
        # "Expected to have finished reduction".
        accelerator_kwargs["kwargs_handlers"].append(
            DistributedDataParallelKwargs(find_unused_parameters=True)
        )
        print("[Dist] DDP find_unused_parameters enabled", flush=True)
    accelerator = Accelerator(
        **accelerator_kwargs,
    )
    set_seed(seed, device_specific=True)
    print(
        f"[Seed] rank={accelerator.process_index} "
        f"device_specific={seed + accelerator.process_index}",
        flush=True,
    )
    device = accelerator.device
    if not use_fsdp:
        model = model.to(device)

    # Reference model — frozen copy of the SFT init. Wrap in the SAME structure as the
    # policy so the adaptive reference can copy the policy's CURRENT weights (base +
    # trained adapter) into it after each curriculum merge (adaptive reference resetting).
    print("[Model] Loading reference model (frozen)...")
    if sft_is_lora:
        ref_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="cpu", local_files_only=True,
        )
        ref_model = PeftModel.from_pretrained(ref_model, sft_ckpt, is_trainable=False)
    else:
        # SFT full as base; same structural wrap as the policy (LoRA on top if lora_rank>0).
        ref_model = _load_full_sft_model(sft_ckpt, sft_format)
        if int(llamacfg.get("lora_rank", 0) or 0) > 0:
            # Ref doesn't need trainable LoRA — just a structural match for state_dict copies.
            # Reuse the SAME LoraConfig so adapter names/shapes line up.
            lora_rank = int(llamacfg.get("lora_rank", 0) or 0)
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=int(llamacfg.get("lora_alpha", 2 * lora_rank)),
                lora_dropout=float(llamacfg.get("lora_dropout", 0.0)),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
                bias="none",
            )
            ref_model = PeftModel(ref_model, lora_cfg)
    ref_model.eval()
    ref_model.config.use_cache = False
    for p in ref_model.parameters():
        p.requires_grad = False

    if use_fsdp:
        _is_full_checkpoint_mode = (
            not sft_is_lora and int(llamacfg.get("lora_rank", 0) or 0) == 0
        )
        if not _is_full_checkpoint_mode:
            raise ValueError("distributed.strategy=fsdp is reserved for full-param RL")
        if accelerator.num_processes != 8:
            raise ValueError(
                "the strict full-RL FSDP contract requires exactly 8 processes, "
                f"found {accelerator.num_processes}"
            )
        policy_wrap = _fsdp_auto_wrap_policy(model)
        # Resolve the reference policy independently so the callable contains
        # the concrete layer classes from that model instance/version too.
        reference_wrap = _fsdp_auto_wrap_policy(ref_model)
        model = _wrap_full_model_fsdp(model, device, policy_wrap)
        ref_model = _wrap_full_model_fsdp(ref_model, device, reference_wrap)
        model.train()
        ref_model.eval()
        print("[FSDP] policy and frozen reference use FULL_SHARD", flush=True)
    else:
        ref_model = ref_model.to(device)

    # Trainer
    from trainer.curriculum_scheduler import CurriculumScheduler, CellQueueSampler
    from rewards.quality_reward import QualityRewardPipeline
    from trainer.grpo_trainer import GRPOTrainer

    scheduler = CurriculumScheduler(
        n_buckets=curriculum_cfg.get("K", 4),
        M=curriculum_cfg.get("M", 512),
        gamma=curriculum_cfg.get("gamma", 0.5),
        revisit_decay=curriculum_cfg.get("revisit_decay", 0.5),
        min_weight=curriculum_cfg.get("min_weight", 0.1),
        progression_mode=curriculum_cfg.get("progression_mode", "legacy"),
        min_frontier_prompts=curriculum_cfg.get("min_frontier_prompts", 0),
        consecutive_mastery_windows=curriculum_cfg.get(
            "consecutive_mastery_windows", 1,
        ),
        mastery_by_family=curriculum_cfg.get("mastery_by_family") or None,
        quota_seed=seed,
    )

    reward_cfg = rl_cfg.get("reward", {})

    reward_pipeline = QualityRewardPipeline(
        # Format reward weight: before T_f the format signal is summed with the
        # quality signal; keeping it well below 1.0 prevents a well-formatted but
        # WRONG answer from dominating the advantage. Tune freely.
        format_weight=reward_cfg.get("format_weight", 1.0),
    )

    kl_cfg = rl_cfg.get("kl", {})
    comp_cfg = rl_cfg.get("compression", {})
    aux_cfg = rl_cfg.get("aux_loss", {})

    gen_cfg = llamacfg.get("generation", {}) or {}
    # Qwen2.5-Omni's generation EOS is the omni-specific token (151645), not
    # the base tokenizer's eos — the SFT eval used it. Without it, generation
    # never stops early and every response runs to max_new_tokens.
    omni_eos = llamacfg.get("eos_token_id", 151645)
    generation_config = GenerationConfig(
        max_new_tokens=llamacfg.get("max_new_tokens", gen_cfg.get("max_new_tokens", 512)),
        do_sample=llamacfg.get("do_sample", gen_cfg.get("do_sample", True)),
        temperature=llamacfg.get("temperature", gen_cfg.get("temperature", 1.0)),
        top_p=llamacfg.get("top_p", gen_cfg.get("top_p", 1.0)),
        top_k=llamacfg.get("top_k", gen_cfg.get("top_k", 0)),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=omni_eos,
    )
    # Log-probs below are computed from the model's raw logits. Training
    # rollouts must therefore sample from that exact categorical distribution;
    # temperature/top-p/top-k warpers would require matching warped log-probs
    # and a separately stored behavior policy.
    if (
        not generation_config.do_sample
        or float(generation_config.temperature) != 1.0
        or float(generation_config.top_p) != 1.0
        or int(generation_config.top_k) != 0
    ):
        raise ValueError(
            "GRPO rollout/logprob contract requires do_sample=true, "
            "temperature=1.0, top_p=1.0, top_k=0"
        )

    # Dataset — loads audio on demand (LRU-cached)
    audio_root = rl_cfg.get("paths", {}).get(
        "audio_root",
        os.environ.get("AUDIO_ROOT", ""),
    )
    from data.curriculum_dataset import CurriculumDataset
    dataset = CurriculumDataset(
        curriculum_file, audio_root=audio_root,
        audio_cache_cap=rl_cfg.get("paths", {}).get("audio_cache_cap", 512),
    )
    data_pipeline_manifest = {
        "source_num_items": len(dataset),
        "filter": {},
        "option_permutation": {"enabled": False},
    }
    # Optional data filter (for debug: restrict to subsets/types with real reward signal)
    df_cfg = rl_cfg.get("data_filter", {})
    if df_cfg:
        # exclude_open: keep S2TT + choice only (open questions would otherwise
        # get a constant 0.5 and waste compute).
        if df_cfg.get("exclude_open"):
            keep = [i for i, it in enumerate(dataset.items)
                    if it.get("subset") == "S2TT" or it.get("type") == "choice"]
            reason = "exclude_open=True"
        else:
            keep = [i for i, it in enumerate(dataset.items)
                    if (not df_cfg.get("subsets") or it.get("subset") in df_cfg["subsets"])
                    and (not df_cfg.get("types") or it.get("type") in df_cfg["types"])]
            reason = f"subsets={df_cfg.get('subsets')}, types={df_cfg.get('types')}"
        dataset.items = [dataset.items[i] for i in keep]
        dataset.bucket_to_idx = {}
        for i, it in enumerate(dataset.items):
            dataset.bucket_to_idx.setdefault(it.get("bucket_id", 0), []).append(i)
        dataset.n_buckets = len(dataset.bucket_to_idx)
        print(f"[Data] Filtered to {len(dataset.items)} items ({reason})")
        data_pipeline_manifest["filter"] = {
            "reason": reason,
            "num_items": len(dataset),
        }

    augmentation_cfg = rl_cfg.get("data_augmentation", {}) or {}
    option_permutation_cfg = augmentation_cfg.get("option_permutation", {}) or {}
    option_permutation_enabled = bool(option_permutation_cfg.get("enabled", False))
    option_permutation_seed = int(option_permutation_cfg.get("seed", seed))
    if option_permutation_enabled:
        from data.choice_permutation import parse_choice_item

        if any(task_family(item) != "choice" for item in dataset.items):
            raise ValueError(
                "option permutation requires a choice-only filtered population"
            )
        valid_items = []
        original_answers = Counter()
        invalid_count = 0
        for item in dataset.items:
            parsed = parse_choice_item(item)
            if parsed is None:
                invalid_count += 1
                continue
            valid_items.append(item)
            original_answers["ABCD"[parsed.correct_index]] += 1
        if invalid_count and not option_permutation_cfg.get("drop_invalid", False):
            raise ValueError(
                f"option permutation found {invalid_count} invalid choice rows; "
                "set drop_invalid=true to exclude them explicitly"
            )
        if invalid_count:
            dataset.items = valid_items
            dataset.bucket_to_idx = {}
            for index, item in enumerate(dataset.items):
                dataset.bucket_to_idx.setdefault(
                    int(item.get("bucket_id", 0)), []
                ).append(index)
            dataset.n_buckets = len(dataset.bucket_to_idx)
        data_pipeline_manifest["option_permutation"] = {
            "enabled": True,
            "seed": option_permutation_seed,
            "drop_invalid": bool(
                option_permutation_cfg.get("drop_invalid", False)
            ),
            "invalid_rows_dropped": invalid_count,
            "num_items": len(dataset),
            "original_answer_counts": dict(sorted(original_answers.items())),
        }
        print(
            "[Augment] option_permutation enabled: "
            f"items={len(dataset)}, dropped_invalid={invalid_count}, "
            f"original_answers={dict(sorted(original_answers.items()))}",
            flush=True,
        )
    # Per-step curriculum sampling.
    # Each rank owns a disjoint shard of the dataset (DDP data-parallel
    # generation) and re-samples a batch EVERY step from the CURRENT curriculum
    # weights — so when the scheduler merges B1/B2/B3, the training data
    # actually changes. The old "sample a fixed pool once" approach never
    # reflected merges and effectively trained only bucket 0 forever.
    bs = llamacfg.get("per_device_train_batch_size", 4)
    world = accelerator.num_processes
    rank = accelerator.process_index
    rank_indices = list(range(rank, len(dataset), world))
    rank_bucket_indices: dict = {}
    for i in rank_indices:
        bid = dataset.items[i].get("bucket_id", 0)
        rank_bucket_indices.setdefault(bid, []).append(i)
    rank_rng = random.Random(seed + rank)
    print(f"[Dist] rank{rank}: owns {len(rank_indices)} items across "
          f"{len(rank_bucket_indices)} buckets", flush=True)

    sampling_cfg = rl_cfg.get("sampling", {}) or {}
    sampling_mode = str(sampling_cfg.get("mode", "legacy_weighted_pool"))
    strict_global_prompts = sampling_cfg.get("require_global_prompts_per_step")
    strict_micro_slots = sampling_cfg.get("micro_slots_per_rank")
    if strict_global_prompts is not None:
        strict_global_prompts = int(strict_global_prompts)
        strict_micro_slots = int(strict_micro_slots or bs)
        if strict_global_prompts <= 0 or strict_micro_slots <= 0:
            raise ValueError("strict global prompt and micro-slot counts must be positive")
        if strict_global_prompts > world * strict_micro_slots:
            raise ValueError(
                "global prompts exceed the available synchronized micro slots: "
                f"{strict_global_prompts} > {world}*{strict_micro_slots}"
            )
        if strict_global_prompts < world:
            raise ValueError("strict sampling requires at least one real prompt per rank")
    else:
        strict_global_prompts = world * bs
        strict_micro_slots = bs
    required_world = sampling_cfg.get("require_world_size")
    if required_world is not None and world != int(required_world):
        raise ValueError(
            f"sampling requires world_size={int(required_world)}, found {world}"
        )
    required_bs = sampling_cfg.get("require_per_device_batch_size")
    if required_bs is not None and bs != int(required_bs):
        raise ValueError(
            "sampling requires per_device_train_batch_size="
            f"{int(required_bs)}, found {bs}"
        )
    cell_sampler = None
    if sampling_mode == "global_residual_quota":
        expected_sha256 = sampling_cfg.get("require_curriculum_sha256")
        if (
            expected_sha256 is not None
            and dataset.data_sha256.casefold() != str(expected_sha256).casefold()
        ):
            raise ValueError(
                "global_residual_quota curriculum SHA-256 mismatch: "
                f"expected {expected_sha256}, found {dataset.data_sha256}"
            )
        expected_items = sampling_cfg.get("require_num_items")
        if expected_items is not None and len(dataset) != int(expected_items):
            raise ValueError(
                "global_residual_quota item-count mismatch: "
                f"expected {int(expected_items)}, found {len(dataset)}"
            )
        expected_version = sampling_cfg.get("require_curriculum_version")
        if expected_version:
            versions = {
                str(item.get("curriculum_version", "")) for item in dataset.items
            }
            if versions != {str(expected_version)}:
                raise ValueError(
                    "global_residual_quota curriculum_version mismatch: "
                    f"expected {expected_version!r}, found {sorted(versions)!r}"
                )

        observed_buckets = {
            int(item.get("bucket_id", 0)) for item in dataset.items
        }
        expected_buckets = set(range(scheduler.K))
        if observed_buckets != expected_buckets:
            raise ValueError(
                "B+ bucket IDs must be contiguous and match K: "
                f"expected {sorted(expected_buckets)}, found "
                f"{sorted(observed_buckets)}"
            )

        stratum_counts = Counter(bplus_stratum(item) for item in dataset.items)
        scheduler.configure_quota(stratum_counts)
        all_cell_counts = Counter(
            (int(item.get("bucket_id", 0)), bplus_stratum(item))
            for item in dataset.items
        )
        expected_cells = {
            (bucket, stratum)
            for bucket in expected_buckets
            for stratum in stratum_counts
        }
        missing_global = sorted(expected_cells - set(all_cell_counts))
        if missing_global:
            raise ValueError(
                f"B+ is missing {len(missing_global)} bucket/stratum cells; "
                f"first={missing_global[:5]!r}"
            )

        rank_cell_indices: dict = {}
        for index in rank_indices:
            item = dataset.items[index]
            cell = (int(item.get("bucket_id", 0)), bplus_stratum(item))
            rank_cell_indices.setdefault(cell, []).append(index)
        missing_local = sorted(expected_cells - set(rank_cell_indices))
        if missing_local:
            raise ValueError(
                f"rank{rank} has no local rows for {len(missing_local)} B+ cells; "
                f"first={missing_local[:5]!r}. The global quota contract needs "
                "every rank shard to cover every cell."
            )
        cell_sampler = CellQueueSampler(rank_cell_indices, seed=seed, rank=rank)
        print(
            f"[Sampling] global_residual_quota: {len(stratum_counts)} strata, "
            f"{len(expected_cells)} cells, global_batch={strict_global_prompts}, "
            f"micro_slots_per_rank={strict_micro_slots}",
            flush=True,
        )
    elif sampling_mode != "legacy_weighted_pool":
        raise ValueError(f"unknown sampling.mode: {sampling_mode!r}")

    sample_step = 0

    def sample_batch(batch_size: int) -> dict:
        """Sample one per-rank batch from the live curriculum."""
        nonlocal sample_step
        if sampling_mode == "global_residual_quota":
            import torch.distributed as dist

            if accelerator.is_main_process:
                global_cells = scheduler.plan_quota_cells(strict_global_prompts)
            else:
                global_cells = None
            if world > 1:
                objects = [global_cells]
                dist.broadcast_object_list(objects, src=0)
                global_cells = objects[0]
            if global_cells is None or len(global_cells) != strict_global_prompts:
                raise RuntimeError("invalid global quota plan broadcast")
            base_count, extra = divmod(strict_global_prompts, world)
            # Rotate the ranks receiving the extra real prompt so every shard
            # contributes equally over time. All ranks still execute exactly
            # strict_micro_slots FSDP forwards; the trainer pads with zero-loss
            # collective-only slots where necessary.
            extra_start = (sample_step * max(extra, 1)) % world
            extra_ranks = {
                (extra_start + offset) % world for offset in range(extra)
            }
            rank_counts = [
                base_count + (1 if r in extra_ranks else 0)
                for r in range(world)
            ]
            if max(rank_counts) > strict_micro_slots:
                raise RuntimeError("rank prompt allocation exceeds micro slots")
            offset = sum(rank_counts[:rank])
            local_cells = global_cells[offset:offset + rank_counts[rank]]
            idx = cell_sampler.sample_cells(local_cells)
        else:
            idx = scheduler.sample_indices(
                rank_bucket_indices, batch_size, rng=rank_rng,
            )
            if not idx:
                # Active buckets have no items in this rank's shard — fall back
                # to anything the rank owns (smoke runs only).
                all_rank = [
                    index
                    for indices in rank_bucket_indices.values()
                    for index in indices
                ]
                idx = rank_rng.choices(all_rank, k=batch_size) if all_rank else []
        items = [dataset[i] for i in idx]
        if option_permutation_enabled:
            from data.choice_permutation import (
                exposure_permutation,
                permute_choice_item,
            )

            items = [
                permute_choice_item(
                    item,
                    exposure_permutation(
                        item,
                        seed=option_permutation_seed,
                        rank=rank,
                        step=sample_step,
                        slot=slot,
                    ),
                )
                for slot, item in enumerate(items)
            ]
        sample_step += 1
        return collate_fn(items)

    def sampling_state_dict() -> dict:
        """Per-rank stochastic/data state required for an exact FSDP resume."""
        return {
            "sample_step": int(sample_step),
            "cell_sampler": (
                cell_sampler.state_dict() if cell_sampler is not None else None
            ),
            "rank_rng": rank_rng.getstate(),
            "python_rng": random.getstate(),
            "numpy_rng": __import__("numpy").random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": (
                torch.cuda.get_rng_state(device)
                if torch.cuda.is_available() else None
            ),
        }

    def load_sampling_state_dict(state: dict) -> None:
        nonlocal sample_step
        if "sample_step" not in state:
            raise ValueError("unsupported RL sampling state")
        sample_step = int(state["sample_step"])
        saved_cell_sampler = state.get("cell_sampler")
        if (saved_cell_sampler is None) != (cell_sampler is None):
            raise ValueError("checkpoint cell-sampler mode differs from config")
        if cell_sampler is not None:
            cell_sampler.load_state_dict(saved_cell_sampler)
        rank_rng.setstate(state["rank_rng"])
        random.setstate(state["python_rng"])
        __import__("numpy").random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_cpu_rng"])
        if state.get("torch_cuda_rng") is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("checkpoint has CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state(state["torch_cuda_rng"], device=device)
        print(
            f"[Resume] rank={rank} sampling/RNG state restored at "
            f"sample_step={sample_step}",
            flush=True,
        )

    # Optimizer — for LoRA training vanilla AdamW is fine (few params). For FULL-param
    # RL (lora_rank=0) we MUST use 8-bit AdamW (~2 bytes/param for two 8-bit
    # moments, about 14GB for 7B parameters) to fit
    # in 80GB GPU; vanilla fp32 AdamW states (8B/param → 56GB for 7B) OOM with activations.
    _trainable = [p for p in model.parameters() if p.requires_grad]
    _n_train = sum(p.numel() for p in _trainable)
    _is_full = not sft_is_lora and int(llamacfg.get("lora_rank", 0) or 0) == 0
    if _is_full and use_fsdp:
        optimizer = AdamW(
            _trainable,
            lr=llamacfg.get("learning_rate", 1e-6),
            weight_decay=0.0,
            fused=bool((llamacfg.get("optimizer") or {}).get("fused", True)),
        )
        print(
            f"[Optim] sharded fused AdamW over {_global_trainable_params/1e9:.2f}B "
            "global params",
            flush=True,
        )
    elif _is_full and HAS_BNB:
        paged_optimizer = bool(llamacfg.get("paged_optimizer", False))
        if paged_optimizer:
            optimizer_cls = getattr(bnb.optim, "PagedAdamW8bit", None)
            if optimizer_cls is None:
                raise RuntimeError(
                    "paged_optimizer=true but this training image does not "
                    "provide bitsandbytes.optim.PagedAdamW8bit"
                )
            optimizer = optimizer_cls(
                _trainable,
                lr=llamacfg.get("learning_rate", 1e-6), weight_decay=0.0,
            )
            print(
                f"[Optim] bitsandbytes PagedAdamW8bit over {_n_train/1e9:.2f}B "
                "params (paged optimizer state)",
                flush=True,
            )
        else:
            optimizer = bnb.optim.AdamW8bit(
                _trainable,
                lr=llamacfg.get("learning_rate", 1e-6), weight_decay=0.0,
            )
            print(f"[Optim] bitsandbytes AdamW8bit over {_n_train/1e9:.2f}B params "
                  f"(≈{_n_train*2/1e9:.1f}GB moment state)", flush=True)
    elif _is_full and not HAS_BNB:
        raise RuntimeError(
            "bitsandbytes is required for full-parameter RL: vanilla AdamW "
            f"would allocate about {_n_train*8/1e9:.1f}GB of optimizer state "
            "per rank and is not a safe fallback on 80GB GPUs. Use the "
            "validated training SIF or set lora_rank>0 for a LoRA run."
        )
    else:
        optimizer = AdamW(
            _trainable,
            lr=llamacfg.get("learning_rate", 1e-6), weight_decay=0.0,
        )
        print(f"[Optim] AdamW (LoRA) over {_n_train/1e6:.1f}M params", flush=True)
    max_steps = int(llamacfg.get("max_steps", 10))
    lr_schedule = str(llamacfg.get("lr_schedule", "linear")).strip().lower()
    if lr_schedule == "constant":
        # 1e-6 is intentionally kept unchanged for every logical step.  The
        # custom loop has no dataloader-driven scheduler cadence.
        lr_scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    elif lr_schedule == "linear":
        lr_scheduler = LambdaLR(
            optimizer, lambda s: max(0.0, 1.0 - s / max_steps)
        )
    else:
        raise ValueError(
            f"Unsupported lr_schedule={lr_schedule!r}; use 'constant' or 'linear'"
        )
    print(
        f"[Optim] lr_schedule={lr_schedule} "
        f"base_lr={llamacfg.get('learning_rate', 1e-6):.3e} "
        f"max_steps={max_steps} "
        "scheduler_step_per_logical_step=1",
        flush=True,
    )

    def _make_trainer(t_model):
        return GRPOTrainer(
            model=t_model, ref_model=ref_model,
            tokenizer=tokenizer, processor=processor,
            accelerator=accelerator,
            curriculum_scheduler=scheduler,
            reward_pipeline=reward_pipeline,
            generation_config=generation_config,
            G=grpo_cfg.get("G", 6),
            beta=grpo_cfg.get("beta", 0.04),
            clip_eps=grpo_cfg.get("clip_eps", 0.2),
            T_f=grpo_cfg.get("T_f", 64),
            max_grad_norm=1.0,
            use_dynamic_kl=kl_cfg.get("dynamic", True),
            reset_reference_on_merge=kl_cfg.get(
                "reset_reference_on_merge", True,
            ),
            comp_caps=comp_cfg.get("caps", [200, 350, 500, 700, 900]),
            comp_caps_by_family=comp_cfg.get("caps_by_family") or {},
            comp_edges_by_family=comp_cfg.get("difficulty_edges_by_family") or {},
            comp_mode=comp_cfg.get("mode", "global_rms"),
            comp_section_caps_by_family=(
                comp_cfg.get("section_caps_by_family") or {}
            ),
            comp_summary_caps_by_family=(
                comp_cfg.get("summary_caps_by_family") or {}
            ),
            comp_invalid_reward=comp_cfg.get("invalid_reward", -0.5),
            comp_rating_levels=comp_cfg.get("rating_levels", 5),
            comp_coeff=comp_cfg.get("coeff", 0.15),
            comp_under_coeff=comp_cfg.get("under_coeff", 0.1),
            comp_min_frac=comp_cfg.get("min_frac", 0.3),
            aux_loss_coeff=aux_cfg.get("coeff", 0.1),
            soft_variance=aux_cfg.get("soft_variance", 0.5),
            num_ratings=aux_cfg.get("num_ratings", 5),
            # Per-type generation lengths (S2TT/choice short, open longer).
            # NOTE: the yaml is FLAT (### sections are comments), so read from
            # the top level, not from gen_cfg["generation"].
            max_new_tokens_by_type=llamacfg.get("max_new_tokens_by_type") or {},
            # Per-task-type competence baseline for the curriculum scheduler.
            baseline_by_type=curriculum_cfg.get("baseline_by_type") or {},
            streaming_backward=bool(use_fsdp),
            micro_slots_per_rank=int(strict_micro_slots),
            global_prompts_per_step=int(strict_global_prompts),
            sampling_state_getter=sampling_state_dict,
            sampling_state_loader=load_sampling_state_dict,
        )

    world_size = accelerator.num_processes
    if use_fsdp:
        # Models were wrapped directly so both policy and frozen reference use
        # identical FULL_SHARD layouts. Preparing again would add DDP on top.
        trainer = _make_trainer(model)
    elif world_size > 1:
        # Multi-GPU DDP: prepare first, trainer gets the DDP-wrapped model so
        # backward gradients are all-reduced across ranks.
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler,
        )
        trainer = _make_trainer(model)
    else:
        # Single GPU: create trainer with the raw model BEFORE prepare
        # (proven reliable baseline).
        trainer = _make_trainer(model)
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler,
        )

    print(f"[Dist] World size: {world_size}")

    # A formal run owns a fresh output directory.  Persist the exact resolved
    # config and immutable source/code provenance before step 1; a crash leaves
    # status=running, while normal completion updates the same manifest.
    output_dir = Path(llamacfg.get("output_dir", "./output")).resolve()
    run_manifest = None
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=False)
        resolved_config = dict(llamacfg)
        resolved_config["rl"] = rl_cfg
        with (output_dir / "config_resolved.yaml").open(
            "x", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(
                resolved_config,
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        run_manifest = _make_run_manifest(
            config_path=Path(args.config),
            curriculum_file=Path(curriculum_file),
            curriculum_sha256=dataset.data_sha256,
            sft_checkpoint=Path(sft_ckpt),
            sft_format=sft_format,
            model_path=model_path,
            seed=seed,
            world_size=world_size,
            per_device_batch=bs,
            group_size=trainer.G,
            max_steps=max_steps,
            global_prompt_batch=strict_global_prompts,
            micro_slots_per_rank=strict_micro_slots,
            distributed_strategy=distributed_strategy,
        )
        run_manifest["data_pipeline"] = data_pipeline_manifest
        _atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    accelerator.wait_for_everyone()

    print(f"\n{'='*60}")
    print(f"Curriculum GRPO — {'LoRA' if isinstance(accelerator.unwrap_model(model), PeftModel) else 'FULL-param'} training")
    print(f"  Steps={max_steps}, G={trainer.G}, audio={audio_root}")
    print(f"  Device={device}")
    print(f"{'='*60}\n")

    trainer.train(sample_batch=sample_batch, optimizer=optimizer,
                  lr_scheduler=lr_scheduler, max_steps=max_steps,
                  per_device_bs=bs,
                  log_steps=llamacfg.get("logging_steps", 10),
                  save_steps=llamacfg.get("save_steps", 200),
                  output_dir=str(output_dir),
                  resume=llamacfg.get("resume_from_checkpoint"))
    if accelerator.is_main_process:
        run_manifest["status"] = "completed"
        run_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_manifest["final_global_step"] = trainer.global_step
        run_manifest["final_curriculum"] = scheduler.get_progress()
        run_manifest["final_checkpoint"] = str(
            output_dir / "checkpoints" / "step_final"
        )
        _atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    print("[Done]")


if __name__ == "__main__":
    main()
