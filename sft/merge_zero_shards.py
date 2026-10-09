#!/usr/bin/env python3
"""Merge a DeepSpeed ZeRO checkpoint into HF safetensors.

Input dir structure (llama_factory full-finetune output):
    <ckpt_dir>/
        config.json
        tokenizer files...
        global_step<N>/
            zero_pp_rank_0_mp_rank_00_model_states.pt
            ...
            zero_pp_rank_7_mp_rank_00_model_states.pt
            bf16_zero_pp_rank_*_optim_states.pt   (dropped)

ZeRO-3 model-state files contain metadata plus flat parameter partitions, not
independent ``module`` dictionaries. Use DeepSpeed's official reconstruction
API rather than concatenating shard dictionaries.
"""

import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                    help="Top-level checkpoint dir (contains global_step<N>/ and config.json)")
    ap.add_argument("--step_dir", default=None,
                    help="Sub-dir to merge (default: auto-detect single global_step*/ dir)")
    ap.add_argument("--out_dir", required=True, help="Output dir for HF-format weights")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    ckpt_dir = args.ckpt_dir
    if args.step_dir:
        step_dir = os.path.join(ckpt_dir, args.step_dir)
    else:
        subs = [d for d in os.listdir(ckpt_dir)
                if d.startswith("global_step")
                and os.path.isdir(os.path.join(ckpt_dir, d))]
        assert len(subs) == 1, f"Expected 1 global_step dir, found {subs}"
        step_dir = os.path.join(ckpt_dir, subs[0])
    print(f"[Merge] step_dir: {step_dir}")

    # Validate the model-state shards, then let DeepSpeed reconstruct the flat
    # ZeRO-3 parameter partitions (including shared parameters) correctly.
    shard_files = sorted(
        f for f in os.listdir(step_dir)
        if f.startswith("zero_pp_rank_") and f.endswith("_model_states.pt")
    )
    assert shard_files, f"No model_states shards in {step_dir}"
    print(f"[Merge] {len(shard_files)} shards")
    from deepspeed.utils.zero_to_fp32 import (
        get_fp32_state_dict_from_zero_checkpoint,
    )

    tag = os.path.basename(step_dir)
    print(f"[Merge] reconstructing tag={tag}")
    merged = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, tag=tag)
    print(f"[Merge] reconstructed {len(merged)} tensors")

    # Qwen2.5-Omni thinker uses tied embeddings maybe; just load into model skeleton
    cfg_path = os.path.join(ckpt_dir, "config.json")
    cfg = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
    print(f"[Merge] config from {cfg_path}")
    print(f"[Merge] arch: {cfg.architectures}")

    # Instantiate the destination directly in the requested dtype. This avoids
    # an unnecessary second fp32 7B allocation while the reconstructed state
    # dict is resident in CPU memory.
    cls_name = cfg.architectures[0]
    if "Qwen2_5OmniThinker" in cls_name:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        ModelCls = Qwen2_5OmniThinkerForConditionalGeneration
    else:
        ModelCls = AutoModelForCausalLM
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"[Merge] instantiating {ModelCls.__name__} on CPU ({args.dtype})")
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = ModelCls(cfg)
    finally:
        torch.set_default_dtype(previous_default_dtype)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ZeRO checkpoint/config mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[Merge] saving to {args.out_dir} (dtype={args.dtype})")
    model.save_pretrained(args.out_dir, safe_serialization=True)

    # Also copy tokenizer + generation_config etc.
    import shutil
    for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "added_tokens.json", "chat_template.jinja", "generation_config.json",
               "preprocessor_config.json", "merges.txt"]:
        src = os.path.join(ckpt_dir, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out_dir, fn))
            print(f"  + copied {fn}")
    print("[Merge] Done.")


if __name__ == "__main__":
    main()
