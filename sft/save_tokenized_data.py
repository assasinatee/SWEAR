#!/usr/bin/env python3
"""
Preprocess dataset (format conversion + tokenizer) and save to disk.
Run once; subsequent training loads from disk instead of re-processing.

Usage:
  python save_tokenized_data.py --data_dir /path/to/converted_data --model_path /path/to/model
"""
import argparse
import os
import json
import soundfile as sf
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_path",
                        default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Omni-7B"))
    parser.add_argument("--media_dir", default=os.environ.get("AUDIO_ROOT", ""))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--val_size", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "tokenized_data")

    if os.path.exists(args.output_dir):
        print(f"[OK] Tokenized data already exists: {args.output_dir}")
        ds = load_from_disk(args.output_dir)
        print(f"  Train: {len(ds['train'])}, Eval: {len(ds['test'])}")
        return

    # 1. Load raw JSON
    json_path = os.path.join(args.data_dir, "full.json")
    print(f"[1/5] Loading raw dataset: {json_path}")
    raw = load_dataset("json", data_files=json_path, split="train")
    print(f"  Loaded {len(raw)} samples")

    # Split train/eval
    splits = raw.train_test_split(test_size=args.val_size, seed=args.seed)
    dataset = DatasetDict({"train": splits["train"], "test": splits["test"]})
    print(f"  Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")

    # 2. Format conversion: expand sharegpt columns
    print("[2/5] Format conversion...")
    def convert_format(examples):
        batch = {"audios": [], "messages": []}
        for i in range(len(examples["audios"])):
            audio_paths = [os.path.join(args.media_dir, f) for f in examples["audios"][i]]
            batch["audios"].append(audio_paths)
            batch["messages"].append(examples["conversations"][i])
        return batch

    dataset = dataset.map(convert_format, batched=True, num_proc=16,
                          remove_columns=["audios", "conversations"],
                          desc="Format conversion")
    print(f"  Features: {list(dataset['train'].features.keys())}")

    # 3. Load processor
    print("[3/5] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # 4. Tokenizer step
    print("[4/5] Tokenizer (audio + text)...")
    def tokenize_fn(examples):
        texts = []
        for msgs in examples["messages"]:
            texts.append(processor.tokenizer.apply_chat_template(msgs, tokenize=False))
        audio_arrays = []
        for paths in examples["audios"]:
            arrs = []
            for p in paths:
                audio, sr = sf.read(p)
                arrs.append(audio)
            audio_arrays.append(arrs)
        return processor(
            text=texts, audios=audio_arrays,
            sampling_rate=16000, padding=False, return_tensors=None,
        )

    dataset = dataset.map(tokenize_fn, batched=True, num_proc=4,
                          remove_columns=["audios", "messages"],
                          desc="Tokenizer")
    print(f"  Features: {list(dataset['train'].features.keys())[:10]}...")

    # 5. Save to disk
    print(f"[5/5] Saving to {args.output_dir}...")
    dataset.save_to_disk(args.output_dir)
    print(f"[OK] Done! Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")


if __name__ == "__main__":
    main()
