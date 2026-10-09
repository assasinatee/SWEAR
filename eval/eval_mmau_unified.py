#!/usr/bin/env python3
"""MMAU-test-mini evaluation for base / LoRA checkpoints.

Uses the "which" prompt (matching training), letter-labeled options, and
conservative final-block answer parsing. Scoring is exact label match with
exact duplicate choices treated as equivalent. Per-sample records and
category/difficulty/task breakdowns are saved for offline re-analysis.
"""

import argparse
import ast
import io
import json
import os
import re
import statistics

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CRYPTOGRAPHY_OPENSSL_NO_LEGACY", "1")

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, Qwen2_5OmniProcessor, GenerationConfig
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from peft import PeftModel

from baseline_answer_parser import resolve_choice
from choice_answer_scoring import score_choice

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Omni-7B")
SFT_CKPT = os.environ.get("SFT_CKPT", "")
RAW_CKPT = os.environ.get("RAW_CKPT", "")
MMAU = os.environ.get("MMAU_PARQUET", "MMAU-test-mini/test_mini.parquet")
DEFAULT_OUT = os.environ.get("MMAU_OUT", "results/eval_mmau_unified.json")
DEFAULT_MAX_TOKENS = 1536

# Used only for the RESPONSE-block coverage diagnostic.  Prediction parsing is
# centralized in baseline_answer_parser.py.
_RESPONSE_TAG_RE = re.compile(r"<RESPONSE>(.*?)</RESPONSE>", re.IGNORECASE | re.DOTALL)


def resolve_pred(resp, choices):
    """Return the model's selected label and an auditable parse method."""

    parsed = resolve_choice(resp, choices)
    return parsed.label, parsed.method


def extract_gt(answer_field):
    if not isinstance(answer_field, str):
        return None
    m = re.match(r"\s*\(([A-H])\)", answer_field)
    if m:
        return m.group(1).upper()
    m = re.search(r"([A-H])", answer_field)
    return m.group(1).upper() if m else None


def to_list(x):
    """Flatten MMAU choices (list / array / nested list / str) into strings."""
    if isinstance(x, np.ndarray):
        return [str(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if isinstance(v, np.ndarray):
                out.extend(str(t) for t in v.tolist())
            else:
                out.append(str(v))
        return out
    if isinstance(x, str):
        try:
            y = ast.literal_eval(x)
            if isinstance(y, (list, tuple)):
                return to_list(y)
        except Exception:
            pass
    return [str(x)]


def build_which_prompt(question, choices):
    """Training-format prompt: keep existing option letters, else label (a)(b)(c)..."""
    parts = []
    for i, c in enumerate(choices):
        c_str = str(c).strip()
        m = re.match(r"\(([A-D])\)\s*(.*)", c_str)
        if m:
            parts.append(f"({m.group(1).lower()}) {m.group(2).strip()}")
        else:
            parts.append(f"({chr(97 + i)}) {c_str}")
    return f"Which of the following is correct? {question.strip()} " + " ".join(parts)


def build_options_prompt(question, choices, suffix):
    return f"Question: {question}\nOptions:\n" + "\n".join(choices) + "\n" + suffix


PROMPTS = {
    "which": build_which_prompt,
    "direct": lambda q, cs: build_options_prompt(q, cs, "Please output the letter of the correct answer directly."),
    "cot": lambda q, cs: build_options_prompt(q, cs, "Please analyze the audio and answer the question."),
}


def load_model(args):
    print(f"[mmau] model {args.model} + ckpt {args.ckpt if not args.no_lora else '(full/base)'}, "
          f"merge={args.merge}, max_new_tokens={args.max_new_tokens}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(
        args.processor or args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu", local_files_only=True)
    if not args.no_lora:
        if args.merge == "peft":
            model = PeftModel.from_pretrained(model, args.ckpt, is_trainable=False)
        else:
            from safetensors.torch import load_file
            scale = args.alpha / args.r
            adapter = load_file(os.path.join(args.ckpt, "adapter_model.safetensors"))
            param_map = {n: p for n, p in model.named_parameters()}
            lora_a = {}
            for k, v in adapter.items():
                rel = k.replace("base_model.model.", "")
                if rel.endswith("lora_A.weight"):
                    lora_a[rel.replace(".lora_A.weight", ".weight")] = v.to(torch.float16).to("cuda")
            merged = 0
            for k, v in adapter.items():
                rel = k.replace("base_model.model.", "")
                if not rel.endswith("lora_B.weight"):
                    continue
                bk = rel.replace(".lora_B.weight", ".weight")
                if bk in lora_a and bk in param_map:
                    param_map[bk].data.add_(scale * (v.to(torch.float16).to("cuda") @ lora_a[bk]))
                    merged += 1
            del adapter, lora_a, param_map
            print(f"[mmau] manual-merged {merged} LoRA pairs (scale={scale})", flush=True)
    model = model.to("cuda").eval()
    gen = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0,
                           pad_token_id=tokenizer.pad_token_id, eos_token_id=151645)
    return processor, model, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="HF model directory; use --no-lora for a merged/full model")
    ap.add_argument("--processor", default=None, help="Optional processor directory when a full checkpoint exports weights/tokenizer but no preprocessor_config.json")
    ap.add_argument("--ckpt", default=SFT_CKPT, help="LoRA checkpoint directory (base model if --no-lora)")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--merge", choices=["peft", "manual"], default="peft",
                    help="manual merge scales LoRA by alpha/r (default 128/64=2.0)")
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--prompt", choices=list(PROMPTS), default="which",
                    help="which matches training (default); direct/cot are baselines")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--n", type=int, default=0, help="evaluate only the first n samples")
    ap.add_argument("--parquet", default=MMAU,
                     help="MMAU test_mini.parquet path")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    processor, model, gen = load_model(args)
    df = pd.read_parquet(args.parquet)
    if args.n:
        df = df.iloc[:args.n]
    print(f"[mmau] {len(df)} items, prompt={args.prompt}", flush=True)

    predictions, gts, methods, n_toks, truncated, responses = [], [], [], [], [], []
    matches, strict_matches, duplicate_matches = [], [], []
    accepted_labels, score_reasons = [], []
    questions, all_choices, answers, all_metadata = [], [], [], []
    for idx, row in df.iterrows():
        audio, _ = librosa.load(io.BytesIO(row["context"]["bytes"]), sr=16000)
        question = str(row["instruction"])
        choices = to_list(row["choices"])
        user_text = PROMPTS[args.prompt](question, choices)
        messages = [
            {"role": "system", "content": [{"type": "text",
                                            "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": user_text}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, audio=audio, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen)
        out_ids = out[0] if isinstance(out, (tuple, list)) else out
        resp = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        n_tok = int(out_ids.shape[1] - inputs["input_ids"].shape[1])
        pred, method = resolve_pred(resp, choices)
        gt = extract_gt(row["answer"])
        score = score_choice(pred, gt, choices)
        predictions.append(pred); gts.append(gt); methods.append(method)
        n_toks.append(n_tok); truncated.append(n_tok >= args.max_new_tokens); responses.append(resp)
        matches.append(score.match); strict_matches.append(score.strict_match)
        duplicate_matches.append(score.duplicate_equivalent_match)
        accepted_labels.append(list(score.accepted_labels)); score_reasons.append(score.reason)
        questions.append(question); all_choices.append(choices); answers.append(str(row["answer"]))
        try:
            all_metadata.append(json.loads(row["other_attributes"] or "{}"))
        except Exception:
            all_metadata.append({})
        if idx % 100 == 0:
            print(f"  [{idx}/{len(df)}] done", flush=True)

    correct = sum(matches)
    strict_correct = sum(strict_matches)
    duplicate_corrected = sum(duplicate_matches)
    acc = correct / len(df) * 100
    strict_acc = strict_correct / len(df) * 100
    resp_hit = sum(1 for r in responses if _RESPONSE_TAG_RE.search(r))
    trunc_rate = sum(truncated) / len(truncated) * 100
    method_dist = {}
    for m in methods:
        method_dist[m] = method_dist.get(m, 0) + 1

    print(f"\n[mmau] correct={correct}/{len(df)} accuracy={acc:.2f}% "
          f"(strict={strict_correct}/{len(df)}={strict_acc:.2f}%, "
          f"duplicate_corrected=+{duplicate_corrected})")
    print(f"[mmau] method_dist={method_dist}")
    print(f"[mmau] <RESPONSE> hit={resp_hit}/{len(df)} = {resp_hit/len(df)*100:.1f}%")
    print(f"[mmau] truncated={trunc_rate:.1f}%  avg_tok={statistics.mean(n_toks):.1f}")

    # category / difficulty / task breakdowns
    buckets = {}
    for i, row in enumerate(df.itertuples()):
        try:
            meta = json.loads(row.other_attributes or "{}")
        except Exception:
            meta = {}
        for key in ("category", "difficulty", "task"):
            v = str(meta.get(key, "?"))
            b = buckets.setdefault(key, {})
            s = b.setdefault(v, [0, 0])
            g = gts[i]
            s[1] += 1
            if matches[i]:
                s[0] += 1
    print("\n[mmau] per-difficulty/task/category accuracy (n>=10):")
    for key in ("category", "difficulty", "task"):
        for v, (c, t) in sorted(buckets[key].items(), key=lambda x: -x[1][1]):
            if t >= 10:
                print(f"  {key:12s} {v:12s} {c:4d}/{t:<4d} = {c/t*100:5.1f}%")

    mtag = "base" if args.no_lora else ("rl" + (args.ckpt.split("step_")[1] if "step_" in args.ckpt else "sft"))
    out = {
        "model": MODEL, "ckpt": "base" if args.no_lora else args.ckpt, "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens, "merge": args.merge,
        "correct": correct, "total": len(df), "accuracy": round(acc, 2),
        "strict_correct": strict_correct, "strict_accuracy": round(strict_acc, 2),
        "duplicate_equivalent_corrected": duplicate_corrected,
        "metric": "option accuracy with exact duplicate choices treated as equivalent",
        "method_dist": method_dist, "resp_block_hit": resp_hit,
        "truncated_rate": round(trunc_rate, 2), "avg_tokens": round(statistics.mean(n_toks), 1),
        "buckets": buckets,
        "records": [{"idx": i, "question": questions[i], "choices": all_choices[i],
                     "answer": answers[i], "metadata": all_metadata[i],
                     "gt": gts[i], "accepted_gt_labels": accepted_labels[i],
                     "pred": predictions[i], "method": methods[i],
                     "match": matches[i], "strict_match": strict_matches[i],
                     "duplicate_equivalent_match": duplicate_matches[i],
                     "score_reason": score_reasons[i], "n_tok": n_toks[i],
                     "truncated": truncated[i], "resp": responses[i]}
                    for i in range(len(df))],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[mmau] saved {args.out}")


if __name__ == "__main__":
    main()
