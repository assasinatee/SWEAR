#!/usr/bin/env python3
"""MMAR evaluation for base / LoRA checkpoints.

Uses the same "which" prompt as MMAU, letter-labeled options, and
conservative final-block answer parsing. The benchmark answer is
independently mapped to its option label before the two labels are
compared (resolved-reference option accuracy). Official full-text and
RESPONSE-block string-match values are retained as diagnostics only.
"""

import argparse
import json
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CRYPTOGRAPHY_OPENSSL_NO_LEGACY", "1")

import librosa
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2_5OmniProcessor, GenerationConfig
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from peft import PeftModel

from baseline_answer_parser import PARSER_VERSION, resolve_choice
from mmar_answer_scoring import MMAR_SCORER_VERSION, score_mmar_answer

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Omni-7B")
SFT_CKPT = os.environ.get("SFT_CKPT", "")
MMAR_META = os.environ.get("MMAR_META", "MMAR/MMAR-meta.json")
MMAR_AUDIO = os.environ.get("MMAR_AUDIO_DIR", "MMAR/audio")
DEFAULT_OUT = os.environ.get("MMAR_OUT", "results/eval_mmar_unified.json")
DEFAULT_MAX_TOKENS = 1536

# Used only for the RESPONSE-block coverage diagnostic.  Prediction parsing is
# centralized in baseline_answer_parser.py.
_RESPONSE_TAG_RE = re.compile(r"<RESPONSE>(.*?)</RESPONSE>", re.IGNORECASE | re.DOTALL)


def tokenize(t):
    return set(re.findall(r"\b\w+\b", str(t).lower()))


def string_match(answer, prediction, choices):
    """Official MMAR string_match, kept verbatim as a diagnostic."""
    pt = tokenize(prediction)
    at = tokenize(answer)
    if not pt:
        return False, False, False
    incorrect = set()
    for ch in choices:
        ct = tokenize(ch)
        if ct != at:
            incorrect.update(ct - at)
    cond1 = at.issubset(pt)
    cond2 = pt.isdisjoint(incorrect)
    return cond1, cond2, (cond1 and cond2)


def extract_response_block(text):
    m = _RESPONSE_TAG_RE.search(text)
    return m.group(1).strip() if m else text


def to_list(x):
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        try:
            y = json.loads(x) if x.startswith("[") else eval(x)
            if isinstance(y, list):
                return [str(v) for v in y]
        except Exception:
            pass
    if isinstance(x, np.ndarray):
        return x.tolist()
    return [str(x)]


def build_which_prompt(item):
    """Which of the following is correct? {question} (a) ... (b) ...  (same as MMAU)."""
    q = item["question"]
    choices = list(item["choices"])
    parts = []
    for i, c in enumerate(choices):
        c_str = str(c).strip()
        m = re.match(r"\(([A-Z])\)\s*(.*)", c_str)
        if m:
            parts.append(f"({m.group(1).lower()}) {m.group(2).strip()}")
        else:
            parts.append(f"({chr(97 + i)}) {c_str}")
    return f"Which of the following is correct? {q} " + " ".join(parts)


def load_model(args):
    print(f"[mmar] model {args.model} + ckpt {args.ckpt if not args.no_lora else '(full/base)'}, "
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
            print(f"[mmar] manual-merged {merged} LoRA pairs (scale={scale})", flush=True)
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
    ap.add_argument("--merge", choices=["peft", "manual"], default="peft")
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--n", type=int, default=0, help="evaluate only the first n samples")
    ap.add_argument("--meta", default=MMAR_META, help="MMAR-meta.json path")
    ap.add_argument("--audio-dir", default=MMAR_AUDIO, help="MMAR audio directory")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    processor, model, gen = load_model(args)
    with open(args.meta) as f:
        meta = json.load(f)
    if args.n:
        meta = meta[:args.n]
    print(f"[mmar] {len(meta)} items", flush=True)

    records = []
    from collections import Counter
    method_dist = Counter()
    for idx, it in enumerate(meta):
        wav = os.path.join(args.audio_dir, os.path.basename(it["audio_path"]))
        if not os.path.exists(wav):
            print(f"[mmar] MISSING audio {wav}", flush=True)
            continue
        audio, _ = librosa.load(wav, sr=16000)
        prompt_text = build_which_prompt(it)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": prompt_text}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, audio=audio, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen)
        out_ids = out[0] if isinstance(out, (tuple, list)) else out
        resp = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        n_tok = int(out_ids.shape[1] - inputs["input_ids"].shape[1])

        choices = list(it["choices"])
        answer = it["answer"].strip()
        parsed = resolve_choice(resp, choices)
        rb = extract_response_block(resp)
        score = score_mmar_answer(parsed.label, answer, choices, rb)
        match, method = score.match, parsed.method
        matched_text = None
        if parsed.label is not None:
            selected_index = ord(parsed.label) - ord("A")
            if 0 <= selected_index < len(choices):
                matched_text = str(choices[selected_index])
        method_dist[method] += 1
        # diagnostics: official strict string_match (RESPONSE block / full text) + letter-to-choice
        c1, c2, m_full = string_match(answer, resp, choices)
        c1r, c2r, m_rb = string_match(answer, rb, choices)
        letter = parsed.label
        letter_match = False
        if letter:
            li = ord(letter) - ord("A")
            if li < len(choices):
                _, _, letter_match = string_match(answer, str(choices[li]), choices)

        records.append({
            "idx": idx, "id": it["id"], "modality": it["modality"], "category": it["category"],
            "sub_category": it.get("sub-category", None),
            "answer": answer, "choices": choices, "prediction": resp,
            "match": match, "method": method,
            "matched_text": matched_text,
            "reference_labels": list(score.reference_labels),
            "reference_mode": score.reference_mode,
            "score_reason": score.reason,
            "diag_official_full": m_full, "diag_official_respblk": m_rb,
            "diag_letter": letter, "diag_letter_match": letter_match,
            "n_tok": n_tok,
        })
        if idx % 50 == 0 or idx == len(meta) - 1:
            run_acc = sum(1 for r in records if r["match"]) / len(records) * 100
            print(f"  [{idx}/{len(meta)}] ans={answer!r} match={match} method={method} running_acc={run_acc:.1f}%", flush=True)

    n = len(records)
    corr = sum(1 for r in records if r["match"])
    print(f"\n{'*'*30}\nTotal Accuracy (option-matching): {corr}/{n} = {corr/n*100:.2f}%")
    print("Method distribution:")
    for m, c in method_dist.most_common():
        print(f"  {m:42s} {c} ({c/n*100:.1f}%)")
    for title, key in [("diag_official_full (official / full text)", "diag_official_full"),
                       ("diag_official_respblk (official / RESPONSE block)", "diag_official_respblk"),
                       ("diag_letter_match (letter-to-choice)", "diag_letter_match")]:
        c = sum(1 for r in records if r[key])
        print(f"  {title}: {c}/{n} = {c/n*100:.1f}%")

    def bucket(key):
        mp = {}
        for r in records:
            v = r[key]
            s = mp.setdefault(v, [0, 0])
            s[0] += r["match"]; s[1] += 1
        return mp
    for title, key in [("Modality", "modality"), ("Category", "category"), ("Sub-category", "sub_category")]:
        print(f"\n{'*'*30}\n{title}-wise Accuracy:")
        for k, (c, t) in sorted(bucket(key).items(), key=lambda x: -x[1][1]):
            print(f"  {k:40s} {c/t*100:.2f}% over {t} samples")

    out = {
        "model": args.model, "ckpt": "base" if args.no_lora else args.ckpt,
        "prompt": "which",
        "metric": "resolved-reference option accuracy + official string-match diagnostics",
        "parser_version": PARSER_VERSION, "scorer_version": MMAR_SCORER_VERSION,
        "correct": corr, "total": n, "accuracy": round(corr / n * 100, 2),
        "method_dist": dict(method_dist), "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[mmar] saved {args.out}")


if __name__ == "__main__":
    main()
