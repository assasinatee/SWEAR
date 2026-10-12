#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stratified non-overlapping SFT / RL split over compressed-label JSONs.

Samples are stratified by (subset, difficulty bucket) from
scores.Reasoning_Demand (A: 1-2, B: 3, C: 4-5) and split by --ratio with a
fixed seed. Outputs sample-ID whitelists (sft_ids.jsonl / rl_ids.jsonl) rather
than filtered JSONs so alignment with the source parquets is preserved; pass
them to prepare_full_data.py via --include_ids.

Usage:
  python split_sft_rl.py --compress-root ... --out ... [--ratio 0.7] [--seed 42]
"""
import json
import glob
import os
import random
import argparse

SUBSETS = ["audiocap", "clotho", "complex_audio", "meld", "multispeaker", "musicbench", "S2TT"]


def difficulty_bucket(sc: dict) -> str:
    rd = sc.get("Reasoning_Demand")
    if isinstance(rd, (int, float)):
        if rd <= 2:
            return "A"
        if rd == 3:
            return "B"
        return "C"
    return "B"  # missing scores default to the middle bucket


def collect_samples(compress_root: str):
    """Return [(sample_id, subset, bucket), ...]."""
    samples = []
    for sub in SUBSETS:
        for f in sorted(glob.glob(os.path.join(compress_root, sub, "*.json"))):
            try:
                data = json.load(open(f))
            except Exception:
                print(f"[warn] skipping unparseable {f}")
                continue
            if not isinstance(data, list):
                continue
            fname = os.path.basename(f)
            for idx, s in enumerate(data):
                # only count samples with compressed answers
                ca = s.get("compressed_answer")
                if not (isinstance(ca, str) and ca.strip()):
                    continue
                sc = s.get("scores") or s.get("score") or {}
                samples.append((f"{sub}/{fname}#{idx}", sub, difficulty_bucket(sc)))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compress-root", type=str, default="./compress/result")
    ap.add_argument("--out", type=str, default="./split_output")
    ap.add_argument("--ratio", type=float, default=0.7, help="SFT fraction (default 0.7)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed)
    samples = collect_samples(a.compress_root)
    print(f"Total compressed samples: {len(samples):,}")

    # stratify by (subset, bucket)
    strata = {}
    for sid, sub, bucket in samples:
        strata.setdefault((sub, bucket), []).append(sid)
    strata = {k: v for k, v in strata.items() if v}

    sft_ids, rl_ids = [], []
    print("\n=== Stratified counts ===")
    print(f"{'subset':<15}{'bucket':<6}{'total':>8}{'SFT':>8}{'RL':>8}")
    for (sub, bucket), ids in sorted(strata.items()):
        random.shuffle(ids)
        n_sft = round(len(ids) * a.ratio)
        sft_ids += ids[:n_sft]
        rl_ids += ids[n_sft:]
        print(f"{sub:<15}{bucket:<6}{len(ids):>8}{n_sft:>8}{len(ids)-n_sft:>8}")

    print(f"\nSplit: SFT={len(sft_ids):,}  RL={len(rl_ids):,}  total={len(sft_ids)+len(rl_ids):,}")
    print(f"SFT fraction: {len(sft_ids)/(len(sft_ids)+len(rl_ids))*100:.1f}%")

    if a.dry_run:
        print("\n[dry-run] no files written.")
        return

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sft_ids.jsonl"), "w") as f:
        for sid in sft_ids:
            f.write(json.dumps({"id": sid}) + "\n")
    with open(os.path.join(a.out, "rl_ids.jsonl"), "w") as f:
        for sid in rl_ids:
            f.write(json.dumps({"id": sid}) + "\n")

    summary = {
        "total": len(sft_ids) + len(rl_ids),
        "sft": len(sft_ids),
        "rl": len(rl_ids),
        "ratio_sft": a.ratio,
        "seed": a.seed,
        "strata": {f"{s}/{b}": len(v) for (s, b), v in sorted(strata.items())},
    }
    with open(os.path.join(a.out, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nWritten: {a.out}/{{sft_ids.jsonl, rl_ids.jsonl, split_summary.json}}")
    print("Next: pass sft_ids.jsonl to prepare_full_data.py via --include_ids for SFT data.")


if __name__ == "__main__":
    main()
