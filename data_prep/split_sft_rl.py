#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT / RL 数据切分：分层随机、非重叠、可复现。

数据：compress/result/<subset>/*.json（每条含 compressed_answer + scores）
分层：(subset, 难度桶)。难度桶由 scores.Reasoning_Demand 划分：
      - bucket A: 1-2（低推理需求）
      - bucket B: 3  （中）
      - bucket C: 4-5（高推理需求）
每个 (subset, 桶) 内按 ratio 随机切 → SFT / RL，保证两边子集与难度分布一致。

⚠️ 音频对齐：不直接过滤 JSON（会破坏与 CoTA parquet 的按 idx 对齐）。
   输出的是样本 ID 清单，prepare_full_data.py 需加 --include_ids 白名单，
   遍历完整 JSON（对齐保留）时跳过不在白名单的样本。

输出：
  <out>/sft_ids.jsonl   每行 {"id": "<subset>/<file>#<idx>"}
  <out>/rl_ids.jsonl
  <out>/split_summary.json  各子集/桶/切分统计

用法：
  python split_sft_rl.py --compress-root ... --out ... [--ratio 0.7] [--seed 42] [--dry-run]
"""
import json
import glob
import os
import sys
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
    return "B"  # 缺失分数默认中桶


def collect_samples(compress_root: str):
    """返回 [(sample_id, subset, bucket), ...]"""
    samples = []
    for sub in SUBSETS:
        for f in sorted(glob.glob(os.path.join(compress_root, sub, "*.json"))):
            try:
                data = json.load(open(f))
            except Exception:
                print(f"[warn] 无法解析 {f}，跳过")
                continue
            if not isinstance(data, list):
                continue
            fname = os.path.basename(f)
            for idx, s in enumerate(data):
                # 仅统计有压缩结果的样本
                ca = s.get("compressed_answer")
                if not (isinstance(ca, str) and ca.strip()):
                    continue
                sc = s.get("scores") or s.get("score") or {}
                samples.append((f"{sub}/{fname}#{idx}", sub, difficulty_bucket(sc)))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compress-root", type=str, default="/root/workspace/mount/share/liyixuan/audio_reasoning/compress/result")
    ap.add_argument("--out", type=str, default="/root/workspace/mount/share/liyixuan/audio_reasoning/split_output")
    ap.add_argument("--ratio", type=float, default=0.7, help="SFT 占比（默认 0.7）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed)
    samples = collect_samples(a.compress_root)
    print(f"压缩样本总数: {len(samples):,}")

    # 按 (subset, bucket) 分层
    strata = {}
    for sid, sub, bucket in samples:
        strata.setdefault((sub, bucket), []).append(sid)
    strata = {k: v for k, v in strata.items() if v}

    sft_ids, rl_ids = [], []
    print("\n=== 分层统计 ===")
    print(f"{'subset':<15}{'bucket':<6}{'总数':>8}{'SFT':>8}{'RL':>8}")
    for (sub, bucket), ids in sorted(strata.items()):
        random.shuffle(ids)
        n_sft = round(len(ids) * a.ratio)
        sft_ids += ids[:n_sft]
        rl_ids += ids[n_sft:]
        print(f"{sub:<15}{bucket:<6}{len(ids):>8}{n_sft:>8}{len(ids)-n_sft:>8}")

    print(f"\n切分结果: SFT={len(sft_ids):,}  RL={len(rl_ids):,}  合计={len(sft_ids)+len(rl_ids):,}")
    print(f"SFT 占比: {len(sft_ids)/(len(sft_ids)+len(rl_ids))*100:.1f}%")

    if a.dry_run:
        print("\n[dry-run] 不写文件。")
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

    print(f"\n已写入: {a.out}/{{sft_ids.jsonl, rl_ids.jsonl, split_summary.json}}")
    print("下一步：prepare_full_data.py 加 --include_ids 白名单后，用 sft_ids.jsonl 跑 SFT 数据。")


if __name__ == "__main__":
    main()
