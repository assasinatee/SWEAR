#!/usr/bin/env python3
"""Calibrate per-difficulty output-length caps from the compress data.

The compression reward caps the model's output length by its SELF-rated <R>
level. The cap for level r should be the compress phase's actual per-difficulty
target length: `compression_meta.target_total` for items whose teacher
difficulty is r.

Difficulty axis (2026-08-13 FIX): use the RAW teacher `Reasoning_Demand`
(1-5) — NOT the derived D_R. D_R = 0.5·rd/5 + 0.5·is/5 floors at 0.2, so a
S_mean·4+1 level never assigns R1, even though RD=1 is 42.8% of the data (195k
items, the largest single class; mostly S2TT). `compression_meta.target_total`
is monotonic in RD (median 99 → 313), so it is the right difficulty→length map.

Usage:
  python scripts/analysis/calibrate_length_caps.py <compress_root> [-o caps.json] [--pct 85]
  → prints [R1..R5] caps; put them in config `compression.caps`.
"""
import argparse
import json
import os
import statistics
from collections import defaultdict

import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("compress_root", help="compress/result root (subset dirs of *.json)")
    ap.add_argument("-o", "--out", default=None, help="write caps.json")
    ap.add_argument("--pct", type=float, default=85, help="percentile per level (default 85)")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.compress_root, "*", "*.json"))
    by_rd = defaultdict(list)  # RD → target_total
    cnt = defaultdict(int)
    n_skip = 0
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if not isinstance(d, list):
            d = [d]
        for it in d:
            if it.get("error") or it.get("compression_error"):
                n_skip += 1
                continue
            cm = it.get("compression_meta") or {}
            tt = cm.get("target_total", 0)
            if not tt:
                continue
            rd = int((it.get("scores") or {}).get("Reasoning_Demand", 1))
            by_rd[rd].append(tt)
            cnt[rd] += 1

    caps = []
    print(f"files: {len(files)}  skipped: {n_skip}  percentile: {args.pct}")
    for lv in range(1, 6):
        vals = sorted(by_rd[lv])
        if not vals:
            caps.append(0)
            print(f"  R{lv}: no items")
            continue
        k = max(0, int(round((args.pct / 100.0) * (len(vals) - 1))))
        cap = vals[k]
        caps.append(cap)
        print(f"  R{lv}: n={len(vals):>7}  p{args.pct:.0f}={cap:5d}  (median={statistics.median(vals):5.0f})")

    print(f"\ncaps = {caps}")
    print("→ 填进 config 的 `compression.caps`（保留合理下限，例如 < 100 的设 100）")
    if args.out:
        json.dump({"caps": caps, "pct": args.pct, "n": dict(cnt)}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
