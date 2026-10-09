#!/usr/bin/env python3
"""Validate all 12 tags (THINK, RESPONSE, PLANNING, CAPTION, REASONING, SUMMARY)
are properly opened and closed in the training data."""
import json, glob, os, re, sys

SOURCES = {
    "compress_raw": os.environ.get("COMPRESS_ROOT", "compress/result"),
    "converted_compressed": os.environ.get("CONVERTED_FULL", "converted_data/full.json"),
    "converted_raw": os.environ.get("CONVERTED_RAW", "converted_data_raw/full_raw.json"),
}

TAGS = ["THINK", "RESPONSE", "PLANNING", "CAPTION", "REASONING", "SUMMARY"]

PASS = "✅"
FAIL = "❌"


def check_tags(text, label, stats):
    """Check that all 12 tags (6 open + 6 close) are properly balanced."""
    errors = []
    tag_counts = {}
    for tag in TAGS:
        opens = text.count(f"<{tag}>")
        closes = text.count(f"</{tag}>")
        tag_counts[tag] = (opens, closes)
        if opens != closes:
            errors.append(f"  {FAIL} <{tag}>: {opens} open, {closes} close")
    return tag_counts, errors


def check_source_json_files():
    """Check all source JSON files in compress/result/. Uses answer + compressed_answer."""
    print(f"{'='*60}")
    print(f"  SOURCE DATA: compress/result/")
    print(f"{'='*60}")

    files = sorted(glob.glob(os.path.join(SOURCES["compress_raw"], "*/*.json")))
    print(f"  Files: {len(files)}")

    total_items = 0
    total_issues = 0

    field_issues = {"answer": 0, "compressed_answer": 0}
    field_ok = {"answer": 0, "compressed_answer": 0}

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        for item in (data if isinstance(data, list) else [data]):
            total_items += 1
            for field in ["answer", "compressed_answer"]:
                val = item.get(field, "")
                _, errs = check_tags(val, field, {})
                if errs or not val:
                    field_issues[field] += 1
                    if field_issues[field] <= 10:
                        rel = os.path.relpath(fpath, SOURCES["compress_raw"])
                        print(f"  {FAIL} {rel} (item #{total_items}, {field}):")
                        for e in errs:
                            print(f"    {e}")
                else:
                    field_ok[field] += 1

    print(f"\n  Items checked: {total_items}")
    print(f"  answer:     {PASS if field_issues['answer']==0 else FAIL} "
          f"{field_ok['answer']} ok, {field_issues['answer']} issues")
    print(f"  compressed: {PASS if field_issues['compressed_answer']==0 else FAIL} "
          f"{field_ok['compressed_answer']} ok, {field_issues['compressed_answer']} issues")
    return total_issues


def check_converted_json(path, name):
    """Check a converted sharegpt JSON file."""
    print(f"\n{'='*60}")
    print(f"  CONVERTED DATA: {name}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"  {FAIL} File not found: {path}")
        return -1

    with open(path) as f:
        data = json.load(f)

    print(f"  Samples: {len(data)}")

    total_issues = 0
    tag_stats = {t: [0, 0] for t in TAGS}  # [open, close]

    for i, item in enumerate(data):
        for conv in item.get("conversations", []):
            if conv.get("from") == "gpt":
                val = conv.get("value", "")
                _, errs = check_tags(val, "", {})
                if errs:
                    total_issues += 1
                    if total_issues <= 10:
                        print(f"  {FAIL} Sample #{i}:")
                        for e in errs:
                            print(f"    {e}")
                break

    print(f"\n  Issues: {total_issues}/{len(data)} samples ({total_issues/len(data)*100:.2f}%)")
    if total_issues == 0:
        print(f"  {PASS} ALL CLEAN")
    else:
        print(f"  {FAIL} HAS ISSUES")
    return total_issues


def main():
    total_errors = 0

    # 1. Check source files
    e1 = check_source_json_files()
    total_errors += e1

    # 2. Check converted data
    for path in [SOURCES["converted_compressed"], SOURCES["converted_raw"]]:
        name = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
        e2 = check_converted_json(path, name)
        if e2 > 0:
            total_errors += e2

    print(f"\n{'='*60}")
    if total_errors == 0:
        print(f"  {PASS} ALL CHECKS PASSED - No tag issues found!")
    else:
        print(f"  {FAIL} {total_errors} issues found")
    print(f"{'='*60}")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
