#!/usr/bin/env python3
"""Build the immutable no-translation SFT / choice-only RL ID split.

The existing 70/30 assignment is preserved.  SFT drops every S2TT sample;
RL drops every S2TT sample and retains only records classified as ``choice``
by the already-isolated RL curriculum.  No source data is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


VERSION = "no_translation_sft_choice_rl_v3"
EXPECTED_SFT = 257_564
EXPECTED_RL = 51_308


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                item_id = value["id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid ID record at {path}:{line_number}") from exc
            if not isinstance(item_id, str) or "#" not in item_id:
                raise ValueError(f"invalid ID at {path}:{line_number}: {item_id!r}")
            ids.append(item_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {path}")
    return ids


def read_exclusions(path: Path) -> dict[str, dict]:
    exclusions: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                item_id = value["id"]
                reason = value["reason"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"invalid quality exclusion at {path}:{line_number}"
                ) from exc
            if not isinstance(item_id, str) or not isinstance(reason, str) or not reason:
                raise ValueError(f"invalid quality exclusion at {path}:{line_number}")
            if item_id in exclusions:
                raise ValueError(f"duplicate quality exclusion: {item_id}")
            exclusions[item_id] = value
    return exclusions


def read_rl_types(path: Path) -> tuple[dict[str, tuple[str, str]], Counter[str]]:
    records: dict[str, tuple[str, str]] = {}
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                source_file = value["source_file"]
                source_idx = int(value["source_idx"])
                type_name = str(value["type"]).casefold()
                subset = str(value["subset"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid curriculum record at {path}:{line_number}"
                ) from exc
            item_id = f"{source_file}#{source_idx}"
            if item_id in records:
                raise ValueError(f"duplicate curriculum source ID: {item_id}")
            records[item_id] = (subset, type_name)
            counts[f"{subset}/{type_name}"] += 1
    return records, counts


def atomic_write_jsonl(path: Path, ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for item_id in ids:
                handle.write(json.dumps({"id": item_id}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def subset_counts(ids: Iterable[str]) -> dict[str, int]:
    counts = Counter(item_id.split("/", 1)[0] for item_id in ids)
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-ids", type=Path, default=Path("split_output/sft_ids.jsonl"))
    parser.add_argument("--rl-ids", type=Path, default=Path("split_output/rl_ids.jsonl"))
    parser.add_argument(
        "--rl-curriculum",
        type=Path,
        default=Path("RL/data/curriculum/iso_k4/curriculum_data.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("split_output/no_translation_sft_choice_rl_v3"),
    )
    parser.add_argument(
        "--quality-exclusions",
        type=Path,
        default=Path("split_output/quality_exclusions_no_translation_v2.jsonl"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_paths = [
        output_dir / "sft_ids.jsonl",
        output_dir / "rl_ids.jsonl",
        output_dir / "manifest.json",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite immutable outputs: "
            + ", ".join(os.fspath(path) for path in existing)
        )

    source_sft = read_ids(args.sft_ids)
    source_rl = read_ids(args.rl_ids)
    exclusions = read_exclusions(args.quality_exclusions)
    if set(source_sft) & set(source_rl):
        raise SystemExit("source SFT and RL IDs overlap")

    rl_types, source_type_counts = read_rl_types(args.rl_curriculum)
    missing_rl = sorted(set(source_rl) - set(rl_types))
    unexpected_rl = sorted(set(rl_types) - set(source_rl))
    if missing_rl or unexpected_rl:
        raise SystemExit(
            "RL curriculum and source IDs differ: "
            f"missing={len(missing_rl)} unexpected={len(unexpected_rl)}"
        )

    source_union = set(source_sft) | set(source_rl)
    unknown_exclusions = sorted(set(exclusions) - source_union)
    if unknown_exclusions:
        raise SystemExit(f"quality exclusions not present in source split: {unknown_exclusions}")

    sft_ids = [
        item_id
        for item_id in source_sft
        if not item_id.startswith("S2TT/") and item_id not in exclusions
    ]
    rl_ids = [
        item_id
        for item_id in source_rl
        if (
            not item_id.startswith("S2TT/")
            and rl_types[item_id][1] == "choice"
            and item_id not in exclusions
        )
    ]

    if len(sft_ids) != EXPECTED_SFT:
        raise SystemExit(f"unexpected SFT count: {len(sft_ids)} != {EXPECTED_SFT}")
    if len(rl_ids) != EXPECTED_RL:
        raise SystemExit(f"unexpected RL count: {len(rl_ids)} != {EXPECTED_RL}")
    if set(sft_ids) & set(rl_ids):
        raise SystemExit("filtered SFT and RL IDs overlap")
    if any(item_id.startswith("S2TT/") for item_id in sft_ids + rl_ids):
        raise SystemExit("S2TT ID survived filtering")
    if any(rl_types[item_id][1] != "choice" for item_id in rl_ids):
        raise SystemExit("non-choice ID survived RL filtering")

    atomic_write_jsonl(output_paths[0], sft_ids)
    atomic_write_jsonl(output_paths[1], rl_ids)
    manifest = {
        "schema_version": "filtered_split_manifest.v1",
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "preserve_existing_split_assignment": True,
            "sft": "exclude source IDs whose subset is S2TT",
            "rl": "exclude S2TT and retain curriculum type == choice",
            "quality": "exclude the reviewed malformed compressed-answer denylist",
        },
        "sources": {
            "sft_ids": {
                "path": str(args.sft_ids.resolve()),
                "count": len(source_sft),
                "sha256": sha256_file(args.sft_ids),
            },
            "rl_ids": {
                "path": str(args.rl_ids.resolve()),
                "count": len(source_rl),
                "sha256": sha256_file(args.rl_ids),
            },
            "rl_curriculum": {
                "path": str(args.rl_curriculum.resolve()),
                "count": len(rl_types),
                "sha256": sha256_file(args.rl_curriculum),
                "type_counts": dict(sorted(source_type_counts.items())),
            },
            "quality_exclusions": {
                "path": str(args.quality_exclusions.resolve()),
                "count": len(exclusions),
                "sha256": sha256_file(args.quality_exclusions),
                "sft_count": len(set(source_sft) & set(exclusions)),
                "rl_count": len(set(source_rl) & set(exclusions)),
            },
        },
        "outputs": {
            "sft": {
                "path": str(output_paths[0].resolve()),
                "count": len(sft_ids),
                "removed": len(source_sft) - len(sft_ids),
                "quality_removed": len(set(source_sft) & set(exclusions)),
                "subset_counts": subset_counts(sft_ids),
            },
            "rl": {
                "path": str(output_paths[1].resolve()),
                "count": len(rl_ids),
                "removed": len(source_rl) - len(rl_ids),
                "quality_removed": len(set(source_rl) & set(exclusions)),
                "subset_counts": subset_counts(rl_ids),
            },
        },
        "validation": {
            "source_split_disjoint": True,
            "filtered_split_disjoint": True,
            "s2tt_rows": 0,
            "rl_non_choice_rows": 0,
            "quality_exclusions_remaining": len(
                (set(sft_ids) | set(rl_ids)) & set(exclusions)
            ),
        },
    }
    atomic_write_json(output_paths[2], manifest)

    # Record output hashes only after the deterministic payloads exist.
    manifest["outputs"]["sft"]["sha256"] = sha256_file(output_paths[0])
    manifest["outputs"]["rl"]["sha256"] = sha256_file(output_paths[1])
    atomic_write_json(output_paths[2], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
