#!/usr/bin/env python3
"""Build B+ curriculum JSONL files from a source curriculum.

For each K, publishes curriculum_data.jsonl + manifest.json atomically (a
completed temp directory is renamed into place; existing output is never
overwritten). Default mode keeps closed-answer items only; --include-open
opts into full data with explicit expected counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from audit_closed_difficulty import (  # noqa: E402
    AuditItem,
    _bucket_report,
    _stratum_spread,
    bplus_stratum,
    build_buckets,
    load_items,
    partition_integrity,
    task_family,
)


MANIFEST_SCHEMA_VERSION = "bplus_curriculum_manifest"
CURRICULUM_VERSION = "bplus_closed_drms"
DEFAULT_K = (1, 3, 4)
DEFAULT_EXPECTED_INPUT_COUNT = 172_498
DEFAULT_EXPECTED_CLOSED_COUNT = 113_410
DEFAULT_EXPECTED_INPUT_SHA256 = (
    "d583ce9cdd299d05c9ec63ae3e374e3ec0d5be4115bebccc85c76b96d7621467"
)
OUTPUT_JSONL = "curriculum_data.jsonl"
OUTPUT_MANIFEST = "manifest.json"


class InputValidationError(ValueError):
    """Raised before publication when the source violates the B+ contract."""


def _parse_ks(values: Sequence[str]) -> List[int]:
    parsed: List[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                parsed.append(int(token))
    return parsed


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _lexists(path: Path) -> bool:
    """Like Path.exists(), but also reject a dangling symlink."""

    return os.path.lexists(os.fspath(path))


def _stable_source_ties(items: Sequence[AuditItem]) -> List[AuditItem]:
    """Use source-ID-only hashes so tie assignments survive input reordering."""

    return [
        replace(
            item,
            tie_key=hashlib.sha256(item.item_id.encode("utf-8")).hexdigest(),
        )
        for item in items
    ]


def _validate_source(
    input_path: Path,
    expected_input_sha256: Optional[str],
    expected_input_count: Optional[int],
    expected_closed_count: Optional[int],
    include_open: bool = False,
) -> Tuple[List[AuditItem], dict, dict]:
    items, input_info, validation = load_items(
        input_path, closed_only=not include_open,
    )
    faults: List[str] = []

    for field in ("blank_lines", "json_errors", "non_object_rows"):
        if validation[field]:
            faults.append(f"{field}={validation[field]}")

    if validation["invalid_difficulty_rows"]:
        faults.append(
            f"invalid_difficulty_rows={validation['invalid_difficulty_rows']}"
        )
    if validation["valid_difficulty_rows"] != validation["selected_rows"]:
        faults.append(
            "valid_difficulty_rows does not equal selected_rows "
            f"({validation['valid_difficulty_rows']} != {validation['selected_rows']})"
        )

    # difficulty_norm is recomputed, so a stale stored value is not an error.
    unexpected_missing = {
        key: count
        for key, count in validation["missing_fields"].items()
        if key != "difficulty_norm" and count
    }
    unexpected_invalid = {
        key: count
        for key, count in validation["invalid_fields"].items()
        if not key.startswith("difficulty_norm:") and count
    }
    if unexpected_missing:
        faults.append(f"missing required fields={unexpected_missing}")
    if unexpected_invalid:
        faults.append(f"invalid required fields={unexpected_invalid}")

    duplicate_ids = validation["duplicate_source_ids"]
    if duplicate_ids["distinct_ids"]:
        faults.append(
            "duplicate source IDs are forbidden "
            f"(distinct={duplicate_ids['distinct_ids']}, extra={duplicate_ids['extra_rows']})"
        )

    if expected_input_sha256 is not None:
        normalized_expected_sha = expected_input_sha256.strip().casefold()
        if len(normalized_expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_expected_sha
        ):
            faults.append(f"expected input SHA-256 is malformed: {expected_input_sha256!r}")
        elif input_info["sha256"] != normalized_expected_sha:
            faults.append(
                f"input_sha256={input_info['sha256']}, expected={normalized_expected_sha}"
            )
    if expected_input_count is not None and validation["input_lines"] != expected_input_count:
        faults.append(
            f"input_lines={validation['input_lines']}, expected={expected_input_count}"
        )
    if expected_closed_count is not None and validation["selected_rows"] != expected_closed_count:
        faults.append(
            f"selected_rows={validation['selected_rows']}, expected={expected_closed_count}"
        )
    if include_open:
        if validation["parsed_rows"] != validation["selected_rows"]:
            faults.append("full-data selection did not retain every parsed row")
    elif (
        validation["parsed_rows"]
        != validation["excluded_non_closed_rows"] + validation["selected_rows"]
    ):
        faults.append("parsed rows do not partition exactly into open and closed rows")
    if not items:
        faults.append("no valid selected rows")

    if faults:
        raise InputValidationError("source curriculum failed validation: " + "; ".join(faults))
    return _stable_source_ties(items), input_info, validation


def _build_assignments(
    items: Sequence[AuditItem], ks: Sequence[int]
) -> Dict[int, dict]:
    plans: Dict[int, dict] = {}
    for k in ks:
        buckets = build_buckets(
            items,
            metric_name="continuous_rms",
            k=k,
            scheme="subset-task-stratified",
        )
        integrity = partition_integrity(items, buckets)
        if not integrity["ok"]:
            raise RuntimeError(f"K={k} produced an invalid partition: {integrity}")

        assignment = [-1] * len(items)
        for bucket_id, bucket in enumerate(buckets):
            for item in bucket:
                if assignment[item.ordinal] != -1:
                    raise RuntimeError(f"duplicate assignment for ordinal {item.ordinal}")
                assignment[item.ordinal] = bucket_id
        if any(bucket_id < 0 for bucket_id in assignment):
            raise RuntimeError(f"K={k} left rows unassigned")

        stratum_totals = Counter(bplus_stratum(item) for item in items)
        plans[k] = {
            "assignment": assignment,
            "integrity": integrity,
            "bucket_sizes": [len(bucket) for bucket in buckets],
            "stratum_totals": dict(sorted(stratum_totals.items())),
            "stratum_balance": _stratum_spread(buckets, bplus_stratum),
            "bucket_reports": [
                _bucket_report(bucket, "continuous_rms", bucket_id)
                for bucket_id, bucket in enumerate(buckets)
            ],
        }
    return plans


def _strict_source_id(row: Mapping[str, object], line_number: int) -> str:
    source_file = row.get("source_file")
    source_idx = row.get("source_idx")
    if not isinstance(source_file, str) or not source_file.strip():
        raise RuntimeError(f"source_file became invalid on second pass at line {line_number}")
    if source_idx in (None, ""):
        raise RuntimeError(f"source_idx became invalid on second pass at line {line_number}")
    return f"{source_file}#{source_idx}"


def _manifest(
    *,
    curriculum_version: str,
    k: int,
    input_info: Mapping[str, object],
    validation: Mapping[str, object],
    plan: Mapping[str, object],
    final_jsonl: Path,
    output_sha256: str,
    output_size_bytes: int,
    output_records: int,
    include_open: bool,
) -> dict:
    open_rows = sum(
        int(count)
        for stratum, count in plan["stratum_totals"].items()
        if str(stratum).startswith("open|")
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "curriculum_version": curriculum_version,
        "input": dict(input_info),
        "output": {
            "path": str(final_jsonl.resolve()),
            "sha256": output_sha256,
            "size_bytes": output_size_bytes,
            "records": output_records,
        },
        "parameters": {
            "K": k,
            "selection": "full" if include_open else "closed",
            "closed_rule": "subset.casefold() == 's2tt' OR type.casefold() == 'choice'",
            "difficulty_field": "difficulty_norm",
            "difficulty_formula": "sqrt((D_P^2 + D_C^2 + D_R^2) / 3)",
            "difficulty_axis_contract": "D_P, D_C, D_R must each be finite and in [0, 1]",
            "strata_rule": (
                "S2TT is one stratum; each non-S2TT family is keyed by raw subset+type"
            ),
            "bucket_rule": (
                "stable equal-frequency split within every stratum; each stratum has "
                "max bucket-count spread <= 1; remainders favor currently smallest "
                "global buckets with stable stratum-hash rotation"
            ),
            "stable_tie_break": (
                "sha256(UTF-8 source_file#source_idx); duplicate source IDs rejected"
            ),
            "implementation": "RL/scripts/analysis/audit_closed_difficulty.py",
            "scheme": "subset-task-stratified (B+)",
        },
        "counts": {
            "input_rows": validation["input_lines"],
            "excluded_open_rows": validation["excluded_non_closed_rows"],
            "selected_rows": validation["selected_rows"],
            "closed_rows": output_records - open_rows,
            "open_rows": open_rows,
            "output_rows": output_records,
        },
        "input_validation": dict(validation),
        "partition_integrity": plan["integrity"],
        "bucket_sizes": plan["bucket_sizes"],
        "stratum_totals": plan["stratum_totals"],
        "stratum_balance": plan["stratum_balance"],
        "buckets": plan["bucket_reports"],
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_bplus_curricula(
    input_path: Path,
    output_root: Path,
    ks: Sequence[int] = DEFAULT_K,
    *,
    expected_input_sha256: Optional[str] = DEFAULT_EXPECTED_INPUT_SHA256,
    expected_input_count: Optional[int] = DEFAULT_EXPECTED_INPUT_COUNT,
    expected_closed_count: Optional[int] = DEFAULT_EXPECTED_CLOSED_COUNT,
    curriculum_version: str = CURRICULUM_VERSION,
    include_open: bool = False,
) -> dict:
    """Build one or more K variants and return a concise publication summary."""

    source = Path(input_path).resolve()
    root = Path(output_root).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input curriculum does not exist: {source}")

    clean_ks = sorted(set(int(k) for k in ks))
    if not clean_ks or any(k <= 0 for k in clean_ks):
        raise ValueError(f"K values must be positive: {clean_ks}")
    if include_open and curriculum_version == CURRICULUM_VERSION:
        raise ValueError(
            "--include-open requires an explicit non-closed curriculum version"
        )

    # Complete all expensive source validation and assignment work before
    # creating any destination path.
    items, input_info, validation = _validate_source(
        source,
        expected_input_sha256=expected_input_sha256,
        expected_input_count=expected_input_count,
        expected_closed_count=expected_closed_count,
        include_open=include_open,
    )
    plans = _build_assignments(items, clean_ks)

    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise NotADirectoryError(f"output root is not a directory: {root}")

    slots: Dict[int, dict] = {}
    acquired_locks: List[Path] = []
    try:
        # Reserve all requested variants up front.  This also means an existing
        # K prevents the entire invocation before any new K is written.
        for k in clean_ks:
            final_dir = root / f"k{k}"
            lock_dir = root / f".k{k}.lock"
            if _lexists(final_dir):
                raise FileExistsError(f"refusing to overwrite existing output: {final_dir}")
            if _lexists(lock_dir):
                raise FileExistsError(f"another build appears active: {lock_dir}")
            lock_dir.mkdir()
            acquired_locks.append(lock_dir)
            if _lexists(final_dir):
                raise FileExistsError(f"output appeared while acquiring lock: {final_dir}")

        for k in clean_ks:
            temp_dir = Path(tempfile.mkdtemp(prefix=f".k{k}.tmp-", dir=root))
            slots[k] = {
                "temp_dir": temp_dir,
                "final_dir": root / f"k{k}",
                "jsonl": temp_dir / OUTPUT_JSONL,
                "manifest": temp_dir / OUTPUT_MANIFEST,
                "sha256": hashlib.sha256(),
                "size_bytes": 0,
                "records": 0,
            }

        second_pass_sha = hashlib.sha256()
        selected_index = 0
        with ExitStack() as stack:
            for k in clean_ks:
                slots[k]["handle"] = stack.enter_context(slots[k]["jsonl"].open("xb"))

            with source.open("rb") as source_handle:
                for line_number, raw_line in enumerate(source_handle, 1):
                    second_pass_sha.update(raw_line)
                    if not raw_line.strip():
                        raise RuntimeError(f"input changed: blank line {line_number} on second pass")
                    row = json.loads(raw_line)
                    if not isinstance(row, dict):
                        raise RuntimeError(
                            f"input changed: non-object row {line_number} on second pass"
                        )

                    subset_raw = row.get("subset")
                    type_raw = row.get("type")
                    subset = str(subset_raw) if subset_raw not in (None, "") else "<missing>"
                    type_name = str(type_raw) if type_raw not in (None, "") else "<missing>"
                    if not include_open and task_family(subset, type_name) == "open":
                        continue
                    if selected_index >= len(items):
                        raise RuntimeError("input gained selected rows between passes")
                    item = items[selected_index]
                    item_id = _strict_source_id(row, line_number)
                    if (
                        item_id != item.item_id
                        or subset != item.subset
                        or type_name != item.type_name
                    ):
                        raise RuntimeError(
                            "input ordering/content changed between passes at selected row "
                            f"{selected_index}: expected {item.item_id}, found {item_id}"
                        )

                    for k in clean_ks:
                        output_row = dict(row)
                        output_row.update(
                            {
                                "bucket_id": plans[k]["assignment"][item.ordinal],
                                "difficulty_norm": item.continuous_rms,
                                "bplus_stratum": bplus_stratum(item),
                                "curriculum_version": curriculum_version,
                            }
                        )
                        encoded = (
                            json.dumps(
                                output_row,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        ).encode("utf-8")
                        target = slots[k]
                        target["handle"].write(encoded)
                        target["sha256"].update(encoded)
                        target["size_bytes"] += len(encoded)
                        target["records"] += 1
                    selected_index += 1

            if second_pass_sha.hexdigest() != input_info["sha256"]:
                raise RuntimeError("input SHA-256 changed between validation and copy passes")
            if selected_index != len(items):
                raise RuntimeError(
                    f"second pass copied {selected_index} selected rows; expected {len(items)}"
                )
            for k in clean_ks:
                handle = slots[k]["handle"]
                handle.flush()
                os.fsync(handle.fileno())

        for k in clean_ks:
            target = slots[k]
            if target["records"] != len(items):
                raise RuntimeError(
                    f"K={k} wrote {target['records']} rows; expected {len(items)}"
                )
            final_jsonl = target["final_dir"] / OUTPUT_JSONL
            manifest = _manifest(
                curriculum_version=curriculum_version,
                k=k,
                input_info=input_info,
                validation=validation,
                plan=plans[k],
                final_jsonl=final_jsonl,
                output_sha256=target["sha256"].hexdigest(),
                output_size_bytes=target["size_bytes"],
                output_records=target["records"],
                include_open=include_open,
            )
            with target["manifest"].open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(target["temp_dir"])

        # A directory rename publishes the JSONL and its matching manifest as
        # one visible unit for each K.
        for k in clean_ks:
            target = slots[k]
            if _lexists(target["final_dir"]):
                raise FileExistsError(
                    f"refusing to overwrite output created during build: {target['final_dir']}"
                )
            os.rename(target["temp_dir"], target["final_dir"])
        _fsync_directory(root)
    finally:
        for target in slots.values():
            temp_dir = target["temp_dir"]
            if _lexists(temp_dir):
                shutil.rmtree(temp_dir)
        for lock_dir in reversed(acquired_locks):
            if _lexists(lock_dir):
                lock_dir.rmdir()

    outputs = {}
    for k in clean_ks:
        final_dir = root / f"k{k}"
        final_jsonl = final_dir / OUTPUT_JSONL
        final_manifest = final_dir / OUTPUT_MANIFEST
        outputs[str(k)] = {
            "directory": str(final_dir),
            "curriculum": str(final_jsonl),
            "curriculum_sha256": _sha256_file(final_jsonl),
            "manifest": str(final_manifest),
            "manifest_sha256": _sha256_file(final_manifest),
            "records": len(items),
        }
    open_rows = sum(1 for item in items if item.family == "open")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "input_sha256": input_info["sha256"],
        "selection": "full" if include_open else "closed",
        "selected_rows": len(items),
        "closed_rows": len(items) - open_rows,
        "open_rows": open_rows,
        "outputs": outputs,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source curriculum JSONL")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/curriculum_bplus"),
        help=(
            "parent directory; each K is published under k<K>/ "
            "(default: data/curriculum_bplus)"
        ),
    )
    parser.add_argument(
        "--K",
        dest="ks",
        nargs="+",
        default=[str(k) for k in DEFAULT_K],
        help="positive K values, space- or comma-separated (default: 1 3 4)",
    )
    parser.add_argument(
        "--expected-input-sha256",
        default=DEFAULT_EXPECTED_INPUT_SHA256,
        help="strict source SHA-256 (default: the reviewed 172,498-row source)",
    )
    parser.add_argument(
        "--expected-input-count",
        type=int,
        default=DEFAULT_EXPECTED_INPUT_COUNT,
        help=f"strict input line count (default: {DEFAULT_EXPECTED_INPUT_COUNT})",
    )
    parser.add_argument(
        "--expected-closed-count",
        "--expected-selected-count",
        dest="expected_closed_count",
        type=int,
        default=DEFAULT_EXPECTED_CLOSED_COUNT,
        help=(
            "strict post-selection count; defaults to the closed population "
            f"({DEFAULT_EXPECTED_CLOSED_COUNT})"
        ),
    )
    parser.add_argument(
        "--include-open",
        action="store_true",
        help="retain the full population, including non-S2TT open questions",
    )
    parser.add_argument(
        "--curriculum-version",
        default=CURRICULUM_VERSION,
        help="curriculum tag stored in output rows/manifests",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = build_bplus_curricula(
        input_path=args.input,
        output_root=args.output_root,
        ks=_parse_ks(args.ks),
        expected_input_sha256=args.expected_input_sha256,
        expected_input_count=args.expected_input_count,
        expected_closed_count=args.expected_closed_count,
        curriculum_version=args.curriculum_version,
        include_open=args.include_open,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
