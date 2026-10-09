#!/usr/bin/env python3
"""Calibrate model-token compression caps from a curriculum JSONL.

The calibration target is the compressed answer that initialized the policy.
Only the bodies of PLANNING, CAPTION, REASONING, and SUMMARY are counted;
section tags, rating tags, RESPONSE, and tokenizer-added special tokens are
excluded. By default difficulty uses the RMS of the three SFT-discretized
teacher ratings, exactly matching the online P/C/R tags; continuous teacher
RMS remains available as an explicit diagnostic mode. Quintiles are computed
independently for each task family.

The script is read-only unless ``--out`` is supplied.  Output files are opened
exclusively and are never overwritten.

Example (closed-only is the default)::

    python scripts/analysis/calibrate_token_caps.py \
      data/curriculum/iso_k4/curriculum_data.jsonl \
      --tokenizer Qwen/Qwen2.5-Omni-7B

Use ``--include-open`` to calibrate the open family as well.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "token_caps"
SECTIONS: Tuple[str, ...] = (
    "PLANNING",
    "CAPTION",
    "REASONING",
    "SUMMARY",
)
RATING_SECTIONS: Tuple[str, ...] = SECTIONS[:3]
FAMILY_ORDER: Tuple[str, ...] = ("s2tt", "choice", "open")
DIFFICULTY_FORMULAS = {
    "continuous-rms": "sqrt((D_P**2 + D_C**2 + D_R**2) / 3)",
    "discrete-rms": (
        "sqrt(mean(((sft_rating(D_i)-1)/4)**2 for "
        "D_i in (D_P,D_C,D_R)))"
    ),
}
DIFFICULTY_FORMULA = DIFFICULTY_FORMULAS["continuous-rms"]

_SECTION_PATTERNS = {
    section: re.compile(
        rf"<\s*{re.escape(section)}(?:\s+[^>]*)?\s*>(.*?)"
        rf"</\s*{re.escape(section)}\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for section in SECTIONS
}
_MARKUP_TAG_RE = re.compile(r"<[^>\n]+>")
_RESPONSE_BLOCK_RE = re.compile(
    r"<\s*RESPONSE(?:\s+[^>]*)?\s*>.*?(?:</\s*RESPONSE\s*>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def classify_family(record: Mapping[str, Any]) -> str:
    """Match the training-time family routing: S2TT, choice, then open."""
    subset = str(record.get("subset") or "").strip().lower()
    item_type = str(record.get("type") or "").strip().lower()
    if subset == "s2tt":
        return "s2tt"
    if item_type == "choice":
        return "choice"
    return "open"


def difficulty_rms(
    record: Mapping[str, Any], mode: str = "continuous-rms",
) -> Optional[float]:
    """Return teacher-axis RMS in continuous or SFT-discretized form."""
    if mode not in DIFFICULTY_FORMULAS:
        raise ValueError(f"unknown difficulty mode: {mode}")
    nested = record.get("scores")
    scores = nested if isinstance(nested, Mapping) else {}
    values: List[float] = []
    for key in ("D_P", "D_C", "D_R"):
        raw = record.get(key, scores.get(key))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        if not 0.0 <= value <= 1.0:
            return None
        values.append(value)
    if mode == "discrete-rms":
        def _rating_unit(value: float) -> float:
            if value < 0.3:
                rating = 1
            elif value < 0.5:
                rating = 2
            elif value < 0.7:
                rating = 3
            elif value < 0.9:
                rating = 4
            else:
                rating = 5
            return (rating - 1) / 4.0
        values = [_rating_unit(value) for value in values]
    return math.sqrt(sum(value * value for value in values) / 3.0)


def sft_rating(value: float) -> int:
    """Map a continuous teacher axis to the exact R1..R5 SFT bins."""
    if value < 0.3:
        return 1
    if value < 0.5:
        return 2
    if value < 0.7:
        return 3
    if value < 0.9:
        return 4
    return 5


def teacher_axis(record: Mapping[str, Any], key: str) -> Optional[float]:
    nested = record.get("scores")
    scores = nested if isinstance(nested, Mapping) else {}
    try:
        value = float(record.get(key, scores.get(key)))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def extract_section_bodies(
    text: str,
    sections: Sequence[str] = SECTIONS,
) -> Dict[str, List[str]]:
    """Extract cleaned bodies for the requested sections.

    Any markup-looking tag inside a body (for example ``<R3>`` or an HF
    special token) is replaced with a space before tokenization.  RESPONSE is
    never searched, so its content cannot contribute to the cap.
    """
    # Normally RESPONSE is a sibling of THINK.  Remove it first anyway so a
    # malformed/nested RESPONSE block can never leak answer tokens into a
    # target section.
    target_text = _RESPONSE_BLOCK_RE.sub(" ", text or "")
    result: Dict[str, List[str]] = {}
    for section in sections:
        pattern = _SECTION_PATTERNS.get(section)
        if pattern is None:
            pattern = re.compile(
                rf"<\s*{re.escape(section)}(?:\s+[^>]*)?\s*>(.*?)"
                rf"</\s*{re.escape(section)}\s*>",
                flags=re.IGNORECASE | re.DOTALL,
            )
        bodies = []
        for match in pattern.finditer(target_text):
            body = _MARKUP_TAG_RE.sub(" ", match.group(1)).strip()
            bodies.append(body)
        result[section] = bodies
    return result


def count_section_tokens(
    text: str,
    tokenizer: Any,
    sections: Sequence[str] = SECTIONS,
) -> Tuple[int, Dict[str, int], int]:
    """Count model tokens in section bodies without adding special tokens.

    The tokenizer only needs an ``encode(text, add_special_tokens=False)``
    method, which keeps this function testable without transformers.
    Returns ``(total_tokens, per_section_tokens, sections_found)``.
    """
    bodies = extract_section_bodies(text, sections)
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    per_section: Dict[str, int] = {}
    sections_found = 0
    for section in sections:
        section_total = 0
        section_bodies = bodies.get(section, [])
        sections_found += len(section_bodies)
        for body in section_bodies:
            if not body:
                continue
            token_ids = tokenizer.encode(body, add_special_tokens=False)
            section_total += sum(token_id not in special_ids for token_id in token_ids)
        per_section[section] = section_total
    return sum(per_section.values()), per_section, sections_found


def linear_quantile(values: Sequence[float], q: float) -> float:
    """NumPy-style linear empirical quantile, implemented with stdlib only."""
    if not values:
        raise ValueError("linear_quantile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def nearest_rank_percentile(values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile; suitable for an integer upper-length cap."""
    if not values:
        raise ValueError("nearest_rank_percentile requires at least one value")
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    ordered = sorted(int(value) for value in values)
    rank = max(1, int(math.ceil(q * len(ordered))))
    return ordered[rank - 1]


def cumulative_max(values: Sequence[Optional[int]]) -> List[Optional[int]]:
    """Fill empty bins from the nearest lower/upper bin, then monotonicize."""
    if not any(value is not None for value in values):
        return [None for _ in values]

    filled = list(values)
    first_observed = next(value for value in filled if value is not None)
    running: Optional[int] = first_observed
    for index, value in enumerate(filled):
        if value is None:
            filled[index] = running
        else:
            running = value

    result: List[Optional[int]] = []
    running: Optional[int] = None
    for value in filled:
        if value is not None:
            running = value if running is None else max(running, value)
        result.append(running)
    return result


def calibrate_family(
    samples: Sequence[Tuple[float, int]],
    p85_q: float = 0.85,
) -> Dict[str, Any]:
    """Calibrate five value-based difficulty bins for one task family."""
    if not samples:
        return {
            "n": 0,
            "difficulty_inner_boundaries": [None, None, None, None],
            "raw_p85_tokens": [None, None, None, None, None],
            "monotonic_caps": [None, None, None, None, None],
            "bins": [
                {
                    "bin": index + 1,
                    "n": 0,
                    "raw_p85_tokens": None,
                    "monotonic_cap_tokens": None,
                    "difficulty_min": None,
                    "difficulty_max": None,
                }
                for index in range(5)
            ],
        }

    difficulties = [float(difficulty) for difficulty, _ in samples]
    boundaries = [
        linear_quantile(difficulties, q) for q in (0.2, 0.4, 0.6, 0.8)
    ]
    bin_samples: List[List[Tuple[float, int]]] = [[] for _ in range(5)]
    for difficulty, token_count in samples:
        # Equality goes to the higher bin: [b0,b1), [b1,b2), ...
        bin_index = bisect.bisect_right(boundaries, float(difficulty))
        bin_samples[bin_index].append((float(difficulty), int(token_count)))

    raw_p85: List[Optional[int]] = []
    for items in bin_samples:
        lengths = [token_count for _, token_count in items]
        raw_p85.append(
            nearest_rank_percentile(lengths, p85_q) if lengths else None
        )
    caps = cumulative_max(raw_p85)

    bins = []
    for index, items in enumerate(bin_samples):
        observed_difficulties = [difficulty for difficulty, _ in items]
        bins.append({
            "bin": index + 1,
            "n": len(items),
            "raw_p85_tokens": raw_p85[index],
            "monotonic_cap_tokens": caps[index],
            "difficulty_min": (
                float(min(observed_difficulties))
                if observed_difficulties else None
            ),
            "difficulty_max": (
                float(max(observed_difficulties))
                if observed_difficulties else None
            ),
        })

    return {
        "n": len(samples),
        # Preserve exact Python floats. Online discrete RMS values sit exactly
        # on many boundaries, so decimal rounding could send a rating triplet
        # to a different side of bisect_right than the calibration sample.
        "difficulty_inner_boundaries": [float(value) for value in boundaries],
        "raw_p85_tokens": raw_p85,
        "monotonic_caps": caps,
        "bins": bins,
    }


def calibrate_rating_caps(
    samples_by_rating: Sequence[Sequence[int]], p85_q: float = 0.85,
) -> Dict[str, Any]:
    """Calibrate monotonic P85 caps for the five emitted rating levels."""
    if len(samples_by_rating) != 5:
        raise ValueError("samples_by_rating must contain exactly five bins")
    raw_caps: List[Optional[int]] = [
        nearest_rank_percentile(values, p85_q) if values else None
        for values in samples_by_rating
    ]
    return {
        "counts": [len(values) for values in samples_by_rating],
        "raw_p85_tokens": raw_caps,
        "monotonic_caps": cumulative_max(raw_caps),
    }


def write_json_exclusive(path: str, payload: Mapping[str, Any]) -> None:
    """Write JSON to a new file, raising FileExistsError rather than overwrite."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_tokenizer(args: argparse.Namespace) -> Any:
    # Lazy import keeps the pure functions and their tests transformers-free.
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise RuntimeError(
            "transformers is required only for the CLI tokenizer load; "
            "run this script inside the project SIF"
        ) from exc

    return AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast,
        local_files_only=not args.allow_download,
    )


def _tokenizer_metadata(tokenizer: Any, args: argparse.Namespace) -> Dict[str, Any]:
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    metadata: Dict[str, Any] = {
        "requested": args.tokenizer,
        "resolved_name_or_path": str(
            getattr(tokenizer, "name_or_path", args.tokenizer)
        ),
        "class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "special_token_ids_excluded": len(
            set(getattr(tokenizer, "all_special_ids", None) or [])
        ),
        "revision": args.revision,
        "use_fast": bool(args.use_fast),
        "local_files_only": not args.allow_download,
    }
    commit_hash = init_kwargs.get("_commit_hash")
    if commit_hash:
        metadata["commit_hash"] = str(commit_hash)
    for key in ("vocab_size", "model_max_length"):
        value = getattr(tokenizer, key, None)
        if isinstance(value, (int, float, str, bool)) or value is None:
            metadata[key] = value
    return metadata


def build_calibration(
    curriculum_jsonl: str,
    tokenizer: Any,
    tokenizer_metadata: Mapping[str, Any],
    *,
    closed_only: bool = True,
    text_field: str = "compressed_answer",
    p85_q: float = 0.85,
    difficulty_mode: str = "discrete-rms",
    progress_every: int = 10_000,
) -> Dict[str, Any]:
    """Stream the JSONL and return the complete calibration payload."""
    included_families = {"s2tt", "choice"} if closed_only else set(FAMILY_ORDER)
    samples: Dict[str, List[Tuple[float, int]]] = {
        family: [] for family in FAMILY_ORDER
    }
    section_totals: Dict[str, Counter] = {
        family: Counter() for family in FAMILY_ORDER
    }
    section_samples = {
        family: {
            section: [[] for _ in range(5)] for section in RATING_SECTIONS
        }
        for family in FAMILY_ORDER
    }
    summary_samples: Dict[str, List[int]] = {
        family: [] for family in FAMILY_ORDER
    }
    stats: Counter = Counter()
    input_hash = hashlib.sha256()

    with open(curriculum_jsonl, "rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            input_hash.update(raw_line)
            if not raw_line.strip():
                stats["blank_lines"] += 1
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {curriculum_jsonl}:{line_number}: {exc}"
                ) from exc
            stats["records_read"] += 1
            if not isinstance(record, Mapping):
                stats["skipped_non_object"] += 1
                continue

            family = classify_family(record)
            stats[f"seen_{family}"] += 1
            if family not in included_families:
                stats["filtered_open"] += 1
                continue

            difficulty = difficulty_rms(record, mode=difficulty_mode)
            if difficulty is None:
                stats["skipped_invalid_difficulty"] += 1
                continue
            text = record.get(text_field)
            if not isinstance(text, str) or not text.strip():
                stats["skipped_missing_text"] += 1
                continue

            token_count, per_section, sections_found = count_section_tokens(
                text, tokenizer, SECTIONS
            )
            if sections_found == 0:
                stats["skipped_no_target_sections"] += 1
                continue

            samples[family].append((difficulty, token_count))
            section_totals[family].update(per_section)
            for section, axis in zip(
                RATING_SECTIONS, ("D_P", "D_C", "D_R")
            ):
                value = teacher_axis(record, axis)
                if value is not None:
                    section_samples[family][section][sft_rating(value) - 1].append(
                        int(per_section[section])
                    )
            summary_samples[family].append(int(per_section["SUMMARY"]))
            stats["records_used"] += 1
            stats[f"used_{family}"] += 1

            if progress_every > 0 and stats["records_read"] % progress_every == 0:
                print(
                    f"[calibrate_token_caps] read={stats['records_read']} "
                    f"used={stats['records_used']}",
                    file=sys.stderr,
                    flush=True,
                )

    family_payload: Dict[str, Any] = {}
    for family in FAMILY_ORDER:
        result = calibrate_family(samples[family], p85_q=p85_q)
        result["included"] = family in included_families
        result["section_token_totals"] = {
            section: int(section_totals[family][section]) for section in SECTIONS
        }
        result["section_wise"] = {
            "rating_caps": {
                section: calibrate_rating_caps(
                    section_samples[family][section], p85_q=p85_q,
                )
                for section in RATING_SECTIONS
            },
            "summary_fixed_cap": (
                nearest_rank_percentile(summary_samples[family], p85_q)
                if summary_samples[family] else None
            ),
            "summary_count": len(summary_samples[family]),
        }
        family_payload[family] = result

    stat_result = {key: int(value) for key, value in sorted(stats.items())}
    for key in (
        "records_read",
        "records_used",
        "filtered_open",
        "skipped_invalid_difficulty",
        "skipped_missing_text",
        "skipped_no_target_sections",
    ):
        stat_result.setdefault(key, 0)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {
            "path": os.path.abspath(curriculum_jsonl),
            "size_bytes": os.path.getsize(curriculum_jsonl),
            "sha256": input_hash.hexdigest(),
            "text_field": text_field,
            "closed_only": bool(closed_only),
            "included_families": [
                family for family in FAMILY_ORDER if family in included_families
            ],
            # Record what was actually observed as well as the CLI's allowed
            # family set.  A choice-only file calibrated with --closed-only
            # permits S2TT by policy but contains none; this field makes that
            # provenance explicit instead of suggesting S2TT data was used.
            "observed_families": [
                family
                for family in FAMILY_ORDER
                if stats.get(f"seen_{family}", 0) > 0
            ],
        },
        "tokenizer": dict(tokenizer_metadata),
        "method": {
            "difficulty_mode": difficulty_mode,
            "difficulty_formula": DIFFICULTY_FORMULAS[difficulty_mode],
            "difficulty_axes": ["D_P", "D_C", "D_R"],
            "difficulty_binning": (
                "independent value-based quintiles within each family; "
                "linear empirical 20/40/60/80% boundaries; boundary ties go "
                "to the higher bin"
            ),
            "sections": list(SECTIONS),
            "section_extraction": (
                "sum tokenizer.encode(body, add_special_tokens=False) over "
                "PLANNING/CAPTION/REASONING/SUMMARY; strip markup tags; "
                "exclude RESPONSE"
            ),
            "raw_cap_percentile": p85_q * 100.0,
            "raw_cap_percentile_method": "nearest-rank",
            "section_wise_binning": (
                "PLANNING/CAPTION/REASONING use their matching D_P/D_C/D_R "
                "SFT rating bins; SUMMARY uses one family-specific fixed cap"
            ),
            "monotonicization": "left-to-right cumulative maximum of raw P85 caps",
            "empty_bin_fill": (
                "forward-fill from lower bin; leading empty bins use first "
                "observed cap before cumulative maximum"
            ),
        },
        "counts": stat_result,
        "families": family_payload,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curriculum_jsonl", help="existing curriculum_data.jsonl")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer repo ID or local model/tokenizer directory",
    )
    parser.add_argument(
        "--text-field",
        default="compressed_answer",
        help="curriculum text field to measure (default: compressed_answer)",
    )
    parser.add_argument(
        "--difficulty-mode",
        choices=sorted(DIFFICULTY_FORMULAS),
        default="discrete-rms",
        help=(
            "difficulty used for online cap lookup; discrete-rms matches the "
            "three emitted R tags (default)"
        ),
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--closed-only",
        dest="closed_only",
        action="store_true",
        help="include S2TT and choice, excluding open (default)",
    )
    scope.add_argument(
        "--include-open",
        dest="closed_only",
        action="store_false",
        help="also calibrate the open family",
    )
    parser.set_defaults(closed_only=True)
    parser.add_argument("--revision", default=None, help="optional HF revision")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow HF downloads; default is local_files_only=True",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="progress interval on stderr; 0 disables (default: 10000)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write JSON to a new path instead of stdout; never overwrites",
    )
    args = parser.parse_args(argv)
    if not os.path.isfile(args.curriculum_jsonl):
        parser.error(f"curriculum JSONL not found: {args.curriculum_jsonl}")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.out and os.path.exists(args.out):
        parser.error(f"refusing to overwrite existing --out: {args.out}")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tokenizer = _load_tokenizer(args)
    payload = build_calibration(
        args.curriculum_jsonl,
        tokenizer,
        _tokenizer_metadata(tokenizer, args),
        closed_only=args.closed_only,
        text_field=args.text_field,
        p85_q=0.85,
        difficulty_mode=args.difficulty_mode,
        progress_every=args.progress_every,
    )

    if args.out:
        try:
            write_json_exclusive(args.out, payload)
        except FileExistsError:
            print(
                f"error: refusing to overwrite existing --out: {args.out}",
                file=sys.stderr,
            )
            return 2
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
