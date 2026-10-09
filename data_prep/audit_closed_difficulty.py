#!/usr/bin/env python3
"""Read-only audit of closed-set curriculum difficulty and candidate buckets.

The script reads an existing ``curriculum_data.jsonl`` and never writes or
rewrites curriculum data.  By default it prints one JSON report to stdout.
An analysis report is written only when ``--out REPORT.json`` is supplied;
the output path is guarded against aliasing the input file.

Difficulty definitions mirror the SFT data preparation:

* continuous_rms = sqrt((D_P^2 + D_C^2 + D_R^2) / 3)
* each continuous axis is mapped to R1..R5 with SFT boundaries
  [0, .3, .5, .7, .9, 1], then
  discrete_rms = sqrt(mean(((R_i - 1) / 4)^2))

Candidate schemes:

* global: global equal-frequency buckets
* task-stratified: equal-frequency within S2TT versus choice
* subset-task-stratified: S2TT is one stratum; choices are stratified by
  source subset (the recommended B+ design)

Examples:

  python RL/scripts/analysis/audit_closed_difficulty.py \
    --input RL/data/curriculum/iso_k4/curriculum_data.jsonl

  python RL/scripts/analysis/audit_closed_difficulty.py \
    --input RL/data/curriculum/iso_k4/curriculum_data.jsonl \
    --K 1 3 4 5 8 --out /tmp/closed_difficulty_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "audit_closed_difficulty.v1"
DEFAULT_K = (1, 3, 4, 5, 8)
SCHEMES = ("global", "task-stratified", "subset-task-stratified")
METRICS = ("continuous_rms", "discrete_rms")
AXES = ("D_P", "D_C", "D_R")


@dataclass(frozen=True, slots=True)
class AuditItem:
    """Minimal metadata retained from one valid curriculum row."""

    ordinal: int
    item_id: str
    tie_key: str
    subset: str
    type_name: str
    family: str
    d_p: float
    d_c: float
    d_r: float
    r_p: int
    r_c: int
    r_r: int
    continuous_rms: float
    discrete_rms: float

    def metric(self, name: str) -> float:
        if name == "continuous_rms":
            return self.continuous_rms
        if name == "discrete_rms":
            return self.discrete_rms
        raise ValueError(f"unknown metric: {name}")


def sft_rating(value: float) -> int:
    """Map a validated [0, 1] teacher axis to the SFT R1..R5 label."""

    if value < 0.3:
        return 1
    if value < 0.5:
        return 2
    if value < 0.7:
        return 3
    if value < 0.9:
        return 4
    return 5


def normalized_rms(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("normalized_rms needs at least one value")
    return math.sqrt(sum(v * v for v in values) / len(values))


def task_family(subset: str, type_name: str) -> str:
    """Match the trainer's reward-route precedence: S2TT, then choice, then open."""

    if subset.casefold() == "s2tt":
        return "s2tt"
    if type_name.casefold() == "choice":
        return "choice"
    return "open"


def bplus_stratum(item: AuditItem) -> str:
    """Recommended B+ stratum: one S2TT stratum, choice split by subset."""

    if item.family == "s2tt":
        return "s2tt"
    return f"{item.family}|subset={item.subset}|type={item.type_name}"


def _coerce_axis(
    row: Mapping[str, object],
    key: str,
    missing_fields: Counter,
    invalid_fields: Counter,
) -> Optional[float]:
    if key not in row or row[key] is None:
        missing_fields[key] += 1
        return None
    value = row[key]
    if isinstance(value, bool):
        invalid_fields[f"{key}:boolean"] += 1
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        invalid_fields[f"{key}:non_numeric"] += 1
        return None
    if not math.isfinite(number):
        invalid_fields[f"{key}:non_finite"] += 1
        return None
    if not 0.0 <= number <= 1.0:
        invalid_fields[f"{key}:out_of_range"] += 1
        return None
    return number


def _canonical_id(row: Mapping[str, object], line_number: int, missing: Counter) -> str:
    source_file = row.get("source_file")
    source_idx = row.get("source_idx")
    if not isinstance(source_file, str) or not source_file.strip():
        missing["source_file"] += 1
        source_file = f"<line:{line_number}>"
    if source_idx in (None, ""):
        missing["source_idx"] += 1
        source_idx = line_number
    return f"{source_file}#{source_idx}"


def load_items(path: Path, closed_only: bool = True) -> Tuple[List[AuditItem], dict, dict]:
    """Stream a JSONL file, hash exact bytes, and retain only audit metadata."""

    input_path = Path(path)
    hasher = hashlib.sha256()
    items: List[AuditItem] = []
    canonical_counts: Counter = Counter()
    missing_fields: Counter = Counter()
    invalid_fields: Counter = Counter()
    json_error_examples: List[dict] = []
    stats: Counter = Counter()

    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            hasher.update(raw_line)
            stats["input_lines"] += 1
            if not raw_line.strip():
                stats["blank_lines"] += 1
                continue
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                stats["json_errors"] += 1
                if len(json_error_examples) < 5:
                    json_error_examples.append({"line": line_number, "error": str(exc)})
                continue
            if not isinstance(row, dict):
                stats["non_object_rows"] += 1
                continue
            stats["parsed_rows"] += 1

            subset_raw = row.get("subset")
            type_raw = row.get("type")
            subset = str(subset_raw) if subset_raw not in (None, "") else "<missing>"
            type_name = str(type_raw) if type_raw not in (None, "") else "<missing>"
            family = task_family(subset, type_name)
            if closed_only and family == "open":
                stats["excluded_non_closed_rows"] += 1
                continue
            stats["selected_rows"] += 1
            if subset_raw in (None, ""):
                missing_fields["subset"] += 1
            if type_raw in (None, ""):
                missing_fields["type"] += 1

            axes = [_coerce_axis(row, key, missing_fields, invalid_fields) for key in AXES]
            if any(value is None for value in axes):
                stats["invalid_difficulty_rows"] += 1
                continue
            d_p, d_c, d_r = (float(value) for value in axes)
            continuous = normalized_rms((d_p, d_c, d_r))
            ratings = tuple(sft_rating(v) for v in (d_p, d_c, d_r))
            units = tuple((rating - 1) / 4.0 for rating in ratings)
            discrete = normalized_rms(units)

            stored_norm = row.get("difficulty_norm")
            if stored_norm is None:
                missing_fields["difficulty_norm"] += 1
            else:
                try:
                    stored_value = float(stored_norm)
                    expected_raw_l2 = continuous * math.sqrt(3.0)
                    if not math.isfinite(stored_value):
                        invalid_fields["difficulty_norm:non_finite"] += 1
                    elif abs(stored_value - expected_raw_l2) > 2e-6:
                        stats["stored_difficulty_mismatch_rows"] += 1
                except (TypeError, ValueError):
                    invalid_fields["difficulty_norm:non_numeric"] += 1

            item_id = _canonical_id(row, line_number, missing_fields)
            canonical_counts[item_id] += 1
            # Include the physical line in the tie key so duplicate source IDs remain
            # deterministic yet separately auditable.
            tie_key = hashlib.sha256(f"{item_id}@line={line_number}".encode("utf-8")).hexdigest()
            items.append(
                AuditItem(
                    ordinal=len(items),
                    item_id=item_id,
                    tie_key=tie_key,
                    subset=subset,
                    type_name=type_name,
                    family=family,
                    d_p=d_p,
                    d_c=d_c,
                    d_r=d_r,
                    r_p=ratings[0],
                    r_c=ratings[1],
                    r_r=ratings[2],
                    continuous_rms=continuous,
                    discrete_rms=discrete,
                )
            )
            stats["valid_difficulty_rows"] += 1

    duplicate_counts = [(item_id, count) for item_id, count in canonical_counts.items() if count > 1]
    duplicate_counts.sort(key=lambda pair: (-pair[1], pair[0]))
    for key in (
        "input_lines",
        "blank_lines",
        "json_errors",
        "non_object_rows",
        "parsed_rows",
        "excluded_non_closed_rows",
        "selected_rows",
        "invalid_difficulty_rows",
        "valid_difficulty_rows",
        "stored_difficulty_mismatch_rows",
    ):
        stats[key] += 0
    validation = {
        "scope": (
            "selected rows after closed-only filtering"
            if closed_only
            else "all parsed rows"
        ),
        "limitations": (
            "Validates stored D_P/D_C/D_R only. Raw PQ/PC are not stored in the "
            "curriculum JSONL and must be audited from compress/result separately."
        ),
        **{key: int(value) for key, value in sorted(stats.items())},
        "missing_fields": dict(sorted(missing_fields.items())),
        "invalid_fields": dict(sorted(invalid_fields.items())),
        "duplicate_source_ids": {
            "distinct_ids": len(duplicate_counts),
            "extra_rows": sum(count - 1 for _, count in duplicate_counts),
            "top": [
                {"item_id": item_id, "count": count}
                for item_id, count in duplicate_counts[:10]
            ],
        },
        "json_error_examples": json_error_examples,
    }
    input_info = {
        "path": str(input_path.resolve()),
        "size_bytes": input_path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }
    return items, input_info, validation


def _chunks_with_sizes(
    sorted_items: Sequence[AuditItem], sizes: Sequence[int]
) -> List[List[AuditItem]]:
    if sum(sizes) != len(sorted_items):
        raise ValueError(f"chunk sizes sum to {sum(sizes)}, expected {len(sorted_items)}")
    chunks: List[List[AuditItem]] = []
    start = 0
    for size in sizes:
        chunks.append(list(sorted_items[start : start + size]))
        start += size
    return chunks


def _balanced_chunks(sorted_items: Sequence[AuditItem], k: int) -> List[List[AuditItem]]:
    if k <= 0:
        raise ValueError(f"K must be positive, got {k}")
    quotient, remainder = divmod(len(sorted_items), k)
    sizes = [quotient + (1 if bucket_id < remainder else 0) for bucket_id in range(k)]
    return _chunks_with_sizes(sorted_items, sizes)


def _scheme_stratum(scheme: str) -> Optional[Callable[[AuditItem], str]]:
    if scheme == "global":
        return None
    if scheme == "task-stratified":
        return lambda item: item.family
    if scheme == "subset-task-stratified":
        return bplus_stratum
    raise ValueError(f"unknown scheme: {scheme}")


def build_buckets(
    items: Sequence[AuditItem], metric_name: str, k: int, scheme: str
) -> List[List[AuditItem]]:
    """Build in-memory candidate buckets without mutating or writing input data."""

    if metric_name not in METRICS:
        raise ValueError(f"unknown metric: {metric_name}")
    stratum_fn = _scheme_stratum(scheme)
    sort_key = lambda item: (item.metric(metric_name), item.tie_key)
    if stratum_fn is None:
        return _balanced_chunks(sorted(items, key=sort_key), k)

    grouped: Dict[str, List[AuditItem]] = defaultdict(list)
    for item in items:
        grouped[stratum_fn(item)].append(item)
    buckets: List[List[AuditItem]] = [[] for _ in range(k)]
    bucket_sizes = [0] * k
    for stratum in sorted(grouped):
        stratum_items = sorted(grouped[stratum], key=sort_key)
        quotient, remainder = divmod(len(stratum_items), k)
        # Give this stratum's remainder rows to the currently smallest global
        # buckets. A stable hash rotation avoids always favoring low bucket IDs
        # when totals tie. Each stratum remains equal-frequency (spread <= 1),
        # while aggregate bucket sizes also remain balanced.
        rotation = int(hashlib.sha256(stratum.encode("utf-8")).hexdigest()[:8], 16) % k
        extra_order = sorted(
            range(k), key=lambda bucket_id: (bucket_sizes[bucket_id], (bucket_id - rotation) % k)
        )
        extra_buckets = set(extra_order[:remainder])
        sizes = [quotient + (1 if bucket_id in extra_buckets else 0) for bucket_id in range(k)]
        chunks = _chunks_with_sizes(stratum_items, sizes)
        for bucket_id, chunk in enumerate(chunks):
            buckets[bucket_id].extend(chunk)
            bucket_sizes[bucket_id] += len(chunk)
    for bucket in buckets:
        bucket.sort(key=sort_key)
    return buckets


def partition_integrity(items: Sequence[AuditItem], buckets: Sequence[Sequence[AuditItem]]) -> dict:
    expected = {item.ordinal for item in items}
    assigned_counts: Counter = Counter(item.ordinal for bucket in buckets for item in bucket)
    assigned = set(assigned_counts)
    return {
        "expected_items": len(items),
        "assigned_items": sum(assigned_counts.values()),
        "unique_assigned_items": len(assigned),
        "missing_items": len(expected - assigned),
        "unexpected_items": len(assigned - expected),
        "duplicate_assignments": sum(count - 1 for count in assigned_counts.values() if count > 1),
        "ok": expected == assigned and all(count == 1 for count in assigned_counts.values()),
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> Optional[float]:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _rounded(value: Optional[float], digits: int = 9) -> Optional[float]:
    return None if value is None else round(value, digits)


def _composition(items: Sequence[AuditItem], key: Callable[[AuditItem], str]) -> dict:
    counts = Counter(key(item) for item in items)
    total = len(items)
    return {
        "counts": dict(sorted(counts.items())),
        "shares": {
            label: round(count / total, 9) if total else None
            for label, count in sorted(counts.items())
        },
    }


def _bucket_report(bucket: Sequence[AuditItem], metric_name: str, bucket_id: int) -> dict:
    values = sorted(item.metric(metric_name) for item in bucket)
    report = {
        "bucket_id": bucket_id,
        "size": len(bucket),
        "difficulty": {
            "min": _rounded(values[0]) if values else None,
            "q25": _rounded(_quantile(values, 0.25)),
            "median": _rounded(_quantile(values, 0.50)),
            "q75": _rounded(_quantile(values, 0.75)),
            "max": _rounded(values[-1]) if values else None,
            "mean": _rounded(sum(values) / len(values)) if values else None,
        },
        "composition": {
            "family": _composition(bucket, lambda item: item.family),
            "subset": _composition(bucket, lambda item: item.subset),
            "type": _composition(bucket, lambda item: item.type_name),
            "bplus_stratum": _composition(bucket, bplus_stratum),
        },
    }
    return report


def _stratum_spread(
    buckets: Sequence[Sequence[AuditItem]], stratum_fn: Optional[Callable[[AuditItem], str]]
) -> Optional[dict]:
    if stratum_fn is None:
        return None
    all_strata = sorted({stratum_fn(item) for bucket in buckets for item in bucket})
    per_stratum = {}
    max_spread = 0
    for stratum in all_strata:
        counts = [sum(1 for item in bucket if stratum_fn(item) == stratum) for bucket in buckets]
        spread = max(counts) - min(counts) if counts else 0
        max_spread = max(max_spread, spread)
        per_stratum[stratum] = {"counts": counts, "max_minus_min": spread}
    return {"max_count_spread": max_spread, "per_stratum": per_stratum}


def _ties(values: Iterable[float], top_n: int) -> dict:
    counts = Counter(round(value, 12) for value in values)
    top = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:top_n]
    return {
        "unique_values": len(counts),
        "values_with_ties": sum(1 for count in counts.values() if count > 1),
        "max_tie_size": max(counts.values(), default=0),
        "top": [{"value": value, "count": count} for value, count in top],
    }


def _largest_remainder_quota(counts: Mapping[str, int], window_size: int) -> dict:
    total = sum(counts.values())
    if window_size <= 0:
        raise ValueError("quota window must be positive")
    if total == 0:
        return {}
    expected = {label: count * window_size / total for label, count in counts.items()}
    allocated = {label: int(math.floor(value)) for label, value in expected.items()}
    remaining = window_size - sum(allocated.values())
    order = sorted(expected, key=lambda label: (-(expected[label] - allocated[label]), label))
    for label in order[:remaining]:
        allocated[label] += 1
    return {
        label: {
            "count": counts[label],
            "share": round(counts[label] / total, 9),
            "expected_per_window": round(expected[label], 9),
            "largest_remainder_allocation": allocated[label],
        }
        for label in sorted(counts)
    }


def natural_quota(items: Sequence[AuditItem], window_size: int) -> dict:
    dimensions = {
        "family": Counter(item.family for item in items),
        "subset": Counter(item.subset for item in items),
        "type": Counter(item.type_name for item in items),
        "bplus_stratum": Counter(bplus_stratum(item) for item in items),
    }
    return {
        "population": len(items),
        "window_size": window_size,
        "method": "largest remainder; use a residual accumulator across windows for exact long-run shares",
        "dimensions": {
            name: _largest_remainder_quota(counts, window_size)
            for name, counts in dimensions.items()
        },
    }


def analyze_metric(
    items: Sequence[AuditItem],
    metric_name: str,
    ks: Sequence[int],
    schemes: Sequence[str],
    top_ties: int,
) -> dict:
    values = sorted(item.metric(metric_name) for item in items)
    metric_report = {
        "definition": (
            "sqrt((D_P^2 + D_C^2 + D_R^2) / 3)"
            if metric_name == "continuous_rms"
            else "sqrt(mean(((R_i - 1) / 4)^2)); R_i uses SFT boundaries 0,.3,.5,.7,.9,1"
        ),
        "population": len(items),
        "range": [_rounded(values[0]), _rounded(values[-1])] if values else [None, None],
        "quantiles": {
            f"q{int(probability * 100):02d}": _rounded(_quantile(values, probability))
            for probability in (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
        },
        "ties": _ties(values, top_ties),
        "ratings": {
            "P": dict(sorted(Counter(f"R{item.r_p}" for item in items).items())),
            "C": dict(sorted(Counter(f"R{item.r_c}" for item in items).items())),
            "R": dict(sorted(Counter(f"R{item.r_r}" for item in items).items())),
            "top_tuples": [
                {"ratings": list(ratings), "count": count}
                for ratings, count in Counter(
                    (item.r_p, item.r_c, item.r_r) for item in items
                ).most_common(top_ties)
            ],
        },
        "K": {},
    }
    for k in ks:
        k_report = {}
        for scheme in schemes:
            buckets = build_buckets(items, metric_name, k, scheme)
            stratum_fn = _scheme_stratum(scheme)
            k_report[scheme] = {
                "bucket_sizes": [len(bucket) for bucket in buckets],
                "integrity": partition_integrity(items, buckets),
                "stratum_balance": _stratum_spread(buckets, stratum_fn),
                "buckets": [
                    _bucket_report(bucket, metric_name, bucket_id)
                    for bucket_id, bucket in enumerate(buckets)
                ],
            }
        metric_report["K"][str(k)] = k_report
    return metric_report


def build_report(
    input_path: Path,
    closed_only: bool = True,
    ks: Sequence[int] = DEFAULT_K,
    schemes: Sequence[str] = SCHEMES,
    quota_window: int = 512,
    top_ties: int = 10,
) -> dict:
    items, input_info, validation = load_items(Path(input_path), closed_only=closed_only)
    if not items:
        raise RuntimeError("no valid difficulty rows remain after filtering")
    clean_ks = sorted(set(int(k) for k in ks))
    if any(k <= 0 for k in clean_ks):
        raise ValueError(f"all K values must be positive: {clean_ks}")
    clean_schemes = list(dict.fromkeys(schemes))
    unknown = sorted(set(clean_schemes) - set(SCHEMES))
    if unknown:
        raise ValueError(f"unknown schemes: {unknown}; valid={list(SCHEMES)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "read_only": True,
        "input": input_info,
        "parameters": {
            "closed_only": closed_only,
            "closed_rule": "subset.casefold() == 's2tt' OR type.casefold() == 'choice'",
            "K": clean_ks,
            "schemes": clean_schemes,
            "quota_window": quota_window,
            "stable_tie_break": "sha256(source_file#source_idx@physical_line)",
            "stratum_remainders": (
                "allocated to currently smallest buckets with a stable hash rotation"
            ),
        },
        "validation": validation,
        "natural_quota": natural_quota(items, quota_window),
        "metrics": {
            metric: analyze_metric(items, metric, clean_ks, clean_schemes, top_ties)
            for metric in METRICS
        },
    }


def _parse_ks(values: Sequence[str]) -> List[int]:
    parsed: List[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                parsed.append(int(token))
    return parsed


def _parse_schemes(values: Sequence[str]) -> List[str]:
    parsed: List[str] = []
    for value in values:
        parsed.extend(token.strip() for token in value.split(",") if token.strip())
    return parsed


def write_report(report: Mapping[str, object], input_path: Path, out_path: Path, force: bool) -> None:
    """Atomically write an explicitly requested JSON report, never the input."""

    source = Path(input_path).resolve()
    destination = Path(out_path).resolve()
    if source == destination:
        raise ValueError("--out must not point to the input curriculum file")
    if destination.name == "curriculum_data.jsonl":
        raise ValueError("refusing to write an analysis report as curriculum_data.jsonl")
    if destination.suffix.casefold() != ".json":
        raise ValueError("--out must end in .json")
    if destination.exists() and not force:
        raise FileExistsError(f"report already exists (pass --force to replace): {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not force:
            raise FileExistsError(f"report appeared while writing: {destination}")
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="existing curriculum JSONL")
    parser.add_argument(
        "--closed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep S2TT plus choice only (default: true)",
    )
    parser.add_argument(
        "--K",
        "--ks",
        dest="ks",
        nargs="+",
        default=[",".join(str(k) for k in DEFAULT_K)],
        help="K values, separated by spaces and/or commas (default: 1,3,4,5,8)",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=[",".join(SCHEMES)],
        help=f"schemes separated by spaces/commas (default: {','.join(SCHEMES)})",
    )
    parser.add_argument("--quota-window", type=int, default=512)
    parser.add_argument("--top-ties", type=int, default=10)
    parser.add_argument("--out", type=Path, help="optional JSON analysis report path")
    parser.add_argument("--force", action="store_true", help="replace an existing --out report")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    if args.force and args.out is None:
        raise ValueError("--force is only meaningful with --out")
    report = build_report(
        input_path=args.input,
        closed_only=args.closed_only,
        ks=_parse_ks(args.ks),
        schemes=_parse_schemes(args.schemes),
        quota_window=args.quota_window,
        top_ties=args.top_ties,
    )
    # Default behavior is stdout only.  Keep stdout machine-readable even when
    # --out is also requested.
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.out is not None:
        write_report(report, args.input, args.out, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
