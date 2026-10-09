#!/usr/bin/env python3
"""
Combine compress JSON + CoTA parquet → sharegpt JSON, wav uploaded to S3 directly.

Pipeline:
  1. Read compress result JSON (question, compressed_answer, scores)
  2. Read audio waveform from CoTA parquet (mounted S3)
  3. Compute difficulty scores (D_P, D_C, D_R), insert <R1>-<R5> rating tokens
  4. Upload wav to S3 via boto3 (no local disk writes for audio)
  5. Write sharegpt JSON (text only, tiny)

The JSON audio paths are relative filenames (e.g. "train_000001.wav").
At training time, mount s3://liyixuan/data/CoTA/audio_files/ as media_dir.

Usage:
  python prepare_full_data.py \\
    --compress_root /path/to/compress/result \\
    --cota_root /path/to/mounted/CoTA \\
    --output_dir /path/to/converted_data \\
    --s3_bucket liyixuan \\
    --s3_prefix data/CoTA/audio_files \\
    --s3_endpoint ${S3_ENDPOINT}
"""

import argparse
import json
import os
import re
import random
import io
import signal
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import soundfile as sf
import boto3
from botocore.config import Config
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Retry helper for flaky S3-mount reads
# ---------------------------------------------------------------------------

def with_retry(fn, max_retries=5, base_delay=1.0, backoff=2.0, exc_types=(Exception,)):
    """Call fn() with exponential backoff retry on transient errors."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except exc_types as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = base_delay * (backoff ** attempt)
                print(f"  [Retry] {type(e).__name__}: {e} — retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Difficulty score computation (mirrors compress.py)
# ---------------------------------------------------------------------------

DEFAULT_BETA1 = 0.5
DEFAULT_BETA2 = 0.5
DEFAULT_GAMMA1 = 0.5
DEFAULT_GAMMA2 = 0.5


def compute_difficulty_scores(score_dict, beta1=DEFAULT_BETA1, beta2=DEFAULT_BETA2,
                               gamma1=DEFAULT_GAMMA1, gamma2=DEFAULT_GAMMA2):
    qq = score_dict.get("Question_Quality", 1) / 5.0
    is_ = score_dict.get("Integration_Score", 1) / 5.0
    rd = score_dict.get("Reasoning_Demand", 1) / 5.0
    pq = score_dict.get("PQ", 0) / 10.0
    pc = score_dict.get("PC", 0) / 10.0
    D_P = 1.0 - qq
    D_C = beta1 * is_ + beta2 * (1.0 - (pq + pc) / 2.0)
    D_R = gamma1 * rd + gamma2 * is_
    return D_P, D_C, D_R


# ---------------------------------------------------------------------------
# Bucketing: score * 5 → <R1>–<R5>
# ---------------------------------------------------------------------------

RATING_RANGES = [
    (0.0, 1.5, "<R1>"),
    (1.5, 2.5, "<R2>"),
    (2.5, 3.5, "<R3>"),
    (3.5, 4.5, "<R4>"),
    (4.5, 5.0 + 1e-9, "<R5>"),
]


def bucket_score(score):
    scaled = score * 5.0
    for lo, hi, token in RATING_RANGES:
        if lo <= scaled < hi:
            return token
    return "<R5>"


# ---------------------------------------------------------------------------
# Insert rating tokens into compressed_answer
# ---------------------------------------------------------------------------

STRUCTURE_TAGS = ["PLANNING", "CAPTION", "REASONING"]
TAG_TO_DIM = {"PLANNING": 0, "CAPTION": 1, "REASONING": 2}


def insert_rating_tokens(text, D_P, D_C, D_R):
    scores = (D_P, D_C, D_R)
    for tag in STRUCTURE_TAGS:
        dim = TAG_TO_DIM[tag]
        rating_token = bucket_score(scores[dim])
        text = re.sub(r"<{}>".format(tag), "<{}>{}".format(tag, rating_token), text)
    return text


# ---------------------------------------------------------------------------
# Audio processing + S3 upload
# ---------------------------------------------------------------------------

def waveform_to_wav_bytes(waveform, samplerate=16000):
    """Convert waveform (numpy array) to PCM float32 wav bytes."""
    buf = io.BytesIO()
    # Ensure float32
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    sf.write(buf, waveform, samplerate, format='WAV', subtype='FLOAT')
    buf.seek(0)
    return buf


def upload_wav_to_s3(s3_client, bucket, key, wav_bytes):
    """Upload wav bytes to S3."""
    s3_client.upload_fileobj(wav_bytes, bucket, key)


# ---------------------------------------------------------------------------
# Find JSON files
# ---------------------------------------------------------------------------

def find_json_files(root_dir):
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for ff in filenames:
            if ff.endswith(".json"):
                full_path = os.path.join(dirpath, ff)
                rel_path = os.path.relpath(full_path, root_dir)
                json_files.append((full_path, rel_path))
    return sorted(json_files)


# ---------------------------------------------------------------------------
# Process one batch (a single JSON file + its matching parquet)
# ---------------------------------------------------------------------------

VALID_ITEMS_CACHE = {}  # json_rel -> list of valid item dicts


def _make_s3_client(endpoint):
    """Create an S3 client (call inside worker to avoid pickle issues)."""
    from botocore.config import Config
    config = Config(
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=30,
        read_timeout=60,
    )
    return boto3.client("s3", endpoint_url=endpoint, config=config)


def process_json_file(args):
    """Process one JSON file -> list of sharegpt samples, upload wavs to S3.

    Returns: (json_rel, n_valid, sharegpt_samples)
    """
    json_full, json_rel, compress_root, cota_root, s3_bucket, s3_prefix, \
        s3_endpoint, skip_invalid, samplerate, include_ids, use_raw, skip_upload = args

    # Each worker creates its own S3 client (can't pickle boto3 client)
    s3_client = _make_s3_client(s3_endpoint)

    # SFT/RL 切分白名单：样本 ID = <subset>/<basename>#<idx>
    subset0 = json_rel.split(os.sep)[0]
    base0 = os.path.basename(json_rel)

    # Map JSON to parquet
    parquet_rel = json_rel.replace(".json", ".parquet")
    parquet_full = os.path.join(cota_root, parquet_rel)

    if not os.path.exists(parquet_full):
        print(f"[Skip] Parquet not found: {parquet_full}")
        return (json_rel, 0, [])

    # Load JSON
    with open(json_full, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Load parquet with retry (goofys mount can have transient errors)。
    # 部分 CoTA parquet 的 audio 列 pyarrow 读不了（List index overflow），duckdb 能读；
    # 优先 duckdb，失败回退 pandas。
    def _read_parquet():
        try:
            import duckdb
            return duckdb.connect().execute(
                f"SELECT * FROM read_parquet('{parquet_full}')"
            ).df()
        except Exception:
            return pd.read_parquet(parquet_full)

    # skip_upload（raw 版）不需要读 parquet / 音频：文件名由 subset/file/idx 决定，与 compress 版一致。
    # 避免并发 worker 把大文件 audio 列整列读进内存导致 OOM。
    df = None
    audio_waveforms = None
    if not skip_upload:
        try:
            df = with_retry(_read_parquet, exc_types=(Exception,))
        except Exception as e:
            print(f"[Skip] Read parquet failed after retries: {parquet_full} - {e}")
            return (json_rel, 0, [])

        process_len = min(len(df), len(items))
        if len(df) != len(items):
            print(f"[Warn] {json_rel}: parquet({len(df)}) != json({process_len}), using min")

        # Eagerly load audio data into memory to avoid lazy reads from flaky mount。
        # audio 列可能是 list/ndarray(波形) 或 bytes(编码音频，如 FLAC/WAV)，统一转 float32 波形。
        def _extract_audio_column():
            audio_list = []
            col = df["audio"]
            for i in range(len(col)):
                try:
                    val = col.iloc[i]
                    if isinstance(val, (list, np.ndarray)):
                        audio_list.append(np.array(val, dtype=np.float32))
                    elif isinstance(val, (bytes, bytearray)):
                        buf = io.BytesIO(bytes(val))
                        waveform, _ = sf.read(buf, dtype="float32")
                        audio_list.append(waveform)
                    else:
                        audio_list.append(None)
                except Exception:
                    audio_list.append(None)
            return audio_list

        try:
            audio_waveforms = with_retry(_extract_audio_column, exc_types=(Exception,))
        except Exception as e:
            print(f"[Skip] Failed to extract audio column: {parquet_full} - {e}")
            return (json_rel, 0, [])
    else:
        process_len = len(items)

    # Build subset tag from rel path
    subset = json_rel.split(os.sep)[0]

    samples = []
    valid_count = 0

    for idx in range(process_len):
        item = items[idx]
        compressed = item.get("compressed_answer")
        err = item.get("compression_error")

        if not use_raw and skip_invalid and (compressed is None or err is not None):
            continue

        # SFT/RL 切分白名单：不在 include_ids 里的样本跳过（保持 parquet 按 idx 对齐）
        if include_ids is not None:
            sid = f"{subset0}/{base0}#{idx}"
            if sid not in include_ids:
                continue

        question = item.get("question", "")
        # Strip <audio> tag
        user_text = question.replace("<audio>", "").strip()

        if use_raw:
            # raw 模式：直接用未压缩的原始答案，不插 rating token
            annotated_answer = item.get("answer", "")
            if not (isinstance(annotated_answer, str) and annotated_answer.strip()):
                print(f"[Warn] Item {idx} in {json_rel}: raw answer empty, skip")
                continue
        else:
            # Compute difficulty scores
            score1 = item.get("scores", {})
            score2 = item.get("audiobox", {})
            score_dict = score1 | score2
            D_P, D_C, D_R = compute_difficulty_scores(score_dict)

            # Insert rating tokens
            annotated_answer = insert_rating_tokens(compressed, D_P, D_C, D_R)

        # Read audio from eagerly-loaded in-memory list（skip_upload 时不需要）
        if not skip_upload:
            waveform = audio_waveforms[idx]
            if waveform is None:
                print(f"[Warn] Item {idx} in {json_rel}: audio is not waveform array or read failed")
                continue

        # Generate filename unique within this split
        audio_filename = f"{subset}_{os.path.basename(json_rel).replace('.json', '')}_{idx:06d}.wav"

        # Upload wav to S3 (with retry)。skip_upload 时跳过上传：compress 版已上传同名 wav，
        # raw 版复用（压缩/raw 的音频文件名都由 subset/file/idx 决定，完全一致）。
        if not skip_upload:
            try:
                wav_bytes = waveform_to_wav_bytes(waveform, samplerate)
                s3_key = f"{s3_prefix}/{audio_filename}"

                def _upload():
                    wav_bytes.seek(0)  # rewind for retry
                    s3_client.upload_fileobj(wav_bytes, s3_bucket, s3_key)

                with_retry(_upload, exc_types=(Exception,))
            except Exception as e:
                print(f"[Error] Failed to upload {audio_filename} to S3 after retries: {e}")
                continue

        # Build sharegpt sample
        sample = {
            "audios": [audio_filename],
            "conversations": [
                {"from": "human", "value": f"<audio>\n{user_text}"},
                {"from": "gpt", "value": annotated_answer},
            ],
        }
        samples.append(sample)
        valid_count += 1

    print(f"  [{subset}] {json_rel}: {valid_count} valid items processed")
    return (json_rel, valid_count, samples)


def init_worker():
    """Initialize worker - ignore SIGINT so Pool can handle it at parent level."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare full SFT data with S3 audio upload."
    )
    parser.add_argument("--compress_root", type=str, required=True,
                        help="Path to compress/result directory")
    parser.add_argument("--cota_root", type=str, required=True,
                        help="Path to mounted CoTA directory")
    parser.add_argument("--include_ids", type=str, default=None,
                        help="SFT/RL 切分白名单 jsonl（每行 {\"id\": \"<subset>/<file>#<idx>\"}）。"
                             "提供后只处理白名单内的样本（保持 parquet 按 idx 对齐）。")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write sharegpt JSON + dataset_info.json")
    parser.add_argument("--s3_bucket", type=str, default=os.environ.get("S3_BUCKET", ""),
                        help="S3 bucket name")
    parser.add_argument("--s3_prefix", type=str, default="data/CoTA/audio_files",
                        help="S3 prefix (path) for uploaded wav files")
    parser.add_argument("--s3_endpoint", type=str,
                        default=os.environ.get("S3_ENDPOINT", ""),
                        help="S3 endpoint URL")
    parser.add_argument("--use_raw", action="store_true", default=False,
                        help="raw 模式：用未压缩的原始 answer 字段（不插 rating token）。"
                             "输出 full_raw.json，dataset 名 cota_sft_raw")
    parser.add_argument("--output_name", type=str, default=None,
                        help="输出 sharegpt 文件名（默认 full.json，--use_raw 时 full_raw.json）")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="dataset_info 里的 dataset 名（默认 cota_sft，--use_raw 时 cota_sft_raw）")
    parser.add_argument("--skip_upload", action="store_true", default=False,
                        help="跳过 S3 上传：compress 版已上传同名 wav，raw 版复用（音频文件名一致）")
    parser.add_argument("--skip_invalid", action="store_true", default=True)
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Proportion for validation split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Number of parallel processes for JSON file processing")
    parser.add_argument("--samplerate", type=int, default=16000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize S3 client (boto3 picks up credentials from env / ~/.aws)
    s3_config = Config(
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=30,
        read_timeout=60,
    )
    s3_client = boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint,
        config=s3_config,
    )
    # Verify bucket access
    try:
        s3_client.head_bucket(Bucket=args.s3_bucket)
        print(f"[OK] S3 bucket '{args.s3_bucket}' accessible")
    except Exception as e:
        print(f"[Error] Cannot access S3 bucket '{args.s3_bucket}': {e}")
        return

    # Find all JSON files
    json_files = find_json_files(args.compress_root)
    print(f"[Info] Found {len(json_files)} JSON files in {args.compress_root}")

    # 加载 SFT/RL 切分白名单
    include_ids = None
    if args.include_ids:
        include_ids = set()
        with open(args.include_ids) as f:
            for line in f:
                line = line.strip()
                if line:
                    include_ids.add(json.loads(line)["id"])
        print(f"[Info] 加载 include_ids {len(include_ids):,} 条样本")

    # Prepare args for parallel processing
    process_args = []
    for json_full, json_rel in json_files:
        process_args.append((
            json_full, json_rel, args.compress_root, args.cota_root,
            args.s3_bucket, args.s3_prefix, args.s3_endpoint,
            args.skip_invalid, args.samplerate, include_ids, args.use_raw,
            args.skip_upload,
        ))

    # Process in parallel
    n_proc = min(args.num_proc, len(process_args))
    print(f"[Info] Processing with {n_proc} workers...")

    all_samples = []
    try:
        with Pool(n_proc, init_worker) as pool:
            for result in tqdm(
                pool.imap_unordered(process_json_file, process_args),
                total=len(process_args),
                desc="Processing JSON files",
            ):
                json_rel, n_valid, samples = result
                if samples:
                    all_samples.extend(samples)
    except KeyboardInterrupt:
        print("\n[Interrupt] Received SIGINT, stopping...")
        sys.exit(1)

    print(f"\n[Info] Total valid samples: {len(all_samples)}")

    if not all_samples:
        print("[Error] No valid samples found. Exiting.")
        return

    # Shuffle deterministically
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)

    # 输出文件名与 dataset 名（--use_raw 时用 raw 专属命名，避免覆盖 compress 版）
    output_name = args.output_name or ("full_raw.json" if args.use_raw else "full.json")
    dataset_name = args.dataset_name or ("cota_sft_raw" if args.use_raw else "cota_sft")

    # Write single full.json (train/eval split is handled by LLaMA-Factory --val_size)
    output_path = os.path.join(args.output_dir, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {len(all_samples)} samples -> {output_path}")

    # Write dataset_info.json
    dataset_info = {
        dataset_name: {
            "file_name": output_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "audios": "audios",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved dataset_info.json -> {info_path}")

    print("\n======= Preparation Complete =======")
    print(f"Total: {len(all_samples)} samples (single {output_name}, LLaMA-Factory --val_size splits)")
    print(f"Audio uploaded to s3://{args.s3_bucket}/{args.s3_prefix}/")
    print(f"JSON output: {args.output_dir}/{output_name}")


if __name__ == "__main__":
    main()
