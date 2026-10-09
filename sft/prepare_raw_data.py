#!/usr/bin/env python3
"""
Generate sharegpt JSON using the ORIGINAL uncompressed answer field.
"""
import json, glob, os, re
from multiprocessing import Pool

COMPRESS_DIR = os.environ.get("COMPRESS_ROOT", "compress/result")
OUTPUT_DIR = os.environ.get("RAW_OUTPUT_DIR", "converted_data_raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_file(fpath):
    """Read one JSON file, extract answer field (uncompressed), return sharegpt samples."""
    with open(fpath) as f:
        data = json.load(f)

    samples = []
    for item in (data if isinstance(data, list) else [data]):
        audios = item.get("audios", [])
        question = item.get("question", "")
        answer = item.get("answer", "")

        if not question or not answer:
            continue

        # Strip <audio> tag from question
        question = re.sub(r"<audio>\s*", "", question).strip()

        samples.append({
            "audios": audios,
            "conversations": [
                {"from": "human", "value": f"<audio>\n{question}"},
                {"from": "gpt", "value": answer},
            ]
        })

    return (os.path.basename(fpath), len(samples), samples)


def main():
    files = sorted(glob.glob(os.path.join(COMPRESS_DIR, "*/*.json")))
    print(f"[Info] Found {len(files)} JSON files")

    with Pool(8) as pool:
        results = pool.map(process_file, files)

    all_samples = []
    for fname, n, samples in results:
        all_samples.extend(samples)
        print(f"  {fname}: {n} samples")

    print(f"\nTotal: {len(all_samples)} samples")

    # Write sharegpt JSON
    outpath = os.path.join(OUTPUT_DIR, "full_raw.json")
    with open(outpath, "w") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"Written to {outpath}")

    # Write dataset_info.json
    info = {
        "cota_sft_raw": {
            "file_name": "full_raw.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "audios": "audios"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            }
        }
    }
    infopath = os.path.join(OUTPUT_DIR, "dataset_info.json")
    with open(infopath, "w") as f:
        json.dump(info, f, indent=2)
    print(f"dataset_info written to {infopath}")


if __name__ == "__main__":
    main()
