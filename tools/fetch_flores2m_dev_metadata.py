"""Pin the public 2M-Flores-ASL dev inventory without touching devtest.

Only metadata and per-clip source URLs are retrieved. No videos are downloaded.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "facebook/2M-Flores-ASL"
EXPECTED_ROWS = 999


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected metadata response: {response.status}")
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    repository = get_json(f"https://huggingface.co/api/datasets/{DATASET}")
    revision = repository["sha"]
    if repository["cardData"]["license"] != "cc-by-sa-4.0":
        raise ValueError("dataset license differs from the reviewed source")
    rows = []
    for offset in range(0, EXPECTED_ROWS, 100):
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": "dev",
                "offset": offset,
                "length": min(100, EXPECTED_ROWS - offset),
            }
        )
        page = get_json(f"https://datasets-server.huggingface.co/rows?{query}")
        if page["partial"] or page["num_rows_total"] != EXPECTED_ROWS:
            raise ValueError("dev metadata inventory is incomplete")
        for item in page["rows"]:
            if item["row_idx"] != len(rows):
                raise ValueError("dev metadata row indices are not sequential")
            row = item["row"]
            video_url = row["video"]["src"]
            prefix = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/data/dev/"
            if not video_url.startswith(prefix) or not video_url.lower().endswith((".mov", ".mp4")):
                raise ValueError("dev video URL differs from pinned repository")
            if not isinstance(row["sentence"], str) or not row["sentence"].strip():
                raise ValueError("dev row lacks a translation")
            rows.append(
                {
                    "row_idx": item["row_idx"],
                    "id": row["id"],
                    "signer_local_id": row["signer"],
                    "source_page": row["URL"],
                    "domain": row["domain"],
                    "topic": row["topic"],
                    "sentence": row["sentence"],
                    "gloss": row["gloss"],
                    "video_url": video_url,
                }
            )
    if len(rows) != EXPECTED_ROWS or len({row["video_url"] for row in rows}) != len(rows):
        raise ValueError("dev inventory count or video URLs differ from expected")
    output = {
        "schema": "umi-studio-flores2m-dev-inventory/1",
        "dataset": DATASET,
        "repository_revision": revision,
        "license": "CC BY-SA 4.0",
        "split": "dev",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"pinned {len(rows)} dev rows at repository revision {revision}")


if __name__ == "__main__":
    main()
