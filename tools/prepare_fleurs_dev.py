"""Freeze a signer-disjoint FLEURS-ASL development split from public metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

SOURCE_METADATA_SHA256 = "d9eeeffacf24e6c20ad1f01629a70225158ef5a67df17348bc49d667eedda1c6"
SOURCE_URL = "https://www.kaggle.com/datasets/googleai/fleurs-asl"
TRAIN_SIGNER = 3
VALIDATION_SIGNER = 4


def _selected_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("zs_split") != "val" or row.get("sd_split") not in {"train", "val"}:
        return None
    signer = row.get("signer")
    if signer not in {TRAIN_SIGNER, VALIDATION_SIGNER}:
        return None
    if not isinstance(row.get("start_time_sec"), int | float) or not isinstance(
        row.get("end_time_sec"), int | float
    ):
        raise ValueError("selected row has invalid timestamps")
    start = float(row["start_time_sec"])
    end = float(row["end_time_sec"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("selected row has invalid interval")
    if not 2 <= end - start <= 15:
        return None
    video = row.get("video_name")
    example_id = row.get("example_id")
    document_id = row.get("document_id")
    sentence_id = row.get("sentence_id")
    reference = row.get("eng")
    if (
        not isinstance(video, str)
        or Path(video).name != video
        or video != f"{document_id}_signer_{signer}.mp4"
        or not isinstance(document_id, str)
        or not document_id.startswith("dev_document_")
        or not isinstance(sentence_id, str)
        or not sentence_id.startswith("dev_sentence_")
        or not isinstance(example_id, str)
        or not example_id.startswith(f"{document_id}_sentence_")
        or not example_id.endswith(f"_signer_{signer}")
        or not isinstance(reference, str)
        or not reference.strip()
    ):
        raise ValueError("selected row has invalid development identity or reference")
    return {
        "document_id": document_id,
        "end_time_sec": end,
        "example_id": example_id,
        "reference": reference,
        "sentence_id": sentence_id,
        "signer": signer,
        "source_video": f"fleurs-asl-v1/videos/{video}",
        "start_time_sec": start,
    }


def _check_cross_split(train: list[dict[str, Any]], val: list[dict[str, Any]]) -> None:
    for key in ("example_id", "document_id", "sentence_id", "source_video"):
        if {row[key] for row in train} & {row[key] for row in val}:
            raise ValueError(f"train and validation share {key}")
    if {row["signer"] for row in train} & {row["signer"] for row in val}:
        raise ValueError("train and validation share a signer")
    if {row["reference"].casefold().strip() for row in train} & {
        row["reference"].casefold().strip() for row in val
    }:
        raise ValueError("train and validation share an exact normalized reference")


def _interval_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["source_video"]].append(row)
    overlap_count = 0
    maximum_overlap = 0.0
    for video_rows in by_video.values():
        ordered = sorted(video_rows, key=lambda row: row["start_time_sec"])
        for left, right in pairwise(ordered):
            overlap = left["end_time_sec"] - right["start_time_sec"]
            if overlap > 0:
                overlap_count += 1
                maximum_overlap = max(maximum_overlap, overlap)
    return {
        "examples": len(rows),
        "interval_overlaps_within_source_video": overlap_count,
        "maximum_interval_overlap_seconds": round(maximum_overlap, 3),
        "source_videos": len(by_video),
        "total_segment_seconds": round(
            sum(row["end_time_sec"] - row["start_time_sec"] for row in rows), 3
        ),
    }


def prepare(metadata: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(metadata).hexdigest()
    if digest != SOURCE_METADATA_SHA256:
        raise ValueError("FLEURS-ASL metadata differs from the pinned public version")
    source = json.loads(metadata)
    if not isinstance(source, list):
        raise ValueError("FLEURS-ASL metadata must be a list")
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    seen: set[str] = set()
    for original in source:
        if not isinstance(original, dict):
            raise ValueError("FLEURS-ASL metadata contains a non-object row")
        row = _selected_row(original)
        if row is None:
            continue
        if row["example_id"] in seen:
            raise ValueError("FLEURS-ASL metadata contains a duplicate selected example")
        seen.add(row["example_id"])
        (train if row["signer"] == TRAIN_SIGNER else val).append(row)
    train.sort(key=lambda row: row["example_id"])
    val.sort(key=lambda row: row["example_id"])
    if not train or not val:
        raise ValueError("signer-disjoint development split is empty")
    _check_cross_split(train, val)
    return {
        "schema": "umi-studio-fleurs-dev-split/1",
        "source": {
            "dataset": "FLEURS-ASL version 2",
            "license": "CC BY-SA 4.0",
            "metadata_sha256": digest,
            "url": SOURCE_URL,
        },
        "selection": {
            "duration_seconds": [2, 15],
            "excluded_sd_split": "test",
            "excluded_zs_split": "test",
            "training_signer": TRAIN_SIGNER,
            "validation_signer": VALIDATION_SIGNER,
            "zs_split": "val",
        },
        "audit": {
            "training": _interval_audit(train),
            "validation": _interval_audit(val),
        },
        "training": train,
        "validation": val,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.metadata.read_bytes())
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(payload + "\n")
    print(json.dumps(result["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
