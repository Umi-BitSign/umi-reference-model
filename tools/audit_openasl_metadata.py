"""Audit pinned OpenASL metadata for a source-disjoint public-data pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import normalize, sha256

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "data/openasl/repository"
TSV = REPOSITORY / "data/openasl-v1.0.tsv"
BBOX = REPOSITORY / "data/bbox-v1.0.json"
LICENSE = REPOSITORY / "LICENSE.md"
HOW2SIGN_SELECTION = ROOT / "runs/how2sign-validation-v1/selection.json"
FLEURS_SELECTION = ROOT / "runs/fleurs-dev-v1/selection.json"


def normalized(text: str) -> str:
    return " ".join(normalize(text))


def timestamp_seconds(value: str) -> float:
    fields = value.split(":")
    if len(fields) != 3:
        raise ValueError("OpenASL timestamp shape differs")
    hours, minutes, seconds = fields
    result = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if not math.isfinite(result) or result < 0:
        raise ValueError("OpenASL timestamp is invalid")
    return result


def summarize_rows(
    rows: list[dict], bboxes: dict[str, list[float]], protected: set[str]
) -> tuple[dict, list[dict]]:
    vids = Counter(row["vid"] for row in rows)
    split_rows = Counter(row["split"] for row in rows)
    split_yids = defaultdict(set)
    protected_rows = 0
    invalid_times = 0
    invalid_bboxes = 0
    eligible = []
    for row in rows:
        split_yids[row["split"]].add(row["yid"])
        reference = normalized(row["raw-text"])
        if reference in protected:
            protected_rows += 1
        try:
            start = timestamp_seconds(row["start"])
            end = timestamp_seconds(row["end"])
            duration = end - start
        except (TypeError, ValueError):
            invalid_times += 1
            continue
        bbox = bboxes.get(row["vid"])
        bbox_valid = (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(value, int | float) and math.isfinite(value)
                for value in bbox
            )
            and bbox[0] < bbox[2]
            and bbox[1] < bbox[3]
        )
        if not bbox_valid:
            invalid_bboxes += 1
            continue
        words = len(reference.split())
        if (
            row["split"] == "train"
            and vids[row["vid"]] == 1
            and 2 <= duration <= 12
            and 3 <= words <= 30
            and reference
            and reference not in protected
        ):
            eligible.append(
                {
                    **row,
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": duration,
                    "normalized_reference": reference,
                    "word_count": words,
                    "bbox": bbox,
                    "source_group": f"openasl:{row['yid']}",
                }
            )
    split_pairs = {}
    for first in sorted(split_yids):
        for second in sorted(split_yids):
            if first < second:
                split_pairs[f"{first}_{second}"] = len(
                    split_yids[first] & split_yids[second]
                )
    summary = {
        "metadata_rows": len(rows),
        "bbox_rows": len(bboxes),
        "unique_clip_ids": len(vids),
        "duplicate_clip_rows": sum(count for count in vids.values() if count > 1),
        "unique_source_videos": len({row["yid"] for row in rows}),
        "rows_by_split": dict(sorted(split_rows.items())),
        "source_videos_by_split": {
            split: len(values) for split, values in sorted(split_yids.items())
        },
        "source_video_overlap_between_splits": split_pairs,
        "metadata_rows_without_bbox": sum(vid not in bboxes for vid in vids),
        "bbox_rows_without_metadata": len(set(bboxes) - set(vids)),
        "invalid_time_rows": invalid_times,
        "invalid_bbox_rows": invalid_bboxes,
        "protected_exact_reference_rows": protected_rows,
        "eligible_training_rows": len(eligible),
        "eligible_source_videos": len({row["yid"] for row in eligible}),
        "eligible_unique_normalized_references": len(
            {row["normalized_reference"] for row in eligible}
        ),
        "eligible_duration_hours": sum(row["duration_sec"] for row in eligible) / 3600,
    }
    return summary, eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("OpenASL audit needs a new run-ledger output")
    with TSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"vid", "yid", "start", "end", "raw-text", "tokenized-text", "gloss", "split"}
    if not rows or set(rows[0]) != required or {row["split"] for row in rows} != {
        "train",
        "valid",
        "test",
    }:
        raise ValueError("OpenASL metadata columns or splits differ")
    bboxes = json.loads(BBOX.read_text())
    how2sign = json.loads(HOW2SIGN_SELECTION.read_text())
    fleurs = json.loads(FLEURS_SELECTION.read_text())
    protected = {
        normalized(row["reference"])
        for row in how2sign["rows"] + fleurs["validation"]
    }
    summary, _eligible = summarize_rows(rows, bboxes, protected)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema": "umi-studio-openasl-metadata-audit/1",
        "repository": "https://github.com/chevalierNoir/OpenASL",
        "repository_commit": revision,
        "metadata_sha256": sha256(TSV),
        "bbox_sha256": sha256(BBOX),
        "license_sha256": sha256(LICENSE),
        "how2sign_protected_selection_sha256": sha256(HOW2SIGN_SELECTION),
        "fleurs_protected_selection_sha256": sha256(FLEURS_SELECTION),
        "filters": {
            "split": "train",
            "duration_sec": [2, 12],
            "word_count": [3, 30],
            "unique_clip_id": True,
            "valid_bbox": True,
            "exclude_protected_exact_reference": True,
        },
        **summary,
        "auditor_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
    temporary.rename(args.output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
