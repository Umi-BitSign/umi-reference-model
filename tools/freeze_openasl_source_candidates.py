"""Freeze source-disjoint OpenASL candidates for bounded availability checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from audit_openasl_metadata import (
    BBOX,
    FLEURS_SELECTION,
    HOW2SIGN_SELECTION,
    LICENSE,
    REPOSITORY,
    TSV,
    normalized,
    summarize_rows,
)
from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
SEED = 50
TARGET_SOURCES = 256
CLIPS_PER_SOURCE = 2


def stable_key(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in (SEED, *values)).encode()
    ).hexdigest()


def select_candidates(
    eligible: list[dict], heldout_yids: set[str]
) -> tuple[list[dict], list[str]]:
    by_source = defaultdict(list)
    for row in eligible:
        if row["yid"] not in heldout_yids:
            by_source[row["yid"]].append(row)
    selected = []
    selected_sources = []
    used_references = set()
    for yid in sorted(by_source, key=lambda value: stable_key("source", value)):
        rows = sorted(
            by_source[yid],
            key=lambda row: stable_key("clip", yid, row["vid"]),
        )
        source_rows = []
        for row in rows:
            reference = row["normalized_reference"]
            if reference in used_references:
                continue
            source_rows.append(row)
            if len(source_rows) == CLIPS_PER_SOURCE:
                break
        if len(source_rows) != CLIPS_PER_SOURCE:
            continue
        selected_sources.append(yid)
        selected.extend(source_rows)
        used_references.update(row["normalized_reference"] for row in source_rows)
        if len(selected_sources) == TARGET_SOURCES:
            break
    if (
        len(selected_sources) != TARGET_SOURCES
        or len(selected) != TARGET_SOURCES * CLIPS_PER_SOURCE
        or len({row["vid"] for row in selected}) != len(selected)
        or len({row["normalized_reference"] for row in selected}) != len(selected)
    ):
        raise ValueError("OpenASL source candidate selection is incomplete")
    return selected, selected_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("OpenASL candidates need a new run-ledger output")
    with TSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    bboxes = json.loads(BBOX.read_text())
    how2sign = json.loads(HOW2SIGN_SELECTION.read_text())
    fleurs = json.loads(FLEURS_SELECTION.read_text())
    protected = {
        normalized(row["reference"])
        for row in how2sign["rows"] + fleurs["validation"]
    }
    _summary, eligible = summarize_rows(rows, bboxes, protected)
    heldout_yids = {row["yid"] for row in rows if row["split"] in {"valid", "test"}}
    selected, sources = select_candidates(eligible, heldout_yids)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema": "umi-studio-openasl-source-candidates/1",
        "seed": SEED,
        "repository_commit": revision,
        "metadata_sha256": sha256(TSV),
        "bbox_sha256": sha256(BBOX),
        "license_sha256": sha256(LICENSE),
        "metadata_audit_sha256": sha256(ROOT / "runs/openasl-v1/metadata-audit.json"),
        "how2sign_protected_selection_sha256": sha256(HOW2SIGN_SELECTION),
        "fleurs_protected_selection_sha256": sha256(FLEURS_SELECTION),
        "target_sources": TARGET_SOURCES,
        "clips_per_source": CLIPS_PER_SOURCE,
        "selected_sources": sources,
        "selected_clips": len(selected),
        "selected_duration_hours": sum(row["duration_sec"] for row in selected) / 3600,
        "records": [
            {
                "example_id": row["vid"],
                "youtube_id": row["yid"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "duration_sec": row["duration_sec"],
                "reference": row["raw-text"],
                "normalized_reference": row["normalized_reference"],
                "bbox": row["bbox"],
                "source_group": row["source_group"],
                "split": "training",
            }
            for row in selected
        ],
        "freezer_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.rename(args.output)
    print(
        json.dumps(
            {
                "sources": len(sources),
                "clips": len(selected),
                "duration_hours": result["selected_duration_hours"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
