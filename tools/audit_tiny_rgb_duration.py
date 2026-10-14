"""Quarantine severely truncated raw clips using annotations, never model scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmark_fleurs_dev import sha256
from tiny_rgb_full_corpus import read_rows
from tiny_rgb_full_run import ROOT, checked_full_plan, training_inventory

PARENT = ROOT / "runs/tiny-rgb-full-v3-buckets/plan.json"
ANNOTATIONS = ROOT / "data/how2sign/source/how2sign_realigned_train.csv"


def findings(rows, headers, annotations):
    indexed = {(r["source"], r["example_id"]): r for r in headers if r["role"] == "training"}
    metadata = {r["SENTENCE_NAME"]: r for r in annotations}
    rejected = []
    for row in rows:
        if not row["source"].startswith("how2sign"):
            continue
        annotation = metadata[row["example_id"]]
        header = indexed[(row["source"], row["example_id"])]
        expected = float(annotation["END_REALIGNED"]) - float(annotation["START_REALIGNED"])
        actual = header["source_frames"] / header["source_fps"]
        if expected <= 0 or actual <= 0:
            raise ValueError("duration audit needs positive source and annotated intervals")
        if expected >= 0.5 and actual < 0.5 * expected:
            rejected.append(
                {
                    "source": row["source"],
                    "example_id": row["example_id"],
                    "video": row["video"],
                    "video_sha256": row["video_sha256"],
                    "expected_duration_seconds": expected,
                    "actual_duration_seconds": actual,
                    "fraction_of_annotated_duration": actual / expected,
                    "reason": "raw_clip_has_less_than_half_annotated_signing_interval",
                    "role": "quarantined_raw_duration",
                    "training_allowed": False,
                    "original_bytes_preserved": True,
                    "recovery_pending": True,
                }
            )
    return sorted(rejected, key=lambda r: (r["source"], r["example_id"]))


def checked_duration_audit(path):
    report = json.loads(path.read_text())
    parent = checked_full_plan(PARENT)
    rows = training_inventory(parent)
    headers = ROOT / parent["headers"]
    if (
        report["parent_plan_sha256"] != sha256(PARENT)
        or report["headers_sha256"] != sha256(headers)
        or report["annotations_sha256"] != sha256(ANNOTATIONS)
        or report["source_sha256"] != sha256(Path(__file__))
        or report["quarantine"]
        != findings(
            rows, read_rows(headers), list(csv.DictReader(ANNOTATIONS.open(), delimiter="\t"))
        )
    ):
        raise ValueError("duration quarantine evidence changed")
    bad = {(r["source"], r["example_id"]) for r in report["quarantine"]}
    if report["valid_training_clips"] != len(rows) - len(bad):
        raise ValueError("duration quarantine counts differ")
    return report, [r for r in rows if (r["source"], r["example_id"]) not in bad]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("duration audit needs a new runs/ directory")
    parent = checked_full_plan(PARENT)
    rows = training_inventory(parent)
    headers = ROOT / parent["headers"]
    quarantine = findings(
        rows, read_rows(headers), list(csv.DictReader(ANNOTATIONS.open(), delimiter="\t"))
    )
    report = {
        "schema": "umi-studio-tiny-rgb-duration-integrity/1",
        "parent_plan_sha256": sha256(PARENT),
        "headers_sha256": sha256(headers),
        "annotations_sha256": sha256(ANNOTATIONS),
        "source_sha256": sha256(Path(__file__)),
        "original_training_clips": len(rows),
        "quarantine": quarantine,
        "valid_training_clips": len(rows) - len(quarantine),
        "model_scores_used": False,
        "original_bytes_deleted": False,
        "qualification": "Annotation-duration integrity only; no semantic alignment guarantee. "
        "Retain source files and repair/re-audit quarantined clips before re-entry.",
    }
    output.mkdir(parents=True)
    path = output / "receipt.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    checked_duration_audit(path)
    print(
        json.dumps(
            {
                "receipt_sha256": sha256(path),
                "quarantined": len(quarantine),
                "valid_training_clips": report["valid_training_clips"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
