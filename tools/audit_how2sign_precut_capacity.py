"""Count aligned pre-cut training intervals beyond the frozen four-per-video cap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data/how2sign/source/how2sign_train.csv"
REALIGNED = ROOT / "data/how2sign/source/how2sign_realigned_train.csv"
SELECTION = ROOT / "runs/how2sign-expanded-v1/selection.json"
SIGNERS = (1, 2, 3, 4, 5, 8, 9, 11)


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    indexed = {row["SENTENCE_NAME"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate How2Sign sentence names: {path}")
    return indexed


def aligned_eligible(original: dict[str, str], realigned: dict[str, str]) -> bool:
    if any(
        original[key] != realigned[key]
        for key in ("VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE")
    ):
        raise ValueError("original and realigned How2Sign identities differ")
    start, end = float(original["START"]), float(original["END"])
    aligned_start = float(realigned["START_REALIGNED"])
    aligned_end = float(realigned["END_REALIGNED"])
    duration = end - start
    aligned_duration = aligned_end - aligned_start
    if not all(math.isfinite(value) for value in (start, end, aligned_start, aligned_end)):
        raise ValueError("nonfinite How2Sign interval")
    if not 2 <= duration <= 12 or len(original["SENTENCE"].split()) < 3:
        return False
    if aligned_start < 0 or aligned_duration <= 0:
        raise ValueError("invalid realigned How2Sign interval")
    overlap = max(0.0, min(end, aligned_end) - max(start, aligned_start))
    return overlap / duration >= 0.8 and overlap / aligned_duration >= 0.8


def audit(original: dict[str, dict], realigned: dict[str, dict], selected: list[dict]) -> dict:
    if set(original) != set(realigned):
        raise ValueError("original and realigned How2Sign sentence sets differ")
    selected_ids = {row["sentence_name"] for row in selected}
    if len(selected_ids) != len(selected) or not selected_ids <= set(original):
        raise ValueError("frozen selection has repeated or unknown How2Sign sentences")
    counts = Counter()
    durations = Counter()
    videos = defaultdict(set)
    selected_counts = Counter()
    selected_durations = Counter()
    for name, row in original.items():
        if not aligned_eligible(row, realigned[name]):
            continue
        signer = int(row["VIDEO_NAME"].rsplit("-", 2)[1])
        if signer not in SIGNERS:
            raise ValueError("unknown How2Sign signer")
        counts[signer] += 1
        durations[signer] += float(row["END"]) - float(row["START"])
        videos[signer].add(row["VIDEO_NAME"])
    for row in selected:
        signer = row["signer"]
        if signer not in SIGNERS or not aligned_eligible(original[row["sentence_name"]], realigned[row["sentence_name"]]):
            raise ValueError("frozen selection does not pass the metadata filter")
        selected_counts[signer] += 1
        selected_durations[signer] += row["end_sec"] - row["start_sec"]
    return {
        "metadata_eligible_examples": sum(counts.values()),
        "selected_examples": sum(selected_counts.values()),
        "metadata_only_unselected_examples": sum(counts.values()) - sum(selected_counts.values()),
        "per_signer": {
            str(signer): {
                "metadata_eligible_examples": counts[signer],
                "selected_examples": selected_counts[signer],
                "metadata_only_unselected_examples": counts[signer] - selected_counts[signer],
                "metadata_eligible_source_videos": len(videos[signer]),
                "metadata_eligible_original_interval_seconds": round(durations[signer], 6),
                "selected_original_interval_seconds": round(selected_durations[signer], 6),
            }
            for signer in SIGNERS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("capacity audit needs a new run-ledger output")
    selection = json.loads(SELECTION.read_text())
    if (
        selection.get("schema") != "umi-studio-how2sign-training-aligned/1"
        or selection.get("metadata_sha256") != sha256(ORIGINAL)
        or selection.get("alignment_metadata_sha256") != sha256(REALIGNED)
        or selection.get("max_clips_per_source_video") != 4
        or selection.get("min_bidirectional_overlap") != 0.8
    ):
        raise ValueError("frozen How2Sign selection provenance differs")
    result = {
        "schema": "umi-studio-how2sign-precut-capacity-audit/1",
        "split": "training",
        "metadata_only": True,
        "archive_membership_and_file_size_verified": False,
        "source_bytes_or_video_duration_verified": False,
        "selection_sha256": sha256(SELECTION),
        "original_metadata_sha256": sha256(ORIGINAL),
        "realigned_metadata_sha256": sha256(REALIGNED),
        "audit_source_sha256": sha256(Path(__file__)),
        "counts": audit(read_rows(ORIGINAL), read_rows(REALIGNED), selection["rows"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished capacity audit needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
