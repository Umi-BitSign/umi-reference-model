"""Audit the publisher's ChicagoFSWild metadata bundle without fetching frames."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data/chicagofswild/metadata/csv.zip"
MEMBERS = ("ChicagoFSWild.csv", "ChicagoFSWildPlus.csv")
FIELDS = (
    "",
    "filename",
    "url",
    "start_time",
    "number_of_frames",
    "width",
    "height",
    "label_proc",
    "label_raw",
    "label_notes",
    "partition",
    "signer",
)
PARTITIONS = ("train", "dev", "test")
PUBLISHER_COUNTS = {
    "ChicagoFSWild.csv": {"sequences": 7_304, "signers": 160},
    "ChicagoFSWildPlus.csv": {"sequences": 55_232, "signers": 260},
}
PUBLISHER_URL = "https://home.ttic.edu/~klivescu/ChicagoFSWild.htm"
METADATA_URL = (
    "https://drive.google.com/uc?export=download&id="
    "1rDahGBMj0v-28mxyHJiZRwiFbhWF9PN_"
)


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def usable_source_url(value: str) -> str | None:
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return None


def read_member(archive: zipfile.ZipFile, name: str) -> tuple[list[dict[str, str]], str]:
    raw = archive.read(name)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError(f"unexpected columns in {name}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"empty metadata member: {name}")
    filenames = set()
    for position, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"malformed CSV row in {name}")
        if row[""] != str(position):
            raise ValueError(f"noncontiguous row index in {name}")
        if not row["filename"] or row["filename"] in filenames:
            raise ValueError(f"empty or duplicate filename in {name}")
        filenames.add(row["filename"])
        if row["partition"] not in PARTITIONS:
            raise ValueError(f"unknown partition in {name}")
        for field in ("number_of_frames", "width", "height"):
            if int(row[field]) <= 0:
                raise ValueError(f"nonpositive {field} in {name}")
        if not row["start_time"] or not row["signer"] or int(row["signer"]) < 0:
            raise ValueError(f"invalid time or signer in {name}")
    return rows, bytes_sha256(raw)


def split_overlap(rows: list[dict[str, str]]) -> dict[str, dict]:
    signers = {
        split: {row["signer"] for row in rows if row["partition"] == split}
        for split in PARTITIONS
    }
    urls = {
        split: {
            url
            for row in rows
            if row["partition"] == split
            if (url := usable_source_url(row["url"])) is not None
        }
        for split in PARTITIONS
    }
    result = {}
    for left, right in combinations(PARTITIONS, 2):
        shared_signers = sorted(signers[left] & signers[right], key=int)
        shared_urls = sorted(urls[left] & urls[right])
        result[f"{left}_{right}"] = {
            "shared_signer_count": len(shared_signers),
            "shared_signers": shared_signers,
            "shared_usable_source_url_count": len(shared_urls),
            "shared_usable_source_urls": shared_urls,
        }
    return result


def candidate_interval_repetitions(rows: list[dict[str, str]]) -> dict[str, int]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["url"], row["start_time"], row["number_of_frames"])].append(row)
    repeated = [group for group in groups.values() if len(group) > 1]
    return {
        "groups": len(repeated),
        "excess_rows": sum(len(group) - 1 for group in repeated),
        "groups_with_label_disagreement": sum(
            len({row["label_proc"] for row in group}) > 1 for group in repeated
        ),
        "groups_crossing_partitions": sum(
            len({row["partition"] for row in group}) > 1 for group in repeated
        ),
        "largest_group": max((len(group) for group in repeated), default=1),
    }


def conservative_train_filter(rows: list[dict[str, str]]) -> dict[str, int | bool]:
    held_out = [row for row in rows if row["partition"] != "train"]
    protected_signers = {row["signer"] for row in held_out}
    protected_urls = {
        url
        for row in held_out
        if (url := usable_source_url(row["url"])) is not None
    }
    training = [row for row in rows if row["partition"] == "train"]
    signer_excluded = {
        row["filename"] for row in training if row["signer"] in protected_signers
    }
    url_excluded = {
        row["filename"]
        for row in training
        if usable_source_url(row["url"]) in protected_urls
    }
    missing_url = sum(usable_source_url(row["url"]) is None for row in training)
    return {
        "publisher_train_rows": len(training),
        "rows_excluded_for_held_out_signer": len(signer_excluded),
        "rows_excluded_for_held_out_usable_source_url": len(url_excluded),
        "rows_excluded_by_union": len(signer_excluded | url_excluded),
        "remaining_rows": len(training) - len(signer_excluded | url_excluded),
        "training_rows_without_usable_source_url": missing_url,
        "source_url_check_complete": missing_url == 0,
    }


def summarize(rows: list[dict[str, str]], member_sha256: str, name: str) -> dict:
    split_counts = Counter(row["partition"] for row in rows)
    labels = [row["label_proc"] for row in rows]
    frame_counts = [int(row["number_of_frames"]) for row in rows]
    dimensions = Counter(f"{row['width']}x{row['height']}" for row in rows)
    return {
        "member_sha256": member_sha256,
        "metadata_rows": len(rows),
        "publisher_page_sequence_count": PUBLISHER_COUNTS[name]["sequences"],
        "publisher_page_sequence_count_difference": (
            len(rows) - PUBLISHER_COUNTS[name]["sequences"]
        ),
        "unique_signers": len({row["signer"] for row in rows}),
        "publisher_page_signer_count": PUBLISHER_COUNTS[name]["signers"],
        "publisher_page_signer_count_difference": (
            len({row["signer"] for row in rows}) - PUBLISHER_COUNTS[name]["signers"]
        ),
        "unique_usable_source_urls": len(
            {
                url
                for row in rows
                if (url := usable_source_url(row["url"])) is not None
            }
        ),
        "rows_without_usable_source_url": sum(
            usable_source_url(row["url"]) is None for row in rows
        ),
        "total_frames": sum(frame_counts),
        "minimum_frames": min(frame_counts),
        "maximum_frames": max(frame_counts),
        "blank_processed_labels": sum(not label for label in labels),
        "minimum_processed_label_characters": min(map(len, labels)),
        "maximum_processed_label_characters": max(map(len, labels)),
        "processed_label_character_inventory": "".join(sorted(set("".join(labels)))),
        "most_common_dimensions": [
            {"dimensions": dimensions, "rows": count}
            for dimensions, count in dimensions.most_common(10)
        ],
        "partitions": {
            split: {
                "rows": split_counts[split],
                "unique_signers": len(
                    {row["signer"] for row in rows if row["partition"] == split}
                ),
                "unique_usable_source_urls": len(
                    {
                        url
                        for row in rows
                        if row["partition"] == split
                        if (url := usable_source_url(row["url"])) is not None
                    }
                ),
            }
            for split in PARTITIONS
        },
        "split_overlap": split_overlap(rows),
        "candidate_repeated_interval_keys": candidate_interval_repetitions(rows),
        "conservative_train_filter": conservative_train_filter(rows),
    }


def audit_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        if tuple(archive.namelist()) != MEMBERS:
            raise ValueError("metadata archive members differ from the expected bundle")
        loaded = {name: read_member(archive, name) for name in MEMBERS}
    rows_by_member = {name: value[0] for name, value in loaded.items()}
    usable_urls = {
        name: {
            url
            for row in rows
            if (url := usable_source_url(row["url"])) is not None
        }
        for name, rows in rows_by_member.items()
    }
    filenames = {
        name: {row["filename"] for row in rows}
        for name, rows in rows_by_member.items()
    }
    return {
        "schema": "umi-studio-chicagofswild-metadata-audit/1",
        "publisher_url": PUBLISHER_URL,
        "metadata_url": METADATA_URL,
        "metadata_archive_sha256": sha256(path),
        "metadata_archive_size_bytes": path.stat().st_size,
        "metadata_only": True,
        "frame_archives_downloaded": False,
        "test_media_downloaded": False,
        "test_rows_used_for_model_selection": False,
        "members": {
            name: summarize(rows, loaded[name][1], name)
            for name, rows in rows_by_member.items()
        },
        "cross_release_overlap": {
            "filenames": len(filenames[MEMBERS[0]] & filenames[MEMBERS[1]]),
            "usable_source_urls": len(usable_urls[MEMBERS[0]] & usable_urls[MEMBERS[1]]),
        },
        "limits": [
            "The publisher page does not state an explicit reuse license.",
            "Frame bytes and metadata-to-frame membership were not checked.",
            "Missing or placeholder source URLs prevent a complete source-video split audit.",
            "Repeated interval keys are metadata-level duplicate candidates, not frame-identity proof.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("audit output must be new and under runs/")
    result = audit_archive(args.archive)
    result["audit_source_sha256"] = sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.json")
    if partial.exists():
        parser.error("unfinished audit output needs inspection")
    try:
        with partial.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        partial.rename(args.output)
    finally:
        partial.unlink(missing_ok=True)
    print(json.dumps({
        name: {
            "metadata_rows": summary["metadata_rows"],
            "remaining_train_rows": summary["conservative_train_filter"]["remaining_rows"],
        }
        for name, summary in result["members"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
