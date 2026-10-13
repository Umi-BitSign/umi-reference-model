"""Freeze a bounded, interpreter-balanced ASL STEM Wiki training selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import zipfile
from collections import Counter
from pathlib import Path

from audit_asl_stem_wiki_metadata import (
    FLEURS_SELECTION,
    HOW2SIGN_SELECTION,
    SOURCE_DIR,
    SOURCE_MANIFEST,
    normalized,
    summarize_rows,
)
from benchmark_fleurs_dev import sha256
from fetch_asl_stem_wiki_metadata import (
    ARCHIVE_NAME,
    ARCHIVE_SIZE,
    AslStemWikiRemoteZip,
    inventory_record,
    validate_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runs/asl-stem-wiki-v1/metadata-audit.json"
TARGET = 1_024
SEED = 20261007
BASE_PARTICIPANT_QUOTA = 30
MAX_PER_ARTICLE = 64
MAX_PER_REFERENCE = 8


def stable_key(seed: int, *values: str) -> str:
    payload = "\0".join((str(seed), *values)).encode()
    return hashlib.sha256(payload).hexdigest()


def participant_quotas(
    eligible: list[dict], target: int, base_quota: int, seed: int
) -> dict[str, int]:
    capacities = Counter(row["participant"] for row in eligible)
    quotas = {
        participant: min(count, base_quota)
        for participant, count in capacities.items()
    }
    remaining = target - sum(quotas.values())
    if remaining < 0:
        raise ValueError("base participant quotas exceed selection target")
    participants = sorted(capacities, key=lambda value: stable_key(seed, value))
    while remaining:
        advanced = False
        for participant in participants:
            if quotas[participant] < capacities[participant]:
                quotas[participant] += 1
                remaining -= 1
                advanced = True
                if remaining == 0:
                    break
        if not advanced:
            raise ValueError("eligible inventory cannot fill selection target")
    return quotas


def select_rows(
    eligible: list[dict],
    *,
    target: int = TARGET,
    base_quota: int = BASE_PARTICIPANT_QUOTA,
    max_per_article: int = MAX_PER_ARTICLE,
    max_per_reference: int = MAX_PER_REFERENCE,
    seed: int = SEED,
) -> tuple[list[dict], dict[str, int]]:
    quotas = participant_quotas(eligible, target, base_quota, seed)
    by_participant = {}
    for participant in quotas:
        by_participant[participant] = sorted(
            (row for row in eligible if row["participant"] == participant),
            key=lambda row: stable_key(
                seed,
                participant,
                row["article"],
                row["section index"],
                row["sentence index"],
                row["video filename"],
            ),
        )
    participants = sorted(quotas, key=lambda value: stable_key(seed, "round", value))
    pointers = Counter()
    selected_counts = Counter()
    article_counts = Counter()
    reference_counts = Counter()
    selected = []
    while len(selected) < target:
        advanced = False
        for participant in participants:
            if selected_counts[participant] >= quotas[participant]:
                continue
            candidates = by_participant[participant]
            while pointers[participant] < len(candidates):
                row = candidates[pointers[participant]]
                pointers[participant] += 1
                if (
                    article_counts[row["article"]] >= max_per_article
                    or reference_counts[row["normalized_reference"]]
                    >= max_per_reference
                ):
                    continue
                selected.append(row)
                selected_counts[participant] += 1
                article_counts[row["article"]] += 1
                reference_counts[row["normalized_reference"]] += 1
                advanced = True
                break
        if not advanced:
            raise ValueError("diversity caps prevent filling the frozen selection")
    if selected_counts != Counter(quotas):
        raise ValueError("selection does not meet participant quotas")
    return selected, quotas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target",
        type=int,
        choices=(1_024, 2_048, 3_072, 4_096),
        default=TARGET,
    )
    parser.add_argument("--require-superset", type=Path)
    parser.add_argument("--exclude-selection", type=Path)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.output.resolve().is_relative_to(ROOT / "runs")
        or (
            args.require_superset is not None
            and not args.require_superset.is_file()
        )
        or (
            args.exclude_selection is not None
            and not args.exclude_selection.is_file()
        )
        or (
            args.require_superset is not None
            and args.exclude_selection is not None
        )
    ):
        parser.error("selection needs a new run-ledger output")
    source = json.loads(SOURCE_MANIFEST.read_text())
    audit = json.loads(AUDIT.read_text())
    if (
        source.get("schema") != "umi-studio-asl-stem-wiki-source/1"
        or audit.get("schema") != "umi-studio-asl-stem-wiki-metadata-audit/1"
        or audit.get("source_manifest_sha256") != sha256(SOURCE_MANIFEST)
    ):
        raise ValueError("ASL STEM Wiki source or audit differs")
    with (SOURCE_DIR / "videos.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    with (SOURCE_DIR / "articles.csv").open(newline="", encoding="utf-8-sig") as stream:
        topics = {row["article"]: row["topic"] for row in csv.DictReader(stream)}
    how2sign = json.loads(HOW2SIGN_SELECTION.read_text())
    fleurs = json.loads(FLEURS_SELECTION.read_text())
    protected = {
        normalized(row["reference"])
        for row in how2sign["rows"] + fleurs["validation"]
    }

    reader = AslStemWikiRemoteZip(
        file_id="direct-download",
        archive_name=ARCHIVE_NAME,
        archive_size=ARCHIVE_SIZE,
    )
    with zipfile.ZipFile(reader) as archive:
        infos = archive.infolist()
    validate_inventory(infos)
    canonical_inventory = json.dumps(
        [inventory_record(info) for info in infos],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    inventory_hash = hashlib.sha256(canonical_inventory).hexdigest()
    if inventory_hash != source["archive_inventory_sha256"]:
        raise ValueError("ASL STEM Wiki archive inventory changed")
    archive_videos = {
        Path(info.filename).name: inventory_record(info)
        for info in infos
        if info.filename.startswith("ASL_STEM_Wiki/videos/") and not info.is_dir()
    }
    summary, eligible = summarize_rows(rows, archive_videos, protected)
    if any(audit.get(key) != value for key, value in summary.items()):
        raise ValueError("ASL STEM Wiki audit summary is not reproducible")
    excluded_hash = None
    excluded_ids = set()
    if args.exclude_selection is not None:
        excluded = json.loads(args.exclude_selection.read_text())
        excluded_ids = {row["example_id"] for row in excluded.get("records", [])}
        if (
            excluded.get("schema")
            != "umi-studio-asl-stem-wiki-training-selection/1"
            or excluded.get("training_only") is not True
            or len(excluded_ids) != excluded.get("selected_examples")
        ):
            raise ValueError("excluded ASL STEM Wiki selection differs")
        excluded_hash = sha256(args.exclude_selection)
        eligible = [
            row
            for row in eligible
            if Path(row["video filename"]).stem not in excluded_ids
        ]
    selected, quotas = select_rows(eligible, target=args.target)
    article_counts = Counter(row["article"] for row in selected)
    reference_counts = Counter(row["normalized_reference"] for row in selected)
    records = []
    for row in selected:
        archive = row["archive"]
        records.append(
            {
                "schema": "umi-studio-asl-stem-wiki-training-example/1",
                "example_id": Path(row["video filename"]).stem,
                "split": "training",
                "participant": row["participant"],
                "article": row["article"],
                "topic": topics[row["article"]],
                "section_index": int(row["section index"]),
                "sentence_index": int(row["sentence index"]),
                "reference": row["sentence text"],
                "normalized_reference": row["normalized_reference"],
                "duration_sec": row["duration_sec"],
                "source_group": (
                    f"asl-stem-wiki:{row['participant']}:{row['article']}"
                ),
                "archive_member": archive["name"],
                "archive_crc32": archive["crc32"],
                "archive_compressed_size": archive["compressed_size"],
                "archive_file_size": archive["file_size"],
                "archive_compression": archive["compression"],
                "archive_header_offset": archive["header_offset"],
            }
        )
    superset_hash = None
    if args.require_superset is not None:
        prior = json.loads(args.require_superset.read_text())
        prior_ids = {row["example_id"] for row in prior.get("records", [])}
        selected_ids = {row["example_id"] for row in records}
        if (
            prior.get("schema")
            != "umi-studio-asl-stem-wiki-training-selection/1"
            or prior.get("training_only") is not True
            or len(prior_ids) != prior.get("selected_examples")
            or not prior_ids.issubset(selected_ids)
        ):
            raise ValueError("required ASL STEM Wiki selection is not a subset")
        superset_hash = sha256(args.require_superset)
    selection = {
        "schema": "umi-studio-asl-stem-wiki-training-selection/1",
        "training_only": True,
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "metadata_audit_sha256": sha256(AUDIT),
        "archive_inventory_sha256": inventory_hash,
        "videos_csv_sha256": sha256(SOURCE_DIR / "videos.csv"),
        "articles_csv_sha256": sha256(SOURCE_DIR / "articles.csv"),
        "license_sha256": sha256(SOURCE_DIR / "use.txt"),
        "how2sign_protected_selection_sha256": sha256(HOW2SIGN_SELECTION),
        "fleurs_protected_selection_sha256": sha256(FLEURS_SELECTION),
        "seed": SEED,
        "target_examples": args.target,
        "base_participant_quota": BASE_PARTICIPANT_QUOTA,
        "max_per_article": MAX_PER_ARTICLE,
        "max_per_normalized_reference": MAX_PER_REFERENCE,
        "selected_examples": len(records),
        "selected_participants": len(quotas),
        "selected_articles": len(article_counts),
        "selected_unique_normalized_references": len(reference_counts),
        "selected_duration_hours": sum(row["duration_sec"] for row in records)
        / 3600,
        "selected_compressed_bytes": sum(
            row["archive_compressed_size"] for row in records
        ),
        "selected_file_bytes": sum(row["archive_file_size"] for row in records),
        "participant_quotas": dict(sorted(quotas.items())),
        "participant_counts": dict(
            sorted(Counter(row["participant"] for row in records).items())
        ),
        "article_count_max": max(article_counts.values()),
        "reference_count_max": max(reference_counts.values()),
        "duration_sec_summary": {
            "min": min(row["duration_sec"] for row in records),
            "median": statistics.median(row["duration_sec"] for row in records),
            "max": max(row["duration_sec"] for row in records),
        },
        "records": records,
        "superset_selection_sha256": superset_hash,
        "excluded_selection_sha256": excluded_hash,
        "excluded_examples": len(excluded_ids),
        "freezer_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(selection, output, sort_keys=True, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.rename(args.output)
    print(
        json.dumps(
            {
                key: selection[key]
                for key in (
                    "selected_examples",
                    "selected_participants",
                    "selected_articles",
                    "selected_unique_normalized_references",
                    "selected_duration_hours",
                    "selected_compressed_bytes",
                    "article_count_max",
                    "reference_count_max",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
