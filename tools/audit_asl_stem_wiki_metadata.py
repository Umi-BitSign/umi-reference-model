"""Audit ASL STEM Wiki metadata against its remote ZIP and protected references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import normalize, sha256
from fetch_asl_stem_wiki_metadata import (
    ARCHIVE_NAME,
    ARCHIVE_SIZE,
    AslStemWikiRemoteZip,
    inventory_record,
    validate_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "runs/asl-stem-wiki-v1/source-manifest.json"
SOURCE_DIR = ROOT / "data/asl-stem-wiki/source"
HOW2SIGN_SELECTION = ROOT / "runs/how2sign-validation-v1/selection.json"
FLEURS_SELECTION = ROOT / "runs/fleurs-dev-v1/selection.json"
MAX_COMPRESSED_BYTES = 32 * 1024 * 1024


def normalized(text: str) -> str:
    return " ".join(normalize(text))


def summarize_rows(
    rows: list[dict], archive_videos: dict[str, dict], protected: set[str]
) -> tuple[dict, list[dict]]:
    filenames = Counter(row["video filename"] for row in rows)
    identities = Counter(
        (
            row["participant"],
            row["article"],
            row["section index"],
            row["sentence index"],
        )
        for row in rows
    )
    references = Counter(normalized(row["sentence text"]) for row in rows)
    eligible = []
    invalid_numeric = 0
    for row in rows:
        try:
            duration = float(row["video length"])
        except ValueError:
            invalid_numeric += 1
            continue
        reference = normalized(row["sentence text"])
        words = len(reference.split())
        member = archive_videos.get(row["video filename"])
        if (
            member is not None
            and filenames[row["video filename"]] == 1
            and identities[
                (
                    row["participant"],
                    row["article"],
                    row["section index"],
                    row["sentence index"],
                )
            ]
            == 1
            and 2 <= duration <= 12
            and 3 <= words <= 30
            and reference
            and reference not in protected
            and member["compressed_size"] <= MAX_COMPRESSED_BYTES
        ):
            eligible.append(
                {
                    **row,
                    "duration_sec": duration,
                    "normalized_reference": reference,
                    "word_count": words,
                    "archive": member,
                }
            )
    eligible_participants = Counter(row["participant"] for row in eligible)
    eligible_articles = Counter(row["article"] for row in eligible)
    summary = {
        "metadata_rows": len(rows),
        "unique_metadata_filenames": len(filenames),
        "archive_videos": len(archive_videos),
        "metadata_rows_missing_archive_video": sum(
            count for name, count in filenames.items() if name not in archive_videos
        ),
        "archive_videos_missing_metadata_row": len(set(archive_videos) - set(filenames)),
        "duplicate_filename_rows": sum(
            count for count in filenames.values() if count > 1
        ),
        "duplicate_identity_rows": sum(
            count for count in identities.values() if count > 1
        ),
        "duplicate_reference_rows": sum(
            count for reference, count in references.items() if reference and count > 1
        ),
        "invalid_numeric_rows": invalid_numeric,
        "participants": len({row["participant"] for row in rows}),
        "articles": len({row["article"] for row in rows}),
        "protected_exact_reference_rows": sum(
            count for reference, count in references.items() if reference in protected
        ),
        "eligible_rows": len(eligible),
        "eligible_participants": len(eligible_participants),
        "eligible_articles": len(eligible_articles),
        "eligible_duration_hours": sum(row["duration_sec"] for row in eligible) / 3600,
        "eligible_compressed_bytes": sum(
            row["archive"]["compressed_size"] for row in eligible
        ),
        "eligible_by_participant": dict(sorted(eligible_participants.items())),
        "eligible_by_article": dict(sorted(eligible_articles.items())),
    }
    return summary, eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("audit needs a new run-ledger output")
    source = json.loads(SOURCE_MANIFEST.read_text())
    if source.get("schema") != "umi-studio-asl-stem-wiki-source/1":
        raise ValueError("ASL STEM Wiki source manifest differs")
    for member in source["metadata_members"]:
        path = SOURCE_DIR / member["local_name"]
        if sha256(path) != member["sha256"]:
            raise ValueError(f"ASL STEM Wiki metadata changed: {path}")
    with (SOURCE_DIR / "videos.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "participant",
        "article",
        "section index",
        "sentence index",
        "video filename",
        "video length",
        "sentence text",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError("ASL STEM Wiki video metadata columns differ")

    reader = AslStemWikiRemoteZip(
        file_id="direct-download",
        archive_name=ARCHIVE_NAME,
        archive_size=ARCHIVE_SIZE,
    )
    import zipfile

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
    how2sign = json.loads(HOW2SIGN_SELECTION.read_text())
    fleurs = json.loads(FLEURS_SELECTION.read_text())
    protected = {
        normalized(row["reference"])
        for row in how2sign["rows"] + fleurs["validation"]
    }
    summary, _ = summarize_rows(rows, archive_videos, protected)
    result = {
        "schema": "umi-studio-asl-stem-wiki-metadata-audit/1",
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "archive_inventory_sha256": inventory_hash,
        "videos_csv_sha256": sha256(SOURCE_DIR / "videos.csv"),
        "license_sha256": sha256(SOURCE_DIR / "use.txt"),
        "how2sign_protected_selection_sha256": sha256(HOW2SIGN_SELECTION),
        "fleurs_protected_selection_sha256": sha256(FLEURS_SELECTION),
        "filters": {
            "duration_sec": [2, 12],
            "word_count": [3, 30],
            "max_compressed_bytes": MAX_COMPRESSED_BYTES,
            "unique_filename": True,
            "unique_participant_article_sentence": True,
            "exclude_protected_exact_reference": True,
        },
        **summary,
        "range_requests": reader.requests,
        "range_bytes_downloaded": reader.bytes_downloaded,
        "auditor_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    temporary.rename(args.output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
