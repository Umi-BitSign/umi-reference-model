"""Select 2M-Flores-ASL dev training clips without FLEURS validation leakage.

All source articles containing a FLEURS validation sentence are excluded. Exact
FLEURS training text is also excluded to add new labels to the training mix.
Only pinned public dev video files are eligible; devtest is never queried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_fleurs_dev import normalize, sha256

SEED = 17
MAX_VIDEO_BYTES = 100_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--fleurs-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    if args.count < 1 or args.output.exists():
        parser.error("count must be positive and output must be new")
    inventory = json.loads(args.inventory.read_text())
    files = json.loads(args.files.read_text())
    fleurs = json.loads(args.fleurs_selection.read_text())
    if (
        inventory["schema"] != "umi-studio-flores2m-dev-inventory/1"
        or files["schema"] != "umi-studio-flores2m-dev-files/1"
        or inventory["repository_revision"] != files["repository_revision"]
        or fleurs["schema"] != "umi-studio-fleurs-dev-split/1"
    ):
        raise ValueError("source inventory or FLEURS selection schema differs")
    by_text = {}
    for row in inventory["rows"]:
        by_text.setdefault(normalize(row["sentence"]), set()).add(row["source_page"])
    validation_text = {normalize(row["reference"]) for row in fleurs["validation"]}
    if len(validation_text) != len(fleurs["validation"]):
        raise ValueError("FLEURS validation has duplicate normalized translations")
    if any(text not in by_text for text in validation_text):
        raise ValueError("a FLEURS validation sentence was not found in 2M-Flores dev")
    protected_pages = {page for text in validation_text for page in by_text[text]}
    train_text = {normalize(row["reference"]) for row in fleurs["training"]}
    by_filename = {Path(row["path"]).name: row for row in files["files"]}
    eligible = []
    for row in inventory["rows"]:
        name = row["video_url"].rsplit("/", 1)[1]
        file = by_filename[name]
        if (
            row["source_page"] in protected_pages
            or normalize(row["sentence"]) in train_text
            or file["size_bytes"] > MAX_VIDEO_BYTES
            or len(row["sentence"].split()) < 5
        ):
            continue
        eligible.append(
            {
                **row,
                "video_size_bytes": file["size_bytes"],
                "video_sha256": file["sha256"],
            }
        )
    ordered = sorted(
        eligible,
        key=lambda row: (
            hashlib.sha256(f"{SEED}:{row['video_url']}".encode()).hexdigest(),
            row["video_url"],
        ),
    )
    selected = []
    selected_pages = set()
    for row in ordered:
        if row["source_page"] in selected_pages:
            continue
        selected.append(row)
        selected_pages.add(row["source_page"])
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise ValueError("not enough eligible source articles for the requested pilot")
    result = {
        "schema": "umi-studio-flores2m-training-pilot/1",
        "source_split": "dev",
        "devtest_used": False,
        "repository_revision": inventory["repository_revision"],
        "inventory_sha256": sha256(args.inventory),
        "files_sha256": sha256(args.files),
        "fleurs_selection_sha256": sha256(args.fleurs_selection),
        "selection_seed": SEED,
        "max_source_video_bytes": MAX_VIDEO_BYTES,
        "fleurs_validation_sentences_excluded": len(validation_text),
        "protected_source_pages_excluded": len(protected_pages),
        "protected_source_page_hashes": sorted(
            hashlib.sha256(page.encode()).hexdigest() for page in protected_pages
        ),
        "eligible_new_text_rows": len(eligible),
        "rows": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        f"selected {len(selected)} dev clips from distinct source articles; "
        f"total source size {sum(row['video_size_bytes'] for row in selected) / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()
