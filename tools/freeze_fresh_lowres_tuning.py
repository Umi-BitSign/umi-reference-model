"""Freeze the already-open tuning clips for fresh low-resolution evaluation."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import sha256

SCHEMA = "umi-studio-fresh-lowres-tuning-example/1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _relative_video(workspace: Path, directory: Path, example_id: str) -> str:
    video = (directory / f"{example_id}.mp4").resolve()
    try:
        relative = video.relative_to(workspace.resolve())
    except ValueError as error:
        raise ValueError("tuning video is outside the workspace") from error
    if not video.is_file():
        raise ValueError(f"tuning video is missing: {relative}")
    return relative.as_posix()


def fleurs_rows(
    workspace: Path,
    selection_path: Path,
    partition_path: Path,
    index_path: Path,
    video_dir: Path,
) -> list[dict]:
    selection = json.loads(selection_path.read_text())
    partition = json.loads(partition_path.read_text())
    index = read_jsonl(index_path)
    if (
        selection.get("schema") != "umi-studio-fleurs-dev-split/1"
        or partition.get("schema") != "umi-studio-fleurs-dev-evaluation-partition/1"
        or partition.get("selection_sha256") != sha256(selection_path)
        or partition.get("tuning_examples") != 7
    ):
        raise ValueError("unexpected FLEURS tuning partition")
    tuning_documents = set(partition["tuning_documents"])
    selected = {
        row["example_id"]: row
        for row in selection["validation"]
        if row["document_id"] in tuning_documents
    }
    if len(selected) != partition["tuning_examples"] or len(index) != len(selected):
        raise ValueError("FLEURS tuning index is incomplete")
    output = []
    for row in index:
        example_id = row.get("example_id")
        chosen = selected.get(example_id)
        if (
            row.get("schema") != "umi-studio-fleurs-dev-encoder-cache-example/1"
            or row.get("split") != "tuning"
            or row.get("clip_fps") != 25
            or chosen is None
            or row.get("reference") != chosen.get("reference")
            or row.get("signer") != chosen.get("signer")
            or row.get("document_id") != chosen.get("document_id")
            or not isinstance(row.get("clip_sha256"), str)
        ):
            raise ValueError(f"FLEURS tuning row differs: {example_id}")
        output.append(
            {
                "schema": SCHEMA,
                "split": "tuning",
                "source": "fleurs",
                "example_id": example_id,
                "source_group": f"fleurs:{row['document_id']}",
                "signer": row["signer"],
                "reference": row["reference"],
                "video": _relative_video(workspace, video_dir, example_id),
                "video_sha256": row["clip_sha256"],
                "video_fps": 25,
                "source_index_sha256": sha256(index_path),
                "source_selection_sha256": sha256(selection_path),
                "source_partition_sha256": sha256(partition_path),
            }
        )
    return output


def how2sign_rows(
    workspace: Path,
    selection_path: Path,
    partition_path: Path,
    index_path: Path,
    video_dir: Path,
) -> list[dict]:
    selection = json.loads(selection_path.read_text())
    partition = json.loads(partition_path.read_text())
    index = read_jsonl(index_path)
    if (
        selection.get("schema") != "umi-studio-how2sign-validation-selection/1"
        or selection.get("training_excluded") is not True
        or partition.get("schema")
        != "umi-studio-how2sign-validation-partition/1"
        or partition.get("selection_sha256") != sha256(selection_path)
        or partition.get("confirmation_predictions_inspected") is not False
    ):
        raise ValueError("unexpected How2Sign tuning partition")
    names = partition.get("tuning_sentence_names")
    if not isinstance(names, list) or len(names) != 16 or len(set(names)) != 16:
        raise ValueError("unexpected How2Sign tuning names")
    selected = {row["sentence_name"]: row for row in selection["rows"]}
    indexed = {row["sentence_name"]: row for row in index}
    output = []
    for name in names:
        chosen = selected.get(name)
        row = indexed.get(name)
        if (
            chosen is None
            or row is None
            or row.get("schema") != "umi-studio-how2sign-encoder-cache-example/1"
            or row.get("split") != "validation"
            or row.get("clip_fps") != 25
            or row.get("selection_sha256") != sha256(selection_path)
            or row.get("reference") != chosen.get("reference")
            or row.get("signer") != chosen.get("signer")
            or not isinstance(row.get("resampled_clip_sha256"), str)
        ):
            raise ValueError(f"How2Sign tuning row differs: {name}")
        output.append(
            {
                "schema": SCHEMA,
                "split": "tuning",
                "source": "how2sign",
                "example_id": name,
                "source_group": f"how2sign:{chosen['video_id']}",
                "signer": chosen["signer"],
                "reference": chosen["reference"],
                "video": _relative_video(workspace, video_dir, name),
                "video_sha256": row["resampled_clip_sha256"],
                "video_fps": 25,
                "source_index_sha256": sha256(index_path),
                "source_selection_sha256": sha256(selection_path),
                "source_partition_sha256": sha256(partition_path),
            }
        )
    return output


def validate_disjoint(rows: list[dict], training_path: Path) -> dict[str, int]:
    training = read_jsonl(training_path)
    training_groups = {row["source_group"] for row in training}
    identities = {(row["source"], row["example_id"]) for row in rows}
    required = {
        "schema",
        "split",
        "source",
        "example_id",
        "source_group",
        "signer",
        "reference",
        "video",
        "video_sha256",
        "video_fps",
        "source_index_sha256",
        "source_selection_sha256",
        "source_partition_sha256",
    }
    if len(identities) != len(rows):
        raise ValueError("fresh tuning inventory repeats an identity")
    for row in rows:
        if (
            set(row) != required
            or row["schema"] != SCHEMA
            or row["split"] != "tuning"
            or row["source_group"] in training_groups
            or type(row["signer"]) is not int
            or not row["reference"].strip()
            or len(row["video_sha256"]) != 64
        ):
            raise ValueError(f"invalid or training-overlapping tuning row: {row}")
    return dict(sorted(Counter(row["source"] for row in rows).items()))


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError("existing fresh tuning inventory differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.jsonl")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-inventory", type=Path, required=True)
    parser.add_argument("--fleurs-selection", type=Path, required=True)
    parser.add_argument("--fleurs-partition", type=Path, required=True)
    parser.add_argument("--fleurs-index", type=Path, required=True)
    parser.add_argument("--fleurs-video-dir", type=Path, required=True)
    parser.add_argument("--how2sign-selection", type=Path, required=True)
    parser.add_argument("--how2sign-partition", type=Path, required=True)
    parser.add_argument("--how2sign-index", type=Path, required=True)
    parser.add_argument("--how2sign-video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    rows = [
        *how2sign_rows(
            workspace,
            args.how2sign_selection,
            args.how2sign_partition,
            args.how2sign_index,
            args.how2sign_video_dir,
        ),
        *fleurs_rows(
            workspace,
            args.fleurs_selection,
            args.fleurs_partition,
            args.fleurs_index,
            args.fleurs_video_dir,
        ),
    ]
    counts = validate_disjoint(rows, args.training_inventory)
    atomic_jsonl(args.output, rows)
    print(
        f"frozen {len(rows)} tuning examples sha256={sha256(args.output)} "
        f"counts={json.dumps(counts, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
