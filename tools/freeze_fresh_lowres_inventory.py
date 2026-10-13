"""Freeze raw-frame training examples for the fresh low-resolution model."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path

from benchmark_fleurs_dev import sha256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _relative_video(workspace: Path, directory: Path, example_id: str) -> str:
    video = (directory / f"{example_id}.mp4").resolve()
    try:
        relative = video.relative_to(workspace.resolve())
    except ValueError as error:
        raise ValueError("training video is outside the workspace") from error
    if not video.is_file():
        raise ValueError(f"training video is missing: {relative}")
    return relative.as_posix()


def how2sign_examples(
    *,
    workspace: Path,
    source: str,
    index_paths: list[Path],
    selection_path: Path,
    video_dir: Path,
    allow_incomplete_selection: bool = False,
) -> list[dict]:
    indexed = [
        (row, index_path)
        for index_path in index_paths
        for row in read_jsonl(index_path)
    ]
    index = [row for row, _ in indexed]
    selection = json.loads(selection_path.read_text())
    if source == "how2sign_aligned":
        if selection.get("schema") != "umi-studio-how2sign-training-aligned/1":
            raise ValueError("unexpected aligned How2Sign selection")
        selected = selection["rows"]
    elif source == "how2sign_raw":
        if (
            selection.get("schema") != "umi-studio-how2sign-raw-training-expansion/2"
            or selection.get("training_only") is not True
        ):
            raise ValueError("unexpected raw How2Sign selection")
        selected = [
            {
                **clip,
                "signer": video["signer"],
                "video_id": video["video_id"],
                "video_name": video["video_name"],
            }
            for video in selection["videos"]
            for clip in video["clips"]
        ]
    else:
        raise ValueError("unknown How2Sign source")
    by_name = {row["sentence_name"]: row for row in selected}
    index_names = [row.get("sentence_name") for row in index]
    if (
        len(by_name) != len(selected)
        or len(index_names) != len(set(index_names))
        or (not allow_incomplete_selection and len(index) != len(selected))
        or len(index) > len(selected)
    ):
        raise ValueError("How2Sign selection has duplicates or an incomplete index")
    output = []
    for row, index_path in indexed:
        name = row.get("sentence_name")
        chosen = by_name.get(name)
        if (
            row.get("schema") != "umi-studio-how2sign-encoder-cache-example/1"
            or row.get("split") != "training"
            or row.get("clip_fps") != 25
            or chosen is None
            or row.get("reference") != chosen.get("reference")
            or row.get("signer") != chosen.get("signer")
            or not isinstance(row.get("resampled_clip_sha256"), str)
        ):
            raise ValueError(f"How2Sign index differs from its selection: {name}")
        output.append(
            {
                "schema": "umi-studio-fresh-lowres-training-example/1",
                "source": source,
                "example_id": name,
                "source_group": f"how2sign:{chosen['video_id']}",
                "signer": row["signer"],
                "reference": row["reference"],
                "video": _relative_video(workspace, video_dir, name),
                "video_sha256": row["resampled_clip_sha256"],
                "video_fps": 25,
                "source_index_sha256": sha256(index_path),
                "source_selection_sha256": sha256(selection_path),
            }
        )
    return output


def how2sign_precut_expansion_examples(
    *,
    workspace: Path,
    selection_path: Path,
    receipts_path: Path,
    video_dir: Path,
    allow_incomplete_selection: bool = False,
) -> list[dict]:
    selection = json.loads(selection_path.read_text())
    receipts = read_jsonl(receipts_path)
    records = selection.get("records", [])
    selected = {row.get("sentence_name"): row for row in records}
    received = {row.get("sentence_name"): row for row in receipts}
    selected_names = [row.get("sentence_name") for row in records]
    received_names = [row.get("sentence_name") for row in receipts]
    selection_hash = sha256(selection_path)
    if (
        selection.get("schema") != "umi-studio-how2sign-precut-expansion/1"
        or selection.get("training_only") is not True
        or selection.get("clips") != len(records)
        or len(selected) != len(records)
        or len(received) != len(receipts)
        or (
            allow_incomplete_selection
            and (
                not receipts
                or len(receipts) > len(records)
                or received_names != selected_names[: len(receipts)]
            )
        )
        or (not allow_incomplete_selection and set(received) != set(selected))
    ):
        raise ValueError("pre-cut How2Sign expansion or receipts are incomplete")
    output = []
    for chosen in records[: len(receipts)]:
        name = chosen["sentence_name"]
        receipt = received[name]
        rate = receipt.get("video_stream", {}).get("r_frame_rate")
        video = video_dir / f"{name}.mp4"
        try:
            video_fps = float(Fraction(rate))
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise ValueError(f"invalid pre-cut How2Sign video rate: {name}") from error
        if (
            receipt.get("schema")
            != "umi-studio-how2sign-precut-expansion-clip/1"
            or receipt.get("selection_sha256") != selection_hash
            or receipt.get("archive_member") != chosen.get("archive_member")
            or receipt.get("archive_crc32") != chosen.get("archive_crc32")
            or receipt.get("clip_size_bytes") != chosen.get("archive_file_size")
            or not isinstance(receipt.get("clip_sha256"), str)
            or len(receipt["clip_sha256"]) != 64
            or type(chosen.get("signer")) is not int
            or not math.isfinite(video_fps)
            or video_fps <= 0
            or not video.is_file()
            or video.stat().st_size != receipt["clip_size_bytes"]
            or sha256(video) != receipt["clip_sha256"]
        ):
            raise ValueError(f"pre-cut How2Sign receipt differs: {name}")
        output.append(
            {
                "schema": "umi-studio-fresh-lowres-training-example/1",
                "source": "how2sign_precut_expansion",
                "example_id": name,
                "source_group": f"how2sign:{chosen['video_id']}",
                "signer": chosen["signer"],
                "reference": chosen["reference"],
                "video": _relative_video(workspace, video_dir, name),
                "video_sha256": receipt["clip_sha256"],
                "video_fps": video_fps,
                "source_index_sha256": sha256(receipts_path),
                "source_selection_sha256": selection_hash,
            }
        )
    return output


def fleurs_examples(
    *,
    workspace: Path,
    index_path: Path,
    selection_path: Path,
    video_dir: Path,
) -> list[dict]:
    index = read_jsonl(index_path)
    selection = json.loads(selection_path.read_text())
    if selection.get("schema") != "umi-studio-fleurs-dev-split/1":
        raise ValueError("unexpected FLEURS selection")
    selected = {row["example_id"]: row for row in selection["training"]}
    if len(selected) != len(selection["training"]) or len(index) != len(selected):
        raise ValueError("FLEURS selection has duplicates or an incomplete index")
    output = []
    for row in index:
        example_id = row.get("example_id")
        chosen = selected.get(example_id)
        if (
            row.get("schema") != "umi-studio-fleurs-dev-encoder-cache-example/1"
            or row.get("split") != "training"
            or row.get("clip_fps") != 25
            or chosen is None
            or row.get("reference") != chosen.get("reference")
            or row.get("signer") != chosen.get("signer")
            or row.get("document_id") != chosen.get("document_id")
            or not isinstance(row.get("clip_sha256"), str)
        ):
            raise ValueError(f"FLEURS index differs from its selection: {example_id}")
        output.append(
            {
                "schema": "umi-studio-fresh-lowres-training-example/1",
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
            }
        )
    return output


def asl_stem_wiki_examples(
    *,
    workspace: Path,
    selection_path: Path,
    receipts_path: Path,
    video_dir: Path,
) -> list[dict]:
    selection = json.loads(selection_path.read_text())
    receipts = read_jsonl(receipts_path)
    records = selection.get("records", [])
    selected = {row.get("example_id"): row for row in records}
    received = {row.get("example_id"): row for row in receipts}
    selection_hash = sha256(selection_path)
    if (
        selection.get("schema")
        != "umi-studio-asl-stem-wiki-training-selection/1"
        or selection.get("training_only") is not True
        or selection.get("selected_examples") != len(records)
        or len(selected) != len(records)
        or len(received) != len(receipts)
        or set(received) != set(selected)
    ):
        raise ValueError("ASL STEM Wiki selection or receipts are incomplete")
    output = []
    for chosen in records:
        example_id = chosen["example_id"]
        receipt = received[example_id]
        participant = chosen.get("participant", "")
        rate = receipt.get("video_probe", {}).get("stream", {}).get("r_frame_rate")
        try:
            video_fps = float(Fraction(rate))
            signer = int(participant.removeprefix("P"))
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise ValueError(f"invalid ASL STEM Wiki video metadata: {example_id}") from error
        video = video_dir / f"{example_id}.mp4"
        if (
            chosen.get("schema")
            != "umi-studio-asl-stem-wiki-training-example/1"
            or chosen.get("split") != "training"
            or not participant.startswith("P")
            or signer < 0
            or not math.isfinite(video_fps)
            or video_fps <= 0
            or receipt.get("schema")
            != "umi-studio-asl-stem-wiki-training-video/1"
            or receipt.get("selection_sha256") != selection_hash
            or receipt.get("participant") != participant
            or receipt.get("article") != chosen.get("article")
            or receipt.get("archive_member") != chosen.get("archive_member")
            or receipt.get("video_size_bytes") != chosen.get("archive_file_size")
            or not isinstance(receipt.get("video_sha256"), str)
            or len(receipt["video_sha256"]) != 64
            or not video.is_file()
            or video.stat().st_size != receipt["video_size_bytes"]
            or sha256(video) != receipt["video_sha256"]
        ):
            raise ValueError(f"ASL STEM Wiki receipt differs: {example_id}")
        output.append(
            {
                "schema": "umi-studio-fresh-lowres-training-example/1",
                "source": "asl_stem_wiki",
                "example_id": example_id,
                "source_group": chosen["source_group"],
                "signer": signer,
                "reference": chosen["reference"],
                "video": _relative_video(workspace, video_dir, example_id),
                "video_sha256": receipt["video_sha256"],
                "video_fps": video_fps,
                "source_index_sha256": sha256(receipts_path),
                "source_selection_sha256": selection_hash,
            }
        )
    return output


def validate_inventory(rows: list[dict], workspace: Path) -> dict[str, int]:
    required = {
        "schema",
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
    }
    identities = [(row["source"], row["example_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("fresh training inventory contains duplicate identities")
    how2sign_ids = [
        row["example_id"] for row in rows if row["source"].startswith("how2sign_")
    ]
    if len(how2sign_ids) != len(set(how2sign_ids)):
        raise ValueError("fresh training inventory repeats a How2Sign clip")
    for row in rows:
        if (
            set(row) != required
            or row["schema"] != "umi-studio-fresh-lowres-training-example/1"
            or row["source"]
            not in {
                "how2sign_aligned",
                "how2sign_raw",
                "how2sign_precut_expansion",
                "fleurs",
                "asl_stem_wiki",
            }
            or type(row["signer"]) is not int
            or not row["reference"].strip()
            or type(row["video_fps"]) not in {int, float}
            or not math.isfinite(row["video_fps"])
            or row["video_fps"] <= 0
            or len(row["video_sha256"]) != 64
            or not (workspace / row["video"]).is_file()
        ):
            raise ValueError(f"invalid fresh training example: {row.get('example_id')}")
    return dict(sorted(Counter(row["source"] for row in rows).items()))


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError("existing fresh inventory differs from current inputs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.jsonl")
    if temporary.exists():
        raise ValueError("unfinished fresh inventory needs inspection")
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
    parser.add_argument(
        "--how2sign-aligned-index", type=Path, action="append", required=True
    )
    parser.add_argument("--allow-incomplete-aligned-selection", action="store_true")
    parser.add_argument("--how2sign-aligned-selection", type=Path, required=True)
    parser.add_argument("--how2sign-aligned-video-dir", type=Path, required=True)
    parser.add_argument("--how2sign-raw-index", type=Path, required=True)
    parser.add_argument("--how2sign-raw-selection", type=Path, required=True)
    parser.add_argument("--how2sign-raw-video-dir", type=Path, required=True)
    parser.add_argument("--how2sign-precut-expansion-selection", type=Path)
    parser.add_argument("--how2sign-precut-expansion-receipts", type=Path)
    parser.add_argument("--how2sign-precut-expansion-video-dir", type=Path)
    parser.add_argument(
        "--allow-incomplete-how2sign-precut-expansion", action="store_true"
    )
    parser.add_argument("--fleurs-index", type=Path, required=True)
    parser.add_argument("--fleurs-selection", type=Path, required=True)
    parser.add_argument("--fleurs-video-dir", type=Path, required=True)
    parser.add_argument("--asl-stem-wiki-selection", type=Path)
    parser.add_argument("--asl-stem-wiki-receipts", type=Path)
    parser.add_argument("--asl-stem-wiki-video-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asl_arguments = (
        args.asl_stem_wiki_selection,
        args.asl_stem_wiki_receipts,
        args.asl_stem_wiki_video_dir,
    )
    if any(asl_arguments) and not all(asl_arguments):
        parser.error("provide all ASL STEM Wiki inputs or none")
    precut_arguments = (
        args.how2sign_precut_expansion_selection,
        args.how2sign_precut_expansion_receipts,
        args.how2sign_precut_expansion_video_dir,
    )
    if any(precut_arguments) and not all(precut_arguments):
        parser.error("provide all pre-cut How2Sign expansion inputs or none")
    workspace = Path(__file__).resolve().parents[1]
    rows = [
        *how2sign_examples(
            workspace=workspace,
            source="how2sign_aligned",
            index_paths=args.how2sign_aligned_index,
            selection_path=args.how2sign_aligned_selection,
            video_dir=args.how2sign_aligned_video_dir,
            allow_incomplete_selection=args.allow_incomplete_aligned_selection,
        ),
        *how2sign_examples(
            workspace=workspace,
            source="how2sign_raw",
            index_paths=[args.how2sign_raw_index],
            selection_path=args.how2sign_raw_selection,
            video_dir=args.how2sign_raw_video_dir,
        ),
        *fleurs_examples(
            workspace=workspace,
            index_path=args.fleurs_index,
            selection_path=args.fleurs_selection,
            video_dir=args.fleurs_video_dir,
        ),
        *(
            how2sign_precut_expansion_examples(
                workspace=workspace,
                selection_path=args.how2sign_precut_expansion_selection,
                receipts_path=args.how2sign_precut_expansion_receipts,
                video_dir=args.how2sign_precut_expansion_video_dir,
                allow_incomplete_selection=(
                    args.allow_incomplete_how2sign_precut_expansion
                ),
            )
            if all(precut_arguments)
            else []
        ),
        *(
            asl_stem_wiki_examples(
                workspace=workspace,
                selection_path=args.asl_stem_wiki_selection,
                receipts_path=args.asl_stem_wiki_receipts,
                video_dir=args.asl_stem_wiki_video_dir,
            )
            if all(asl_arguments)
            else []
        ),
    ]
    counts = validate_inventory(rows, workspace)
    atomic_jsonl(args.output, rows)
    print(
        f"frozen {len(rows)} raw-frame examples sha256={sha256(args.output)} "
        f"counts={json.dumps(counts, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
