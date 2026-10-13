"""Run the frozen baseline on signer-disjoint FLEURS-ASL validation clips.

The JSONL result is a single-reference development diagnostic, not an official
UMI score. Run with the independent baseline environment and low CPU priority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

MAX_MINER_HEALTH_SECONDS = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def miner_revision(url: str, expected: str | None) -> str:
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError("miner health endpoint returned a non-200 status")
        health = json.load(response)
    if time.perf_counter() - started > MAX_MINER_HEALTH_SECONDS:
        raise RuntimeError("miner health endpoint is responding too slowly")
    revision = health.get("model_revision")
    if health.get("ok") is not True or not isinstance(revision, str):
        raise RuntimeError("miner health endpoint is not healthy")
    if expected is not None and revision != expected:
        raise RuntimeError("miner model revision changed during benchmark")
    return revision


def normalize(text: str) -> tuple[str, ...]:
    value = unicodedata.normalize("NFKC", text).lower()
    characters = []
    for index, character in enumerate(value):
        letter_or_number = unicodedata.category(character)[0] in {"L", "N"}
        internal_apostrophe = (
            character in {"'", "\u2019"}
            and 0 < index < len(value) - 1
            and unicodedata.category(value[index - 1])[0] in {"L", "N"}
            and unicodedata.category(value[index + 1])[0] in {"L", "N"}
        )
        characters.append(character if letter_or_number or internal_apostrophe else " ")
    return tuple("".join(characters).split())


def distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_word in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_word in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1]


def extract_clip(source: Path, clip: Path, start: float, end: float, fps: str) -> None:
    if clip.is_file():
        return
    temporary = clip.with_suffix(".partial.mp4")
    if temporary.exists():
        raise RuntimeError(f"unfinished clip extraction must be inspected: {temporary}")
    try:
        command = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-t",
            str(round(end - start, 3)),
            "-map",
            "0:v:0",
            "-an",
        ]
        if fps != "source":
            command.extend(("-vf", f"fps={fps}"))
        command.extend(
            (
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "2",
                str(temporary),
            )
        )
        subprocess.run(command, check=True)
        temporary.rename(clip)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--fps", choices=("source", "15", "25", "30"), default="25")
    parser.add_argument("--source-short-side", type=int)
    parser.add_argument("--visual-crop-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=5, help="maximum new validation examples")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    from resolution_policy import validate_source_short_side, validate_visual_crop_size

    try:
        validate_source_short_side(args.source_short_side)
        validate_visual_crop_size(args.visual_crop_size)
    except ValueError as error:
        parser.error(str(error))
    if not args.source_dir.is_dir() or not args.baseline.is_dir():
        parser.error("source directory and independent baseline must exist")
    selection = json.loads(args.selection.read_bytes())
    receipt = json.loads(args.source_receipt.read_bytes())
    selection_hash = sha256(args.selection)
    if (
        selection.get("schema") != "umi-studio-fleurs-dev-split/1"
        or receipt.get("schema") != "umi-studio-fleurs-dev-source-files/1"
        or receipt.get("split") != "validation"
        or receipt.get("selection_sha256") != selection_hash
    ):
        raise ValueError("selection and source receipt are incompatible")
    source_records = {row["path"]: row for row in receipt["files"]}
    if len(source_records) != len(receipt["files"]):
        raise ValueError("source receipt contains duplicate files")
    completed = {}
    if args.results.exists():
        for line in args.results.read_text().splitlines():
            record = json.loads(line)
            if record["selection_sha256"] != selection_hash:
                raise ValueError("existing benchmark results use another split")
            if record.get("clip_fps", "source") != args.fps:
                raise ValueError("existing benchmark results use another frame-rate transform")
            if (
                record.get("source_short_side") != args.source_short_side
                or record.get("visual_crop_size", 224) != args.visual_crop_size
            ):
                raise ValueError("existing benchmark results use another resolution policy")
            if record["example_id"] in completed:
                raise ValueError("existing benchmark results contain a duplicate example")
            completed[record["example_id"]] = record

    baseline = args.baseline.resolve()
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(baseline / "run/hf"),
        "TORCH_HOME": str(baseline / "run/torch"),
        "MPLCONFIGDIR": str(baseline / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(baseline / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(baseline))
    from runtime import SHuBERTInferenceRuntime

    revision = miner_revision(args.miner_health_url, None)
    model = SHuBERTInferenceRuntime(baseline, device="cpu")
    from resolution_policy import make_short_side_reader, set_visual_crop_size

    if args.source_short_side is not None:
        model._video_reader = make_short_side_reader(
            model._video_reader, args.source_short_side
        )
    visual_policy = set_visual_crop_size(model, args.visual_crop_size)
    revision = miner_revision(args.miner_health_url, revision)
    baseline_identity_hash = sha256(baseline / "MODEL_IDENTITY.json")
    baseline_runtime_hash = sha256(baseline / "runtime.py")
    args.clip_dir.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    verified_sources = set()
    new_count = 0
    for row in selection["validation"]:
        if row["example_id"] in completed:
            continue
        if new_count >= args.limit:
            break
        revision = miner_revision(args.miner_health_url, revision)
        source_name = row["source_video"]
        source = args.source_dir / Path(source_name).name
        source_record = source_records[source_name]
        if source_name not in verified_sources:
            if (
                source.stat().st_size != source_record["size_bytes"]
                or sha256(source) != source_record["sha256"]
            ):
                raise RuntimeError(f"source video no longer matches receipt: {source_name}")
            verified_sources.add(source_name)
        clip = args.clip_dir / f"{row['example_id']}.mp4"
        extract_clip(source, clip, row["start_time_sec"], row["end_time_sec"], args.fps)
        clip_hash = sha256(clip)
        hypothesis = model.translate_path(clip)
        revision = miner_revision(args.miner_health_url, revision)
        reference_words = normalize(row["reference"])
        hypothesis_words = normalize(hypothesis)
        edits = distance(reference_words, hypothesis_words)
        wer = edits / len(reference_words)
        record = {
            "schema": "umi-studio-fleurs-dev-baseline-example/1",
            "example_id": row["example_id"],
            "selection_sha256": selection_hash,
            "source_receipt_sha256": sha256(args.source_receipt),
            "baseline_identity_sha256": baseline_identity_hash,
            "baseline_runtime_sha256": baseline_runtime_hash,
            "source_video": source_name,
            "source_video_sha256": source_record["sha256"],
            "clip_sha256": clip_hash,
            "clip_transform": "ffmpeg input seek + h264 ultrafast crf18",
            "clip_fps": args.fps,
            "source_short_side": args.source_short_side,
            "source_resize_policy": (
                "aspect_preserving_downscale_only"
                if args.source_short_side is not None
                else "native"
            ),
            **visual_policy,
            "clip_start_seconds": row["start_time_sec"],
            "clip_end_seconds": row["end_time_sec"],
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "single_reference_wer": wer,
            "single_reference_word_edits": edits,
            "reference_word_count": len(reference_words),
            "miner_model_revision": revision,
            "diagnostics": model.last_diagnostics,
        }
        with args.results.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        new_count += 1
        print(f"{len(completed) + new_count}: {row['example_id']} WER={wer:.3f}", flush=True)


if __name__ == "__main__":
    main()
