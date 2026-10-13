"""Compare the frozen SHuBERT baseline on the same FSboard validation clips.

Only clips actually scored by the offline Core ML right-hand diagnostic are
eligible. This remains a single-reference development CER, not a UMI score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import distance, miner_revision, sha256
from evaluate_fsboard_coreml import comparable_units
from fetch_fsboard_validation import validate_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clip-receipts", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--sequential-decode", action="store_true")
    parser.add_argument("--source-short-side", type=int)
    parser.add_argument("--visual-crop-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.limit < 1 or not args.baseline.is_dir():
        parser.error("limit must be positive and baseline must exist")
    from resolution_policy import validate_source_short_side, validate_visual_crop_size

    try:
        validate_source_short_side(args.source_short_side)
        validate_visual_crop_size(args.visual_crop_size)
    except ValueError as error:
        parser.error(str(error))
    rows = validate_selection(json.loads(args.selection.read_text()))
    selection_hash = sha256(args.selection)
    candidate_hash = sha256(args.candidate_results)
    eligible = {
        result["clip_filename"]
        for result in map(json.loads, args.candidate_results.read_text().splitlines())
        if result["schema"] == "umi-studio-fsboard-coreml-diagnostic/2"
        and result["status"] == "right_hand_evaluated"
        and result["selection_sha256"] == selection_hash
    }
    receipts = {
        row["clip_filename"]: row
        for row in map(json.loads, args.clip_receipts.read_text().splitlines())
    }
    previous = set()
    if args.results.exists():
        for line in args.results.read_text().splitlines():
            record = json.loads(line)
            name = record["clip_filename"]
            if (
                name in previous
                or name not in eligible
                or record["selection_sha256"] != selection_hash
                or record["candidate_results_sha256"] != candidate_hash
                or record.get("source_short_side") != args.source_short_side
                or record.get("visual_crop_size", 224) != args.visual_crop_size
            ):
                raise ValueError("existing baseline FSboard results differ from inputs")
            previous.add(name)
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
    if args.sequential_decode:
        from sequential_video_reader import make_sequential_reader

        model._video_reader = make_sequential_reader(model._video_reader)
    from resolution_policy import make_short_side_reader, set_visual_crop_size

    if args.source_short_side is not None:
        model._video_reader = make_short_side_reader(
            model._video_reader, args.source_short_side
        )
    visual_policy = set_visual_crop_size(model, args.visual_crop_size)
    revision = miner_revision(args.miner_health_url, revision)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        name = row["clip_filename"]
        if name not in eligible or name in previous or count >= args.limit:
            continue
        clip = args.clips_dir / name
        receipt = receipts.get(name)
        if (
            receipt is None
            or receipt["selection_sha256"] != selection_hash
            or not clip.is_file()
            or sha256(clip) != receipt["sha256"]
        ):
            raise ValueError(f"FSboard source clip differs from receipt: {name}")
        revision = miner_revision(args.miner_health_url, revision)
        hypothesis = model.translate_path(clip)
        revision = miner_revision(args.miner_health_url, revision)
        reference_units = comparable_units(row["reference"])
        hypothesis_units = comparable_units(hypothesis)
        record = {
            "schema": "umi-studio-fsboard-baseline-diagnostic/1",
            "clip_filename": name,
            "selection_sha256": selection_hash,
            "candidate_results_sha256": candidate_hash,
            "clip_sha256": receipt["sha256"],
            "video_decode_mode": "sequential" if args.sequential_decode else "random_access",
            "source_short_side": args.source_short_side,
            "source_resize_policy": (
                "aspect_preserving_downscale_only"
                if args.source_short_side is not None
                else "native"
            ),
            **visual_policy,
            "baseline_identity_sha256": sha256(baseline / "MODEL_IDENTITY.json"),
            "baseline_runtime_sha256": sha256(baseline / "runtime.py"),
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "single_reference_cer": distance(reference_units, hypothesis_units)
            / max(1, len(reference_units)),
            "miner_model_revision": revision,
            "diagnostics": model.last_diagnostics,
        }
        with args.results.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        count += 1
        print(f"{len(previous) + count}: baseline evaluated {name}", flush=True)


if __name__ == "__main__":
    main()
