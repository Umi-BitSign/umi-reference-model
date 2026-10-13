"""Prepare matched, aspect-preserving raw-pixel fit cases from audited training clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from c5_experiment_sources import archive_sources
from c5_full_coverage_mps import load_training
from tiny_rgb_data import decode_rgb_video
from tiny_rgb_fit import RECIPE, ROOT, SEED, SIDES, private_path, select_fit_cases, sources, storage
from tiny_rgb_model import TinyRgbConfig
from train_fresh_lowres import language_artifact_receipt


def preview(bindings: list[dict], output: Path) -> None:
    """Exact resized pixels at native display size, with no learned detections or crops."""
    import cv2

    chosen = []
    seen = set()
    for binding in bindings:
        domain = binding["source"].split("_")[0]
        if domain not in seen:
            chosen.append(binding)
            seen.add(domain)
    row_height = max(v["shape"][1] for b in chosen for v in b["views"].values()) + 43
    column_width = max(400, max(v["shape"][2] for b in chosen for v in b["views"].values()) + 12)
    sheet = np.full((len(chosen) * row_height, column_width * 2, 3), 245, dtype=np.uint8)
    for index, binding in enumerate(chosen):
        for column, side in enumerate(SIDES):
            with np.load(ROOT / binding["views"][str(side)]["path"], allow_pickle=False) as data:
                frames = data["pixels"]
                frame = frames[len(frames) // 2]
            height, width = frame.shape[:2]
            # The panel does not resize its already-downscaled pixels.
            y, x = index * row_height + 37, column * column_width + (column_width - width) // 2
            sheet[y : y + height, x : x + width] = frame
            text = f"{binding['source']} cap{side}: {width}x{height}"
            cv2.putText(
                sheet,
                text,
                (column * column_width + 4, index * row_height + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
            )
    if not cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("could not persist RGB pixel preview")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gpu-budget-policy", type=Path, required=True)
    args = parser.parse_args()
    output, data_dir = args.output_dir.resolve(), args.data_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("RGB preparation requires a new runs/ directory")
    if data_dir.exists() or not data_dir.is_relative_to(ROOT / "data"):
        raise ValueError("RGB preparation requires a new private data/ directory")
    raw_dir = ROOT / "runs/tiny-rgb-corpus-v1"
    audit_path, inventory = raw_dir / "audit.json", raw_dir / "training-inventory.jsonl"
    audit = json.loads(audit_path.read_text())
    if sha256(inventory) != audit["training_inventory_sha256"]:
        raise ValueError("audited raw training inventory changed")
    approved = load_training(
        ROOT / "runs/c5-all-source-corpus-v2/audit.json",
        ROOT / "runs/c5-all-source-corpus-v2/training-streams.jsonl",
    )
    rows = [json.loads(s) for s in inventory.read_text().splitlines()]
    if {(r["source"], r["example_id"]) for r in rows} != {(r["source"], r["id"]) for r in approved}:
        raise ValueError("raw RGB audit no longer matches protected approved training identities")
    dependency_hashes = {
        str(audit_path.relative_to(ROOT)): sha256(audit_path),
        str(inventory.relative_to(ROOT)): sha256(inventory),
        str(args.gpu_budget_policy.resolve().relative_to(ROOT)): sha256(args.gpu_budget_policy),
        **audit["source_inventory_sha256"],
    }
    if any(sha256(ROOT / name) != digest for name, digest in dependency_hashes.items()):
        raise ValueError("raw RGB audit dependencies changed")
    language_path = ROOT / "artifacts/t5-small-df1b051c"
    language = language_artifact_receipt(ROOT, language_path)
    language_assets = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in language_path.iterdir()
        if p.is_file() and p.suffix in (".json", ".model", ".safetensors")
    }
    examples = select_fit_cases(rows)
    source_hashes = sources()
    output.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    bindings = []
    with output.joinpath("pixels.jsonl").open("x") as ledger:
        for index, row in enumerate(examples):
            if sha256(private_path(row["video"], "data")) != row["video_sha256"]:
                raise ValueError("selected RGB source changed")
            binding = {k: row[k] for k in ("source", "example_id", "video_sha256")}
            binding["views"] = {}
            common_indices = None
            for side in SIDES:
                frames, indices, metadata = decode_rgb_video(
                    row["video"], side, RECIPE["target_fps"], RECIPE["max_frames"]
                )
                if common_indices is not None and not np.array_equal(common_indices, indices):
                    raise ValueError("RGB resolution arms differ in temporal sampling")
                if abs(metadata["source_fps"] - row["video_fps"]) > 0.02:
                    raise ValueError("RGB frame rate differs from source binding")
                common_indices = indices
                storage(data_dir, frames.nbytes + indices.nbytes + 4096)
                target = data_dir / f"case-{index:02d}-cap{side}.npz"
                with target.open("xb") as stream:
                    np.savez_compressed(stream, pixels=frames, frame_indices=indices)
                binding["views"][str(side)] = {
                    "path": str(target.relative_to(ROOT)),
                    "sha256": sha256(target),
                    "shape": list(frames.shape),
                    "metadata": metadata,
                    "frame_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                }
            bindings.append(binding)
            ledger.write(json.dumps(binding, sort_keys=True) + "\n")
            ledger.flush()
            print(
                json.dumps({"prepared_cases": index + 1, "total_cases": len(examples)}), flush=True
            )
    if sources() != source_hashes:
        raise ValueError("RGB preparation implementation changed while running")
    plan = {
        "schema": "umi-studio-tiny-rgb-fit-plan/1",
        "seed": SEED,
        "recipe": RECIPE,
        "short_sides": list(SIDES),
        "visual_config": asdict(TinyRgbConfig()),
        "examples": examples,
        "raw_inventory": str(inventory.relative_to(ROOT)),
        "data_dir": str(data_dir.relative_to(ROOT)),
        "source_sha256": source_hashes,
        "dependency_sha256": dependency_hashes,
        "language_asset_sha256": language_assets,
        "language": language,
        "language_path": str(language_path.relative_to(ROOT)),
        "pixel_ledger_sha256": sha256(output / "pixels.jsonl"),
        "gpu_budget_policy": str(args.gpu_budget_policy.resolve().relative_to(ROOT)),
        "miner_mode": "user_authorized_maintenance",
        "miner_health_claim": False,
        "pretrained_visual_weights": False,
        "learned_feature_inputs": False,
        "c5_teacher_targets": False,
        "language_initialization": "pinned_generic_t5_small",
        "qualification": (
            "Both resolution arms start from identical fresh visual initialization and identical "
            "pretrained language bytes. Complete matched 12.5fps clips, no augmentation. All24 "
            "cases are training only and may be reused for fitting diagnostics. Success is a "
            "learnability prerequisite, not held-out generalization, official score or promotion. "
            "No frozen original/current baseline identity changes. Subsequent full-corpus "
            "training and protected source-held-out quality checks remain required."
        ),
    }
    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(plan_path, output / "source-snapshot")
    preview(bindings, output / "resolution-preview.png")
    print(
        json.dumps(
            {
                "plan_sha256": sha256(plan_path),
                "prepared_cases": len(bindings),
                "new_pixel_cache_bytes": sum(p.stat().st_size for p in data_dir.iterdir()),
            }
        )
    )


if __name__ == "__main__":
    main()
