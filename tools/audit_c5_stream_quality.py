"""Measure blank/imputed C5 inputs on deterministic training and open-tuning clips."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

from benchmark_fleurs_dev import sha256
from c5_all_source_training import load_corpus
from freeze_c5_stem_continuation_plan import BASELINE, ROOT
from train_c5_stem_continuation import STREAM_NAMES, stride2_streams


def sample_training(rows: list[dict], per_group: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = ("how2sign", str(row["signer"])) if row["source"].startswith("how2sign") else (row["source"], "all")
        groups[key].append(row)
    selected = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: hashlib.sha256(
            f"c5-stream-quality-v1\0{row['source']}\0{row['id']}".encode()
        ).hexdigest())
        if len(ordered) < per_group:
            raise ValueError("stream audit lacks the required training group")
        selected.extend(ordered[:per_group])
    return selected


def stream_statistics(streams: dict, blanks: dict, rms_threshold: float = 1e-3) -> dict:
    import numpy as np

    result = {}
    for name in STREAM_NAMES:
        values = streams[name]
        if not np.isfinite(values).all() or values.ndim != 2 or len(values) < 1:
            raise ValueError("stream audit has invalid cached features")
        repeated = np.all(values[1:] == values[:-1], axis=1)
        summary = {
            "frames": len(values),
            "exact_adjacent_repeat_fraction": float(repeated.mean()) if len(repeated) else 0.,
            "mean_vector_norm": float(np.linalg.norm(values, axis=1).mean()),
            "mean_adjacent_rms_change": float(np.sqrt(np.square(np.diff(values, axis=0)).mean(axis=1)).mean()) if len(values) > 1 else 0.,
        }
        if name in blanks:
            delta = np.sqrt(np.square(values - blanks[name]).mean(axis=1))
            summary["blank_template_match_fraction"] = float(np.mean(delta <= rms_threshold))
        else:
            summary["all_missing_pose_fraction"] = float(np.mean(np.all(values == -9999., axis=1)))
        result[name] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-training-group", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs") or not 1 <= args.per_training_group <= 16:
        parser.error("stream audit requires a new runs/ output and bounded sample")
    plan = json.loads(args.plan.read_text())
    corpus, _, _ = load_corpus(plan)
    training = sample_training(corpus, args.per_training_group)
    for key, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TORCH_HOME": str(BASELINE / "run/torch"),
        "SHUBERT_DINOV2_SOURCE": str(BASELINE / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(BASELINE / "vendor/demo"))
    import numpy as np
    import torch
    from dinov2_features import DINOEmbedder

    torch.set_num_threads(2)
    blanks = {}
    asset_hashes = {}
    expected = {
        "hand": "630a11c0aef6bca4f27bcb2f68dbf35ad33d365af877c1fca7c42e1e748064ac",
        "face": "61e96b40c9b3b841520fb2fcf89b2a2fdda706d13723669d11304fc2064d295e",
    }
    for kind in ("hand", "face"):
        path = BASELINE / f"models/dinov2{kind}.safetensors"
        asset_hashes[kind] = sha256(path)
        if asset_hashes[kind] != expected[kind]:
            raise ValueError("stream audit DINO asset changed")
        embedder = DINOEmbedder(str(path), batch_size=1, device="cpu")
        vector = embedder.extract_embedding_from_single_image(np.zeros((224, 224, 3), dtype=np.uint8))[0]
        for name in (("left_hand_features", "right_hand_features") if kind == "hand" else ("face_features",)):
            blanks[name] = vector
        del embedder
    rows = [{**row, "audit_split": "training"} for row in training]
    from run_c5_stem_tuning import LAYOUT
    for source, layout in LAYOUT.items():
        index_path = layout["index"]
        for row in [json.loads(line) for line in index_path.read_text().splitlines()]:
            if source == "fleurs" and row["split"] != "tuning":
                continue
            key = row.get("sentence_name", row.get("example_id"))
            rows.append({
                "source": source, "id": key, "signer": row["signer"], "reference": row["reference"],
                "cache": str(layout["cache"] / f"{key}.npz"), "cache_sha256": row["encoder_cache_sha256"],
                "sidecar": None, "frames": row["encoder_shape"][0], "audit_split": "open_tuning",
            })
    report_rows = []
    grouped = defaultdict(list)
    for row in rows:
        streams = stride2_streams(row)
        metrics = stream_statistics(streams, blanks)
        report_rows.append({
            "source": row["source"], "example_id": row["id"], "signer": row["signer"],
            "reference": row["reference"], "split": row["audit_split"], "cache_sha256": row["cache_sha256"],
            "streams": metrics,
        })
        key = f"{row['audit_split']}:{row['source']}:{row['signer']}"
        grouped[key].append(metrics)
    summary = {}
    for key, metrics in sorted(grouped.items()):
        summary[key] = {"clips": len(metrics), "streams": {
            name: {metric: float(np.mean([row[name][metric] for row in metrics])) for metric in metrics[0][name]}
            for name in STREAM_NAMES
        }}
    report = {
        "schema": "umi-studio-c5-stream-quality-audit/1", "plan_sha256": sha256(args.plan),
        "audit_source_sha256": sha256(Path(__file__)), "dino_asset_sha256": asset_hashes,
        "training_clips": len(training), "open_tuning_clips": len(rows) - len(training),
        "selection": "SHA-256 order, eight clips per How2Sign signer and per other prepared source",
        "blank_match_rms_threshold": 1e-3, "summary": summary, "records": report_rows,
        "device": "cpu", "cpu_threads": 2, "miner_changed": False, "confirmation_decoded": False,
        "qualification": "Blank-template matches measure blank crop embeddings. Exact repeated features can reflect imputation or genuinely static crops. These metrics do not establish semantic video/transcript alignment or ASL translation quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"training_clips": len(training), "open_tuning_clips": len(rows)-len(training), "output": str(args.output)}))


if __name__ == "__main__":
    main()
