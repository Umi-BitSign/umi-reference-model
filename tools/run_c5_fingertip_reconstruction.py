# ruff: noqa: UP017
"""Freeze and run one small CPU-only auxiliary geometry experiment.

Uses the pre-existing fit/validation split; confirmation arrays stay unopened.
This entry point never loads a translation model or imports a GPU framework.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from c5_fingertip_probe_data import Packet, require
from c5_fingertip_probe_metrics import clip_xy, paired_group_comparison
from c5_fingertip_reconstruction import METHODS, OFFSETS, RIDGE, SCALE_FLOOR, features, fit, predict

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "runs/c5-fingertip-probe-split-v1/plan.json"
SPLIT_SHA = "14f5f08067fdd8ea7cdc498777091070b1cbcfc000be4b2ad937cc9e61216a95"
BUDGET = 16 * 1024**2
PROTECTED = (
    "c5-word-semantic-training-v1",
    "c5-mixed-native112-data-v1",
    "c5-word-semantic-training-v2-recovery",
    "terra-negation-rubric-study-v1",
    "terra-negation-failed-context-v1",
    "c5-polarity-semantic-gradient-v1",
    "c5-polarity-word-training-v1",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_sources():
    result = {}
    for name in PROTECTED:
        plan = json.loads((ROOT / "runs" / name / "plan.json").read_text())
        for path, digest in plan["source_sha256"].items():
            require(sha(ROOT / path) == digest, f"protected source changed: {path}")
        result[name] = len(plan["source_sha256"])
    return result


def summarize(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["score"] is not None:
            grouped[row["family"]][row["group"]].append(row["score"])
    families = {
        family: float(np.mean([np.mean(values) for values in groups.values()]))
        for family, groups in grouped.items()
    }
    require(len(families) == 3, "missing scored source family")
    joint_means = {}
    for joint in rows[0]["per_joint"]:
        joint_rows = [{**row, "score": row["per_joint"][joint]["mae"]} for row in rows]
        values = defaultdict(lambda: defaultdict(list))
        for row in joint_rows:
            if row["score"] is not None:
                values[row["family"]][row["group"]].append(row["score"])
        require(len(values) == 3, "missing joint source family")
        joint_means[joint] = float(
            np.mean(
                [
                    np.mean([np.mean(clips) for clips in groups.values()])
                    for groups in values.values()
                ]
            )
        )
    return {
        "equal_family_group_clip_mae": float(np.mean(list(families.values()))),
        "family_mae": families,
        "per_joint_mae": joint_means,
        "family_scored_groups": {family: len(groups) for family, groups in grouped.items()},
        "clips": len(rows),
        "unscored_clips": [row["clip_id"] for row in rows if row["score"] is None],
        "input_complete_hands": sum(row["input_complete_hands"] for row in rows),
        "known_target_hands": sum(row["known_target_hands"] for row in rows),
        "scored_target_points": sum(row["scored_target_points"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        require(os.environ.get(key) == "2", f"CPU2 environment required: {key}")
    out = args.output.resolve()
    require(out.parent == ROOT / "runs" and not out.exists(), "fresh runs directory required")
    require(sha(SPLIT) == SPLIT_SHA, "split changed")
    protected_before = protected_sources()
    assignments = json.loads(SPLIT.read_text())["assignments"]
    roles = {
        role: [row for row in assignments if row["role"] == role] for role in ("fit", "validation")
    }
    require([len(roles[role]) for role in roles] == [328, 104], "unexpected split roles")

    def guard(extra):
        used = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        require(used + extra <= BUDGET, "auxiliary output cap")
        # The parent trainer is terminal. This small independent probe reserves
        # its entire prospective output, the physical floor and safety margin.
        require(
            shutil.disk_usage(ROOT).free >= 80 * 1024**3 + 128 * 1024**2 + BUDGET,
            "auxiliary physical storage guard",
        )

    def write_bytes(name, data):
        guard(len(data))
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}

    def write(name, value):
        return write_bytes(
            name, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        )

    guard(BUDGET)
    out.mkdir()
    source_paths = [
        Path(__file__),
        ROOT / "tools/c5_fingertip_reconstruction.py",
        ROOT / "tools/c5_fingertip_probe_data.py",
        ROOT / "tools/c5_fingertip_probe_metrics.py",
        ROOT / "tests/test_c5_fingertip_reconstruction.py",
    ]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    for path in source_paths:
        write_bytes(str(Path("sources") / path.relative_to(ROOT)), path.read_bytes())
    plan = write(
        "plan.json",
        {
            "schema": "umi-studio-tip-geometry-probe/1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": {"path": str(SPLIT.relative_to(ROOT)), "sha256": SPLIT_SHA},
            "methods": list(METHODS),
            "ridge_on_weighted_standardized_features": RIDGE,
            "tip_scale_floor": SCALE_FLOOR,
            "temporal_offsets_original_25fps_frames": list(OFFSETS),
            "fit": roles["fit"],
            "validation": roles["validation"],
            "confirmation_feature_arrays_opened": False,
            "feature_inputs": "XY tips, input finite masks, tip-derived continuity only",
            "targets": "withheld detector wrist and index/middle/ring/little MCP XY",
            "fit_weighting": "equal families/groups/clips/eligible hand frames for each joint",
            "geometry_score": "equal families/groups/clips; common support across all arms",
            "hyperparameter_search_or_validation_refit": False,
            "bootstrap": "4000 paired whole-group samples within source family; exploratory",
            "primary_comparisons": ["all arms vs centroid", "all arms vs normalized_linear"],
            "limitations": [
                "detector targets are not anatomical ground truth",
                "detector errors can correlate between tips and withheld joints",
                "temporal arm uses future frames up to 240ms",
                "two unnamed tracks do not verify speaker ownership or physical identity",
                "source-group disjointness for OpenASL does not prove signer disjointness",
                "no sign boundary, recognition, semantic translation or serving evidence",
                "exploratory intervals have no multiple-comparison correction",
            ],
            "source_sha256": sources,
            "cpu_threads": 2,
            "gpu_api_or_translation_optimizer_calls": 0,
            "output_byte_budget": BUDGET,
            "parent_training_restarted": False,
        },
    )
    started = time.monotonic()
    packet = Packet(ROOT)
    fit_sequences = []
    for assignment in roles["fit"]:
        sequence = packet.load(assignment["record_index"])
        require(
            (sequence.metadata["source"], sequence.metadata["example_id"])
            == (assignment["source"], assignment["example_id"]),
            "fit identity mismatch",
        )
        fit_sequences.append((assignment, sequence))
    models, model_descriptors = {}, {}
    for method in METHODS:
        if method == "centroid":
            models[method] = None
            continue
        rows = []
        for assignment, sequence in fit_sequences:
            rows.append(
                {
                    "features": features(sequence.tips, sequence.continuity, method),
                    "targets": sequence.targets.reshape(-1, 5, 2),
                    "target_valid": sequence.target_valid.reshape(-1, 5),
                    "family": assignment["family"],
                    "group": assignment["component_id"],
                    "role": assignment["role"],
                }
            )
        models[method] = fit(rows)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **models[method])
        model_descriptors[method] = write_bytes(f"models/{method}.npz", buffer.getvalue())
        print(json.dumps({"fitted": method, "fit_clips": len(rows)}), flush=True)
    scores, summaries, comparisons, frames = {}, {}, {}, {}
    for role in roles:
        scores[role] = {method: [] for method in METHODS}
        frames[role] = 0
        iterator = (
            fit_sequences
            if role == "fit"
            else (
                (assignment, packet.load(assignment["record_index"])) for assignment in roles[role]
            )
        )
        for assignment, sequence in iterator:
            require(
                (sequence.metadata["source"], sequence.metadata["example_id"])
                == (assignment["source"], assignment["example_id"]),
                "score identity mismatch",
            )
            frames[role] += len(sequence.seconds)
            for method in METHODS:
                view = features(sequence.tips, sequence.continuity, method)
                score = clip_xy(
                    predict(view, models[method]),
                    sequence.targets,
                    sequence.target_valid,
                    sequence.tip_valid,
                )
                scores[role][method].append(
                    {
                        **score,
                        "clip_id": str(assignment["record_index"]),
                        "family": assignment["family"],
                        "group": assignment["component_id"],
                    }
                )
        summaries[role] = {method: summarize(rows) for method, rows in scores[role].items()}
        print(json.dumps({"scored": role, "clips": len(roles[role])}), flush=True)
    for baseline in ("centroid", "normalized_linear"):
        comparisons[baseline] = {
            method: paired_group_comparison(scores["validation"][baseline], rows)
            for method, rows in scores["validation"].items()
            if method != baseline
        }
    require(frames == {"fit": 49815, "validation": 13447}, "timeline count differs")
    rows_descriptor = write("rows.json", scores)
    require(protected_sources() == protected_before, "protected inventory changed")
    require(all(sha(ROOT / path) == digest for path, digest in sources.items()), "sources changed")
    result = write(
        "result.json",
        {
            "schema": "umi-studio-tip-geometry-probe-result/1",
            "complete": True,
            "plan": plan,
            "models": model_descriptors,
            "rows": rows_descriptor,
            "summary": summaries,
            "validation_comparisons": comparisons,
            "original_frames": frames,
            "cpu_wall_seconds": time.monotonic() - started,
            "frozen_current_source_maps_unchanged": protected_before,
            "confirmation_feature_arrays_opened": False,
            "learned_auxiliary_ridge_models": len(model_descriptors),
            "gpu_api_or_translation_optimizer_calls": 0,
            "semantic_quality_or_linguistic_spans_measured": False,
            "parent_training_restarted": False,
        },
    )
    print(json.dumps({"complete": True, "result": result}), flush=True)


if __name__ == "__main__":
    main()
