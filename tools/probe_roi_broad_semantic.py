"""Qualify unchanged Terra-only ROI learning on the prepared cross-source stage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import FunctionType

import probe_roi_semantic_learning as shared
import torch
from benchmark_fleurs_dev import sha256
from evaluate_roi_short_translation import terminal
from launch_terra_visual_job import identity, now, write
from prepare_roi_appearance import ROOT, preview
from prepare_roi_broad_corpus import LAYOUT
from prepare_roi_broad_corpus import checked as checked_data_plan
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage
from train_roi_semantic import qualified

DATA_PLAN = ROOT / "runs/roi-broad-short-data-v1/plan.json"
DATA = ROOT / "checkpoints/roi-broad-short-data-v1/data.json"


def training_rows(plan, data):
    if (
        data.get("complete") is not True
        or data.get("all_streamed_pixels_equal_native_reference") is not True
        or len(data["records"]) != len(plan["records"])
    ):
        raise ValueError("broad ROI proof needs complete, native-parity data")
    reserved = set(plan["reserved_STEM_participants"] + plan["ROI_diagnostic_participants"])
    result, known = [], set()
    for blueprint, row in zip(plan["records"], data["records"], strict=True):
        if any(row.get(key) != value for key, value in blueprint.items()):
            raise ValueError("broad ROI cached roles/identities changed after preparation")
        if row["roi_training_allowed"] is False:
            if row["roi_role"] != "feature_diagnostic":
                raise ValueError("broad ROI nontraining role changed")
            continue
        source = row["original_row"]
        key = (source["source"], source["example_id"])
        if (
            row["roi_training_allowed"] is not True
            or row["roi_role"] != "training"
            or source["training_allowed"] is not True
            or source["role"] != "training"
            or source["source"] not in LAYOUT
            or (source["source"].startswith("asl_stem") and source["signer"] in reserved)
            or row["frames"] != source["original_pair"]["header"]["sampled_frames"]
            or row["streamed_pixels_equal_native_reference"] is not True
            or key in known
        ):
            raise ValueError("broad ROI proof cannot use reserved, incomplete or duplicate media")
        known.add(key)
        result.append(row)
    return result


def checked_data():
    plan = checked_data_plan(DATA_PLAN)
    terminal(DATA)
    data = json.loads(DATA.read_text())
    if (
        data.get("plan_sha256") != sha256(DATA_PLAN)
        or data.get("counts") != {"training": 318, "feature_diagnostic": 2}
        or data.get("coverage_counts") != plan["coverage_counts"]
    ):
        raise ValueError("broad ROI proof requires the complete318/2 preparation receipt")
    rows = training_rows(plan, data)
    if len(rows) != 318 or sum(r["frames"] for r in rows) != 26729:
        raise ValueError("broad ROI training coverage changed")
    for record in data["records"]:
        if artifact(ROOT / record["cache"]["path"]) != record["cache"]:
            raise ValueError("broad ROI complete cache bytes changed")
    return plan, data, rows


def proof_indices(rows):
    """Longest complete clip per family, plus longest byte caption if distinct."""
    if {r["original_row"]["source"] for r in rows} != set(LAYOUT):
        raise ValueError("broad ROI proof must cover every prepared source family")
    selected = []
    for source in LAYOUT:
        selected.append(
            min(
                (i for i, row in enumerate(rows) if row["original_row"]["source"] == source),
                key=lambda i: (-rows[i]["frames"], rows[i]["original_row"]["example_id"]),
            )
        )
    longest_caption = min(
        range(len(rows)),
        key=lambda i: (
            -len(rows[i]["original_row"]["reference"].encode()),
            rows[i]["original_row"]["source"],
            rows[i]["original_row"]["example_id"],
        ),
    )
    if longest_caption not in selected:
        selected.append(longest_caption)
    references = [rows[i]["original_row"]["reference"] for i in selected]
    if any(references[i] == references[(i + 1) % len(references)] for i in range(len(references))):
        raise ValueError("predeclared broad proof donors need distinct adjacent training captions")
    return selected


def specification():
    base, proof = qualified()
    data_plan, _, rows = checked_data()
    indices = proof_indices(rows)
    source_names = (
        set(base["source_sha256"])
        | set(data_plan["source_sha256"])
        | {
            "tools/probe_roi_broad_semantic.py",
            "tools/train_roi_semantic.py",
            "tools/roi_semantic_resume.py",
        }
    )
    return {
        **base,
        "schema": "umi-studio-roi-broad-semantic-gradient-proof/1",
        "policy": {
            **base["policy"],
            "scope": "one complete longest clip per prepared family plus longest byte caption",
            "complete_stage_training_clips": len(rows),
            "maximum_stage_caption_words": 6,
            "shared_phase_change": "only bind the complete broader plan validator",
        },
        "records": [rows[i] for i in indices],
        "record_indices": indices,
        "data": artifact(DATA),
        "data_plan": artifact(DATA_PLAN),
        "original_semantic_proof": {
            "path": "checkpoints/roi-semantic-gradient-v2/qualify/qualification.json",
            "sha256": sha256(
                ROOT / "checkpoints/roi-semantic-gradient-v2/qualify/qualification.json"
            ),
        },
        "original_initialization_sha256": proof["vision_initialization_sha256"],
        "source_sha256": {name: sha256(ROOT / name) for name in sorted(source_names)},
        "promotion_authorized": False,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("broad ROI proof data, implementation or scientific policy changed")
    verify_snapshot(path, plan)
    return plan


def bound_phase(mode):
    """Reuse exact qualified phase bytecode with one control-plane binding.

    The shared module and all numerical/scoring globals remain untouched. Its
    old end-of-phase validator describes the88-clip lineage, so a private globals
    copy replaces only that validator with the broader immutable-plan check.
    """
    function = {"candidates": shared.prepare, "qualify": shared.qualify}[mode]
    return FunctionType(
        function.__code__,
        {**function.__globals__, "checked": checked},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("broad ROI proof needs a fresh owned plan")
    torch.set_num_threads(2)
    plan = specification()
    storage()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path), "cases": len(plan["records"])}), flush=True)


def review(output):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("broad ROI preview needs a fresh owned runs/ directory")
    _, _, rows = checked_data()
    chosen = [rows[i] for i in proof_indices(rows)]
    storage()
    output.mkdir(parents=True)
    preview(output, chosen)
    write(
        output / "receipt.json",
        {
            "data_plan": artifact(DATA_PLAN),
            "data": artifact(DATA),
            "selected_training_records": chosen,
            "preview": artifact(output / "preview.png"),
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
        },
    )
    print(json.dumps({"preview": str(output / "preview.png"), "rows": len(chosen)}), flush=True)


def run(path, output, mode, candidates=None):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("broad ROI proof needs a fresh owned output")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        if mode == "candidates":
            bound_phase(mode)(path, output, plan)
        else:
            from train_terra_c5_tail import gpu_lease

            if mode != "qualify" or candidates is None:
                raise ValueError("broad ROI numerics need completed actual candidate rewards")
            with gpu_lease():
                bound_phase(mode)(path, output, plan, candidates.resolve())
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def pipeline(path, output):
    path, output = path.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("broad ROI pipeline needs a fresh owned output")
    plan = json.loads(path.read_text())
    verify_snapshot(path, plan)
    if plan["source_sha256"]["tools/probe_roi_broad_semantic.py"] != sha256(Path(__file__)):
        raise ValueError("broad ROI pipeline source differs from its frozen plan")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        for mode in ("candidates", "qualify"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--plan",
                str(path),
                "--output",
                str(output / mode),
                "--mode",
                mode,
            ]
            if mode == "qualify":
                command.extend(["--candidates", str(output / "candidates/candidates.json")])
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"broad ROI phase {mode} stopped with exit{result.returncode}; no retry"
                )
        write(
            output / "complete.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "qualification": artifact(output / "qualify/qualification.json"),
                "optimizer_updates": 0,
                "automatic_retries": 0,
            },
        )
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("candidates", "qualify", "pipeline"))
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()
    if args.review:
        review(args.review)
    elif args.freeze:
        freeze(args.freeze)
        if args.output and args.mode == "pipeline":
            pipeline(args.freeze, args.output)
    elif args.plan and args.output and args.mode == "pipeline":
        pipeline(args.plan, args.output)
    elif args.plan and args.output and args.mode:
        run(args.plan, args.output, args.mode, args.candidates)
    else:
        parser.error("choose --review, --freeze, or --plan --output --mode")


if __name__ == "__main__":
    main()
