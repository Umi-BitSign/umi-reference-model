"""Use the unchanged paired Terra gate only after complete native112 inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import FunctionType

import evaluate_roi_semantic as shared
import torch
from benchmark_fleurs_dev import sha256
from evaluate_c5_small_input import ROOT, checked
from launch_terra_visual_job import identity, now, write
from probe_roi_seen_translation import freeze
from terra_training_recovery import artifact, verify_snapshot
from terra_training_storage import storage


def bound_phase(name):
    """Reuse bytecode and all scientific globals; replace only plan validation."""
    function = {"specification": shared.score_specification, "run": shared.run}[name]
    replacements = (
        {"checked": checked} if name == "specification" else {"checked_scores": checked_scores}
    )
    result = FunctionType(
        function.__code__,
        {**function.__globals__, **replacements},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = (
        None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    )
    return result


def checked_scores(path):
    plan = json.loads(path.read_text())
    if plan != bound_phase("specification")(
        ROOT / plan["inference_plan"]["path"], ROOT / plan["decoded"]["path"]
    ):
        raise ValueError("native112 scoring changed actual outputs, model, rubric or paired gate")
    verify_snapshot(path, plan)
    return plan


def freeze_scores(path, inference_plan, decoded):
    torch.set_num_threads(2)
    freeze(path, bound_phase("specification")(inference_plan, decoded))


def run(path, output):
    # Never expose the historical function's alternative ROI inference mode.
    return bound_phase("run")(path, output, scoring=True)


def pipeline(path, output):
    """Run each own child once, freezing actual scores only after inference exit0."""
    path, output = path.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("native112 comparison requires a fresh owned pipeline output")
    plan = json.loads(path.read_text())
    verify_snapshot(path, plan)
    if plan["source_sha256"].get("tools/score_c5_small_input.py") != sha256(Path(__file__)):
        raise ValueError("native112 comparison controller differs from its pinned source")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        command = [
            sys.executable,
            str(ROOT / "tools/evaluate_c5_small_input.py"),
            "--plan",
            str(path),
            "--output",
            str(output / "inference"),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise RuntimeError("native112 inference stopped; scoring not started, no retry")
        score_plan = path.parent.with_name(path.parent.name + "-scores") / "plan.json"
        freeze_scores(score_plan, path, output / "inference/receipt.json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--plan",
            str(score_plan),
            "--output",
            str(output / "scoring"),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise RuntimeError("native112 semantic scoring stopped; no retry or model fallback")
        write(
            output / "complete.json",
            {
                "complete": True,
                "inference_plan": artifact(path),
                "scoring_plan": artifact(score_plan),
                "decision": artifact(output / "scoring/decision.json"),
                "optimizer_updates": 0,
                "automatic_retries": 0,
                "promotion_authorized": False,
            },
        )
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("freeze", "inference_plan", "decoded", "plan", "output"):
        parser.add_argument("--" + field.replace("_", "-"), type=Path)
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.freeze and args.inference_plan and args.decoded:
        freeze_scores(args.freeze, args.inference_plan, args.decoded)
    elif args.pipeline and args.inference_plan and args.output:
        pipeline(args.inference_plan, args.output)
    elif args.plan and args.output and not args.pipeline:
        run(args.plan, args.output)
    else:
        parser.error("choose --freeze with inference dependencies, --pipeline, or --plan/--output")


if __name__ == "__main__":
    main()
