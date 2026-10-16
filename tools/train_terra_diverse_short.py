"""Change only the fixed training candidate pools; reuse the qualified replay driver."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import train_terra_frozen_short as original_driver
from benchmark_fleurs_dev import sha256
from grade_terra_diverse_short import PENDING, apply_scores
from grade_terra_diverse_short import checked as checked_grading
from launch_terra_visual_job import write
from prepare_terra_short_curriculum import ROOT
from probe_jeff_terra_quality import terra_payload
from semantic_quality_judge import parse_terra_scores
from terra_frozen_short import RECIPE, key, proof_cases
from terra_training_recovery import artifact, verify_snapshot

PARENT = ROOT / "runs/terra-frozen-short-training-v1/plan.json"
GRADING = ROOT / "runs/terra-diverse-short-grading-v1/plan.json"
REWARDS = ROOT / "checkpoints/terra-diverse-short-grading-v1/graded-pools.json"
REWARDS_SHA = "97003ea75e594865d6eb787a210f0ff46e7a7b2e5ab9899d484feb80c70beb45"
DRIVER_SHA = "9e066c8a8f1c74806f4ab911cc40c94c9a5105658bb6807146a79f180bd3ad0d"
PROFILE = {
    **RECIPE,
    "candidate_selection": (
        "original authenticated pools plus four reference-blind original-C5 draws"
    ),
}


def graded_rewards(data):
    """Reparse every real response and reconstruct all labels before optimizer work."""
    if sha256(REWARDS) != REWARDS_SHA:
        raise ValueError("diverse short training labels changed")
    plan = checked_grading(GRADING)
    document = json.loads(REWARDS.read_text())
    terminal = json.loads(REWARDS.with_name("exit.json").read_text())
    worker = json.loads(REWARDS.with_name("worker.json").read_text())
    if (
        document.get("schema") != "umi-studio-terra-diverse-short-graded-pools/1"
        or document.get("plan_sha256") != sha256(GRADING)
        or document.get("pending_candidates") != artifact(PENDING)
        or document.get("model_training_ready") is not True
        or document.get("bookend_calibration_passed") is not True
        or document.get("original_scores_preserved") is not True
        or document.get("development_used") is not False
        or document.get("optimizer_updates") != 0
        or document.get("new_Terra_requests") != len(plan["batches"])
        or terminal.get("exit_code") != 0
        or terminal.get("pid") != worker.get("pid")
        or len(document["new_score_receipts"]) != len(plan["batches"])
    ):
        raise ValueError("diverse short training requires complete calibrated grading")
    scores = [None] * len(plan["items"])
    for number, (batch, receipt) in enumerate(
        zip(plan["batches"], document["new_score_receipts"], strict=True)
    ):
        directory = REWARDS.parent / f"batch-{number:03d}"
        if receipt != {
            "batch": batch,
            **{
                name: artifact(directory / f"{name}.json")
                for name in ("request", "response", "scores")
            },
        }:
            raise ValueError("diverse short score provenance changed")
        rows = [
            {
                k: (plan["controls"] if r["kind"] == "control" else plan["items"])[r["index"]][k]
                for k in ("reference", "candidate")
            }
            for r in batch
        ]
        response = json.loads((directory / "response.json").read_text())
        result = json.loads((directory / "scores.json").read_text())
        parsed = parse_terra_scores(response, [f"pair_{i:02d}" for i in range(len(batch))])
        values = [parsed[f"pair_{i:02d}"] for i in range(len(batch))]
        if (
            response.get("model") != "gpt-5.6-terra"
            or response.get("reasoning", {}).get("effort") != "medium"
            or json.loads((directory / "request.json").read_text()) != terra_payload(rows)
            or result["response_sha256"] != sha256(directory / "response.json")
            or result["scores"] != values
        ):
            raise ValueError("diverse short scores differ from actual Terra-medium response")
        for entry, value in zip(batch, values, strict=True):
            if entry["kind"] == "control":
                lo, hi = plan["controls"][entry["index"]]["range"]
                if not lo <= value <= hi:
                    raise ValueError("diverse short grading calibration does not pass")
            else:
                scores[entry["index"]] = value
    expected = apply_scores(json.loads(PENDING.read_text())["records"], plan["positions"], scores)
    if document["records"] != expected:
        raise ValueError("diverse short label arrays do not match real scores")
    allowed = {
        key(r): r
        for r in data["records"]
        if r["role"] == "training" and r["training_allowed"] is True
    }
    if (
        len(expected) != len(allowed)
        or len({key(r["data_row"]) for r in expected}) != len(expected)
        or any(r["data_row"] != allowed.get(key(r["data_row"])) for r in expected)
    ):
        raise ValueError("diverse short training coverage or strict role changed")
    return expected


def specification():
    parent = original_driver.checked(PARENT)
    data = original_driver.checked_data(ROOT / parent["data"]["path"])
    rewards = graded_rewards(data)
    grading = json.loads(GRADING.read_text())
    parent.update(
        {
            "schema": "umi-studio-terra-diverse-short-training/1",
            "recipe": PROFILE,
            "historical_training_rewards": artifact(REWARDS),
            "historical_reward_grading_plan": artifact(GRADING),
            "original_optimizer_driver": {
                "path": "tools/train_terra_frozen_short.py",
                "sha256": DRIVER_SHA,
            },
            "integration_proof_identities": [
                list(key(r["data_row"])) for r in proof_cases(rewards)
            ],
            "only_training_candidate_pool_changes": True,
            "training_score_preparation_requests": len(grading["batches"]),
        }
    )
    parent["source_sha256"].update(grading["source_sha256"])
    parent["source_sha256"]["tools/train_terra_diverse_short.py"] = sha256(Path(__file__))
    return parent


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("diverse short training profile, sources or data changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("diverse short training requires a fresh owned plan")
    storage()
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps({"plan_sha256": sha256(path), "updates": plan["planned_optimizer_updates"]}),
        flush=True,
    )


def runtime():
    """Isolate three explicit profile dependencies; never edit frozen driver sources."""
    source = ROOT / "tools/train_terra_frozen_short.py"
    if sha256(source) != DRIVER_SHA:
        raise ValueError("qualified optimizer driver changed")
    # All numerical methods, sampler, schedule, checkpointing and evaluation remain
    # the prior driver. A private module keeps original parent validators unchanged.
    spec = importlib.util.spec_from_file_location("_terra_diverse_short_driver", source)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.RECIPE = PROFILE
    driver.checked_rewards = graded_rewards
    driver.checked = checked
    return driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output:
        runtime().run(
            args.plan,
            args.output,
            qualification_only=args.qualification_only,
            qualification=args.qualification,
        )
    else:
        parser.error("provide --freeze or --plan and --output with a qualification mode")


if __name__ == "__main__":
    main()
