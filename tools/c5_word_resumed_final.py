"""Full856 word-final verification with an explicit inherited committed prefix."""

from __future__ import annotations

import json
from pathlib import Path

import c5_word_final as original
from benchmark_fleurs_dev import sha256
from c5_word_resume import validate_inheritance
from evaluate_roi_short_translation import terminal
from terra_training_recovery import artifact
from train_c5_word_resume import checked

ROOT = Path(__file__).resolve().parents[1]


def checked_final(training_path, result_path):
    from transformers import ByT5Tokenizer

    training_path, result_path = training_path.resolve(), result_path.resolve()
    if not result_path.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("word continued final result must remain in owned checkpoints/")
    plan = checked(training_path)
    terminal(result_path)
    result = json.loads(result_path.read_text())
    original.require_final(result, plan, sha256(training_path))
    directory = result_path.parent
    if list(directory.rglob("failure.json")) or list(directory.rglob("stopped.json")):
        raise ValueError("word continued final cannot consume another stopped lineage")
    inheritance, inherited = validate_inheritance(directory, plan)
    run_path, steps_path = directory / "run.json", directory / "steps.jsonl"
    run = json.loads(run_path.read_text())
    resume = plan["resume"]
    if (
        run.get("plan_sha256") != sha256(training_path)
        or run.get("initial_visual_sha256") != plan["vision_initialization_sha256"]
        or run.get("fixed_backbones") != plan["model_plan"]["initial_reference_hashes"]
        or run.get("recipe") != plan["recipe"]
        or run.get("unique_training_clips") != len(plan["records"])
        or run.get("training_examples") != resume["training_examples"]
        or run.get("optimizer_updates") != resume["optimizer_updates"]
        or run.get("trainable_parameters") != 43260672
        or run.get("resume") != resume
        or run.get("inheritance") != artifact(directory / "inheritance.json")
        or run.get("restored_visual_state_sha256") != resume["starting_visual_state_sha256"]
    ):
        raise ValueError("word continued final lost truthful restored counters or provenance")
    claim_path = (ROOT / resume["source_plan"]["path"]).with_name("word-manual-continuation.json")
    claim = json.loads(claim_path.read_text())
    if (
        claim.get("plan") != artifact(training_path)
        or claim.get("worker") != artifact(directory / "worker.json")
        or claim.get("output") != str(directory.relative_to(ROOT))
    ):
        raise ValueError("word continued final lost its single-continuation ownership claim")
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    original.verify_steps(plan, steps, result, resource_reset_batches={resume["completed_batches"]})
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["model_plan"]["reference"]["bundle_root"] / "model/models/byt5_base",
        local_files_only=True,
    )
    assets = [
        artifact(run_path),
        artifact(steps_path),
        artifact(directory / "inheritance.json"),
        artifact(claim_path),
    ]
    assets.extend(value for value in resume.values() if isinstance(value, dict))
    assets.extend(original.verified_batch_evidence(plan, directory, steps, tokenizer))
    assets.extend(
        original.checked_checkpoints(
            plan, result, directory, sha256(training_path), steps, inherited=inherited
        )
    )
    # Retain original reference fields alongside the explicitly authenticated inheritance.
    if inheritance["source_plan"] != resume["source_plan"]:
        raise ValueError("word continued final changed inherited source")
    return plan, result, assets
