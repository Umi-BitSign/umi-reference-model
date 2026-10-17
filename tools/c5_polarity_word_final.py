"""Complete fresh polarity-word finals only; no legacy/resumed or partial checkpoint admission."""

from __future__ import annotations

import json
from pathlib import Path

import c5_word_final as original
import train_c5_polarity_words as training
from terra_polarity_judge import bind, verified_training_rewards
from terra_training_recovery import artifact

ROOT = Path(__file__).resolve().parents[1]


def verified_batch_evidence(plan, directory, steps, tokenizer):
    return bind(original.verified_batch_evidence, verified_rewards=verified_training_rewards)(
        plan, directory, steps, tokenizer
    )


def checked_final(training_path, result_path):
    training_path, result_path = training_path.resolve(), result_path.resolve()
    metadata = json.loads(training_path.read_text())
    if "resume" in metadata or metadata.get("schema") != training.SCHEMA:
        raise ValueError("polarity final requires a fresh public-C5 curriculum without inheritance")
    directory = result_path.parent
    if list(directory.rglob("failure.json")) or list(directory.rglob("stopped.json")):
        raise ValueError("polarity final cannot consume a stopped or failed curriculum")
    plan, result, evidence = bind(
        original.checked_final,
        checked_training=training.checked,
        verified_batch_evidence=verified_batch_evidence,
    )(training_path, result_path)
    claim_path = training_path.with_name("execution-claim.json")
    proof_claim_path = (ROOT / plan["qualified_plan"]["path"]).with_name(
        "optimizer-child-claim.json"
    )
    record = json.loads(claim_path.read_text())
    if (
        record != json.loads(proof_claim_path.read_text())
        or record.get("plan") != artifact(training_path)
        or record.get("worker") != artifact(directory / "worker.json")
        or record.get("output") != str(directory.relative_to(ROOT))
        or record.get("qualified_plan") != plan["qualified_plan"]
        or record.get("single_execution") is not True
        or record.get("inherited_exposures") != 0
        or record.get("inherited_optimizer_updates") != 0
        or record.get("manual_continuations_authorized") != 0
    ):
        raise ValueError("polarity final lost its single fresh execution claim")
    return plan, result, [*evidence, artifact(claim_path), artifact(proof_claim_path)]
