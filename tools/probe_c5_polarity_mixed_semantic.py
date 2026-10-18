"""Prepare a zero-update mixed proof only after full fresh-word polarity evaluations."""

from __future__ import annotations

import json

import c5_polarity_mixed_evidence as evidence
import probe_c5_mixed_semantic as original
import terra_mixed_semantic_protocol as mixed
from terra_polarity_judge import Pacer, bind, verified_training_rewards
from terra_training_recovery import verify_snapshot

ROOT = original.ROOT
TRAINING_PLAN = ROOT / "runs/c5-polarity-word-training-v1/plan.json"
TRAINING_RESULT = ROOT / "checkpoints/c5-polarity-word-training-v1/result.json"
WORD_PARENT = ROOT / "checkpoints/c5-polarity-word-evaluation-v1/complete.json"
SENTENCE_PARENT = ROOT / "checkpoints/c5-polarity-sentence-evaluation-v1/complete.json"


def specification(
    training_path=TRAINING_PLAN,
    result_path=TRAINING_RESULT,
    input_plan=original.INPUT_PLAN,
    input_result=original.INPUT_RESULT,
    word_parent=WORD_PARENT,
    sentence_parent=SENTENCE_PARENT,
    proposal_path=original.PROPOSAL,
):
    plan = bind(original.specification, evidence_specification=evidence.specification)(
        training_path,
        result_path,
        input_plan,
        input_result,
        word_parent,
        sentence_parent,
        proposal_path,
    )
    return {**plan, "schema": "umi-studio-C5-polarity-mixed-semantic-gradient-proof/1"}


def checked(path):
    plan = json.loads(path.read_text())
    initialization = plan["initialization"]
    if plan != specification(
        ROOT / initialization["word_training_plan"]["path"],
        ROOT / initialization["word_training_result"]["path"],
        ROOT / plan["input_plan"]["path"],
        ROOT / plan["input_result"]["path"],
        ROOT / plan["preceding_evaluations"]["word"]["parent_complete"]["path"],
        ROOT / plan["preceding_evaluations"]["sentence"]["parent_complete"]["path"],
        ROOT / plan["metadata_proposal"]["path"],
    ):
        raise ValueError("mixed polarity proof changed complete predecessors, inputs or protocol")
    verify_snapshot(path, plan)
    return plan


def freeze(path, training_path=TRAINING_PLAN, result_path=TRAINING_RESULT):
    return bind(original.freeze, specification=specification)(path, training_path, result_path)


def verified_proof_rewards(paired, directory, seed):
    return bind(mixed.verified_proof_rewards, verified_rewards=verified_training_rewards)(
        paired, directory, seed
    )


def prepare(path, output, plan):
    return bind(
        original.prepare,
        load_student=evidence.load_student,
        RecoveryPacer=Pacer,
        verified_proof_rewards=verified_proof_rewards,
        checked=checked,
    )(path, output, plan)


def checked_candidates(path, plan_path, plan, tokenizer):
    return bind(original.checked_candidates, verified_proof_rewards=verified_proof_rewards)(
        path, plan_path, plan, tokenizer
    )


def bound_qualification():
    return bind(
        original.bound_qualification(),
        load_student=evidence.load_student,
        checked_candidates=checked_candidates,
        checked=checked,
    )


def qualify(path, output, plan, candidates):
    return bound_qualification()(path, output, plan, candidates)


def checked_qualification(result_path, plan_path, plan):
    return bind(original.checked_qualification, checked_candidates=checked_candidates)(
        result_path, plan_path, plan
    )


def run(path, output, mode, candidates=None):
    return bind(original.run, checked=checked, prepare=prepare, qualify=qualify)(
        path, output, mode, candidates
    )


def pipeline(path, output):
    return bind(
        original.pipeline,
        __file__=__file__,
        checked=checked,
        checked_candidates=checked_candidates,
        checked_qualification=checked_qualification,
    )(path, output)


def main(argv=None):
    original.torch.set_num_threads(2)
    return bind(
        original.main,
        TRAINING_PLAN=TRAINING_PLAN,
        TRAINING_RESULT=TRAINING_RESULT,
        freeze=freeze,
        run=run,
        pipeline=pipeline,
    )(argv)


if __name__ == "__main__":
    main()
