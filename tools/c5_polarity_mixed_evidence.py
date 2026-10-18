"""Gate a later mixed proof on the audited fresh word final and full polarity evaluations."""

from __future__ import annotations

import c5_mixed_risk_evidence as original
import c5_mixed_semantic_model as model
import evaluate_c5_polarity_final as final
from benchmark_fleurs_dev import sha256
from terra_polarity_judge import bind, protocol
from terra_training_recovery import artifact

ROOT = original.ROOT


def word_initialization(training_path, result_path):
    return final.prerequisites(
        training_path, result_path, training_path.with_name("terminal-audit.json")
    )


def previous_evaluation(path, kind, initialization):
    # Evaluation-only decline recording must never authorize this next proof.
    return final.previous_evaluation(path, kind, initialization, evaluation_only=False)


def checked_initialization(binding):
    return bind(model.checked_initialization, word_initialization=word_initialization)(binding)


def load_student(plan):
    return bind(model.load_student, checked_initialization=checked_initialization)(plan)


def specification(
    training_path,
    result_path,
    input_plan,
    input_result,
    word_parent,
    sentence_parent,
    proposal_path=original.PROPOSAL,
):
    evidence = bind(
        original.specification,
        word_initialization=word_initialization,
        previous_evaluation=previous_evaluation,
    )(
        training_path,
        result_path,
        input_plan,
        input_result,
        word_parent,
        sentence_parent,
        proposal_path,
    )
    if evidence["initialization"]["original_model_plan"]["judge_protocol"] != protocol():
        raise ValueError("mixed polarity proof requires the same qualified word judge protocol")
    sources = dict(evidence["source_sha256"])
    for name in ("c5_polarity_mixed_evidence.py", "probe_c5_polarity_mixed_semantic.py"):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        **evidence,
        "source_sha256": sources,
        "judge_protocol": protocol(),
        "word_terminal_audit": artifact(training_path.with_name("terminal-audit.json")),
        "fresh_mixed_optimizer_training_authorized": False,
    }
