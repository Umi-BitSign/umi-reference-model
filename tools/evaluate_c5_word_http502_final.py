"""Serial full96/full34 evaluation after the independent second-recovery audit."""

from __future__ import annotations

import json
from pathlib import Path

import c5_mixed_risk_evidence as evidence
import c5_mixed_semantic_model as model
import evaluate_c5_word_http502_sentences as sentences
import evaluate_c5_word_http502_words as words
import evaluate_c5_word_storage_final as template
from audit_c5_word_http502_lineage import storage_audit_fields
from benchmark_fleurs_dev import sha256
from c5_word_http502_final import checked_final
from terra_polarity_judge import bind
from train_c5_word_http502_recovery import SOURCES

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = "tools/audit_c5_word_http502_lineage.py"


def word_initialization(training_path, result_path):
    return bind(model.word_initialization, checked_final=checked_final)(training_path, result_path)


def previous_evaluation(path, kind, initialization, *, evaluation_only=False):
    return bind(evidence.previous_evaluation, word_evaluation=words, sentence_evaluation=sentences)(
        path, kind, initialization, evaluation_only=evaluation_only
    )


def derived_prerequisites_ast():
    return bind(template.derived_prerequisites_ast, AUDITOR=AUDITOR)()


def bound_prerequisites():
    return bind(
        template.bound_prerequisites,
        ROOT=ROOT,
        word_initialization=word_initialization,
        derived_prerequisites_ast=derived_prerequisites_ast,
    )()


def prerequisites(training_path, result_path, audit_path):
    initialization = bound_prerequisites()(training_path, result_path, audit_path)
    plan = json.loads(training_path.read_text())
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    fields = storage_audit_fields(
        plan, result, result_path.parent, initialization["word_training_evidence"]
    )
    if (
        any(audit.get(key) != value for key, value in fields.items())
        or plan["lineage_audit_source"]["sha256"] != plan["source_sha256"][AUDITOR]
        or audit["audit_source"] != plan["lineage_audit_source"]
    ):
        raise ValueError("HTTP502 final audit lost nested accounting, approvals or working history")
    return initialization


def control_sources():
    sources = template.control_sources()
    for name in SOURCES:
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return sources


def locations():
    return {
        "word": (
            ROOT / "runs/c5-http502-word-evaluation-v1/plan.json",
            ROOT / "checkpoints/c5-http502-word-evaluation-v1",
            "evaluate_c5_word_http502_words.py",
        ),
        "sentence": (
            ROOT / "runs/c5-http502-sentence-evaluation-v1/plan.json",
            ROOT / "checkpoints/c5-http502-sentence-evaluation-v1",
            "evaluate_c5_word_http502_sentences.py",
        ),
    }


def pipeline(training_path, result_path, audit_path, output):
    return bind(
        template.original.pipeline,
        prerequisites=prerequisites,
        locations=locations,
        control_sources=control_sources,
        previous_evaluation=previous_evaluation,
        __file__=__file__,
    )(training_path, result_path, audit_path, output)


def main(argv=None):
    return bind(template.main, pipeline=pipeline)(argv)


if __name__ == "__main__":
    main()
