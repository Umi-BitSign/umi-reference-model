"""Serial full96/full34 evaluation after the approved storage-recovery audit."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path

import c5_mixed_risk_evidence as evidence
import c5_mixed_semantic_model as model
import evaluate_c5_word_final as original
import evaluate_c5_word_storage_sentences as sentences
import evaluate_c5_word_storage_words as words
from audit_c5_word_storage_lineage import storage_audit_fields
from benchmark_fleurs_dev import sha256
from c5_word_storage_final import checked_final
from terra_polarity_judge import bind

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = "tools/audit_c5_word_storage_lineage.py"


def word_initialization(training_path, result_path):
    return bind(model.word_initialization, checked_final=checked_final)(training_path, result_path)


def previous_evaluation(path, kind, initialization, *, evaluation_only=False):
    return bind(evidence.previous_evaluation, word_evaluation=words, sentence_evaluation=sentences)(
        path, kind, initialization, evaluation_only=evaluation_only
    )


def derived_prerequisites_ast():
    definition = ast.parse(inspect.getsource(original.prerequisites)).body[0]
    count = 0
    for node in ast.walk(definition):
        if isinstance(node, ast.Constant) and node.value == "tools/audit_c5_word_lineage.py":
            node.value = AUDITOR
            count += 1
    if count != 1:
        raise ValueError("cannot bind the reviewed storage auditor to original final prerequisites")
    return ast.fix_missing_locations(definition)


def bound_prerequisites():
    namespace = {
        **original.prerequisites.__globals__,
        "ROOT": ROOT,
        "word_initialization": word_initialization,
    }
    exec(
        compile(
            ast.Module(body=[derived_prerequisites_ast()], type_ignores=[]),
            "<storage-final-prerequisites>",
            "exec",
        ),
        namespace,
    )
    return namespace["prerequisites"]


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
        raise ValueError(
            "storage final audit lost its accounting, approval, protocol or working history"
        )
    return initialization


def control_sources():
    sources = original.control_sources()
    for name in (
        "evaluate_c5_word_storage_final.py",
        "evaluate_c5_word_storage_words.py",
        "evaluate_c5_word_storage_sentences.py",
        "c5_word_storage_final.py",
        "train_c5_word_storage_recovery.py",
        "c5_word_storage_recovery.py",
        "c5_word_working_checkpoints.py",
        "audit_c5_word_storage_lineage.py",
        "audit_c5_word_lineage.py",
        "c5_word_resumed_final.py",
        "c5_polarity_word_final.py",
        "terra_polarity_judge.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return sources


def locations():
    return {
        "word": (
            ROOT / "runs/c5-storage-word-evaluation-v1/plan.json",
            ROOT / "checkpoints/c5-storage-word-evaluation-v1",
            "evaluate_c5_word_storage_words.py",
        ),
        "sentence": (
            ROOT / "runs/c5-storage-sentence-evaluation-v1/plan.json",
            ROOT / "checkpoints/c5-storage-sentence-evaluation-v1",
            "evaluate_c5_word_storage_sentences.py",
        ),
    }


def pipeline(training_path, result_path, audit_path, output):
    return bind(
        original.pipeline,
        prerequisites=prerequisites,
        locations=locations,
        control_sources=control_sources,
        previous_evaluation=previous_evaluation,
        __file__=__file__,
    )(training_path, result_path, audit_path, output)


def main(argv=None):
    original.torch.set_num_threads(2)
    original.torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-plan", "training-result", "terminal-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    pipeline(args.training_plan, args.training_result, args.terminal_audit, args.output)


if __name__ == "__main__":
    main()
