"""Serial full96/full34 polarity evaluations after the independent complete-training audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import c5_mixed_risk_evidence as evidence
import c5_mixed_semantic_model as model
import evaluate_c5_polarity_sentences as sentences
import evaluate_c5_polarity_words as words
import evaluate_c5_word_final as original
from benchmark_fleurs_dev import sha256
from c5_polarity_word_final import checked_final
from terra_polarity_judge import bind

ROOT = Path(__file__).resolve().parents[1]
TRAINING_PLAN = ROOT / "runs/c5-polarity-word-training-v1/plan.json"
TRAINING_RESULT = ROOT / "checkpoints/c5-polarity-word-training-v1/result.json"


def word_initialization(training_path, result_path):
    return bind(model.word_initialization, checked_final=checked_final)(training_path, result_path)


def previous_evaluation(path, kind, initialization, *, evaluation_only=False):
    return bind(evidence.previous_evaluation, word_evaluation=words, sentence_evaluation=sentences)(
        path, kind, initialization, evaluation_only=evaluation_only
    )


def prerequisites(training_path, result_path, audit_path):
    initialization = bind(
        original.prerequisites,
        word_initialization=word_initialization,
        TERMINAL_AUDIT_SOURCE_SHA256=sha256(ROOT / "tools/audit_c5_polarity_word.py"),
    )(training_path, result_path, audit_path)
    plan = json.loads(training_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (
        audit["audit_source"] != plan["terminal_audit_source"]
        or plan["terminal_audit_source"]["sha256"]
        != plan["source_sha256"]["tools/audit_c5_polarity_word.py"]
        or audit.get("judge_protocol") != plan["judge_protocol"]
        or audit.get("original_public_initialization") is not True
    ):
        raise ValueError(
            "polarity final audit lost its original-initializer, source or protocol binding"
        )
    return initialization


def control_sources():
    sources = original.control_sources()
    for name in (
        "evaluate_c5_polarity_final.py",
        "evaluate_c5_polarity_words.py",
        "evaluate_c5_polarity_sentences.py",
        "c5_polarity_word_final.py",
        "train_c5_polarity_words.py",
        "audit_c5_polarity_word.py",
        "terra_polarity_judge.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return sources


def locations():
    return {
        "word": (
            ROOT / "runs/c5-polarity-word-evaluation-v1/plan.json",
            ROOT / "checkpoints/c5-polarity-word-evaluation-v1",
            "evaluate_c5_polarity_words.py",
        ),
        "sentence": (
            ROOT / "runs/c5-polarity-sentence-evaluation-v1/plan.json",
            ROOT / "checkpoints/c5-polarity-sentence-evaluation-v1",
            "evaluate_c5_polarity_sentences.py",
        ),
    }


def pipeline(training_path, result_path, audit_path, output):
    return bind(
        original.pipeline,
        prerequisites=prerequisites,
        locations=locations,
        control_sources=control_sources,
        previous_evaluation=previous_evaluation,
    )(training_path, result_path, audit_path, output)


def main(argv=None):
    original.torch.set_num_threads(2)
    original.torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, default=TRAINING_PLAN)
    parser.add_argument("--training-result", type=Path, default=TRAINING_RESULT)
    parser.add_argument(
        "--terminal-audit", type=Path, default=TRAINING_PLAN.with_name("terminal-audit.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    pipeline(args.training_plan, args.training_result, args.terminal_audit, args.output)


if __name__ == "__main__":
    main()
