"""Full34 fixed-final sentences after the complete second approved word recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_c5_word_sentences as original
from benchmark_fleurs_dev import sha256
from c5_word_http502_final import checked_final
from evaluate_c5_word_http502_words import authenticated_passes
from terra_polarity_judge import Pacer, bind, protocol
from terra_training_recovery import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]


def specification(training_path, result_path):
    plan = bind(original.specification, checked_final=checked_final)(training_path, result_path)
    return {
        **plan,
        "schema": "umi-studio-C5-HTTP502-recovery-sentence-evaluation/1",
        "judge_protocol": protocol(),
        "source_sha256": {
            **plan["source_sha256"],
            "tools/evaluate_c5_word_http502_sentences.py": sha256(Path(__file__)),
            "tools/evaluate_c5_word_http502_words.py": sha256(
                ROOT / "tools/evaluate_c5_word_http502_words.py"
            ),
        },
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification(
        ROOT / plan["training_plan"]["path"], ROOT / plan["training_result"]["path"]
    ):
        raise ValueError("polarity sentence evaluation changed complete final, controls or prompt")
    verify_snapshot(path, plan)
    return plan


def freeze(path, training_path, result_path):
    return bind(original.freeze, specification=specification)(path, training_path, result_path)


def bound_inference():
    return bind(
        original.native.infer,
        appearance_model=original.candidate_appearance,
        checked=checked,
        storage=original.storage,
    )


def infer(path, output, plan):
    return bind(original.infer, bound_inference=bound_inference)(path, output, plan)


def run_inference(path, output):
    return bind(original.native.run, checked=checked, infer=infer, storage=original.storage)(
        path, output
    )


def score_specification(inference_path, decoded_path):
    plan = bind(original.scoring.score_specification, checked=checked)(inference_path, decoded_path)
    return {
        **plan,
        "schema": "umi-studio-C5-HTTP502-recovery-sentence-semantic-scoring/1",
        "judge_protocol": protocol(),
    }


def checked_scores(path):
    plan = json.loads(path.read_text())
    if plan != score_specification(
        ROOT / plan["inference_plan"]["path"], ROOT / plan["decoded"]["path"]
    ):
        raise ValueError(
            "polarity sentence scores changed complete outputs, prompt or semantic gate"
        )
    verify_snapshot(path, plan)
    return plan


def score_comparison(comparison, output, pacer):
    measured = original.scoring.score_comparison(comparison, output, pacer)
    actual = authenticated_passes(comparison, output)
    if actual != measured:
        raise ValueError("sentence score results differ from complete actual raw response passes")
    return actual


def run_scores(path, output):
    return bind(
        original.scoring.run,
        checked_scores=checked_scores,
        storage=original.storage,
        RecoveryPacer=Pacer,
        score_comparison=score_comparison,
    )(path, output, scoring=True)


def pipeline(path, output):
    return bind(
        original.pipeline,
        __file__=__file__,
        checked=checked,
        score_specification=score_specification,
    )(path, output)


def main(argv=None):
    original.torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "training-plan", "training-result", "plan", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--mode", choices=("inference", "scores"))
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args(argv)
    if args.mode not in ("inference", "scores"):
        original.torch.set_num_interop_threads(1)
    if args.freeze and args.training_plan and args.training_result:
        freeze(args.freeze, args.training_plan, args.training_result)
    path = args.plan or args.freeze
    if path and args.output and args.pipeline:
        pipeline(path, args.output)
    elif path and args.output and args.mode:
        (run_scores if args.mode == "scores" else run_inference)(path, args.output)
    elif not args.freeze or args.output:
        parser.error(
            "freeze from a complete final, or execute a complete frozen sentence comparison"
        )


if __name__ == "__main__":
    main()
