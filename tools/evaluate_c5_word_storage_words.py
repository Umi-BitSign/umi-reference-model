"""Full96 fixed-final words after an approved, complete storage recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_c5_words as original
from benchmark_fleurs_dev import sha256
from c5_word_storage_final import checked_final
from terra_polarity_judge import Pacer, bind, payload, protocol
from terra_training_recovery import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]


def specification(training_path, result_path):
    plan = bind(original.specification, checked_final=checked_final)(training_path, result_path)
    return {
        **plan,
        "schema": "umi-studio-C5-storage-recovery-word-evaluation/1",
        "judge_protocol": protocol(),
        "source_sha256": {
            **plan["source_sha256"],
            "tools/evaluate_c5_word_storage_words.py": sha256(Path(__file__)),
        },
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification(
        ROOT / plan["training_plan"]["path"], ROOT / plan["training_result"]["path"]
    ):
        raise ValueError("polarity word evaluation changed complete training, controls or prompt")
    verify_snapshot(path, plan)
    return plan


def freeze(path, training_path, result_path):
    return bind(original.freeze, specification=specification)(path, training_path, result_path)


def bound_baseline():
    return original.torch.no_grad()(bind(original.baseline.__wrapped__, checked=checked))


def bound_inference():
    return original.torch.no_grad()(bind(original.infer.__wrapped__, checked=checked))


def baseline(path, output, plan):
    return bound_baseline()(path, output, plan)


def infer(path, output, plan, baseline_path):
    return bound_inference()(path, output, plan, baseline_path)


def score_specification(plan_path, decoded_path):
    plan = bind(original.score_specification, checked=checked)(plan_path, decoded_path)
    return {
        **plan,
        "schema": "umi-studio-C5-storage-recovery-word-semantic-scoring/1",
        "judge_protocol": protocol(),
    }


def checked_scores(path):
    plan = json.loads(path.read_text())
    if plan != score_specification(
        ROOT / plan["inference_plan"]["path"], ROOT / plan["decoded"]["path"]
    ):
        raise ValueError("polarity word scores changed actual complete outputs, prompt or gate")
    verify_snapshot(path, plan)
    return plan


def authenticated_passes(comparison, directory):
    return bind(original.authenticated_passes, terra_payload=payload)(comparison, directory)


def score(path, output, plan):
    return bind(
        original.score,
        RecoveryPacer=Pacer,
        authenticated_passes=authenticated_passes,
        checked_scores=checked_scores,
    )(path, output, plan)


def run(path, output, mode, baseline_path=None):
    return bind(
        original.run,
        checked=checked,
        checked_scores=checked_scores,
        baseline=baseline,
        infer=infer,
        score=score,
    )(path, output, mode, baseline_path)


def pipeline(path, output):
    return bind(
        original.pipeline,
        __file__=__file__,
        checked=checked,
        score_specification=score_specification,
    )(path, output)


def main(argv=None):
    original.torch.set_num_threads(2)
    original.torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "training-plan", "training-result", "plan", "output", "baseline"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--mode", choices=("baseline", "inference", "scores"))
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze and args.training_plan and args.training_result:
        freeze(args.freeze, args.training_plan, args.training_result)
    path = args.plan or args.freeze
    if path and args.output and args.pipeline:
        pipeline(path, args.output)
    elif path and args.output and args.mode:
        run(path, args.output, args.mode, args.baseline)
    elif not args.freeze or args.output:
        parser.error("freeze from complete training, or execute one complete frozen comparison")


if __name__ == "__main__":
    main()
