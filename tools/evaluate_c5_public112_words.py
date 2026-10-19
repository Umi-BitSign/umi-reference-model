"""Post-hoc resolution diagnostic on the already-open96-word development set.

The visual model starts from unchanged public values. No trained checkpoint,
partial epoch or independent confirmation clip is used for candidate inference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import evaluate_c5_small_input as native
import evaluate_c5_word_http502_words as trained
import evaluate_c5_words as original
from benchmark_fleurs_dev import sha256
from c5_experiment_sources import archive_sources
from launch_terra_visual_job import identity, now, write
from probe_c5_mixed_semantic import wait_for_predecessor
from terra_polarity_judge import Pacer, bind, payload
from terra_training_recovery import artifact, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PLAN = ROOT / "runs/c5-http502-word-evaluation-v1/plan.json"
BASELINE = ROOT / "checkpoints/c5-http502-word-evaluation-v1/baseline/receipt.json"
FINAL = ROOT / "checkpoints/c5-http502-final-evaluation-v1/complete.json"
DECLINE = ROOT / "checkpoints/c5-http502-word-evaluation-v1/scoring/decision.json"


def checked_release():
    original.terminal(FINAL)
    complete = json.loads(FINAL.read_text())
    decline = json.loads(DECLINE.read_text())
    if (
        complete.get("complete") is not True
        or set(complete["evaluations"]) != {"word", "sentence"}
        or decline.get("calibration_passed") is not True
        or decline.get("stop_for_semantic_decline") is not True
    ):
        raise ValueError(
            "resolution diagnostic requires the completed serial evaluation and real regression"
        )
    return {
        "complete": artifact(FINAL),
        "decline": artifact(DECLINE),
        "exit": artifact(FINAL.with_name("exit.json")),
        "worker": artifact(FINAL.with_name("worker.json")),
    }


def specification():
    release = checked_release()  # Refuse overlapping the active complete evaluation.
    plan = trained.checked(ORIGINAL_PLAN)
    result = {
        **plan,
        "schema": "umi-studio-public112-word-resolution-diagnostic/1",
        "original_evaluation_plan": artifact(ORIGINAL_PLAN),
        "baseline_control": artifact(BASELINE),
        "predecessor_release": release,
        "candidate_visual_state_sha256": plan["model_plan"]["pretrained_state_sha256"],
        "source_sha256": {
            **plan["source_sha256"],
            str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
        },
        "policy": {
            **plan["policy"],
            "scope": "post-hoc diagnosis on already-open96 development clips",
            "candidate": "original public visual values at native112; no training",
            "independent_confirmation": False,
            "promotion_authorized": False,
        },
        "trained_result_is_context_only": True,
        "trained_or_partial_visual_checkpoint_loaded": False,
        "post_hoc_diagnostic_only": True,
    }
    del result["candidate_visual"]
    return result


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError(
            "public112 diagnostic changed inputs, original values or completed predecessor"
        )
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    if path.parent.exists() or not path.absolute().is_relative_to(ROOT / "runs"):
        raise ValueError("public112 diagnostic needs a fresh owned plan")
    plan = specification()
    original.storage()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))


def appearance(plan, hands, face):
    # This path never opens the rejected visual tensor file.
    value = native.appearance_model(plan["model_plan"], hands, face)
    if original.state_digest(value.state_dict()) != plan["candidate_visual_state_sha256"]:
        raise ValueError("public112 diagnostic did not load exact original visual values")
    return value


def checked_baseline(receipt_path, plan_path, plan, tokenizer):
    if (
        artifact(receipt_path) != plan["baseline_control"]
        or artifact(ORIGINAL_PLAN) != plan["original_evaluation_plan"]
    ):
        raise ValueError("public112 diagnostic cannot replace its original224 controls")
    source = trained.checked(ORIGINAL_PLAN)
    if any(
        plan[k] != source[k]
        for k in ("records", "reference", "model_plan", "initial_reference_hashes")
    ):
        raise ValueError("public112 diagnostic changed baseline inputs or fixed values")
    return original.checked_baseline(receipt_path, ORIGINAL_PLAN, source, tokenizer)


def infer(path, output, plan, baseline_path):
    return original.torch.no_grad()(
        bind(
            original.infer.__wrapped__,
            checked=checked,
            candidate_appearance=appearance,
            checked_baseline=checked_baseline,
        )
    )(path, output, plan, baseline_path)


def score_specification(plan_path, decoded_path):
    result = bind(original.score_specification, checked=checked, checked_baseline=checked_baseline)(
        plan_path, decoded_path
    )
    return {
        **result,
        "schema": "umi-studio-public112-word-diagnostic-scoring/1",
        "judge_protocol": checked(plan_path)["judge_protocol"],
        "post_hoc_diagnostic_only": True,
        "no_trained_model_win_can_be_established": True,
    }


def checked_scores(path):
    value = json.loads(path.read_text())
    if value != score_specification(
        ROOT / value["inference_plan"]["path"], ROOT / value["decoded"]["path"]
    ):
        raise ValueError("public112 diagnostic scoring changed its actual complete inference")
    verify_snapshot(path, value)
    return value


def authenticated_passes(comparison, directory):
    return bind(original.authenticated_passes, terra_payload=payload)(comparison, directory)


def wait_after_predecessor():
    release = checked_release()
    predecessor = json.loads(FINAL.read_text())["evaluations"]["sentence"]
    return {
        "predecessor": release["complete"],
        "not_before": predecessor["API_start_not_before"],
        "ready_at": wait_for_predecessor(predecessor),
    }


def score(path, output, plan):
    write(output / "API-spacing.json", wait_after_predecessor())
    bind(
        original.score,
        RecoveryPacer=Pacer,
        authenticated_passes=authenticated_passes,
        checked_scores=checked_scores,
    )(path, output, plan)
    # Keep the original decision bytes. A separate receipt makes the scope explicit.
    write(
        output / "diagnostic-scope.json",
        {
            "post_hoc_diagnostic_only": True,
            "decision": artifact(output / "decision.json"),
            "optimizer_updates": 0,
            "trained_model_win_established": False,
            "promotion_authorized": False,
        },
    )


def run(path, output, mode):
    return bind(
        original.run, checked=checked, checked_scores=checked_scores, infer=infer, score=score
    )(path, output, mode, BASELINE if mode == "inference" else None)


def pipeline(path, output):
    from probe_roi_seen_translation import freeze as freeze_scoring

    checked(path)
    if output.exists() or not output.absolute().is_relative_to(ROOT / "checkpoints"):
        raise ValueError("public112 diagnostic needs a fresh owned output")
    original.storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        score_path = path.parent.with_name(path.parent.name + "-scores") / "plan.json"
        for mode in ("inference", "scores"):
            stage_plan = path
            if mode == "scores":
                freeze_scoring(
                    score_path, score_specification(path, output / "inference/receipt.json")
                )
                stage_plan = score_path
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--plan",
                str(stage_plan),
                "--output",
                str(output / mode),
                "--mode",
                mode,
            ]
            if subprocess.run(command, check=False).returncode:
                raise RuntimeError("public112 diagnostic child failed; no retry or next stage")
            original.terminal(
                output / mode / ("receipt.json" if mode == "inference" else "decision.json")
            )
        checked(path)
        write(
            output / "complete.json",
            {
                "complete": True,
                "plan": artifact(path),
                "inference": artifact(output / "inference/receipt.json"),
                "decision": artifact(output / "scores/decision.json"),
                "optimizer_updates": 0,
                "post_hoc_diagnostic_only": True,
                "promotion_authorized": False,
            },
        )
        code = 0
    except BaseException as error:
        write(output / "failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    original.torch.set_num_threads(2)
    original.torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=lambda p: Path(p).absolute())
    parser.add_argument("--plan", type=lambda p: Path(p).absolute())
    parser.add_argument("--output", type=lambda p: Path(p).absolute())
    parser.add_argument("--mode", choices=("inference", "scores"))
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    path = args.plan or args.freeze
    if path and args.output and args.pipeline:
        pipeline(path, args.output)
    elif path and args.output and args.mode:
        run(path, args.output, args.mode)
    elif not args.freeze:
        parser.error("freeze, or execute the complete public112 diagnostic")


if __name__ == "__main__":
    main()
