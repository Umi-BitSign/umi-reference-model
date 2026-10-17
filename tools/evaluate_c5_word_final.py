"""Serial fixed-final word/sentence evaluation after the independent terminal audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import torch
from benchmark_fleurs_dev import sha256
from c5_mixed_risk_evidence import previous_evaluation, require_terminal
from c5_mixed_semantic_model import word_initialization
from c5_word_evaluation_io import storage
from launch_terra_visual_job import identity, now, write
from probe_c5_mixed_semantic import wait_for_predecessor
from terra_training_recovery import artifact

ROOT = Path(__file__).resolve().parents[1]
TRAINING_PLAN = ROOT / "runs/c5-word-semantic-training-v1/plan.json"
TRAINING_RESULT = ROOT / "checkpoints/c5-word-semantic-training-v1/result.json"
TERMINAL_AUDIT = TRAINING_PLAN.with_name("terminal-audit.json")
TERMINAL_AUDIT_SOURCE_SHA256 = "2c8e775919b1d21430764afea406230edc1909610cb7d34fdf6a28a6371fe678"
CONTROL_SOURCES = (
    "evaluate_c5_word_final.py",
    "c5_mixed_risk_evidence.py",
    "c5_mixed_semantic_model.py",
    "probe_c5_mixed_semantic.py",
    "c5_word_final.py",
    "c5_word_evaluation_io.py",
)


def control_sources():
    return {f"tools/{name}": sha256(ROOT / "tools" / name) for name in CONTROL_SOURCES}


def training_boundary(result_path):
    record = json.loads(result_path.with_name("exit.json").read_text())
    exited = datetime.fromisoformat(record["observed_at"])
    if exited.tzinfo is None:
        raise ValueError("training-to-evaluation cooldown needs a timezone-aware exit")
    return {"API_start_not_before": (exited + timedelta(seconds=60)).isoformat()}


def prerequisites(training_path, result_path, audit_path):
    # Authenticate all856 exposures and their updates/checkpoints before any child or plan exists.
    initialization = word_initialization(training_path, result_path)
    require_terminal(result_path)
    audit = json.loads(audit_path.read_text())
    result = json.loads(result_path.read_text())
    if (
        audit.get("schema") != "umi-studio-complete-word-training-terminal-audit/1"
        or audit.get("complete") is not True
        or audit.get("same_worker_exit0") is not True
        or audit.get("all_original_350_tensor_checkpoints_and_actual_reward_evidence_validated")
        is not True
        or audit.get("all_pinned_sources_and_model_assets_match") is not True
        or audit.get("raw_request_response_pairings_revalidated_by_checked_final") is not True
        or audit.get("training_examples") != 856
        or audit.get("optimizer_updates") != result["optimizer_updates"]
        or audit.get("visual_initialization_sha256") != result["visual_initialization_sha256"]
        or audit.get("visual_final_sha256") != initialization["initial_visual_state_sha256"]
        or audit.get("plan") != initialization["word_training_plan"]
        or audit.get("result") != initialization["word_training_result"]
        or audit.get("training_evidence") != initialization["word_training_evidence"]
        or audit.get("worker") != artifact(result_path.with_name("worker.json"))
        or audit.get("exit") != artifact(result_path.with_name("exit.json"))
        or audit.get("new_Terra_requests_during_audit") != 0
        or audit.get("optimizer_updates_during_audit") != 0
        or audit.get("held_out_semantic_quality_evaluated") is not False
        or audit.get("promotion_authorized") is not False
    ):
        raise ValueError("final evaluation requires complete independently audited training")
    source = ROOT / audit["audit_source"]["path"]
    if (
        artifact(source) != audit["audit_source"]
        or audit["audit_source"]["sha256"] != TERMINAL_AUDIT_SOURCE_SHA256
    ):
        raise ValueError("final evaluation terminal-audit program differs from reviewed source")
    if len(audit["checkpoints"]) != 4 or len(audit["epochs"]) != 4:
        raise ValueError("final evaluation terminal audit omitted a complete epoch")
    updates = 0
    for index, (epoch, checkpoint, visual) in enumerate(
        zip(audit["epochs"], audit["checkpoints"], result["checkpoints"], strict=True)
    ):
        updates += epoch["optimizer_updates"]
        if (
            epoch.get("epoch") != index
            or epoch.get("unique_clips") != 214
            or epoch.get("exposures") != 214
            or epoch.get("tail_clips") != 2
            or checkpoint.get("training_examples") != (index + 1) * 214
            or checkpoint.get("parameter_tensors") != 350
            or checkpoint.get("all_Adam_steps") != updates
            or checkpoint.get("visual") != visual
        ):
            raise ValueError("final evaluation terminal audit changed complete350-tensor epochs")
        commit_path = ROOT / checkpoint["commit"]["path"]
        commit = json.loads(commit_path.read_text())
        if (
            artifact(commit_path) != checkpoint["commit"]
            or checkpoint["all_Adam_steps"] != commit["optimizer_updates"]
            or checkpoint["optimizer"] != commit["optimizer"]
            or artifact(ROOT / checkpoint["optimizer"]["path"]) != checkpoint["optimizer"]
        ):
            raise ValueError("final evaluation terminal Adam audit lost its actual commit binding")
    if updates != result["optimizer_updates"]:
        raise ValueError("final evaluation terminal epoch update counts differ from final training")
    return initialization


def locations():
    return {
        "word": (
            ROOT / "runs/c5-word-semantic-evaluation-v1/plan.json",
            ROOT / "checkpoints/c5-word-semantic-evaluation-v1",
            "evaluate_c5_words.py",
        ),
        "sentence": (
            ROOT / "runs/c5-word-sentence-evaluation-v1/plan.json",
            ROOT / "checkpoints/c5-word-sentence-evaluation-v1",
            "evaluate_c5_word_sentences.py",
        ),
    }


def pipeline(training_path, result_path, audit_path, output):
    training_path, result_path, audit_path, output = (
        path.resolve() for path in (training_path, result_path, audit_path, output)
    )
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("final evaluation coordinator needs a fresh owned output")
    initialization = prerequisites(training_path, result_path, audit_path)
    paths = locations()
    if any(
        plan.parent.exists()
        or plan.parent.with_name(plan.parent.name + "-scores").exists()
        or child_output.exists()
        for plan, child_output, _ in paths.values()
    ):
        raise ValueError("final evaluation refuses to replay existing plans or child outputs")
    first_request_boundary = training_boundary(result_path)
    sources = control_sources()
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        for name, digest in sources.items():
            source = ROOT / name
            target = output / "control-sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
            if sha256(target) != digest:
                raise ValueError("final evaluation control source changed before archival")
        write(
            output / "handoff.json",
            {
                "coordinator_source_sha256": sha256(Path(__file__)),
                "source_sha256": sources,
                "initialization": initialization,
                "terminal_audit": artifact(audit_path),
                "complete_evaluation_order": ["word96", "sentence34"],
                "stop_on_failed_calibration": True,
                "semantic_decline_blocks_later_training_without_omitting_the_sentence_gate": True,
                "cross_pipeline_minimum_start_gap_seconds": 60,
                "automatic_retries": 0,
                "automatic_training_continuation": False,
                "promotion_authorized": False,
            },
        )
        evaluated = {}
        for kind, (plan_path, child_output, script) in paths.items():
            if control_sources() != sources:
                raise ValueError("final evaluation control sources changed before dependent work")
            storage(output=output)
            predecessor = evaluated["word"] if evaluated else first_request_boundary
            spacing_path = (
                "word-to-sentence-spacing.json" if evaluated else "training-to-word-spacing.json"
            )
            write(
                output / spacing_path,
                {
                    "not_before": predecessor["API_start_not_before"],
                    "ready_at": wait_for_predecessor(predecessor),
                },
            )
            command = [
                sys.executable,
                str(ROOT / "tools" / script),
                "--freeze",
                str(plan_path),
                "--training-plan",
                str(training_path),
                "--training-result",
                str(result_path),
                "--pipeline",
                "--output",
                str(child_output),
            ]
            child = subprocess.run(command, check=False)
            if child.returncode:
                raise RuntimeError(f"final {kind} evaluation stopped; no retry or next pipeline")
            evaluated[kind] = previous_evaluation(
                child_output / "complete.json", kind, initialization, evaluation_only=True
            )
            write(output / f"{kind}-authenticated.json", evaluated[kind])
        if prerequisites(training_path, result_path, audit_path) != initialization:
            raise ValueError("fixed-final training evidence changed during evaluation")
        # Reauthenticate both complete results before committing the parent completion.
        for kind, (_, child_output, _) in paths.items():
            if (
                previous_evaluation(
                    child_output / "complete.json", kind, initialization, evaluation_only=True
                )
                != evaluated[kind]
            ):
                raise ValueError("final paired evaluation evidence changed during handoff")
        if control_sources() != sources or any(
            sha256(output / "control-sources" / name) != digest for name, digest in sources.items()
        ):
            raise ValueError("final evaluation control sources changed during handoff")
        storage(output=output)
        write(
            output / "complete.json",
            {
                "complete": True,
                "handoff": artifact(output / "handoff.json"),
                "API_spacing": [
                    artifact(output / "training-to-word-spacing.json"),
                    artifact(output / "word-to-sentence-spacing.json"),
                ],
                "initialization": initialization,
                "terminal_audit": artifact(audit_path),
                "evaluations": evaluated,
                "optimizer_updates": 0,
                "automatic_retries": 0,
                "automatic_training_continuation": False,
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
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, default=TRAINING_PLAN)
    parser.add_argument("--training-result", type=Path, default=TRAINING_RESULT)
    parser.add_argument("--terminal-audit", type=Path, default=TERMINAL_AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pipeline(args.training_plan, args.training_result, args.terminal_audit, args.output)


if __name__ == "__main__":
    main()
