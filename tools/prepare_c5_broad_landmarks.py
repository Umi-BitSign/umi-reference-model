"""Bind reserved confirmation media to offline landmarks after training exits."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections import Counter
from pathlib import Path
from types import FunctionType

import prepare_terra_word_landmarks as original
import fetch_c5_broad_how2sign as media_admission
from fetch_c5_broad_how2sign import ROOT, admit, artifact, read, require, sha256
from launch_terra_visual_job import write as original_write
from prepare_asl_stem_wiki_landmarks import DOCKER_SOCKET, IMAGE, docker_image_id

MEDIA = ROOT / "runs/c5-broad-confirmation-media-qualified-v1/plan.json"
TRAINING = ROOT / "runs/c5-polarity-word-http502-recovery-v1/plan.json"
TRAINING_OUTPUT = ROOT / "checkpoints/c5-polarity-word-http502-recovery-v1"
POLICY = {
    **original.POLICY,
    "CPU_threads": 1,
    "physical_free_floor_bytes": 84 * 1024**3,
    "maximum_batch_clips": 16,
    "training_allowed": False,
    "training_session_terminal_receipt_required": 31973,
    "complete_training_audit_required": True,
}


def bind(function, **overrides):
    result = FunctionType(function.__code__, {**function.__globals__, **overrides},
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def specification():
    p = read(MEDIA)
    reservation_path = ROOT / p["reservation"]["path"]
    reservation, _ = admit(reservation_path)
    review = read(MEDIA.with_name("independent-review.json"))
    require(
        p["schema"] == "umi-studio-broad-semantic-confirmation-qualified-media/1"
        and review["plan"] == artifact(MEDIA)
        and p["reservation"] == artifact(reservation_path)
        and p["confirmation_consumed"] is False and p["candidate_bound"] is False
        and p["model_predictions_inspected"] is False and p["training_allowed"] is False
        and p["reserved_source_groups"] == reservation["reserved_source_groups"]
        and p["candidate_policy"] == reservation["candidate_policy"]
        and p["scoring_policy"] == reservation["scoring_policy"],
        "landmarks require reviewed unconsumed confirmation media and unchanged gates",
    )
    records = []
    for example in p["examples"]:
        receipt = ROOT / example["source_receipt"]["path"]
        require(artifact(receipt) == example["source_receipt"], "media receipt changed")
        saved = read(receipt)
        require(
            original.media_artifact(ROOT / example["video"]["path"]) == example["video"]
            and saved["video"] == example["video"] and saved["frames"] == example["frames"]
            and example["fps"] == 25 and example["training_allowed"] is False
            and example["source_group"] in p["reserved_source_groups"],
            "landmark clip identity, timeline or role changed",
        )
        stream = saved["probe"]["streams"][0]
        records.append({
            "original": {"row": {"example_id": example["clip_id"], "role": "confirmation",
                                   "training_allowed": False}},
            "example": example, "clip": example["video"], "normalized_receipt": example["source_receipt"],
            "normalized_probe": {"width": stream["width"], "height": stream["height"],
                                 "decoded_frames": example["frames"]},
            "ROI_lineage_training_allowed": False,
        })
    require(len(records) == 48 and Counter(r["example"]["family"] for r in records)
            == {"how2sign": 24, "openasl": 24}, "landmarks changed confirmation coverage")
    sources = dict(read(TRAINING)["source_sha256"])
    for name in (*media_admission.SOURCES, "prepare_c5_broad_landmarks.py"):
        sources["tools/" + name] = sha256(ROOT / "tools" / name)
    require(all(sha256(ROOT / name) == digest for name, digest in sources.items()),
            "frozen training/landmark source changed")
    return {
        "schema": "umi-studio-broad-confirmation-landmarks/1", "media": artifact(MEDIA),
        "media_review": artifact(MEDIA.with_name("independent-review.json")),
        "training_plan": artifact(TRAINING), "records": records, "policy": POLICY,
        "source_sha256": sources, "docker_image_id": docker_image_id(DOCKER_SOCKET),
        "dockerfile_sha256": sha256(original.DOCKERFILE),
        "model_assets": {str(path.relative_to(ROOT)): sha256(path) for path in (
            original.ROOT / "artifacts/baseline-6fe8df59/models" / name
            for name in ("face_landmarker_v2_with_blendshapes.task", "hand_landmarker.task"))},
        "training_allowed": False, "confirmation_consumed": False, "candidate_bound": False,
    }


def checked(path):
    plan = read(path)
    require(plan == specification(), "broad landmark plan, media or source changed")
    original.verify_snapshot(path, plan)
    return plan


freeze = bind(original.freeze, specification=specification)


def container_command(**kwargs):
    command = original.container_command(**kwargs)
    command[command.index("--cpus") + 1] = "1"
    position = command.index(IMAGE)
    command[position:position] = ["--pull", "never", "--env", "OMP_NUM_THREADS=1",
                                 "--env", "OPENBLAS_NUM_THREADS=1",
                                 "--env", "PYTHONDONTWRITEBYTECODE=1"]
    return command


def write(path, value):
    if value.get("schema") == "umi-studio-word-landmark-receipt/1":
        value = {**value, "schema": "umi-studio-broad-confirmation-landmark-receipt/1",
                 "training_allowed": False, "confirmation_consumed": False}
    original_write(path, value)


def admit_release(terminal_poll, terminal_audit):
    require(all(p.absolute() == p.resolve() and p.resolve().is_relative_to(ROOT / "runs")
                and p.is_file() for p in (terminal_poll, terminal_audit)),
            "training release receipts must be canonical owned run files")
    poll = read(terminal_poll)
    result = poll["tool_result"]
    require(poll.get("tool") == "write_stdin" and poll.get("requested_session_id") == 31973
            and type(result.get("exit_code")) is int and result["exit_code"] == 0
            and result.get("session_id") is None, "training session must actually exit0 first")
    audit, plan = read(terminal_audit), read(TRAINING)
    require(
        audit.get("schema") == "umi-studio-complete-word-training-terminal-audit/1"
        and audit.get("complete") is True and audit.get("same_worker_exit0") is True
        and audit.get("training_examples") == 856
        and audit.get("all_original_350_tensor_checkpoints_and_actual_reward_evidence_validated") is True
        and audit.get("all_pinned_sources_and_model_assets_match") is True
        and audit.get("raw_request_response_pairings_revalidated_by_checked_final") is True
        and audit.get("plan") == artifact(TRAINING)
        and audit.get("audit_source") == plan["lineage_audit_source"],
        "landmarks require the completed independent training audit",
    )
    for key in ("audit_source", "plan", "result", "exit", "worker"):
        descriptor = audit[key]
        require(artifact(ROOT / descriptor["path"]) == descriptor, "training release evidence changed")
    require(audit["result"] == artifact(TRAINING_OUTPUT / "result.json")
            and audit["exit"] == artifact(TRAINING_OUTPUT / "exit.json")
            and audit["worker"] == artifact(TRAINING_OUTPUT / "worker.json")
            and read(TRAINING_OUTPUT / "exit.json")["exit_code"] == 0,
            "release did not bind the actual completed training worker")
    worker = read(TRAINING_OUTPUT / "worker.json")
    require(worker["uid"] == os.getuid(), "training worker ownership changed")
    live = subprocess.run(["ps", "-p", str(worker["pid"]), "-o", "pid="], capture_output=True, text=True)
    require(live.returncode == 1 and not live.stdout.strip(), "training worker is still present")
    return {"terminal_poll": artifact(terminal_poll), "terminal_audit": artifact(terminal_audit)}


def prepare(path, data, receipts, job, limit, terminal_poll, terminal_audit):
    release = admit_release(terminal_poll, terminal_audit)
    require(not job.exists(), "broad landmark execution needs a fresh job")
    # The inherited worker records exact command, PID, container result and each
    # complete sidecar. Existing accepted sidecars are validated before reuse.
    def bound_write(target, value):
        write(target, value)
        if target == job.resolve() / "worker.json":
            write(job.resolve() / "training-release.json", release)

    bound = bind(original.prepare, checked=checked, POLICY=POLICY, write=bound_write,
                 container_command=container_command, guard=bind(original.guard, POLICY=POLICY))
    bound(path, data, receipts, job, limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "plan", "data", "receipts", "job", "terminal-poll", "terminal-audit"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif all((args.plan, args.data, args.receipts, args.job, args.terminal_poll, args.terminal_audit)):
        prepare(args.plan, args.data, args.receipts, args.job, args.limit,
                args.terminal_poll, args.terminal_audit)
    else:
        parser.error("freeze a plan, or provide execution paths and actual training release receipts")


if __name__ == "__main__":
    main()
