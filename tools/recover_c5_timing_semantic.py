"""One explicit timing-proof grading recovery after an unanswered terminal HTTP502.

Preserves the failed attempt, complete original native identity evidence and exact
candidate pools. Records the one orchestrated request replay; the API client has
no transport retry. A second error ends this recovery without another attempt.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import probe_c5_timing_semantic as original
from c5_timing_feature_data import (
    ROOT,
    TimingFeatureDataset,
    artifact,
    bound,
    owned,
    read,
    require,
    sha,
)
from launch_terra_visual_job import identity, now, write

UTC = timezone.utc  # noqa: UP017 - pinned runtime is Python3.10

SCHEMA = "umi-studio-explicit-timing-proof-HTTP502-recovery-plan/1"
PREDECESSOR_PLAN = ROOT / "runs/c5-timing-semantic-proof-v1/plan.json"
PREDECESSOR_OUTPUT = ROOT / "checkpoints/c5-timing-semantic-proof-v1"
PREDECESSOR_POLL = ROOT / "runs/c5-timing-semantic-proof-launch-v1/terminal-tool-poll.json"
REVIEW = ROOT / "runs/c5-timing-semantic-proof-launch-v1/terminal-review.json"
REVIEW_SHA256 = "a2e51532163b0f73f0e59b55e80b5a42134e607713ea92a10ca273d8bfda48a3"


def verify_review(evidence):
    require(
        sha(REVIEW) == REVIEW_SHA256 and read(REVIEW)["evidence"] == evidence,
        "timing recovery predecessor differs from its actual terminal review",
    )
    return artifact(REVIEW)


def require_failure(failure):
    require(
        type(failure.get("http_status")) is int
        and failure["http_status"] == 502
        and failure.get("ambiguous_completion") is False
        and failure.get("automatic_retry_allowed") is False,
        "timing recovery accepts only explicit unanswered HTTP502, never ambiguous completion",
    )


def predecessor():
    from terra_polarity_judge import payload

    plan = original.checked(PREDECESSOR_PLAN)
    output = PREDECESSOR_OUTPUT
    worker, exited, poll = (
        read(output / "worker.json"),
        read(output / "exit.json"),
        read(PREDECESSOR_POLL),
    )
    require(
        worker["uid"] == os.getuid()
        and worker["pid"] == exited["pid"] == poll["actual_owned_pid"]
        and exited["exit_code"] == poll["tool_result"]["exit_code"] == 1
        and poll["tool"] == "write_stdin"
        and type(poll["requested_session_id"]) is int
        and poll["tool_result"].get("session_id") is None,
        "timing recovery requires actual same-owner terminal failure tool evidence",
    )
    argv = worker["argv"]
    require(
        Path(argv[1]).absolute() == ROOT / "tools/probe_c5_timing_semantic.py"
        and Path(argv[argv.index("--plan") + 1]).absolute() == PREDECESSOR_PLAN
        and Path(argv[argv.index("--output") + 1]).absolute() == output,
        "timing recovery predecessor changed its actual program or inputs",
    )
    live = subprocess.run(
        ["ps", "-p", str(worker["pid"]), "-o", "uid=,lstart="], capture_output=True, text=True
    )
    require(
        live.returncode != 0
        or live.stdout.strip().split(maxsplit=1)
        != [str(worker["uid"]), worker["ps_started_local"]],
        "timing recovery predecessor is still live",
    )
    exited_at = datetime.fromisoformat(exited["observed_at"])
    require(exited_at.tzinfo is not None, "timing recovery needs an actual timezone-aware exit")
    semantic = output / "semantic"
    paired = read(output / "comparison.json")
    requests = original.grading_requests(paired)
    require(
        read(semantic / "requests.json") == requests, "timing predecessor grading order changed"
    )
    sub = semantic / "request-000"
    require_failure(read(sub / "failure.json"))
    first = [paired["items"][i] for i in requests[0]["indices"]]
    started = read(sub / "started.json")
    require(
        read(sub / "request.json") == payload(first)
        and started["request_sha256"] == sha(sub / "request.json")
        and started["pairs"] == len(first)
        and started["model"] == "gpt-5.6-terra"
        and started["reasoning_effort"] == "medium"
        and started["automatic_retries"] == 0,
        "timing predecessor failed request lost exact model/prompt/candidate provenance",
    )
    expected = (
        {
            output / name
            for name in (
                "worker.json",
                "exit.json",
                "feature-identity.json",
                "pools.json",
                "comparison.json",
                "identity-00.json",
                "identity-01.json",
                "identity-02.json",
                "identity-03.json",
            )
        }
        | {semantic / "requests.json"}
        | {sub / name for name in ("request.json", "started.json", "failure.json")}
    )
    actual = list(output.rglob("*"))
    require(
        all(not p.is_symlink() for p in actual) and {p for p in actual if p.is_file()} == expected,
        "timing recovery refuses a response, score, later attempt or unbound predecessor output",
    )
    require(
        read(output / "feature-identity.json")
        == {"clips": 532, "bitwise_identity_including_signed_zero": True},
        "timing predecessor lacks complete actual initial feature identity",
    )
    for index, record in enumerate(plan["records"]):
        saved = read(output / f"identity-{index:02d}.json")
        require(
            saved["row"] == record["row"]
            and saved["memory_bitwise_equal"] is True
            and saved["teacher_forcing_logits_and_loss_bitwise_equal"] is True
            and saved["native_generation_tokens_equal"] is True
            and saved["generation"] == plan["model_plan"]["reference"]["generation"],
            "timing predecessor lacks complete actual native identity checks",
        )
    claim = read(PREDECESSOR_PLAN.with_name("execution-claim.json"))
    require(
        claim["plan"] == artifact(PREDECESSOR_PLAN)
        and claim["worker"] == artifact(output / "worker.json"),
        "timing predecessor lost its one actual execution claim",
    )
    evidence = {
        "predecessor_plan": artifact(PREDECESSOR_PLAN),
        "terminal_poll": artifact(PREDECESSOR_POLL),
        "evidence": [artifact(p) for p in sorted(expected)],
        "execution_claim": artifact(PREDECESSOR_PLAN.with_name("execution-claim.json")),
        "feature_identity": artifact(output / "feature-identity.json"),
        "native_identities": [artifact(output / f"identity-{i:02d}.json") for i in range(4)],
        "pools": artifact(output / "pools.json"),
        "comparison": artifact(output / "comparison.json"),
        "explicit_failed_API_attempts": 1,
        "successful_predecessor_scores": 0,
        "orchestrated_failed_request_replays_authorized": 1,
        "transport_automatic_retries": 0,
        "maximum_recovery_executions": 1,
        "API_start_not_before": (exited_at + timedelta(seconds=300)).isoformat(),
        "same_candidates_and_rubric_required": True,
        "failed_attempt_reclassified_as_success": False,
        "authorization": "standing user approval for indefinite model-training continuations",
    }
    evidence["terminal_review"] = verify_review(evidence)
    return plan, evidence


def specification():
    base, recovery = predecessor()
    return {
        **base,
        "schema": SCHEMA,
        "recovery": recovery,
        "source_sha256": {
            **base["source_sha256"],
            "tools/recover_c5_timing_semantic.py": sha(Path(__file__)),
        },
    }


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = read(path)
    require(plan == specification(), "timing recovery changed its actual failure or fixed proof")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(not path.parent.exists(), "timing recovery requires a fresh plan")
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))


def require_completion(plan, result, directory):
    require(
        plan["schema"] == SCHEMA
        and result["recovery"] == plan["recovery"]
        and result["orchestrated_failed_request_replays"] == 1
        and result["API_attempts_including_predecessor"] == result["new_Terra_requests"] + 1
        and not list(directory.rglob("failure.json")),
        "timing recovery result lost its explicit replay count or complete successful scope",
    )
    source = bound(result["plan"])
    parent = bound(plan["recovery"]["predecessor_plan"])
    claim = read(source.with_name("execution-claim.json"))
    require(
        claim == read(parent.with_name("grading-recovery-child-claim.json"))
        and claim["plan"] == result["plan"]
        and claim["worker"] == result["worker"]
        and claim["maximum_recovery_executions"] == 1
        and claim["orchestrated_failed_request_replays"] == 1,
        "timing recovery lost its one actual separately recorded execution claim",
    )


def qualify(path, output, plan):
    from c5_timing_semantic_model import load_student, require_model
    from terra_polarity_judge import Pacer

    dataset = TimingFeatureDataset(bound(plan["data_audit"]))
    values = original.selected_values(dataset, plan)
    model, tokenizer = load_student(plan["model_plan"])
    original.require_zero(model)
    pools = read(bound(plan["recovery"]["pools"]))
    original.validate_pools(values, pools, tokenizer)
    paired = original.paired_inputs(values, pools)
    require(
        paired == read(bound(plan["recovery"]["comparison"])),
        "timing recovery changed exact candidate pairs",
    )
    requests = original.grading_requests(paired)
    semantic = output / "semantic"
    semantic.mkdir()
    write(semantic / "requests.json", requests)
    pacer = Pacer()
    for index, request in enumerate(requests):
        original.guard(output)
        require(
            datetime.now(UTC) >= datetime.fromisoformat(plan["recovery"]["API_start_not_before"]),
            "timing recovery cannot query before its five-minute cooldown",
        )
        pacer.score(
            [paired["items"][i] for i in request["indices"]], semantic / f"request-{index:03d}"
        )
    judged, raw = original.authenticated_passes(paired, semantic)
    gradients = original.actual_gradients(model, values, pools, judged, output=output)
    require_model(model, plan["model_plan"])
    require(plan == checked(path), "timing recovery data or sources changed during execution")
    original.guard(output)
    result = {
        "schema": "umi-studio-native224-timing-semantic-qualification/1",
        "complete": True,
        "plan": artifact(path),
        "worker": artifact(output / "worker.json"),
        "feature_identity": plan["recovery"]["feature_identity"],
        "native_identities": plan["recovery"]["native_identities"],
        "pools": plan["recovery"]["pools"],
        "comparison": plan["recovery"]["comparison"],
        "grading": judged,
        "raw_grading_evidence": raw,
        "gradients": gradients,
        "complete_grading_passes": 2,
        "new_Terra_requests": len(requests),
        "original_model_values_unchanged": True,
        "optimizer_updates": 0,
        "CPU_threads": 2,
        "GPU_use": False,
        "training_only": True,
        "quality_gain_established": False,
        "promotion_authorized": False,
        "recovery": plan["recovery"],
        "orchestrated_failed_request_replays": 1,
        "API_attempts_including_predecessor": len(requests) + 1,
    }
    require_completion(plan, result, output)
    write(output / "qualification.json", result)


def run(path, output):
    import torch
    from prepare_c5_timing_features import require_environment
    from train_terra_c5_tail import gpu_lease

    require_environment()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan = checked(path)
    owned(output, "checkpoints")
    require(not output.exists(), "timing recovery needs a fresh output")
    require(
        datetime.now(UTC) >= datetime.fromisoformat(plan["recovery"]["API_start_not_before"]),
        "timing recovery cooldown has not elapsed",
    )
    original.guard(output)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        claim = {
            "plan": artifact(path),
            "worker": artifact(output / "worker.json"),
            "claimed_at": now(),
            "maximum_recovery_executions": 1,
            "orchestrated_failed_request_replays": 1,
        }
        write(PREDECESSOR_PLAN.with_name("grading-recovery-child-claim.json"), claim)
        write(path.with_name("execution-claim.json"), claim)
        with gpu_lease():
            qualify(path, output, plan)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "plan", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute())
    args = parser.parse_args()
    if args.freeze and not args.plan:
        freeze(args.freeze)
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output and not args.freeze:
        run(args.plan, args.output)
    else:
        parser.error("freeze one reviewed HTTP502 recovery, or run it once")


if __name__ == "__main__":
    main()
