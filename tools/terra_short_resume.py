"""Explicit short-pilot recovery; scientific sources and completed sample order stay fixed."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone

from prepare_terra_short_curriculum import ROOT
from terra_short_curriculum_learning import RECIPE, JudgePacer, order
from terra_training_recovery import artifact, verify_snapshot

CONTROL_SOURCES = {"tools/train_terra_short_curriculum.py", "tools/terra_short_resume.py"}


def compatible(previous, current):
    excluded = {"source_sha256", "resume"}
    if {k: v for k, v in previous.items() if k not in excluded} != {
        k: v for k, v in current.items() if k not in excluded
    }:
        raise ValueError("short recovery cannot change data, loss, sampler, generation or gates")
    a, b = previous["source_sha256"], current["source_sha256"]
    changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if not changed <= CONTROL_SOURCES:
        raise ValueError("short recovery changed a scientific component")


def worker_still_present(worker):
    # Check only the saved worker's UID/start, not a reused unrelated PID's command.
    values = [
        subprocess.run(
            ["ps", "-p", str(worker["pid"]), "-o", field + "="],
            capture_output=True,
            text=True,
        ).stdout.strip()
        for field in ("uid", "lstart")
    ]
    return values == [str(worker["uid"]), worker["ps_started_local"]]


def cli_path(argv, name, *, root):
    try:
        return (root / argv[argv.index(name) + 1]).resolve()
    except (ValueError, IndexError, TypeError) as error:
        raise ValueError("short recovery worker command is incomplete") from error


def make_resume(previous_path, checkpoint, current, *, root=ROOT):
    from safetensors import safe_open

    previous_path, checkpoint = previous_path.resolve(), checkpoint.resolve()
    previous = json.loads(previous_path.read_text())
    compatible(previous, current)
    verify_snapshot(previous_path, previous)
    source_plan = artifact(previous_path, root=root)
    policy = artifact(checkpoint, root=root)
    output = checkpoint.parent
    commit_path = checkpoint.with_suffix(".json")
    commit = json.loads(commit_path.read_text())
    optimizer_path = checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")
    optimizer = artifact(optimizer_path, root=root)
    examples, updates = commit["training_examples"], commit["optimizer_updates"]
    if (
        commit.get("complete") is not True
        or commit.get("visual") != policy
        or commit.get("optimizer") != optimizer
        or type(examples) is not int
        or type(updates) is not int
        or not 0 < examples < current["planned_training_examples"]
        or updates < 1
    ):
        raise ValueError("short recovery checkpoint pair/counters are incomplete")
    for path in (checkpoint, optimizer_path):
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            metadata = saved.metadata()
        if metadata.get("plan_sha256") != source_plan["sha256"] or metadata.get("update") != str(
            examples
        ):
            raise ValueError("short recovery checkpoint belongs to a different plan or offset")
        if path == checkpoint and (
            metadata.get("optimizer_updates") != str(updates)
            or metadata.get("fixed_winner_checkpoint")
            != current["reference"]["translator_checkpoint_sha256"]
        ):
            raise ValueError("short recovery policy counters or reference differ")
    names = {
        k: output / v
        for k, v in {
            "source_run": "run.json",
            "source_steps": "steps.jsonl",
            "source_pools": "candidate-pools.jsonl",
            "source_worker": "worker.json",
            "source_terminal": "exit.json",
            "source_stop": "preupdate-stop.json",
        }.items()
    }
    run, worker, terminal, stop = [
        json.loads(names[k].read_text())
        for k in ("source_run", "source_worker", "source_terminal", "source_stop")
    ]
    if (
        run.get("plan_sha256") != source_plan["sha256"]
        or terminal.get("pid") != worker.get("pid")
        or terminal.get("exit_code") != 1
        or worker_still_present(worker)
        or cli_path(worker["argv"], "--plan", root=root) != previous_path
        or cli_path(worker["argv"], "--output-dir", root=root) != output
        or stop.get("stage") != "terra_scoring"
        or stop.get("training_examples") != examples
        or stop.get("optimizer_updates") != updates
        or stop.get("failed_batch") != updates + 1
        or stop.get("failed_batch_optimizer_applied") is not False
        or stop.get("automatic_retries") != 0
    ):
        raise ValueError("short recovery lacks a matching terminal pre-update scoring stop")
    failure_path = output / "terra" / f"batch-{updates + 1:06d}" / "failure.json"
    failure = json.loads(failure_path.read_text())
    if failure.get("ambiguous_completion") is not False or failure.get("http_status") not in (
        502,
        503,
    ):
        raise ValueError("short recovery cannot replay an ambiguous or unrelated failure")
    error = json.loads(failure.get("message", "{}"))
    if error.get("error", {}).get("code") != "server_is_overloaded":
        raise ValueError("short recovery needs the recorded explicit overload response")
    if run.get("resume") != previous.get("resume"):
        raise ValueError("short recovery source run has a different restored prefix")
    base = run.get("resume") or {}
    if base:
        paths(previous, root=root)
    base_examples, base_updates = base.get("training_examples", 0), base.get("optimizer_updates", 0)
    steps = [json.loads(s) for s in names["source_steps"].read_text().splitlines()]
    pools = [json.loads(s) for s in names["source_pools"].read_text().splitlines()]
    data = json.loads((root / current["data"]["path"]).read_text())
    ordered = order(data["records"])
    offset = base_examples
    if len(steps) != updates - base_updates or len(pools) != len(steps):
        raise ValueError("short recovery has uncommitted or missing optimizer/sample records")
    for i, (step, pool) in enumerate(zip(steps, pools, strict=True), base_updates + 1):
        identities = pool["identities"]
        boundary = next(n for n in current["checks_after_examples"] if n > offset)
        expected_count = min(RECIPE["batch_clips"], boundary - offset)
        expected = [
            {k: p["raw"][k] for k in ("source", "example_id")}
            for p in ordered[offset : offset + len(identities)]
        ]
        offset += len(identities)
        if (
            step["optimizer_update"] != i
            or pool["optimizer_update"] != i
            or step["training_examples"] != offset
            or identities != expected
            or len(identities) != expected_count
        ):
            raise ValueError("short recovery would replay or skip completed sampler identities")
    if offset != examples:
        raise ValueError("short recovery sampler offset differs from the checkpoint")
    decisions_path = output / "semantic-decisions.jsonl"
    decisions = []
    if decisions_path.exists():
        decisions = [json.loads(s) for s in decisions_path.read_text().splitlines()]
        if any(
            d.get("complete_scoring") is not True
            or d.get("calibration_passed") is not True
            or d["training_examples"] not in current["checks_after_examples"]
            or d["training_examples"] > examples
            or d["stop_for_semantic_decline"]
            or d["stop_for_failed_calibration"]
            or d["control_stage_mastered"]
            or d["candidate_stage_mastered"]
            for d in decisions
        ):
            raise ValueError("short recovery cannot override a semantic stop/mastery decision")
    if any(
        n <= examples and not any(d["training_examples"] == n for d in decisions)
        for n in current["checks_after_examples"]
    ):
        raise ValueError("short recovery must finish a pending semantic check before training")
    when = datetime.fromisoformat(terminal["observed_at"].replace("Z", "+00:00"))
    if when.tzinfo is None:
        raise ValueError("short recovery terminal timestamp requires a timezone")
    result = {
        "source_plan": source_plan,
        "visual": policy,
        "optimizer": optimizer,
        "checkpoint_commit": artifact(commit_path, root=root),
        **{k: artifact(v, root=root) for k, v in names.items()},
        "source_failure": artifact(failure_path, root=root),
        "source_failed_request": artifact(failure_path.with_name("request.json"), root=root),
        "training_examples": examples,
        "optimizer_updates": updates,
        "unsaved_examples_to_recompute": 0,
        "not_before": (when + timedelta(seconds=1800)).isoformat(),
        "minimum_request_gap_seconds": 60,
        "automatic_retries": 0,
    }
    if decisions_path.exists():
        result["source_decisions"] = artifact(decisions_path, root=root)
    return result


def paths(plan, *, root=ROOT):
    result = {}
    for key, value in plan["resume"].items():
        if isinstance(value, dict):
            path = root / value["path"]
            if artifact(path, root=root) != value:
                raise ValueError("short recovery artifact changed")
            result[key] = path
    return result


def check_cooldown(plan):
    if datetime.now(timezone.utc) < datetime.fromisoformat(plan["resume"]["not_before"]):  # noqa: UP017
        raise RuntimeError("short recovery must finish the explicit thirty-minute cooldown")


class RecoveryPacer(JudgePacer):
    def score(self, rows, output):
        if self.previous_start is not None:
            remaining = 60 - (self.clock() - self.previous_start)
            if remaining > 0:
                self.sleep(remaining)
        return super().score(rows, output)
