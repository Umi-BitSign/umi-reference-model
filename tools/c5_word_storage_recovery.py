"""A proposed one-time, explicitly authorized rollback from a storage stop.

This module never creates authorization records. A local snapshot-deletion
approval cannot authorize training. The original stopped run, its discarded
accepted work, and its consumed execution claim remain immutable evidence.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import c5_word_resume as previous
import train_c5_polarity_words as training
from benchmark_fleurs_dev import sha256
from c5_word_epoch_audit import optimizer_inventory
from c5_word_training import storage
from launch_terra_visual_job import now, write
from terra_polarity_judge import bind
from terra_training_recovery import artifact

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "runs/c5-polarity-storage-recovery-preparation-v1/proposal.json"
PROPOSAL_SHA = "4d95aba8918d574b69f1d9cd43a5eab1db9df9a83e0c2fe91a72f1f4a4095698"
SCOPE = "one-polarity-word-storage-recovery-from-428"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def checked_authorization(path):
    """Authenticate a recorded explicit user approval before any model loading."""
    path = Path(path).resolve()
    require(
        path.is_relative_to(ROOT / "runs") and path.is_file(),
        "explicit storage-recovery authorization is required",
    )
    require(sha256(PROPOSAL) == PROPOSAL_SHA, "reviewed rollback proposal changed")
    proposal = read(PROPOSAL)
    approval = read(path)
    require(
        approval.get("scope") == SCOPE
        and approval.get("approved") is True
        and approval.get("source") == "explicit_user_message"
        and isinstance(approval.get("user_message"), str)
        and approval["user_message"].strip()
        and approval.get("proposal") == artifact(PROPOSAL)
        and approval.get("source_checkpoint") == proposal["visual"]
        and type(approval.get("manual_continuation_limit")) is int
        and approval["manual_continuation_limit"] == 1
        and type(approval.get("automatic_retries")) is int
        and approval["automatic_retries"] == 0
        and type(approval.get("automatic_further_continuations")) is int
        and approval["automatic_further_continuations"] == 0,
        "training needs its own explicit one-time rollback approval",
    )
    received = datetime.fromisoformat(approval["received_at"].replace("Z", "+00:00"))
    require(received.tzinfo is not None, "recorded approval needs a timezone-aware timestamp")
    require(
        proposal["inherited_exposures"] == 428
        and proposal["rolled_back_accepted_exposures"] == 104
        and proposal["actual_total_lineage_exposures_if_complete"] == 960,
        "approval lost truthful rollback accounting",
    )
    return artifact(path), proposal


def rollback_prefix(plan, steps, committed_examples):
    """Select a complete epoch prefix; later accepted steps stay outside it."""
    actual_examples, actual_updates = previous.prefix_state(plan, steps)
    ends = [
        index + 1
        for index, step in enumerate(steps)
        if step["training_examples"] == committed_examples
    ]
    require(len(ends) == 1, "checkpoint does not identify one accepted prefix")
    cutoff = ends[0]
    require(
        cutoff < len(steps)
        and cutoff < len(plan["batches"])
        and plan["batches"][cutoff - 1]["epoch"] != plan["batches"][cutoff]["epoch"],
        "rollback must use an earlier complete epoch boundary",
    )
    inherited_examples, inherited_updates = previous.prefix_state(plan, steps[:cutoff])
    return {
        "training_examples": inherited_examples,
        "optimizer_updates": inherited_updates,
        "completed_batches": cutoff,
        "actual_source_training_examples": actual_examples,
        "actual_source_optimizer_updates": actual_updates,
        "rolled_back_accepted_exposures": actual_examples - inherited_examples,
        "rolled_back_optimizer_updates": actual_updates - inherited_updates,
    }


def make_binding(source_path, checkpoint, audit_path, authorization_path):
    authorization, proposal = checked_authorization(authorization_path)
    source_path, checkpoint, audit_path = (
        Path(path).resolve() for path in (source_path, checkpoint, audit_path)
    )
    require(
        artifact(source_path) == proposal["source_plan"]
        and artifact(checkpoint) == proposal["visual"]
        and artifact(audit_path) == proposal["terminal_failure_audit"],
        "storage recovery cannot fork another source or select another checkpoint",
    )
    source = training.checked(source_path)
    require(
        "resume" not in source and source["manual_continuations_authorized"] == 0,
        "storage recovery cannot become a continuation chain",
    )
    audit = read(audit_path)
    require(
        audit["schema"] == "umi-studio-polarity-word-storage-stop-audit/1"
        and audit["audit_complete"] is True
        and audit["complete"] is False
        and audit["same_original_session_exit1_authenticated"] is True
        and audit["plan"] == artifact(source_path),
        "recovery requires the exact authenticated storage stop",
    )
    for descriptor in [
        *[
            audit[key]
            for key in (
                "observer_source",
                "session_poll",
                "previous_observation",
                "worker",
                "run",
                "execution_claim",
                "proof_claim",
                "completed_steps_prefix",
            )
        ],
        *audit["actual_raw_training_evidence"],
    ]:
        require(
            artifact(ROOT / descriptor["path"]) == descriptor, "audited source evidence changed"
        )
    directory = checkpoint.parent
    stop, exited, worker, run = [
        read(directory / name) for name in ("stopped.json", "exit.json", "worker.json", "run.json")
    ]
    require(
        stop["error"] == "word training exceeded its4GiB reserve or unchanged80GiB floor"
        and exited["exit_code"] == 1
        and exited["pid"] == worker["pid"]
        and stop["automatic_continuations"] == stop["automatic_retries"] == 0
        and stop["recovery"] is None
        and stop["recovery_is_committed_prefix"] is False
        and stop["uncommitted_optimizer_mutation_possible"] is False
        and not (directory / "result.json").exists(),
        "source is not the reviewed stopped, nonrecovering worker",
    )
    status = audit["owned_process_terminal_status"]
    require(
        status["original_pid_absent"] and status["same_session_exit_code"] == 1,
        "source terminal identity changed",
    )
    for key in ("exit", "stopped", "failure"):
        require(
            artifact(directory / (key + ".json")) == status[key], "source terminal receipt changed"
        )
    raw = (directory / "steps.jsonl").read_bytes()
    require(
        raw == (ROOT / audit["completed_steps_prefix"]["path"]).read_bytes()
        and raw.endswith(b"\n"),
        "source accepted ledger changed",
    )
    steps = [json.loads(line) for line in raw.splitlines()]
    commit_path = checkpoint.with_suffix(".json")
    commit = read(commit_path)
    prefix = rollback_prefix(source, steps, commit["training_examples"])
    require(
        prefix
        == {
            "training_examples": 428,
            "optimizer_updates": 108,
            "completed_batches": 108,
            "actual_source_training_examples": 532,
            "actual_source_optimizer_updates": 134,
            "rolled_back_accepted_exposures": 104,
            "rolled_back_optimizer_updates": 26,
        }
        and stop["training_examples"] == 532
        and stop["optimizer_updates"] == 134,
        "source counters differ from the authorized rollback",
    )
    chosen = ROOT / proposal["candidate_inherited_prefix"]["path"]
    require(
        artifact(chosen) == proposal["candidate_inherited_prefix"]
        and chosen.read_bytes() == b"".join(raw.splitlines(keepends=True)[:108]),
        "candidate inherited prefix changed",
    )
    epochs = []
    for descriptor in audit["epoch_checkpoint_commits_bytes_bound"]:
        saved = read(ROOT / descriptor["path"])
        require(
            artifact(ROOT / descriptor["path"]) == descriptor
            and saved["complete"]
            and saved["committed_batch_RNG"],
            "source epoch commit changed",
        )
        for key in ("visual", "optimizer"):
            require(artifact(ROOT / saved[key]["path"]) == saved[key], "source epoch bytes changed")
        epochs.append(
            {
                "training_examples": saved["training_examples"],
                "visual": saved["visual"],
                "optimizer": saved["optimizer"],
                "commit": descriptor,
            }
        )
    require(
        [item["training_examples"] for item in epochs] == [214, 428],
        "latest complete epoch is not the reviewed428 prefix",
    )
    require(
        commit["visual"] == artifact(checkpoint)
        and commit["optimizer"] == proposal["optimizer"]
        and commit["optimizer_updates"] == 108,
        "source visual/Adam/RNG transaction changed",
    )
    require(
        run["plan_sha256"] == sha256(source_path)
        and run["initial_visual_sha256"] == source["vision_initialization_sha256"]
        and run["fixed_backbones"] == source["model_plan"]["initial_reference_hashes"]
        and run["recipe"] == source["recipe"]
        and run["training_examples"] == run["optimizer_updates"] == 0,
        "source lost original public initialization",
    )
    shapes = previous.shapes_for(source)
    optimizer = ROOT / commit["optimizer"]["path"]
    optimizer_inventory(optimizer, shapes, 108, source["recipe"])
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    for path in (checkpoint, optimizer):
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            metadata = saved.metadata()
            require(
                metadata.get("plan_sha256") == sha256(source_path)
                and metadata.get("update") == "428",
                "checkpoint tensor provenance changed",
            )
            if path == checkpoint:
                require(
                    metadata.get("optimizer_updates") == "108"
                    and metadata.get("fixed_winner_checkpoint")
                    == source["model_plan"]["reference"]["translator_checkpoint_sha256"],
                    "checkpoint optimizer count or frozen C5 origin changed",
                )

    values = load_file(str(checkpoint), device="cpu")
    require(
        set(values) == set(shapes)
        and all(
            list(value.shape) == shapes[name]
            and value.dtype == torch.float32
            and bool(torch.isfinite(value).all())
            for name, value in values.items()
        ),
        "source visual values or topology changed",
    )
    evidence = [
        descriptor
        for descriptor in audit["actual_raw_training_evidence"]
        if (path := Path(descriptor["path"]).relative_to(directory.relative_to(ROOT)))
        .parts[0]
        .startswith("batch-")
        and int(path.parts[0][6:]) < 108
    ]
    return {
        **prefix,
        "source_plan": artifact(source_path),
        "visual": artifact(checkpoint),
        "optimizer": artifact(optimizer),
        "source_steps": artifact(chosen),
        "actual_source_steps": artifact(directory / "steps.jsonl"),
        "source_output": str(directory.relative_to(ROOT)),
        "source_run": artifact(directory / "run.json"),
        "source_stop": artifact(directory / "stopped.json"),
        "source_failure": artifact(directory / "failure.json"),
        "source_exit": artifact(directory / "exit.json"),
        "source_worker": artifact(directory / "worker.json"),
        "source_commit": artifact(commit_path),
        "terminal_audit": artifact(audit_path),
        "authorization": authorization,
        "proposal": artifact(PROPOSAL),
        "starting_visual_state_sha256": previous.original.state_digest(
            {name.removeprefix("vision."): value for name, value in values.items()}
        ),
        "source_evidence": evidence,
        "actual_source_evidence": audit["actual_raw_training_evidence"],
        "source_epochs": epochs,
        "failed_batch_grades_discarded": False,
        "rolled_back_accepted_rewards_reused": False,
        "manual_continuation_limit": 1,
    }


def checked_binding(binding):
    expected = make_binding(
        ROOT / binding["source_plan"]["path"],
        ROOT / binding["visual"]["path"],
        ROOT / binding["terminal_audit"]["path"],
        ROOT / binding["authorization"]["path"],
    )
    require(binding == expected, "authorized rollback binding changed")
    return expected


def claim(path, output, plan):
    binding = checked_binding(plan["resume"])
    target = (ROOT / binding["source_plan"]["path"]).with_name("word-manual-continuation.json")
    write(
        target,
        {
            "plan": artifact(path),
            "output": str(output.relative_to(ROOT)),
            "worker": artifact(output / "worker.json"),
            "claimed_at": now(),
            "authorization": binding["authorization"],
            "manual_continuation_limit": 1,
            "automatic_continuations": 0,
            "rolled_back_accepted_exposures": 104,
        },
    )


def inherit_prefix(output, plan):
    binding = checked_binding(plan["resume"])
    directory = ROOT / binding["source_output"]
    copied = []
    for descriptor in binding["source_evidence"]:
        source = ROOT / descriptor["path"]
        relative = source.relative_to(directory)
        require(
            relative.parts[0].startswith("batch-")
            and int(relative.parts[0][6:]) < 108
            and source.suffix == ".json",
            "rollback cannot inherit discarded or unrelated evidence",
        )
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        storage(source.stat().st_size, output=output)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        require(sha256(target) == descriptor["sha256"], "inherited evidence changed")
        copied.append({"source": descriptor, "copy": artifact(target)})
    raw = (ROOT / binding["source_steps"]["path"]).read_bytes()
    storage(len(raw), output=output)
    with (output / "steps.jsonl").open("xb") as stream:
        stream.write(raw)
    inherited = []
    for item in binding["source_epochs"]:
        new = {}
        for key in ("visual", "optimizer"):
            source = ROOT / item[key]["path"]
            target = output / source.name
            storage(source.stat().st_size, output=output)
            require(not target.exists(), "inherited checkpoints cannot overwrite evidence")
            shutil.copyfile(source, target)
            require(sha256(target) == item[key]["sha256"], "inherited checkpoint bytes changed")
            new[key] = artifact(target)
        commit = read(ROOT / item["commit"]["path"])
        target = output / Path(item["commit"]["path"]).name
        write(target, {**commit, **new, "inherited_from": item["commit"]})
        inherited.append({"source": item, **new, "commit": artifact(target)})
    record = {
        "source_plan": binding["source_plan"],
        "source_steps": binding["source_steps"],
        "inherited_step_bytes": len(raw),
        "inherited_step_sha256": binding["source_steps"]["sha256"],
        "accepted_evidence": copied,
        "epochs": inherited,
        "rejected_batch_evidence_copied": False,
        "rolled_back_accepted_evidence_copied": False,
        "rolled_back_accepted_exposures": 104,
        "actual_source_steps": binding["actual_source_steps"],
    }
    write(output / "inheritance.json", record)
    return record


def validate_inheritance(output, plan):
    binding = checked_binding(plan["resume"])
    # The old validator's source_steps points at the chosen prefix. Rebind the
    # original source-directory calculation without changing its byte checks.
    record = read(output / "inheritance.json")
    raw = (ROOT / binding["source_steps"]["path"]).read_bytes()
    require(
        record["source_plan"] == binding["source_plan"]
        and record["source_steps"] == binding["source_steps"]
        and record["inherited_step_bytes"] == len(raw)
        and record["inherited_step_sha256"] == binding["source_steps"]["sha256"]
        and (output / "steps.jsonl").read_bytes().startswith(raw),
        "inherited prefix changed",
    )
    directory = ROOT / binding["source_output"]
    copies = []
    for descriptor in binding["source_evidence"]:
        target = output / (ROOT / descriptor["path"]).relative_to(directory)
        require(sha256(target) == descriptor["sha256"], "inherited raw evidence changed")
        copies.append({"source": descriptor, "copy": artifact(target)})
    require(
        record["accepted_evidence"] == copies
        and record["rolled_back_accepted_evidence_copied"] is False
        and record["rolled_back_accepted_exposures"] == 104
        and record["rejected_batch_evidence_copied"] is False
        and record["actual_source_steps"] == binding["actual_source_steps"],
        "discarded accepted work entered inheritance",
    )
    expected = {ROOT / item["copy"]["path"] for item in copies}
    actual = {
        file
        for number in range(108)
        for file in (output / f"batch-{number:04d}").rglob("*")
        if file.is_file()
    }
    require(actual == expected, "inherited directory inventory changed")
    require(
        len(record["epochs"]) == len(binding["source_epochs"]) == 2,
        "inherited epoch inventory changed",
    )
    epochs = {}
    for source, copied in zip(binding["source_epochs"], record["epochs"], strict=True):
        require(copied["source"] == source, "inherited epoch source changed")
        for key in ("visual", "optimizer"):
            expected_path = output / Path(source[key]["path"]).name
            require(
                copied[key]["path"] == str(expected_path.relative_to(ROOT))
                and artifact(expected_path) == copied[key]
                and copied[key]["sha256"] == source[key]["sha256"],
                "inherited epoch tensors changed",
            )
        commit_path = output / Path(source["commit"]["path"]).name
        require(
            copied["commit"]["path"] == str(commit_path.relative_to(ROOT))
            and artifact(commit_path) == copied["commit"]
            and read(commit_path)
            == {
                **read(ROOT / source["commit"]["path"]),
                "visual": copied["visual"],
                "optimizer": copied["optimizer"],
                "inherited_from": source["commit"],
            },
            "inherited epoch provenance changed",
        )
        epochs[source["training_examples"]] = {
            "source_plan_sha256": binding["source_plan"]["sha256"],
            "commit": source["commit"],
        }
    return record, epochs


def restore_committed(plan, model, optimizer, duty):
    counts = bind(previous.restore_committed, checked_binding=checked_binding)(
        plan, model, optimizer, duty
    )
    import torch
    from safetensors import safe_open

    binding = plan["resume"]
    with safe_open(str(ROOT / binding["optimizer"]["path"]), framework="pt", device="cpu") as saved:
        expected = {key: saved.get_tensor("rng." + key) for key in ("cpu", "mps")}
    actual = duty.run(lambda: {"cpu": torch.get_rng_state(), "mps": torch.mps.get_rng_state()})
    require(
        all(torch.equal(actual[key], expected[key]) for key in expected),
        "restored CPU/MPS RNG bytes differ from the committed checkpoint",
    )
    return counts
