"""Bind and inherit one committed word prefix without changing its scientific recipe."""

from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import train_c5_word_semantic as original
from benchmark_fleurs_dev import sha256
from c5_word_epoch_audit import optimizer_inventory
from c5_word_semantic_model import PARAMETERS, TENSORS
from c5_word_training import storage
from launch_terra_visual_job import now, write
from probe_c5_word_semantic import require_resources
from roi_semantic_resume import restore as restore
from terra_training_recovery import artifact, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
STOP_AUDIT_SHA256 = "d95a949e09a802df79f831d81a33fdb9e28bb8af44fd60539a7472a2141cdabd"
UTC = timezone.utc  # noqa: UP017 - pinned model runtime is Python3.10


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def prefix_state(plan, steps):
    require(0 < len(steps) < len(plan["batches"]), "word recovery needs an unfinished prefix")
    examples = updates = 0
    for number, step in enumerate(steps):
        batch = plan["batches"][number]
        qualities = step["Terra_qualities"]
        require(
            all(step.get(key) == value for key, value in batch.items())
            and step.get("batch") == number
            and len(qualities) == len(batch["training_record_indices"]),
            "word recovery changed original complete batch order",
        )
        require(
            all(
                quality
                and quality[0] == 10
                and all(type(value) is int and 0 <= value <= 10 for value in quality)
                for quality in qualities
            ),
            "word recovery changed actual grades",
        )
        contrast = sum(len(set(quality)) > 1 for quality in qualities)
        norm = step["gradient_norm_before_clip"]
        require(
            math.isfinite(norm) and norm > 0 if contrast else norm is None,
            "word recovery prefix has an invalid applied-step norm",
        )
        examples += len(batch["training_record_indices"])
        updates += bool(contrast)
        require(
            step["training_examples"] == examples
            and step["optimizer_updates"] == updates
            and step["contrasting_clips"] == contrast,
            "word recovery counter differs from grades",
        )
        require_resources(step["resources"])
    return examples, updates


def shapes_for(plan):
    descriptor = plan["model_plan"]["visual_scope"]
    path = ROOT / descriptor["path"]
    require(artifact(path) == descriptor, "word recovery original shape scope changed")
    shapes = {"vision." + row["parameter"]: row["shape"] for row in read(path)["parameters"]}
    require(
        len(shapes) == TENSORS
        and sum(math.prod(shape) for shape in shapes.values()) == PARAMETERS
        and set(shapes) == set(plan["model_plan"]["expected_visual_gradient_names"]),
        "word recovery changed exact original350-tensor scope",
    )
    return shapes


def make_binding(source_path, checkpoint, audit_path):
    from safetensors import safe_open
    from safetensors.torch import load_file

    source_path, checkpoint, audit_path = (
        path.resolve() for path in (source_path, checkpoint, audit_path)
    )
    source = original.checked(source_path)
    require("resume" not in source, "word permits one continuation without a continuation chain")
    verify_snapshot(source_path, source)
    require(
        sha256(audit_path) == STOP_AUDIT_SHA256, "word recovery requires the reviewed stop audit"
    )
    audit = read(audit_path)
    directory = checkpoint.parent
    stop, exited, worker, run = [
        read(directory / name) for name in ("stopped.json", "exit.json", "worker.json", "run.json")
    ]
    raw = (directory / "steps.jsonl").read_bytes()
    require(raw.endswith(b"\n"), "word recovery cannot consume a partial ledger append")
    steps = [json.loads(line) for line in raw.splitlines()]
    examples, updates = prefix_state(source, steps)
    require(
        exited.get("exit_code") == 1
        and exited.get("pid") == worker.get("pid")
        and type(worker.get("pid")) is int
        and worker["pid"] > 0,
        "word recovery needs actual same-worker exit1",
    )
    require(
        stop.get("error") == "word judge failed complete integer rewards or original calibration"
        and stop.get("automatic_retries") == stop.get("automatic_continuations") == 0
        and stop.get("recovery_is_committed_prefix") is True
        and stop.get("uncommitted_optimizer_mutation_possible") is False
        and stop.get("recovery_error") is None
        and stop.get("training_examples") == examples
        and stop.get("optimizer_updates") == updates
        and updates > 0
        and not (directory / "result.json").exists(),
        "word recovery must follow a committed preupdate calibration stop",
    )
    require(
        audit.get("schema") == "umi-studio-original-word-stopped-terminal-audit/1"
        and audit.get("audit_complete") is True
        and audit.get("same_original_worker_exit1") is True
        and audit.get("original_PID_absent") is True
        and audit.get("plan") == artifact(source_path)
        and audit.get("worker") == artifact(directory / "worker.json")
        and audit.get("exit") == artifact(directory / "exit.json")
        and audit.get("stopped") == artifact(directory / "stopped.json")
        and audit.get("training_examples") == examples
        and audit.get("optimizer_updates") == updates
        and audit.get("completed_batches") == len(steps)
        and audit.get("failed_batch") == len(steps)
        and audit.get("failed_batch_rewards_admitted_or_optimizer_updated") is False
        and audit.get("failed_batch_control_violations")
        and audit.get("curriculum_complete") is False
        and audit.get("fixed_final_evaluation_eligible") is False
        and audit.get("all_pinned_sources_and_model_assets_match") is True,
        "word recovery lost its independently authenticated actual stop",
    )
    previous_observation = read(ROOT / audit["previous_accepted_observation"]["path"])
    descriptors = [
        audit["audit_source"],
        audit["failure"],
        audit["previous_accepted_observation"],
        previous_observation["original_worker"],
        previous_observation["original_run"],
        audit["completed_steps_prefix"],
        *previous_observation["epoch_checkpoint_commits"],
        *audit["actual_completed_evidence"],
        *audit["actual_failed_evidence"],
    ]
    for descriptor in descriptors:
        require(
            artifact(ROOT / descriptor["path"]) == descriptor,
            "word recovery audited evidence changed",
        )
    require(
        (ROOT / audit["completed_steps_prefix"]["path"]).read_bytes() == raw,
        "word recovery prefix differs from its audited committed ledger",
    )
    require(
        run["plan_sha256"] == sha256(source_path)
        and run["initial_visual_sha256"] == source["vision_initialization_sha256"]
        and run["fixed_backbones"] == source["model_plan"]["initial_reference_hashes"]
        and run["recipe"] == source["recipe"]
        and run["training_examples"] == run["optimizer_updates"] == 0,
        "word recovery changed original initialization or recipe",
    )
    optimizer = checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")
    commit_path = checkpoint.with_suffix(".json")
    commit = read(commit_path)
    require(
        checkpoint.name == f"visual-recovery-after-{examples:06d}.safetensors"
        and stop["recovery"] == artifact(checkpoint) == commit["visual"] == audit["recovery_visual"]
        and artifact(optimizer) == commit["optimizer"] == audit["recovery_optimizer"]
        and artifact(commit_path) == audit["recovery_commit"]
        and commit["complete"] is True
        and commit["committed_batch_RNG"] is True
        and commit["training_examples"] == examples
        and commit["optimizer_updates"] == updates,
        "word recovery lost complete visual/Adam/committed-RNG transaction",
    )
    shapes = shapes_for(source)
    for path in (checkpoint, optimizer):
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            metadata = saved.metadata()
            require(
                metadata.get("plan_sha256") == sha256(source_path)
                and metadata.get("update") == str(examples),
                "word recovery tensor provenance changed",
            )
            if path == checkpoint:
                require(
                    metadata.get("optimizer_updates") == str(updates)
                    and metadata.get("fixed_winner_checkpoint")
                    == source["model_plan"]["reference"]["translator_checkpoint_sha256"]
                    and set(saved.keys()) == set(shapes),
                    "word recovery tensor topology changed",
                )
    optimizer_inventory(optimizer, shapes, updates, source["recipe"])
    values = load_file(str(checkpoint), device="cpu")
    import torch

    require(
        all(
            list(value.shape) == shapes[name]
            and value.dtype == torch.float32
            and bool(torch.isfinite(value).all())
            for name, value in values.items()
        ),
        "word recovery visual values are invalid",
    )
    epochs = []
    for number, step in enumerate(steps):
        if (
            number + 1 < len(source["batches"])
            and source["batches"][number + 1]["epoch"] != step["epoch"]
        ):
            if step["optimizer_updates"] > 0:
                visual = directory / f"visual-after-{step['training_examples']:06d}.safetensors"
                previous_commit = visual.with_suffix(".json")
                previous_optimizer = visual.with_name(visual.stem + "-optimizer.safetensors")
                saved = read(previous_commit)
                require(
                    saved["visual"] == artifact(visual)
                    and saved["optimizer"] == artifact(previous_optimizer)
                    and saved["complete"] is True
                    and saved["committed_batch_RNG"] is True
                    and saved["training_examples"] == step["training_examples"]
                    and saved["optimizer_updates"] == step["optimizer_updates"],
                    "inherited word epoch changed",
                )
                epochs.append(
                    {
                        "training_examples": step["training_examples"],
                        "visual": artifact(visual),
                        "optimizer": artifact(previous_optimizer),
                        "commit": artifact(previous_commit),
                    }
                )
    require(
        set(directory.glob("visual-after-*.json"))
        == {ROOT / item["commit"]["path"] for item in epochs},
        "word source changed exact complete epoch commit inventory",
    )
    require(
        [item["commit"] for item in epochs] == previous_observation["epoch_checkpoint_commits"],
        "word recovery epoch is not the previously accepted checkpoint",
    )
    return {
        "source_plan": artifact(source_path),
        "visual": artifact(checkpoint),
        "optimizer": artifact(optimizer),
        "source_steps": artifact(directory / "steps.jsonl"),
        "source_run": artifact(directory / "run.json"),
        "source_stop": artifact(directory / "stopped.json"),
        "source_failure": artifact(directory / "failure.json"),
        "source_exit": artifact(directory / "exit.json"),
        "source_worker": artifact(directory / "worker.json"),
        "source_commit": artifact(commit_path),
        "terminal_audit": artifact(audit_path),
        "training_examples": examples,
        "optimizer_updates": updates,
        "completed_batches": len(steps),
        "starting_visual_state_sha256": original.state_digest(
            {name.removeprefix("vision."): value for name, value in values.items()}
        ),
        "source_evidence": audit["actual_completed_evidence"],
        "source_epochs": epochs,
        "failed_batch_grades_discarded": True,
        "manual_continuation_limit": 1,
    }


def checked_binding(binding):
    expected = make_binding(
        ROOT / binding["source_plan"]["path"],
        ROOT / binding["visual"]["path"],
        ROOT / binding["terminal_audit"]["path"],
    )
    require(binding == expected, "word recovery binding differs from actual audited source")
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
            "manual_continuation_limit": 1,
            "automatic_continuations": 0,
        },
    )


def source_cooldown(binding, *, clock=lambda: datetime.now(UTC), sleep=time.sleep):
    exited = datetime.fromisoformat(read(ROOT / binding["source_exit"]["path"])["observed_at"])
    require(exited.tzinfo is not None, "word recovery request boundary needs a timezone-aware exit")
    boundary = exited + timedelta(seconds=60)
    while (remaining := (boundary - clock()).total_seconds()) > 0:
        sleep(min(30, remaining))
    return {
        "source_exit": binding["source_exit"],
        "not_before": boundary.isoformat(),
        "ready_at": clock().isoformat(),
    }


def inherit_prefix(output, plan):
    binding = checked_binding(plan["resume"])
    directory = (ROOT / binding["source_steps"]["path"]).parent
    copied = []
    for descriptor in binding["source_evidence"]:
        source = ROOT / descriptor["path"]
        relative = source.relative_to(directory)
        first = relative.parts[0]
        require(
            first.startswith("batch-")
            and first[6:].isdigit()
            and int(first[6:]) < binding["completed_batches"]
            and source.suffix == ".json",
            "word inheritance cannot copy rejected or unrelated source evidence",
        )
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        storage(source.stat().st_size, output=output)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        require(sha256(target) == descriptor["sha256"], "inherited word evidence bytes changed")
        copied.append({"source": descriptor, "copy": artifact(target)})
    raw = (ROOT / binding["source_steps"]["path"]).read_bytes()
    with (output / "steps.jsonl").open("xb") as stream:
        stream.write(raw)
    inherited = []
    for item in binding["source_epochs"]:
        new = {}
        for kind in ("visual", "optimizer"):
            source = ROOT / item[kind]["path"]
            target = output / source.name
            require(not target.exists(), "word inherited checkpoint cannot overwrite evidence")
            storage(source.stat().st_size, output=output)
            shutil.copyfile(source, target)
            require(
                sha256(target) == item[kind]["sha256"], "inherited word checkpoint bytes changed"
            )
            new[kind] = artifact(target)
        original_commit = read(ROOT / item["commit"]["path"])
        target = output / Path(item["commit"]["path"]).name
        write(target, {**original_commit, **new, "inherited_from": item["commit"]})
        inherited.append({"source": item, **new, "commit": artifact(target)})
    record = {
        "source_plan": binding["source_plan"],
        "source_steps": binding["source_steps"],
        "inherited_step_bytes": len(raw),
        "inherited_step_sha256": binding["source_steps"]["sha256"],
        "accepted_evidence": copied,
        "epochs": inherited,
        "rejected_batch_evidence_copied": False,
    }
    write(output / "inheritance.json", record)
    return record


def prior_order(plan):
    return [
        index
        for batch in plan["batches"][: plan["resume"]["completed_batches"]]
        for index in batch["training_record_indices"]
    ]


def validate_inheritance(output, plan):
    binding = checked_binding(plan["resume"])
    record = read(output / "inheritance.json")
    directory = (ROOT / binding["source_steps"]["path"]).parent
    raw = (ROOT / binding["source_steps"]["path"]).read_bytes()
    require(
        record["source_plan"] == binding["source_plan"]
        and record["source_steps"] == binding["source_steps"]
        and record["inherited_step_bytes"] == len(raw)
        and record["inherited_step_sha256"] == binding["source_steps"]["sha256"]
        and (output / "steps.jsonl").read_bytes().startswith(raw)
        and record["rejected_batch_evidence_copied"] is False,
        "word continuation changed its inherited exact ledger prefix",
    )
    copies = []
    for descriptor in binding["source_evidence"]:
        source = ROOT / descriptor["path"]
        target = output / source.relative_to(directory)
        require(sha256(target) == descriptor["sha256"], "word inherited accepted evidence changed")
        copies.append({"source": descriptor, "copy": artifact(target)})
    require(record["accepted_evidence"] == copies, "word continuation inheritance manifest changed")
    expected_files = {ROOT / item["copy"]["path"] for item in copies}
    actual_files = {
        path
        for number in range(binding["completed_batches"])
        for path in (output / f"batch-{number:04d}").rglob("*")
        if path.is_file()
    }
    require(actual_files == expected_files, "word inherited accepted directory inventory changed")
    require(
        len(record["epochs"]) == len(binding["source_epochs"]), "word inherited epoch count changed"
    )
    inherited = {}
    for source, copied in zip(binding["source_epochs"], record["epochs"], strict=True):
        require(copied["source"] == source, "word inherited epoch source changed")
        for key in ("visual", "optimizer"):
            path = output / Path(source[key]["path"]).name
            require(
                artifact(path) == copied[key] and copied[key]["sha256"] == source[key]["sha256"],
                "word inherited epoch tensor bytes changed",
            )
        commit_path = output / Path(source["commit"]["path"]).name
        expected_commit = {
            **read(ROOT / source["commit"]["path"]),
            "visual": copied["visual"],
            "optimizer": copied["optimizer"],
            "inherited_from": source["commit"],
        }
        require(
            artifact(commit_path) == copied["commit"] and read(commit_path) == expected_commit,
            "word inherited epoch commit lost its original provenance",
        )
        inherited[source["training_examples"]] = {
            "source_plan_sha256": binding["source_plan"]["sha256"],
            "commit": source["commit"],
        }
    return record, inherited


def prior_checkpoints(output):
    return [item["visual"] for item in read(output / "inheritance.json")["epochs"]]


def restore_committed(plan, model, optimizer, duty):
    binding = checked_binding(plan["resume"])
    expected_fixed = plan["model_plan"]["initial_reference_hashes"]
    masks = read(ROOT / binding["source_run"]["path"])["frozen_mask_sha256"]
    counts = duty.run(lambda: restore(plan, model, optimizer))

    def verify():
        parameters = model.trainable_parameters()
        digest = original.state_digest(
            {name.removeprefix("vision."): value for name, value in parameters.items()}
        )
        require(
            digest == binding["starting_visual_state_sha256"]
            and original.fixed_hashes(model) == expected_fixed
            and original.mask_digest(model) == masks,
            "word restore changed committed visual values, original C5 backbones or masks",
        )

    duty.run(verify)
    require(
        counts == (binding["training_examples"], binding["optimizer_updates"]),
        "word restore counters changed",
    )
    return counts
