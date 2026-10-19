"""Read-only admission for a proposed second, explicitly approved word recovery.

The exact source stopped on HTTP502 at a durably saved mid-epoch checkpoint.
This module creates no approval, claim, plan, output, model or API request.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "runs/c5-polarity-word-http502-recovery-preparation-v1/proposal.json"
PROPOSAL_SHA = "dbeddd0193413ea36bebd7560efc6096cc2eed94f9c592a0f574f4ff6cf6bd7b"
SCOPE = "one-additional-polarity-word-HTTP502-recovery-from-512"
ERROR = "Terra request rejected with HTTP 502; receipt preserved"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).absolute()
    require(
        path.resolve() == path
        and (path.is_relative_to(ROOT / "runs") or path.is_relative_to(ROOT / "checkpoints")),
        "HTTP502 recovery evidence must be a canonical owned private file",
    )
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def bound(descriptor):
    require(set(descriptor) == {"path", "sha256"}, "invalid recovery evidence descriptor")
    path = ROOT / descriptor["path"]
    require(artifact(path) == descriptor, "HTTP502 recovery source evidence changed")
    return path


def reviewed_proposal():
    require(sha(PROPOSAL) == PROPOSAL_SHA, "reviewed HTTP502 proposal changed")
    proposal = read(PROPOSAL)
    require(
        proposal["inherited_accepted_exposures"] == 512
        and proposal["inherited_accepted_updates"] == proposal["inherited_completed_batches"] == 129
        and proposal["remaining_exposures"] == 344
        and proposal["remaining_batches"] == 87
        and proposal["actual_accepted_exposures_before_new_work"] == 616
        and proposal["actual_updates_before_new_work"] == 155
        and proposal["actual_final_accepted_exposures_if_complete"] == 960
        and proposal["newly_rolled_back_accepted_exposures"]
        == proposal["newly_rolled_back_accepted_updates"]
        == 0
        and proposal["original_rolled_back_parent_exposures_preserved"] == 104
        and proposal["original_rolled_back_parent_updates_preserved"] == 26,
        "HTTP502 proposal lost exact saved-prefix or actual-work accounting",
    )
    return proposal


def checked_authorization(path):
    """Require a new recorded user approval, without inventing an expiry."""
    path = Path(path).absolute()
    require(path.is_file(), "new explicit HTTP502 recovery authorization is required")
    descriptor = artifact(path)
    proposal = reviewed_proposal()
    approval = read(path)
    require(
        approval.get("scope") == SCOPE
        and approval.get("approved") is True
        and approval.get("source") == "explicit_user_message"
        and isinstance(approval.get("user_message"), str)
        and approval["user_message"].strip()
        and approval.get("proposal") == artifact(PROPOSAL)
        and approval.get("source_checkpoint") == proposal["starting_working_state"]["visual"]
        and all(
            type(approval.get(key)) is int and approval[key] == expected
            for key, expected in (
                ("additional_manual_continuation_limit", 1),
                ("automatic_retries", 0),
                ("automatic_further_continuations", 0),
            )
        ),
        "another continuation requires its own exact explicit HTTP502 approval",
    )
    received = datetime.fromisoformat(approval["received_at"].replace("Z", "+00:00"))
    require(received.tzinfo is not None, "HTTP502 approval requires a timezone-aware timestamp")
    return descriptor, proposal


def check_terminal(proposal, source, audit, directory):
    """Verify stop identity and recompute nested accepted-work accounting."""
    require(
        audit.get("schema") == "umi-studio-polarity-storage-recovery-http502-stop-audit/1"
        and audit.get("audit_complete") is True
        and audit.get("complete") is False
        and audit.get("same_approved_recovery_worker_terminal_exit1_authenticated") is True
        and audit["plan"] == proposal["source_plan"]
        and audit["durable_working_state"] == proposal["starting_working_state"]
        and audit["training_examples"] == 512
        and audit["optimizer_updates"] == audit["complete_batch_records"] == 129,
        "HTTP502 recovery lacks its exact terminal saved-prefix audit",
    )
    stopped, failed, exited, worker = [
        read(directory / name)
        for name in ("stopped.json", "failure.json", "exit.json", "worker.json")
    ]
    require(
        stopped["error"] == failed["error"] == ERROR
        and stopped["automatic_continuations"]
        == stopped["automatic_retries"]
        == failed["automatic_retries"]
        == 0
        and stopped["recovery_error"] is None
        and stopped["recovery_is_committed_prefix"] is True
        and stopped["uncommitted_optimizer_mutation_possible"] is False
        and stopped["training_examples"] == 512
        and stopped["optimizer_updates"] == 129
        and stopped["recovery"] == proposal["starting_working_state"]["visual"]
        and worker["uid"] == os.getuid()
        and worker["pid"] == exited["pid"]
        and exited["exit_code"] == 1
        and not (directory / "result.json").exists(),
        "HTTP502 source has changed or lacks an exact committed preupdate stop",
    )
    for key in ("exit", "stopped", "failure"):
        require(
            artifact(directory / (key + ".json")) == audit["owned_process_ps"][key]
            and audit["owned_process_ps"][key]
            == audit["owned_process_ps_after_authentication"][key],
            "HTTP502 terminal receipts changed",
        )
    parent = source["resume"]
    actual_examples = parent["actual_source_training_examples"] + 512 - parent["training_examples"]
    actual_updates = parent["actual_source_optimizer_updates"] + 129 - parent["optimizer_updates"]
    require(
        actual_examples == audit["actual_total_lineage_exposures"] == 616
        and actual_updates == audit["actual_total_lineage_optimizer_updates"] == 155
        and parent["rolled_back_accepted_exposures"] == 104
        and parent["rolled_back_optimizer_updates"] == 26
        and source["manual_continuations_authorized"] == 1
        and source["automatic_further_continuations"] == 0,
        "HTTP502 recovery lost original rollback or nested actual-work accounting",
    )
    failure = audit["failed_batch"]
    require(
        failure["batch"] == 129
        and failure["accepted_exposures"] == failure["optimizer_updates"] == 0
        and failure["completed_raw_requests"] == failure["HTTP502_failed_requests"] == 1
        and failure["partial_scores_not_used_for_update_or_recovery"] is True,
        "partial HTTP502 batch became accepted training work",
    )
    return actual_examples, actual_updates


def source_binding():
    """Read-only preparation; creates no runnable authorization or training plan.

    The exact immutable audited source is authenticated without recursively
    repeating its already completed model/gradient qualification. Its complete
    source snapshot, original assets, ancestor binding and retained proof bytes
    are checked before numerical checkpoint inventory.
    """
    proposal = reviewed_proposal()
    source_path, audit_path = (bound(proposal[key]) for key in ("source_plan", "terminal_audit"))
    source, audit = read(source_path), read(audit_path)
    require(
        source["schema"] == "umi-studio-C5-polarity-storage-recovery/1"
        and source["resume"]["training_examples"] == 428
        and source["resume"]["optimizer_updates"] == 108,
        "HTTP502 recovery cannot select another ancestry",
    )
    for name, expected in {**source["source_sha256"], **source["model_plan"]["assets"]}.items():
        path = ROOT / name
        require(path.resolve() == path and sha(path) == expected, "pinned HTTP502 ancestor changed")

    import torch
    from audit_c5_word_lineage import optimizer_inventory
    from c5_word_resume import prefix_state, shapes_for
    from c5_word_storage_recovery import checked_authorization as first_authorization
    from c5_word_storage_recovery import validate_inheritance as first_inheritance
    from safetensors import safe_open
    from safetensors.torch import load_file
    from terra_polarity_judge import bind, protocol
    from terra_training_recovery import verify_snapshot
    from train_c5_word_semantic import state_digest

    verify_snapshot(source_path, source)
    require(source["judge_protocol"] == protocol(), "HTTP502 polarity protocol changed")
    first, _ = first_authorization(ROOT / source["resume"]["authorization"]["path"])
    require(first == source["resume"]["authorization"], "first recovery approval changed")

    def check_bound(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                bound(value)
            else:
                for child in value.values():
                    check_bound(child)
        elif isinstance(value, list):
            for child in value:
                check_bound(child)

    check_bound(source["resume"])
    directory = bound(audit["worker"]).parent
    actual_examples, actual_updates = check_terminal(proposal, source, audit, directory)
    assets = [
        *audit["actual_raw_training_evidence"],
        *audit["working_history_bound_assets"],
        *audit["failed_batch"]["evidence"],
        *[
            audit[key]
            for key in (
                "observer_source",
                "session_poll",
                "previous_observation",
                "worker",
                "run",
                "execution_claim",
                "inheritance",
                "completed_steps_prefix",
            )
        ],
    ]
    seen = {}
    for descriptor in assets:
        require(
            descriptor["path"] not in seen or seen[descriptor["path"]] == descriptor["sha256"],
            "conflicting HTTP502 evidence",
        )
        if descriptor["path"] not in seen:
            bound(descriptor)
            seen[descriptor["path"]] = descriptor["sha256"]
    binding = source["resume"]

    def known_binding(value):
        require(value == binding, "first recovery binding changed")
        return binding

    _, origins = bind(first_inheritance, checked_binding=known_binding)(directory, source)
    raw = (directory / "steps.jsonl").read_bytes()
    require(
        raw == bound(audit["completed_steps_prefix"]).read_bytes() and raw.endswith(b"\n"),
        "HTTP502 accepted ledger changed",
    )
    steps = [json.loads(line) for line in raw.splitlines()]
    require(
        len(steps) == 129 and prefix_state(source, steps) == (512, 129),
        "HTTP502 accepted prefix changed",
    )
    require(
        len(audit["original_calibration_scores"]) == 129
        and all(
            x[:3] == [10, 10, 0] and 0 <= x[3] <= 3 for x in audit["original_calibration_scores"]
        ),
        "accepted HTTP502 calibration changed",
    )
    state = proposal["starting_working_state"]
    visual, optimizer, commit = (bound(state[key]) for key in ("visual", "optimizer", "commit"))
    require(
        read(commit)["visual"] == state["visual"]
        and read(commit)["optimizer"] == state["optimizer"],
        "HTTP502 working pair transaction changed",
    )
    shapes = shapes_for(source)
    for path in (visual, optimizer):
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            metadata = saved.metadata()
            require(
                metadata["plan_sha256"] == proposal["source_plan"]["sha256"]
                and metadata["update"] == "512",
                "HTTP502 working tensor provenance changed",
            )
            if path == visual:
                require(
                    set(saved.keys()) == set(shapes)
                    and metadata["optimizer_updates"] == "129"
                    and metadata["fixed_winner_checkpoint"]
                    == source["model_plan"]["reference"]["translator_checkpoint_sha256"],
                    "HTTP502 visual scope or frozen translator changed",
                )
    optimizer_inventory(optimizer, shapes, 129, source["recipe"])
    values = load_file(str(visual), device="cpu")
    require(
        all(
            list(value.shape) == shapes[name]
            and value.dtype == torch.float32
            and bool(torch.isfinite(value).all())
            for name, value in values.items()
        ),
        "HTTP502 visual values are invalid",
    )
    epochs = []
    for descriptor in proposal["inherited_epoch_commits"]:
        saved = read(bound(descriptor))
        count = saved["training_examples"]
        require(
            count in origins and saved["complete"] is True and saved["committed_batch_RNG"] is True,
            "HTTP502 inherited epoch changed",
        )
        epochs.append(
            {
                "training_examples": count,
                "visual": saved["visual"],
                "optimizer": saved["optimizer"],
                "commit": descriptor,
                "tensor_plan_sha256": origins[count]["source_plan_sha256"],
            }
        )
        for key in ("visual", "optimizer"):
            path = bound(saved[key])
            with safe_open(str(path), framework="np") as tensors:
                require(
                    tensors.metadata()["plan_sha256"] == epochs[-1]["tensor_plan_sha256"],
                    "inherited original tensor-plan provenance changed",
                )
    require(
        [item["training_examples"] for item in epochs] == [214, 428],
        "HTTP502 inherited complete epochs changed",
    )
    evidence = []
    for descriptor in audit["actual_raw_training_evidence"]:
        path = Path(descriptor["path"]).relative_to(directory.relative_to(ROOT))
        if path.parts[0].startswith("batch-") and int(path.parts[0][6:]) < 129:
            require(path.suffix == ".json", "HTTP502 accepted inheritance contains nonreceipt data")
            evidence.append(descriptor)
    return {
        "source_plan": proposal["source_plan"],
        "terminal_audit": proposal["terminal_audit"],
        "proposal": artifact(PROPOSAL),
        "source_output": str(directory.relative_to(ROOT)),
        "visual": state["visual"],
        "optimizer": state["optimizer"],
        "source_commit": state["commit"],
        "source_steps": artifact(directory / "steps.jsonl"),
        "actual_source_steps": artifact(directory / "steps.jsonl"),
        **{
            key: artifact(directory / filename)
            for key, filename in (
                ("source_run", "run.json"),
                ("source_stop", "stopped.json"),
                ("source_failure", "failure.json"),
                ("source_exit", "exit.json"),
                ("source_worker", "worker.json"),
            )
        },
        "training_examples": 512,
        "optimizer_updates": 129,
        "completed_batches": 129,
        "actual_source_training_examples": actual_examples,
        "actual_source_optimizer_updates": actual_updates,
        "rolled_back_accepted_exposures": 104,
        "rolled_back_optimizer_updates": 26,
        "newly_rolled_back_accepted_exposures": 0,
        "newly_rolled_back_optimizer_updates": 0,
        "resource_reset_batches": [108, 129],
        "starting_visual_state_sha256": state_digest(
            {name.removeprefix("vision."): value for name, value in values.items()}
        ),
        "source_evidence": evidence,
        "source_epochs": epochs,
        # Existing restore/final consumers treat top-level dictionaries as file
        # descriptors. Keep compound ancestry in lists to preserve that contract.
        "source_working_states": [state],
        "ancestor_bindings": [binding],
        "actual_source_evidence": [*binding["actual_source_evidence"], *assets],
        "failed_batch_grades_discarded": True,
        "failed_batch_evidence": audit["failed_batch"]["evidence"],
        "failed_batch_rollouts_reused": False,
        "rolled_back_accepted_rewards_reused": False,
        "additional_manual_continuation_limit": 1,
    }


def make_binding(source_path, checkpoint, audit_path, authorization_path):
    authorization, proposal = checked_authorization(authorization_path)
    require(
        artifact(source_path) == proposal["source_plan"]
        and artifact(checkpoint) == proposal["starting_working_state"]["visual"]
        and artifact(audit_path) == proposal["terminal_audit"],
        "HTTP502 recovery cannot select another source or checkpoint",
    )
    return {**source_binding(), "authorization": authorization}


def checked_binding(binding):
    expected = make_binding(
        ROOT / binding["source_plan"]["path"],
        ROOT / binding["visual"]["path"],
        ROOT / binding["terminal_audit"]["path"],
        ROOT / binding["authorization"]["path"],
    )
    require(expected == binding, "HTTP502 recovery binding changed")
    return expected


def claim(path, output, plan):
    from launch_terra_visual_job import now, write

    binding = checked_binding(plan["resume"])
    target = (ROOT / binding["source_plan"]["path"]).with_name("word-manual-continuation.json")
    write(
        target,
        {
            "plan": artifact(path),
            "worker": artifact(output / "worker.json"),
            "output": str(output.relative_to(ROOT)),
            "authorization": binding["authorization"],
            "claimed_at": now(),
            "manual_continuation_limit": 1,
            "additional_manual_continuation_limit": 1,
            "manual_continuation_number": 2,
            "automatic_continuations": 0,
            "rolled_back_accepted_exposures": 104,
            "newly_rolled_back_accepted_exposures": 0,
        },
    )


def inherit_prefix(output, plan):
    """Reuse accepted-receipt/epoch copying; keep the working seed separate."""
    import shutil

    import c5_word_resume as previous
    from c5_word_training import storage
    from launch_terra_visual_job import write
    from terra_polarity_judge import bind

    binding = checked_binding(plan["resume"])

    def publish(path, record):
        if path == output / "inheritance.json":
            source = binding["source_working_states"][0]
            copied = {}
            for key, suffix in (
                ("visual", ".safetensors"),
                ("optimizer", "-optimizer.safetensors"),
            ):
                original = bound(source[key])
                target = output / ("resume-seed-000512" + suffix)
                require(not target.exists(), "HTTP502 seed cannot overwrite existing evidence")
                storage(original.stat().st_size, output=output)
                shutil.copyfile(original, target)
                require(sha(target) == source[key]["sha256"], "HTTP502 seed copy changed")
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                copied[key] = artifact(target)
            commit = output / "resume-seed-000512.json"
            source_commit = read(bound(source["commit"]))
            storage(16384, output=output)
            write(commit, {**source_commit, **copied, "inherited_from": source["commit"]})
            record.update(
                {
                    "working_seed": {"source": source, **copied, "commit": artifact(commit)},
                    "working_seed_is_complete_epoch": False,
                    "ancestor_bindings": binding["ancestor_bindings"],
                    "rolled_back_accepted_exposures": 104,
                    "newly_rolled_back_accepted_exposures": 0,
                }
            )
        storage(len(json.dumps(record).encode()) + 4096, output=output)
        write(path, record)

    return bind(
        previous.inherit_prefix,
        ROOT=ROOT,
        checked_binding=checked_binding,
        artifact=artifact,
        storage=storage,
        write=publish,
    )(output, plan)


def validate_inheritance(output, plan):
    import c5_word_resume as previous
    from terra_polarity_judge import bind

    binding = checked_binding(plan["resume"])
    record, epochs = bind(
        previous.validate_inheritance,
        ROOT=ROOT,
        checked_binding=checked_binding,
        artifact=artifact,
    )(output, plan)
    seed = record["working_seed"]
    source = binding["source_working_states"][0]
    require(
        seed["source"] == source
        and record["working_seed_is_complete_epoch"] is False
        and record["ancestor_bindings"] == binding["ancestor_bindings"]
        and record["rolled_back_accepted_exposures"] == 104
        and record["newly_rolled_back_accepted_exposures"] == 0,
        "HTTP502 inheritance lost its separate working seed or prior rollback",
    )
    for key, suffix in (("visual", ".safetensors"), ("optimizer", "-optimizer.safetensors")):
        target = output / ("resume-seed-000512" + suffix)
        require(
            artifact(target) == seed[key]
            and seed[key]["sha256"] == source[key]["sha256"]
            and not os.path.samefile(target, bound(source[key])),
            "HTTP502 inherited seed bytes or file ownership changed",
        )
    commit = output / "resume-seed-000512.json"
    require(
        artifact(commit) == seed["commit"]
        and read(commit)
        == {
            **read(bound(source["commit"])),
            "visual": seed["visual"],
            "optimizer": seed["optimizer"],
            "inherited_from": source["commit"],
        },
        "HTTP502 working seed commit changed",
    )
    for descriptor, copied in zip(binding["source_epochs"], record["epochs"], strict=True):
        count = descriptor["training_examples"]
        epochs[count]["source_plan_sha256"] = descriptor["tensor_plan_sha256"]
        for key in ("visual", "optimizer"):
            require(
                not os.path.samefile(bound(copied[key]), bound(descriptor[key])),
                "inherited epoch aliases its source",
            )
    for item in record["accepted_evidence"]:
        require(
            not os.path.samefile(bound(item["source"]), bound(item["copy"])),
            "inherited receipt aliases its source",
        )
    return record, epochs


def restore_committed(plan, model, optimizer, duty):
    from c5_word_storage_recovery import restore_committed as original
    from terra_polarity_judge import bind

    # Preserve original visual/frozen-mask/Adam/counter and actual CPU/MPS RNG
    # restore checks. Every top-level dictionary remains a file descriptor.
    require(
        all(
            set(value) == {"path", "sha256"}
            for value in plan["resume"].values()
            if isinstance(value, dict)
        ),
        "HTTP502 restore binding contains a compound nonartifact dictionary",
    )
    return bind(original, checked_binding=checked_binding)(plan, model, optimizer, duty)
