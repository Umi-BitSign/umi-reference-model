"""Validate a complete, explicitly authorized polarity storage-recovery lineage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import c5_polarity_word_final as polarity
import c5_word_final as original
import c5_word_resumed_final as resumed
import train_c5_word_storage_recovery as training
from benchmark_fleurs_dev import sha256
from c5_word_storage_recovery import validate_inheritance
from c5_word_working_checkpoints import KEEP, SCHEMA, WorkingCheckpoints, digest, require
from terra_polarity_judge import bind
from terra_training_recovery import artifact

ROOT = Path(__file__).resolve().parents[1]


def same_tensor_state(left, right):
    """Require equal tensors and semantic metadata, independent of file encoding."""
    import torch
    from safetensors import safe_open

    def metadata(saved):
        result = saved.metadata()
        for key in ("parameter_order", "groups_without_parameters"):
            if key in result:
                result[key] = json.loads(result[key])
        return result

    with (
        safe_open(str(left), framework="pt", device="cpu") as first,
        safe_open(str(right), framework="pt", device="cpu") as second,
    ):
        require(
            set(first.keys()) == set(second.keys()) and metadata(first) == metadata(second),
            "working and epoch tensor topology or metadata differ",
        )
        for name in first.keys():
            a, b = first.get_tensor(name), second.get_tensor(name)
            require(
                a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b),
                "working and epoch tensor values differ: " + name,
            )


def checked_working_history(directory, plan, plan_sha, steps, result):
    """Read-only verification of every save and retirement in a successful run.

    Retired tensor bytes cannot be re-read. Their preserved complete commit
    receipts and exact retirement transactions remain bound to the full ledger.
    The final two pairs are intact; the latest equals the fixed epoch final.
    """
    store = WorkingCheckpoints(
        directory,
        workspace=ROOT,
        plan_sha=plan_sha,
        storage_check=lambda _: None,
        pair_reserve_bytes=1,
    )
    events = store._events()  # A successful final cannot contain a torn append.
    raw, actual_steps = store._ledger()
    require(actual_steps == steps, "working history and final ledger differ")
    seed_batches = plan["resume"]["completed_batches"]
    expected_counts = [seed_batches, *range(seed_batches + 1, len(steps) + 1)]
    states = [event["state"] for event in events if event.get("kind") == "committed"]
    require(
        [state.get("ledger_rows") for state in states] == expected_counts,
        "working history must include the seed and every new accepted batch exactly once",
    )
    prefixes = [b""]
    for line in raw.splitlines(keepends=True):
        prefixes.append(prefixes[-1] + line)
    expected_events, assets, expected_files = [], [], set()
    for index, state in enumerate(states):
        count = expected_counts[index]
        prefix, step = prefixes[count], steps[count - 1]
        require(
            set(state)
            == {
                "schema",
                "plan_sha256",
                "inherited_seed",
                "training_examples",
                "optimizer_updates",
                "ledger_bytes",
                "ledger_rows",
                "ledger_sha256",
                "visual",
                "optimizer",
                "commit",
            }
            and state["schema"] == SCHEMA
            and state["plan_sha256"] == plan_sha
            and state["inherited_seed"] is (index == 0)
            and type(state["ledger_rows"]) is int
            and type(state["ledger_bytes"]) is int
            and state["ledger_bytes"] == len(prefix)
            and state["ledger_sha256"] == digest(prefix)
            and all(
                type(state[key]) is int and state[key] == step[key]
                for key in ("training_examples", "optimizer_updates")
            ),
            "working state lost its exact ledger prefix or counters",
        )
        commit_path = store._check_artifact(state["commit"])
        commit = json.loads(commit_path.read_text())
        assets.append(state["commit"])
        expected_directory = directory if index == 0 else store.directory / f"batch-{count:06d}"
        require(commit_path.parent == expected_directory, "working commit escaped its batch")
        require(
            commit.get("complete") is True
            and commit.get("committed_batch_RNG") is True
            and all(
                commit.get(key) == state[key]
                for key in ("training_examples", "optimizer_updates", "visual", "optimizer")
            )
            and state["visual"]["path"] != state["optimizer"]["path"],
            "working commit lost complete model/Adam/RNG provenance",
        )
        if index == 0:
            inherited = json.loads((directory / "inheritance.json").read_text())["epochs"][-1]
            require(
                all(state[key] == inherited[key] for key in ("visual", "optimizer", "commit")),
                "working seed is not the selected inherited epoch",
            )
        else:
            require("inherited_from" not in commit, "new working state cannot claim inheritance")
            expected_files.add(commit_path)
        retained = index == 0 or index >= len(states) - KEEP
        for key in ("visual", "optimizer"):
            value = state[key]
            require(set(value) == {"path", "sha256"}, "invalid working tensor descriptor")
            path = store._path(value["path"])
            require(
                path.parent == expected_directory and path.suffix == ".safetensors",
                "working tensor escaped its batch",
            )
            if retained:
                store._check_artifact(value)
                assets.append(value)
                if index:
                    expected_files.add(path)
            else:
                require(not path.exists(), "retired working tensor remains present")
        if retained:
            require(
                store._capture(commit_path, prefix, steps[:count], inherited=index == 0) == state,
                "retained working state does not match its committed pair",
            )
        expected_events.extend(
            [
                {"kind": "prepared", "state": state},
                {"kind": "committed", "state": state},
            ]
        )
        if index > KEEP:
            retired = states[index - KEEP]
            for key in ("visual", "optimizer"):
                expected_events.extend(
                    [
                        {"kind": "retire_intent", "artifact": retired[key]},
                        {"kind": "retired", "artifact": retired[key]},
                    ]
                )
    require(events == expected_events, "incomplete, extra or reordered working transactions")
    require(store.latest() == states[-1], "latest pointer does not identify the complete final")
    paths = list(store.directory.rglob("*"))
    require(all(not path.is_symlink() for path in paths), "aliased working history")
    require(
        {path for path in paths if path.is_file()} == expected_files
        and {path for path in paths if path.is_dir()} == {path.parent for path in expected_files},
        "working history contains orphan or missing artifacts",
    )
    final_path = ROOT / result["visual"]["path"]
    final_commit = json.loads(final_path.with_suffix(".json").read_text())
    for key in ("visual", "optimizer"):
        same_tensor_state(ROOT / states[-1][key]["path"], ROOT / final_commit[key]["path"])
    report = {
        "complete": True,
        "seed_batches": seed_batches,
        "new_committed_pairs": len(states) - 1,
        "retained_working_pairs": KEEP,
        "retired_working_pairs": len(states) - 1 - KEEP,
        "complete_transaction_events": len(events),
        "all_commit_receipts_and_ledger_prefixes_valid": True,
        "exact_retirement_history_and_no_orphans": True,
        "latest": states[-1],
        "latest_equals_fixed_final_tensors_and_metadata": True,
        "journal": artifact(store.journal),
        "pointer": artifact(store.pointer),
    }
    return report, [*assets, report["journal"], report["pointer"]]


def checked_accounting(plan, result):
    expected = training.lineage_counts(
        plan, result["training_examples"], result["optimizer_updates"]
    )
    require(result.get("lineage_accounting") == expected, "final lineage accounting changed")
    require(
        expected["parent_actual_exposures"] == 532
        and expected["inherited_exposures"] == expected["worker_exposures"] == 428
        and expected["rolled_back_accepted_exposures"] == 104
        and expected["selected_model_trajectory_exposures"] == 856
        and expected["actual_total_lineage_exposures"] == 960
        and expected["parent_actual_optimizer_updates"] == 134
        and expected["inherited_optimizer_updates"] == 108
        and expected["rolled_back_optimizer_updates"] == 26,
        "storage recovery lost its approved rollback boundaries",
    )
    return expected


def checked_final(training_path, result_path):
    original_view = SimpleNamespace(
        **{**vars(original), "verified_batch_evidence": polarity.verified_batch_evidence}
    )
    plan, result, assets = bind(
        resumed.checked_final,
        checked=training.checked,
        validate_inheritance=validate_inheritance,
        original=original_view,
    )(training_path, result_path)
    checked_accounting(plan, result)
    binding = plan["resume"]
    claim_path = (ROOT / binding["source_plan"]["path"]).with_name("word-manual-continuation.json")
    claim = json.loads(claim_path.read_text())
    require(
        claim.get("authorization") == binding["authorization"]
        and claim.get("manual_continuation_limit") == 1
        and claim.get("automatic_continuations") == 0
        and claim.get("rolled_back_accepted_exposures") == 104,
        "storage final lost explicit recovery ownership or authorization",
    )
    directory = result_path.resolve().parent
    steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    _, working_assets = checked_working_history(
        directory, plan, sha256(training_path), steps, result
    )
    return plan, result, [*assets, *binding["actual_source_evidence"], *working_assets]
