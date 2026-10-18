"""Archived independent full-training auditor for one approved storage recovery.

Preserves the original auditor's deterministic order, calibration, raw rewards,
350-tensor epochs and Adam/RNG checks, then adds recovery checks before writing.
"""

# ruff: noqa: E402

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[2]
    if Path(__file__).parent.parent.name == "runs"
    else Path(__file__).resolve().parents[1]
)
sys.path.insert(0, str(ROOT / "tools"))

import audit_c5_word_lineage as original
import torch
from c5_word_storage_final import checked_final, checked_working_history
from c5_word_working_checkpoints import require
from safetensors import safe_open
from terra_training_recovery import artifact


def storage_audit_fields(plan, result, output, evidence):
    binding = plan["resume"]
    parent = ROOT / binding["actual_source_steps"]["path"]
    prefix = ROOT / binding["source_steps"]["path"]
    require(artifact(parent) == binding["actual_source_steps"], "actual parent ledger changed")
    require(artifact(prefix) == binding["source_steps"], "selected parent prefix changed")
    raw = parent.read_bytes().splitlines(keepends=True)
    chosen = prefix.read_bytes()
    require(
        len(raw) == 134 and chosen == b"".join(raw[:108]),
        "independent recovery audit lost the actual parent versus chosen prefix",
    )
    parent_last, inherited_last = json.loads(raw[-1]), json.loads(raw[107])
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    require((output / "steps.jsonl").read_bytes().startswith(chosen), "inherited prefix changed")
    selected, worker = steps[-1], result["optimizer_updates"] - inherited_last["optimizer_updates"]
    accounting = {
        "parent_actual_exposures": parent_last["training_examples"],
        "inherited_exposures": inherited_last["training_examples"],
        "worker_exposures": selected["training_examples"] - inherited_last["training_examples"],
        "actual_total_lineage_exposures": parent_last["training_examples"]
        + selected["training_examples"]
        - inherited_last["training_examples"],
        "selected_model_trajectory_exposures": selected["training_examples"],
        "rolled_back_accepted_exposures": parent_last["training_examples"]
        - inherited_last["training_examples"],
        "parent_actual_optimizer_updates": parent_last["optimizer_updates"],
        "inherited_optimizer_updates": inherited_last["optimizer_updates"],
        "worker_optimizer_updates": worker,
        "actual_total_lineage_optimizer_updates": parent_last["optimizer_updates"] + worker,
        "selected_model_optimizer_updates": selected["optimizer_updates"],
        "rolled_back_optimizer_updates": parent_last["optimizer_updates"]
        - inherited_last["optimizer_updates"],
    }
    require(
        accounting == result["lineage_accounting"]
        and accounting["selected_model_trajectory_exposures"] == 856
        and accounting["actual_total_lineage_exposures"] == 960
        and accounting["rolled_back_accepted_exposures"] == 104,
        "independent recovery audit disagrees with truthful work accounting",
    )
    working, _ = checked_working_history(output, plan, result["plan_sha256"], steps, result)
    scope = json.loads((ROOT / plan["model_plan"]["visual_scope"]["path"]).read_text())
    shapes = {"vision." + row["parameter"]: row["shape"] for row in scope["parameters"]}
    states = [
        event["state"]
        for line in (output / "working-checkpoints.jsonl").read_text().splitlines()
        if (event := json.loads(line))["kind"] == "committed"
        and not event["state"]["inherited_seed"]
    ][-2:]
    retained = []
    for state in states:
        for kind in ("visual", "optimizer"):
            path = ROOT / state[kind]["path"]
            require(artifact(path) == state[kind], "retained working bytes changed")
            with safe_open(str(path), framework="pt", device="cpu") as saved:
                metadata = saved.metadata()
                require(
                    metadata.get("plan_sha256") == result["plan_sha256"]
                    and metadata.get("update") == str(state["training_examples"]),
                    "retained working tensor provenance changed",
                )
                if kind == "visual":
                    require(
                        set(saved.keys()) == set(shapes)
                        and len(shapes) == 350
                        and metadata.get("optimizer_updates") == str(state["optimizer_updates"])
                        and metadata.get("fixed_winner_checkpoint")
                        == plan["model_plan"]["reference"]["translator_checkpoint_sha256"],
                        "retained working visual topology or provenance changed",
                    )
                    for name, shape in shapes.items():
                        value = saved.get_tensor(name)
                        require(
                            list(value.shape) == shape
                            and value.dtype == torch.float32
                            and bool(torch.isfinite(value).all()),
                            "retained working visual contains invalid values",
                        )
        retained.append(
            {
                "state": state,
                **original.optimizer_inventory(
                    ROOT / state["optimizer"]["path"],
                    shapes,
                    state["optimizer_updates"],
                    plan["recipe"],
                ),
            }
        )
    require(len(retained) == 2, "storage final omitted a retained working pair")
    require(
        all(item in evidence for item in binding["actual_source_evidence"]),
        "audit evidence omitted rolled-back accepted parent work",
    )
    return {
        "lineage_accounting": accounting,
        "manual_recovery_authorization": binding["authorization"],
        "parent_storage_failure_audit": binding["terminal_audit"],
        "judge_protocol": plan["judge_protocol"],
        "working_checkpoint_history": working,
        "retained_working_inventory": retained,
        "discarded_parent_work_preserved_and_not_reused": True,
        "original_public_initialization_with_explicit_rollback": True,
        "further_automatic_continuations": 0,
    }


def derived_main_ast():
    definition = ast.parse(inspect.getsource(original.main)).body[0]
    matches = 0
    for index, statement in enumerate(definition.body):
        if isinstance(statement, ast.With) and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and isinstance(item.context_expr.func.value, ast.Name)
            and item.context_expr.func.value.id == "target"
            and item.context_expr.func.attr == "open"
            for item in statement.items
        ):
            definition.body.insert(
                index,
                ast.parse(
                    "audit.update(storage_audit_fields(plan, result, output, evidence))"
                ).body[0],
            )
            matches += 1
            break
    require(matches == 1, "cannot bind recovery checks before the independent audit write")
    return ast.fix_missing_locations(definition)


def bound_main():
    namespace = {
        **original.main.__globals__,
        "ROOT": ROOT,
        "__file__": __file__,
        "checked_final": checked_final,
        "storage_audit_fields": storage_audit_fields,
    }
    exec(
        compile(
            ast.Module(body=[derived_main_ast()], type_ignores=[]),
            "<storage-lineage-auditor>",
            "exec",
        ),
        namespace,
    )
    return namespace["main"]


def main():
    return bound_main()()


if __name__ == "__main__":
    main()
