"""Independent complete-training audit after the second approved word recovery."""

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

import audit_c5_word_storage_lineage as template
from c5_word_http502_final import checked_final, checked_working_history
from c5_word_working_checkpoints import require
from terra_polarity_judge import bind
from terra_training_recovery import artifact


def independent_counts(plan, result, output):
    """Reconstruct all three attempts from immutable ledgers, including rollback."""
    binding = plan["resume"]
    ancestor = binding["ancestor_bindings"][0]
    paths = [
        ROOT / descriptor["path"]
        for descriptor in (
            ancestor["actual_source_steps"],
            ancestor["source_steps"],
            binding["actual_source_steps"],
        )
    ]
    for path, descriptor in zip(
        paths,
        (ancestor["actual_source_steps"], ancestor["source_steps"], binding["actual_source_steps"]),
        strict=True,
    ):
        require(artifact(path) == descriptor, "independent HTTP502 ancestor ledger changed")
    original, inherited, parent = [path.read_bytes().splitlines(keepends=True) for path in paths]
    require(
        len(original) == 134
        and len(inherited) == 108
        and len(parent) == 129
        and inherited == original[:108] == parent[:108],
        "independent HTTP502 prefix/rollback boundaries changed",
    )
    raw = (output / "steps.jsonl").read_bytes()
    require(
        raw.startswith(b"".join(parent)) and len(raw.splitlines()) == 216,
        "complete HTTP502 ledger lost saved512 prefix",
    )
    old, chosen, prior, final = [
        json.loads(lines[-1]) for lines in (original, inherited, parent, raw.splitlines())
    ]
    require(
        (old["training_examples"], old["optimizer_updates"]) == (532, 134)
        and (chosen["training_examples"], chosen["optimizer_updates"]) == (428, 108)
        and (prior["training_examples"], prior["optimizer_updates"]) == (512, 129),
        "independent HTTP502 source counters changed",
    )
    actual_parent_exposures = (
        old["training_examples"] + prior["training_examples"] - chosen["training_examples"]
    )
    actual_parent_updates = (
        old["optimizer_updates"] + prior["optimizer_updates"] - chosen["optimizer_updates"]
    )
    counts = {
        "parent_actual_exposures": actual_parent_exposures,
        "inherited_exposures": prior["training_examples"],
        "worker_exposures": final["training_examples"] - prior["training_examples"],
        "actual_total_lineage_exposures": actual_parent_exposures
        + final["training_examples"]
        - prior["training_examples"],
        "selected_model_trajectory_exposures": final["training_examples"],
        "rolled_back_accepted_exposures": old["training_examples"] - chosen["training_examples"],
        "parent_actual_optimizer_updates": actual_parent_updates,
        "inherited_optimizer_updates": prior["optimizer_updates"],
        "worker_optimizer_updates": final["optimizer_updates"] - prior["optimizer_updates"],
        "actual_total_lineage_optimizer_updates": actual_parent_updates
        + final["optimizer_updates"]
        - prior["optimizer_updates"],
        "selected_model_optimizer_updates": final["optimizer_updates"],
        "rolled_back_optimizer_updates": old["optimizer_updates"] - chosen["optimizer_updates"],
    }
    require(
        counts == result["lineage_accounting"]
        and counts["selected_model_trajectory_exposures"] == result["training_examples"] == 856
        and counts["actual_total_lineage_exposures"] == 960
        and counts["selected_model_optimizer_updates"] == result["optimizer_updates"]
        and counts["rolled_back_accepted_exposures"] == 104
        and counts["rolled_back_optimizer_updates"] == 26,
        "independent HTTP502 audit disagrees with actual accepted work",
    )
    return counts


def derived_fields_ast():
    definition = ast.parse(inspect.getsource(template.storage_audit_fields)).body[0]
    positions = [
        index
        for index, node in enumerate(definition.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "checked_working_history"
    ]
    require(len(positions) == 1, "cannot preserve original retained-working tensor audit")
    definition.body = (
        ast.parse(
            'binding = plan["resume"]\naccounting = independent_counts(plan, result, output)'
        ).body
        + definition.body[positions[0] :]
    )
    renamed = 0
    for node in ast.walk(definition):
        if isinstance(node, ast.Constant) and node.value == "parent_storage_failure_audit":
            node.value = "parent_HTTP502_failure_audit"
            renamed += 1
    require(renamed == 1, "cannot label the actual HTTP502 parent audit")
    return ast.fix_missing_locations(definition)


def storage_audit_fields(plan, result, output, evidence):
    namespace = {
        **template.storage_audit_fields.__globals__,
        "ROOT": ROOT,
        "checked_working_history": checked_working_history,
        "independent_counts": independent_counts,
    }
    exec(
        compile(
            ast.Module(body=[derived_fields_ast()], type_ignores=[]),
            "<HTTP502-lineage-fields>",
            "exec",
        ),
        namespace,
    )
    fields = namespace["storage_audit_fields"](plan, result, output, evidence)
    binding = plan["resume"]
    return {
        **fields,
        "original_storage_failure_audit": binding["ancestor_bindings"][0]["terminal_audit"],
        "manual_recovery_authorizations": [
            binding["ancestor_bindings"][0]["authorization"],
            binding["authorization"],
        ],
        "no_newly_rolled_back_accepted_work": True,
        "resource_reset_batches": [108, 129],
        "failed_HTTP502_batch_partial_evidence": binding["failed_batch_evidence"],
    }


derived_main_ast = template.derived_main_ast


def bound_main():
    return bind(
        template.bound_main,
        ROOT=ROOT,
        __file__=__file__,
        checked_final=checked_final,
        storage_audit_fields=storage_audit_fields,
    )()


def main():
    return bound_main()()


if __name__ == "__main__":
    main()
