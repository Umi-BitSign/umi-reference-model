"""Complete856 verification after an explicitly approved second continuation."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import c5_polarity_word_final as polarity
import c5_word_final as original
import c5_word_resumed_final as resumed
import c5_word_storage_final as template
import train_c5_word_http502_recovery as training
from benchmark_fleurs_dev import sha256
from c5_word_http502_recovery import validate_inheritance
from c5_word_working_checkpoints import require
from terra_polarity_judge import bind

ROOT = Path(__file__).resolve().parents[1]


def derived_working_history_ast():
    source = inspect.getsource(template.checked_working_history)
    before = 'json.loads((directory / "inheritance.json").read_text())["epochs"][-1]'
    after = 'json.loads((directory / "inheritance.json").read_text())["working_seed"]'
    require(
        source.count(before) == 1,
        "cannot bind the separate working seed to original history checks",
    )
    return ast.parse(source.replace(before, after)).body[0]


def checked_working_history(directory, plan, plan_sha, steps, result):
    namespace = {**template.checked_working_history.__globals__, "ROOT": ROOT}
    exec(
        compile(
            ast.Module(body=[derived_working_history_ast()], type_ignores=[]),
            "<HTTP502-working-history>",
            "exec",
        ),
        namespace,
    )
    return namespace["checked_working_history"](directory, plan, plan_sha, steps, result)


def derived_verify_steps_ast():
    definition = ast.parse(inspect.getsource(original.verify_steps)).body[0]
    expected = ast.parse(
        'if resets and ("resume" not in plan or resets != {plan["resume"]["completed_batches"]}):\n'
        '    raise ValueError("word resources may reset only at the authenticated worker handoff")'
    ).body[0]
    require(
        ast.dump(definition.body[1], include_attributes=False)
        == ast.dump(expected, include_attributes=False),
        "cannot identify the original single-handoff resource guard",
    )
    definition.body[1] = ast.parse(
        'if ("resume" not in plan or resets != {108, 129}\n'
        '    or plan["resume"]["completed_batches"] != 129\n'
        '    or plan["resume"]["resource_reset_batches"] != [108, 129]):\n'
        '    raise ValueError("HTTP502 resource reset boundaries changed")'
    ).body[0]
    return ast.fix_missing_locations(definition)


def verify_steps(plan, steps, result, *, resource_reset_batches=()):
    require(
        plan["resume"]["resource_reset_batches"] == [108, 129]
        and set(resource_reset_batches).issubset({129}),
        "HTTP502 resource reset boundaries changed",
    )
    namespace = dict(original.verify_steps.__globals__)
    exec(
        compile(
            ast.Module(body=[derived_verify_steps_ast()], type_ignores=[]),
            "<HTTP502-resource-guard>",
            "exec",
        ),
        namespace,
    )
    return namespace["verify_steps"](plan, steps, result, resource_reset_batches={108, 129})


def checked_accounting(plan, result):
    expected = training.lineage_counts(
        plan, result["training_examples"], result["optimizer_updates"]
    )
    require(result.get("lineage_accounting") == expected, "HTTP502 final work accounting changed")
    require(
        expected["parent_actual_exposures"] == 616
        and expected["inherited_exposures"] == 512
        and expected["worker_exposures"] == 344
        and expected["selected_model_trajectory_exposures"] == 856
        and expected["actual_total_lineage_exposures"] == 960
        and expected["rolled_back_accepted_exposures"] == 104
        and expected["parent_actual_optimizer_updates"] == 155
        and expected["inherited_optimizer_updates"] == 129
        and expected["rolled_back_optimizer_updates"] == 26
        and plan["resume"]["newly_rolled_back_accepted_exposures"] == 0
        and plan["resume"]["newly_rolled_back_optimizer_updates"] == 0,
        "HTTP502 final lost original rollback or saved512 boundaries",
    )
    return expected


def checked_final(training_path, result_path):
    original_view = SimpleNamespace(
        **{
            **vars(original),
            "verified_batch_evidence": polarity.verified_batch_evidence,
            "verify_steps": verify_steps,
        }
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
        and claim.get("manual_continuation_limit")
        == claim.get("additional_manual_continuation_limit")
        == 1
        and claim.get("manual_continuation_number") == 2
        and claim.get("automatic_continuations") == 0
        and claim.get("rolled_back_accepted_exposures") == 104
        and claim.get("newly_rolled_back_accepted_exposures") == 0,
        "HTTP502 final lost its explicit second-recovery claim",
    )
    directory = result_path.resolve().parent
    steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    _, working_assets = checked_working_history(
        directory, plan, sha256(training_path), steps, result
    )
    return plan, result, [*assets, *binding["actual_source_evidence"], *working_assets]
