"""Prepared one-time storage recovery; requires separate explicit authorization."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import c5_word_training as word
import train_c5_word_resume as resumed
import train_c5_word_semantic as original
from benchmark_fleurs_dev import sha256
from c5_word_storage_recovery import (
    checked_authorization,
    checked_binding,
    claim,
    inherit_prefix,
    make_binding,
    restore_committed,
)
from c5_word_working_checkpoints import WorkingCheckpoints
from launch_terra_visual_job import write
from terra_polarity_judge import Pacer, bind, score_training_batch
from terra_training_recovery import artifact, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "umi-studio-C5-polarity-storage-recovery/1"
SOURCES = (
    "train_c5_word_storage_recovery.py",
    "c5_word_storage_recovery.py",
    "c5_word_working_checkpoints.py",
    "c5_word_resume.py",
    "train_c5_word_resume.py",
    "roi_semantic_resume.py",
    "c5_word_resumed_final.py",
    "audit_c5_word_lineage.py",
    "c5_word_storage_final.py",
    "audit_c5_word_storage_lineage.py",
    "evaluate_c5_word_storage_final.py",
    "evaluate_c5_word_storage_words.py",
    "evaluate_c5_word_storage_sentences.py",
)


def lineage_counts(plan, examples, updates):
    parent = plan["resume"]
    return {
        "parent_actual_exposures": parent["actual_source_training_examples"],
        "inherited_exposures": parent["training_examples"],
        "worker_exposures": examples - parent["training_examples"],
        "actual_total_lineage_exposures": parent["actual_source_training_examples"]
        + examples
        - parent["training_examples"],
        "selected_model_trajectory_exposures": examples,
        "rolled_back_accepted_exposures": parent["rolled_back_accepted_exposures"],
        "parent_actual_optimizer_updates": parent["actual_source_optimizer_updates"],
        "inherited_optimizer_updates": parent["optimizer_updates"],
        "worker_optimizer_updates": updates - parent["optimizer_updates"],
        "actual_total_lineage_optimizer_updates": parent["actual_source_optimizer_updates"]
        + updates
        - parent["optimizer_updates"],
        "selected_model_optimizer_updates": updates,
        "rolled_back_optimizer_updates": parent["rolled_back_optimizer_updates"],
    }


def store_for(path, output):
    return WorkingCheckpoints(
        output,
        workspace=ROOT,
        plan_sha=sha256(path),
        storage_check=lambda prospective: word.storage(prospective, output=output),
        pair_reserve_bytes=word.STORAGE["checkpoint_pair_reserve_bytes"],
    )


def commit_working_checkpoint(path, output, model, optimizer, duty, examples, updates, rng):
    def writer(directory):
        save = bind(
            word.save_checkpoint,
            storage=lambda prospective=0, **_: word.storage(prospective, output=output),
        )
        plan = json.loads(path.read_text())
        visual = duty.run(
            lambda: save(
                model,
                optimizer,
                directory,
                path,
                plan["model_plan"]["reference"]["translator_checkpoint_sha256"],
                examples,
                updates,
                rng,
            )
        )
        return visual.with_suffix(".json")

    return store_for(path, output).commit(examples=examples, updates=updates, writer=writer)


def save_checkpoint(
    model, optimizer, output, path, reference_sha, examples, updates, rng, *, recovery=False
):
    if not recovery:
        return word.save_checkpoint(
            model, optimizer, output, path, reference_sha, examples, updates, rng
        )
    # A storage failure cannot require another half-GiB write just to preserve
    # a state already committed after the preceding accepted batch.
    state = store_for(path, output).latest()
    if (state["training_examples"], state["optimizer_updates"]) != (examples, updates):
        raise ValueError("latest durable working state is behind current in-memory counters")
    return ROOT / state["visual"]["path"]


def derived_fit_ast():
    definition = resumed.derived_fit_ast()
    inserted, final_report = 0, 0
    for statement in definition.body:
        if isinstance(statement, ast.Try):
            loop = statement.body[0]
            if not isinstance(loop, ast.For):
                continue
            for index, node in enumerate(loop.body):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "state_is_committed"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is True
                ):
                    loop.body.insert(
                        index + 1,
                        ast.parse(
                            "commit_working_checkpoint(path, output, gpu, optimizer, duty, "
                            "examples, updates, committed_rng)"
                        ).body[0],
                    )
                    inserted += 1
                    break
            for node in statement.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = node.value
                    if (
                        isinstance(call.func, ast.Name)
                        and call.func.id == "write"
                        and isinstance(call.args[0], ast.BinOp)
                        and isinstance(call.args[0].right, ast.Constant)
                        and call.args[0].right.value == "result.json"
                    ):
                        report = call.args[1]
                        report.keys.append(ast.Constant("lineage_accounting"))
                        report.values.append(
                            ast.parse("lineage_counts(plan, examples, updates)").body[0].value
                        )
                        final_report += 1
    if inserted != 1 or final_report != 1:
        raise ValueError("storage recovery could not identify the exact checkpoint boundaries")
    return ast.fix_missing_locations(definition)


def bound_fit():
    definition = derived_fit_ast()
    namespace = {
        **original.fit.__globals__,
        "checked": checked,
        "restore_committed": restore_committed,
        "prior_checkpoints": resumed.prior_checkpoints,
        "prior_order": resumed.prior_order,
        "RecoveryPacer": Pacer,
        "score_batch": score_training_batch,
        "commit_working_checkpoint": commit_working_checkpoint,
        "save_checkpoint": save_checkpoint,
        "lineage_counts": lineage_counts,
    }
    exec(
        compile(
            ast.Module(body=[definition], type_ignores=[]),
            "<authorized-word-storage-recovery>",
            "exec",
        ),
        namespace,
    )
    return namespace["fit"]


def specification(path, source_path, checkpoint, audit_path, authorization_path):
    binding = make_binding(source_path, checkpoint, audit_path, authorization_path)
    source = original_plan(source_path)
    sources = dict(source["source_sha256"])
    # All final-audit/evaluation adapters must exist and be pinned before this
    # can become an executable freeze. Missing components stop preparation.
    for name in SOURCES:
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        **source,
        "schema": SCHEMA,
        "source_sha256": sources,
        "resume": binding,
        "initialization": "explicitly approved428 checkpoint from original public-C5 lineage",
        "manual_continuations_authorized": 1,
        "single_optimizer_execution_per_proof": False,
        "explicit_one_time_storage_recovery_exception": binding["authorization"],
        "automatic_further_continuations": 0,
        "working_checkpoint_policy": {
            "after_each_accepted_batch": True,
            "retained_working_pairs": 2,
            "all_epoch_pairs_and_working_receipts_retained": True,
        },
        "lineage_audit_source": {
            "path": str(path.with_name("terminal-audit.py").relative_to(ROOT)),
            "sha256": sha256(ROOT / "tools/audit_c5_word_storage_lineage.py"),
        },
    }


def original_plan(path):
    import train_c5_polarity_words

    return train_c5_polarity_words.checked(path)


def checked(path):
    path = path.resolve()
    plan = json.loads(path.read_text())
    binding = plan["resume"]
    expected = specification(
        path,
        ROOT / binding["source_plan"]["path"],
        ROOT / binding["visual"]["path"],
        ROOT / binding["terminal_audit"]["path"],
        ROOT / binding["authorization"]["path"],
    )
    if (
        plan != expected
        or artifact(ROOT / plan["lineage_audit_source"]["path"]) != plan["lineage_audit_source"]
    ):
        raise ValueError(
            "storage recovery changed its approved source, recipe or final-audit binding"
        )
    verify_snapshot(path, plan)
    return plan


def freeze(path, source_path, checkpoint, audit_path, authorization_path):
    from c5_experiment_sources import archive_sources

    checked_authorization(authorization_path)
    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("storage recovery requires a fresh owned plan")
    word.storage(word.STORAGE["checkpoint_pair_reserve_bytes"])
    plan = specification(
        path,
        source_path.resolve(),
        checkpoint.resolve(),
        audit_path.resolve(),
        authorization_path.resolve(),
    )
    path.parent.mkdir(parents=True)
    with path.with_name("terminal-audit.py").open("xb") as stream:
        stream.write((ROOT / "tools/audit_c5_word_storage_lineage.py").read_bytes())
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan": artifact(path),
                "inherited_exposures": 428,
                "rolled_back_exposures": 104,
                "remaining_exposures": 428,
                "actual_lineage_exposures_if_complete": 960,
            }
        ),
        flush=True,
    )


def fit(path, output, plan):
    checked_binding(plan["resume"])
    write(output / "resume-spacing.json", resumed.source_cooldown(plan["resume"]))
    claim(path, output, plan)
    inheritance = inherit_prefix(output, plan)
    store_for(path, output).seed(ROOT / inheritance["epochs"][-1]["commit"]["path"])
    return bound_fit()(path, output, plan)


def run(path, output):
    return bind(original.run, checked=checked, fit=fit)(path, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "freeze",
        "source-plan",
        "checkpoint",
        "terminal-audit",
        "authorization",
        "plan",
        "output",
    ):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.freeze and all(
        (args.source_plan, args.checkpoint, args.terminal_audit, args.authorization)
    ):
        checked_authorization(args.authorization)
        original.torch.set_num_interop_threads(1)
        freeze(
            args.freeze, args.source_plan, args.checkpoint, args.terminal_audit, args.authorization
        )
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output:
        checked(args.plan)
        original.torch.set_num_interop_threads(1)
        run(args.plan, args.output)
    else:
        parser.error(
            "prepare only after explicit training approval, or run one fully frozen recovery"
        )


if __name__ == "__main__":
    main()
