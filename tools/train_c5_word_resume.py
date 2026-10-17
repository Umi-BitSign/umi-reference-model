"""One reviewed word continuation with the original complete per-batch numerical loop."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from types import FunctionType

import train_c5_word_semantic as original
from benchmark_fleurs_dev import sha256
from c5_word_resume import (
    claim,
    inherit_prefix,
    make_binding,
    prior_checkpoints,
    prior_order,
    restore_committed,
    source_cooldown,
)
from launch_terra_visual_job import write
from terra_training_recovery import artifact, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
CONTROL_SOURCES = (
    "train_c5_word_resume.py",
    "c5_word_resume.py",
    "c5_word_resumed_final.py",
    "audit_c5_word_lineage.py",
    "roi_semantic_resume.py",
    "c5_word_epoch_audit.py",
    "c5_word_final.py",
)


def derived_fit_ast():
    definition = ast.parse(inspect.getsource(original.fit)).body[0]
    changes = {"counters": 0, "order": 0, "run": 0, "skip": 0}
    for index, statement in enumerate(definition.body):
        if isinstance(statement, ast.Assign):
            names = [target.id for target in statement.targets if isinstance(target, ast.Name)]
            if names == ["examples", "updates"]:
                definition.body[index] = ast.parse(
                    "examples, updates = restore_committed(plan, gpu, optimizer, duty)"
                ).body[0]
                changes["counters"] += 1
            elif len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Tuple):
                if [node.id for node in statement.targets[0].elts] == ["saved", "observed_order"]:
                    definition.body[index] = ast.parse(
                        "saved, observed_order = prior_checkpoints(output), prior_order(plan)"
                    ).body[0]
                    changes["order"] += 1
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if isinstance(call.func, ast.Name) and call.func.id == "write":
                if isinstance(call.args[1], ast.Dict):
                    report = call.args[1]
                    for position, key in enumerate(report.keys):
                        if isinstance(key, ast.Constant) and key.value in (
                            "training_examples",
                            "optimizer_updates",
                        ):
                            report.values[position] = ast.Name(
                                id="examples" if key.value == "training_examples" else "updates",
                                ctx=ast.Load(),
                            )
                    extra = (
                        ast.parse(
                            "{'resume': plan['resume'], "
                            "'inheritance': artifact(output / 'inheritance.json'), "
                            "'restored_visual_state_sha256': "
                            "plan['resume']['starting_visual_state_sha256']}"
                        )
                        .body[0]
                        .value
                    )
                    report.keys.extend(extra.keys)
                    report.values.extend(extra.values)
                    changes["run"] += 1
        elif isinstance(statement, ast.Try):
            loop = statement.body[0]
            if isinstance(loop, ast.For):
                loop.body.insert(
                    0,
                    ast.parse(
                        "if batch_number < plan['resume']['completed_batches']: continue"
                    ).body[0],
                )
                changes["skip"] += 1
    if changes != {key: 1 for key in changes}:
        raise ValueError("word resume cannot identify exact original control statements")
    return ast.fix_missing_locations(definition)


def bound_fit():
    definition = derived_fit_ast()
    namespace = {
        **original.fit.__globals__,
        "checked": checked,
        "restore_committed": restore_committed,
        "prior_checkpoints": prior_checkpoints,
        "prior_order": prior_order,
    }
    exec(
        compile(
            ast.Module(body=[definition], type_ignores=[]), "<derived-word-continuation>", "exec"
        ),
        namespace,
    )
    return namespace["fit"]


def specification(path, source_path, checkpoint, audit_path):
    source = original.checked(source_path)
    binding = make_binding(source_path, checkpoint, audit_path)
    sources = dict(source["source_sha256"])
    for name in CONTROL_SOURCES:
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        **source,
        "source_sha256": sources,
        "resume": binding,
        "lineage_audit_source": {
            "path": str(path.with_name("terminal-audit.py").relative_to(ROOT)),
            "sha256": sha256(ROOT / "tools/audit_c5_word_lineage.py"),
        },
    }


def checked(path):
    path = path.resolve()
    plan = json.loads(path.read_text())
    binding = plan["resume"]
    expected = specification(
        path,
        ROOT / binding["source_plan"]["path"],
        ROOT / binding["visual"]["path"],
        ROOT / binding["terminal_audit"]["path"],
    )
    if (
        plan != expected
        or artifact(ROOT / plan["lineage_audit_source"]["path"]) != plan["lineage_audit_source"]
    ):
        raise ValueError("word continuation changed its same-recipe lineage or reviewed controls")
    verify_snapshot(path, plan)
    return plan


def freeze(path, source_path, checkpoint, audit_path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("word continuation requires a fresh owned plan")
    plan = specification(path, source_path.resolve(), checkpoint.resolve(), audit_path.resolve())
    original.storage(original.STORAGE["checkpoint_pair_reserve_bytes"])
    path.parent.mkdir(parents=True)
    with path.with_name("terminal-audit.py").open("xb") as stream:
        stream.write((ROOT / "tools/audit_c5_word_lineage.py").read_bytes())
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan": artifact(path),
                "source_exposures": plan["resume"]["training_examples"],
                "remaining_exposures": plan["planned_training_examples"]
                - plan["resume"]["training_examples"],
                "manual_continuation_limit": 1,
            }
        ),
        flush=True,
    )


def fit(path, output, plan):
    write(output / "resume-spacing.json", source_cooldown(plan["resume"]))
    claim(path, output, plan)
    inherit_prefix(output, plan)
    return bound_fit()(path, output, plan)


def run(path, output):
    function = original.run
    bound = FunctionType(
        function.__code__,
        {**function.__globals__, "checked": checked, "fit": fit},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    return bound(path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "source-plan", "checkpoint", "terminal-audit", "plan", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.freeze and args.source_plan and args.checkpoint and args.terminal_audit:
        freeze(args.freeze, args.source_plan, args.checkpoint, args.terminal_audit)
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output:
        run(args.plan, args.output)
    else:
        parser.error("freeze from reviewed committed stop, or run one existing continuation plan")


if __name__ == "__main__":
    main()
