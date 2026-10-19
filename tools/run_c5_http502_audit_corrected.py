"""Disclosed one-statement runtime correction for the frozen HTTP502 auditor.

Frozen training/evaluation programs and model/score bytes remain untouched.
The restored ledger read is the parent's original local assignment.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from fetch_c5_broad_how2sign import ROOT, artifact, read, require, sha256, write
from launch_terra_visual_job import identity, now

TRAINING = ROOT / "runs/c5-polarity-word-http502-recovery-v1/plan.json"
OUTPUT = ROOT / "checkpoints/c5-polarity-word-http502-recovery-v1"
PREPARATION = ROOT / "runs/c5-http502-audit-correction-preparation-v1/validation.json"


def corrected_ast(original, parent):
    definition = copy.deepcopy(original)
    assignments = [n for n in parent.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "steps" for t in n.targets)]
    expected = ast.parse('steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]').body[0]
    require(len(assignments) == 1 and ast.dump(assignments[0], include_attributes=False)
            == ast.dump(expected, include_attributes=False), "parent ledger assignment changed")
    require(not any(isinstance(n, ast.Name) and n.id == "steps" and isinstance(n.ctx, ast.Store)
                    for n in ast.walk(definition)), "HTTP502 audit already defines its ledger; refuse another correction")
    positions = [i for i, node in enumerate(definition.body) if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "checked_working_history"]
    require(positions == [2], "HTTP502 audit retained-history boundary changed")
    definition.body.insert(2, copy.deepcopy(assignments[0]))
    check = copy.deepcopy(definition)
    del check.body[2]
    require(ast.dump(check, include_attributes=False) == ast.dump(original, include_attributes=False),
            "correction changed another audit statement")
    return ast.fix_missing_locations(definition)


def patch(module):
    if hasattr(module, "_umi_original_derived_fields_ast"):
        return
    original = module.derived_fields_ast
    parent = ast.parse(inspect.getsource(module.template.storage_audit_fields)).body[0]
    corrected_ast(original(), parent)  # Refuse an unexpected generator before mutation.

    def generated():
        return corrected_ast(original(), parent)

    module._umi_original_derived_fields_ast = original
    module.derived_fields_ast = generated


def checked_preparation():
    p = read(PREPARATION)
    require(p["schema"] == "umi-studio-http502-audit-runtime-correction-preparation/1"
            and p["tests_passed"] >= 3 and p["only_scientific_change"] == "restore_parent_local_steps_assignment",
            "audit correction requires its reviewed qualification")
    for name, digest in p["source_sha256"].items():
        path = ROOT / name
        require(path.resolve() == path and path.is_relative_to(ROOT)
                and sha256(path) == digest, "audit correction qualification source changed")
    return artifact(PREPARATION)


def install():
    plan = read(TRAINING)
    require(all(sha256(ROOT / name) == digest for name, digest in plan["source_sha256"].items()),
            "frozen training source changed")
    import audit_c5_word_http502_lineage as module

    require(sha256(Path(module.__file__)) == plan["lineage_audit_source"]["sha256"]
            and artifact(ROOT / plan["lineage_audit_source"]["path"]) == plan["lineage_audit_source"],
            "correction does not match the original frozen auditor")
    patch(module)
    return module


def forward_main(function, arguments):
    class Parser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            return super().parse_args(arguments if args is None else args, namespace)

    proxy = SimpleNamespace(**{**vars(argparse), "ArgumentParser": Parser})
    bound = FunctionType(function.__code__, {**function.__globals__, "argparse": proxy},
                         function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound()


def verify_audit(path):
    value = read(path)
    correction = value["runtime_correction"]
    require(correction["entrypoint"] == artifact(Path(__file__))
            and correction["qualification"] == checked_preparation(),
            "corrected terminal audit lost its actual supplemental program")
    for key in ("raw_audit", "invocation", "corrected_derived_fields", "completed_steps"):
        descriptor = correction[key]
        require(artifact(ROOT / descriptor["path"]) == descriptor, "corrected audit evidence changed")
    raw = read(ROOT / correction["raw_audit"]["path"])
    require(value == {**raw, "runtime_correction": correction}
            and raw["audit_source"] == read(TRAINING)["lineage_audit_source"]
            and raw["complete"] is True and raw["training_examples"] == 856,
            "corrected audit reclassified or changed original audited values")
    module = install()
    actual_ast = ast.parse((ROOT / correction["corrected_derived_fields"]["path"]).read_text()).body[0]
    require(ast.dump(actual_ast, include_attributes=False)
            == ast.dump(module.derived_fields_ast(), include_attributes=False),
            "corrected audit changed another generated check")
    require(correction["completed_steps"] == artifact(OUTPUT / "steps.jsonl"),
            "corrected audit was rebound to another completed ledger")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "evaluate", "export", "parity", "full-runtime"), required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    record_path = args.record.absolute()
    require(record_path.resolve() == record_path and record_path.is_relative_to(ROOT / "runs")
            and not record_path.exists(), "correction invocation needs a fresh owned record")
    qualification = checked_preparation()
    module = install()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write(record_path, {"mode": args.mode, "actual_entrypoint": artifact(Path(__file__)),
                        "actual_argv": [sys.executable, *sys.argv], "requested_arguments": arguments,
                        "worker": identity(os.getpid(), [sys.executable, *sys.argv]),
                        "qualification": qualification, "original_auditor": read(TRAINING)["lineage_audit_source"],
                        "scientific_change": "restore_parent_local_steps_assignment",
                        "started_at": now(), "training_or_grade_values_changed": False})
    code = 1
    try:
        if args.mode == "audit":
            requested = Path(arguments[arguments.index("--output") + 1]).absolute()
            require(requested.resolve() == requested and requested.is_relative_to(ROOT / "runs")
                    and not requested.exists(), "corrected audit needs a fresh owned output")
            original = ROOT / read(TRAINING)["lineage_audit_source"]["path"]
            spec = importlib.util.spec_from_file_location("archived_http502_corrected_execution", original)
            archived = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(archived)
            patch(archived)
            derived = record_path.with_name("corrected-derived-fields.py")
            with derived.open("x") as f:
                f.write(ast.unparse(archived.derived_fields_ast()) + "\n")
            raw_path = record_path.with_name("raw-terminal-audit.json")
            actual_arguments = list(arguments)
            actual_arguments[actual_arguments.index("--output") + 1] = str(raw_path)
            forward_main(archived.bound_main(), actual_arguments)
            raw = read(raw_path)
            correction = {"entrypoint": artifact(Path(__file__)), "qualification": qualification,
                          "invocation": artifact(record_path), "raw_audit": artifact(raw_path),
                          "corrected_derived_fields": artifact(derived),
                          "completed_steps": artifact(OUTPUT / "steps.jsonl"),
                          "only_scientific_change": "restore_parent_local_steps_assignment"}
            write(requested, {**raw, "runtime_correction": correction})
            verify_audit(requested)
            print(json.dumps({"corrected_audit": artifact(requested), "training_examples": raw["training_examples"],
                              "optimizer_updates": raw["optimizer_updates"]}), flush=True)
        elif args.mode == "evaluate":
            audit = Path(arguments[arguments.index("--terminal-audit") + 1]).absolute()
            verify_audit(audit)
            import evaluate_c5_word_http502_final as evaluation
            evaluation.main(arguments)
        elif args.mode == "export":
            audit = Path(arguments[arguments.index("--terminal-audit") + 1]).absolute()
            verify_audit(audit)
            import export_c5_native112_visual as exporter
            exporter.main(arguments)
        elif args.mode == "parity":
            export = read(Path(arguments[arguments.index("--export") + 1]).absolute())
            descriptor = export["evidence"]["terminal_audit"]
            require(artifact(ROOT / descriptor["path"]) == descriptor, "export audit binding changed")
            verify_audit(ROOT / descriptor["path"])
            import compare_c5_native112_export as parity
            parity.main(arguments)
        else:
            export = read(Path(arguments[arguments.index("--export") + 1]).absolute())
            descriptor = export["evidence"]["terminal_audit"]
            require(artifact(ROOT / descriptor["path"]) == descriptor, "export audit binding changed")
            verify_audit(ROOT / descriptor["path"])
            import probe_c5_native112_full_runtime as full
            forward_main(full.main, arguments)
        code = 0
    except BaseException as error:
        write(record_path.with_name(record_path.stem + "-failure.json"),
              {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write(record_path.with_name(record_path.stem + "-exit.json"),
              {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


if __name__ == "__main__":
    main()

