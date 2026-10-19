"""Generated-auditor regression; stops before tensor/model work."""
from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import sys
from pathlib import Path
from types import FunctionType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_c5_http502_audit_corrected as correction
import audit_c5_word_http502_lineage as auditor


def definition():
    original = getattr(auditor, "_umi_original_derived_fields_ast", auditor.derived_fields_ast)
    parent = ast.parse(inspect.getsource(auditor.template.storage_audit_fields)).body[0]
    return original(), parent


def test_actual_generated_audit_reproduces_missing_local_and_corrected_function_passes_real_ledger(tmp_path):
    original, parent = definition()
    ledger = [{"synthetic": i} for i in range(3)]
    (tmp_path / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
    seen = []

    class ReachedRetainedChecks(Exception):
        pass

    def history(output, plan, digest, steps, result):
        seen.append(steps)
        assert output == tmp_path and digest == "synthetic-plan"
        raise ReachedRetainedChecks()

    namespace = {**auditor.template.storage_audit_fields.__globals__,
                 "independent_counts": lambda *args: {}, "checked_working_history": history}
    namespace.pop("steps", None)
    exec(compile(ast.Module(body=[original], type_ignores=[]), "<unchanged-failed-audit>", "exec"), namespace)
    with pytest.raises(NameError, match="steps"):
        namespace["storage_audit_fields"]({"resume": {}}, {"plan_sha256": "synthetic-plan"}, tmp_path, [])
    assert seen == []
    fixed = correction.corrected_ast(original, parent)
    exec(compile(ast.Module(body=[fixed], type_ignores=[]), "<restored-local-audit>", "exec"), namespace)
    with pytest.raises(ReachedRetainedChecks):
        namespace["storage_audit_fields"]({"resume": {}}, {"plan_sha256": "synthetic-plan"}, tmp_path, [])
    assert seen == [ledger]


def test_only_parent_steps_assignment_is_added_and_every_other_check_is_identical():
    original, parent = definition()
    before = ast.dump(original, include_attributes=False)
    fixed = correction.corrected_ast(original, parent)
    assignment = fixed.body.pop(2)
    assert isinstance(assignment, ast.Assign) and assignment.targets[0].id == "steps"
    assert ast.dump(fixed, include_attributes=False) == before
    assert ast.dump(original, include_attributes=False) == before


def test_changed_or_already_corrected_generator_is_refused():
    original, parent = definition()
    fixed = correction.corrected_ast(original, parent)
    with pytest.raises(ValueError, match="already defines"):
        correction.corrected_ast(fixed, parent)
    changed = copy.deepcopy(parent)
    step = next(n for n in changed.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "steps" for t in n.targets))
    step.value = ast.Constant(value=None)
    with pytest.raises(ValueError, match="parent ledger"):
        correction.corrected_ast(original, changed)


def test_forwarded_argparse_inputs_do_not_relabel_actual_entrypoint_argv(monkeypatch):
    actual = ["actual-wrapper.py", "--mode", "synthetic", "--", "--value", "requested"]
    monkeypatch.setattr(sys, "argv", actual)

    def program():
        parser = argparse.ArgumentParser()
        parser.add_argument("--value", required=True)
        return parser.parse_args().value, list(sys.argv)

    # program's globals deliberately include the real argparse module.
    result = correction.forward_main(program, ["--value", "requested"])
    assert result == ("requested", actual)
    assert sys.argv == actual

