from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.numpy import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_c5_native112_visual as exporter
from test_c5_native112_export_values import fixture as arrays_fixture


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def gate_fixture():
    initial, audit = {"synthetic_final": True}, {"synthetic_audit": True}
    evaluations = {
        kind: {
            "kind": kind,
            "semantic_decline_recorded_for_evaluation_only": False,
            "decision_values": {
                "calibration_passed": True,
                "passes_semantic_research_gate": True,
                "stop_for_semantic_decline": False,
            },
        }
        for kind in ("word", "sentence")
    }
    complete = {
        "complete": True,
        "initialization": initial,
        "terminal_audit": audit,
        "evaluations": evaluations,
        "optimizer_updates": 0,
        "automatic_retries": 0,
        "automatic_training_continuation": False,
        "promotion_authorized": False,
    }
    handoff = {
        "initialization": initial,
        "terminal_audit": audit,
        "complete_evaluation_order": ["word96", "sentence34"],
        "stop_on_failed_calibration": True,
        "cross_pipeline_minimum_start_gap_seconds": 60,
        "automatic_retries": 0,
        "automatic_training_continuation": False,
        "promotion_authorized": False,
    }
    return complete, handoff, initial, audit, evaluations


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "incomplete",
        "other_final",
        "missing_sentence",
        "calibration",
        "no_gain",
        "decline",
        "wrong_kind",
        "handoff_order",
        "retry",
        "auto_training",
        "promotion",
    ],
)
def test_export_requires_both_same_final_complete_calibrated_semantic_gates(fault):
    # Receipt-logic fixture only, not authentication of a real model or grade.
    complete, handoff, initial, audit, evaluations = gate_fixture()
    if fault == "incomplete":
        complete["complete"] = False
    elif fault == "other_final":
        complete["initialization"] = {"other": True}
    elif fault == "missing_sentence":
        del evaluations["sentence"]
    elif fault == "calibration":
        evaluations["word"]["decision_values"]["calibration_passed"] = False
    elif fault == "no_gain":
        evaluations["sentence"]["decision_values"]["passes_semantic_research_gate"] = False
    elif fault == "decline":
        evaluations["word"]["decision_values"]["stop_for_semantic_decline"] = True
    elif fault == "wrong_kind":
        evaluations["word"]["kind"] = "sentence"
    elif fault == "handoff_order":
        handoff["complete_evaluation_order"].reverse()
    elif fault == "retry":
        handoff["automatic_retries"] = 1
    elif fault == "auto_training":
        complete["automatic_training_continuation"] = True
    elif fault == "promotion":
        complete["promotion_authorized"] = True
    if fault is None:
        exporter.checked_coordinator(complete, handoff, initial, audit, evaluations)
    else:
        with pytest.raises(ValueError):
            exporter.checked_coordinator(complete, handoff, initial, audit, evaluations)


@pytest.fixture
def small_export(tmp_path, monkeypatch):
    """Tiny file transaction; actual complete-training authentication is stubbed."""
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    monkeypatch.setattr(
        exporter.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3)
    )
    originals, trained, shapes = arrays_fixture()
    monkeypatch.setattr(exporter, "qualified_shapes", lambda _: copy.deepcopy(shapes))
    plan_path = tmp_path / "runs/synthetic/plan.json"
    result_path = tmp_path / "checkpoints/synthetic/result.json"
    audit = tmp_path / "runs/synthetic/audit.json"
    other = [
        tmp_path / "checkpoints" / name / "complete.json"
        for name in ("evaluation", "word", "sentence")
    ]
    scope_path = tmp_path / "runs/synthetic/scope.json"
    source_code = tmp_path / "synthetic-export-source.py"
    source_code.write_text("# synthetic code binding\n")
    monkeypatch.setattr(
        exporter,
        "export_sources",
        lambda: {"synthetic-export-source.py": exporter.sha(source_code)},
    )
    put(scope_path, {"synthetic": True})
    original_paths = []
    for branch, filename in exporter.FILES.items():
        path = tmp_path / "artifacts/original/model/models" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(originals[branch], str(path), metadata={"format": "pt", "branch": branch})
        original_paths.append(path)
    plan = {
        "planned_training_examples": 856,
        "model_plan": {
            "visual_scope": exporter.artifact(scope_path),
            "expected_visual_gradient_names": list(shapes),
            "reference": {
                "bundle_root": "artifacts/original",
                "translator_checkpoint_sha256": "public-C5",
            },
            "assets": {
                str(path.relative_to(tmp_path)): exporter.sha(path) for path in original_paths
            },
        },
    }
    put(plan_path, plan)
    visual = result_path.with_name("visual-after-000856.safetensors")
    visual.parent.mkdir(parents=True)
    metadata = {
        "plan_sha256": exporter.sha(plan_path),
        "update": "856",
        "optimizer_updates": "216",
        "fixed_winner_checkpoint": "public-C5",
    }
    save_file(trained, str(visual), metadata=metadata)
    put(
        result_path,
        {"training_examples": 856, "optimizer_updates": 216, "visual": exporter.artifact(visual)},
    )
    for path in [audit, *other]:
        put(path, {"synthetic": True})
    inputs = [plan_path, result_path, audit, *other]
    monkeypatch.setattr(
        exporter,
        "authenticate",
        lambda *args: {str(i): exporter.artifact(p) for i, p in enumerate(inputs)},
    )
    return SimpleNamespace(
        root=tmp_path,
        inputs=inputs,
        originals=originals,
        trained=trained,
        shapes=shapes,
        visual=visual,
        metadata=metadata,
        source_code=source_code,
        output=tmp_path / "checkpoints/export",
        original_paths=original_paths,
    )


def test_small_serialized_export_has_exact_values_provenance_and_atomic_commit(small_export):
    f = small_export
    before = {p: p.read_bytes() for p in f.original_paths}
    manifest = exporter.run(*f.inputs, f.output)
    assert (f.output / "EXPORT.json").is_file()
    assert not f.output.with_name(".export.partial").exists()
    assert manifest["promotion_or_deployment_authorized"] is False
    assert manifest["serving_feature_token_or_latency_equivalence_established"] is False
    assert manifest["bound_source_artifacts_reverified"] == 6
    loaded = {}
    from safetensors import safe_open

    for branch, file in manifest["files"].items():
        path = exporter.bound(file)
        loaded[branch] = load_file(str(path))
        with safe_open(str(path), framework="np") as saved:
            assert saved.metadata()["format"] == "pt"
            assert saved.metadata()["branch"] == branch
            assert saved.metadata()["training_plan_sha256"] == exporter.sha(f.inputs[0])
    exporter.verify_visual_values(f.originals, f.trained, loaded, f.shapes)
    assert before == {p: p.read_bytes() for p in f.original_paths}
    with pytest.raises(ValueError, match="fresh owned output"):
        exporter.run(*f.inputs, f.output)


def test_refused_authentication_creates_no_export_directory(small_export, monkeypatch):
    f = small_export

    def refused(*_):
        raise ValueError("synthetic failed actual-training admission")

    monkeypatch.setattr(exporter, "authenticate", refused)
    with pytest.raises(ValueError, match="admission"):
        exporter.run(*f.inputs, f.output)
    assert not f.output.exists() and not f.output.with_name(".export.partial").exists()


def test_missing_complete_results_refuse_before_model_helpers_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    monkeypatch.setattr(exporter, "export_sources", lambda: {})
    paths = [
        tmp_path / "runs/plan.json",
        tmp_path / "checkpoints/training/result.json",
        tmp_path / "runs/audit.json",
        *(
            tmp_path / "checkpoints" / name / "complete.json"
            for name in ("evaluation", "word", "sentence")
        ),
    ]
    with pytest.raises(ValueError, match="complete owned training"):
        exporter.run(*paths, tmp_path / "checkpoints/not-created")
    assert not (tmp_path / "checkpoints").exists()


def test_mid_write_failure_keeps_partial_evidence_and_prevents_replay(small_export, monkeypatch):
    import safetensors.numpy

    f = small_export

    def interrupted(values, filename, **kwargs):
        if Path(filename).name == exporter.FILES["hands"]:
            raise OSError("synthetic second-file failure")
        return save_file(values, filename, **kwargs)

    monkeypatch.setattr(safetensors.numpy, "save_file", interrupted)
    with pytest.raises(OSError, match="second-file"):
        exporter.run(*f.inputs, f.output)
    partial = f.output.with_name(".export.partial")
    assert not f.output.exists() and (partial / exporter.FILES["face"]).is_file()
    assert not (partial / "EXPORT.json").exists()
    with pytest.raises(ValueError, match="partial exports are not retried"):
        exporter.run(*f.inputs, f.output)


@pytest.mark.parametrize(
    "field,value", [("update", "428"), ("plan_sha256", "other-plan"), ("optimizer_updates", "108")]
)
def test_trained_tensor_metadata_must_match_the_complete_final(small_export, field, value):
    f = small_export
    metadata = {**f.metadata, field: value}
    f.visual.unlink()
    save_file(f.trained, str(f.visual), metadata=metadata)
    result = json.loads(f.inputs[1].read_text())
    put(f.inputs[1], {**result, "visual": exporter.artifact(f.visual)})
    with pytest.raises(ValueError, match="tensor metadata"):
        exporter.run(*f.inputs, f.output)
    assert not f.output.exists() and not f.output.with_name(".export.partial").exists()


def test_changed_evaluation_evidence_during_export_blocks_publication(small_export, monkeypatch):
    import safetensors.numpy

    f = small_export

    def tampered(values, filename, **kwargs):
        answer = save_file(values, filename, **kwargs)
        put(f.inputs[3], {"changed_after_authentication": True})
        return answer

    monkeypatch.setattr(safetensors.numpy, "save_file", tampered)
    with pytest.raises(ValueError, match="source bytes changed"):
        exporter.run(*f.inputs, f.output)
    assert not f.output.exists() and f.output.with_name(".export.partial").is_dir()


def test_conflicting_nested_evidence_bindings_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    path = tmp_path / "source.json"
    put(path, {"synthetic": True})
    descriptor = exporter.artifact(path)
    with pytest.raises(ValueError, match="conflicting"):
        exporter.verify_evidence([descriptor, {**descriptor, "sha256": "0" * 64}])


def test_low_physical_space_blocks_export_before_any_write(small_export, monkeypatch):
    f = small_export
    monkeypatch.setattr(exporter.shutil, "disk_usage", lambda _: SimpleNamespace(free=79 * 1024**3))
    with pytest.raises(ValueError, match="80GiB floor"):
        exporter.run(*f.inputs, f.output)
    assert not f.output.exists() and not f.output.with_name(".export.partial").exists()
