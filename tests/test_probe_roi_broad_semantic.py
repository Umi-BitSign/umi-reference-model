from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_roi_broad_semantic as probe


def row(source, number, *, frames=10, reference=None, signer="P100"):
    return {
        "roi_role": "training",
        "roi_training_allowed": True,
        "frames": frames,
        "streamed_pixels_equal_native_reference": True,
        "cache": {"path": f"checkpoints/unit/clip-{number}.npz", "sha256": str(number)},
        "original_row": {
            "source": source,
            "example_id": str(number),
            "signer": signer,
            "role": "training",
            "training_allowed": True,
            "reference": reference or f"Training phrase {number}",
            "original_pair": {"header": {"sampled_frames": frames}},
        },
    }


def inventory():
    rows = [row(source, i) for i, source in enumerate(probe.LAYOUT)]
    plan = {
        "records": copy.deepcopy(rows),
        "reserved_STEM_participants": ["P4"],
        "ROI_diagnostic_participants": ["P19", "P0"],
    }
    data = {"complete": True, "all_streamed_pixels_equal_native_reference": True, "records": rows}
    return plan, data


def test_training_rows_keep_every_family_and_namespace_reserved_STEM_signers():
    plan, data = inventory()
    assert probe.training_rows(plan, data) == data["records"]
    for container in (plan, data):
        container["records"][-1]["original_row"]["signer"] = "P4"
    assert len(probe.training_rows(plan, data)) == 4
    for container in (plan, data):
        container["records"][0]["original_row"]["signer"] = "P4"
    with pytest.raises(ValueError, match="reserved"):
        probe.training_rows(plan, data)


@pytest.mark.parametrize(
    "mutation", ["incomplete", "parity", "drop", "identity", "diagnostic", "duplicate", "frame"]
)
def test_training_rows_reject_changed_or_incomplete_media(mutation):
    plan, data = inventory()
    if mutation == "incomplete":
        data["complete"] = False
    elif mutation == "parity":
        data["all_streamed_pixels_equal_native_reference"] = False
    elif mutation == "drop":
        data["records"].pop()
    elif mutation == "identity":
        data["records"][0]["original_row"]["example_id"] = "changed"
    elif mutation == "diagnostic":
        for container in (plan, data):
            container["records"][0]["roi_role"] = "feature_diagnostic"
    elif mutation == "duplicate":
        for container in (plan, data):
            container["records"][1] = copy.deepcopy(container["records"][0])
    else:
        for container in (plan, data):
            container["records"][0]["frames"] = 9
    with pytest.raises(ValueError):
        probe.training_rows(plan, data)


def test_original_diagnostics_are_retained_but_never_returned_for_gradients():
    plan, data = inventory()
    diagnostic = row("asl_stem_original", 100, signer="P19")
    diagnostic.update(roi_role="feature_diagnostic", roi_training_allowed=False)
    plan["records"].append(copy.deepcopy(diagnostic))
    data["records"].append(diagnostic)
    assert len(probe.training_rows(plan, data)) == 4
    assert all(r["roi_training_allowed"] for r in probe.training_rows(plan, data))


def test_proof_selection_covers_each_family_longest_timeline_and_longest_byte_caption():
    _, data = inventory()
    rows = data["records"]
    rows.extend(
        [
            row("asl_stem_original", 10, frames=149),
            row("asl_stem_supplemental", 11, frames=100),
            row("how2sign_precut_expansion", 12, frames=20),
            row("openasl_scale", 13, frames=60),
            row(
                "openasl_scale",
                14,
                frames=5,
                reference="A substantially longer complete training caption",
            ),
        ]
    )
    selected = probe.proof_indices(rows)
    assert selected == [4, 5, 6, 7, 8]
    assert len(selected) == len(set(selected))
    assert {rows[i]["original_row"]["source"] for i in selected} == set(probe.LAYOUT)
    assert probe.proof_indices(copy.deepcopy(rows)) == selected


def test_proof_selection_does_not_duplicate_a_longest_caption_case():
    _, data = inventory()
    data["records"][0]["original_row"]["reference"] = "the longest complete training caption"
    assert probe.proof_indices(data["records"]) == [0, 1, 2, 3]


def test_proof_requires_all_families_and_distinct_adjacent_caption_donors():
    _, data = inventory()
    with pytest.raises(ValueError, match="every"):
        probe.proof_indices(data["records"][:-1])
    data["records"][1]["original_row"]["reference"] = data["records"][0]["original_row"][
        "reference"
    ]
    with pytest.raises(ValueError, match="donors"):
        probe.proof_indices(data["records"])


@pytest.mark.parametrize("mode", ["candidates", "qualify"])
def test_shared_phase_keeps_bytecode_and_all_globals_except_the_plan_validator(mode):
    original = {"candidates": probe.shared.prepare, "qualify": probe.shared.qualify}[mode]
    previous = dict(original.__globals__)
    bound = probe.bound_phase(mode)
    assert bound.__code__ is original.__code__
    assert bound.__defaults__ is original.__defaults__
    assert bound.__closure__ is original.__closure__
    assert bound.__globals__ is not original.__globals__
    assert bound.__globals__["checked"] is probe.checked
    assert set(bound.__globals__) == set(previous)
    assert all(bound.__globals__[k] is v for k, v in previous.items() if k != "checked")
    assert all(original.__globals__[k] is v for k, v in previous.items())


def test_adapter_has_no_unrequested_phase_or_model_fallback():
    with pytest.raises(KeyError):
        probe.bound_phase("fallback")
    assert probe.shared.POLICY["judge"] == "gpt-5.6-terra"
    assert probe.shared.POLICY["reasoning_effort"] == "medium"
    assert probe.shared.POLICY["caption_weight"] == 0
    assert probe.shared.POLICY["automatic_retries"] == 0


@pytest.mark.parametrize("first_exit", [0, 1])
def test_pipeline_blocks_on_own_children_and_stops_without_retry(tmp_path, monkeypatch, first_exit):
    path = tmp_path / "runs/plan.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"source_sha256": {"tools/probe_roi_broad_semantic.py": "bound"}}))
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(probe, "storage", lambda: None)
    monkeypatch.setattr(probe, "sha256", lambda *args: "bound")
    monkeypatch.setattr(probe, "identity", lambda *args: {"pid": 1})
    monkeypatch.setattr(probe, "artifact", lambda p: {"path": str(p), "sha256": "bound"})
    calls = []

    def child(command, check):
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=first_exit if len(calls) == 1 else 0)

    monkeypatch.setattr(probe.subprocess, "run", child)
    output = tmp_path / "checkpoints/pipeline"
    if first_exit:
        with pytest.raises(RuntimeError, match="no retry"):
            probe.pipeline(path, output)
        assert len(calls) == 1
        assert not (output / "complete.json").exists()
    else:
        probe.pipeline(path, output)
        assert len(calls) == 2
        assert [c[c.index("--mode") + 1] for c in calls] == ["candidates", "qualify"]
        assert "--candidates" in calls[1]
        assert json.loads((output / "complete.json").read_text())["optimizer_updates"] == 0
