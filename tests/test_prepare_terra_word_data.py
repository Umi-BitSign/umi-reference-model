from __future__ import annotations

import copy
import io
import json
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_terra_word_data as module


def rows():
    original = [
        {
            "participant": f"P{i}",
            "article": "Fish",
            "section index": "0",
            "sentence index": "0",
            "video filename": f"clip{i}.mp4",
            "video length": "3",
            "sentence text": "Fish.",
        }
        for i in range(4)
    ]
    draft = [
        {
            **r,
            "label": "fish",
            "role": "development" if i == 3 else "training",
            "training_allowed": i != 3,
        }
        for i, r in enumerate(original)
    ]
    meta = {
        "heldout_participants": ["P3"],
        "selected_by_role": {"training": 3, "development": 1},
        "selected_labels": ["fish"],
    }
    return draft, meta, original


def test_metadata_preserves_full_reference_and_signer_disjoint_roles():
    result = module.validate_rows(*rows(), set())
    assert len(result) == 4
    assert all(r["reference"] == "Fish." for r in result)
    assert [r["training_allowed"] for r in result] == [True, True, True, False]
    assert result[-1]["source_group"] == "asl_stem:P3:Fish"


@pytest.mark.parametrize(
    "kind",
    ["label", "source_text", "dev_permission", "dev_signer", "duplicate", "path", "protected"],
)
def test_changed_labels_roles_sources_and_protected_references_are_rejected(kind):
    draft, meta, original = rows()
    protected = set()
    if kind == "label":
        draft[0]["label"] = "bird"
    if kind == "source_text":
        draft[0]["sentence text"] = "A fish swims."
    if kind == "dev_permission":
        draft[-1]["training_allowed"] = True
    if kind == "dev_signer":
        meta["heldout_participants"] = []
    if kind == "duplicate":
        draft.append(copy.deepcopy(draft[0]))
    if kind == "path":
        draft[0]["video filename"] = "../outside.mp4"
    if kind == "protected":
        protected.add(("fish",))
    with pytest.raises(ValueError):
        module.validate_rows(draft, meta, original, protected)


def valid_probe():
    return {
        "streams": [
            {"width": 640, "height": 480, "avg_frame_rate": "25/1", "nb_read_frames": "75"}
        ],
        "format": {"duration": "3.0"},
    }


@pytest.mark.parametrize(
    "kind", ["one_frame", "zero_fps", "nan_duration", "bad_duration", "bad_dimensions"]
)
def test_decoded_video_validation_catches_truncation_and_invalid_metadata(kind):
    value = valid_probe()
    if kind == "one_frame":
        value["streams"][0]["nb_read_frames"] = "1"
    if kind == "zero_fps":
        value["streams"][0]["avg_frame_rate"] = "0/0"
    if kind == "nan_duration":
        value["format"]["duration"] = "nan"
    if kind == "bad_duration":
        value["format"]["duration"] = "1"
    if kind == "bad_dimensions":
        value["streams"][0]["width"] = 0
    with pytest.raises(ValueError):
        module.validate_probe(value, 3)


def test_complete_decoded_video_metadata_is_retained():
    assert module.validate_probe(valid_probe(), 3) == {
        "duration_sec": 3,
        "decoded_frames": 75,
        "fps": 25,
        "width": 640,
        "height": 480,
    }


def test_space_guard_precedes_network_or_plan_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "metadata", lambda: ({}, {}, [], {}))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=79 * 1024**3))
    monkeypatch.setattr(module, "remote_archive", lambda _: pytest.fail("no network below floor"))
    plan = tmp_path / "runs" / "fresh" / "plan.json"
    with pytest.raises(RuntimeError, match="80GiB"):
        module.freeze(plan)
    assert not plan.parent.exists()


@contextmanager
def archive():
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as writer:
        for i in range(4):
            writer.writestr(f"ASL_STEM_Wiki/videos/clip{i}.mp4", f"test bytes{i}")
    data.seek(0)
    with zipfile.ZipFile(data) as opened:
        yield opened, SimpleNamespace(requests=0, bytes_downloaded=0)


def bound_rows():
    selected = module.validate_rows(*rows(), set())
    with archive() as (opened, _):
        return module.bind_members(selected, opened)


def setup_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    path = tmp_path / "plan.json"
    path.write_text("{}")
    plan = {"records": bound_rows()}
    monkeypatch.setattr(module, "checked", lambda _: (plan, {}))
    monkeypatch.setattr(module, "remote_archive", lambda _: archive())
    monkeypatch.setattr(module, "guard_space", lambda *a: None)
    monkeypatch.setattr(module, "probe", lambda p, d: module.validate_probe(valid_probe(), d))
    return path, tmp_path / "data" / "words", tmp_path / "runs" / "receipts", plan


def test_bounded_fetch_preserves_roles_and_completed_resume_does_not_download(
    monkeypatch, tmp_path
):
    path, data, receipts, plan = setup_fetch(monkeypatch, tmp_path)
    module.fetch(path, data, receipts, 4)
    for row in plan["records"]:
        video, record = module.paths_for(row, data, receipts)
        value = json.loads(record.read_text())
        assert video.is_file() and value["row"] == row
        assert not value["model_training_ready"] and not value["label_semantics_reviewed"]
    monkeypatch.setattr(
        module, "remote_archive", lambda _: pytest.fail("completed sources must not replay")
    )
    module.fetch(path, data, receipts, 4)


@pytest.mark.parametrize("kind", ["orphan", "partial", "foreign", "low_space", "bad_probe"])
def test_fetch_failures_do_not_silently_overwrite_adopt_or_approve_data(
    monkeypatch, tmp_path, kind
):
    path, data, receipts, plan = setup_fetch(monkeypatch, tmp_path)
    target, _ = module.paths_for(plan["records"][0], data, receipts)
    target.parent.mkdir(parents=True)
    if kind == "orphan":
        target.write_bytes(b"unreceipted")
    if kind == "partial":
        target.with_suffix(".partial.mp4").write_bytes(b"incomplete")
    if kind == "foreign":
        (data / "unrelated.txt").write_text("untouched")
    if kind == "low_space":
        monkeypatch.setattr(
            module, "guard_space", lambda *a: (_ for _ in ()).throw(RuntimeError("low space"))
        )
    if kind == "bad_probe":
        monkeypatch.setattr(
            module, "probe", lambda *a: (_ for _ in ()).throw(ValueError("bad video"))
        )
    with pytest.raises((ValueError, RuntimeError)):
        module.fetch(path, data, receipts, 4)
    assert not list(receipts.rglob("*.json")) if receipts.exists() else True
    if kind == "orphan":
        assert target.read_bytes() == b"unreceipted"
    if kind == "bad_probe":
        assert target.with_suffix(".partial.mp4").is_file()


def test_archive_sizes_cannot_exceed_total_data_budget(monkeypatch):
    monkeypatch.setitem(module.POLICY, "maximum_total_raw_bytes", 1)
    with pytest.raises(ValueError, match="2GiB"):
        bound_rows()
