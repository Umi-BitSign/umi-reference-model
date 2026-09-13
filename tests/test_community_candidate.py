from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "candidates" / "community-baseline-v0.2"


@pytest.fixture
def checker(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "candidate_asset_checker", CANDIDATE / "verify_model_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def record(path="models/test.safetensors", payload=b"inert test asset"):
    return {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def manifest(checker, records):
    (checker.ROOT / "MODEL_ARTIFACT.json").write_text(json.dumps({"model_files": records}))


def asset(checker, name="models/test.safetensors", data=b"inert test asset"):
    path = checker.ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_candidate_assets_match_importer_identity_and_include_no_binaries():
    value = json.loads((CANDIDATE / "MODEL_ARTIFACT.json").read_text())
    assert value["archive"]["sha256"] == (
        "f78979599486456e06e7886b126169f8d45e5e1e7d16d7a2b617648615eef3d4"
    )
    assert value["archive"]["bytes"] == 2_934_700_086
    assert len(value["model_files"]) == 8
    paths = {item["path"] for item in value["model_files"]}
    assert len(paths) == 8
    for item in value["model_files"]:
        path = CANDIDATE / item["path"]
        if path.suffix == ".json":
            assert path.stat().st_size == item["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        else:
            assert path.suffix in {".safetensors", ".task"}
            assert not path.exists()
    identity = json.loads((CANDIDATE / "MODEL_IDENTITY.json").read_text())
    assert identity["model_id"] == value["model_id"]
    assert identity["generation"] == {"num_beams": 5, "max_length": 2048}
    assert identity["qualification"] == "Functional smoke only; no benchmark claim"


def test_checker_accepts_matching_assets_without_importing_model_code(checker, capsys):
    manifest(checker, [record()])
    asset(checker)
    (checker.ROOT / "runtime.py").write_text("raise AssertionError('model code must not run')")
    checker.verify()
    assert capsys.readouterr().out == "All model assets verified\n"


@pytest.mark.parametrize("failure", ("missing", "size", "hash", "escape"))
def test_checker_rejects_missing_altered_and_escaping_assets(checker, failure):
    manifest(checker, [record()])
    path = asset(checker)
    if failure == "missing":
        path.unlink()
    elif failure == "size":
        path.write_bytes(b"short")
    elif failure == "hash":
        path.write_bytes(b"x" * len(b"inert test asset"))
    else:
        outside = checker.ROOT.parent / "outside-asset"
        outside.write_bytes(b"inert test asset")
        path.unlink()
        path.symlink_to(outside)
    with pytest.raises(ValueError):
        checker.verify()


@pytest.mark.parametrize(
    "records",
    [
        [],
        [record(), record()],
        [record("../outside")],
        [record("/absolute")],
        [record("models/../outside")],
        [record("models//test.safetensors")],
        [record("runtime.py")],
        [{**record(), "bytes": True}],
        [{**record(), "bytes": 5 * 1024**3}],
        [{**record(), "sha256": "not-a-digest"}],
    ],
    ids=(
        "empty",
        "duplicate",
        "parent",
        "absolute",
        "traversal",
        "alias",
        "non-model",
        "boolean-size",
        "oversized",
        "digest",
    ),
)
def test_checker_rejects_invalid_inventories(checker, records):
    asset(checker)
    manifest(checker, records)
    with pytest.raises(ValueError):
        checker.verify()
