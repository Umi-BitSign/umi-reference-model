from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location(
        "community_bundle_verification", ROOT / "community/bundle_verification.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(root, verifier, document):
    manifest = root / "manifest.json"
    if manifest.exists():
        manifest.chmod(0o600)
    raw = verifier._canonical(document)
    manifest.write_bytes(raw)
    manifest.chmod(0o400)
    return hashlib.sha256(verifier.DOMAIN + raw).hexdigest()


@pytest.fixture
def bundle(tmp_path, verifier):
    root = tmp_path / "bundle"
    root.mkdir(mode=0o700)
    model = root / "model"
    model.mkdir(mode=0o700)
    payloads = {
        "LICENSE": ("license", b"inert test license"),
        "models/config.json": ("config", b"{}"),
        "models/weights.safetensors": ("weights", b"inert weights"),
        "runtime.py": ("dependency", b"raise AssertionError('must not import model')"),
        "umi_inference.py": ("inference", b"raise AssertionError('must not execute entrypoint')"),
    }
    records = []
    for name, (role, data) in sorted(payloads.items()):
        path = model / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o400)
        records.append(
            {
                "path": name,
                "role": role,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    document = {
        "schema": "umi-model-bundle/1",
        "profile": "offline_bundle/1",
        "parent_baseline_sha256": None,
        "license_id": "MIT",
        "files": records,
    }
    revision = publish(root, verifier, document)
    return root, document, revision


def check(verifier, bundle):
    root, _, revision = bundle
    return verifier.verify_imported_bundle(root, expected_bundle_sha256=revision)


def test_checks_every_artifact_without_executing_or_changing_it(verifier, bundle):
    root, document, revision = bundle
    before = {p: (p.read_bytes(), p.stat().st_mode) for p in root.rglob("*") if p.is_file()}
    result = check(verifier, bundle)
    assert result == {
        "model_bundle_sha256": revision,
        "manifest_file_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "files": 5,
        "bytes": sum(r["size_bytes"] for r in document["files"]),
    }
    assert before == {p: (p.read_bytes(), p.stat().st_mode) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "name",
    [
        "LICENSE",
        "models/config.json",
        "models/weights.safetensors",
        "runtime.py",
        "umi_inference.py",
    ],
)
def test_changes_to_any_declared_file_fail(verifier, bundle, name):
    path = bundle[0] / "model" / name
    original = path.read_bytes()
    path.chmod(0o600)
    path.write_bytes(bytes(b ^ 1 for b in original))
    path.chmod(0o400)
    with pytest.raises(ValueError, match="checksum"):
        check(verifier, bundle)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "extra_directory",
        "symlink",
        "parent_link",
        "hardlink",
        "fifo",
        "writable_file",
        "writable_directory",
        "public_root",
    ],
)
def test_refuses_invalid_filesystem_layout(verifier, bundle, tmp_path, mutation):
    root = bundle[0]
    path = root / "model/runtime.py"
    if mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        (root / "model/unknown.py").write_text("unexpected")
    elif mutation == "extra_directory":
        (root / "model/empty").mkdir()
    elif mutation == "symlink":
        path.rename(tmp_path / "outside")
        path.symlink_to(tmp_path / "outside")
    elif mutation == "parent_link":
        parent = root / "model/models"
        parent.rename(tmp_path / "outside")
        parent.symlink_to(tmp_path / "outside", target_is_directory=True)
    elif mutation == "hardlink":
        os.link(path, tmp_path / "outside")
    elif mutation == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif mutation == "writable_file":
        path.chmod(0o600)
    elif mutation == "writable_directory":
        path.parent.chmod(0o777)
    else:
        root.chmod(0o755)
    with pytest.raises((ValueError, OSError)):
        check(verifier, bundle)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a/../b", "a//b", "a\\b", "", ".", "a/./b"]
)
def test_invalid_manifest_paths_rejected_before_traversal(verifier, bundle, name):
    root, document, _ = bundle
    document["files"][0]["path"] = name
    revision = publish(root, verifier, document)
    with pytest.raises(ValueError, match="path"):
        verifier.verify_imported_bundle(root, expected_bundle_sha256=revision)


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_role",
        "bool_size",
        "negative_size",
        "oversized",
        "bad_digest",
        "unsorted",
        "duplicate",
        "case_alias",
        "file_directory_alias",
        "entrypoint",
        "unknown_field",
        "wrong_profile",
        "wrong_parent",
        "wrong_license",
        "empty",
        "non_list",
    ],
)
def test_invalid_manifest_inventory(verifier, bundle, mutation):
    root, document, _ = bundle
    record = document["files"][0]
    if mutation == "bad_role":
        record["role"] = "not-a-role"
    elif mutation == "bool_size":
        record["size_bytes"] = True
    elif mutation == "negative_size":
        record["size_bytes"] = -1
    elif mutation == "oversized":
        record["size_bytes"] = verifier.MAXIMUM_BYTES + 1
    elif mutation == "bad_digest":
        record["sha256"] = "z" * 64
    elif mutation == "unsorted":
        document["files"].reverse()
    elif mutation in {"duplicate", "case_alias", "file_directory_alias"}:
        extra = dict(record)
        extra["path"] = {
            "duplicate": "LICENSE",
            "case_alias": "license",
            "file_directory_alias": "LICENSE/child",
        }[mutation]
        document["files"].append(extra)
        document["files"].sort(key=lambda r: r["path"])
    elif mutation == "entrypoint":
        document["files"][-1]["role"] = "dependency"
    elif mutation == "unknown_field":
        document["trust_me"] = True
    elif mutation == "wrong_profile":
        document["profile"] = "native"
    elif mutation == "wrong_parent":
        document["parent_baseline_sha256"] = "a" * 64
    elif mutation == "wrong_license":
        document["license_id"] = "no-approval"
    elif mutation == "empty":
        document["files"] = []
    else:
        document["files"] = {}
    revision = publish(root, verifier, document)
    with pytest.raises(ValueError):
        verifier.verify_imported_bundle(root, expected_bundle_sha256=revision)


def test_caller_pin_is_required_and_intake_report_is_not_trusted(verifier, bundle):
    root = bundle[0]
    (root / "intake.json").write_text('{"model_bundle_sha256":"' + "f" * 64 + '"}')
    assert check(verifier, bundle)["model_bundle_sha256"] == bundle[2]
    with pytest.raises(ValueError, match="caller-pinned"):
        verifier.verify_imported_bundle(root, expected_bundle_sha256="f" * 64)
    with pytest.raises(ValueError, match="digest"):
        verifier.verify_imported_bundle(root, expected_bundle_sha256="invalid")
    with pytest.raises(ValueError, match="absolute"):
        verifier.verify_imported_bundle(Path("relative"), expected_bundle_sha256=bundle[2])


def test_bundle_root_alias_is_rejected(verifier, bundle, tmp_path):
    alias = tmp_path / "bundle-alias"
    alias.symlink_to(bundle[0], target_is_directory=True)
    with pytest.raises(ValueError, match="aliases"):
        verifier.verify_imported_bundle(alias, expected_bundle_sha256=bundle[2])


@pytest.mark.parametrize(
    "mutation", ["noncanonical", "duplicate_key", "oversize", "symlink", "hardlink", "writable"]
)
def test_manifest_file_checks(verifier, bundle, tmp_path, mutation):
    root, document, revision = bundle
    manifest = root / "manifest.json"
    manifest.chmod(0o600)
    if mutation == "noncanonical":
        manifest.write_text(json.dumps(document, indent=2))
    elif mutation == "duplicate_key":
        raw = manifest.read_bytes()
        manifest.write_bytes(b'{"schema":"umi-model-bundle/1",' + raw[1:])
    elif mutation == "oversize":
        manifest.write_bytes(b"x" * (verifier.MAXIMUM_MANIFEST_BYTES + 1))
    elif mutation == "symlink":
        manifest.rename(tmp_path / "manifest")
        manifest.symlink_to(tmp_path / "manifest")
    elif mutation == "hardlink":
        os.link(manifest, tmp_path / "manifest")
    if mutation != "writable":
        manifest.chmod(0o400)
    revision = hashlib.sha256(verifier.DOMAIN + manifest.read_bytes()).hexdigest()
    with pytest.raises((ValueError, OSError)):
        verifier.verify_imported_bundle(root, expected_bundle_sha256=revision)


def test_mid_scan_manifest_replacement_is_rejected(verifier, bundle, monkeypatch):
    real_tree = verifier._tree

    def replace(root_fd, records):
        real_tree(root_fd, records)
        manifest = bundle[0] / "manifest.json"
        replacement = bundle[0] / "replacement"
        replacement.write_bytes(manifest.read_bytes())
        replacement.chmod(0o400)
        replacement.replace(manifest)

    monkeypatch.setattr(verifier, "_tree", replace)
    with pytest.raises(ValueError, match="directory changed"):
        check(verifier, bundle)


@pytest.mark.parametrize("umask", [0o000, 0o002, 0o077])
def test_importer_output_is_accepted(tmp_path, monkeypatch, verifier, umask):
    # Resolve the shared fixture by file so this works both in the complete
    # tests package and in a dependency-light snapshot of these test files.
    spec = importlib.util.spec_from_file_location(
        "community_baseline_test_helpers", ROOT / "tests/test_community_baseline.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    importer = helpers.load("tools/import_community_baseline.py")
    archive = helpers.archive_fixture(tmp_path, importer, monkeypatch)
    root = tmp_path / "staged"
    previous_umask = os.umask(umask)
    try:
        intake = importer.stage_baseline(archive, root)
    finally:
        os.umask(previous_umask)
    for directory in (root, *root.rglob("*")):
        if directory.is_dir():
            assert directory.stat().st_mode & 0o777 == 0o700
    result = verifier.verify_imported_bundle(
        root, expected_bundle_sha256=intake["model_bundle_sha256"]
    )
    assert result["manifest_file_sha256"] == intake["manifest_file_sha256"]
    assert result["files"] == intake["files"]
    assert result["bytes"] == intake["bytes"]
