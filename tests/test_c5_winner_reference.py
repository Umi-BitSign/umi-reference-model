from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_winner_reference import canonical, verify_tree


def bundle(tmp_path):
    root = (tmp_path / "bundle").resolve()
    (root / "model").mkdir(parents=True)
    payload = b"safe test bytes, not a model"
    target = root / "model/weights.safetensors"
    target.write_bytes(payload)
    target.chmod(0o400)
    manifest = {
        "schema": "umi-model-bundle/1",
        "profile": "offline_bundle/1",
        "parent_baseline_sha256": "a" * 64,
        "license_id": "CC-BY-SA-4.0",
        "files": [
            {
                "path": target.name,
                "role": "weights",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    raw = canonical(manifest)
    (root / "manifest.json").write_bytes(raw)
    identity = hashlib.sha256(b"umi-open-competition-v1\0" + raw).hexdigest()
    return root, identity, manifest


def test_verified_derivative_parent_and_license_are_not_forced_to_old_baseline(tmp_path):
    root, identity, manifest = bundle(tmp_path)
    assert verify_tree(root, identity) == manifest


def test_wrong_model_identity_is_rejected(tmp_path):
    root, _, _ = bundle(tmp_path)
    with pytest.raises(ValueError, match="model identity"):
        verify_tree(root, "b" * 64)


@pytest.mark.parametrize("kind", ("changed", "extra", "link", "writable"))
def test_changed_undeclared_linked_and_writable_assets_are_rejected(tmp_path, kind):
    root, identity, _ = bundle(tmp_path)
    target = root / "model/weights.safetensors"
    if kind == "extra":
        (root / "model/extra.py").write_text("raise RuntimeError('never execute')")
    elif kind == "link":
        (root / "model/link").symlink_to(target)
    else:
        target.chmod(0o600)
        if kind == "changed":
            target.write_bytes(b"corrupted")
            target.chmod(0o400)
    with pytest.raises(ValueError, match="C5 reference"):
        verify_tree(root, identity)


def test_noncanonical_or_duplicate_manifest_keys_are_rejected(tmp_path):
    root, identity, manifest = bundle(tmp_path)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with pytest.raises(ValueError, match="noncanonical"):
        verify_tree(root, identity)
    (root / "manifest.json").write_bytes(b'{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError, match="duplicate"):
        verify_tree(root, identity)
