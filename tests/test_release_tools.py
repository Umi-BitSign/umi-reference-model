from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bundle(root: Path, revision: str) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    payloads = {
        "bundle-manifest.json": b"{}",
        "inference-identity.json": json.dumps(
            {"inference_revision": revision}, sort_keys=True, separators=(",", ":")
        ).encode(),
        "model-config.json": b"{}",
        "model.safetensors": b"tensor",
        "tokenizer.json": b"{}",
        "tokenizer.model": b"tokenizer",
    }
    for name, payload in payloads.items():
        (bundle / name).write_bytes(payload)
    return bundle


def test_bundle_package_is_deterministic(tmp_path: Path) -> None:
    tool = _load("package_bundle")
    bundle = _bundle(tmp_path, "12" * 32)
    first = tool.package_bundle(bundle, tmp_path / "first.zip")
    second = tool.package_bundle(bundle, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]
    assert (tmp_path / "first.zip").read_bytes() == (tmp_path / "second.zip").read_bytes()


def test_release_metadata_binds_and_verifies_artifacts(tmp_path: Path) -> None:
    package_tool = _load("package_bundle")
    release_tool = _load("release_artifacts")
    revision = "34" * 32
    package_tool.package_bundle(
        _bundle(tmp_path, revision), tmp_path / release_tool.MODEL_FILENAME
    )
    arguments = argparse.Namespace(
        release_id="umi-s1-baseline-v0",
        inference_revision=revision,
        rights_decision_sha256="56" * 32,
        source_git_revision="78" * 20,
        umi_git_revision="9a" * 20,
        artifact=[
            f"model={tmp_path / release_tool.MODEL_FILENAME}",
        ],
        output_directory=tmp_path,
        replace=False,
    )
    created = release_tool.create_release(arguments)
    assert release_tool.verify_release(tmp_path / "release-manifest.json") == created
    (tmp_path / release_tool.MODEL_FILENAME).write_bytes(b"changed")
    with pytest.raises(release_tool.ReleaseArtifactError, match="digest differs"):
        release_tool.verify_release(tmp_path / "release-manifest.json")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_release_commit_has_source_parent_and_only_metadata(tmp_path: Path) -> None:
    package_tool = _load("package_bundle")
    release_tool = _load("release_artifacts")
    repository = tmp_path / "repository"
    release = repository / "release"
    release.mkdir(parents=True)
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Release Test")
    _git(repository, "config", "user.email", "release@example.invalid")
    _git(repository, "config", "commit.gpgsign", "false")
    (repository / ".gitignore").write_text("release/*.zip\n", encoding="utf-8")
    (repository / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "source.py")
    _git(repository, "commit", "-m", "source")
    source_revision = _git(repository, "rev-parse", "HEAD")

    revision = "34" * 32
    package_tool.package_bundle(
        _bundle(tmp_path, revision), release / release_tool.MODEL_FILENAME
    )
    created = release_tool.create_release(
        argparse.Namespace(
            release_id="umi-s1-baseline-v0",
            inference_revision=revision,
            rights_decision_sha256="56" * 32,
            source_git_revision=source_revision,
            umi_git_revision="9a" * 20,
            artifact=[f"model={release / release_tool.MODEL_FILENAME}"],
            output_directory=release,
            replace=False,
        )
    )
    _git(repository, "add", "release/release-manifest.json", "release/SHA256SUMS")
    _git(repository, "commit", "-m", "release metadata")
    release_revision = _git(repository, "rev-parse", "HEAD")

    assert (
        release_tool.verify_release_commit(
            created,
            release / "release-manifest.json",
            repository,
            release_revision,
        )
        == release_revision
    )

    (repository / "unexpected.txt").write_text("not metadata\n", encoding="utf-8")
    _git(repository, "add", "unexpected.txt")
    _git(repository, "commit", "--amend", "--no-edit")
    with pytest.raises(release_tool.ReleaseArtifactError, match="not metadata-only"):
        release_tool.verify_release_commit(
            created, release / "release-manifest.json", repository, "HEAD"
        )


def test_runtime_staging_uses_only_manifest_paths(tmp_path: Path) -> None:
    tool = _load("stage_runtime")
    source = tmp_path / "private"
    destination = tmp_path / "public"
    source.mkdir()
    destination.mkdir()
    (source / "src").mkdir()
    (source / "src" / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("src/runtime.py\n", encoding="utf-8")
    assert tool.stage(source, destination, manifest, check_only=True, replace=False) == 1
    assert tool.stage(source, destination, manifest, check_only=False, replace=False) == 1
    assert (destination / "src" / "runtime.py").read_text() == "VALUE = 1\n"
    assert tool.stage(source, destination, manifest, check_only=True, replace=False) == 1


def test_runtime_staging_rejects_intermediate_source_symlink(tmp_path: Path) -> None:
    tool = _load("stage_runtime")
    source = tmp_path / "private"
    destination = tmp_path / "public"
    outside = tmp_path / "outside"
    source.mkdir()
    destination.mkdir()
    outside.mkdir()
    (outside / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "src").symlink_to(outside, target_is_directory=True)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("src/runtime.py\n", encoding="utf-8")
    with pytest.raises(tool.StagingError, match="parent contains a symlink"):
        tool.stage(source, destination, manifest, check_only=True, replace=False)


def test_repository_guard_accepts_public_tree() -> None:
    tool = _load("repo_guard")
    files, size = tool.check_repository(ROOT)
    assert files > 10
    assert size > 10_000


def test_miner_runbook_uses_locked_no_build_source_paths() -> None:
    runbook = (ROOT / "docs" / "RUN_MINER.md").read_text(encoding="utf-8")
    assert "--editable" not in runbook
    assert "export PYTHONPATH=" not in runbook
    assert "umi-reference-model-source.pth" in runbook
    assert "umi-source.pth" in runbook
    assert ".venv/bin/python -m bitsign_motion.local_extractor_release" in runbook
    assert ".venv/bin/python -m bitsign_motion.local_bundle_rebind" in runbook
    assert "-m bitsign_motion.umi_reference_backend probe" in runbook
    assert 'bin/python" -m umi.miner' in runbook


def test_tooling_docs_do_not_require_unversioned_python_command() -> None:
    for relative in ("docs/RELEASE.md", "docs/SOURCE_STAGING.md"):
        document = (ROOT / relative).read_text(encoding="utf-8")
        document = document.replace("uv run --frozen --extra dev python tools/", "")
        assert "python tools/" not in document
        assert "uv run python tools/" not in document
