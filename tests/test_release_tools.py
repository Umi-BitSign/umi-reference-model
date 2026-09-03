from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import torch

from bitsign_motion import s1_portable_runtime as portable_module
from bitsign_motion import s1_release_evidence as release_evidence
from bitsign_motion.canonical import canonical_json_bytes, canonical_json_sha256
from bitsign_motion.portable_model import PortableS1, PortableS1Config
from bitsign_motion.s1_portable_runtime import (
    S1_PUBLIC_FINETUNE_CLAIM_PROFILE,
    build_s1_validation_word_prefix_contract,
    export_s1_portable_bundle,
    seal_s1_validation_word_prefix_selection,
)
from bitsign_motion.s1_state_digest import s1_tensor_set_sha256

from .release_evidence_fixtures import (
    make_e2e,
    make_motion,
    make_rights,
    make_selection,
    write_evidence_set,
)
from .test_s1_portable_runtime import (
    _experiment_rights,
    _preprocessing,
    _public_finetune_claim_fixture,
    _rights,
    _synthetic_tokenizer,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _invalid_bundle(root: Path, revision: str) -> Path:
    bundle = root / "invalid-bundle"
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


def _unchecked_package(bundle: Path, output: Path) -> None:
    tool = _load("package_bundle")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for name in tool.EXPECTED_FILES:
            info = zipfile.ZipInfo(name, tool.FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (bundle / name).read_bytes())


@pytest.fixture(scope="module")
def valid_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str]:
    root = tmp_path_factory.mktemp("release-tools-valid-bundle")
    tokenizer_model, tokenizer_record = _synthetic_tokenizer()
    tokenizer_model_path = root / "tokenizer.model"
    tokenizer_record_path = root / "tokenizer.json"
    tokenizer_model_path.write_bytes(tokenizer_model)
    tokenizer_record_path.write_bytes(tokenizer_record)
    rights = _rights()
    rights["source_id"] = "fleurs-asl-v1"
    rights["source_version"] = "kaggle-v2-2024-08-29"
    rights["redistribution_blocked_pending_final_rights_review"] = False
    rights_decision = "56" * 32
    rights["release_rights_decision_sha256"] = rights_decision
    selection = seal_s1_validation_word_prefix_selection(
        {
            "schema": "umi-s1-word-prefix-selection/1",
            "partition": "fleurs_val",
            "normalization_revision": "synthetic-normalization/1",
            "source_prediction_set_sha256": "51" * 32,
            "sample_count": 2,
            "candidate_maximum_words": [8],
            "candidate_scores": [
                {
                    "maximum_words": 8,
                    "mean_normalized_score": {
                        "numerator": "1",
                        "denominator": "2",
                    },
                }
            ],
            "selected_maximum_words": 8,
            "selection_rule": "greatest-score-then-smallest-cap-v1",
            "selected_arm": "fixture-arm",
            "selected_epoch": 1,
            "test_partition_opened": False,
        }
    )
    torch.manual_seed(17)
    output = root / "bundle"
    report = export_s1_portable_bundle(
        PortableS1(PortableS1Config()).eval(),
        output,
        tokenizer_model_path=tokenizer_model_path,
        tokenizer_record_path=tokenizer_record_path,
        preprocessing=_preprocessing(),
        rights=rights,
        upstream_release_identity_sha256="41" * 32,
        text_postprocess=build_s1_validation_word_prefix_contract(selection),
    )
    return output, str(report["inference_revision"]), rights_decision


def _copy_companions(release_tool: ModuleType, output: Path) -> None:
    for release_filename, source_relative in release_tool.COMPANION_ARTIFACT_SOURCES.values():
        shutil.copyfile(ROOT / source_relative, output / release_filename)


def _reseal_selection(record: dict[str, object]) -> None:
    record.pop("content_sha256", None)
    record["content_sha256"] = canonical_json_sha256(
        record,
        domain=release_evidence._SELECTION_DOMAIN,
    )


def _artifact_arguments(release_tool: ModuleType, output: Path) -> list[str]:
    return [
        f"{label}={output / filename}"
        for label, filename in release_tool.EXPECTED_ARTIFACTS.items()
    ]


def _stage_release_inputs(
    release_tool: ModuleType,
    package_tool: ModuleType,
    output: Path,
    bundle: Path,
    revision: str,
    rights_decision: str,
    umi_revision: str,
    *,
    tested_revision: str = "13" * 20,
) -> dict[str, object]:
    package_tool.package_bundle(bundle, output / release_tool.MODEL_FILENAME)
    _copy_companions(release_tool, output)
    identity = json.loads((bundle / "inference-identity.json").read_bytes())
    write_evidence_set(
        output,
        identity,
        revision,
        rights_decision,
        umi_revision,
        tested_revision=tested_revision,
    )
    return identity


def _release_arguments(
    release_tool: ModuleType,
    output: Path,
    revision: str,
    rights_decision: str,
    umi_revision: str,
    source_revision: str = "78" * 20,
) -> argparse.Namespace:
    return argparse.Namespace(
        release_id="umi-s1-baseline-v0",
        inference_revision=revision,
        rights_decision_sha256=rights_decision,
        source_git_revision=source_revision,
        umi_git_revision=umi_revision,
        artifact=_artifact_arguments(release_tool, output),
        output_directory=output,
        replace=False,
    )


def test_bundle_package_is_deterministic(
    valid_bundle: tuple[Path, str, str], tmp_path: Path
) -> None:
    tool = _load("package_bundle")
    bundle, _revision, _rights_decision = valid_bundle
    first = tool.package_bundle(bundle, tmp_path / "first.zip")
    second = tool.package_bundle(bundle, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]
    assert (tmp_path / "first.zip").read_bytes() == (tmp_path / "second.zip").read_bytes()


def test_bundle_packager_accepts_public_finetune_profile(tmp_path: Path) -> None:
    tool = _load("package_bundle")
    tokenizer_model, tokenizer_record = _synthetic_tokenizer()
    tokenizer_model_path = tmp_path / "tokenizer.model"
    tokenizer_record_path = tmp_path / "tokenizer.json"
    tokenizer_model_path.write_bytes(tokenizer_model)
    tokenizer_record_path.write_bytes(tokenizer_record)
    model = PortableS1(PortableS1Config()).eval()
    model_state_sha256 = s1_tensor_set_sha256(model.state_dict())
    rights = _experiment_rights()
    rights_sha256 = canonical_json_sha256(rights)
    evidence, arguments = _public_finetune_claim_fixture()
    evidence["candidate"]["selected_model_state_sha256"] = model_state_sha256
    evidence["tokenizer"]["model_sha256"] = hashlib.sha256(tokenizer_model).hexdigest()
    evidence["tokenizer"]["record_sha256"] = hashlib.sha256(tokenizer_record).hexdigest()
    evidence["rights_record_sha256"] = rights_sha256
    evidence["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in evidence.items() if key != "content_sha256"},
        domain=portable_module._PUBLIC_FINETUNE_RELEASE_IDENTITY_DOMAIN,
    )
    upstream_identity_sha256 = canonical_json_sha256(evidence)
    report = export_s1_portable_bundle(
        model,
        tmp_path / "public-finetune-bundle",
        tokenizer_model_path=tokenizer_model_path,
        tokenizer_record_path=tokenizer_record_path,
        preprocessing=_preprocessing(),
        rights=rights,
        upstream_release_identity_sha256=upstream_identity_sha256,
        text_postprocess=arguments["text_postprocess"],
        claim_boundary_profile=S1_PUBLIC_FINETUNE_CLAIM_PROFILE,
        claim_boundary_evidence=evidence,
    )

    packaged = tool.package_bundle(
        tmp_path / "public-finetune-bundle",
        tmp_path / "public-finetune.zip",
    )

    assert packaged["inference_revision"] == report["inference_revision"]
    assert packaged["archive"] == "public-finetune.zip"


def test_selection_evidence_discloses_source_rebind_transfer(
    valid_bundle: tuple[Path, str, str],
) -> None:
    bundle, revision, _rights_decision = valid_bundle
    identity = json.loads((bundle / "inference-identity.json").read_bytes())
    selection = make_selection(identity, revision)
    candidate = selection["release_candidate"]
    assert isinstance(candidate, dict)
    candidate["evaluated_inference_revision"] = release_evidence._EVALUATED_BEAM_INFERENCE_REVISION
    candidate["evaluation_transfer_basis"] = release_evidence._SOURCE_REBIND_EVALUATION_TRANSFER
    _reseal_selection(selection)

    release_evidence.validate_selection_ledger(
        selection,
        expected_inference_revision=revision,
    )

    invalid = copy.deepcopy(selection)
    invalid_candidate = invalid["release_candidate"]
    assert isinstance(invalid_candidate, dict)
    invalid_candidate["evaluation_transfer_basis"] = release_evidence._DIRECT_EVALUATION_TRANSFER
    _reseal_selection(invalid)
    with pytest.raises(
        release_evidence.S1ReleaseEvidenceError,
        match="release candidate differs",
    ):
        release_evidence.validate_selection_ledger(
            invalid,
            expected_inference_revision=revision,
        )


def test_bundle_packager_rejects_invalid_portable_bundle(tmp_path: Path) -> None:
    tool = _load("package_bundle")
    with pytest.raises(tool.BundlePackagingError, match="strict runtime loading"):
        tool.package_bundle(_invalid_bundle(tmp_path, "12" * 32), tmp_path / "invalid.zip")
    assert not (tmp_path / "invalid.zip").exists()


def test_release_metadata_binds_complete_artifact_set(
    valid_bundle: tuple[Path, str, str], tmp_path: Path
) -> None:
    package_tool = _load("package_bundle")
    release_tool = _load("release_artifacts")
    bundle, revision, rights_decision = valid_bundle
    umi_revision = "9a" * 20
    _stage_release_inputs(
        release_tool,
        package_tool,
        tmp_path,
        bundle,
        revision,
        rights_decision,
        umi_revision,
    )
    created = release_tool.create_release(
        _release_arguments(
            release_tool,
            tmp_path,
            revision,
            rights_decision,
            umi_revision,
        )
    )
    assert set(created["evidence"]) == set(release_tool.EVIDENCE_SCHEMAS)
    assert created["licenses"] == release_tool.LICENSE_DECLARATION
    assert release_tool.verify_release(tmp_path / "release-manifest.json") == created
    checksum_names = {
        line.split("  ", 1)[1] for line in (tmp_path / "SHA256SUMS").read_text().splitlines()
    }
    assert checksum_names == set(release_tool.EXPECTED_ARTIFACTS.values())


def test_release_rejects_missing_or_changed_fixed_artifacts(
    valid_bundle: tuple[Path, str, str], tmp_path: Path
) -> None:
    package_tool = _load("package_bundle")
    release_tool = _load("release_artifacts")
    bundle, revision, rights_decision = valid_bundle
    umi_revision = "9a" * 20
    _stage_release_inputs(
        release_tool,
        package_tool,
        tmp_path,
        bundle,
        revision,
        rights_decision,
        umi_revision,
    )
    arguments = _release_arguments(
        release_tool,
        tmp_path,
        revision,
        rights_decision,
        umi_revision,
    )
    arguments.artifact.pop()
    with pytest.raises(release_tool.ReleaseArtifactError, match="complete fixed artifact"):
        release_tool.create_release(arguments)

    arguments.artifact = _artifact_arguments(release_tool, tmp_path)
    (tmp_path / "NOTICE").write_bytes(b"changed\n")
    with pytest.raises(release_tool.ReleaseArtifactError, match="canonical repository source"):
        release_tool.create_release(arguments)


def test_release_rejects_artifact_mutation_and_rights_mismatch(
    valid_bundle: tuple[Path, str, str], tmp_path: Path
) -> None:
    package_tool = _load("package_bundle")
    release_tool = _load("release_artifacts")
    bundle, revision, rights_decision = valid_bundle
    umi_revision = "9a" * 20
    _stage_release_inputs(
        release_tool,
        package_tool,
        tmp_path,
        bundle,
        revision,
        rights_decision,
        umi_revision,
    )
    created = release_tool.create_release(
        _release_arguments(
            release_tool,
            tmp_path,
            revision,
            rights_decision,
            umi_revision,
        )
    )
    evidence_path = tmp_path / release_tool.SELECTION_LEDGER_FILENAME
    evidence_path.write_bytes(evidence_path.read_bytes() + b"\n")
    with pytest.raises(release_tool.ReleaseArtifactError):
        release_tool.verify_release(tmp_path / "release-manifest.json")
    assert created["rights_decision_sha256"] == rights_decision

    second = tmp_path / "second"
    second.mkdir()
    _stage_release_inputs(
        release_tool,
        package_tool,
        second,
        bundle,
        revision,
        rights_decision,
        umi_revision,
    )
    arguments = _release_arguments(
        release_tool,
        second,
        revision,
        "57" * 32,
        umi_revision,
    )
    with pytest.raises(release_tool.ReleaseArtifactError, match="rights"):
        release_tool.create_release(arguments)


def test_release_model_archive_rejects_invalid_internals(tmp_path: Path) -> None:
    release_tool = _load("release_artifacts")
    invalid = _invalid_bundle(tmp_path, "12" * 32)
    archive = tmp_path / "invalid.zip"
    _unchecked_package(invalid, archive)
    with pytest.raises(release_tool.ReleaseArtifactError, match="strict runtime loading"):
        release_tool._validate_model_archive(archive.read_bytes(), "12" * 32)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_release_history_requires_tested_evidence_and_metadata_only_commits(
    valid_bundle: tuple[Path, str, str], tmp_path: Path
) -> None:
    package_tool = _load("package_bundle")
    release_tool = _load("release_artifacts")
    bundle, revision, rights_decision = valid_bundle
    identity = json.loads((bundle / "inference-identity.json").read_bytes())
    umi_revision = "9a" * 20
    repository = tmp_path / "repository"
    release = repository / "release"
    release.mkdir(parents=True)
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Release Test")
    _git(repository, "config", "user.email", "release@example.invalid")
    _git(repository, "config", "commit.gpgsign", "false")

    for relative in release_tool._identity_source_closure(identity):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    rebinder_relative = Path("src/bitsign_motion/local_bundle_rebind.py")
    rebinder_source = (ROOT / rebinder_relative).read_text(encoding="utf-8")
    current_base = release_tool._base_revision_from_source(rebinder_source.encode())
    (repository / rebinder_relative).write_text(
        rebinder_source.replace(current_base, revision), encoding="utf-8"
    )
    for _label, (
        release_filename,
        source_relative,
    ) in release_tool.COMPANION_ARTIFACT_SOURCES.items():
        canonical_destination = repository / source_relative
        canonical_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source_relative, canonical_destination)
        shutil.copyfile(ROOT / source_relative, release / release_filename)
    package_tool.package_bundle(bundle, release / release_tool.MODEL_FILENAME)
    for label, record in {
        "selection-ledger": make_selection(identity, revision),
        "motion-ablation-evidence": make_motion(identity, revision),
        "rights-evidence": make_rights(identity, rights_decision),
    }.items():
        (release / release_tool.EVIDENCE_FILES[label]).write_bytes(canonical_json_bytes(record))
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "tested release source and artifacts")
    tested_revision = _git(repository, "rev-parse", "HEAD")

    e2e = make_e2e(identity, revision, umi_revision, tested_revision)
    (release / release_tool.RELEASE_E2E_FILENAME).write_bytes(canonical_json_bytes(e2e))
    _git(repository, "add", f"release/{release_tool.RELEASE_E2E_FILENAME}")
    _git(repository, "commit", "-m", "record machine-derived release E2E")
    source_revision = _git(repository, "rev-parse", "HEAD")

    created = release_tool.create_release(
        _release_arguments(
            release_tool,
            release,
            revision,
            rights_decision,
            umi_revision,
            source_revision,
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
            created,
            release / "release-manifest.json",
            repository,
            "HEAD",
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


def test_runtime_file_manifest_describes_versioned_shared_source_closure() -> None:
    lines = (ROOT / "release/runtime-files.txt").read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [
        "# Reviewed source-closure allowlist for v0 and public-s1-finetune/1 runtimes.",
        "# Each release binds its exact source bytes through its inference identity.",
    ]
    assert "src/bitsign_motion/s1_portable_runtime.py" in lines


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
