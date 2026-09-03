from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import bitsign_motion.local_bundle_rebind as rebind_module
import bitsign_motion.local_extractor_release as extractor_module
from bitsign_motion.amd64_holistic_container import HolisticContainerImage
from bitsign_motion.canonical import canonical_json_bytes
from bitsign_motion.local_bundle_rebind import LocalBundleRebindError
from bitsign_motion.local_extractor_release import LocalExtractorReleaseError

from .test_release_tools import _invalid_bundle, _unchecked_package


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _local_image() -> HolisticContainerImage:
    image_id = "sha256:" + "ab" * 32
    return HolisticContainerImage(
        requested_reference=image_id,
        image_id=image_id,
        platform="linux/amd64",
    )


def _dependencies() -> dict[str, Any]:
    return {
        "ffmpeg": extractor_module.extractor.EXPECTED_FFMPEG_VERSION,
        "packages": extractor_module._locked_packages(),
        "pip_check": "passed",
    }


def test_local_build_record_binds_source_image_and_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    docker = _executable(tmp_path / "docker")
    image = _local_image()

    def builder(**kwargs: Any) -> HolisticContainerImage:
        assert kwargs["platform"] == "linux/amd64"
        assert kwargs["image_tag"] == "local:test"
        return image

    monkeypatch.setattr(
        extractor_module,
        "validate_local_extractor",
        lambda *_args, **_kwargs: (image, _dependencies()),
    )
    record = extractor_module.build_local_extractor(
        docker_executable=docker,
        image_tag="local:test",
        timeout_seconds=900,
        builder=builder,
    )
    assert record["image_id"] == image.image_id
    assert record["status"] == "component_test_no_weight"
    assert record["sources"] == extractor_module._source_record()
    assert (
        extractor_module.validate_local_extractor_record(
            record,
            docker_executable=docker,
        )
        == record
    )

    changed = copy.deepcopy(record)
    changed["sources"]["worker_sha256"] = "00" * 32
    with pytest.raises(LocalExtractorReleaseError, match="identity differs"):
        extractor_module.validate_local_extractor_record(
            changed,
            docker_executable=docker,
        )


def test_rebinder_extracts_only_the_deterministic_published_bundle(tmp_path: Path) -> None:
    archive = tmp_path / "base.zip"
    _unchecked_package(_invalid_bundle(tmp_path, "12" * 32), archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = tmp_path / "extracted"
    destination.mkdir(mode=0o700)
    rebind_module._extract_base_archive(
        archive,
        destination,
        expected_sha256=digest,
    )
    assert tuple(sorted(path.name for path in destination.iterdir())) == tuple(
        sorted(rebind_module._EXPECTED_FILES)
    )

    with pytest.raises(LocalBundleRebindError, match="digest differs"):
        rebind_module._extract_base_archive(
            archive,
            tmp_path / "unused",
            expected_sha256="00" * 32,
        )


def test_rebinder_preserves_every_identity_field_except_local_amd64_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "base.zip"
    _unchecked_package(_invalid_bundle(tmp_path, "12" * 32), archive)
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    docker = _executable(tmp_path / "docker")
    record_path = tmp_path / "local-extractor.json"
    record_path.write_bytes(canonical_json_bytes({"fixture": True}))
    source_hash = "56" * 32
    local_image = "sha256:" + "78" * 32
    build_record = {
        "image_id": local_image,
        "sources": {
            "amd64_container_host_source_sha256": source_hash,
            "requirements_sha256": source_hash,
            "worker_sha256": source_hash,
        },
    }
    monkeypatch.setattr(
        rebind_module,
        "validate_local_extractor_record",
        lambda *_args, **_kwargs: build_record,
    )
    base_identity = {
        "inference_revision": rebind_module.BASE_INFERENCE_REVISION,
        "upstream_release_identity_sha256": "90" * 32,
        "preprocessing": {
            "supported_oci_images": {
                "linux/amd64": "sha256:" + "12" * 32,
                "linux/arm64": "sha256:" + "34" * 32,
            },
            "sources": {
                "amd64_container_host_source_sha256": source_hash,
                "amd64_container_requirements_sha256": source_hash,
                "amd64_container_worker_source_sha256": source_hash,
            },
        },
        "rights": {"license": "CC-BY-SA-4.0"},
        "text_postprocess": {"maximum_words": 8},
        "model": {"state_sha256": "ab" * 32},
        "runtime": {"revision": "fixture"},
        "claim_boundary": "fixture",
    }
    derived_revision = "cd" * 32
    derived_identity = copy.deepcopy(base_identity)
    derived_identity["preprocessing"]["supported_oci_images"]["linux/amd64"] = local_image
    derived_identity["inference_revision"] = derived_revision
    loads = iter(
        (
            SimpleNamespace(identity=base_identity, _model=object()),
            SimpleNamespace(identity=derived_identity),
        )
    )
    monkeypatch.setattr(rebind_module, "load_s1_portable_bundle", lambda *_a, **_k: next(loads))

    monkeypatch.setattr(
        rebind_module,
        "_reseal_verified_base",
        lambda *_args, **_kwargs: (_args[1].mkdir() or derived_revision),
    )
    output = tmp_path / "derived"
    result = rebind_module.rebind_local_extractor(
        base_archive=archive,
        expected_base_sha256=archive_digest,
        build_record_path=record_path,
        docker_executable=docker,
        output=output,
    )
    assert result["inference_revision"] == derived_revision
    assert result["local_extractor_image_id"] == local_image


def test_identity_comparison_rejects_unrelated_derived_change() -> None:
    base = {
        "inference_revision": "12" * 32,
        "preprocessing": {"supported_oci_images": {"linux/amd64": "sha256:" + "34" * 32}},
        "model": {"state": "fixed"},
    }
    derived = copy.deepcopy(base)
    derived["inference_revision"] = "56" * 32
    derived["preprocessing"]["supported_oci_images"]["linux/amd64"] = "sha256:" + "78" * 32
    rebind_module._assert_preserved_identity(
        base,
        derived,
        image_id="sha256:" + "78" * 32,
    )
    derived["model"]["state"] = "changed"
    with pytest.raises(LocalBundleRebindError, match="other than"):
        rebind_module._assert_preserved_identity(
            base,
            derived,
            image_id="sha256:" + "78" * 32,
        )


def test_rebinder_accepts_an_explicit_nonlegacy_base_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "base.zip"
    _unchecked_package(_invalid_bundle(tmp_path, "12" * 32), archive)
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    docker = _executable(tmp_path / "docker")
    record_path = tmp_path / "local-extractor.json"
    record_path.write_bytes(canonical_json_bytes({"fixture": True}))
    source_hash = "56" * 32
    local_image = "sha256:" + "78" * 32
    base_revision = "91" * 32
    derived_revision = "cd" * 32
    base_identity = {
        "inference_revision": base_revision,
        "preprocessing": {
            "supported_oci_images": {
                "linux/amd64": "sha256:" + "12" * 32,
                "linux/arm64": "sha256:" + "34" * 32,
            },
            "sources": {
                "amd64_container_host_source_sha256": source_hash,
                "amd64_container_requirements_sha256": source_hash,
                "amd64_container_worker_source_sha256": source_hash,
            },
        },
    }
    derived_identity = copy.deepcopy(base_identity)
    derived_identity["preprocessing"]["supported_oci_images"]["linux/amd64"] = local_image
    derived_identity["inference_revision"] = derived_revision
    loads = iter(
        (
            SimpleNamespace(identity=base_identity, _model=object()),
            SimpleNamespace(identity=derived_identity),
        )
    )
    observed_revisions: list[str] = []

    def load(_root: Path, *, expected_inference_revision: str) -> SimpleNamespace:
        observed_revisions.append(expected_inference_revision)
        return next(loads)

    monkeypatch.setattr(
        rebind_module,
        "validate_local_extractor_record",
        lambda *_args, **_kwargs: {
            "image_id": local_image,
            "sources": {
                "amd64_container_host_source_sha256": source_hash,
                "requirements_sha256": source_hash,
                "worker_sha256": source_hash,
            },
        },
    )
    monkeypatch.setattr(rebind_module, "load_s1_portable_bundle", load)
    monkeypatch.setattr(
        rebind_module,
        "_reseal_verified_base",
        lambda *_args, **_kwargs: (_args[1].mkdir() or derived_revision),
    )

    result = rebind_module.rebind_local_extractor(
        base_archive=archive,
        expected_base_sha256=archive_digest,
        expected_base_inference_revision=base_revision,
        build_record_path=record_path,
        docker_executable=docker,
        output=tmp_path / "derived",
    )

    assert result["base_inference_revision"] == base_revision
    assert observed_revisions == [base_revision, derived_revision]
