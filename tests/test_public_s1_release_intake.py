from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from bitsign_motion import public_s1_release_intake as intake
from bitsign_motion import s1_portable_runtime as portable
from bitsign_motion.canonical import canonical_json_bytes, canonical_json_sha256
from bitsign_motion.portable_model import PortableS1, PortableS1Config
from bitsign_motion.s1_portable_runtime import export_s1_portable_bundle
from bitsign_motion.s1_state_digest import s1_tensor_set_sha256

from .test_s1_portable_runtime import (
    _preprocessing,
    _public_finetune_claim_fixture,
    _synthetic_tokenizer,
)


def _write(path: Path, value: dict[str, Any]) -> bytes:
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    return payload


def _seal(value: dict[str, Any], domain: bytes) -> None:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_json_sha256(value, domain=domain)


def _source_license_payload(source_id: str) -> bytes:
    return f"Fixture license for {source_id}.\n".encode()


def _sources() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, source_id in enumerate(intake._EXPECTED_SOURCE_IDS, start=1):
        notice = f"Fixture attribution for {source_id}. No source records are distributed."
        rows.append(
            {
                "source_id": source_id,
                "source_version": f"fixture-v{index}",
                "source_name": f"Fixture source {index}",
                "source_entry_sha256": f"{index:02x}" * 32,
                "license_id": "CC-BY-SA-4.0" if index in (1, 2) else "CC-BY-4.0",
                "license_sha256": hashlib.sha256(_source_license_payload(source_id)).hexdigest(),
                "terms_sha256": f"{index + 32:02x}" * 32,
                "source_use_policy_sha256": f"{index + 48:02x}" * 32,
                "attribution_notice": notice,
                "attribution_notice_sha256": hashlib.sha256(notice.encode()).hexdigest(),
                "public_weight_eligible": True,
                "public_data_eligible": False,
                "raw_data_release": False,
                "public_data_release": False,
            }
        )
    return rows


def _policy_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        field: source[field]
        for field in (
            "source_id",
            "source_name",
            "source_version",
            "license_id",
            "license_sha256",
            "terms_sha256",
            "source_use_policy_sha256",
            "attribution_notice_sha256",
            "source_entry_sha256",
        )
    }


def _build_release(root: Path) -> tuple[Path, Path]:
    source = root / "source-release"
    source.mkdir()
    tokenizer_model, tokenizer_record = _synthetic_tokenizer()
    tokenizer_model_path = root / "tokenizer.model"
    tokenizer_record_path = root / "tokenizer.json"
    tokenizer_model_path.write_bytes(tokenizer_model)
    tokenizer_record_path.write_bytes(tokenizer_record)
    model = PortableS1(PortableS1Config()).eval()
    model_state_sha256 = s1_tensor_set_sha256(model.state_dict())
    identity, arguments = _public_finetune_claim_fixture()
    identity["candidate"]["selected_model_state_sha256"] = model_state_sha256
    identity["candidate"]["training_authority_file_sha256"] = identity["candidate"][
        "training_authority_sha256"
    ]
    identity["text_pretraining"]["training_authority_file_sha256"] = identity["text_pretraining"][
        "training_authority_sha256"
    ]
    identity["text_pretraining"]["claim"] = intake._TEXT_PRETRAINING_CLAIM
    identity["tokenizer"]["model_sha256"] = hashlib.sha256(tokenizer_model).hexdigest()
    identity["tokenizer"]["record_sha256"] = hashlib.sha256(tokenizer_record).hexdigest()
    identity["base_preprocessing"]["preprocessing_sha256"] = canonical_json_sha256(_preprocessing())
    sources = _sources()
    by_id = {row["source_id"]: row for row in sources}
    identity["motion_sources"]["fleurs"]["source_entry_sha256"] = by_id["fleurs-asl-v1"][
        "source_entry_sha256"
    ]
    identity["motion_sources"]["two_m_flores"]["source_entry_sha256"] = by_id[
        "facebook/2M-Flores-ASL"
    ]["source_entry_sha256"]
    identity["text_pretraining"]["source_entry_sha256"] = by_id[
        "google-research-datasets/taskmaster/TM-1-2019"
    ]["source_entry_sha256"]
    review: dict[str, Any] = {
        "schema": intake.PUBLIC_S1_FINETUNE_RELEASE_REVIEW_SCHEMA,
        "effective_at": "2020-01-01T00:00:00Z",
        "decision_authority": intake._RIGHTS_DECISION_AUTHORITY,
        "review_class": intake._RIGHTS_REVIEW_CLASS,
        "candidate": {
            field: identity["candidate"][field]
            for field in (
                "training_authority_sha256",
                "training_authority_content_sha256",
                "final_report_sha256",
                "final_report_content_sha256",
                "selected_epoch",
                "selected_model_state_sha256",
                "selected_checkpoint_manifest_sha256",
            )
        },
        "sources": sources,
        "release_terms": {
            "public_weight_redistribution_approved": True,
            "weight_license": intake._WEIGHT_LICENSE,
            "runtime_code_license": intake._RUNTIME_CODE_LICENSE,
            "raw_source_data_redistribution_approved": False,
            "source_annotations_redistribution_approved": False,
            "tokenizer_artifacts_included": True,
        },
        "claim_boundary": intake._RIGHTS_CLAIM_BOUNDARY,
    }
    _seal(review, intake._REVIEW_DOMAIN)
    review_sha256 = canonical_json_sha256(review)
    rights = {
        "schema": portable.S1_PORTABLE_MULTI_SOURCE_RIGHTS_SCHEMA,
        "sources": sources,
        "rights_as_of": review["effective_at"],
        "runtime_code_license": intake._RUNTIME_CODE_LICENSE,
        "intended_public_weight_license": intake._WEIGHT_LICENSE,
        "redistribution_blocked_pending_final_rights_review": False,
        "release_rights_decision_sha256": review_sha256,
        "upstream_rights_sha256": review_sha256,
        "claim_boundary": intake._RIGHTS_RECORD_CLAIM_BOUNDARY,
    }
    identity["rights_review_sha256"] = review_sha256
    identity["rights_record_sha256"] = canonical_json_sha256(rights)
    _seal(identity, intake._IDENTITY_DOMAIN)
    arguments.update(
        {
            "model_state_sha256": model_state_sha256,
            "upstream_release_identity_sha256": canonical_json_sha256(identity),
            "tokenizer_model_sha256": hashlib.sha256(tokenizer_model).hexdigest(),
            "tokenizer_record_sha256": hashlib.sha256(tokenizer_record).hexdigest(),
            "rights_sha256": canonical_json_sha256(rights),
            "claim_boundary_evidence": identity,
        }
    )
    portable_result = export_s1_portable_bundle(
        model,
        source / "portable",
        tokenizer_model_path=tokenizer_model_path,
        tokenizer_record_path=tokenizer_record_path,
        preprocessing=_preprocessing(),
        rights=rights,
        upstream_release_identity_sha256=arguments["upstream_release_identity_sha256"],
        text_postprocess=arguments["text_postprocess"],
        claim_boundary_profile=portable.S1_PUBLIC_FINETUNE_CLAIM_PROFILE,
        claim_boundary_evidence=identity,
    )
    identity_bytes = _write(source / "release-identity.json", identity)
    review_bytes = _write(source / "rights-review.json", review)
    manifest: dict[str, Any] = {
        "schema": intake.PUBLIC_S1_FINETUNE_RELEASE_MANIFEST_SCHEMA,
        "release_identity": {
            "sha256": hashlib.sha256(identity_bytes).hexdigest(),
            "content_sha256": identity["content_sha256"],
            "size_bytes": len(identity_bytes),
        },
        "rights_review": {
            "sha256": hashlib.sha256(review_bytes).hexdigest(),
            "content_sha256": review["content_sha256"],
            "size_bytes": len(review_bytes),
        },
        "portable": portable_result,
        "raw_data_included": False,
    }
    _seal(manifest, intake._MANIFEST_DOMAIN)
    manifest_bytes = _write(source / "release-manifest.json", manifest)
    policy: dict[str, Any] = {
        "schema": intake.PUBLIC_S1_FINETUNE_INTAKE_POLICY_SCHEMA,
        "release_profile": portable.S1_PUBLIC_FINETUNE_CLAIM_PROFILE,
        "source_release": {
            "release_identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
            "rights_review_sha256": hashlib.sha256(review_bytes).hexdigest(),
            "release_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "gate": identity["gate"]["gate"],
        "tokenizer": identity["tokenizer"],
        "base_preprocessing": identity["base_preprocessing"],
        "sources": [_policy_source(row) for row in sources],
        "runtime_code_license": intake._RUNTIME_CODE_LICENSE,
        "weight_license": intake._WEIGHT_LICENSE,
    }
    _seal(policy, intake._POLICY_DOMAIN)
    policy_path = root / "intake-policy.json"
    _write(policy_path, policy)
    return source, policy_path


@pytest.fixture(scope="module")
def public_release(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return _build_release(tmp_path_factory.mktemp("public-s1-intake"))


def _rebuild_manifest(source: Path) -> None:
    identity = json.loads((source / "release-identity.json").read_bytes())
    review = json.loads((source / "rights-review.json").read_bytes())
    manifest = json.loads((source / "release-manifest.json").read_bytes())
    identity_bytes = canonical_json_bytes(identity)
    review_bytes = canonical_json_bytes(review)
    manifest["release_identity"] = {
        "sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "content_sha256": identity["content_sha256"],
        "size_bytes": len(identity_bytes),
    }
    manifest["rights_review"] = {
        "sha256": hashlib.sha256(review_bytes).hexdigest(),
        "content_sha256": review["content_sha256"],
        "size_bytes": len(review_bytes),
    }
    _seal(manifest, intake._MANIFEST_DOMAIN)
    _write(source / "release-manifest.json", manifest)


def test_intake_projects_only_aggregate_records_and_is_deterministic(
    public_release: tuple[Path, Path], tmp_path: Path
) -> None:
    source, policy = public_release
    first = tmp_path / "first"
    second = tmp_path / "second"

    result = intake.intake_public_s1_finetune_release(source, first, policy_path=policy)
    repeated = intake.intake_public_s1_finetune_release(source, second, policy_path=policy)

    assert result == repeated
    assert intake.verify_public_s1_finetune_intake(first, policy_path=policy) == result
    assert sorted(path.name for path in first.iterdir()) == list(intake._EXPECTED_INTAKE_ROOT)
    assert (first / intake.PUBLIC_S1_FINETUNE_ARCHIVE_FILENAME).read_bytes() == (
        second / intake.PUBLIC_S1_FINETUNE_ARCHIVE_FILENAME
    ).read_bytes()
    evidence = json.loads((first / intake.PUBLIC_S1_FINETUNE_EVIDENCE_FILENAME).read_bytes())
    intake._assert_aggregate_safe(evidence)
    assert evidence["release_identity"]["release_contents"] == {
        "safe_tensor_weights": True,
        "tokenizer_artifacts": True,
        "raw_training_or_validation_data": False,
        "source_video_or_annotations": False,
        "prediction_plaintext": False,
    }
    assert evidence["portable_archive"]["inference_revision"] == result["inference_revision"]


def test_intake_rejects_unexpected_source_content(
    public_release: tuple[Path, Path], tmp_path: Path
) -> None:
    source, policy = public_release
    mutated = tmp_path / "source"
    shutil.copytree(source, mutated)
    (mutated / "predictions.json").write_text("[]", encoding="utf-8")

    with pytest.raises(intake.PublicS1ReleaseIntakeError, match="inventory"):
        intake.intake_public_s1_finetune_release(mutated, tmp_path / "output", policy_path=policy)


def test_intake_rejects_symlinked_source_ancestor(
    public_release: tuple[Path, Path], tmp_path: Path
) -> None:
    source, policy = public_release
    linked = tmp_path / "linked-source"
    linked.symlink_to(source, target_is_directory=True)

    with pytest.raises(intake.PublicS1ReleaseIntakeError, match="contains a symlink"):
        intake.intake_public_s1_finetune_release(linked, tmp_path / "output", policy_path=policy)


def test_intake_rejects_resealed_outer_identity_not_bound_by_portable(
    public_release: tuple[Path, Path], tmp_path: Path
) -> None:
    source, policy = public_release
    mutated = tmp_path / "source"
    shutil.copytree(source, mutated)
    identity = json.loads((mutated / "release-identity.json").read_bytes())
    identity["candidate"]["selected_epoch"] += 1
    _seal(identity, intake._IDENTITY_DOMAIN)
    _write(mutated / "release-identity.json", identity)
    _rebuild_manifest(mutated)

    with pytest.raises(intake.PublicS1ReleaseIntakeError, match="trusted intake policy"):
        intake.intake_public_s1_finetune_release(mutated, tmp_path / "output", policy_path=policy)


def test_intake_rejects_candidate_selected_gate_not_authorized_by_policy(
    public_release: tuple[Path, Path], tmp_path: Path
) -> None:
    source, policy_path = public_release
    policy = copy.deepcopy(json.loads(policy_path.read_bytes()))
    policy["gate"]["minimum_real_score"] = {"numerator": "1", "denominator": "100"}
    _seal(policy, intake._POLICY_DOMAIN)
    changed_policy = tmp_path / "policy.json"
    _write(changed_policy, policy)

    with pytest.raises(intake.PublicS1ReleaseIntakeError, match="gate differs"):
        intake.intake_public_s1_finetune_release(
            source, tmp_path / "output", policy_path=changed_policy
        )


def test_intake_rejects_unbound_rights_review(
    public_release: tuple[Path, Path], tmp_path: Path
) -> None:
    source, policy = public_release
    mutated = tmp_path / "source"
    shutil.copytree(source, mutated)
    review = json.loads((mutated / "rights-review.json").read_bytes())
    review["effective_at"] = "2020-01-02T00:00:00Z"
    _seal(review, intake._REVIEW_DOMAIN)
    _write(mutated / "rights-review.json", review)
    _rebuild_manifest(mutated)

    with pytest.raises(intake.PublicS1ReleaseIntakeError, match="trusted intake policy"):
        intake.intake_public_s1_finetune_release(mutated, tmp_path / "output", policy_path=policy)


def test_intake_rejects_archive_mutation_and_output_collision(
    public_release: tuple[Path, Path], tmp_path: Path
) -> None:
    source, policy = public_release
    output = tmp_path / "output"
    intake.intake_public_s1_finetune_release(source, output, policy_path=policy)
    archive = output / intake.PUBLIC_S1_FINETUNE_ARCHIVE_FILENAME
    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 1
    archive.write_bytes(raw)
    with pytest.raises(intake.PublicS1ReleaseIntakeError):
        intake.verify_public_s1_finetune_intake(output, policy_path=policy)
    with pytest.raises(intake.PublicS1ReleaseIntakeError, match="already exists"):
        intake.intake_public_s1_finetune_release(source, output, policy_path=policy)


def test_atomic_publication_never_replaces_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    marker = destination / "owner-data"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(intake.PublicS1ReleaseIntakeError, match="already exists"):
        intake._rename_no_replace(source, destination)

    assert source.is_dir()
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_verifier_rejects_resealed_embedded_policy_substitution(
    public_release: tuple[Path, Path], tmp_path: Path
) -> None:
    source, policy = public_release
    output = tmp_path / "output"
    intake.intake_public_s1_finetune_release(source, output, policy_path=policy)
    evidence_path = output / intake.PUBLIC_S1_FINETUNE_EVIDENCE_FILENAME
    evidence = json.loads(evidence_path.read_bytes())
    evidence["intake_policy"]["gate"]["minimum_real_score"] = {
        "numerator": "1",
        "denominator": "100",
    }
    _seal(evidence["intake_policy"], intake._POLICY_DOMAIN)
    _seal(evidence, intake._INTAKE_DOMAIN)
    _write(evidence_path, evidence)

    with pytest.raises(intake.PublicS1ReleaseIntakeError, match="trusted policy"):
        intake.verify_public_s1_finetune_intake(output, policy_path=policy)


def test_intake_policy_pins_bounded_source_name(
    public_release: tuple[Path, Path], tmp_path: Path
) -> None:
    source, policy_path = public_release
    policy = copy.deepcopy(json.loads(policy_path.read_bytes()))
    policy["sources"][0]["source_name"] = "Changed reviewed source name"
    _seal(policy, intake._POLICY_DOMAIN)
    changed_policy = tmp_path / "policy.json"
    _write(changed_policy, policy)

    with pytest.raises(intake.PublicS1ReleaseIntakeError, match="source rights differ"):
        intake.intake_public_s1_finetune_release(
            source, tmp_path / "output", policy_path=changed_policy
        )

    policy["sources"][0]["source_name"] = "x" * 513
    _seal(policy, intake._POLICY_DOMAIN)
    _write(changed_policy, policy)
    with pytest.raises(intake.PublicS1ReleaseIntakeError, match="source name is invalid"):
        intake.load_public_s1_finetune_intake_policy(changed_policy)
