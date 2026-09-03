from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
from typing import Any

from bitsign_motion import s1_portable_runtime as portable
from bitsign_motion import s1_release_evidence as evidence
from bitsign_motion.canonical import canonical_json_bytes, canonical_json_sha256


def _rational(value: Fraction) -> dict[str, str]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
    }


def _result(identity_key: str, identity: object, score: Fraction) -> dict[str, Any]:
    return {
        identity_key: identity,
        "sample_count": 285,
        "mean_wer": _rational(1 - score),
        "mean_normalized_score": _rational(score),
        "unique_output_count": 200,
    }


def _binding(role: str, schema: str, seed: int) -> dict[str, str]:
    return {
        "role": role,
        "schema": schema,
        "content_sha256": f"{seed:064x}",
        "file_sha256": f"{seed + 1:064x}",
    }


def _seal(record: dict[str, Any], domain: bytes) -> dict[str, Any]:
    record["content_sha256"] = canonical_json_sha256(record, domain=domain)
    return record


def make_selection(identity: dict[str, Any], revision: str) -> dict[str, Any]:
    model = identity["model"]
    tokenizer = identity["tokenizer"]
    runtime = identity["runtime"]
    rights = identity["rights"]
    baseline = _result("arm", "published_greedy", Fraction(1, 4))
    candidate = _result("arm", "candidate_beam", Fraction(1, 2))
    checkpoint_results = [
        _result(
            "epoch",
            epoch,
            Fraction(1, 2) if epoch == 20 else Fraction(1, epoch + 2),
        )
        for epoch in evidence._CHECKPOINT_EPOCHS
    ]
    scales = [
        ("0x0.0p+0", "0000000000000000"),
        ("0x1.0000000000000p-2", "3fd0000000000000"),
        ("0x1.0000000000000p-1", "3fe0000000000000"),
        ("0x1.0000000000000p+0", "3ff0000000000000"),
        ("0x1.0000000000000p+1", "4000000000000000"),
        ("0x1.0000000000000p+2", "4010000000000000"),
        ("0x1.0000000000000p+3", "4020000000000000"),
    ]
    guidance_results = []
    for index, (hexadecimal, binary) in enumerate(scales):
        row = _result(
            "scale_index", index, Fraction(1, 2) if index == 0 else Fraction(1, 10 + index)
        )
        row["scale"] = {
            "hexadecimal": hexadecimal,
            "ieee754_binary64_be": binary,
        }
        guidance_results.append(row)
    return _seal(
        {
            "schema": evidence.SELECTION_LEDGER_SCHEMA,
            "release_id": "umi-s1-baseline-v0",
            "purpose": "aggregate post-test-open validation selection ledger",
            "partition": "fleurs_val",
            "sample_count": 285,
            "source": {
                "source_id": rights["source_id"],
                "source_version": rights["source_version"],
                "license_id": rights["source_license_id"],
                "public_data_eligible": False,
            },
            "evaluation_context": {
                "test_partition_already_open": True,
                "test_evaluation_invocation_count": 0,
                "inference_backend": "mps",
                "machine_architecture": "arm64",
                "input_materialization": "bound-arm64-validation-motion-tensors",
                "linux_amd64_end_to_end_quality_evidence": False,
            },
            "private_reports": [
                _binding(
                    "decoder_comparison",
                    "umi-s1-validation-decoder-comparison/1",
                    1,
                ),
                _binding(
                    "checkpoint_sweep",
                    "umi-s1-validation-checkpoint-sweep/1",
                    3,
                ),
                _binding(
                    "motion_guidance_sweep",
                    "umi-s1-validation-motion-guidance-sweep/1",
                    5,
                ),
            ],
            "release_candidate": {
                "release_inference_revision": revision,
                "evaluated_inference_revision": revision,
                "runtime_revision": runtime["revision"],
                "selected_epoch": 20,
                "selected_model_state_sha256": model["state_sha256"],
                "tokenizer_model_sha256": tokenizer["model_sha256"],
                "tokenizer_record_sha256": tokenizer["record_sha256"],
                "evaluation_transfer_basis": (
                    "same-model-state-tokenizer-runtime-and-decoder-contract"
                ),
            },
            "decoder_selection": {
                "profiles": [
                    "published-greedy-max128-eight-word-cap",
                    "beam2-max24-no-repeat3-eight-word-cap",
                ],
                "results": [baseline, candidate],
                "paired_counts": {
                    "candidate_better": 100,
                    "equal": 150,
                    "candidate_worse": 35,
                },
                "candidate_minus_baseline": _rational(Fraction(1, 4)),
                "relative_improvement_over_baseline": _rational(Fraction(1)),
                "selected_profile": "beam2-max24-no-repeat3-eight-word-cap",
            },
            "checkpoint_selection": {
                "rule": "highest-exact-validation-mean-then-lowest-epoch",
                "results": checkpoint_results,
                "selected_epoch": 20,
                "selected_model_state_sha256": model["state_sha256"],
            },
            "guidance_selection": {
                "rule": "highest-exact-validation-mean-then-lowest-scale",
                "results": guidance_results,
                "selected_scale_index": 0,
                "scale_zero_matches_portable_decoder": True,
                "scale_zero_match_count": 285,
            },
            "public_data": {
                "aggregate_only": True,
                "source_rows_included": False,
                "per_example_outputs_included": False,
            },
            "claim_boundary": evidence._SELECTION_CLAIM_BOUNDARY,
        },
        evidence._SELECTION_DOMAIN,
    )


def make_motion(identity: dict[str, Any], revision: str) -> dict[str, Any]:
    model = identity["model"]
    tokenizer = identity["tokenizer"]
    real = Fraction(1, 3)
    zero = Fraction(1, 2)
    deranged = Fraction(1, 4)
    return _seal(
        {
            "schema": evidence.MOTION_ABLATION_SCHEMA,
            "release_id": "umi-s1-baseline-v0",
            "purpose": "aggregate post-test-open validation motion diagnostic",
            "partition": "fleurs_val",
            "sample_count": 285,
            "private_report": _binding("motion_ablation", "umi-s1-validation-motion-ablation/2", 7),
            "release_candidate": {
                "release_inference_revision": revision,
                "selected_model_state_sha256": model["state_sha256"],
                "tokenizer_model_sha256": tokenizer["model_sha256"],
                "tokenizer_record_sha256": tokenizer["record_sha256"],
                "decoder_profile": "beam2-max24-no-repeat3-eight-word-cap",
            },
            "evaluation_context": {
                "test_partition_already_open": True,
                "test_evaluation_invocation_count": 0,
                "inference_backend": "mps",
                "machine_architecture": "arm64",
                "input_materialization": "bound-arm64-validation-motion-tensors",
                "linux_amd64_end_to_end_quality_evidence": False,
            },
            "conditions": [
                _result("condition", "real_motion", real),
                _result("condition", "zero_motion", zero),
                _result("condition", "deranged_motion", deranged),
            ],
            "effects": {
                "real_minus_zero": _rational(real - zero),
                "real_minus_deranged": _rational(real - deranged),
            },
            "paired_counts": {
                "real_vs_zero": {
                    "real_better": 50,
                    "equal": 127,
                    "zero_better": 108,
                },
                "real_vs_deranged": {
                    "real_better": 98,
                    "equal": 114,
                    "deranged_better": 73,
                },
            },
            "conclusions": {
                "zero_motion_has_higher_aggregate_score_than_real_motion": True,
                "real_motion_has_higher_aggregate_score_than_deranged_motion": True,
                "useful_motion_grounding_established": False,
            },
            "public_data": {
                "aggregate_only": True,
                "source_rows_included": False,
                "per_example_outputs_included": False,
            },
            "claim_boundary": evidence._MOTION_CLAIM_BOUNDARY,
        },
        evidence._MOTION_DOMAIN,
    )


def make_rights(identity: dict[str, Any], rights_decision: str) -> dict[str, Any]:
    rights = identity["rights"]
    return _seal(
        {
            "schema": evidence.RIGHTS_EVIDENCE_SCHEMA,
            "release_id": "umi-s1-baseline-v0",
            "decision_authority": "project-owner-release-decision/1",
            "private_review": {
                "role": "final_rights_review",
                "schema": "umi-s1-final-rights-review/1",
                "content_sha256": rights_decision,
                "file_sha256": "09" * 32,
            },
            "release_terms": {
                "review_class": "project-release-decision-not-legal-opinion",
                "runtime_code_license": rights["runtime_code_license"],
                "weight_license": rights["intended_public_weight_license"],
                "public_weight_redistribution_approved": True,
                "raw_source_data_redistribution_approved": False,
                "source_annotations_redistribution_approved": False,
                "mediapipe_task_redistribution_approved": False,
            },
            "sources": [
                {
                    "source_id": rights["source_id"],
                    "source_version": rights["source_version"],
                    "license_id": rights["source_license_id"],
                    "public_data_eligible": False,
                    "public_weight_eligible": True,
                    "public_data_release": False,
                    "raw_data_release": False,
                },
                {
                    "source_id": "fsboard-v3",
                    "source_version": "kaggle-v13-2025-07-26",
                    "license_id": "CC-BY-4.0",
                    "public_data_eligible": False,
                    "public_weight_eligible": True,
                    "public_data_release": False,
                    "raw_data_release": False,
                },
            ],
            "claim_boundary": evidence._RIGHTS_CLAIM_BOUNDARY,
        },
        evidence._RIGHTS_DOMAIN,
    )


def make_e2e(
    identity: dict[str, Any],
    revision: str,
    umi_revision: str,
    tested_revision: str = "13" * 20,
) -> dict[str, Any]:
    extractor_image_id = "sha256:" + "12" * 32
    rebound = copy.deepcopy(identity)
    del rebound["inference_revision"]
    rebound["preprocessing"]["supported_oci_images"]["linux/amd64"] = extractor_image_id
    derived_revision = canonical_json_sha256(rebound, domain=portable._IDENTITY_DOMAIN)
    capture = {
        "release_id": "umi-s1-baseline-v0",
        "status": "passed",
        "base_inference_revision": revision,
        "derived_inference_revision": derived_revision,
        "tested_reference_model_git_revision": tested_revision,
        "umi_git_revision": umi_revision,
        "fixture": {
            "fixture_class": "rights-cleared-private-video",
            "distributed": False,
            "rights_cleared_for_private_testing": True,
            "video_sha256": "14" * 32,
        },
        "started_at_utc": "2026-09-02T00:00:00Z",
        "finished_at_utc": "2026-09-02T00:01:00Z",
        "runtime": {
            "host_operating_system": "Linux",
            "host_architecture": "x86_64",
            "container_platform": "linux/amd64",
            "model_device": "cpu",
            "python_version": "3.12.11",
            "torch_version": "2.7.0",
            "numpy_version": "2.3.5",
            "safetensors_version": "0.8.0",
            "bittensor_version": "9.10.1",
            "docker_engine_version": "client=28.3.3;server=28.3.3",
            "extractor_image_id": extractor_image_id,
            "mediapipe_task_model_sha256": evidence._TASK_MODEL_SHA256,
        },
        "timeouts_seconds": {
            "inner_model_hard_deadline": 150,
            "outer_inference_timeout": 180,
            "outer_admission_timeout": 10,
            "outer_lifecycle_timeout": 60,
        },
        "execution": {
            "request_count": 2,
            "valid_video_status": "ok",
            "invalid_video_status": "error",
            "invalid_video_error_code": "inference_failed",
            "signed_envelopes_verified": True,
            "plaintext_unavailable_before_reveal": True,
            "decryption_after_reveal_verified": True,
            "response_bindings_verified": True,
            "maximum_output_words": 8,
            "output_limit_passed": True,
            "temporary_job_cleanup_passed": True,
            "extractor_container_cleanup_passed": True,
            "video_fixture_distributed": False,
            "translation_quality_measured": False,
        },
        "evidence": {
            "extractor_build_record_content_sha256": "15" * 32,
            "extractor_build_record_file_sha256": "16" * 32,
            "valid_wire_response_sha256": "17" * 32,
            "invalid_wire_response_sha256": "18" * 32,
            "post_reveal_plaintext_set_sha256": "19" * 32,
            "run_log_sha256": "1a" * 32,
        },
    }
    private = evidence.seal_private_release_e2e(capture)
    public = {
        "schema": evidence.RELEASE_E2E_SCHEMA,
        "release_id": private["release_id"],
        "status": private["status"],
        "base_inference_revision": private["base_inference_revision"],
        "derived_inference_revision": private["derived_inference_revision"],
        "tested_reference_model_git_revision": tested_revision,
        "umi_git_revision": private["umi_git_revision"],
        "private_run": {
            "role": "release_e2e",
            "schema": evidence.RELEASE_E2E_RUN_SCHEMA,
            "content_sha256": private["content_sha256"],
            "file_sha256": "1b" * 32,
        },
        "test_fixture": {
            "fixture_class": "rights-cleared-private-video",
            "distributed": False,
            "source_identity_published": False,
        },
        "started_at_utc": private["started_at_utc"],
        "finished_at_utc": private["finished_at_utc"],
        "runtime": private["runtime"],
        "timeouts_seconds": private["timeouts_seconds"],
        "execution": private["execution"],
        "claim_boundary": evidence._E2E_CLAIM_BOUNDARY,
    }
    return _seal(public, evidence._E2E_DOMAIN)


def write_evidence_set(
    root: Path,
    identity: dict[str, Any],
    revision: str,
    rights_decision: str,
    umi_revision: str,
    *,
    tested_revision: str = "13" * 20,
) -> dict[str, dict[str, Any]]:
    records = {
        "selection-ledger": make_selection(identity, revision),
        "motion-ablation-evidence": make_motion(identity, revision),
        "rights-evidence": make_rights(identity, rights_decision),
        "release-e2e-evidence": make_e2e(identity, revision, umi_revision, tested_revision),
    }
    for label, record in records.items():
        (root / evidence.EVIDENCE_FILES[label]).write_bytes(canonical_json_bytes(record))
    return records
