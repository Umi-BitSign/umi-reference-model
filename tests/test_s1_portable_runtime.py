from __future__ import annotations

import copy
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save as save_safetensors

import bitsign_motion.s1_portable_runtime as portable_module
from bitsign_motion.canonical import canonical_json_bytes, canonical_json_sha256
from bitsign_motion.portable_model import MOTION_FEATURE_DIM, PortableS1, PortableS1Config
from bitsign_motion.s1_decode_tokenizer import S1DecodeTokenizer, S1DecodeTokenizerError
from bitsign_motion.s1_portable_runtime import (
    S1_EXPERIMENT_BOOTSTRAP_CLAIM_PROFILE,
    S1_PUBLIC_FINETUNE_CLAIM_PROFILE,
    S1PortableError,
    _portable_claim_boundary,
    build_s1_authority_word_prefix_contract,
    build_s1_multi_source_rights_record,
    build_s1_preprocessing_contract,
    build_s1_rights_record,
    build_s1_validation_word_prefix_contract,
    export_s1_portable_bundle,
    load_s1_portable_bundle,
    seal_s1_validation_word_prefix_selection,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fraction(numerator: int, denominator: int) -> dict[str, str]:
    return {"numerator": str(numerator), "denominator": str(denominator)}


def _synthetic_tokenizer() -> tuple[bytes, bytes]:
    model = b"synthetic-fixed-s1-sentencepiece-model"
    pieces: list[dict[str, object]] = []
    for identifier in range(4_096):
        if identifier == 0:
            text, byte, control, unknown = "<pad>", False, True, False
        elif identifier == 1:
            text, byte, control, unknown = "<bos>", False, True, False
        elif identifier == 2:
            text, byte, control, unknown = "<eos>", False, True, False
        elif identifier == 3:
            text, byte, control, unknown = "<unk>", False, False, True
        elif identifier < 260:
            text, byte, control, unknown = f"<0x{identifier - 4:02X}>", True, False, False
        else:
            text, byte, control, unknown = f"▁piece{identifier}", False, False, False
        pieces.append(
            {
                "byte": byte,
                "control": control,
                "id": identifier,
                "piece": text,
                "score": -identifier,
                "unknown": unknown,
                "unused": False,
            }
        )
    record = {
        "schema": "umi-unigram-tokenizer/1",
        "configuration": {
            "vocabulary_size": 4_096,
            "max_output_tokens": 128,
            "byte_fallback": True,
        },
        "corpus_sha256": "11" * 32,
        "model_sha256": _sha256(model),
        "normalization_revision": "synthetic-normalization/1",
        "pieces": pieces,
        "reserved_ids": {"bos": 1, "eos": 2, "pad": 0, "unk": 3},
        "sentence_count": 1,
        "training_partition_sha256": "22" * 32,
    }
    return model, canonical_json_bytes(record)


def _preprocessing() -> dict[str, Any]:
    return build_s1_preprocessing_contract(
        extractor_image_sha256="b9054e439a0e259b4905356ece9d8368aec6c0e3af9a2029ebe369c919555a57",
        amd64_extractor_image_sha256=(
            "a36ae75cd3b096dc5859e6c470928c978db3263277c8dc3249a3c1b104a9cacb"
        ),
        extractor_model_sha256="e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8",
        mapping_profile="normalized-monocular-3d/2-ex203-candidate",
        mapping_policy_sha256="dda0ea20afe7934e7636fc8b62845df7fc40d42d98abcd5530379cb72dd04b3e",
        motion_feature_profile="normalized-monocular-3d/1",
        motion_feature_policy_sha256="8abfb7ef2fc2a88c768ee2534a3b194d6702bfe3f60472a62acf48296f335147",
    )


def _rights() -> dict[str, Any]:
    source = {
        "source_id": "synthetic-source",
        "source_version": "1",
        "license_id": "CC-BY-SA-4.0",
        "license_sha256": "31" * 32,
        "terms_sha256": "32" * 32,
        "source_use_policy_sha256": "33" * 32,
        "attribution_notice_sha256": "34" * 32,
        "source_entry_sha256": "35" * 32,
        "rights_as_of": "2026-09-02T00:00:00Z",
        "public_weight_eligible": True,
        "public_data_eligible": False,
    }
    return build_s1_rights_record(
        source=source,
        runtime_code_license="Apache-2.0",
        intended_public_weight_license="CC-BY-SA-4.0",
        redistribution_blocked_pending_final_rights_review=True,
        release_rights_decision_sha256=None,
        upstream_rights={"fixture": True},
    )


def _experiment_rights() -> dict[str, Any]:
    return {
        "schema": "umi-s1-portable-multi-source-rights/1",
        "sources": [
            {
                "source_id": "fleurs-asl-v1",
                "source_version": "kaggle-v2-2024-08-29",
                "source_name": "FLEURS-ASL",
                "source_entry_sha256": (
                    "0431b6492b9d1a67d2c78d93ecaf69024c9313daa5972bb33b93ed566577be35"
                ),
                "license_id": "CC-BY-SA-4.0",
                "license_sha256": (
                    "23ee78c8bae49cf08ea2f0c84945c66b987ebe4520881fb51b3dad4fb43d07c2"
                ),
                "terms_sha256": (
                    "ff9e2ea4bcaf5534cbd0bfd564216d5d4428fd4cd40c88105ec277b3b2c5e81b"
                ),
                "source_use_policy_sha256": (
                    "47f3286fe98be02ce503cdf5d7c4814a28b19610e507ba853553c50b2dac2e05"
                ),
                "attribution_notice": (
                    "FLEURS-ASL by Garrett Tanzer and Google AI is licensed under CC BY-SA 4.0 "
                    "(https://creativecommons.org/licenses/by-sa/4.0/). Source: "
                    "https://www.kaggle.com/datasets/googleai/fleurs-asl. Cite Garrett Tanzer, "
                    '"FLEURS-ASL: Including American Sign Language in Massively Multilingual '
                    'Multitask Evaluation," arXiv:2408.13585 (2024). Changes: UMI selects '
                    "2-to-15-second segments, derives skeletal motion representations, trains "
                    "model weights, and does not redistribute source videos or annotations. "
                    "Dependent public model weights are licensed under CC BY-SA 4.0 pending "
                    "final rights review."
                ),
                "attribution_notice_sha256": (
                    "df4d82cf63e09d0513ac08ab4ff01141583ea7774b6abfe2bad2a99c6b8bebe3"
                ),
                "public_weight_eligible": True,
                "public_data_eligible": False,
                "raw_data_release": False,
                "public_data_release": False,
            },
            {
                "source_id": "fsboard-v3",
                "source_version": "kaggle-v13-2025-07-26",
                "source_name": "FSboard (daun_v3 and dmk_v3)",
                "source_entry_sha256": (
                    "a0419739fe4cf40eaf06db604bbced794504dcaf8925596c2ef73a88ef1af3c5"
                ),
                "license_id": "CC-BY-4.0",
                "license_sha256": (
                    "96218feb836f9005261b0346501f349d11e4019deb6bbfce899483d94639c7dd"
                ),
                "terms_sha256": (
                    "4cac0eacba5f92084e440ea2a378e5f761b30119a616b5da8b263e754b98c1a4"
                ),
                "source_use_policy_sha256": (
                    "02edff2779b75ab6fffec2b4bab4ddecbb4304904eb0aa008e084d7d12f41884"
                ),
                "attribution_notice": (
                    "FSboard dataset Copyright 2025 Google, licensed under CC BY 4.0 "
                    "(https://creativecommons.org/licenses/by/4.0/). Cite Manfred Georg et al., "
                    '"FSboard: Over 3 million characters of ASL fingerspelling collected via '
                    'smartphones," arXiv:2407.15806 (2024). Changes: UMI filters sensitive and '
                    "over-length records, uses MediaPipe landmark sequences to train model "
                    "weights, and does not redistribute FSboard records."
                ),
                "attribution_notice_sha256": (
                    "0cfe5b50a410a6fd9d2a6101de37cc9c0a98e1478761b1bf4e5acf9d0a7cd88f"
                ),
                "public_weight_eligible": True,
                "public_data_eligible": False,
                "raw_data_release": False,
                "public_data_release": False,
            },
        ],
        "rights_as_of": "2026-09-03T08:00:00Z",
        "runtime_code_license": "Apache-2.0",
        "intended_public_weight_license": "CC-BY-SA-4.0",
        "redistribution_blocked_pending_final_rights_review": False,
        "release_rights_decision_sha256": (
            "95205d508c43a41dafc232fe7ed81c73f1e6c65ae962dc8857ee179271e0ff2b"
        ),
        "upstream_rights_sha256": (
            "dfbbcb433e8e6038f59d2eebbb16eedde67816e10be74d78457e3f8060b35ea1"
        ),
        "claim_boundary": (
            "The portable artifact carries weights, tokenizer data, and readable attribution "
            "for every model source. It grants no right to redistribute source videos or "
            "annotations. Runtime code and model-weight distribution remain governed by every "
            "recorded source license and the final review state."
        ),
    }


def test_cleared_portable_rights_require_eligible_sources_and_a_review_digest() -> None:
    source = {
        "source_id": "synthetic-source",
        "source_version": "1",
        "license_id": "CC-BY-SA-4.0",
        "license_sha256": "31" * 32,
        "terms_sha256": "32" * 32,
        "source_use_policy_sha256": "33" * 32,
        "attribution_notice_sha256": "34" * 32,
        "source_entry_sha256": "35" * 32,
        "rights_as_of": "2026-09-02T00:00:00Z",
        "public_weight_eligible": True,
        "public_data_eligible": False,
    }
    cleared = build_s1_rights_record(
        source=source,
        runtime_code_license="Apache-2.0",
        intended_public_weight_license="CC-BY-SA-4.0",
        redistribution_blocked_pending_final_rights_review=False,
        release_rights_decision_sha256="36" * 32,
        upstream_rights={"fixture": True},
    )
    assert cleared["redistribution_blocked_pending_final_rights_review"] is False
    assert cleared["release_rights_decision_sha256"] == "36" * 32

    with pytest.raises(S1PortableError, match="release rights decision"):
        build_s1_rights_record(
            source=source,
            runtime_code_license="Apache-2.0",
            intended_public_weight_license="CC-BY-SA-4.0",
            redistribution_blocked_pending_final_rights_review=False,
            release_rights_decision_sha256=None,
            upstream_rights={"fixture": True},
        )
    ineligible = dict(source)
    ineligible["public_weight_eligible"] = False
    with pytest.raises(S1PortableError, match="eligible source rights"):
        build_s1_rights_record(
            source=ineligible,
            runtime_code_license="Apache-2.0",
            intended_public_weight_license="CC-BY-SA-4.0",
            redistribution_blocked_pending_final_rights_review=False,
            release_rights_decision_sha256="36" * 32,
            upstream_rights={"fixture": True},
        )


def test_multi_source_rights_and_attribution_survive_export_reload(tmp_path: Path) -> None:
    def source(source_id: str, license_id: str, notice: str, digest_byte: str) -> dict[str, Any]:
        return {
            "source_id": source_id,
            "source_version": "fixture-v1",
            "source_name": f"{source_id} fixture",
            "source_entry_sha256": digest_byte * 64,
            "license_id": license_id,
            "license_sha256": "a" * 64,
            "terms_sha256": "b" * 64,
            "source_use_policy_sha256": "c" * 64,
            "attribution_notice": notice,
            "attribution_notice_sha256": _sha256(notice.encode()),
            "public_weight_eligible": True,
            "public_data_eligible": False,
            "raw_data_release": False,
            "public_data_release": False,
        }

    sources = [
        source("fleurs-asl-v1", "CC-BY-SA-4.0", "FLEURS-ASL attribution", "1"),
        source("fsboard-v3", "CC-BY-4.0", "FSboard attribution", "2"),
    ]
    rights = build_s1_multi_source_rights_record(
        sources=sources,
        rights_as_of="2026-09-03T08:00:00Z",
        runtime_code_license="Apache-2.0",
        intended_public_weight_license="CC-BY-SA-4.0",
        redistribution_blocked_pending_final_rights_review=False,
        release_rights_decision_sha256="36" * 32,
        upstream_rights={"fixture": True},
    )
    model_bytes, record_bytes = _synthetic_tokenizer()
    model_path = tmp_path / "tokenizer.model"
    record_path = tmp_path / "tokenizer.json"
    model_path.write_bytes(model_bytes)
    record_path.write_bytes(record_bytes)
    report = export_s1_portable_bundle(
        PortableS1(PortableS1Config()).eval(),
        tmp_path / "multi-source-bundle",
        tokenizer_model_path=model_path,
        tokenizer_record_path=record_path,
        preprocessing=_preprocessing(),
        rights=rights,
        upstream_release_identity_sha256="44" * 32,
    )
    runtime = load_s1_portable_bundle(
        tmp_path / "multi-source-bundle",
        expected_inference_revision=report["inference_revision"],
    )
    assert runtime.identity["rights"]["sources"] == sources

    tampered = dict(rights)
    tampered["sources"] = [dict(row) for row in sources]
    tampered["sources"][1]["attribution_notice"] = "missing required attribution"
    with pytest.raises(S1PortableError, match="attribution notice differs"):
        portable_module._validate_rights(tampered)


@pytest.fixture(scope="module")
def portable_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("portable-s1")
    model_bytes, record_bytes = _synthetic_tokenizer()
    model_path = root / "tokenizer.model"
    record_path = root / "tokenizer.json"
    model_path.write_bytes(model_bytes)
    record_path.write_bytes(record_bytes)
    torch.manual_seed(17)
    model = PortableS1(PortableS1Config()).eval()
    report = export_s1_portable_bundle(
        model,
        root / "bundle",
        tokenizer_model_path=model_path,
        tokenizer_record_path=record_path,
        preprocessing=_preprocessing(),
        rights=_rights(),
        upstream_release_identity_sha256="41" * 32,
    )
    return root / "bundle", report


def _copy_bundle(source: Path, tmp_path: Path) -> Path:
    output = tmp_path / "bundle"
    shutil.copytree(source, output)
    return output


def _reseal(root: Path, identity: dict[str, Any]) -> str:
    unsigned_identity = dict(identity)
    unsigned_identity.pop("inference_revision", None)
    revision = canonical_json_sha256(unsigned_identity, domain=portable_module._IDENTITY_DOMAIN)
    identity["inference_revision"] = revision
    (root / "inference-identity.json").write_bytes(canonical_json_bytes(identity))
    payload_names = sorted(set(portable_module._EXPECTED_FILES) - {"bundle-manifest.json"})
    manifest: dict[str, Any] = {
        "schema": portable_module.S1_PORTABLE_MANIFEST_SCHEMA,
        "inference_revision": revision,
        "files": [
            {
                "name": name,
                "sha256": _sha256((root / name).read_bytes()),
                "size_bytes": (root / name).stat().st_size,
            }
            for name in payload_names
        ],
    }
    manifest["content_sha256"] = canonical_json_sha256(
        manifest, domain=portable_module._MANIFEST_DOMAIN
    )
    (root / "bundle-manifest.json").write_bytes(canonical_json_bytes(manifest))
    return revision


def test_decode_only_tokenizer_matches_sentencepiece_piece_semantics() -> None:
    model, record = _synthetic_tokenizer()
    tokenizer = S1DecodeTokenizer.from_bytes(
        record_bytes=record,
        model_bytes=model,
        expected_record_sha256=_sha256(record),
        expected_model_sha256=_sha256(model),
    )
    assert tokenizer.decode([1, 260, 261, 2, 262]) == "piece260 piece261"
    assert tokenizer.decode([199, 173, 2]) == "é"
    assert tokenizer.decode([-1, 2]) == " ⁇ "
    with pytest.raises(S1DecodeTokenizerError, match="integer"):
        tokenizer.decode([260, True])
    with pytest.raises(S1DecodeTokenizerError, match="ceiling"):
        tokenizer.decode([260] * 129)


def test_portable_bundle_loads_without_training_or_source_data_and_runs_beam_motion(
    portable_bundle: tuple[Path, dict[str, Any]],
) -> None:
    root, report = portable_bundle
    runtime = load_s1_portable_bundle(
        root, expected_inference_revision=report["inference_revision"]
    )
    motion = np.zeros((120, MOTION_FEATURE_DIM), dtype=np.float32)
    frame_mask = np.zeros(120, dtype=np.int32)
    frame_mask[:8] = 1
    first = runtime.infer_motion(motion, frame_mask, max_new_tokens=4)
    second = runtime.infer_motion(motion, frame_mask, max_new_tokens=4)
    assert first == second
    assert len(first.token_ids) == 4
    assert isinstance(first.text, str)
    assert first.raw_text == first.text
    assert first.text_postprocess_revision == "raw-tokenizer-output/1"
    assert runtime.require_preprocessing_platform("linux/arm64").startswith("sha256:")
    assert runtime.require_preprocessing_platform("linux/amd64") == (
        "sha256:a36ae75cd3b096dc5859e6c470928c978db3263277c8dc3249a3c1b104a9cacb"
    )


def test_validation_selected_word_prefix_is_exact_bounded_and_identity_bound() -> None:
    selection = seal_s1_validation_word_prefix_selection(
        {
            "schema": "umi-s1-word-prefix-selection/1",
            "partition": "fleurs_val",
            "normalization_revision": "fixture-normalization/1",
            "source_prediction_set_sha256": "41" * 32,
            "sample_count": 2,
            "candidate_maximum_words": [4, 8],
            "candidate_scores": [
                {
                    "maximum_words": 4,
                    "mean_normalized_score": {"numerator": "3", "denominator": "4"},
                },
                {
                    "maximum_words": 8,
                    "mean_normalized_score": {"numerator": "1", "denominator": "2"},
                },
            ],
            "selected_maximum_words": 4,
            "selection_rule": "greatest-score-then-smallest-cap-v1",
            "selected_arm": "fixture-arm",
            "selected_epoch": 1,
            "test_partition_opened": False,
        }
    )
    policy = build_s1_validation_word_prefix_contract(selection)
    assert portable_module._apply_text_postprocess(" one  two three four five ", policy) == (
        "one two three four"
    )

    tampered = dict(policy)
    tampered["maximum_words"] = 8
    with pytest.raises(S1PortableError, match="differs"):
        portable_module._apply_text_postprocess("one two three four five", tampered)


def test_training_authority_word_prefix_is_fixed_before_optimization() -> None:
    policy = build_s1_authority_word_prefix_contract(
        training_authority_sha256="41" * 32,
        training_authority_content_sha256="42" * 32,
        final_report_content_sha256="43" * 32,
        maximum_words=8,
    )
    assert policy["selected_from_validation"] is False
    assert policy["basis"] == "training-target-policy-fixed-before-optimization"
    assert (
        portable_module._apply_text_postprocess(
            "one two three four five six seven eight nine", policy
        )
        == "one two three four five six seven eight"
    )

    tampered = dict(policy)
    tampered["selected_from_validation"] = True
    with pytest.raises(S1PortableError, match="differs"):
        portable_module._apply_text_postprocess("one two three", tampered)
    with pytest.raises(S1PortableError, match="deploy profile"):
        build_s1_authority_word_prefix_contract(
            training_authority_sha256="41" * 32,
            training_authority_content_sha256="42" * 32,
            final_report_content_sha256="43" * 32,
            maximum_words=7,
        )


def test_experiment_claim_profile_is_fixed_and_legacy_default_is_unchanged(
    portable_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    legacy_root, legacy_report = portable_bundle
    legacy = load_s1_portable_bundle(
        legacy_root, expected_inference_revision=legacy_report["inference_revision"]
    )
    assert "zero motion outscored real motion" in legacy.identity["claim_boundary"]

    model_bytes, record_bytes = _synthetic_tokenizer()
    model_path = tmp_path / "tokenizer.model"
    record_path = tmp_path / "tokenizer.json"
    model_path.write_bytes(model_bytes)
    record_path.write_bytes(record_bytes)
    model = PortableS1(PortableS1Config()).eval()
    experiment_postprocess = build_s1_authority_word_prefix_contract(
        training_authority_sha256=(
            "f57a189360f80f7c88cfc435df033c5ef8e0c2f7084fa6fe835f45f5f57a54be"
        ),
        training_authority_content_sha256=(
            "6e33714687a942df27d0c3729ede68eb0d68e857aea17cb27b1e9f135c70343a"
        ),
        final_report_content_sha256=(
            "fa81389b82a95edd671719157c2f06decd3f684bf9a724ef0ac6e687e9154735"
        ),
        maximum_words=8,
    )
    rights = _experiment_rights()
    portable_module._validate_rights(rights)
    rights_sha256 = canonical_json_sha256(rights)
    assert rights_sha256 == "adc1dc03f23568498759d2f14c35b55969973d2c1a7d43164ef03d2a77850634"
    claim_arguments = {
        "model_state_sha256": ("137b2733af1803d853a87396c31e5332f4170a4b5c5beffdfb246c60bb681521"),
        "upstream_release_identity_sha256": (
            "48292a46e555bf4a1cb63788d3d4b78a7e8fb4c4d99cc67d68bd37aa12ddc9b3"
        ),
        "tokenizer_model_sha256": (
            "7d8abdec60dab3a2a1969c1a7f194bdaa26f4c23884b23a204365b1535cf6783"
        ),
        "tokenizer_record_sha256": (
            "eaaa5055b60f756c4e93befbec725344bfd3395da7d84a5ae2da78d3d24f57c0"
        ),
        "text_postprocess": experiment_postprocess,
    }
    claim = _portable_claim_boundary(
        S1_EXPERIMENT_BOOTSTRAP_CLAIM_PROFILE,
        rights_sha256=rights_sha256,
        **claim_arguments,
    )
    assert "low-accuracy validation-only" in claim
    assert "No FLEURS test result is claimed" in claim
    assert "zero motion outscored real motion" not in claim

    mutated_rights = copy.deepcopy(rights)
    mutated_rights.update(
        {
            "rights_as_of": "2099-01-01T00:00:00Z",
            "runtime_code_license": "caller-chosen",
            "intended_public_weight_license": "caller-chosen",
            "redistribution_blocked_pending_final_rights_review": True,
            "release_rights_decision_sha256": None,
            "upstream_rights_sha256": "0" * 64,
        }
    )
    mutated_rights["sources"][0]["terms_sha256"] = "0" * 64
    portable_module._validate_rights(mutated_rights)
    with pytest.raises(S1PortableError, match="claim-boundary evidence differs"):
        _portable_claim_boundary(
            S1_EXPERIMENT_BOOTSTRAP_CLAIM_PROFILE,
            rights_sha256=canonical_json_sha256(mutated_rights),
            **claim_arguments,
        )

    with pytest.raises(S1PortableError, match="claim-boundary evidence differs"):
        export_s1_portable_bundle(
            model,
            tmp_path / "experiment-bundle",
            tokenizer_model_path=model_path,
            tokenizer_record_path=record_path,
            preprocessing=_preprocessing(),
            rights=_rights(),
            upstream_release_identity_sha256="44" * 32,
            text_postprocess=experiment_postprocess,
            claim_boundary_profile=S1_EXPERIMENT_BOOTSTRAP_CLAIM_PROFILE,
        )

    with pytest.raises(S1PortableError, match="claim-boundary profile"):
        export_s1_portable_bundle(
            model,
            tmp_path / "invalid-claim-bundle",
            tokenizer_model_path=model_path,
            tokenizer_record_path=record_path,
            preprocessing=_preprocessing(),
            rights=_rights(),
            upstream_release_identity_sha256="44" * 32,
            claim_boundary_profile="caller-controlled-text/1",
        )


def _public_finetune_claim_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    state_sha256 = "51" * 32
    tokenizer_model_sha256 = "52" * 32
    tokenizer_record_sha256 = "53" * 32
    rights_sha256 = "54" * 32
    training_authority_sha256 = "55" * 32
    training_authority_content_sha256 = "56" * 32
    final_report_content_sha256 = "57" * 32
    postprocess = build_s1_authority_word_prefix_contract(
        training_authority_sha256=training_authority_sha256,
        training_authority_content_sha256=training_authority_content_sha256,
        final_report_content_sha256=final_report_content_sha256,
        maximum_words=8,
    )
    candidate = {
        "training_authority_sha256": training_authority_sha256,
        "training_authority_file_sha256": "58" * 32,
        "training_authority_content_sha256": training_authority_content_sha256,
        "final_report_sha256": "59" * 32,
        "final_report_content_sha256": final_report_content_sha256,
        "selected_epoch": 6,
        "selected_epoch_report_content_sha256": "5a" * 32,
        "selected_validation_sha256": "5b" * 32,
        "selected_validation_content_sha256": "5c" * 32,
        "selected_checkpoint_manifest_sha256": "5d" * 32,
        "selected_checkpoint_metadata_sha256": "5e" * 32,
        "selected_model_state_sha256": state_sha256,
    }
    gate = {
        "passed": True,
        "real_motion_score": _fraction(1, 10),
        "zero_motion_score": _fraction(9, 100),
        "deranged_motion_score": _fraction(2, 25),
        "real_minus_zero": _fraction(1, 100),
        "real_minus_deranged": _fraction(1, 50),
        "prediction_count": 285,
        "unique_real_hypotheses": 250,
        "maximum_real_hypothesis_multiplicity": 4,
        "gate": {
            "minimum_real_score": _fraction(9, 100),
            "minimum_control_advantage": _fraction(1, 100),
            "minimum_unique_real_hypotheses": 240,
            "maximum_real_hypothesis_multiplicity": 6,
            "reject_empty_hypotheses": True,
            "reject_unk_token_id": 3,
        },
    }
    text_pretraining = {
        "training_authority_sha256": "61" * 32,
        "training_authority_content_sha256": "62" * 32,
        "final_report_sha256": "63" * 32,
        "final_report_content_sha256": "64" * 32,
        "selected_checkpoint_manifest_sha256": "65" * 32,
        "selected_checkpoint_metadata_sha256": "66" * 32,
        "selected_model_state_sha256": "67" * 32,
        "selected_epoch": 4,
        "release_lineage_allowed": True,
        "motion_encoder_invocation_count": 0,
        "cross_attention_invocation_count": 0,
        "claim": "fixture value is bound but never copied into portable claim prose",
        "training_authority_file_sha256": "68" * 32,
        "corpus_manifest_sha256": "69" * 32,
        "corpus_manifest_content_sha256": "6a" * 32,
        "source_entry_sha256": "6b" * 32,
    }
    evidence: dict[str, Any] = {
        "schema": "umi-public-s1-finetune-release-identity/1",
        "candidate": candidate,
        "gate": gate,
        "tokenizer": {
            "binding_sha256": "71" * 32,
            "model_sha256": tokenizer_model_sha256,
            "record_sha256": tokenizer_record_sha256,
            "training_report_sha256": "72" * 32,
        },
        "text_pretraining": text_pretraining,
        "motion_sources": {
            "fleurs": {
                "manifest_sha256": "73" * 32,
                "manifest_content_sha256": "74" * 32,
                "source_entry_sha256": "75" * 32,
                "source_ledger_sha256": "76" * 32,
                "training_rows_loaded": 492,
                "validation_rows_loaded": 285,
                "test_inference_rows": 0,
            },
            "two_m_flores": {
                "manifest_sha256": "77" * 32,
                "manifest_content_sha256": "78" * 32,
                "binding_content_sha256": "79" * 32,
                "source_entry_sha256": "7a" * 32,
                "source_ledger_sha256": "7b" * 32,
                "sample_count": 1234,
                "devtest_rows_loaded": 0,
                "evaluation_rows_loaded": 0,
            },
        },
        "rights_review_sha256": "7c" * 32,
        "rights_record_sha256": rights_sha256,
        "base_preprocessing": {
            "inference_revision": "7d" * 32,
            "identity_sha256": "7e" * 32,
            "preprocessing_sha256": "7f" * 32,
        },
        "release_contents": {
            "safe_tensor_weights": True,
            "tokenizer_artifacts": True,
            "raw_training_or_validation_data": False,
            "source_video_or_annotations": False,
            "prediction_plaintext": False,
        },
        "claim_boundary": portable_module._PUBLIC_FINETUNE_RELEASE_CLAIM_BOUNDARY,
    }
    evidence["content_sha256"] = canonical_json_sha256(
        evidence, domain=portable_module._PUBLIC_FINETUNE_RELEASE_IDENTITY_DOMAIN
    )
    arguments = {
        "model_state_sha256": state_sha256,
        "upstream_release_identity_sha256": canonical_json_sha256(evidence),
        "tokenizer_model_sha256": tokenizer_model_sha256,
        "tokenizer_record_sha256": tokenizer_record_sha256,
        "text_postprocess": postprocess,
        "rights_sha256": rights_sha256,
        "claim_boundary_evidence": evidence,
    }
    return evidence, arguments


def test_public_finetune_claim_is_generated_only_from_complete_bound_evidence() -> None:
    _evidence, arguments = _public_finetune_claim_fixture()

    claim = _portable_claim_boundary(S1_PUBLIC_FINETUNE_CLAIM_PROFILE, **arguments)

    assert "285-sample FLEURS validation" in claim
    assert "real motion 1/10, zero motion 9/100, and deranged motion 2/25" in claim
    assert "1/100 over zero" in claim
    assert "1/50 over deranged" in claim
    assert "250 unique" in claim
    assert "maximum multiplicity 4" in claim
    assert "No FLEURS test inference or 2M-Flores devtest/evaluation inference" in claim
    assert "UMI activation" in claim
    assert "accessibility certification" in claim
    assert "fixture value" not in claim


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("state", "model state differs"),
        ("tokenizer", "tokenizer differs"),
        ("rights", "rights record differs"),
        ("raw", "non-portable data"),
        ("fleurs_test", "test, devtest, or evaluation"),
        ("two_m_devtest", "test, devtest, or evaluation"),
        ("score_arithmetic", "metric or diversity"),
        ("diversity", "metric or diversity"),
        ("content", "release identity differs"),
    ],
)
def test_public_finetune_claim_rejects_incomplete_or_unbound_evidence(
    mutation: str, match: str
) -> None:
    evidence, arguments = _public_finetune_claim_fixture()
    if mutation == "state":
        arguments["model_state_sha256"] = "00" * 32
    elif mutation == "tokenizer":
        arguments["tokenizer_record_sha256"] = "00" * 32
    elif mutation == "rights":
        arguments["rights_sha256"] = "00" * 32
    elif mutation == "raw":
        evidence["release_contents"]["raw_training_or_validation_data"] = True
    elif mutation == "fleurs_test":
        evidence["motion_sources"]["fleurs"]["test_inference_rows"] = 1
    elif mutation == "two_m_devtest":
        evidence["motion_sources"]["two_m_flores"]["devtest_rows_loaded"] = 1
    elif mutation == "score_arithmetic":
        evidence["gate"]["real_minus_zero"] = _fraction(1, 50)
    elif mutation == "diversity":
        evidence["gate"]["unique_real_hypotheses"] = 239
    elif mutation == "content":
        evidence["claim_boundary"] = "caller prose"
    if mutation not in {"state", "tokenizer", "rights", "content"}:
        evidence["content_sha256"] = canonical_json_sha256(
            {key: value for key, value in evidence.items() if key != "content_sha256"},
            domain=portable_module._PUBLIC_FINETUNE_RELEASE_IDENTITY_DOMAIN,
        )
        arguments["upstream_release_identity_sha256"] = canonical_json_sha256(evidence)

    with pytest.raises(S1PortableError, match=match):
        _portable_claim_boundary(S1_PUBLIC_FINETUNE_CLAIM_PROFILE, **arguments)


def test_public_finetune_profile_requires_evidence_and_fixed_profile_rejects_it() -> None:
    evidence, arguments = _public_finetune_claim_fixture()
    no_evidence = dict(arguments)
    del no_evidence["claim_boundary_evidence"]
    with pytest.raises(S1PortableError, match="requires release evidence"):
        _portable_claim_boundary(S1_PUBLIC_FINETUNE_CLAIM_PROFILE, **no_evidence)

    fixed_arguments = dict(arguments)
    fixed_arguments.update(
        {
            "model_state_sha256": portable_module._EXPERIMENT_BOOTSTRAP_MODEL_STATE_SHA256,
            "upstream_release_identity_sha256": (
                portable_module._EXPERIMENT_BOOTSTRAP_RELEASE_IDENTITY_SHA256
            ),
            "tokenizer_model_sha256": (
                portable_module._EXPERIMENT_BOOTSTRAP_TOKENIZER_MODEL_SHA256
            ),
            "tokenizer_record_sha256": (
                portable_module._EXPERIMENT_BOOTSTRAP_TOKENIZER_RECORD_SHA256
            ),
            "rights_sha256": portable_module._EXPERIMENT_BOOTSTRAP_RIGHTS_SHA256,
        }
    )
    del fixed_arguments["claim_boundary_evidence"]
    fixed_arguments["text_postprocess"] = build_s1_authority_word_prefix_contract(
        training_authority_sha256=(
            "f57a189360f80f7c88cfc435df033c5ef8e0c2f7084fa6fe835f45f5f57a54be"
        ),
        training_authority_content_sha256=(
            "6e33714687a942df27d0c3729ede68eb0d68e857aea17cb27b1e9f135c70343a"
        ),
        final_report_content_sha256=(
            "fa81389b82a95edd671719157c2f06decd3f684bf9a724ef0ac6e687e9154735"
        ),
        maximum_words=8,
    )
    with pytest.raises(S1PortableError, match="does not accept external evidence"):
        _portable_claim_boundary(
            S1_EXPERIMENT_BOOTSTRAP_CLAIM_PROFILE,
            claim_boundary_evidence=evidence,
            **fixed_arguments,
        )


def test_portable_runtime_identity_binds_every_local_inference_dependency(
    portable_bundle: tuple[Path, dict[str, Any]],
) -> None:
    root, report = portable_bundle
    runtime = load_s1_portable_bundle(
        root, expected_inference_revision=report["inference_revision"]
    )
    modules = {row["module"] for row in runtime.identity["runtime"]["modules"]}
    assert modules == {
        "bitsign_motion",
        "bitsign_motion.amd64_holistic_container",
        "bitsign_motion.canonical",
        "bitsign_motion.constants",
        "bitsign_motion.holistic_container",
        "bitsign_motion.holistic_motion",
        "bitsign_motion.mediapipe_mapping",
        "bitsign_motion.motion_artifact",
        "bitsign_motion.portable_model",
        "bitsign_motion.s1_decode_tokenizer",
        "bitsign_motion.s1_portable_runtime",
        "bitsign_motion.s1_state_digest",
        "bitsign_motion.umi_reference_backend",
    }
    assert runtime.identity["runtime"]["rfc8785_version"] == "0.1.4"
    assert runtime.identity["runtime"]["decoding"] == {
        "algorithm": "beam-search",
        "beam_width": 2,
        "default_maximum_decode_tokens": 24,
        "eos_token_id": 2,
        "length_normalization": False,
        "no_repeat_ngram_size": 3,
        "score": "cumulative-log-probability",
        "suppressed_token_ids": [0, 1],
        "tie_break": "lexicographically-smallest-token-sequence",
    }


def test_portable_runtime_uses_bound_beam_controls_and_configurable_token_budget(
    portable_bundle: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, report = portable_bundle
    runtime = load_s1_portable_bundle(
        root, expected_inference_revision=report["inference_revision"]
    )
    calls: list[tuple[int | None, int, int]] = []

    def beam_decode(
        model: PortableS1,
        motion: torch.Tensor,
        frame_mask: torch.Tensor,
        *,
        max_new_tokens: int | None,
        beam_width: int,
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        del model, frame_mask
        calls.append((max_new_tokens, beam_width, no_repeat_ngram_size))
        assert max_new_tokens is not None
        result = torch.zeros(
            (motion.shape[0], max_new_tokens), dtype=torch.int64, device=motion.device
        )
        result[:, 0] = 2
        return result

    monkeypatch.setattr(PortableS1, "beam_decode", beam_decode)
    motion = np.zeros((120, MOTION_FEATURE_DIM), dtype=np.float32)
    frame_mask = np.zeros(120, dtype=np.int32)
    frame_mask[:8] = 1

    default = runtime.infer_motion(motion, frame_mask)
    overridden = runtime.infer_motion(motion, frame_mask, max_new_tokens=7)

    assert default.token_ids == (2,)
    assert overridden.token_ids == (2,)
    assert calls == [(24, 2, 3), (7, 2, 3)]


@pytest.mark.parametrize("extra_name", ["extra", "nested"])
def test_portable_loader_rejects_extra_files_and_directories(
    portable_bundle: tuple[Path, dict[str, Any]], tmp_path: Path, extra_name: str
) -> None:
    source, report = portable_bundle
    root = _copy_bundle(source, tmp_path)
    extra = root / extra_name
    if extra_name == "nested":
        extra.mkdir()
    else:
        extra.write_bytes(b"extra")
    with pytest.raises(S1PortableError, match="field set"):
        load_s1_portable_bundle(root, expected_inference_revision=report["inference_revision"])


def test_portable_loader_rejects_symlinks_and_special_files(
    portable_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, report = portable_bundle
    symlink_root = _copy_bundle(source, tmp_path / "symlink-case")
    model = symlink_root / "tokenizer.model"
    model.unlink()
    model.symlink_to(source / "tokenizer.model")
    with pytest.raises(S1PortableError, match="cannot be opened"):
        load_s1_portable_bundle(
            symlink_root, expected_inference_revision=report["inference_revision"]
        )

    fifo_root = _copy_bundle(source, tmp_path / "fifo-case")
    config = fifo_root / "model-config.json"
    config.unlink()
    os.mkfifo(config)
    with pytest.raises(S1PortableError, match="violates its contract"):
        load_s1_portable_bundle(fifo_root, expected_inference_revision=report["inference_revision"])


def test_portable_loader_rejects_corruption_oversize_and_revision_mismatch(
    portable_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, report = portable_bundle
    corrupt = _copy_bundle(source, tmp_path / "corrupt-case")
    checkpoint = corrupt / "model.safetensors"
    payload = bytearray(checkpoint.read_bytes())
    payload[-1] ^= 1
    checkpoint.write_bytes(payload)
    with pytest.raises(S1PortableError, match="inventory"):
        load_s1_portable_bundle(corrupt, expected_inference_revision=report["inference_revision"])

    oversized = _copy_bundle(source, tmp_path / "oversize-case")
    (oversized / "bundle-manifest.json").write_bytes(b"{" + b" " * (1024 * 1024))
    with pytest.raises(S1PortableError, match="violates its contract"):
        load_s1_portable_bundle(oversized, expected_inference_revision=report["inference_revision"])
    with pytest.raises(S1PortableError, match="identity"):
        load_s1_portable_bundle(source, expected_inference_revision="00" * 32)


def test_portable_loader_rejects_resealed_config_and_preprocessing_revision_mismatches(
    portable_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _report = portable_bundle
    config_root = _copy_bundle(source, tmp_path / "config-case")
    config = portable_module._strict_json(
        (config_root / "model-config.json").read_bytes(),
        maximum_bytes=256 * 1024,
        label="fixture config",
    )
    config["model_config"]["hidden_size"] = 128
    config_bytes = canonical_json_bytes(config)
    (config_root / "model-config.json").write_bytes(config_bytes)
    identity = portable_module._strict_json(
        (config_root / "inference-identity.json").read_bytes(),
        maximum_bytes=2 * 1024 * 1024,
        label="fixture identity",
    )
    identity["model"]["config_sha256"] = _sha256(config_bytes)
    revision = _reseal(config_root, identity)
    with pytest.raises(S1PortableError, match="frozen public S1"):
        load_s1_portable_bundle(config_root, expected_inference_revision=revision)

    preprocessing_root = _copy_bundle(source, tmp_path / "preprocessing-case")
    identity = portable_module._strict_json(
        (preprocessing_root / "inference-identity.json").read_bytes(),
        maximum_bytes=2 * 1024 * 1024,
        label="fixture identity",
    )
    identity["preprocessing"]["mapping"]["profile"] = "unapproved-mapping/1"
    revision = _reseal(preprocessing_root, identity)
    with pytest.raises(S1PortableError, match="fixed S1 candidate"):
        load_s1_portable_bundle(preprocessing_root, expected_inference_revision=revision)

    postprocess_root = _copy_bundle(source, tmp_path / "postprocess-case")
    identity = portable_module._strict_json(
        (postprocess_root / "inference-identity.json").read_bytes(),
        maximum_bytes=2 * 1024 * 1024,
        label="fixture identity",
    )
    identity["text_postprocess"]["revision"] = "collapse-repetitions/1"
    revision = _reseal(postprocess_root, identity)
    with pytest.raises(S1PortableError, match="text postprocess"):
        load_s1_portable_bundle(postprocess_root, expected_inference_revision=revision)


def test_portable_loader_rejects_resealed_tensor_and_tokenizer_mismatches(
    portable_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _report = portable_bundle
    tensor_root = _copy_bundle(source, tmp_path / "tensor-case")
    model = PortableS1(PortableS1Config())
    tensors = {
        name: value.detach().contiguous().clone() for name, value in model.state_dict().items()
    }
    tensors.pop("encoder_norm.bias")
    identity = portable_module._strict_json(
        (tensor_root / "inference-identity.json").read_bytes(),
        maximum_bytes=2 * 1024 * 1024,
        label="fixture identity",
    )
    state_sha256 = identity["model"]["state_sha256"]
    tensor_bytes = save_safetensors(
        tensors,
        metadata={
            "schema": portable_module.S1_PORTABLE_TENSOR_SCHEMA,
            "model": "S1",
            "model_state_sha256": state_sha256,
        },
    )
    (tensor_root / "model.safetensors").write_bytes(tensor_bytes)
    identity["model"]["checkpoint_sha256"] = _sha256(tensor_bytes)
    identity["model"]["checkpoint_size_bytes"] = len(tensor_bytes)
    revision = _reseal(tensor_root, identity)
    with pytest.raises(S1PortableError, match="tensor set"):
        load_s1_portable_bundle(tensor_root, expected_inference_revision=revision)

    tokenizer_root = _copy_bundle(source, tmp_path / "tokenizer-case")
    tokenizer = portable_module._strict_json(
        (tokenizer_root / "tokenizer.json").read_bytes(),
        maximum_bytes=2 * 1024 * 1024,
        label="fixture tokenizer",
    )
    tokenizer["pieces"].pop()
    tokenizer_bytes = canonical_json_bytes(tokenizer)
    (tokenizer_root / "tokenizer.json").write_bytes(tokenizer_bytes)
    identity = portable_module._strict_json(
        (tokenizer_root / "inference-identity.json").read_bytes(),
        maximum_bytes=2 * 1024 * 1024,
        label="fixture identity",
    )
    identity["tokenizer"]["record_sha256"] = _sha256(tokenizer_bytes)
    identity["tokenizer"]["record_size_bytes"] = len(tokenizer_bytes)
    revision = _reseal(tokenizer_root, identity)
    with pytest.raises(S1PortableError, match="tokenizer is invalid"):
        load_s1_portable_bundle(tokenizer_root, expected_inference_revision=revision)


def test_portable_inference_rejects_invalid_motion_contract(
    portable_bundle: tuple[Path, dict[str, Any]],
) -> None:
    source, report = portable_bundle
    runtime = load_s1_portable_bundle(
        source, expected_inference_revision=report["inference_revision"]
    )
    motion = np.zeros((120, MOTION_FEATURE_DIM), dtype=np.float32)
    frame_mask = np.zeros(120, dtype=np.int32)
    frame_mask[[0, 2]] = 1
    with pytest.raises(S1PortableError, match="contiguous prefix"):
        runtime.infer_motion(motion, frame_mask, max_new_tokens=1)
    frame_mask[:] = 0
    frame_mask[:2] = 1
    motion[2, 0] = np.float32(-0.0)
    with pytest.raises(S1PortableError, match="positive zero"):
        runtime.infer_motion(motion, frame_mask, max_new_tokens=1)
