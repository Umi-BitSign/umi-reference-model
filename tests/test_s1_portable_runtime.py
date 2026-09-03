from __future__ import annotations

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
    S1PortableError,
    build_s1_preprocessing_contract,
    build_s1_rights_record,
    build_s1_validation_word_prefix_contract,
    export_s1_portable_bundle,
    load_s1_portable_bundle,
    seal_s1_validation_word_prefix_selection,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
