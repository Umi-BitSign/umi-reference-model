from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_word_training as word
import probe_roi_semantic_learning as original_proof
import terra_mixed_semantic_cases as cases_protocol
import terra_mixed_semantic_protocol as protocol
import terra_visual_learning
import test_c5_word_training as training_fixtures
import test_terra_mixed_semantic_cases as case_fixtures
from c5_small_input_semantic import Native112Clip
from terra_mixed_native_input import expected_header, training_row


@pytest.fixture
def cases():
    selected, binding = case_fixtures.complete_binding.__wrapped__()
    return cases_protocol.bind_cases(selected, binding)


@pytest.fixture
def tokenizer():
    return training_fixtures.Tokenizer()


def pools(cases, tokenizer):
    generated = []
    for index, case in enumerate(cases):
        text = training_row(case["case"])["reference"]
        generated.append(
            [
                {
                    "tokens": tokenizer(text, truncation=False).input_ids,
                    "text": text,
                    "kind": "human_anchor",
                    "termination": "eos",
                    "truncated": False,
                },
                protocol.training_negative(case, cases[(index + 1) % 3], tokenizer),
            ]
        )
    return generated


def test_actual_pairings_include_exact_mixed_input_and_cyclic_donor_receipts(cases, tokenizer):
    paired = protocol.paired_inputs(cases, pools(cases, tokenizer), tokenizer)
    assert len(paired["cases"]) == 3 and len(paired["items"]) <= 16
    for index, (case, mapping) in enumerate(zip(cases, paired["cases"], strict=True)):
        assert mapping["mixed_input_receipt"] == case["receipt"]
        assert mapping["donor_input_receipt"] == cases[(index + 1) % 3]["receipt"]
        assert (
            paired["items"][mapping["indices"][0]]["reference"]
            == training_row(case["case"])["reference"]
        )
    assert protocol.qualities is original_proof.qualities
    assert protocol.candidate_pool is terra_visual_learning.candidate_pool
    assert protocol.verified_rewards is word.verified_rewards
    assert protocol.shuffled_indices is word.shuffled_indices


def test_full_four_candidate_pools_fit_the_original_single_sixteen_pair_request(cases, tokenizer):
    generated = pools(cases, tokenizer)
    for pool in generated:
        for kind, text in (("greedy", "Greedy output"), ("sampled", "Sampled output")):
            pool.insert(
                -1,
                {
                    "tokens": tokenizer(text, truncation=False).input_ids,
                    "text": text,
                    "kind": kind,
                    "termination": "eos",
                    "truncated": False,
                },
            )
    paired = protocol.paired_inputs(cases, generated, tokenizer)
    assert len(paired["items"]) == 16
    assert [len(mapping["indices"]) for mapping in paired["cases"]] == [4, 4, 4]
    assert len(paired["controls"]) == 4


def test_donor_caption_must_roundtrip_without_alteration(cases):
    class ChangedTokenizer(training_fixtures.Tokenizer):
        def decode(self, tokens, **kwargs):
            return "Altered training caption"

    with pytest.raises(ValueError, match="cannot alter or truncate"):
        protocol.training_negative(cases[0], cases[1], ChangedTokenizer())


@pytest.mark.parametrize("mutation", ["anchor", "tokens", "donor", "kind", "duplicate"])
def test_drifted_anchor_tokens_donor_or_candidate_scope_is_rejected(cases, tokenizer, mutation):
    generated = pools(cases, tokenizer)
    if mutation == "anchor":
        generated[0][0]["text"] = "Changed anchor"
    elif mutation == "tokens":
        generated[0][-1]["tokens"][0] += 1
    elif mutation == "donor":
        generated[0][-1]["donor_input_receipt"] = cases[0]["receipt"]
    elif mutation == "kind":
        generated[0][0]["kind"] = "greedy"
    else:
        generated[0].append(copy.deepcopy(generated[0][0]))
    with pytest.raises(ValueError):
        protocol.validate_pools(cases, generated, tokenizer)


def test_forbidden_donor_role_fails_before_caption_tokenization(cases):
    donor = copy.deepcopy(cases[1])
    donor["case"]["phrase"]["roi_training_allowed"] = False

    def forbidden(*args, **kwargs):
        pytest.fail("forbidden donor reached tokenization")

    with pytest.raises(ValueError):
        protocol.training_negative(cases[0], donor, forbidden)


@pytest.mark.parametrize("duplicate_rollout", [False, True])
def test_pool_generation_preserves_complete_inputs_and_observed_duplicate_donors(
    cases, tokenizer, monkeypatch, duplicate_rollout
):
    model = nn.Linear(1, 1).eval()
    calls = []

    def clip(case):
        frames = expected_header(case["case"])["sampled_frames"]
        return Native112Clip(
            torch.zeros(frames, 3, 3, 112, 112), torch.zeros(frames, 14)
        ), torch.ones(1, frames, dtype=torch.bool)

    def generate(student, actual_tokenizer, video, mask, reference, **kwargs):
        index = len(calls)
        assert student is model and actual_tokenizer is tokenizer
        assert len(video.pixels) == expected_header(cases[index]["case"])["sampled_frames"]
        assert mask.shape == (1, len(video.pixels)) and mask.all()
        assert reference == training_row(cases[index]["case"])["reference"]
        assert kwargs["seed"] == protocol.POLICY["training_rollout_seed"] + index
        assert kwargs["maximum_tokens"] == 512
        observed = pools(cases, tokenizer)[index]
        if duplicate_rollout:
            observed[-1]["kind"] = "greedy"
            observed[-1].pop("donor_input_receipt")
            result = observed
        else:
            result = observed[:1]
        calls.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(protocol, "load_clip", clip)
    monkeypatch.setattr(protocol, "candidate_pool", generate)
    generated = protocol.prepare_pools(model, tokenizer, cases)
    assert len(calls) == 3
    protocol.validate_pools(cases, generated, tokenizer)
    for index, pool in enumerate(generated):
        assert len(pool) == 2
        if duplicate_rollout:
            assert pool == calls[index] and pool[-1]["kind"] == "greedy"
        else:
            assert pool[-1] == protocol.training_negative(
                cases[index], cases[(index + 1) % 3], tokenizer
            )


def test_nonCPU_rollout_model_fails_before_input_access(cases, tokenizer, monkeypatch):
    model = nn.Linear(1, 1, device="meta")
    monkeypatch.setattr(
        protocol, "load_clip", lambda *a: pytest.fail("read inputs for nonCPU model")
    )
    with pytest.raises(ValueError, match="CPU-only"):
        protocol.prepare_pools(model, tokenizer, cases)


@pytest.mark.parametrize("tied_rewards", [False, True])
def test_shared_raw_response_authentication_rejoins_shuffled_mixed_training_rewards(
    tmp_path, monkeypatch, cases, tokenizer, tied_rewards
):
    monkeypatch.setattr(word, "ROOT", tmp_path)
    paired = protocol.paired_inputs(cases, pools(cases, tokenizer), tokenizer)
    directory = tmp_path / "rewards"
    directory.mkdir()
    (directory / "blinded-inputs.json").write_text(json.dumps(paired))
    indices = protocol.shuffled_indices(paired, protocol.POLICY["training_rollout_seed"])
    pacer = training_fixtures.FakePacer(tie_calls=(1,) if tied_rewards else ())
    for offset in range(0, len(indices), 16):
        chosen = indices[offset : offset + 16]
        pacer.score(
            [paired["items"][index] for index in chosen], directory / f"request-{offset // 16:03d}"
        )
    result = protocol.verified_rewards(paired, directory, protocol.POLICY["training_rollout_seed"])
    if tied_rewards:
        assert result["qualities"] == [[10, 10]] * 3
        with pytest.raises(ValueError, match="contrasting"):
            protocol.verified_proof_rewards(
                paired, directory, protocol.POLICY["training_rollout_seed"]
            )
        return
    assert result["qualities"] == [[10, 3]] * 3
    assert (
        protocol.verified_proof_rewards(paired, directory, protocol.POLICY["training_rollout_seed"])
        == result
    )
    assert protocol.qualities(paired, result["scores"]) == result["qualities"]
    saved = json.loads((directory / "blinded-inputs.json").read_text())
    saved["cases"][0]["donor_input_receipt"] = cases[0]["receipt"]
    (directory / "blinded-inputs.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="pairings"):
        protocol.verified_rewards(paired, directory, protocol.POLICY["training_rollout_seed"])


def test_unchanged_proof_checker_rejects_failed_calibration_and_tied_case_rewards(cases, tokenizer):
    paired = protocol.paired_inputs(cases, pools(cases, tokenizer), tokenizer)
    scores = [0] * len(paired["items"])
    for mapping in paired["cases"]:
        scores[mapping["indices"][0]] = 10
    for control in paired["controls"]:
        scores[control["index"]] = control["range"][0]
    assert protocol.qualities(paired, scores) == [[10, 0]] * 3
    tied = list(scores)
    for mapping in paired["cases"]:
        for index in mapping["indices"]:
            tied[index] = 10
    with pytest.raises(ValueError, match="contrasting"):
        protocol.qualities(paired, tied)
    generic = next(control for control in paired["controls"] if control["range"] == [0, 0])
    scores[generic["index"]] = 1
    with pytest.raises(ValueError, match="rubric controls"):
        protocol.qualities(paired, scores)
