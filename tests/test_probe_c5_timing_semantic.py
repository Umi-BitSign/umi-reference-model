"""Synthetic qualification boundaries and real raw-score parsing; no real C5 run."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_timing_feature_data as data
import probe_c5_timing_semantic as proof
from c5_monotone_feature_timing import WIDTHS
from test_c5_timing_semantic_model import student


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, *, truncation):
        assert truncation is False
        return SimpleNamespace(input_ids=[ord(c) - ord("a") + 2 for c in text] + [1])

    def decode(self, tokens, **kwargs):
        return "".join(chr(token - 2 + ord("a")) for token in tokens if token > 1)


@pytest.fixture
def cases():
    result = []
    generator = np.random.default_rng(73)
    for index in range(4):
        result.append(
            {
                "row": {
                    "source": "synthetic",
                    "example_id": str(index),
                    "reference": chr(ord("a") + index),
                    "role": "training",
                    "training_allowed": True,
                },
                "streams": {
                    k: (
                        generator.normal(size=(13, width)) * np.linspace(0.1, 2, 13)[:, None]
                    ).astype(np.float32)
                    for k, width in WIDTHS.items()
                },
                "valid_intervals": np.ones(12, dtype=np.bool_),
            }
        )
    return result


def pools_for(values):
    tokenizer = Tokenizer()
    pools = []
    for index, value in enumerate(values):
        anchor = value["row"]["reference"]
        pools.append(
            [
                {
                    "tokens": tokenizer(anchor, truncation=False).input_ids,
                    "text": anchor,
                    "kind": "human_anchor",
                    "truncated": False,
                    "termination": "eos",
                },
                proof.donor_candidate(
                    value, values[(index + proof.POLICY["donor_offset"]) % 4], tokenizer
                ),
            ]
        )
    return pools


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def grades(cases, tmp_path, monkeypatch):
    import terra_polarity_judge as judge

    monkeypatch.setattr(proof, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    directory = tmp_path / "semantic"
    directory.mkdir()
    pools = pools_for(cases)
    paired = proof.paired_inputs(cases, pools)
    scores = [10 if item["reference"] == item["candidate"] else 0 for item in paired["items"]]
    for control in paired["controls"]:
        scores[control["index"]] = control["range"][1]
    requests = proof.grading_requests(paired)
    save(directory / "requests.json", requests)
    for index, request in enumerate(requests):
        sub = directory / f"request-{index:03d}"
        sub.mkdir()
        rows = [paired["items"][i] for i in request["indices"]]
        actual = [scores[i] for i in request["indices"]]
        save(sub / "request.json", judge.payload(rows))
        save(
            sub / "started.json",
            {
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "pairs": len(rows),
                "automatic_retries": 0,
                "request_sha256": data.sha(sub / "request.json"),
            },
        )
        save(
            sub / "response.json",
            {
                "model": "gpt-5.6-terra",
                "reasoning": {"effort": "medium"},
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "scores": [
                                            {"id": f"pair_{i:02d}", "score": score}
                                            for i, score in enumerate(actual)
                                        ]
                                    }
                                ),
                            }
                        ],
                    }
                ],
            },
        )
        save(
            sub / "scores.json",
            {
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "scores": actual,
                "response_sha256": data.sha(sub / "response.json"),
            },
        )
    return paired, directory, cases, pools


def test_freeze_refuses_missing_complete_data_before_creating_plan_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)

    def refuse(path):
        raise ValueError("no completed actual data audit")

    monkeypatch.setattr(proof, "checked_data", refuse)
    plan = tmp_path / "runs/new/plan.json"
    with pytest.raises(ValueError, match="completed actual data"):
        proof.freeze(plan, tmp_path / "runs/audit.json")
    assert not plan.parent.exists()


def test_fixed_cases_cannot_change_order_or_roles(cases):
    plan = {"records": [{"row": value["row"]} for value in cases]}

    class Dataset:
        def load(self, index):
            return copy.deepcopy(cases[proof.POLICY["case_indices"].index(index)])

    dataset = Dataset()
    assert len(proof.selected_values(dataset, plan)) == 4
    plan["records"].reverse()
    with pytest.raises(ValueError, match="four complete training"):
        proof.selected_values(dataset, plan)


def test_all_feature_identity_and_real_interface_generation_on_synthetic_backbones(cases):
    class Dataset:
        def __len__(self):
            return len(cases)

        def load(self, index):
            return cases[index]

    model = student().eval()
    fake_generate = model.language.generate

    def native_generate(*, encoder_outputs, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones(
                encoder_outputs.last_hidden_state.shape[:2], dtype=torch.long
            )
        return fake_generate(
            encoder_outputs=encoder_outputs, attention_mask=attention_mask, **kwargs
        )

    model.language.generate = native_generate
    assert proof.all_feature_identities(model, Dataset())["clips"] == 4
    for value in cases:
        report = proof.identity_check(
            model,
            Tokenizer(),
            value,
            {
                "reference": {
                    "generation": {
                        "length_penalty": 0.6,
                        "max_length": 2048,
                        "num_beams": 5,
                        "no_repeat_ngram_size": 16,
                    }
                }
            },
        )
        assert report["memory_bitwise_equal"] and report["native_generation_tokens_equal"]
        assert report["teacher_forcing_logits_and_loss_bitwise_equal"]
    with torch.no_grad():
        model.timing.rate_weights[0] = 0.01
    with pytest.raises(ValueError, match="identity initializer"):
        proof.all_feature_identities(model, Dataset())


@pytest.mark.parametrize("change", ["anchor_text", "anchor_tokens", "donor", "duplicate", "omit"])
def test_pool_validation_binds_exact_anchor_donor_and_observed_tokens(cases, change):
    pools = pools_for(cases)
    proof.validate_pools(cases, pools, Tokenizer())
    if change == "anchor_text":
        pools[0][0]["text"] = "c"
    elif change == "anchor_tokens":
        pools[0][0]["tokens"] = [3, 1]
    elif change == "donor":
        pools[0][-1] = proof.donor_candidate(cases[0], cases[1], Tokenizer())
    elif change == "duplicate":
        pools[0].append(copy.deepcopy(pools[0][0]))
    else:
        pools.pop()
    with pytest.raises(ValueError):
        proof.validate_pools(cases, pools, Tokenizer())


def test_two_complete_raw_authenticated_passes_and_actual_risk_backward(grades):
    paired, directory, cases, pools = grades
    passes, raw = proof.authenticated_passes(paired, directory)
    assert [p["pass"] for p in passes] == [1, 2]
    assert all(p["qualities"] == [[10, 0]] * 4 for p in passes)
    assert len(raw) == 1 + 4 * len(proof.grading_requests(paired))
    model = student()
    gradients = proof.actual_gradients(model, cases, pools, passes)
    assert len(gradients) == 8
    assert all(g["gradients"]["gradient_norm"] > 0 for g in gradients)
    assert torch.count_nonzero(model.timing.rate_weights) == 0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize(
    "change", ["wrong_model", "prompt", "saved_score", "order", "missing", "failure", "extra"]
)
def test_raw_grading_rejects_changed_or_incomplete_evidence(grades, change):
    paired, directory, _, _ = grades
    sub = directory / "request-000"
    if change == "wrong_model":
        path = sub / "response.json"
        value = data.read(path)
        value["model"] = "different"
        save(path, value)
    elif change == "prompt":
        path = sub / "request.json"
        value = data.read(path)
        value["instructions"] += " changed"
        save(path, value)
    elif change == "saved_score":
        path = sub / "scores.json"
        value = data.read(path)
        value["scores"][0] = 5
        save(path, value)
    elif change == "order":
        path = directory / "requests.json"
        value = data.read(path)
        value.reverse()
        save(path, value)
    elif change == "missing":
        (sub / "scores.json").unlink()
    elif change == "failure":
        save(sub / "failure.json", {"http_status": 502})
    else:
        save(directory / "unbound.json", {})
    with pytest.raises((ValueError, FileNotFoundError)):
        proof.authenticated_passes(paired, directory)


def test_original_calibration_failure_is_never_clamped_or_used_as_reward(grades):
    paired, directory, _, _ = grades
    control = paired["controls"][-1]["index"]
    requests = proof.grading_requests(paired)
    for index, request in enumerate(requests):
        if control not in request["indices"]:
            continue
        sub = directory / f"request-{index:03d}"
        position = request["indices"].index(control)
        response = data.read(sub / "response.json")
        actual = json.loads(response["output"][0]["content"][0]["text"])
        actual["scores"][position]["score"] = 4
        response["output"][0]["content"][0]["text"] = json.dumps(actual)
        save(sub / "response.json", response)
        saved = data.read(sub / "scores.json")
        saved["scores"][position] = 4
        saved["response_sha256"] = data.sha(sub / "response.json")
        save(sub / "scores.json", saved)
    with pytest.raises(ValueError, match="original rubric controls"):
        proof.authenticated_passes(paired, directory)


def test_fixed_plan_has_no_optimizer_or_reserved_confirmation_use():
    assert proof.POLICY["case_indices"] == [0, 1, 214, 215]
    assert proof.POLICY["optimizer_updates"] == 0
    assert proof.POLICY["complete_grading_passes"] == 2
    assert proof.POLICY["GPU_use"] is False
