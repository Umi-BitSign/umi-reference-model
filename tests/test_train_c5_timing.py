"""Complete recipe and authentic grading boundaries; no actual model training."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_timing_feature_data as data
import train_c5_timing as training
from test_probe_c5_timing_semantic import Tokenizer


@pytest.fixture
def records():
    return [
        {
            "index": i,
            "row": {
                "source": "synthetic",
                "example_id": str(i),
                "reference": chr(ord("a") + i % 4),
                "role": "training",
                "training_allowed": True,
            },
            "features": {"path": f"features-{i:04d}.npz", "sha256": "0" * 64},
        }
        for i in range(532)
    ]


def pool_for(record, other):
    tokenizer = Tokenizer()
    text = record["row"]["reference"]
    donor = {
        "tokens": tokenizer(other["row"]["reference"], truncation=False).input_ids,
        "text": other["row"]["reference"],
        "kind": "other_training_caption",
        "truncated": False,
        "termination": "eos",
        "donor_features": other["features"],
    }
    return {
        "row": record["row"],
        "candidates": [
            {
                "tokens": tokenizer(text, truncation=False).input_ids,
                "text": text,
                "kind": "human_anchor",
                "truncated": False,
                "termination": "eos",
            },
            donor,
        ],
        "donor": donor,
        "donor_duplicate_index": None,
    }


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def rewards(records, tmp_path, monkeypatch):
    from terra_polarity_judge import payload

    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    donors = training.donor_indices(records)
    pools = [pool_for(row, records[donors[i]]) for i, row in enumerate(records)]
    paired = training.comparison(records, pools)
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    for index, request in enumerate(paired["requests"]):
        sub = semantic / f"request-{index:03d}"
        sub.mkdir()
        scores = [
            10 if row["reference"] == row["candidate"] else 0 for row in request["rows"][:-4]
        ] + [10, 10, 0, 3]
        save(sub / "request.json", payload(request["rows"]))
        save(
            sub / "started.json",
            {
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "pairs": len(scores),
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
                                            {"id": f"pair_{i:02d}", "score": value}
                                            for i, value in enumerate(scores)
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
                "scores": scores,
                "response_sha256": data.sha(sub / "response.json"),
            },
        )
    return paired, semantic


def test_schedule_has_exactly_two_complete_distinct_epochs_and_fixed_final():
    batches = training.batches()
    assert batches == training.batches()
    assert len(batches) == training.RECIPE["planned_optimizer_updates"] == 134
    assert sum(len(batch["indices"]) for batch in batches) == 1064
    orders = []
    for epoch in (1, 2):
        order = [
            index for batch in batches if batch["epoch"] == epoch for index in batch["indices"]
        ]
        assert sorted(order) == list(range(532)) and len(set(order)) == 532
        orders.append(order)
    assert orders[0] != orders[1]
    assert [len(batches[i]["indices"]) for i in (66, 133)] == [4, 4]
    assert training.RECIPE["caption_weight"] == 0
    assert training.RECIPE["GPU_use"] is False


def test_donor_uses_next_distinct_caption_without_dropping_repeated_labels(records):
    records[1]["row"]["reference"] = records[0]["row"]["reference"]
    donors = training.donor_indices(records)
    assert len(donors) == 532 and donors[0] == donors[1] == 2
    for index, other in enumerate(donors):
        assert records[index]["row"]["reference"] != records[other]["row"]["reference"]
    for record in records:
        record["row"]["reference"] = "a"
    with pytest.raises(ValueError, match="distinct training-caption donor"):
        training.donor_indices(records)


def test_pool_deduplicates_observed_donor_tokens_without_inventing_a_second_caption(records):
    saved = pool_for(records[0], records[1])
    training.validate_pool(records[0], records[1], saved, Tokenizer())
    observed = copy.deepcopy(saved["donor"])
    observed.pop("donor_features")
    observed["kind"] = "greedy"
    saved["candidates"][-1] = observed
    saved["donor_duplicate_index"] = 1
    training.validate_pool(records[0], records[1], saved, Tokenizer())
    saved["donor_duplicate_index"] = 0
    with pytest.raises(ValueError, match="donor deduplication"):
        training.validate_pool(records[0], records[1], saved, Tokenizer())


@pytest.mark.parametrize("change", ["role", "anchor", "donor_source", "donor_tokens", "missing"])
def test_pool_refuses_changed_approved_role_source_or_likelihood_binding(records, change):
    saved = pool_for(records[0], records[1])
    if change == "role":
        records[0]["row"]["role"] = "development"
    elif change == "anchor":
        saved["candidates"][0]["text"] = "b"
    elif change == "donor_source":
        saved["donor"] = copy.deepcopy(saved["donor"])
        saved["donor"]["donor_features"]["sha256"] = "1" * 64
    elif change == "donor_tokens":
        saved["donor"]["tokens"] = [5, 1]
    else:
        saved["candidates"].pop()
    with pytest.raises(ValueError):
        training.validate_pool(records[0], records[1], saved, Tokenizer())


def test_complete_raw_rewards_preserve_all532_cases_and_original_control_vectors(rewards):
    paired, directory = rewards
    result = training.authenticated_rewards(paired, directory)
    assert len(result["qualities"]) == len(paired["cases"]) == 532
    assert all(row == [10, 0] for row in result["qualities"])
    assert all(values == [10, 10, 0, 3] for values in result["controls"])
    assert len(result["raw"]) == len(paired["requests"]) * 4
    assert all(len(request["rows"]) <= 16 for request in paired["requests"])


@pytest.mark.parametrize("change", ["failure", "extra", "missing", "prompt", "score"])
def test_training_rewards_reject_partial_or_modified_raw_evidence(rewards, change):
    paired, directory = rewards
    sub = directory / "request-000"
    if change == "failure":
        save(sub / "failure.json", {"http_status": 502})
    elif change == "extra":
        save(directory / "extra.json", {})
    elif change == "missing":
        (sub / "response.json").unlink()
    elif change == "prompt":
        value = data.read(sub / "request.json")
        value["instructions"] += " altered"
        save(sub / "request.json", value)
    else:
        value = data.read(sub / "scores.json")
        value["scores"][0] = 4
        save(sub / "scores.json", value)
    with pytest.raises((ValueError, FileNotFoundError)):
        training.authenticated_rewards(paired, directory)


def test_no_training_plan_can_freeze_before_actual_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)

    def refuse(*args):
        raise ValueError("actual C5 qualification remains unavailable")

    monkeypatch.setattr(training, "qualification", refuse)
    path = tmp_path / "runs/new/plan.json"
    with pytest.raises(ValueError, match="actual C5 qualification"):
        training.freeze(path, tmp_path / "a", tmp_path / "b", tmp_path / "c")
    assert not path.parent.exists()


@pytest.mark.parametrize("negation", [3, 4])
def test_per_request_original_negation_boundary_is_not_relaxed(rewards, negation):
    paired, directory = rewards
    sub = directory / "request-000"
    response = data.read(sub / "response.json")
    content = json.loads(response["output"][0]["content"][0]["text"])
    content["scores"][-1]["score"] = negation
    response["output"][0]["content"][0]["text"] = json.dumps(content)
    save(sub / "response.json", response)
    saved = data.read(sub / "scores.json")
    saved["scores"][-1] = negation
    saved["response_sha256"] = data.sha(sub / "response.json")
    save(sub / "scores.json", saved)
    if negation == 4:
        with pytest.raises(ValueError, match="original controls"):
            training.authenticated_rewards(paired, directory)
    else:
        assert training.authenticated_rewards(paired, directory)["controls"][0][-1] == 3


@pytest.fixture
def fit_fixture(records, tmp_path, monkeypatch):
    import c5_timing_semantic_model as semantic_model
    import numpy as np
    import terra_polarity_judge as judge
    from c5_monotone_feature_timing import WIDTHS
    from test_c5_timing_semantic_model import student

    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    model, tokenizer = student(), Tokenizer()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    generator = np.random.default_rng(37)
    streams = {
        k: (generator.normal(size=(13, width)) * np.linspace(0.1, 2, 13)[:, None]).astype(
            np.float32
        )
        for k, width in WIDTHS.items()
    }
    loads = []

    class Dataset:
        def __init__(self, path):
            pass

        def load(self, index):
            loads.append(index)
            return {
                "row": records[index]["row"],
                "streams": streams,
                "valid_intervals": np.full(12, index % 7 != 0, dtype=np.bool_),
            }

    monkeypatch.setattr(training, "TimingFeatureDataset", Dataset)
    monkeypatch.setattr(semantic_model, "load_student", lambda plan: (model, tokenizer))
    monkeypatch.setattr(semantic_model, "require_model", lambda model, plan: model.require_fixed())
    monkeypatch.setattr(training, "guard", lambda output: None)
    audit = tmp_path / "audit.json"
    save(audit, {"synthetic": True})
    plan = {
        "records": records,
        "batches": training.batches(),
        "model_plan": {},
        "data_audit": data.artifact(audit),
    }
    plan_path = tmp_path / "plan.json"
    save(plan_path, plan)
    monkeypatch.setattr(training, "checked", lambda path: copy.deepcopy(plan))
    output = tmp_path / "output"
    output.mkdir()
    save(output / "worker.json", {"synthetic": True})
    donors = training.donor_indices(records)
    pools = [pool_for(row, records[donors[i]]) for i, row in enumerate(records)]

    def prepare(*args):
        directory = output / "pools"
        directory.mkdir()
        for index, pool in enumerate(pools):
            save(directory / f"clip-{index:04d}.json", pool)
        return pools

    monkeypatch.setattr(training, "prepare_pools", prepare)
    reward_values = {
        "scores": [],
        "qualities": [[10, 0] for _ in records],
        "controls": [[10, 10, 0, 3]],
        "raw": [],
    }
    monkeypatch.setattr(
        training, "authenticated_rewards", lambda paired, directory: copy.deepcopy(reward_values)
    )

    class NoNetworkPacer:
        def score(self, rows, output):
            return {"synthetic": True}

    monkeypatch.setattr(judge, "Pacer", NoNetworkPacer)
    return plan_path, output, plan, model, before, loads


def test_full_synthetic_optimizer_keeps_every_exposure_and_only_serializes_timing(fit_fixture):
    from collections import Counter

    import torch

    path, output, plan, model, before, loads = fit_fixture
    training.train(path, output, plan)
    result = data.read(output / "result.json")
    assert result["completed_epochs"] == 2
    assert result["exposures"] == len(loads) == 1064
    assert Counter(loads) == Counter({i: 2 for i in range(532)})
    assert result["optimizer_updates"] == len(result["steps"]) == 134
    assert result["quality_gain_established"] is False
    assert result["original_backbones_unchanged"] is True
    assert torch.count_nonzero(model.timing.rate_weights) > 0
    for name, value in model.state_dict().items():
        if name != "timing.rate_weights":
            assert torch.equal(value, before[name])
    saved = torch.load(output / "final-adapter.pt", weights_only=True)
    assert set(saved["timing"]) == {"rate_weights"}
    assert saved["exposures"] == 1064 and saved["updates"] == 134
    samples = [
        sample for step in result["steps"] for sample in data.read(data.bound(step))["samples"]
    ]
    unsupported = [sample for sample in samples if sample["index"] % 7 == 0]
    assert unsupported and all(
        not s["risk"]["timing_memory_has_gradient_path"] for s in unsupported
    )


def test_failed_grading_stops_before_any_synthetic_optimizer_creation(fit_fixture, monkeypatch):
    import torch

    path, output, plan, model, before, loads = fit_fixture

    def failed(*args):
        raise ValueError("actual calibration failed")

    monkeypatch.setattr(training, "authenticated_rewards", failed)
    monkeypatch.setattr(
        torch.optim,
        "AdamW",
        lambda *args, **kwargs: pytest.fail("optimizer created after failed calibration"),
    )
    with pytest.raises(ValueError, match="calibration failed"):
        training.train(path, output, plan)
    assert not loads and not (output / "final-adapter.pt").exists()
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


@pytest.fixture
def qualification_fixture(records, tmp_path, monkeypatch):
    import os

    import probe_c5_timing_semantic as proof

    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    rows = records[:4]
    generation = {
        "num_beams": 5,
        "max_length": 2048,
        "length_penalty": 0.6,
        "no_repeat_ngram_size": 16,
    }
    plan_value = {"records": rows, "model_plan": {"reference": {"generation": generation}}}
    monkeypatch.setattr(proof, "checked", lambda path: copy.deepcopy(plan_value))
    judged = [{"pass": number, "qualities": [[10, 0]] * 4, "scores": []} for number in (1, 2)]
    monkeypatch.setattr(proof, "authenticated_passes", lambda *args: (copy.deepcopy(judged), []))
    plan = tmp_path / "runs/proof/plan.json"
    plan.parent.mkdir(parents=True)
    save(plan, plan_value)
    output = tmp_path / "checkpoints/proof"
    output.mkdir(parents=True)
    worker = {
        "pid": 123,
        "uid": os.getuid(),
        "argv": [
            sys.executable,
            str(tmp_path / "tools/probe_c5_timing_semantic.py"),
            "--plan",
            str(plan),
            "--output",
            str(output),
        ],
    }
    save(output / "worker.json", worker)
    save(output / "exit.json", {"pid": 123, "exit_code": 0})
    poll = tmp_path / "runs/poll.json"
    save(
        poll,
        {
            "tool": "write_stdin",
            "requested_session_id": 9,
            "actual_owned_pid": 123,
            "tool_result": {"exit_code": 0},
        },
    )
    save(output / "comparison.json", {"synthetic": True})
    save(output / "pools.json", [])
    save(
        output / "feature-identity.json",
        {"clips": 532, "bitwise_identity_including_signed_zero": True},
    )
    identities = []
    for index, row in enumerate(rows):
        path = output / f"identity-{index:02d}.json"
        save(
            path,
            {
                "row": row["row"],
                "memory_bitwise_equal": True,
                "teacher_forcing_logits_and_loss_bitwise_equal": True,
                "native_generation_tokens_equal": True,
                "generation": generation,
            },
        )
        identities.append(data.artifact(path))
    gradient = {
        "parameters": 4,
        "gradient_values": [1.0, 0.0, 0.0, 0.0],
        "gradient_norm": 1.0,
        "frozen_backbone_gradients_absent": True,
    }
    gradients = []
    for number in (1, 2):
        for index, row in enumerate(rows):
            gradients.append(
                {
                    "pass": number,
                    "case": index,
                    "row": row["row"],
                    "actual_grades": [10, 0],
                    "gradients": copy.deepcopy(gradient),
                    "single_encoder_gradients": copy.deepcopy(gradient),
                    "single_encoder_risk": {"loss": 4.0, "differentiable_encoder_calls": 1},
                    "risk": {"loss": 4.0},
                    "maximum_absolute_gradient_difference": 0.0,
                }
            )
    result = output / "qualification.json"
    save(
        result,
        {
            "schema": "umi-studio-native224-timing-semantic-qualification/1",
            "complete": True,
            "plan": data.artifact(plan),
            "worker": data.artifact(output / "worker.json"),
            "complete_grading_passes": 2,
            "optimizer_updates": 0,
            "original_model_values_unchanged": True,
            "training_only": True,
            "quality_gain_established": False,
            "promotion_authorized": False,
            "comparison": data.artifact(output / "comparison.json"),
            "grading": judged,
            "raw_grading_evidence": [],
            "feature_identity": data.artifact(output / "feature-identity.json"),
            "native_identities": identities,
            "gradients": gradients,
            "pools": data.artifact(output / "pools.json"),
        },
    )
    return plan, result, poll


def test_complete_qualification_contract_is_accepted_only_with_terminal_owner_join(
    qualification_fixture,
):
    _, result, _ = training.qualification(*qualification_fixture)
    assert result["complete"] is True


@pytest.mark.parametrize(
    "change",
    [
        "live",
        "pid",
        "partial_pass",
        "gradient",
        "nonfinite",
        "family_zero",
        "row_join",
        "failed_exit",
        "wrong_program",
    ],
)
def test_training_qualification_refuses_incomplete_or_inconsistent_actual_evidence(
    qualification_fixture, change
):
    plan, result, poll = qualification_fixture
    value = data.read(result)
    if change in ("live", "pid"):
        record = data.read(poll)
        if change == "live":
            record["tool_result"]["session_id"] = 9
        else:
            record["actual_owned_pid"] = 999
        save(poll, record)
    elif change == "partial_pass":
        value["complete_grading_passes"] = 1
    elif change == "gradient":
        value["gradients"][0]["single_encoder_gradients"]["gradient_values"][0] = 2.0
        value["gradients"][0]["single_encoder_gradients"]["gradient_norm"] = 2.0
    elif change == "nonfinite":
        value["gradients"][0]["single_encoder_gradients"]["gradient_values"][0] = float("nan")
    elif change == "family_zero":
        for gradient in value["gradients"]:
            if gradient["case"] < 2:
                for name in ("gradients", "single_encoder_gradients"):
                    gradient[name]["gradient_values"] = [0.0] * 4
                    gradient[name]["gradient_norm"] = 0.0
    elif change == "row_join":
        value["gradients"][0]["row"] = value["gradients"][1]["row"]
    elif change == "failed_exit":
        save(result.parent / "exit.json", {"pid": 123, "exit_code": 1})
    else:
        worker = data.read(result.parent / "worker.json")
        worker["argv"][1] = str(plan.parent / "wrong.py")
        save(result.parent / "worker.json", worker)
    save(result, value)
    with pytest.raises(ValueError):
        training.qualification(plan, result, poll)
