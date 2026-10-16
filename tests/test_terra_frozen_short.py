from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import terra_frozen_short as module
import train_terra_frozen_short as trainer


def record(index, *, development=False):
    raw = {"source": "stem", "example_id": str(index), "reference": "Short phrase"}
    return {
        **raw,
        "role": "development" if development else "training",
        "training_allowed": not development,
        "original_pair": {"raw": raw, "feature_pretraining_allowed": True},
    }


@pytest.fixture
def rewards_fixture(monkeypatch, tmp_path):
    rows = [record(0), record(1)]
    pools = [
        {
            "data_row": row,
            "pool": [{"text": row["reference"], "kind": "human_anchor"}],
            "quality": [10],
        }
        for row in rows
    ]
    source = tmp_path / "runs/source.json"
    source.parent.mkdir()
    source.write_text("authentic source")
    path = source.with_name("rewards.json")
    document = {
        "schema": "umi-studio-terra-frozen-short-pools/1",
        "evaluation_material_used_as_training_rewards": False,
        "new_Terra_requests": 0,
        "optimizer_updates": 0,
        "training_identities": 2,
        "source_sha256": {"runs/source.json": module.sha256(source)},
        "records": pools,
    }
    path.write_text(json.dumps(document))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "REWARDS", path)
    monkeypatch.setattr(module, "REWARDS_SHA", module.sha256(path))
    return {"records": [*rows, record(2, development=True)]}, document, path, source


def test_rewards_require_exact_training_coverage_and_authenticated_dependencies(rewards_fixture):
    data, document, _, _ = rewards_fixture
    assert module.checked_rewards(data) == document["records"]


def test_changed_reward_bytes_are_rejected(rewards_fixture):
    data, _, path, _ = rewards_fixture
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="rewards changed"):
        module.checked_rewards(data)


def test_changed_original_response_is_rejected(rewards_fixture):
    data, _, _, source = rewards_fixture
    source.write_text("changed")
    with pytest.raises(ValueError, match="dependency changed"):
        module.checked_rewards(data)


@pytest.mark.parametrize(
    "kind",
    ["dev", "role", "duplicate", "missing", "anchor", "bool_score", "range", "count", "label"],
)
def test_hash_binding_does_not_replace_role_and_label_checks(rewards_fixture, monkeypatch, kind):
    data, document, path, _ = rewards_fixture
    row = document["records"][0]
    if kind == "dev":
        row["data_row"] = data["records"][-1]
    elif kind == "role":
        row["data_row"]["training_allowed"] = False
    elif kind == "duplicate":
        document["records"][1] = copy.deepcopy(row)
    elif kind == "missing":
        document["records"].pop()
    elif kind == "anchor":
        row["quality"][0] = 9
    elif kind == "bool_score":
        row["quality"][0] = True
    elif kind == "range":
        row["quality"].append(11)
        row["pool"].append({"text": "Wrong", "kind": "greedy"})
    elif kind == "count":
        row["quality"].append(1)
    elif kind == "label":
        row["pool"][0]["text"] = "Changed caption"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(module, "REWARDS_SHA", module.sha256(path))
    with pytest.raises(ValueError):
        module.checked_rewards(data)


def test_outer_order_is_unchanged_but_real_update_and_exposure_counts_grow():
    data = {"records": [record(i) for i in range(116)] + [record(200, development=True)]}
    rewards = [{"data_row": r} for r in data["records"] if r["training_allowed"]]
    schedule = list(module.batches(data, rewards))
    actual = [module.key(r["data_row"]) for _, batch in schedule for r in batch]
    expected = [module.key(p["raw"]) for p in module.order(data["records"])]
    assert actual == expected
    assert len(schedule) == 59
    assert len(schedule) * module.RECIPE["local_steps_per_outer_batch"] == 944
    assert len(actual) * module.RECIPE["local_steps_per_outer_batch"] == 7424
    assert [o + len(rs) for o, rs in schedule if len(rs) != 8] == [116, 232]
    assert all(module.rate(o, len(rs)) > 0 for o, rs in schedule)


def test_proof_uses_deterministic_distinct_nonzero_actual_contrasts():
    rows = [
        {"quality": q, "identity": i}
        for i, q in enumerate(([10], [10, 10], [10, 9], [10, 9], [10, 0]))
    ]
    assert [r["identity"] for r in module.proof_cases(rows)] == [2, 4]
    with pytest.raises(ValueError, match="two distinct"):
        module.proof_cases(rows[:4])


def test_recipe_preserves_primary_semantics_and_never_calls_frozen_grades_on_policy():
    recipe = module.RECIPE
    assert recipe["WER_is_diagnostic_only"] is True
    assert recipe["training_Terra_requests"] == 0
    assert recipe["on_policy_training_claimed"] is False
    assert recipe["evaluation_uses_fresh_outputs_and_Terra_scores"] is True
    assert recipe["judge_model"] == "gpt-5.6-terra"
    assert recipe["judge_reasoning_effort"] == "medium"
    assert recipe["risk_alpha"] == module.LIVE_RECIPE["risk_alpha"]
    assert recipe["caption_weight"] == module.LIVE_RECIPE["caption_weight"]
    assert recipe["minimum_seconds_between_request_starts"] == 60


def test_new_plan_distinguishes_outer_exposures_from_repeated_work(monkeypatch, tmp_path):
    data = {"records": [record(i) for i in range(116)]}
    rewards = [{"data_row": r, "quality": [10, i % 5]} for i, r in enumerate(data["records"])]
    monkeypatch.setattr(trainer, "checked_data", lambda _: data)
    monkeypatch.setattr(trainer, "checked_rewards", lambda _: rewards)
    monkeypatch.setattr(
        trainer,
        "short_spec",
        lambda *_: {
            "source_sha256": {},
            "planned_training_examples": 464,
            "checks_after_examples": [116, 232, 464],
            "WER_is_diagnostic_only": True,
            "promotion_authorized": False,
        },
    )
    monkeypatch.setattr(trainer, "artifact", lambda _: {"path": "fixed", "sha256": "hash"})
    result = trainer.specification()
    assert result["planned_training_examples"] == 7424
    assert result["outer_training_exposures"] == 464
    assert result["planned_optimizer_updates"] == 944
    assert result["checks_after_examples"] == [1856, 3712, 7424]
    assert result["WER_is_diagnostic_only"]
    assert not result["promotion_authorized"]


@pytest.fixture
def actual_proof(monkeypatch, tmp_path):
    cases = [
        {"data_row": record(0), "quality": [10, 0]},
        {"data_row": record(1), "quality": [10, 9]},
    ]
    path = tmp_path / "plan.json"
    path.write_text("frozen plan")
    comparisons = tmp_path / "qualification-comparisons.jsonl"
    rows = [
        {
            "identity": list(module.key(r["data_row"])),
            "quality": r["quality"],
            "caption_weight": 0,
            "terra_gradient_coefficient_l1": 0.02,
            "comparison": {"actual_comparison": True},
        }
        for r in cases
    ]
    comparisons.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    receipt = tmp_path / "qualification.json"
    result = {
        "complete": True,
        "actual_Terra_only_gradients": True,
        "plan_sha256": module.sha256(path),
        "initial_policy_digest": "initial",
        "fixed_backbones": {"encoder": "frozen"},
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "caption_weight_during_proof": 0,
        "inputs": cases,
        "comparisons_sha256": module.sha256(comparisons),
    }
    receipt.write_text(json.dumps(result))
    (tmp_path / "exit.json").write_text(json.dumps({"pid": 123, "exit_code": 0}))
    (tmp_path / "worker.json").write_text(json.dumps({"pid": 123}))
    monkeypatch.setattr(trainer, "artifact", lambda _: {"verified": True})
    import tiny_rgb_capacity_protocol

    calls = []
    monkeypatch.setattr(
        tiny_rgb_capacity_protocol, "verified_comparison", lambda r, n: calls.append((r, n))
    )
    return receipt, path, cases, result, calls


def test_actual_gradient_proof_checks_both_eight_tensor_comparisons(actual_proof):
    receipt, path, cases, _, calls = actual_proof
    assert trainer.checked_proof(receipt, path, "initial", {"encoder": "frozen"}, cases)
    assert len(calls) == 2
    assert all(n == 8 for _, n in calls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual_Terra_only_gradients", False),
        ("optimizer_updates", 1),
        ("new_Terra_requests", 1),
        ("caption_weight_during_proof", 0.1),
        ("initial_policy_digest", "changed"),
        ("inputs", []),
        ("comparisons_sha256", "changed"),
    ],
)
def test_caption_only_modified_or_uncommitted_proof_is_rejected(actual_proof, field, value):
    receipt, path, cases, result, _ = actual_proof
    result[field] = value
    receipt.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="proof differs"):
        trainer.checked_proof(receipt, path, "initial", {"encoder": "frozen"}, cases)
