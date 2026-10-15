from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from terra_quality_training import (
    minimum_risk,
    responses_endpoint,
    safe_response,
    sequence_log_probability,
)
from terra_visual_learning import complete_candidate


def test_configured_gateway_is_used_instead_of_leaking_its_key_to_public_api():
    assert (
        responses_endpoint({"OPENAI_BASE_URL": "http://127.0.0.1:2455/v1/"})
        == "http://127.0.0.1:2455/v1/responses"
    )
    assert responses_endpoint({}) == "https://api.openai.com/v1/responses"
    with pytest.raises(ValueError, match="disagree"):
        responses_endpoint({"OPENAI_BASE_URL": "https://a/v1", "OPENAI_API_BASE": "https://b/v1"})
    for value in (
        "http://remote/v1",
        "https://user:password@host/v1",
        "https://host/v1?key=secret",
        "https://host/v1#secret",
        "https://host/v2",
    ):
        with pytest.raises(ValueError):
            responses_endpoint({"OPENAI_BASE_URL": value})


def test_response_evidence_omits_encrypted_reasoning_and_account_fields():
    value = safe_response(
        {
            "status": "completed",
            "safety_identifier": "private",
            "output": [
                {"type": "reasoning", "encrypted_content": "secret"},
                {"type": "message", "content": []},
            ],
        }
    )
    assert value == {"status": "completed", "output": [{"type": "message", "content": []}]}


def test_higher_terra_score_is_reinforced_and_zero_score_is_suppressed():
    log_probabilities = torch.tensor([-4.0, -2.0], requires_grad=True)
    measured = minimum_risk(log_probabilities, [10, 0])
    measured["loss"].backward()
    assert log_probabilities.grad[0] < 0 < log_probabilities.grad[1]
    assert torch.allclose(log_probabilities.grad, measured["coefficients"], atol=1e-8)
    assert measured["coefficients"].abs().sum() > 0


def test_analytic_released_graph_gradient_matches_direct_autograd():
    torch.manual_seed(317)
    parameter = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    vectors = torch.randn(4, 3, 4, dtype=torch.float64)
    logp = (vectors * parameter).sum((1, 2))
    risk = minimum_risk(logp, [10, 4, 1, 0])
    direct = torch.autograd.grad(risk["loss"], parameter, retain_graph=True)[0]
    proxy = (risk["coefficients"].detach() * logp).sum()
    analytic = torch.autograd.grad(proxy, parameter)[0]
    assert torch.allclose(direct, analytic, atol=1e-12, rtol=1e-10)


def test_constant_quality_has_zero_risk_gradient_and_needs_caption_anchor():
    logp = torch.tensor([-2.0, -4.0], requires_grad=True)
    risk = minimum_risk(logp, [0, 0])
    risk["loss"].backward()
    assert torch.equal(logp.grad, torch.zeros_like(logp))


def test_likelihood_keeps_eos_and_ignores_only_explicit_target_padding():
    logits = torch.randn(1, 4, 7, requires_grad=True)
    labels = torch.tensor([[3, 4, 1, -100]])
    value = sequence_log_probability(logits, labels)
    expected = (
        logits.log_softmax(-1)[0, 0, 3]
        + logits.log_softmax(-1)[0, 1, 4]
        + logits.log_softmax(-1)[0, 2, 1]
    )
    assert torch.equal(value, expected)
    value.backward()
    assert logits.grad[0, 2].abs().sum() > 0
    assert logits.grad[0, 3].abs().sum() == 0


def test_complete_rollouts_reject_truncation_and_content_after_eos():
    assert complete_candidate(torch.tensor([0, 7, 8, 1, 0])) == [7, 8, 1]
    for values in ([0, 7, 8], [7, 1], [0, 7, 1, 8]):
        with pytest.raises(ValueError):
            complete_candidate(torch.tensor(values))


@pytest.mark.parametrize("quality", [[-1], [11], [True], [1.1], []])
def test_invalid_terra_quality_is_never_silently_used(quality):
    with pytest.raises(ValueError):
        minimum_risk(torch.tensor([-1.0]), quality)


def test_released_model_graphs_match_complete_minimum_risk_plus_caption_gradient():
    import copy
    from types import SimpleNamespace

    from terra_visual_learning import backward_candidate_risk

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.surface = torch.nn.Linear(3, 7)

        def forward(self, video, mask, labels):
            logits = self.surface(video).expand(1, labels.shape[1], 7)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
            return {"language": SimpleNamespace(logits=logits, loss=loss)}

    torch.manual_seed(317)
    direct = Student()
    released = copy.deepcopy(direct)
    video, mask = torch.randn(1, 1, 3), torch.ones(1, 1, dtype=torch.bool)
    labels = [torch.tensor([[3, 1]]), torch.tensor([[4, 5, 1]]), torch.tensor([[6, 1]])]
    outputs = [direct(video, mask, value)["language"] for value in labels]
    logp = torch.stack(
        [
            sequence_log_probability(output.logits, value)
            for output, value in zip(outputs, labels, strict=True)
        ]
    )
    loss = minimum_risk(logp, [10, 3, 0])["loss"] + 0.1 * outputs[0].loss
    loss.backward()
    measured = backward_candidate_risk(
        released, video, mask, labels, [10, 3, 0], alpha=0.005, caption_weight=0.1
    )
    assert measured["loss"] == pytest.approx(float(loss), abs=1e-6)
    for expected, actual in zip(direct.parameters(), released.parameters(), strict=True):
        assert torch.allclose(expected.grad, actual.grad, atol=1e-7, rtol=1e-5)


def test_terra_scoring_uses_one_frozen_request_and_refuses_accidental_replay(monkeypatch, tmp_path):
    import json

    import terra_quality_training as module

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:2455/v1")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    calls = []

    def respond(endpoint, payload, **kwargs):
        calls.append((endpoint, payload))
        return {
            "model": "gpt-5.6-terra",
            "reasoning": {"effort": "medium"},
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"scores": [{"id": "pair_00", "score": 10}]}),
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(module, "request", respond)
    output = tmp_path / "receipt"
    rows = [{"reference": "An apple", "candidate": "An apple"}]
    result = module.score_with_terra(rows, output)
    assert result["scores"] == [10] and len(calls) == 1
    assert calls[0][0] == "http://127.0.0.1:2455/v1/responses"
    with pytest.raises(ValueError, match="fresh"):
        module.score_with_terra(rows, output)
    assert len(calls) == 1


def test_development_reference_cannot_enter_terra_visual_training():
    from terra_visual_learning import training_pixels

    with pytest.raises(ValueError, match="approved training"):
        training_pixels({"raw": {"role": "open_development", "training_allowed": False}}, {})


def test_terra_evaluator_preserves_all_existing_generation_and_quality_gates():
    import ast

    root = Path(__file__).resolve().parents[1] / "tools"
    original = (root / "evaluate_tiny_rgb_feature_winner.py").read_text()
    expected = original.replace(
        "from train_tiny_rgb_capacity_cpu import storage\n"
        "    from train_tiny_rgb_feature_winner_full import checked_plan",
        "from terra_training_storage import storage\n"
        "    from train_terra_visual import checked_plan",
    ).replace('"raw_feature_winner_student_cpu"', '"terra_raw_feature_winner_student_cpu"')
    before = ast.parse(expected)
    after = ast.parse((root / "evaluate_terra_visual.py").read_text())
    before.body.pop(0)
    after.body.pop(0)
    # Imports are sorted by Ruff. Only loader, resource guard and model label
    # differ; generation, metric calculation, rejection and promotion do not.
    for tree in (before, after):
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        imports = [node for node in function.body if isinstance(node, ast.Import | ast.ImportFrom)]
        statements = [node for node in function.body if node not in imports]
        function.body = sorted(imports, key=ast.dump) + statements
    assert ast.dump(before) == ast.dump(after)


def test_legacy_weight_norm_policy_is_independently_loaded_instead_of_deepcopied():
    import ast
    import copy

    layer = torch.nn.utils.weight_norm(torch.nn.Linear(3, 4)).requires_grad_(False)
    with pytest.raises(RuntimeError, match="deepcopy"):
        copy.deepcopy(layer)
    source = Path(__file__).resolve().parents[1] / "tools/train_terra_visual.py"
    tree = ast.parse(source.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr == "deepcopy" for node in calls
    )


def test_training_storage_reserves_future_outputs_without_lowering_ingestion_guards(tmp_path):
    from types import SimpleNamespace

    from terra_training_storage import POLICY, storage

    output = tmp_path / "checkpoints" / "terra-test"
    output.mkdir(parents=True)
    required = (
        POLICY["physical_floor_bytes"] + POLICY["headroom_bytes"] + POLICY["output_budget_bytes"]
    )
    result = storage(
        215000000,
        output=output,
        root=tmp_path,
        disk_usage=lambda root: SimpleNamespace(free=required),
    )
    assert result["remaining_output_reserve_bytes"] == 2 * 1024**3
    with pytest.raises(RuntimeError, match="physical floor"):
        storage(root=tmp_path, disk_usage=lambda root: SimpleNamespace(free=required - 1))
    with pytest.raises(RuntimeError, match="output budget"):
        storage(POLICY["output_budget_bytes"] + 1, root=tmp_path)
    with pytest.raises(ValueError, match="checkpoint subtree"):
        storage(output=tmp_path / "data", root=tmp_path)
