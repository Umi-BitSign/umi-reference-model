from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_terra_tail as module
from benchmark_fleurs_dev import sha256
from probe_jeff_terra_quality import terra_payload
from terra_visual_learning import backward_candidate_risk


class Language(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.block = torch.nn.ModuleList()
        for _ in range(6):
            block = torch.nn.Module()
            attention = torch.nn.Module()
            for name in ("q", "k", "v", "o"):
                attention.add_module(name, torch.nn.Linear(4, 4, bias=False))
            cross = torch.nn.Module()
            cross.EncDecAttention = attention
            block.layer = torch.nn.ModuleList([torch.nn.Identity(), cross])
            self.decoder.block.append(block)
        self.head = torch.nn.Linear(4, 4)

    def forward(self, *, encoder_outputs, attention_mask, labels, **kwargs):
        value = encoder_outputs.last_hidden_state.mean(1, keepdim=True)
        for block in self.decoder.block:
            a = block.layer[1].EncDecAttention
            value = value + (a.q(value) + a.k(value) + a.v(value) + a.o(value)).tanh()
        logits = self.head(value).expand(1, labels.shape[1], 4)
        return SimpleNamespace(
            logits=logits,
            loss=torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
            ),
        )


def test_tail_selection_preserves_values_and_direct_forward():
    torch.manual_seed(317)
    language = Language()
    before = copy.deepcopy(language.state_dict())
    model = module.NativeTailRefiner(language)
    assert len(model.trainable_parameters()) == 8
    assert all("block.4." in k or "block.5." in k for k in model.trainable_parameters())
    assert all(torch.equal(v, before[k]) for k, v in language.state_dict().items())
    model.train(True)
    assert not model.training and not language.training
    memory, mask, labels = (
        torch.randn(1, 3, 4),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[2, 1]]),
    )
    actual = model(memory, mask, labels)["language"]
    expected = language(
        encoder_outputs=SimpleNamespace(last_hidden_state=memory),
        attention_mask=mask,
        labels=labels,
    )
    assert torch.equal(actual.logits, expected.logits)


def test_actual_risk_reaches_only_eight_tail_weights_with_no_caption_term():
    torch.manual_seed(317)
    model = module.NativeTailRefiner(Language())
    memory, mask = torch.randn(1, 3, 4), torch.ones(1, 3, dtype=torch.bool)
    result = backward_candidate_risk(
        model,
        memory,
        mask,
        [torch.tensor([[2, 1]]), torch.tensor([[3, 1]])],
        [10, 0],
        alpha=0.005,
        caption_weight=0,
    )
    assert result["terra_gradient_coefficient_l1"] > 0
    assert len(module.gradients(model)) == 8
    assert all(p.grad is None for p in model.frozen_parameters().values())


def test_policy_sync_copies_only_selected_values_even_to_fully_frozen_rollout():
    torch.manual_seed(317)
    first = module.NativeTailRefiner(Language())
    other = module.NativeTailRefiner(Language()).requires_grad_(False)
    frozen = {k: p.detach().clone() for k, p in other.frozen_parameters().items()}
    module.synchronize_policy(first, other)
    assert all(
        torch.equal(p, other.policy_parameters()[k]) for k, p in first.policy_parameters().items()
    )
    assert all(torch.equal(p, frozen[k]) for k, p in other.frozen_parameters().items())
    assert not other.trainable_parameters()


def test_memory_padding_preserves_all_real_values_and_masks_only_suffix():
    memory = torch.arange(13 * 4).reshape(1, 13, 4).float()
    padded, mask = module.padded_memory(memory)
    assert padded.shape == (1, 16, 4)
    assert torch.equal(padded[:, :13], memory) and not padded[:, 13:].any()
    assert mask[:, :13].all() and not mask[:, 13:].any()
    with pytest.raises(ValueError, match="drop frames"):
        module.padded_memory(torch.zeros(1, 513, 4))


def test_labels_preserve_observed_eos_or_explicit_prefix_never_invent_tokens():
    complete = {"tokens": [7, 1], "truncated": False, "termination": "eos"}
    prefix = {"tokens": [8] * 512, "truncated": True, "termination": "training_token_horizon"}
    labels = module.candidate_labels([complete, prefix])
    assert labels[0].tolist() == [[7, 1, -100, -100, -100, -100, -100, -100]]
    assert labels[1].tolist() == [[8] * 512]
    with pytest.raises(ValueError, match="distinct"):
        module.candidate_labels([complete, complete])
    with pytest.raises(ValueError):
        module.candidate_labels([{**prefix, "tokens": [8] * 511}])


def test_greedy_pool_never_supplies_reference_to_generation():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs == {"truncation": False}
            return SimpleNamespace(input_ids=[7, 1])

        def decode(self, tokens, **kwargs):
            return "native candidate"

    language = Language()

    def generate(**kwargs):
        assert "labels" not in kwargs and "reference" not in kwargs and "input_ids" not in kwargs
        assert kwargs["num_beams"] == 1 and kwargs["max_length"] == 513
        assert not kwargs["do_sample"]
        return torch.tensor([[0, 8, 1]])

    language.generate = generate
    model = module.NativeTailRefiner(language)
    pool = module.candidate_pool(model, Tokenizer(), torch.zeros(1, 3, 4), "human reference")
    assert [c["kind"] for c in pool] == ["human_anchor", "greedy"]
    assert pool[1]["tokens"] == [8, 1]


def test_protected_role_rejected_before_feature_loading():
    with pytest.raises(ValueError, match="approved training"):
        module.native_memory(
            None,
            {
                "feature_pretraining_allowed": True,
                "raw": {"role": "open_development", "training_allowed": False},
                "teacher": {},
            },
            "C5",
        )


def test_existing_judge_reuses_only_authenticated_matching_text_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    directory = tmp_path / "checkpoints/evidence"
    directory.mkdir(parents=True)
    pair = {"raw": {"reference": "correct caption"}}
    pools = [
        [
            {
                "kind": "human_anchor",
                "tokens": [7, 1],
                "text": "correct caption",
                "truncated": False,
                "termination": "eos",
            },
            {
                "kind": "c5_teacher",
                "tokens": [8, 1],
                "text": "native candidate",
                "truncated": False,
                "termination": "eos",
            },
        ]
    ]
    inputs = {"pairs": [pair], "pools": pools, "quality": [[10, 3]]}
    response = {
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
                                        {"id": "pair_00", "score": 10},
                                        {"id": "pair_01", "score": 3},
                                ]
                            }
                        ),
                    }
                ],
            }
        ],
    }
    values = {
        "inputs": inputs,
        "response": response,
        "request": terra_payload(
            [{"reference": "correct caption", "candidate": c["text"]} for c in pools[0]]
        ),
    }
    for key, value in values.items():
        directory.joinpath(key + ".json").write_text(json.dumps(value))
    directory.joinpath("receipt.json").write_text(
        json.dumps(
            {
                "inputs_sha256": sha256(directory / "inputs.json"),
                "judge_response_sha256": sha256(directory / "response.json"),
            }
        )
    )
    plan = {
        "proof_cases": [pair],
        "judge_source": {
            k: {"path": str(p.relative_to(tmp_path)), "sha256": sha256(p)}
            for k in ("inputs", "response", "request", "receipt")
            for p in [directory / (k + ".json")]
        },
    }
    actual = module.existing_judge_cases(plan)
    assert actual[0]["quality"] == [10, 3] and actual[0]["pool"][1]["kind"] == "greedy"
    directory.joinpath("response.json").write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        module.existing_judge_cases(plan)
