"""Terra risk on eight existing C5 cross-attention weights, with fixed native vision."""

from __future__ import annotations

import json

import torch
from benchmark_fleurs_dev import sha256
from c5_winner_reference import ROOT
from probe_tiny_rgb_capacity import state_digest
from terra_teacher_candidates import DECODE, generation_binding
from terra_visual_learning import bounded_candidate
from transformers import StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutput

PARAMETERS = 9437184
TENSORS = 8
TARGET_BUCKETS = (8, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256, 384, 512)


class NativeTailRefiner(torch.nn.Module):
    """Keep initial C5 values and dropout-free behavior; select only tail q/k/v/o."""

    def __init__(self, language):
        super().__init__()
        self.language = language.requires_grad_(False).eval()
        if len(language.decoder.block) != 6:
            raise ValueError("native C5 tail requires the reviewed six-block decoder")
        for block in self.language.decoder.block[-2:]:
            attention = block.layer[1].EncDecAttention
            for projection in (attention.q, attention.k, attention.v, attention.o):
                projection.weight.requires_grad_(True)
        self.policy_names = tuple(name for name, p in self.named_parameters() if p.requires_grad)
        if len(self.policy_names) != TENSORS:
            raise ValueError("native C5 optimizer scope differs from eight projection weights")

    def train(self, mode=True):
        super().train(False)
        return self

    def policy_parameters(self):
        values = dict(self.named_parameters())
        return {name: values[name] for name in self.policy_names}

    def trainable_parameters(self):
        return {name: p for name, p in self.policy_parameters().items() if p.requires_grad}

    def frozen_parameters(self):
        return {name: p for name, p in self.named_parameters() if name not in self.policy_names}

    def forward(self, memory, mask, labels):
        if (
            memory.ndim != 3
            or memory.shape[0] != 1
            or mask.shape != memory.shape[:2]
            or mask.dtype != torch.bool
            or not mask.any()
            or labels.ndim != 2
            or labels.shape[0] != 1
        ):
            raise ValueError("native C5 likelihood requires one complete masked memory/caption")
        return {
            "language": self.language(
                encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                attention_mask=mask.long(),
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
        }


def load_model():
    from c5_winner_language_decoder import load_winner_language

    language, reference = load_winner_language()
    model = NativeTailRefiner(language).eval()
    if sum(p.numel() for p in model.policy_parameters().values()) != PARAMETERS:
        raise ValueError("native C5 tail parameter count differs from CPU preflight")
    return model, reference


def fixed_hashes(model, encoder):
    return {
        "language": state_digest(model.frozen_parameters()),
        "encoder": state_digest(encoder.state_dict()),
    }


def gradients(model):
    selected = model.trainable_parameters()
    if len(selected) != TENSORS:
        raise ValueError("native C5 tail gradient scope changed")
    if any(p.grad is not None for p in model.frozen_parameters().values()):
        raise RuntimeError("frozen C5 language received gradients")
    result = {}
    for name, p in selected.items():
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.norm() <= 0:
            raise RuntimeError("native C5 tail lacks actual finite nonzero Terra gradients")
        result[name] = p.grad.detach().cpu().clone()
    return result


@torch.no_grad()
def synchronize_policy(source, destination):
    before, after = source.policy_parameters(), destination.policy_parameters()
    if before.keys() != after.keys() or len(before) != TENSORS:
        raise ValueError("native C5 rollout/optimizer policy topology differs")
    for name, p in before.items():
        if p.shape != after[name].shape or p.dtype != after[name].dtype:
            raise ValueError("native C5 policy shape/dtype differs")
    for name, p in before.items():
        after[name].copy_(p.detach().to(after[name].device))


@torch.no_grad()
def native_memory(encoder, pair, checkpoint_sha):
    from tiny_rgb_visual_teacher_data import load_targets

    generation_binding(pair, checkpoint_sha)
    teacher = pair["teacher"]
    for field, hash_field in (("cache", "cache_sha256"), ("sidecar", "sidecar_sha256")):
        if teacher.get(field):
            path = (ROOT / teacher[field]).resolve()
            if not path.is_relative_to(ROOT / "data") or sha256(path) != teacher[hash_field]:
                raise ValueError("native approved training feature bytes/path changed")
    if any(p.device.type != "cpu" or p.requires_grad for p in encoder.parameters()):
        raise ValueError("native feature encoder must remain fixed on CPU")
    result = encoder(
        **{name: torch.from_numpy(value)[None] for name, value in load_targets(pair).items()},
        return_dict=True,
    ).last_hidden_state
    if (
        result.shape != (1, pair["header"]["sampled_frames"], 1536)
        or not torch.isfinite(result).all()
    ):
        raise ValueError("native C5 memory timeline/width/finite values differ")
    return result


def padded_memory(memory):
    if memory.ndim != 3 or memory.shape[0] != 1:
        raise ValueError("native C5 memory must contain one complete sequence")
    frames = memory.shape[1]
    size = next((n for n in TARGET_BUCKETS if n >= frames), None)
    if not frames or size is None:
        raise ValueError("native C5 cannot drop frames to fit a memory bucket")
    padded = torch.nn.functional.pad(memory, (0, 0, 0, size - frames))
    mask = torch.arange(size, device=memory.device)[None] < frames
    return padded, mask


def candidate_labels(candidates):
    result = []
    observed = set()
    for candidate in candidates:
        tokens = candidate["tokens"]
        if (
            not tokens
            or any(type(v) is not int or not 0 <= v < 384 for v in tokens)
            or tuple(tokens) in observed
        ):
            raise ValueError("native C5 risk requires distinct nonempty observed token sequences")
        bounded = bounded_candidate(torch.tensor([0, *tokens]), 512)
        if any(candidate[k] != bounded[k] for k in ("tokens", "truncated", "termination")):
            raise ValueError("native C5 candidate termination evidence differs")
        size = next((n for n in TARGET_BUCKETS if n >= len(tokens)), None)
        if size is None:
            raise ValueError("native C5 cannot truncate a candidate target")
        target = torch.full((1, size), -100, dtype=torch.long)
        target[:, : len(tokens)] = torch.tensor(tokens)
        result.append(target)
        observed.add(tuple(tokens))
    if not result:
        raise ValueError("native C5 risk requires a nonempty pool")
    return result


@torch.no_grad()
def candidate_pool(model, tokenizer, memory, reference):
    from evaluate_tiny_rgb_feature_winner import CpuBudget

    gold = tokenizer(reference, truncation=False).input_ids
    if not gold or gold[-1] != 1 or len(gold) > 512:
        raise ValueError("native C5 never truncates the human caption")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("native C5 training rollouts are CPU-only")
    candidates = [
        {
            "tokens": gold,
            "text": reference,
            "kind": "human_anchor",
            "truncated": False,
            "termination": "eos",
        }
    ]
    emitted = bounded_candidate(
        model.language.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=torch.ones(memory.shape[:2], dtype=torch.long),
            **DECODE,
            stopping_criteria=StoppingCriteriaList([CpuBudget()]),
        )[0],
        512,
    )
    if emitted["tokens"] != gold:
        candidates.append(
            {
                **emitted,
                "kind": "greedy",
                "text": tokenizer.decode(
                    emitted["tokens"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            }
        )
    return candidates


def existing_judge_cases(plan):
    """Reuse judgments of identical C5 texts, never the old visual gradient proof."""
    from probe_jeff_terra_quality import terra_payload
    from semantic_quality_judge import parse_terra_scores

    paths = {}
    for key, value in plan["judge_source"].items():
        path = (ROOT / value["path"]).resolve()
        if not path.is_relative_to(ROOT / "checkpoints") or sha256(path) != value["sha256"]:
            raise ValueError("native C5 qualification judge source changed")
        paths[key] = path
    inputs, response, receipt = (
        json.loads(paths[k].read_text()) for k in ("inputs", "response", "receipt")
    )
    if (
        receipt["inputs_sha256"] != sha256(paths["inputs"])
        or receipt["judge_response_sha256"] != sha256(paths["response"])
        or inputs["pairs"] != plan["proof_cases"]
        or response.get("model") != "gpt-5.6-terra"
        or response.get("reasoning", {}).get("effort") != "medium"
    ):
        raise ValueError("native C5 judge provenance, model or training identities differ")
    rows = [
        {"reference": pair["raw"]["reference"], "candidate": candidate["text"]}
        for pair, pool in zip(inputs["pairs"], inputs["pools"], strict=True)
        for candidate in pool
    ]
    if json.loads(paths["request"].read_text()) != terra_payload(rows):
        raise ValueError("native C5 authenticated judge request differs")
    scores = parse_terra_scores(response, [f"pair_{i:02d}" for i in range(len(rows))])
    if [q for qs in inputs["quality"] for q in qs] != [
        scores[f"pair_{i:02d}"] for i in range(len(rows))
    ]:
        raise ValueError("native C5 judge score mapping differs")
    result = []
    for pool, qs in zip(inputs["pools"], inputs["quality"], strict=True):
        i = next(i for i, c in enumerate(pool) if c["kind"] == "c5_teacher")
        candidates = [pool[0], {**pool[i], "kind": "greedy"}]
        quality = [qs[0], qs[i]]
        candidate_labels(candidates)
        if quality[0] != 10:
            raise ValueError("native C5 authenticated human anchor was not ten")
        result.append({"pool": candidates, "quality": quality})
    return result
