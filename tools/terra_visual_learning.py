"""Generate training-only candidate sets and backpropagate Terra's sequence errors."""

from __future__ import annotations

import hashlib

import torch
from terra_quality_training import minimum_risk, sequence_log_probability
from tiny_rgb_data import pixel_tensor
from tiny_rgb_full_training_tools import decode_rgb_video_fast
from tiny_rgb_winner_student import winner_length_buckets
from transformers import StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutput


def training_pixels(pair, recipe):
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_feature_winner_protocol import ROOT

    row = pair["raw"]
    if row["role"] != "training" or row["training_allowed"] is not True:
        raise ValueError("Terra learning accepts approved training clips only")
    if sha256(ROOT / row["video"]) != row["video_sha256"]:
        raise ValueError("Terra training video bytes changed")
    pixels, indices, info = decode_rgb_video_fast(
        row["video"], recipe["short_side"], recipe["target_fps"], recipe["max_frames"]
    )
    if (
        len(pixels) != pair["header"]["sampled_frames"]
        or hashlib.sha256(indices.tobytes()).hexdigest() != pair["header"]["frame_indices_sha256"]
        or info["source_shape"] != pair["header"]["source_shape"]
    ):
        raise ValueError("Terra training frame timeline changed")
    video, mask, _ = winner_length_buckets(pixel_tensor(pixels)[None], None)
    return video, mask


def complete_candidate(tokens, *, start_id=0, eos_id=1, pad_id=0):
    values = tokens.tolist()
    if not values or values[0] != start_id or eos_id not in values[1:]:
        raise ValueError("training rollout did not produce a complete EOS-terminated caption")
    end = values.index(eos_id, 1)
    if any(value != pad_id for value in values[end + 1 :]):
        raise ValueError("training rollout has content after EOS")
    return values[1 : end + 1]


@torch.no_grad()
def candidate_pool(student, tokenizer, video, mask, reference, *, seed, maximum_tokens=512):
    """Reference never conditions generation; it enters the objective afterward."""
    from evaluate_tiny_rgb_feature_winner import CpuBudget

    student.eval()
    gold = tokenizer(reference, truncation=False).input_ids
    if not gold or gold[-1] != 1 or len(gold) > maximum_tokens:
        raise ValueError("Terra learning never truncates the human caption")
    memory, frame_mask = student.encode(video, mask)
    candidates = [{"tokens": gold, "text": reference, "kind": "human_anchor"}]
    torch.manual_seed(seed)
    for sample in (False, True):
        arguments = {
            "encoder_outputs": BaseModelOutput(last_hidden_state=memory),
            "attention_mask": frame_mask.long(),
            "max_length": maximum_tokens + 1,
            "num_beams": 1,
            "do_sample": sample,
            "use_cache": True,
            "stopping_criteria": StoppingCriteriaList([CpuBudget()]),
        }
        if sample:
            arguments.update(temperature=1.0, top_k=0, top_p=1.0)
        sequence = complete_candidate(student.language.generate(**arguments)[0])
        if sequence not in [row["tokens"] for row in candidates]:
            candidates.append(
                {
                    "tokens": sequence,
                    "text": tokenizer.decode(
                        sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ),
                    "kind": "sampled" if sample else "greedy",
                }
            )
    return candidates


def candidate_labels(video, candidates):
    labels = []
    for candidate in candidates:
        if not candidate["tokens"] or candidate["tokens"][-1] != 1:
            raise ValueError("likelihood requires complete candidate EOS")
        _, _, padded = winner_length_buckets(video, torch.tensor([candidate["tokens"]]))
        labels.append(padded)
    if not labels or len({tuple(row["tokens"]) for row in candidates}) != len(candidates):
        raise ValueError("minimum-risk pool is empty or has duplicate sequences")
    return labels


def backward_candidate_risk(
    student, video, mask, labels, quality, *, alpha, caption_weight, divisor=1
):
    """Exact candidate-set gradient using one released graph at a time.

    Detached analytic coefficients equal d(expected Terra error)/d(log P).
    Scores alone are never passed to backward, and CE is separately measured.
    """
    if divisor < 1 or caption_weight < 0 or not quality or quality[0] != 10:
        raise ValueError("invalid accumulation or Terra rejected the exact human anchor")
    student.eval()  # Same dropout-free policy for sampling, coefficients and likelihood gradients.
    device = next(student.parameters()).device
    video, mask = video.to(device), mask.to(device)
    labels = [value.to(device) for value in labels]
    with torch.no_grad():
        log_probabilities = torch.stack(
            [
                sequence_log_probability(student(video, mask, value)["language"].logits, value)
                for value in labels
            ]
        )
        risk = minimum_risk(log_probabilities, quality, alpha=alpha)
        coefficients = risk["coefficients"].detach()
    caption_loss, gold_logits = None, None
    for index, value in enumerate(labels):
        output = student(video, mask, value)["language"]
        log_probability = sequence_log_probability(output.logits, value)
        objective = coefficients[index] * log_probability
        if index == 0:
            caption_loss = float(output.loss.detach())
            gold_logits = output.logits.detach().cpu()
            objective = objective + caption_weight * output.loss
        (objective / divisor).backward()
        del output, objective, log_probability
    return {
        "expected_terra_error": float(risk["loss"]),
        "caption_loss": caption_loss,
        "loss": float(risk["loss"]) + caption_weight * caption_loss,
        "terra_gradient_coefficient_l1": float(coefficients.abs().sum()),
        "candidate_probabilities": risk["probabilities"].cpu().tolist(),
        "candidate_sequence_log_probabilities": log_probabilities.cpu().tolist(),
        "gold_logits": gold_logits,
    }


@torch.no_grad()
def synchronize_visual_policy(source, destination):
    """Copy only fresh visual tensors; optimizer parameters never change devices."""
    incoming = source.trainable_parameters()
    expected = dict(destination.named_parameters())
    if len(incoming) != 90 or not incoming.keys() <= expected.keys():
        raise ValueError("Terra rollout policy topology differs")
    for name, parameter in incoming.items():
        expected[name].copy_(parameter.detach().cpu())
