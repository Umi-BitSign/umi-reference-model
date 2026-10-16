"""Historical Terra labels for a predeclared, training-only short-phrase replay."""

from __future__ import annotations

import json
import math

from benchmark_fleurs_dev import sha256
from prepare_terra_short_curriculum import ROOT, training_pairs
from terra_short_curriculum_learning import RECIPE as LIVE_RECIPE
from terra_short_curriculum_learning import order

REWARDS = ROOT / "runs/terra-frozen-short-pools-v1/data.json"
REWARDS_SHA = "9b6bf1e961f130db1ba6e9361ffc3eb2fd38cbf53aaff9b4c018767ae94138ac"
RECIPE = {
    **LIVE_RECIPE,
    "objective": "fixed-candidate expected historical Terra error plus auxiliary human-caption CE",
    "local_steps_per_outer_batch": 16,
    "candidate_selection": "first authenticated training pool per identity, no score selection",
    "learning_rate_schedule": "original outer-example schedule, held fixed within 16 local steps",
    "training_scores_are_historical": True,
    "on_policy_training_claimed": False,
    "training_Terra_requests": 0,
    "evaluation_uses_fresh_outputs_and_Terra_scores": True,
    "minimum_seconds_between_request_starts": 60,
    "native_memory_CPU_cache_budget_bytes": 128 * 1024**2,
    "checks_after_outer_exposures": [116, 232, 464],
}


def key(row):
    return row["source"], row["example_id"]


def checked_rewards(data):
    if sha256(REWARDS) != REWARDS_SHA:
        raise ValueError("frozen short rewards changed")
    rewards = json.loads(REWARDS.read_text())
    if (
        rewards.get("schema") != "umi-studio-terra-frozen-short-pools/1"
        or rewards.get("evaluation_material_used_as_training_rewards") is not False
        or rewards.get("new_Terra_requests") != 0
        or rewards.get("optimizer_updates") != 0
    ):
        raise ValueError("frozen reward preparation provenance differs")
    for name, digest in rewards["source_sha256"].items():
        path = (ROOT / name).resolve()
        if not any(path.is_relative_to(ROOT / part) for part in ("runs", "checkpoints")):
            raise ValueError("frozen reward dependency is outside owned artifacts")
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError("frozen reward dependency changed")
    allowed = {
        key(r): r
        for r in data["records"]
        if r["role"] == "training" and r["training_allowed"] is True
    }
    records = rewards["records"]
    if (
        len(records) != len(allowed)
        or len({key(r["data_row"]) for r in records}) != len(records)
        or rewards.get("training_identities") != len(allowed)
    ):
        raise ValueError("frozen reward training coverage differs")
    for row in records:
        original = row["data_row"]
        if original != allowed.get(key(original)):
            raise ValueError("frozen reward labels or training roles changed")
        pool, quality = row["pool"], row["quality"]
        if (
            not pool
            or pool[0]["text"] != original["reference"]
            or pool[0]["kind"] != "human_anchor"
            or len(quality) != len(pool)
            or any(type(q) is not int or not 0 <= q <= 10 for q in quality)
            or quality[0] != 10
        ):
            raise ValueError("frozen reward captions, scores or human anchor differ")
    return records


def batches(data, rewards):
    """Keep the prior outer sampler and checkpoint boundaries, count real repeated work."""
    indexed = {key(r["data_row"]): r for r in rewards}
    ordered = order(data["records"])
    offset = 0
    while offset < len(ordered):
        boundary = next(n for n in RECIPE["checks_after_outer_exposures"] if n > offset)
        count = min(RECIPE["batch_clips"], boundary - offset)
        pairs = ordered[offset : offset + count]
        rows = [indexed[key(p["raw"])] for p in pairs]
        yield offset, rows
        offset += count


def rate(offset, count, total=464):
    return (
        RECIPE["learning_rate"]
        * min(1, (offset + count) / RECIPE["warmup_examples"])
        * (0.1 + 0.9 * (1 + math.cos(math.pi * offset / total)) / 2)
    )


def proof_cases(rewards):
    selected, contrasts = [], set()
    for row in rewards:
        quality = tuple(row["quality"])
        if len(set(quality)) > 1 and quality not in contrasts:
            selected.append(row)
            contrasts.add(quality)
        if len(selected) == 2:
            return selected
    raise ValueError("frozen integration proof needs two distinct authentic semantic contrasts")


def pair(row):
    return training_pairs([row["data_row"]])[0]
