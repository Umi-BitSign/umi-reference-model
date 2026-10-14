"""Freeze raw appearance with the complete winning sequence/language backbone."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "runs/tiny-rgb-visual-teacher-corpus-v1/plan.json"
BUDGET = ROOT / "runs/c5-resource-policy-v2-certification48/gpu-budget.json"
RECIPE = {
    "capacity": "large",
    "short_side": 192,
    "target_fps": 12.5,
    "max_frames": 512,
    "max_text_tokens": 512,
    "latent_coefficient": 0.5,
    "feature_coefficient": 1.0,
    "trainable_parameter_tensors": 90,
    "trainable_parameters": 17526750,
    "frozen_language_parameters": 167539584,
    "frozen_semantic_encoder_parameters": 503411144,
    "retained_native_semantic_encoder": True,
    "initial_diagnostic_weights_reused": False,
}


def checked_corpus():
    from benchmark_fleurs_dev import sha256
    from c5_full_coverage_mps import load_training

    plan = json.loads(CORPUS.read_text())
    if (
        plan["schema"] != "umi-studio-raw-rgb-feature-teacher-corpus/1"
        or plan["feature_pretraining_clips"] != 7523
    ):
        raise ValueError("student corpus differs from native teacher audit")
    for name, digest in {**plan["source_sha256"], **plan["dependency_sha256"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("native student corpus source/bytes changed")
    snapshot = json.loads(CORPUS.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(CORPUS):
        raise ValueError("native student corpus snapshot changed")
    pairs = [json.loads(line) for line in (ROOT / plan["native_pairs"]).read_text().splitlines()]
    current_allowed = {
        (row["source"], row["id"])
        for row in load_training(
            ROOT / "runs/c5-all-source-corpus-v2/audit.json",
            ROOT / "runs/c5-all-source-corpus-v2/training-streams.jsonl",
        )
    }
    if (
        len(pairs) != 7523
        or len({(p["raw"]["source"], p["raw"]["example_id"]) for p in pairs}) != 7523
        or any(
            p["raw"]["role"] != "training"
            or p["raw"]["training_allowed"] is not True
            or p["feature_pretraining_allowed"] is not True
            or (p["raw"]["source"], p["raw"]["example_id"]) not in current_allowed
            for p in pairs
        )
    ):
        raise ValueError("student corpus has duplicate/protected/unavailable targets")
    return plan, pairs


def select_cases(pairs):
    if not pairs or any(
        p["raw"]["role"] != "training"
        or p["raw"]["training_allowed"] is not True
        or p["feature_pretraining_allowed"] is not True
        for p in pairs
    ):
        raise ValueError("winner student proof uses available approved training identities only")
    # Byte length is the winner decoder's target surface, unlike T5-Small subwords.
    ordered = sorted(pairs, key=lambda p: (p["raw"]["source"], p["raw"]["example_id"]))
    chosen = [
        min(ordered, key=lambda p: p["header"]["sampled_frames"]),
        max(ordered, key=lambda p: p["header"]["sampled_frames"]),
        max(ordered, key=lambda p: len(p["raw"]["reference"].encode("utf-8"))),
    ]
    for prefix in ("fleurs", "asl_stem", "openasl"):
        chosen.append(next(p for p in ordered if p["raw"]["source"].startswith(prefix)))
    keys = {(p["raw"]["source"], p["raw"]["example_id"]) for p in chosen}
    return [p for p in ordered if (p["raw"]["source"], p["raw"]["example_id"]) in keys]


def sources():
    from benchmark_fleurs_dev import sha256
    from train_tiny_rgb_visual_teacher import sources as feature_sources

    result = feature_sources()
    for name in (
        "tiny_rgb_winner_student.py",
        "c5_winner_language_decoder.py",
        "tiny_rgb_winner_student_protocol.py",
        "tiny_rgb_feature_winner_student.py",
        "tiny_rgb_feature_winner_protocol.py",
        "probe_tiny_rgb_feature_winner.py",
        "probe_tiny_rgb_winner_student.py",
        "c5_winner_reference.py",
        "c5_winner_mps_safe.py",
    ):
        result[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return result


def checked_plan(path):
    from benchmark_fleurs_dev import sha256
    from c5_winner_reference import load_reference
    from tiny_rgb_full_run import runtime_binding

    plan = json.loads(path.read_text())
    if (
        plan["schema"] != "umi-studio-feature-winner-student-proof/1"
        or plan["recipe"] != RECIPE
        or plan["source_sha256"] != sources()
        or plan["runtime_binding"] != runtime_binding()
        or plan["corpus_sha256"] != sha256(CORPUS)
        or plan["resource_policy_sha256"] != sha256(BUDGET)
        or plan["winner_reference"] != load_reference()
    ):
        raise ValueError("winner student proof immutable bindings changed")
    _, pairs = checked_corpus()
    if select_cases(pairs) != plan["cases"]:
        raise ValueError("winner student protected real-case selection changed")
    snapshot = json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("winner student proof differs from source archive")
    return plan


def freeze(path):
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from c5_winner_reference import load_reference
    from tiny_rgb_full_run import runtime_binding
    from train_tiny_rgb_capacity_cpu import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("winner student proof requires new runs/ plan")
    storage()
    _, pairs = checked_corpus()
    plan = {
        "schema": "umi-studio-feature-winner-student-proof/1",
        "recipe": RECIPE,
        "cases": select_cases(pairs),
        "source_sha256": sources(),
        "runtime_binding": runtime_binding(),
        "corpus_sha256": sha256(CORPUS),
        "resource_policy_sha256": sha256(BUDGET),
        "winner_reference": load_reference(),
        "optimizer_updates": 0,
        "training_only": True,
        "miner_modified": False,
        "held_out_quality_verified": False,
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, path.with_name("source-snapshot"))
    checked_plan(path)
    print(
        json.dumps({"plan_sha256": sha256(path), "real_training_cases": len(plan["cases"])}),
        flush=True,
    )
