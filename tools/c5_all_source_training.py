"""Freeze/load the complete audited corpus with balanced objective weights."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
import json

from audit_c5_all_source_corpus import PROTECTED, exclusion_reason, protected_sets
from benchmark_fleurs_dev import sha256
from freeze_c5_stem_continuation_plan import ROOT
from interpolate_visual_adapter import DIRECT_NAMES

RECIPE = {
    "id": "c5-all-source-fused-lora-v1", "rank": 16, "alpha": 32.0,
    "visual_last_layers": 4, "visual_learning_rate": 1e-5, "decoder_learning_rate": 3e-5,
    "weight_decay": 0.01, "label_smoothing": 0.1, "gradient_accumulation": 4,
    "cpu_threads": 4, "train_examples": 7_732,
    "checkpoint_after_examples": [512, 1_024, 2_048, 4_096, 7_732],
    "warmup_optimizer_steps": 100, "minimum_learning_rate_ratio": 0.1,
    "domain_mass": {"how2sign": 0.6, "fleurs": 0.2, "asl_stem": 0.15, "openasl": 0.05},
    "how2sign_signers_balanced": True, "frozen_direct_parameter_names": sorted(DIRECT_NAMES),
}


def domain(row: dict) -> str:
    if row["source"].startswith("how2sign"):
        return "how2sign"
    if row["source"].startswith("asl_stem"):
        return "asl_stem"
    if row["source"] == "openasl_scale":
        return "openasl"
    if row["source"] == "fleurs":
        return "fleurs"
    raise ValueError("unknown audited continuous-sign source")


def objective_weights(rows: list[dict]) -> list[float]:
    counts = Counter(domain(row) for row in rows)
    signer_counts = Counter(row["signer"] for row in rows if domain(row) == "how2sign")
    if set(counts) != set(RECIPE["domain_mass"]) or set(signer_counts) != {1, 2, 3, 4, 5, 8, 9, 11}:
        raise ValueError("full corpus lacks a required domain or How2Sign signer")
    weights = []
    for row in rows:
        group = domain(row)
        mass = RECIPE["domain_mass"][group]
        if group == "how2sign":
            weight = len(rows) * mass / len(signer_counts) / signer_counts[row["signer"]]
        else:
            weight = len(rows) * mass / counts[group]
        weights.append(weight)
    if not math.isclose(sum(weights), len(rows), rel_tol=1e-12):
        raise ValueError("full-corpus objective is not normalized")
    return weights


def learning_rate_factor(step: int, total_steps: int) -> float:
    warmup = RECIPE["warmup_optimizer_steps"]
    if total_steps <= warmup:
        raise ValueError("full-corpus schedule is shorter than warmup")
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / (total_steps - warmup))
    floor = RECIPE["minimum_learning_rate_ratio"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def load_corpus(plan: dict) -> tuple[list[dict], list[dict], dict]:
    audit_path = ROOT / plan["corpus_audit"]
    manifest = ROOT / plan["training_manifest"]
    if sha256(audit_path) != plan["corpus_audit_sha256"] or sha256(manifest) != plan["training_manifest_sha256"]:
        raise ValueError("frozen full-corpus audit or manifest changed")
    audit = json.loads(audit_path.read_text())
    if {name: sha256(path) for name, path in PROTECTED.items()} != audit["protected_source_hashes"]:
        raise ValueError("protected evaluation inventories changed after the corpus audit")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows) != RECIPE["train_examples"] or audit["training_manifest_sha256"] != sha256(manifest):
        raise ValueError("frozen full-corpus count differs")
    groups, references = protected_sets()
    if any(exclusion_reason(row, groups, references) for row in rows):
        raise ValueError("full corpus includes a protected source or reference")
    weights = objective_weights(rows)
    for row, weight in zip(rows, weights, strict=True):
        for name in ("cache", "sidecar"):
            if row.get(name):
                path = (ROOT / row[name]).resolve()
                if not path.is_relative_to(ROOT / "data"):
                    raise ValueError("full-corpus cache path leaves data/")
                row[name] = str(path)
        row["loss_weight"] = weight
    hashes = {
        "corpus_audit_sha256": sha256(audit_path),
        "training_manifest_sha256": sha256(manifest),
        **audit["source_hashes"],
    }
    return rows, [], hashes
