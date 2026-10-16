"""Prepare a no-download short-phrase curriculum and its unchanged C5 CPU control."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from launch_terra_visual_job import identity, now, write
from terra_semantic_selection import POLICY as SEMANTIC_POLICY

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "runs/tiny-rgb-visual-teacher-corpus-v1/plan.json"
RESERVATION = ROOT / "runs/terra-word-curriculum-capacity-v1/metadata-only.json"
NATIVE = ROOT / "runs/terra-c5-tail-training-v3-recovery/plan.json"
POLICY = {
    "minimum_words": 1,
    "maximum_words": 3,
    "minimum_seconds": 0.75,
    "maximum_seconds": 12,
    "minimum_training_clips": 64,
    "minimum_development_clips": 16,
    "caption_filter": (
        "complete ASCII alphabetic words, internal hyphens/apostrophes, optional final punctuation"
    ),
    "text_rewritten_or_video_cropped": False,
    "new_data_downloads": False,
    "initialization": "original C5, never previously locally trained weights",
    "C5_upstream_training_inventory_known": False,
    "primary_metric": "Terra semantic quality 0-10",
    "WER_is_diagnostic_only": True,
    "semantic_selection": SEMANTIC_POLICY,
    "development_signers_excluded_from_all_later_STEM_curriculum_training": True,
    "development_may_have_been_seen_by_historical_research_weights": True,
    "CPU_threads": 2,
    "control_GPU_use": False,
    "miner_or_codex_lb_inspection": False,
}
LABEL = re.compile(r"[A-Za-z]+(?:[ '-][A-Za-z]+){0,4}[.!?]?")


def key(pair):
    return pair["raw"]["source"], pair["raw"]["example_id"]


def reserved(pair, signers):
    return pair["raw"]["source"].startswith("asl_stem") and pair["raw"]["signer"] in signers


def select(pairs, signers):
    if len({key(p) for p in pairs}) != len(pairs):
        raise ValueError("short curriculum repeats an original source identity")
    result = []
    for pair in pairs:
        raw, header = pair["raw"], pair["header"]
        if (
            raw["role"] != "training"
            or raw["training_allowed"] is not True
            or pair.get("feature_pretraining_allowed") is not True
        ):
            raise ValueError("short curriculum cannot repurpose protected source data")
        frames, fps = header["source_frames"], header["source_fps"]
        if (
            type(frames) is not int
            or frames < 1
            or type(fps) not in (int, float)
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("short curriculum has invalid original frame-rate metadata")
        duration = frames / fps
        reference = raw["reference"]
        if (
            not math.isfinite(duration)
            or not POLICY["minimum_seconds"] <= duration <= POLICY["maximum_seconds"]
            or not POLICY["minimum_words"] <= len(reference.split()) <= POLICY["maximum_words"]
            or LABEL.fullmatch(reference) is None
        ):
            continue
        development = reserved(pair, signers)
        result.append(
            {
                "source": raw["source"],
                "example_id": raw["example_id"],
                "source_group": raw["source_group"],
                "signer": raw["signer"],
                "reference": reference,
                "duration_sec": duration,
                "role": "development" if development else "training",
                "training_allowed": not development,
                # Original pair is retained byte-for-byte as data. Its nested training
                # permission predates this split and NEVER overrides the role above.
                "original_pair": pair,
            }
        )
    counts = Counter(r["role"] for r in result)
    training_groups = {r["source_group"] for r in result if r["training_allowed"]}
    dev_groups = {r["source_group"] for r in result if not r["training_allowed"]}
    if (
        counts["training"] < POLICY["minimum_training_clips"]
        or counts["development"] < POLICY["minimum_development_clips"]
        or training_groups & dev_groups
    ):
        raise ValueError("short curriculum lacks disjoint training/development coverage")
    return result


def training_pairs(records):
    """Explicit curriculum roles, rather than historical nested permissions, decide."""
    if any(
        r["role"] not in ("training", "development")
        or r["training_allowed"] is not (r["role"] == "training")
        for r in records
    ):
        raise ValueError("short curriculum role or strict training permission differs")
    return [r["original_pair"] for r in records if r["training_allowed"]]


def source():
    from tiny_rgb_feature_winner_protocol import checked_corpus

    corpus, pairs = checked_corpus()
    native = json.loads(NATIVE.read_text())
    reservation = json.loads(RESERVATION.read_text())
    records = select(pairs, set(reservation["heldout_participants"]))
    assets = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in (CORPUS, ROOT / corpus["native_pairs"], RESERVATION, NATIVE)
    }
    code = {
        **native["source_sha256"],
        str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
    }
    code["tools/terra_semantic_selection.py"] = sha256(ROOT / "tools/terra_semantic_selection.py")
    return {
        "schema": "umi-studio-terra-short-curriculum/1",
        "policy": POLICY,
        "assets": assets,
        "source_sha256": code,
        "reference": native["reference"],
        "resource_policy": native["resource_policy"],
        "resource_policy_sha256": native["resource_policy_sha256"],
        "reserved_STEM_participants": reservation["heldout_participants"],
        "records": records,
        "counts": dict(Counter(r["role"] for r in records)),
        "training_clips": len(training_pairs(records)),
        "protected_original_evaluation_unchanged": True,
        "official_or_unseen_to_C5_generalization_claimed": False,
        "promotion_authorized": False,
    }


def verify_assets(plan):
    for name, digest in {**plan["assets"], **plan["source_sha256"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("short curriculum source/metadata binding changed: " + name)
    if sha256(ROOT / plan["resource_policy"]) != plan["resource_policy_sha256"]:
        raise ValueError("short curriculum compute policy changed")
    for row in plan["records"]:
        pair = row["original_pair"]
        raw = pair["raw"]
        paths = [(raw["video"], raw["video_sha256"])]
        for field in ("cache", "sidecar"):
            if pair["teacher"].get(field):
                paths.append((pair["teacher"][field], pair["teacher"][field + "_sha256"]))
        for name, digest in paths:
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT / "data") or sha256(path) != digest:
                raise ValueError("short curriculum approved raw/cache bytes changed")


def checked(path):
    plan = json.loads(path.read_text())
    if plan != source():
        raise ValueError("short curriculum selection, role or policy changed")
    verify_assets(plan)
    return plan


def freeze(path):
    from terra_training_storage import storage

    path = path.resolve()
    if not path.is_relative_to(ROOT / "runs") or path.parent.exists():
        raise ValueError("short curriculum needs a fresh owned runs/ directory")
    storage()
    plan = source()
    verify_assets(plan)
    path.parent.mkdir(parents=True)
    write(path, plan)
    print(
        json.dumps({"plan_sha256": sha256(path), "counts": plan["counts"], "new_downloads": 0}),
        flush=True,
    )


def baseline(path, output):
    from train_terra_c5_tail import gpu_lease

    # Serialize with our own native trainers. No miner/service inspection.
    with gpu_lease():
        return _baseline(path, output)


def _baseline(path, output):
    import torch
    from c5_terra_tail import fixed_hashes, load_model, native_memory
    from evaluate_tiny_rgb_feature_winner import CpuBudget, peak_bytes
    from probe_tiny_rgb_capacity import state_digest
    from terra_training_storage import storage
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from transformers import ByT5Tokenizer, StoppingCriteriaList
    from transformers.modeling_outputs import BaseModelOutput

    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("short C5 control needs a fresh owned runs/ output")
    storage()
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model, reference = load_model()
        model.requires_grad_(False)
        if reference != plan["reference"]:
            raise ValueError("short C5 control reference differs")
        encoder = load_frozen_winner_encoder(reference).requires_grad_(False).eval()
        fixed = fixed_hashes(model, encoder)
        initial_policy = state_digest(model.policy_parameters())
        tokenizer = ByT5Tokenizer.from_pretrained(
            ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
        )
        predictions = []
        for row in plan["records"]:
            if row["role"] != "development":
                continue
            if row["training_allowed"] is not False:
                raise ValueError("short C5 control lost its reserved role")
            started = time.perf_counter()
            with torch.no_grad():
                memory = native_memory(
                    encoder, row["original_pair"], reference["translator_checkpoint_sha256"]
                )
                tokens = model.language.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                    **reference["generation"],
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([CpuBudget()]),
                )
            hypothesis = tokenizer.decode(tokens[0], skip_special_tokens=True)
            words = len(normalize(row["reference"]))
            edits = distance(normalize(row["reference"]), normalize(hypothesis))
            result = {
                **{
                    k: row[k]
                    for k in (
                        "source",
                        "example_id",
                        "source_group",
                        "signer",
                        "reference",
                        "role",
                        "training_allowed",
                    )
                },
                "hypothesis": hypothesis,
                "reference_words": words,
                "edits": edits,
                "WER_diagnostic": edits / words,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_bytes": peak_bytes(),
                "execution_device": "cpu",
            }
            write(output / f"case-{len(predictions):04d}.json", result)
            predictions.append(result)
            print(
                json.dumps(
                    {"completed_control_clips": len(predictions), "example_id": row["example_id"]}
                ),
                flush=True,
            )
            storage()
        if (
            fixed_hashes(model, encoder) != fixed
            or state_digest(model.policy_parameters()) != initial_policy
        ):
            raise RuntimeError("short C5 control changed model weights")
        checked(path)
        write(
            output / "receipt.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "clips": len(predictions),
                "predictions": predictions,
                "all_weights_unchanged": True,
                "original_C5_generation": reference["generation"],
                "mean_WER_diagnostic": sum(r["WER_diagnostic"] for r in predictions)
                / len(predictions),
                "semantic_scores_pending": True,
                "optimizer_updates": 0,
                "new_Terra_requests": 0,
                "GPU_use": False,
                "cpu_threads": 2,
                "miner_or_codex_lb_inspected_or_modified": False,
            },
        )
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    for name, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ.setdefault(name, value)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--baseline-output", type=Path)
    args = parser.parse_args()
    if args.freeze and not (args.plan or args.baseline_output):
        freeze(args.freeze)
    elif args.plan and args.baseline_output and not args.freeze:
        baseline(args.plan, args.baseline_output)
    else:
        parser.error("provide --freeze OR --plan with --baseline-output")


if __name__ == "__main__":
    main()
