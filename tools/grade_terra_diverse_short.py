"""Authenticate added training candidates with the unchanged Terra-medium rubric."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import identity, now, write
from prepare_terra_diverse_short import ROOT, case_seed, merge_pool
from prepare_terra_diverse_short import checked as checked_generation
from semantic_quality_judge import calibration_controls, parse_terra_scores
from terra_short_resume import RecoveryPacer
from terra_training_recovery import artifact, verify_snapshot

GENERATION_PLAN = ROOT / "runs/terra-diverse-short-generation-v1/plan.json"
PENDING = ROOT / "checkpoints/terra-diverse-short-generation-v1/pending-pools.json"
PENDING_SHA = "77ca0db125ce41c825f9976878875f46c1e7a8bab7b822f59f039034ddeb738d"


def unknown_items(records):
    items, positions = [], []
    for row_number, row in enumerate(records):
        for candidate_index, (candidate, quality) in enumerate(
            zip(row["pool"], row["quality"], strict=True)
        ):
            if quality is None:
                items.append(
                    {"reference": row["data_row"]["reference"], "candidate": candidate["text"]}
                )
                positions.append([row_number, candidate_index])
    if not items:
        raise ValueError("no new candidate texts need grading")
    if len({(r["reference"], r["candidate"]) for r in items}) != len(items):
        raise ValueError("new candidate texts must be deduplicated before paid grading")
    return items, positions


def bookend_batches(count):
    if count < 1:
        raise ValueError("new grading requires candidate texts")
    indices = list(range(count))
    random.Random(20261047).shuffle(indices)
    controls = [{"kind": "control", "index": i} for i in range(4)]
    result = [[*controls, *({"kind": "candidate", "index": i} for i in indices[:12])]]
    remaining = indices[12:]
    while len(remaining) > 12:
        result.append([{"kind": "candidate", "index": i} for i in remaining[:16]])
        remaining = remaining[16:]
    result.append([*({"kind": "candidate", "index": i} for i in remaining), *controls])
    return result


def apply_scores(records, positions, scores):
    if len(positions) != len(scores) or any(type(v) is not int or not 0 <= v <= 10 for v in scores):
        raise ValueError("added-candidate scores are incomplete or outside the rubric")
    result = copy.deepcopy(records)
    for (r, c), value in zip(positions, scores, strict=True):
        if result[r]["quality"][c] is not None:
            raise ValueError("new grading must not overwrite historical judgments")
        result[r]["quality"][c] = value
    if any(q is None for row in result for q in row["quality"]):
        raise ValueError("some added candidates remain ungraded")
    return result


def specification():
    generation, original = checked_generation(GENERATION_PLAN)
    if sha256(PENDING) != PENDING_SHA:
        raise ValueError("pending sampled candidates changed")
    pending = json.loads(PENDING.read_text())
    terminal = json.loads(PENDING.with_name("exit.json").read_text())
    worker = json.loads(PENDING.with_name("worker.json").read_text())
    if (
        terminal["exit_code"] != 0
        or terminal["pid"] != worker["pid"]
        or pending["plan_sha256"] != sha256(GENERATION_PLAN)
        or pending["original_rewards"] != generation["original_rewards"]
        or pending["model_training_ready"] is not False
        or pending["original_C5_and_frozen_backbones_unchanged"] is not True
        or pending["heldout_material_used"] is not False
        or len(pending["records"]) != len(original)
    ):
        raise ValueError("candidate generation is incomplete or not training-only")
    for row, base in zip(pending["records"], original, strict=True):
        origin = row["new_sample_origin"]
        expected = merge_pool(base, origin["observed_samples"])
        expected["new_sample_origin"] = origin
        if (
            row != expected
            or origin["plan_sha256"] != sha256(GENERATION_PLAN)
            or origin["seed"] != case_seed(base["data_row"])
            or len(origin["observed_samples"]) != 4
        ):
            raise ValueError(
                "candidate generation changed original labels, roles or sampling provenance"
            )
    items, positions = unknown_items(pending["records"])
    sources = dict(generation["source_sha256"])
    sources["tools/grade_terra_diverse_short.py"] = sha256(Path(__file__))
    return {
        "schema": "umi-studio-terra-diverse-short-grading/1",
        "generation_plan": artifact(GENERATION_PLAN),
        "pending_candidates": artifact(PENDING),
        "generation_terminal": artifact(PENDING.with_name("exit.json")),
        "source_sha256": sources,
        "items": items,
        "positions": positions,
        "batches": bookend_batches(len(items)),
        "controls": calibration_controls(),
        "judge_model": "gpt-5.6-terra",
        "judge_reasoning_effort": "medium",
        "minimum_seconds_between_starts": 60,
        "maximum_pairs_per_request": 16,
        "own_requests_in_flight": 1,
        "automatic_retries": 0,
        "new_rubric_or_prompt": False,
        "GPU_use": False,
        "optimizer_updates": 0,
        "development_used": False,
        "original_scores_preserved": True,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("added-candidate grading plan or scientific inputs changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("new grading requires a fresh owned plan")
    storage(5 * 1024**2)
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "new_pairs": len(plan["items"]),
                "requests": len(plan["batches"]),
            }
        ),
        flush=True,
    )


def run(path, output):
    from c5_terra_tail import candidate_labels
    from probe_jeff_terra_quality import terra_payload
    from terra_training_storage import storage
    from transformers import ByT5Tokenizer

    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("new grading requires a fresh owned output")
    storage(5 * 1024**2)
    pending = json.loads(PENDING.read_text())
    generation = json.loads(GENERATION_PLAN.read_text())
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / generation["reference"]["bundle_root"] / "model/models/byt5_base",
        local_files_only=True,
    )
    for row in pending["records"]:
        candidate_labels(row["pool"])
        if any(
            tokenizer.decode(
                c["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            != c["text"]
            for c in row["pool"]
        ):
            raise ValueError("added-candidate token/text alignment changed")
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        scores, receipts = [None] * len(plan["items"]), []
        pacer = RecoveryPacer()
        for number, batch in enumerate(plan["batches"]):
            rows = [
                {
                    k: (plan["controls"] if r["kind"] == "control" else plan["items"])[r["index"]][
                        k
                    ]
                    for k in ("reference", "candidate")
                }
                for r in batch
            ]
            directory = output / f"batch-{number:03d}"
            result = pacer.score(rows, directory)
            response = json.loads((directory / "response.json").read_text())
            parsed = parse_terra_scores(response, [f"pair_{i:02d}" for i in range(len(rows))])
            if json.loads((directory / "request.json").read_text()) != terra_payload(
                rows
            ) or result["scores"] != [parsed[f"pair_{i:02d}"] for i in range(len(rows))]:
                raise ValueError("new candidate grades differ from their actual prompt/response")
            for entry, value in zip(batch, result["scores"], strict=True):
                if entry["kind"] == "control":
                    lo, hi = plan["controls"][entry["index"]]["range"]
                    if not lo <= value <= hi:
                        raise RuntimeError(
                            "new candidate scorer failed calibration; no training artifact"
                        )
                else:
                    scores[entry["index"]] = value
            receipts.append(
                {
                    "batch": batch,
                    **{
                        name: artifact(directory / f"{name}.json")
                        for name in ("request", "response", "scores")
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "stage": "grade_diverse_short",
                        "successful_requests": number + 1,
                        "new_pairs_scored": sum(v is not None for v in scores),
                    }
                ),
                flush=True,
            )
        records = apply_scores(pending["records"], plan["positions"], scores)
        checked(path)
        write(
            output / "graded-pools.json",
            {
                "schema": "umi-studio-terra-diverse-short-graded-pools/1",
                "plan_sha256": sha256(path),
                "pending_candidates": artifact(PENDING),
                "records": records,
                "new_score_receipts": receipts,
                "training_identities": len(records),
                "new_scored_pairs": len(scores),
                "new_Terra_requests": len(receipts),
                "original_scores_preserved": True,
                "bookend_calibration_passed": True,
                "model_training_ready": True,
                "optimizer_updates": 0,
                "GPU_use": False,
                "development_used": False,
                "historical_frozen_labels_not_on_policy": True,
            },
        )
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output:
        run(args.plan, args.output)
    else:
        parser.error("provide --freeze or --plan and --output")


if __name__ == "__main__":
    main()
