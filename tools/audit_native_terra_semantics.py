"""Blinded, repeated semantic grading of existing C5/native-tail predictions only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from launch_terra_visual_job import digest, identity, now, write
from semantic_quality_judge import calibration_controls

ROOT = Path(__file__).resolve().parents[1]
PARENT = "runs/terra-c5-tail-training-v2-recovery/plan.json"
PREDICTIONS = (
    "checkpoints/terra-c5-tail-training-v2-recovery/cpu-evaluation-004096/evaluation-results.jsonl"
)
DECISION = str(Path(PREDICTIONS).with_name("decision.json"))
CHECKPOINT = "checkpoints/terra-c5-tail-training-v2-recovery/visual-after-004096.safetensors"
STOP = "runs/terra-c5-tail-training-v2-job/exit.json"
RESTORE = "runs/terra-c5-tail-v3-restore-job/exit.json"
TRAIN = "data/how2sign/source/how2sign_realigned_train.csv"
VAL = "data/how2sign/source/how2sign_realigned_val.csv"
POLICY = {
    "model": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "quality_range": [0, 10],
    "passes": 2,
    "seed": 719,
    "maximum_pairs_per_request": 16,
    "maximum_inflight_requests": 1,
    "minimum_seconds_between_request_starts": 30,
    "automatic_retries": 0,
    "optimizer_updates": 0,
    "GPU_use": False,
    "training_allowed": False,
    "promotion_authorized": False,
    "existing_WER_gates_changed": False,
    "interpretation": "same_reward_judge_diagnostic; not_independent_human_validation",
}


def sources():
    names = (
        Path(__file__).name,
        "launch_terra_visual_job.py",
        "semantic_quality_judge.py",
        "probe_jeff_terra_quality.py",
        "terra_quality_training.py",
        "train_terra_c5_tail.py",
    )
    return {"tools/" + name: digest(ROOT / "tools" / name) for name in names}


def make_cases(evaluation, control, predictions, official):
    def index(rows):
        result = {r["example_id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate diagnostic example identity")
        return result

    expected, baseline, candidate = (index(rows) for rows in (evaluation, control, predictions))
    if not candidate or not set(candidate) <= expected.keys() & baseline.keys():
        raise ValueError("diagnostic coverage differs from frozen controls")
    cases, items, unique = [], [], {}
    for example in sorted(candidate):
        row, initial, current = expected[example], baseline[example], candidate[example]
        for value in (row, initial, current):
            if (
                value["role"] != "open_development"
                or value["training_allowed"] is not False
                or value["domain"] != "how2sign"
                or value["source_group"] != row["source_group"]
                or value["signer"] != row["signer"]
            ):
                raise ValueError("semantic audit accepts only matching already-open controls")
        if initial["checkpoint_examples"] != 0 or current["checkpoint_examples"] != 4096:
            raise ValueError("semantic audit requires C5 versus the fixed4096 checkpoint")
        if official.get(example) not in ("official_train", "official_val"):
            raise ValueError("diagnostic example lacks a unique official-split identity")
        indices = []
        for transcript in (initial["hypothesis"], current["hypothesis"]):
            key = row["reference"], transcript
            if key not in unique:
                unique[key] = len(items)
                items.append({"reference": key[0], "candidate": key[1]})
            indices.append(unique[key])
        cases.append(
            {
                "example_id": example,
                "source_group": row["source_group"],
                "signer": row["signer"],
                "publisher_split": official[example],
                "item_indices_C5_then_native4096": indices,
                "WER_C5_then_native4096": [initial["wer"], current["wer"]],
                "role": "open_development_diagnostic_only",
                "training_allowed": False,
            }
        )
    return cases, items


def batches(count):
    result = []
    for repeat in range(POLICY["passes"]):
        order = list(range(count))
        random.Random(POLICY["seed"] + repeat).shuffle(order)
        result.append([order[i : i + 16] for i in range(0, count, 16)])
    return result


def specification():
    parent = json.loads((ROOT / PARENT).read_text())
    prediction_rows = [json.loads(s) for s in (ROOT / PREDICTIONS).read_text().splitlines()]
    decision = json.loads((ROOT / DECISION).read_text())
    stop, restore = (json.loads((ROOT / name).read_text()) for name in (STOP, RESTORE))
    if (
        len(prediction_rows) != 124
        or decision["evaluated_clips"] != 124
        or decision["checkpoint_sha256"] != digest(ROOT / CHECKPOINT)
        or decision["complete_benchmark_score"] is not False
        or decision["passes_open_development_gate"] is not False
        or stop["pid"] != 3863
        or stop["exit_code"] != 1
        or restore["pid"] != 4409
        or restore["exit_code"] != 0
    ):
        raise ValueError("semantic audit source checkpoint or owned-terminal binding differs")
    official = {}
    for name, split in ((TRAIN, "official_train"), (VAL, "official_val")):
        with (ROOT / name).open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                example = row["SENTENCE_NAME"]
                if example in official:
                    raise ValueError("official How2Sign split identity is duplicated")
                official[example] = split
    cases, items = make_cases(
        parent["evaluation"], parent["winner_control"], prediction_rows, official
    )
    controls = []
    for control in calibration_controls()[:3]:
        controls.append({"item_index": len(items), "id": control["id"], "range": control["range"]})
        items.append({"reference": control["reference"], "candidate": control["candidate"]})
    names = (PARENT, PREDICTIONS, DECISION, CHECKPOINT, STOP, RESTORE, TRAIN, VAL)
    return {
        "schema": "umi-studio-native-terra-semantic-audit/1",
        "policy": POLICY,
        "source_sha256": sources(),
        "assets": {name: digest(ROOT / name) for name in names},
        "not_before": (
            datetime.fromisoformat(stop["observed_at"]) + timedelta(seconds=1801)
        ).isoformat(),
        "cases": cases,
        "items": items,
        "calibration_controls": controls,
        "batches": batches(len(items)),
        "complete_benchmark_score": False,
        "new_video_decodes": 0,
        "GPU_use": False,
        "training_references_or_rewards_modified": False,
    }


def freeze(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT / "runs") or path.parent.exists():
        raise ValueError("semantic audit requires a fresh owned runs/ plan directory")
    spec = specification()
    path.parent.mkdir(parents=True)
    write(path, spec)
    print(
        json.dumps(
            {
                "plan_sha256": digest(path),
                "clips": len(spec["cases"]),
                "items_per_pass": len(spec["items"]),
                "requests": sum(map(len, spec["batches"])),
                "not_before": spec["not_before"],
            }
        ),
        flush=True,
    )


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("semantic diagnostic plan/source/evidence binding changed")
    return plan


def cluster_interval(cases, deltas):
    groups = defaultdict(list)
    for row, value in zip(cases, deltas, strict=True):
        groups[row["source_group"]].append(value)
    keys, means, rng = sorted(groups), [], random.Random(POLICY["seed"])
    for _ in range(5000):
        values = [v for _ in keys for v in groups[rng.choice(keys)]]
        means.append(statistics.mean(values))
    means.sort()
    return [means[125], means[4874]]


def summarize(plan, scores):
    if len(scores) != 2 or any(len(s) != len(plan["items"]) for s in scores):
        raise ValueError("semantic audit needs both complete predeclared passes")
    if any(type(v) is not int or not 0 <= v <= 10 for s in scores for v in s):
        raise ValueError("semantic audit score lies outside the owner rubric")
    controls_pass = all(
        c["range"][0] <= s[c["item_index"]] <= c["range"][1]
        for c in plan["calibration_controls"]
        for s in scores
    )
    cases = plan["cases"]
    per_pass = []
    rows = []
    for case in cases:
        a, b = case["item_indices_C5_then_native4096"]
        quality = [statistics.mean(s[i] for s in scores) for i in (a, b)]
        rows.append(
            {
                **case,
                "quality_C5_then_native4096": quality,
                "quality_delta_native_minus_C5": quality[1] - quality[0],
            }
        )
    for s in scores:
        pairs = [[s[i] for i in c["item_indices_C5_then_native4096"]] for c in cases]
        per_pass.append(
            {
                "C5_mean_quality": statistics.mean(p[0] for p in pairs),
                "native4096_mean_quality": statistics.mean(p[1] for p in pairs),
                "mean_delta": statistics.mean(p[1] - p[0] for p in pairs),
            }
        )
    deltas = [r["quality_delta_native_minus_C5"] for r in rows]
    by_split = {}
    for split in sorted({r["publisher_split"] for r in rows}):
        chosen = [r for r in rows if r["publisher_split"] == split]
        by_split[split] = {
            "clips": len(chosen),
            "mean_quality_delta": statistics.mean(
                r["quality_delta_native_minus_C5"] for r in chosen
            ),
        }
    return {
        "complete": True,
        "clips": len(rows),
        "calibration_controls_passed_both_passes": controls_pass,
        "interpretation_permitted": controls_pass,
        "C5_mean_quality": statistics.mean(r["quality_C5_then_native4096"][0] for r in rows),
        "native4096_mean_quality": statistics.mean(
            r["quality_C5_then_native4096"][1] for r in rows
        ),
        "mean_quality_delta_native_minus_C5": statistics.mean(deltas),
        "source_group_bootstrap95_interval": cluster_interval(cases, deltas),
        "improved_clips": sum(v > 0 for v in deltas),
        "regressed_clips": sum(v < 0 for v in deltas),
        "tied_clips": sum(v == 0 for v in deltas),
        "per_pass": per_pass,
        "by_publisher_split": by_split,
        "mean_absolute_repeat_score_difference": statistics.mean(
            abs(a - b) for a, b in zip(*scores, strict=True)
        ),
        "independent_human_validation": False,
        "complete_benchmark_score": False,
        "existing_WER_gates_changed": False,
        "promotion_authorized": False,
        "training_allowed": False,
        "GPU_use": False,
        "rows": rows,
    }


def run(path, output):
    from terra_quality_training import score_with_terra
    from train_terra_c5_tail import gpu_lease

    path, output = path.resolve(), output.resolve()
    plan = checked(path)
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("semantic audit requires a fresh owned runs/ output")
    if datetime.now(timezone.utc) < datetime.fromisoformat(plan["not_before"]):  # noqa: UP017
        raise RuntimeError("semantic audit must respect the scoring-overload cooldown")
    with gpu_lease():
        # An orchestration fence only: this CPU/API diagnostic never allocates Metal.
        output.mkdir(parents=True)
        write(
            output / "run.json",
            {"plan_sha256": digest(path), "started_at": now(), "policy": POLICY},
        )
        all_scores, previous_start = [], None
        for repeat, groups in enumerate(plan["batches"]):
            scores = [None] * len(plan["items"])
            for batch, indices in enumerate(groups):
                if previous_start is not None:
                    remaining = 30 - (time.monotonic() - previous_start)
                    if remaining > 0:
                        time.sleep(remaining)
                if datetime.now(timezone.utc) < datetime.fromisoformat(plan["not_before"]):  # noqa: UP017
                    raise RuntimeError("clock moved before scoring cooldown")
                previous_start = time.monotonic()
                destination = output / f"pass-{repeat + 1}-batch-{batch + 1:03d}"
                result = score_with_terra([plan["items"][i] for i in indices], destination)
                for i, value in zip(indices, result["scores"], strict=True):
                    scores[i] = value
                print(
                    json.dumps(
                        {
                            "pass": repeat + 1,
                            "completed_batches": batch + 1,
                            "total_batches": len(groups),
                            "pairs": len(indices),
                            "scoring_seconds": result["seconds"],
                        }
                    ),
                    flush=True,
                )
            write(output / f"pass-{repeat + 1}-scores.json", scores)
            all_scores.append(scores)
        checked(path)
        decision = summarize(plan, all_scores)
        decision["plan_sha256"] = digest(path)
        write(output / "decision.json", decision)
        print(json.dumps({k: v for k, v in decision.items() if k != "rows"}), flush=True)


def launch(path, output, job):
    path, output, job = (p.resolve() for p in (path, output, job))
    checked(path)
    if (
        output.exists()
        or job.exists()
        or not all(p.is_relative_to(ROOT / "runs") for p in (path, output, job))
    ):
        raise ValueError("semantic launch needs fresh owned runs/ output and job paths")
    job.mkdir(parents=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--plan",
        str(path),
        "--output",
        str(output),
    ]
    write(
        job / "job.json",
        {
            "plan": str(path),
            "plan_sha256": digest(path),
            "source_sha256": digest(Path(__file__)),
            "command": command,
            "automatic_restarts": 0,
        },
    )
    supervisor = [sys.executable, str(Path(__file__).resolve()), "--supervise", str(job)]
    with (job / "worker.log").open("x") as stream:
        child = subprocess.Popen(
            supervisor,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = identity(child.pid, supervisor)
    write(job / "supervisor.json", receipt)
    print(json.dumps(receipt), flush=True)


def supervise(job):
    job = job.resolve()
    if not job.is_relative_to(ROOT / "runs"):
        raise ValueError("semantic supervisor needs an owned job path")
    record = json.loads((job / "job.json").read_text())
    path = Path(record["plan"])
    command = record["command"]
    if (
        record["source_sha256"] != digest(Path(__file__))
        or record["plan_sha256"] != digest(path)
        or command[:2] != [sys.executable, str(Path(__file__).resolve())]
        or len(command) != 6
        or command[2:5:2] != ["--plan", "--output"]
        or command[3] != str(path)
    ):
        raise ValueError("semantic supervisor source/plan/command changed")
    checked(path)
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write(job / "worker.json", {**identity(child.pid, command), "recorded_at": now()})
    code = child.wait()
    write(job / "exit.json", {"pid": child.pid, "exit_code": code, "observed_at": now()})
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--supervise", type=Path)
    args = parser.parse_args()
    if args.supervise:
        return supervise(args.supervise)
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output:
        if args.job:
            launch(args.plan, args.output, args.job)
        else:
            run(args.plan, args.output)
    else:
        parser.error("provide --freeze or --plan with --output")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
