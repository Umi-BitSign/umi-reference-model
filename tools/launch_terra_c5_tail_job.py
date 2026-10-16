"""Detach a qualified C5-tail job; record exact identity and exit, without retries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import launch_terra_visual_job as detached

ROOT = Path(__file__).resolve().parents[1]


def launch(plan, output, job, *, qualification=None, restore_only=False):
    plan, output, job = (p.resolve() for p in (plan, output, job))
    if (
        not plan.is_relative_to(ROOT / "runs")
        or not output.is_relative_to(ROOT / "checkpoints")
        or not job.is_relative_to(ROOT / "runs")
        or output.exists()
        or job.exists()
    ):
        raise ValueError("C5-tail job needs an owned plan and fresh checkpoint/job paths")
    spec = json.loads(plan.read_text())
    if spec.get("schema") != "umi-studio-terra-c5-tail-training/1":
        raise ValueError("C5-tail detached job requires its own frozen model plan")
    command = [
        sys.executable,
        str(ROOT / "tools/train_terra_c5_tail.py"),
        "--plan",
        str(plan),
        "--output-dir",
        str(output),
    ]
    if "resume" not in spec:
        if qualification is None or restore_only:
            raise ValueError("initial C5-tail training requires its actual completed qualification")
        qualification = qualification.resolve()
        proof = json.loads(qualification.read_text())
        if (
            not qualification.is_relative_to(ROOT / "checkpoints")
            or proof.get("complete") is not True
            or proof.get("numerics_passed") is not True
            or proof.get("plan_sha256") != detached.digest(plan)
            or proof.get("optimizer_updates") != 0
            or proof.get("caption_weight_during_proof") != 0
        ):
            raise ValueError(
                "C5-tail detached job cannot use a missing or different gradient proof"
            )
        command.extend(["--qualification", str(qualification)])
    elif qualification is not None:
        raise ValueError("C5-tail recovery uses its plan-bound qualification only")
    if restore_only:
        command.append("--restore-only")
    manifest = plan.with_name("source-snapshot") / "manifest.json"
    if json.loads(manifest.read_text())["plan_sha256"] != detached.digest(plan):
        raise ValueError("C5-tail detached plan differs from its source archive")
    job.mkdir(parents=True)
    detached.write(
        job / "job.json",
        {
            "schema": "umi-studio-detached-terra-c5-tail-job/1",
            "created_at": detached.now(),
            "plan": str(plan),
            "plan_sha256": detached.digest(plan),
            "launcher_sha256": detached.digest(Path(__file__)),
            "identity_helper_sha256": detached.digest(Path(detached.__file__)),
            "command": command,
            "cwd": str(ROOT),
            "automatic_restarts": 0,
            "miner_or_codex_lb_modified": False,
        },
    )
    argv = [sys.executable, str(Path(__file__)), "--supervise", str(job)]
    with (job / "worker.log").open("x") as log:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = detached.identity(process.pid, argv)
    detached.write(job / "supervisor.json", receipt)
    return receipt


def supervise(job):
    job = job.resolve()
    if not job.is_relative_to(ROOT / "runs"):
        raise ValueError("C5-tail supervisor must remain in the owned runs/ subtree")
    record = json.loads(job.joinpath("job.json").read_text())
    if (
        record["cwd"] != str(ROOT)
        or record["launcher_sha256"] != detached.digest(Path(__file__))
        or record["identity_helper_sha256"] != detached.digest(Path(detached.__file__))
        or record["plan_sha256"] != detached.digest(Path(record["plan"]))
        or record["command"][:2] != [sys.executable, str(ROOT / "tools/train_terra_c5_tail.py")]
    ):
        raise ValueError("C5-tail supervisor source/plan/command binding changed")
    child = subprocess.Popen(
        record["command"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    detached.write(
        job / "worker.json",
        {**detached.identity(child.pid, record["command"]), "recorded_at": detached.now()},
    )
    code = child.wait()
    detached.write(
        job / "exit.json", {"pid": child.pid, "exit_code": code, "observed_at": detached.now()}
    )
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--restore-only", action="store_true")
    parser.add_argument("--supervise", type=Path)
    args = parser.parse_args()
    if args.supervise:
        return supervise(args.supervise)
    if any(p is None for p in (args.plan, args.output_dir, args.job_dir)):
        parser.error("--plan, --output-dir and --job-dir are required")
    print(
        json.dumps(
            launch(
                args.plan,
                args.output_dir,
                args.job_dir,
                qualification=args.qualification,
                restore_only=args.restore_only,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
