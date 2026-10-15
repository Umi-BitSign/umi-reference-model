"""Launch one qualified Terra recovery independently of the chat terminal.

The detached supervisor records its child identity and terminal exit. It never
retries, changes the scientific recipe, or inspects the miner or scoring service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 (Python3.10 training runtime)


def identity(pid, command):
    started = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not started:
        raise RuntimeError("owned process exited before its identity was recorded")
    return {"pid": pid, "uid": os.getuid(), "ps_started_local": started, "argv": command}


def launch(plan, output, job, *, restore_only=False):
    plan, output, job = (path.resolve() for path in (plan, output, job))
    if (
        not plan.is_relative_to(ROOT / "runs")
        or not output.is_relative_to(ROOT / "checkpoints")
        or not job.is_relative_to(ROOT / "runs")
        or output.exists()
        or job.exists()
    ):
        raise ValueError("detached recovery needs a runs/ plan and fresh owned output/job paths")
    record = json.loads(plan.read_text())
    if "resume" not in record or record.get("schema") != "umi-studio-terra-visual-training/1":
        raise ValueError("detached launcher requires a qualified frozen recovery plan")
    if json.loads(plan.with_name("source-snapshot").joinpath("manifest.json").read_text())[
        "plan_sha256"
    ] != digest(plan):
        raise ValueError("detached recovery plan differs from its frozen source archive")
    command = [
        sys.executable,
        str(ROOT / "tools/train_terra_visual.py"),
        "--plan",
        str(plan),
        "--output-dir",
        str(output),
    ]
    if restore_only:
        command.append("--restore-only")
    job.mkdir(parents=True)
    write(
        job / "job.json",
        {
            "schema": "umi-studio-detached-terra-job/1",
            "created_at": now(),
            "plan": str(plan),
            "plan_sha256": digest(plan),
            "launcher_sha256": digest(Path(__file__)),
            "command": command,
            "cwd": str(ROOT),
            "restore_only": restore_only,
            "automatic_restarts": 0,
            "miner_or_codex_lb_modified": False,
        },
    )
    supervisor_command = [sys.executable, str(Path(__file__)), "--supervise", str(job)]
    with (job / "worker.log").open("x") as log:
        process = subprocess.Popen(
            supervisor_command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = identity(process.pid, supervisor_command)
    write(job / "supervisor.json", receipt)
    return receipt


def supervise(job):
    job = job.resolve()
    if not job.is_relative_to(ROOT / "runs"):
        raise ValueError("supervisor receipt must remain in the owned runs/ subtree")
    record = json.loads((job / "job.json").read_text())
    if (
        record["cwd"] != str(ROOT)
        or record["launcher_sha256"] != digest(Path(__file__))
        or record["plan_sha256"] != digest(Path(record["plan"]))
    ):
        raise ValueError("detached launch source/plan binding changed")
    command = record["command"]
    if command[:2] != [sys.executable, str(ROOT / "tools/train_terra_visual.py")]:
        raise ValueError("supervisor may execute only the owned Terra trainer")
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
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--restore-only", action="store_true")
    parser.add_argument("--supervise", type=Path)
    args = parser.parse_args()
    if args.supervise:
        return supervise(args.supervise)
    if any(value is None for value in (args.plan, args.output_dir, args.job_dir)):
        parser.error("--plan, --output-dir and --job-dir are required")
    print(
        json.dumps(launch(args.plan, args.output_dir, args.job_dir, restore_only=args.restore_only))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
