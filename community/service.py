"""Start a model sidecar from an operator-reviewed, hash-bound configuration.

This process has no wallet and cannot publish an endpoint or enable rewards.
Worker commands must include the reviewed sandbox and resource enforcement.
The configuration hash identifies local deployment inputs, not a rights review
or a signed competition policy. Workers still verify their own runtime bundles.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from native_runtime import canonical, digest
from reboot_recovery import RebootRecovery
from sidecar import CommunityModelSidecar, run_sidecar_service
from worker_transport import WarmModelProcess

MAXIMUM_CONFIG_BYTES = 256 * 1024
FIELDS = {"schema", "socket_path", "scoring_policy_sha256", "validator_slot_count", "workers"}
WORKER_FIELDS = {
    "command",
    "environment",
    "cwd",
    "model_revision",
    "startup_seconds",
    "inference_seconds",
    "scratch_directory",
}


class ModelServiceError(RuntimeError):
    """A bounded stage code without local paths, commands or model output."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def read_configuration(path: Path, expected_sha256: str) -> dict:
    digest(expected_sha256)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("configuration must use an absolute path without aliases")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o077
            or not 0 < before.st_size <= MAXIMUM_CONFIG_BYTES
        ):
            raise ValueError("configuration must be a bounded owner-private regular file")
        raw = stream.read(MAXIMUM_CONFIG_BYTES + 1)
        after = os.fstat(stream.fileno())
        linked = path.lstat()
        attributes = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if len(raw) != before.st_size or any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(linked, field)
            for field in attributes
        ):
            raise ValueError("configuration changed while reading")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("configuration differs from reviewed inputs")
    document = json.loads(raw)
    if (
        not isinstance(document, dict)
        or set(document) != FIELDS
        or document["schema"]
        not in {"umi-community-model-service/1", "umi-community-model-standby/1"}
        or canonical(document) != raw
    ):
        raise ValueError("configuration schema or canonical encoding differs")
    if document["schema"] == "umi-community-model-standby/1":
        # A local baseline can run before a reward policy exists. Null is
        # intentional: the miner's policy-bound capacity check rejects it.
        if (
            document["scoring_policy_sha256"] is not None
            or type(document["validator_slot_count"]) is not int
            or document["validator_slot_count"] != 1
            or not isinstance(document["workers"], list)
            or len(document["workers"]) != 1
        ):
            raise ValueError("standby requires one worker and no scoring policy")
    else:
        digest(document["scoring_policy_sha256"])
    return document


@contextlib.contextmanager
def service_lock(socket_path: Path, *, recover_after_reboot: bool = False):
    parent = socket_path.parent
    if not socket_path.is_absolute() or parent.resolve(strict=True) != parent:
        raise ValueError("socket parent must be an absolute directory without aliases")
    metadata = parent.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError("socket parent must be owner-private")
    path = Path(f"{socket_path}.service.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("model service lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        linked = path.lstat()
        if (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("model service lock changed while acquiring")
        if not recover_after_reboot:
            for artifact in (
                socket_path,
                Path(f"{socket_path}.capacity.json"),
                Path(f"{socket_path}.reboot.json"),
            ):
                if artifact.exists() or artifact.is_symlink():
                    raise FileExistsError("existing model artifacts require recovery mode")
        yield
    finally:
        # Leave the lock inode in place so another process cannot lock a new file
        # at the same name while the previous descriptor is still held.
        os.close(descriptor)


def build_sidecar(document: dict) -> CommunityModelSidecar:
    specifications = document["workers"]
    if not isinstance(specifications, list) or not 1 <= len(specifications) <= 256:
        raise ValueError("configuration must declare bounded model workers")
    workers = []
    scratches = []
    for specification in specifications:
        if not isinstance(specification, dict) or set(specification) != WORKER_FIELDS:
            raise ValueError("worker configuration fields differ")
        arguments = dict(specification)
        for field in ("cwd", "scratch_directory"):
            if not isinstance(arguments[field], str) or not arguments[field]:
                raise ValueError("worker paths must be explicit")
            arguments[field] = Path(arguments[field])
        scratch = arguments["scratch_directory"]
        if any(scratch.is_relative_to(old) or old.is_relative_to(scratch) for old in scratches):
            raise ValueError("workers must have separate non-overlapping scratch directories")
        scratches.append(scratch)
        workers.append(WarmModelProcess(**arguments))
    return CommunityModelSidecar(workers, validator_slot_count=document["validator_slot_count"])


def run_configuration(
    path: Path, expected_sha256: str, *, recover_after_reboot: bool = False
) -> None:
    stage = "model_service_configuration_invalid"
    try:
        document = read_configuration(path, expected_sha256)
        if not isinstance(document["socket_path"], str) or not document["socket_path"]:
            raise ValueError("socket path must be explicit")
        socket_path = Path(document["socket_path"])
        stage = "model_service_exclusive_startup_failed"
        with service_lock(socket_path, recover_after_reboot=recover_after_reboot):
            recovery = None
            if recover_after_reboot:
                stage = "model_service_reboot_recovery_failed"
                recovery = RebootRecovery(socket_path, path, expected_sha256, document)
                recovery.prepare()
            stage = "model_service_worker_configuration_invalid"
            sidecar = build_sidecar(document)
            if recovery is not None:
                stage = "model_service_reboot_recovery_failed"
                recovery.begin()
            stage = "model_service_runtime_failed"
            run_sidecar_service(
                sidecar, socket_path, scoring_policy_sha256=document["scoring_policy_sha256"]
            )
            if recovery is not None:
                stage = "model_service_reboot_recovery_failed"
                recovery.finish()
    except Exception as error:
        raise ModelServiceError(stage) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--recover-after-reboot", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        run_configuration(
            args.config, args.expected_config_sha256, recover_after_reboot=args.recover_after_reboot
        )
    except ModelServiceError as error:
        # No configuration, commands, environment values or local paths in logs.
        print(
            json.dumps({"status": "blocked", "reason_code": error.reason_code}),
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
