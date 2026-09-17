"""Verify the supplied community ZIP and stage an offline UMI model bundle.

Standard library only. Does not import the model, install dependencies, fetch
URLs, approve rights, attribute rewards or change the active baseline registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE_SHA256 = "f78979599486456e06e7886b126169f8d45e5e1e7d16d7a2b617648615eef3d4"
ARCHIVE_ROOT = "umi-community-baseline-v0.2/"
MAXIMUM_ARCHIVE_BYTES = 4 * 1024**3
MAXIMUM_CONTENT_BYTES = 4 * 1024**3
MAXIMUM_FILES = 512
MAXIMUM_METADATA_BYTES = 256 * 1024
ROOT = Path(__file__).resolve().parents[1]


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        str(path) != value
        or path.is_absolute()
        or not 1 <= len(value) <= 240
        or any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", p) for p in path.parts)
    ):
        raise ValueError("noncanonical archive path")
    return value


def _json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("duplicate JSON key")
            result[name] = value
        return result

    result = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(result, dict):
        raise ValueError("inventory must be an object")
    return result


def _members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not 1 <= len(infos) <= MAXIMUM_FILES:
        raise ValueError("archive file count exceeds limit")
    members = {}
    folded = set()
    total = 0
    for info in infos:
        if not info.filename.startswith(ARCHIVE_ROOT):
            raise ValueError("archive root mismatch")
        name = _path(info.filename[len(ARCHIVE_ROOT) :])
        mode = info.external_attr >> 16
        if (
            info.is_dir()
            or stat.S_IFMT(mode) not in {0, stat.S_IFREG}
            or info.flag_bits & 1
            or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or name.casefold() in folded
            or info.file_size < 0
        ):
            raise ValueError("archive has duplicate, encrypted or non-regular entries")
        folded.add(name.casefold())
        members[name] = info
        total += info.file_size
    if total > MAXIMUM_CONTENT_BYTES:
        raise ValueError("archive content exceeds limit")
    for name in members:
        if any(str(parent).casefold() in folded for parent in PurePosixPath(name).parents):
            raise ValueError("archive file is also a directory")
    return members


def _inventory(archive: zipfile.ZipFile, members: dict) -> dict:
    for name in ("FILE_LIST.json", "SHA256SUMS"):
        if name not in members or members[name].file_size > MAXIMUM_METADATA_BYTES:
            raise ValueError("missing or oversized integrity metadata")
    records = _json(archive.read(members["FILE_LIST.json"])).get("files")
    if not isinstance(records, list) or len(records) > MAXIMUM_FILES:
        raise ValueError("invalid inventory")
    declared = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
            raise ValueError("invalid inventory record")
        name = _path(record["path"])
        if (
            name in declared
            or type(record["bytes"]) is not int
            or record["bytes"] < 0
            or not isinstance(record["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        ):
            raise ValueError("invalid inventory record")
        declared[name] = record
    if set(declared) | {"FILE_LIST.json", "SHA256SUMS"} != set(members):
        raise ValueError("archive inventory mismatch")
    sums = {}
    for line in archive.read(members["SHA256SUMS"]).decode("ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match[2] in sums:
            raise ValueError("invalid checksum inventory")
        sums[_path(match[2])] = match[1]
    if set(sums) != set(members) - {"SHA256SUMS"}:
        raise ValueError("checksum inventory mismatch")
    for name, record in declared.items():
        if record["bytes"] != members[name].file_size or record["sha256"] != sums[name]:
            raise ValueError("integrity inventories disagree")
    return sums


def _role(name: str) -> str:
    if name == "umi_inference.py":
        return "inference"
    if name.endswith((".safetensors", ".task")):
        return "weights"
    if "LICENSE" in name or name.startswith("licenses/"):
        return "license"
    if "byt5_base/" in name:
        return "processor"
    if name.endswith("config.json") or name == "MODEL_IDENTITY.json":
        return "config"
    if name in {"requirements.txt", "runtime.Dockerfile"}:
        return "environment"
    if name.endswith((".py", ".sh")):
        # offline_bundle/1 selects exactly one Python inference entrypoint.
        return "dependency"
    return "provenance"


def stage_baseline(source: Path, destination: Path) -> dict:
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise ValueError("destination must be a new absolute path")
    parent = destination.parent
    metadata = parent.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise ValueError("destination parent must be an owner-held directory")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("destination parent must be private (0700)")
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAXIMUM_ARCHIVE_BYTES:
            raise ValueError("source must be a bounded regular ZIP")
        hasher = hashlib.sha256()
        hashed_bytes = 0
        while chunk := stream.read(min(1024 * 1024, before.st_size + 1 - hashed_bytes)):
            hashed_bytes += len(chunk)
            if hashed_bytes > before.st_size:
                raise ValueError("source archive grew during intake")
            hasher.update(chunk)
        if hashed_bytes != before.st_size or hasher.hexdigest() != ARCHIVE_SHA256:
            raise ValueError("source archive SHA-256 mismatch")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            members = _members(archive)
            sums = _inventory(archive, members)
            stage = Path(tempfile.mkdtemp(prefix=".community-pending-", dir=parent))
            try:
                model = stage / "model"
                model.mkdir(mode=0o700)
                records = []
                for name, info in sorted(members.items()):
                    target = model / name
                    # pathlib's parents=True applies mode only to the leaf.
                    # Create each level explicitly so a permissive process umask
                    # cannot leave group-writable intermediate directories.
                    parent_directory = model
                    for part in PurePosixPath(name).parts[:-1]:
                        parent_directory = parent_directory / part
                        parent_directory.mkdir(exist_ok=True, mode=0o700)
                    checksum = hashlib.sha256()
                    total = 0
                    with archive.open(info) as incoming, target.open("xb") as outgoing:
                        while data := incoming.read(min(1024 * 1024, info.file_size + 1 - total)):
                            total += len(data)
                            if total > info.file_size:
                                raise ValueError("archive entry exceeds declared size")
                            checksum.update(data)
                            outgoing.write(data)
                        outgoing.flush()
                        os.fsync(outgoing.fileno())
                    if total != info.file_size or (
                        name in sums and checksum.hexdigest() != sums[name]
                    ):
                        raise ValueError("archive entry checksum mismatch")
                    target.chmod(0o400)
                    records.append(
                        {
                            "path": name,
                            "role": _role(name),
                            "sha256": checksum.hexdigest(),
                            "size_bytes": total,
                        }
                    )
                for original, name in (
                    ("umi_inference.py", "umi_inference.py"),
                    ("Dockerfile", "runtime.Dockerfile"),
                ):
                    if name in members:
                        raise ValueError("archive shadows the UMI adapter")
                    content = (ROOT / "community" / original).read_bytes()
                    (model / name).write_bytes(content)
                    (model / name).chmod(0o400)
                    records.append(
                        {
                            "path": name,
                            "role": _role(name),
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "size_bytes": len(content),
                        }
                    )
                manifest = canonical(
                    {
                        "schema": "umi-model-bundle/1",
                        "profile": "offline_bundle/1",
                        "parent_baseline_sha256": None,
                        "license_id": "MIT",
                        "files": sorted(records, key=lambda record: record["path"]),
                    }
                )
                # Match umi.open_competition.digest, including its domain.
                revision = hashlib.sha256(b"umi-open-competition-v1\0" + manifest).hexdigest()
                report = {
                    "schema": "umi-community-baseline-intake/1",
                    "source_archive_sha256": ARCHIVE_SHA256,
                    "source_archive_bytes": before.st_size,
                    "model_bundle_sha256": revision,
                    "manifest_file_sha256": hashlib.sha256(manifest).hexdigest(),
                    "entrypoint": "umi_inference.py",
                    "device": "cpu",
                    "files": len(records),
                    "bytes": sum(r["size_bytes"] for r in records),
                    "inference_verified": False,
                    "rights_reviewed": False,
                    "contributor_attribution": None,
                    "chain_submission_authorized": False,
                }
                (stage / "manifest.json").write_bytes(manifest)
                (stage / "intake.json").write_bytes(canonical(report))
                (stage / "manifest.json").chmod(0o400)
                (stage / "intake.json").chmod(0o400)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError("source archive changed during intake")
                stage.rename(destination)
                return report
            finally:
                if stage.exists():
                    # Only the private directory allocated by this invocation.
                    shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(stage_baseline(args.archive, args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
