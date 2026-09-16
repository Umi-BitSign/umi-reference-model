"""Rebuild the exact preserved MPS baseline without loading a model or wallet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

import import_community_baseline as intake

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_SHA256 = "ce459641c180c680aae32009052985165fa8bbffd165b0dca05d2dade0e619ed"
OVERLAYS = {
    "runtime.py": "candidates/community-baseline-v0.2/runtime.py",
    "umi_inference.py": "community/native/umi_inference.py",
    "umi_landmarks.py": "community/native/umi_landmarks.py",
    "umi_video_reader.py": "community/native/umi_video_reader.py",
    "LOCAL_RUNTIME_DERIVATION.json": "community/native/LOCAL_RUNTIME_DERIVATION.json",
}


def expected_manifest() -> dict:
    manifest = intake._json((ROOT / "community/native/manifest.json").read_bytes())
    checksum = hashlib.sha256(b"umi-open-competition-v1\0" + intake.canonical(manifest)).hexdigest()
    if checksum != BUNDLE_SHA256:
        raise ValueError("native baseline template identity changed")
    return manifest


def overlay_bytes(name: str) -> bytes:
    body = (ROOT / OVERLAYS[name]).read_bytes()
    if name.endswith(".json"):
        body = intake.canonical(intake._json(body))
    return body


def stage_native_baseline(source: Path, destination: Path) -> dict:
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise ValueError("destination must be a new absolute path")
    parent = destination.parent
    info = parent.lstat()
    if (
        parent.resolve(strict=True) != parent
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("destination parent must be an owned private directory without aliases")
    manifest = expected_manifest()
    expected = {record["path"]: record for record in manifest["files"]}
    overlays = {name: overlay_bytes(name) for name in OVERLAYS}
    for name, body in overlays.items():
        entry = expected[name]
        if len(body) != entry["size_bytes"] or hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ValueError("native overlay differs from the preserved baseline")
    temporary = Path(tempfile.mkdtemp(prefix=".native-baseline-pending-", dir=parent))
    try:
        staged = temporary / "bundle"
        report = intake.stage_baseline(source, staged)
        model = staged / "model"
        for name, body in overlays.items():
            path = model / name
            if path.exists():
                path.chmod(0o600)
            with path.open("wb") as stream:
                os.fchmod(stream.fileno(), 0o400)
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        actual = {p.relative_to(model).as_posix() for p in model.rglob("*") if p.is_file()}
        if actual != set(expected):
            raise ValueError("native baseline file inventory differs")
        for name, entry in expected.items():
            path = model / name
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size != entry["size_bytes"]
            ):
                raise ValueError("native baseline file identity differs")
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            if checksum.hexdigest() != entry["sha256"]:
                raise ValueError("native baseline file checksum differs")
        body = intake.canonical(manifest)
        report.update(
            model_bundle_sha256=BUNDLE_SHA256,
            manifest_file_sha256=hashlib.sha256(body).hexdigest(),
            device="mps",
            files=len(expected),
            bytes=sum(record["size_bytes"] for record in expected.values()),
            tensor_bytes_changed=False,
            native_runtime_installed=False,
        )
        for name, data in (("manifest.json", body), ("intake.json", intake.canonical(report))):
            path = staged / name
            path.chmod(0o600)
            with path.open("wb") as stream:
                os.fchmod(stream.fileno(), 0o400)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        if destination.exists() or destination.is_symlink():
            raise ValueError("destination appeared during staging")
        staged.rename(destination)
        return report
    finally:
        # Only our newly allocated temporary directory, never an existing bundle.
        shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_native_baseline(args.archive, args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
