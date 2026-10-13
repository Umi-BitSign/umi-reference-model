"""Verify the user-designated C5 working reference without loading models or changing serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "C5_BASELINE_REFERENCE.json"
MAX_FILES = 512
MAX_BYTES = 4 * 1024**3


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate C5 manifest key")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=pairs)


def checked_name(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 240:
        raise ValueError("invalid C5 asset path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or not path.parts
        or any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", part) for part in path.parts)
    ):
        raise ValueError("invalid C5 asset path")
    return value


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def file_identity(info):
    # Reading can update access time; that does not change the verified file.
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def verify_tree(root: Path, expected_model_id: str) -> dict:
    if root.is_symlink() or root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("C5 reference must be a canonical direct directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 256 * 1024:
        raise ValueError("invalid C5 reference manifest")
    raw = manifest_path.read_bytes()
    document = strict_json(raw)
    if (
        document.get("schema") != "umi-model-bundle/1"
        or document.get("profile") != "offline_bundle/1"
        or raw != canonical(document)
    ):
        raise ValueError("unsupported or noncanonical C5 manifest")
    identity = hashlib.sha256(b"umi-open-competition-v1\0" + raw).hexdigest()
    if identity != expected_model_id:
        raise ValueError("C5 reference model identity differs")
    records = document["files"]
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_FILES:
        raise ValueError("invalid C5 asset inventory")
    names, total = [], 0
    for row in records:
        names.append(checked_name(row["path"]))
        if (
            type(row["size_bytes"]) is not int
            or row["size_bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("invalid C5 asset digest or size")
        total += row["size_bytes"]
    if len(set(name.casefold() for name in names)) != len(names) or total > MAX_BYTES:
        raise ValueError("C5 reference inventory is aliased or oversized")
    model = root / "model"
    if model.is_symlink() or not model.is_dir():
        raise ValueError("invalid C5 model tree")
    actual = set()
    for path in model.rglob("*"):
        if path.is_symlink():
            raise ValueError("C5 reference contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(model).as_posix())
        elif not path.is_dir():
            raise ValueError("C5 reference contains a special file")
    if actual != set(names):
        raise ValueError("C5 reference exact file inventory differs")
    for row in records:
        path = model / row["path"]
        before = path.stat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
            or before.st_size != row["size_bytes"]
            or sha256(path) != row["sha256"]
        ):
            raise ValueError("C5 reference asset checksum, size or permissions differ")
        if file_identity(before) != file_identity(path.stat()):
            raise ValueError("C5 reference changed during verification")
    if manifest_path.read_bytes() != raw:
        raise ValueError("C5 manifest changed during verification")
    return document


def load_reference() -> dict:
    reference = strict_json(REGISTRY.read_bytes())
    if (
        reference["schema"] != "umi-studio-working-baseline-reference/1"
        or reference["status"] != "active_training_reference"
    ):
        raise ValueError("C5 winner is not the designated training reference")
    root = (ROOT / reference["bundle_root"]).absolute()
    if not root.is_relative_to(ROOT / "artifacts"):
        raise ValueError("C5 working reference leaves the private artifacts tree")
    manifest = verify_tree(root, reference["model_bundle_sha256"])
    records = {row["path"]: row for row in manifest["files"]}
    for path_key, hash_key in (
        ("translator_checkpoint", "translator_checkpoint_sha256"),
        ("fingerspelling_checkpoint", "fingerspelling_checkpoint_sha256"),
    ):
        if records[reference[path_key]]["sha256"] != reference[hash_key]:
            raise ValueError("C5 component pin differs from the model manifest")
    for name, key in (
        ("umi_fs_router.py", "router_sha256"),
        ("fs_router.json", "router_config_sha256"),
    ):
        if records[name]["sha256"] != reference[key]:
            raise ValueError("C5 routing pin differs from the model manifest")
    if (
        sha256(root / "manifest.json") != reference["manifest_file_sha256"]
        or manifest["license_id"] != reference["declared_bundle_license"]
    ):
        raise ValueError("C5 reference metadata binding differs")
    router = strict_json((root / "model/fs_router.json").read_bytes())
    if reference["generation"] != {
        "num_beams": 5,
        "max_length": 2048,
        **router["baseline_generate_overrides"],
    }:
        raise ValueError("C5 reference decoding differs from the released composite")
    return reference


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    reference = load_reference()
    print(
        json.dumps(
            {
                "model_id": reference["model_bundle_sha256"],
                "all_files_verified": True,
                "translator_sha256": reference["translator_checkpoint_sha256"],
                "miner_modified": False,
                "quality_verified": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
