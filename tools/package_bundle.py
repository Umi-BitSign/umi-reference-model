from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import zipfile
from pathlib import Path

EXPECTED_FILES = (
    "bundle-manifest.json",
    "inference-identity.json",
    "model-config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer.model",
)
MAXIMUM_FILE_BYTES = 64 * 1024 * 1024
MAXIMUM_TOTAL_BYTES = 80 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}")
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class BundlePackagingError(RuntimeError):
    pass


def _read_bundle(root: Path) -> tuple[dict[str, bytes], str]:
    direct = Path(os.path.abspath(root))
    if direct.is_symlink():
        raise BundlePackagingError("bundle root must not be a symlink")
    bundle = direct.resolve(strict=True)
    if not bundle.is_dir():
        raise BundlePackagingError("bundle root must be a direct directory")
    if tuple(sorted(item.name for item in bundle.iterdir())) != EXPECTED_FILES:
        raise BundlePackagingError("portable bundle has an unexpected file set")
    payloads: dict[str, bytes] = {}
    total = 0
    for name in EXPECTED_FILES:
        path = bundle / name
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not 1 <= metadata.st_size <= MAXIMUM_FILE_BYTES
        ):
            raise BundlePackagingError(f"bundle file violates its contract: {name}")
        payload = path.read_bytes()
        if len(payload) != metadata.st_size:
            raise BundlePackagingError(f"bundle file changed while being read: {name}")
        payloads[name] = payload
        total += len(payload)
    if total > MAXIMUM_TOTAL_BYTES:
        raise BundlePackagingError("portable bundle exceeds its release ceiling")
    try:
        identity = json.loads(payloads["inference-identity.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundlePackagingError("inference identity is not JSON") from exc
    revision = identity.get("inference_revision") if isinstance(identity, dict) else None
    if not isinstance(revision, str) or SHA256.fullmatch(revision) is None:
        raise BundlePackagingError("inference identity has no valid revision")
    return payloads, revision


def package_bundle(root: Path, output: Path) -> dict[str, object]:
    payloads, revision = _read_bundle(root)
    destination = output.absolute()
    destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if destination.parent.resolve(strict=True) != destination.parent:
        raise BundlePackagingError("output parent must not contain a symlink")
    if destination.exists() or destination.is_symlink():
        raise BundlePackagingError("output archive already exists")
    temporary = destination.with_name(f".{destination.name}.package-{os.getpid()}")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED) as archive:
            for name in EXPECTED_FILES:
                info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payloads[name])
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    payload = destination.read_bytes()
    return {
        "schema": "umi-reference-model-package-result/1",
        "inference_revision": revision,
        "archive": destination.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic portable bundle ZIP")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = package_bundle(arguments.bundle, arguments.output)
    except (OSError, BundlePackagingError) as exc:
        print(f"bundle packaging failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
