from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

from bitsign_motion import s1_portable_runtime as portable
from bitsign_motion.s1_portable_runtime import S1PortableError, load_s1_portable_bundle

EXPECTED_FILES = portable._EXPECTED_FILES
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
    try:
        payloads = portable._read_bundle_files(bundle)
        identity = portable._strict_json(
            payloads["inference-identity.json"],
            maximum_bytes=portable._MAXIMUM_FILE_BYTES["inference-identity.json"],
            label="packaged inference identity",
        )
    except S1PortableError as exc:
        raise BundlePackagingError("portable bundle failed strict input validation") from exc
    revision = identity.get("inference_revision") if isinstance(identity, dict) else None
    if not isinstance(revision, str) or SHA256.fullmatch(revision) is None:
        raise BundlePackagingError("inference identity has no valid revision")
    try:
        loaded = load_s1_portable_bundle(bundle, expected_inference_revision=revision)
        final_payloads = portable._read_bundle_files(bundle)
    except S1PortableError as exc:
        raise BundlePackagingError("portable bundle failed strict runtime loading") from exc
    if loaded.identity != identity or final_payloads != payloads:
        raise BundlePackagingError("portable bundle changed during strict runtime loading")
    del loaded
    if sum(len(payload) for payload in payloads.values()) > MAXIMUM_TOTAL_BYTES:
        raise BundlePackagingError("portable bundle exceeds its release ceiling")
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
