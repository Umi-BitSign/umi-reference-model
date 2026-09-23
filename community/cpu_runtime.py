"""Seal and verify a local Linux CPU worker environment without importing a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from bundle_verification import verify_imported_bundle
from native_runtime import MAXIMUM_MANIFEST_BYTES, canonical, digest, file_hash, inventory

DOMAIN = b"umi-community-cpu-runtime-v1\0"
EXECUTION = {
    "device": "cpu",
    "generation_num_beams": 5,
    "generation_max_length": 2048,
    "dino_batch_size": 128,
    "model_execution_concurrency": 1,
    "cpu_threads": 4,
}


def stage_manifest(
    roots: dict[str, Path], *, bundle: Path, bundle_sha256: str, output: Path
) -> str:
    if any(output.resolve().is_relative_to(root.resolve()) for root in [*roots.values(), bundle]):
        raise ValueError("runtime manifest must be outside inventoried roots and bundle")
    verify_imported_bundle(bundle, expected_bundle_sha256=bundle_sha256)
    document = {
        "schema": "umi-community-cpu-runtime/1",
        "platform": "linux/x86_64",
        "python_abi": "cp310",
        "model_bundle_sha256": digest(bundle_sha256),
        "execution": EXECUTION,
        "entries": inventory(roots),
    }
    raw = canonical(document)
    if len(raw) > MAXIMUM_MANIFEST_BYTES:
        raise ValueError("runtime manifest exceeds size limit")
    with output.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o400)
        stream.write(raw)
    return hashlib.sha256(DOMAIN + raw).hexdigest()


def verify_runtime(roots: dict[str, Path], *, manifest: Path, expected_revision: str) -> dict:
    digest(expected_revision)
    checksum, size = file_hash(manifest, maximum_bytes=MAXIMUM_MANIFEST_BYTES)
    raw = manifest.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("runtime manifest changed while reading")
    if hashlib.sha256(DOMAIN + raw).hexdigest() != expected_revision:
        raise ValueError("runtime manifest differs from reviewed release")
    document = json.loads(raw)
    if (
        not isinstance(document, dict)
        or set(document)
        != {"schema", "platform", "python_abi", "model_bundle_sha256", "execution", "entries"}
        or document["schema"] != "umi-community-cpu-runtime/1"
        or document["platform"] != "linux/x86_64"
        or document["python_abi"] != "cp310"
        or document["execution"] != EXECUTION
        or canonical(document) != raw
    ):
        raise ValueError("unsupported CPU runtime manifest")
    digest(document["model_bundle_sha256"])
    if inventory(roots) != document["entries"]:
        raise ValueError("installed runtime differs from reviewed release")
    if manifest.read_bytes() != raw:
        raise ValueError("runtime manifest changed during verification")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code", "environment", "python", "bundle", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    args = parser.parse_args()
    revision = stage_manifest(
        {name: getattr(args, name) for name in ("code", "environment", "python")},
        bundle=args.bundle,
        bundle_sha256=args.expected_bundle_sha256,
        output=args.output,
    )
    print(json.dumps({"model_revision": revision}))


if __name__ == "__main__":
    main()
