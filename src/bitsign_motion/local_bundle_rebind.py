from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Final, cast

from . import s1_portable_runtime as portable
from .canonical import canonical_json_bytes, canonical_json_sha256
from .local_extractor_release import (
    LocalExtractorReleaseError,
    validate_local_extractor_record,
)
from .s1_portable_runtime import S1PortableError, load_s1_portable_bundle

BASE_INFERENCE_REVISION: Final = "2ead0d2d870c082ae7796f7055cdc360fc3ded7f156a98b247d5d554c79cd752"
DERIVED_STATUS: Final = "component_test_no_weight"

_EXPECTED_FILES: Final = (
    "bundle-manifest.json",
    "inference-identity.json",
    "model-config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer.model",
)
_MAXIMUM_ARCHIVE_BYTES: Final = 80 * 1024 * 1024
_MAXIMUM_MEMBER_BYTES: Final = 64 * 1024 * 1024
_MAXIMUM_TOTAL_BYTES: Final = 80 * 1024 * 1024
_MAXIMUM_RECORD_BYTES: Final = 1024 * 1024
_FIXED_ZIP_TIME: Final = (1980, 1, 1, 0, 0, 0)


class LocalBundleRebindError(RuntimeError):
    """Raised when a published bundle cannot be rebound without widening authority."""


def _read_regular(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    direct = Path(os.path.abspath(path))
    try:
        before = direct.lstat()
        payload = direct.read_bytes()
        after = direct.lstat()
    except OSError as exc:
        raise LocalBundleRebindError(f"{label} is unavailable") from exc
    if (
        direct.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 1 <= len(payload) <= maximum_bytes
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise LocalBundleRebindError(f"{label} violates its file contract")
    return payload


def _strict_record(path: Path) -> dict[str, Any]:
    payload = _read_regular(
        path,
        maximum_bytes=_MAXIMUM_RECORD_BYTES,
        label="local extractor build record",
    )
    try:
        value = json.loads(
            payload,
            object_pairs_hook=lambda pairs: _reject_duplicate_pairs(pairs),
            parse_constant=lambda token: (_ for _ in ()).throw(
                LocalBundleRebindError(f"build record contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalBundleRebindError("local extractor build record is invalid JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise LocalBundleRebindError("local extractor build record is not canonical JSON")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise LocalBundleRebindError("build record contains a duplicate member")
        value[key] = item
    return value


def _extract_base_archive(
    archive_path: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    payload = _read_regular(
        archive_path,
        maximum_bytes=_MAXIMUM_ARCHIVE_BYTES,
        label="published base bundle archive",
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise LocalBundleRebindError("published base bundle archive digest differs")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            if tuple(info.filename for info in infos) != _EXPECTED_FILES:
                raise LocalBundleRebindError("published base bundle file set or order differs")
            total = 0
            for info in infos:
                mode = info.external_attr >> 16
                if (
                    info.compress_type != zipfile.ZIP_STORED
                    or info.date_time != _FIXED_ZIP_TIME
                    or info.flag_bits & 0x1
                    or stat.S_IFMT(mode) != stat.S_IFREG
                    or stat.S_IMODE(mode) != 0o644
                    or not 1 <= info.file_size <= _MAXIMUM_MEMBER_BYTES
                    or info.compress_size != info.file_size
                ):
                    raise LocalBundleRebindError(
                        f"published base bundle member differs: {info.filename}"
                    )
                total += info.file_size
                if total > _MAXIMUM_TOTAL_BYTES:
                    raise LocalBundleRebindError("published base bundle exceeds its ceiling")
                member = archive.read(info)
                if len(member) != info.file_size:
                    raise LocalBundleRebindError(
                        f"published base bundle member changed: {info.filename}"
                    )
                target = destination / info.filename
                descriptor = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                )
                try:
                    offset = 0
                    while offset < len(member):
                        written = os.write(descriptor, member[offset:])
                        if written <= 0:
                            raise LocalBundleRebindError("base extraction made no progress")
                        offset += written
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise LocalBundleRebindError("published base bundle cannot be extracted safely") from exc


def _assert_preserved_identity(
    base: dict[str, Any],
    derived: dict[str, Any],
    *,
    image_id: str,
) -> None:
    expected = copy.deepcopy(base)
    expected["preprocessing"]["supported_oci_images"]["linux/amd64"] = image_id
    expected.pop("inference_revision")
    observed = copy.deepcopy(derived)
    observed.pop("inference_revision")
    if observed != expected:
        raise LocalBundleRebindError(
            "derived identity changed a field other than the local AMD64 image ID"
        )


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise LocalBundleRebindError("derived bundle write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reseal_verified_base(
    base_root: Path,
    destination: Path,
    *,
    base_identity: dict[str, Any],
    image_id: str,
) -> str:
    """Copy verified payloads byte-for-byte and reseal only the image authority."""

    identity = copy.deepcopy(base_identity)
    identity["preprocessing"]["supported_oci_images"]["linux/amd64"] = image_id
    identity.pop("inference_revision")
    revision = canonical_json_sha256(identity, domain=portable._IDENTITY_DOMAIN)
    identity["inference_revision"] = revision
    payloads = {
        name: (base_root / name).read_bytes()
        for name in _EXPECTED_FILES
        if name not in {"bundle-manifest.json", "inference-identity.json"}
    }
    payloads["inference-identity.json"] = canonical_json_bytes(identity)
    manifest: dict[str, Any] = {
        "schema": portable.S1_PORTABLE_MANIFEST_SCHEMA,
        "inference_revision": revision,
        "files": [
            {
                "name": name,
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
                "size_bytes": len(payloads[name]),
            }
            for name in sorted(payloads)
        ],
    }
    manifest["content_sha256"] = canonical_json_sha256(
        manifest,
        domain=portable._MANIFEST_DOMAIN,
    )
    payloads["bundle-manifest.json"] = canonical_json_bytes(manifest)
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise LocalBundleRebindError("derived bundle destination appeared during export") from exc
    try:
        for name in _EXPECTED_FILES:
            _write_exclusive(destination / name, payloads[name])
    except BaseException:
        for name in _EXPECTED_FILES:
            try:
                (destination / name).unlink()
            except FileNotFoundError:
                pass
        try:
            destination.rmdir()
        except OSError:
            pass
        raise
    return revision


def _remove_generated_bundle(destination: Path) -> None:
    for name in _EXPECTED_FILES:
        path = destination / name
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            path.unlink()
    try:
        destination.rmdir()
    except FileNotFoundError:
        pass


def rebind_local_extractor(
    *,
    base_archive: Path,
    expected_base_sha256: str,
    expected_base_inference_revision: str = BASE_INFERENCE_REVISION,
    build_record_path: Path,
    docker_executable: Path,
    output: Path,
) -> dict[str, Any]:
    """Re-export the fixed base model with one locally built AMD64 image authority."""

    for value, label in (
        (expected_base_sha256, "base archive digest"),
        (expected_base_inference_revision, "base inference revision"),
    ):
        if (
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LocalBundleRebindError(f"{label} must be lowercase SHA-256")
    destination = Path(os.path.abspath(output))
    if destination.exists() or destination.is_symlink():
        raise LocalBundleRebindError("derived bundle destination must not exist")
    parent = destination.parent.resolve(strict=True)
    if destination.parent != parent:
        raise LocalBundleRebindError("derived bundle parent must be a direct directory")

    build_record = validate_local_extractor_record(
        _strict_record(build_record_path),
        docker_executable=docker_executable,
    )
    image_id = cast(str, build_record["image_id"])

    generated = False
    with tempfile.TemporaryDirectory(prefix=".umi-base-bundle-", dir=parent) as temporary:
        base_root = Path(temporary)
        base_root.chmod(0o700)
        _extract_base_archive(
            base_archive,
            base_root,
            expected_sha256=expected_base_sha256,
        )
        try:
            base_runtime = load_s1_portable_bundle(
                base_root,
                expected_inference_revision=expected_base_inference_revision,
            )
        except S1PortableError as exc:
            raise LocalBundleRebindError("published base bundle failed runtime validation") from exc

        base_identity = copy.deepcopy(base_runtime.identity)
        preprocessing = copy.deepcopy(base_identity["preprocessing"])
        images = preprocessing.get("supported_oci_images")
        sources = preprocessing.get("sources")
        source_record = build_record["sources"]
        if (
            not isinstance(images, dict)
            or set(images) != {"linux/amd64", "linux/arm64"}
            or not isinstance(sources, dict)
            or source_record.get("amd64_container_host_source_sha256")
            != sources.get("amd64_container_host_source_sha256")
            or source_record.get("requirements_sha256")
            != sources.get("amd64_container_requirements_sha256")
            or source_record.get("worker_sha256")
            != sources.get("amd64_container_worker_source_sha256")
        ):
            raise LocalBundleRebindError(
                "local extractor source set differs from the published preprocessing contract"
            )
        if images["linux/amd64"] == image_id:
            raise LocalBundleRebindError(
                "local image already matches the base authority; no derived revision would result"
            )
        images["linux/amd64"] = image_id

        try:
            revision = _reseal_verified_base(
                base_root,
                destination,
                base_identity=base_identity,
                image_id=image_id,
            )
            generated = True
            if revision == expected_base_inference_revision:
                raise LocalBundleRebindError("derived bundle did not produce a new revision")
            derived = load_s1_portable_bundle(
                destination,
                expected_inference_revision=revision,
            )
            _assert_preserved_identity(
                base_identity,
                derived.identity,
                image_id=image_id,
            )
        except (OSError, S1PortableError) as exc:
            if generated:
                _remove_generated_bundle(destination)
            raise LocalBundleRebindError("derived bundle export failed") from exc
        except LocalBundleRebindError:
            if generated:
                _remove_generated_bundle(destination)
            raise
        finally:
            del base_runtime

    if not generated:
        raise LocalBundleRebindError("derived bundle was not generated")
    return {
        "schema": "umi-local-extractor-rebind-result/1",
        "status": DERIVED_STATUS,
        "base_inference_revision": expected_base_inference_revision,
        "local_extractor_image_id": image_id,
        "inference_revision": revision,
        "bundle": str(destination),
        "claim_boundary": (
            "This derived component-test bundle changes only the locally built Linux/AMD64 "
            "extractor image authority. It is not byte-equivalence or UMI activation evidence."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bind a locally built extractor into the published component-test bundle"
    )
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument(
        "--base-inference-revision",
        default=BASE_INFERENCE_REVISION,
        help="expected revision inside the base archive (legacy v0 by default)",
    )
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--docker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = rebind_local_extractor(
            base_archive=arguments.base_archive,
            expected_base_sha256=arguments.base_sha256,
            expected_base_inference_revision=arguments.base_inference_revision,
            build_record_path=arguments.build_record,
            docker_executable=arguments.docker,
            output=arguments.output,
        )
    except (LocalBundleRebindError, LocalExtractorReleaseError, OSError) as exc:
        print(f"local bundle rebind failed: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
