from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

SCHEMA = "umi-reference-model-release/1"
STATUS = "component_test_no_weight"
CONTENT_DOMAIN = b"umi-reference-model-release-v1\0"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_REVISION = re.compile(r"[0-9a-f]{40}")
RELEASE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
LABEL = re.compile(r"[a-z][a-z0-9-]{0,63}")
MAXIMUM_ARTIFACT_BYTES = 128 * 1024 * 1024
MODEL_FILENAME = "umi-s1-baseline-v0-portable.zip"
TASK_MODEL_SHA256 = "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8"
TASK_MODEL_SOURCE = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task?"
    "generation=1703178474695092"
)
CLAIM_BOUNDARY = (
    "This is a component-test reference miner release. UMI translation weights are "
    "inactive, and the release is not activation evidence or a reward guarantee. The "
    "extractor image is built locally and is not a release artifact."
)
EXPECTED_ARTIFACTS = {"model": MODEL_FILENAME}
LOCAL_EXTRACTOR = {
    "binary_distributed": False,
    "build_context": "docker/mediapipe-holistic",
    "build_platform": "linux/amd64",
    "derived_bundle_required": True,
    "workflow": "source-build-validate-rebind-probe/1",
}
RELEASE_COMMIT_POLICY = {
    "source_revision_relation": "direct_parent",
    "allowed_changed_paths": ["release/SHA256SUMS", "release/release-manifest.json"],
}


class ReleaseArtifactError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _content_sha256(record: dict[str, Any]) -> str:
    unsigned = dict(record)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(CONTENT_DOMAIN + _canonical(unsigned)).hexdigest()


def _hash_regular(path: Path) -> tuple[str, int]:
    before = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 1 <= before.st_size <= MAXIMUM_ARTIFACT_BYTES
    ):
        raise ReleaseArtifactError(f"release artifact violates its file contract: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ReleaseArtifactError(f"release artifact changed while hashing: {path.name}")
    return digest.hexdigest(), before.st_size


def _parse_artifacts(values: list[str], output_directory: Path) -> list[dict[str, object]]:
    parsed: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or LABEL.fullmatch(label) is None or label in parsed:
            raise ReleaseArtifactError(f"invalid or duplicate artifact argument: {value}")
        parsed[label] = Path(raw_path).resolve(strict=True)
    if set(parsed) != set(EXPECTED_ARTIFACTS):
        raise ReleaseArtifactError("release requires exactly one portable model artifact")

    output = output_directory.resolve(strict=True)
    records: list[dict[str, object]] = []
    for label, expected_name in EXPECTED_ARTIFACTS.items():
        path = parsed[label]
        if path.parent != output or path.name != expected_name:
            raise ReleaseArtifactError(f"{label} must use the fixed name in the output directory")
        digest, size = _hash_regular(path)
        records.append(
            {"label": label, "filename": path.name, "size_bytes": size, "sha256": digest}
        )
    return sorted(records, key=lambda item: str(item["filename"]))


def _validate_model_archive(path: Path, expected_revision: str) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = tuple(sorted(archive.namelist()))
            expected = (
                "bundle-manifest.json",
                "inference-identity.json",
                "model-config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer.model",
            )
            if names != expected:
                raise ReleaseArtifactError("model archive has an unexpected file set")
            identity_info = archive.getinfo("inference-identity.json")
            if (
                identity_info.compress_type != zipfile.ZIP_STORED
                or not 1 <= identity_info.file_size <= 2 * 1024 * 1024
            ):
                raise ReleaseArtifactError("archived inference identity violates its contract")
            identity_payload = archive.read("inference-identity.json")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ReleaseArtifactError("model archive cannot be inspected") from exc
    try:
        identity = json.loads(identity_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseArtifactError("archived inference identity is invalid") from exc
    if not isinstance(identity, dict) or identity.get("inference_revision") != expected_revision:
        raise ReleaseArtifactError("model archive inference revision differs")


def _write_atomic(path: Path, payload: bytes, *, replace: bool) -> None:
    if (path.exists() or path.is_symlink()) and not replace:
        raise ReleaseArtifactError(f"release metadata already exists: {path.name}")
    temporary = path.with_name(f".{path.name}.release-{os.getpid()}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o644,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise ReleaseArtifactError("metadata write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def create_release(arguments: argparse.Namespace) -> dict[str, Any]:
    for value, pattern, label in (
        (arguments.release_id, RELEASE_ID, "release ID"),
        (arguments.inference_revision, SHA256, "inference revision"),
        (arguments.rights_decision_sha256, SHA256, "rights decision digest"),
        (arguments.source_git_revision, GIT_REVISION, "source Git revision"),
        (arguments.umi_git_revision, GIT_REVISION, "UMI Git revision"),
    ):
        if pattern.fullmatch(value) is None:
            raise ReleaseArtifactError(f"{label} is invalid")
    output = arguments.output_directory.resolve(strict=True)
    artifacts = _parse_artifacts(arguments.artifact, output)
    model_path = output / MODEL_FILENAME
    _validate_model_archive(model_path, arguments.inference_revision)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "release_id": arguments.release_id,
        "status": STATUS,
        "inference_revision": arguments.inference_revision,
        "rights_decision_sha256": arguments.rights_decision_sha256,
        "source_git_revision": arguments.source_git_revision,
        "umi_git_revision": arguments.umi_git_revision,
        "release_commit_policy": RELEASE_COMMIT_POLICY,
        "local_extractor": LOCAL_EXTRACTOR,
        "external_dependencies": {
            "mediapipe_holistic_task_model": {
                "distributed": False,
                "sha256": TASK_MODEL_SHA256,
                "source": TASK_MODEL_SOURCE,
            }
        },
        "licenses": {"code": "Apache-2.0", "model": "CC-BY-SA-4.0"},
        "artifacts": artifacts,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    record["content_sha256"] = _content_sha256(record)
    manifest_payload = _canonical(record) + b"\n"
    checksums = "".join(
        f"{item['sha256']}  {item['filename']}\n" for item in artifacts
    ).encode()
    _write_atomic(output / "release-manifest.json", manifest_payload, replace=arguments.replace)
    _write_atomic(output / "SHA256SUMS", checksums, replace=arguments.replace)
    return record


def verify_release(path: Path, *, artifact_directory: Path | None = None) -> dict[str, Any]:
    manifest_path = path.resolve(strict=True)
    raw = manifest_path.read_bytes()
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseArtifactError("release manifest is invalid JSON") from exc
    if not isinstance(record, dict) or raw != _canonical(record) + b"\n":
        raise ReleaseArtifactError("release manifest is not canonical")
    expected_fields = {
        "schema",
        "release_id",
        "status",
        "inference_revision",
        "rights_decision_sha256",
        "source_git_revision",
        "umi_git_revision",
        "release_commit_policy",
        "local_extractor",
        "external_dependencies",
        "licenses",
        "artifacts",
        "claim_boundary",
        "content_sha256",
    }
    if set(record) != expected_fields:
        raise ReleaseArtifactError("release manifest has an unexpected field set")
    if (
        record["schema"] != SCHEMA
        or record["status"] != STATUS
        or RELEASE_ID.fullmatch(str(record["release_id"])) is None
        or record["local_extractor"] != LOCAL_EXTRACTOR
        or record["release_commit_policy"] != RELEASE_COMMIT_POLICY
        or record["licenses"] != {"code": "Apache-2.0", "model": "CC-BY-SA-4.0"}
        or record["external_dependencies"]
        != {
            "mediapipe_holistic_task_model": {
                "distributed": False,
                "sha256": TASK_MODEL_SHA256,
                "source": TASK_MODEL_SOURCE,
            }
        }
        or record["claim_boundary"] != CLAIM_BOUNDARY
        or record["content_sha256"] != _content_sha256(record)
        or SHA256.fullmatch(str(record["inference_revision"])) is None
        or SHA256.fullmatch(str(record["rights_decision_sha256"])) is None
        or GIT_REVISION.fullmatch(str(record["source_git_revision"])) is None
        or GIT_REVISION.fullmatch(str(record["umi_git_revision"])) is None
    ):
        raise ReleaseArtifactError("release identity or content digest differs")
    artifacts = record["artifacts"]
    artifact_root = (
        artifact_directory.resolve(strict=True)
        if artifact_directory is not None
        else manifest_path.parent
    )
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ReleaseArtifactError("release artifact set is invalid")
    if artifacts != sorted(artifacts, key=lambda item: str(item.get("filename", ""))):
        raise ReleaseArtifactError("release artifact records are not sorted by filename")
    expected_checksum_lines: list[str] = []
    labels: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"label", "filename", "size_bytes", "sha256"}:
            raise ReleaseArtifactError("release artifact record is invalid")
        label = item["label"]
        filename = item["filename"]
        if (
            not isinstance(label, str)
            or label in labels
            or not isinstance(filename, str)
            or EXPECTED_ARTIFACTS.get(label) != filename
            or Path(filename).name != filename
        ):
            raise ReleaseArtifactError("release artifact name or label differs")
        labels.add(label)
        digest, size = _hash_regular(artifact_root / filename)
        if digest != item["sha256"] or size != item["size_bytes"]:
            raise ReleaseArtifactError(f"release artifact digest differs: {filename}")
        expected_checksum_lines.append(f"{digest}  {filename}\n")
    if set(labels) != set(EXPECTED_ARTIFACTS):
        raise ReleaseArtifactError("release artifact labels are incomplete")
    checksum_payload = (manifest_path.parent / "SHA256SUMS").read_text(encoding="utf-8")
    if checksum_payload != "".join(expected_checksum_lines):
        raise ReleaseArtifactError("SHA256SUMS differs from the manifest")
    _validate_model_archive(
        artifact_root / MODEL_FILENAME, str(record["inference_revision"])
    )
    return record


def _git(repository: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repository), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseArtifactError("release Git history cannot be verified") from exc
    return completed.stdout


def verify_release_commit(
    record: dict[str, Any],
    manifest_path: Path,
    repository: Path,
    release_revision: str = "HEAD",
) -> str:
    root = repository.resolve(strict=True)
    manifest = manifest_path.resolve(strict=True)
    try:
        relative_manifest = manifest.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReleaseArtifactError("release manifest is outside the repository") from exc
    if relative_manifest != "release/release-manifest.json":
        raise ReleaseArtifactError("release manifest is not at its fixed repository path")

    release_commit = _git(
        root, "rev-parse", "--verify", f"{release_revision}^{{commit}}"
    ).decode("ascii").strip()
    if GIT_REVISION.fullmatch(release_commit) is None:
        raise ReleaseArtifactError("release Git revision is invalid")
    parent_line = _git(root, "rev-list", "--parents", "-n", "1", release_commit)
    parents = parent_line.decode("ascii").strip().split()
    if len(parents) != 2 or parents[1] != record["source_git_revision"]:
        raise ReleaseArtifactError(
            "source Git revision is not the release commit's sole direct parent"
        )
    changed = {
        value
        for value in _git(
            root,
            "diff",
            "--name-only",
            "--no-renames",
            parents[1],
            release_commit,
        )
        .decode("utf-8")
        .splitlines()
        if value
    }
    expected = set(RELEASE_COMMIT_POLICY["allowed_changed_paths"])
    if changed != expected:
        raise ReleaseArtifactError("release commit is not metadata-only")
    for relative_path, local_path in (
        ("release/release-manifest.json", manifest),
        ("release/SHA256SUMS", manifest.parent / "SHA256SUMS"),
    ):
        committed = _git(root, "show", f"{release_commit}:{relative_path}")
        if committed != local_path.read_bytes():
            raise ReleaseArtifactError(f"working {relative_path} differs from the release commit")
    return release_commit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify reference-model release metadata"
    )
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--release-git-revision")
    parser.add_argument("--release-id")
    parser.add_argument("--inference-revision")
    parser.add_argument("--rights-decision-sha256")
    parser.add_argument("--source-git-revision")
    parser.add_argument("--umi-git-revision")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    try:
        if arguments.verify is not None:
            supplied_generation = any(
                value is not None
                for value in (
                    arguments.release_id,
                    arguments.inference_revision,
                    arguments.rights_decision_sha256,
                    arguments.source_git_revision,
                    arguments.umi_git_revision,
                    arguments.output_directory,
                )
            ) or bool(arguments.artifact) or arguments.replace
            if supplied_generation:
                parser.error("--verify cannot be combined with generation options")
            result = verify_release(
                arguments.verify, artifact_directory=arguments.artifact_directory
            )
            if arguments.release_git_revision is not None and arguments.repository is None:
                parser.error("--release-git-revision requires --repository")
            if arguments.repository is not None:
                verify_release_commit(
                    result,
                    arguments.verify,
                    arguments.repository,
                    arguments.release_git_revision or "HEAD",
                )
        else:
            if any(
                value is not None
                for value in (
                    arguments.artifact_directory,
                    arguments.repository,
                    arguments.release_git_revision,
                )
            ):
                parser.error("repository verification options require --verify")
            required = (
                arguments.release_id,
                arguments.inference_revision,
                arguments.rights_decision_sha256,
                arguments.source_git_revision,
                arguments.umi_git_revision,
                arguments.output_directory,
            )
            if any(value is None for value in required):
                parser.error("generation requires every release identity option")
            result = create_release(arguments)
    except (OSError, ReleaseArtifactError) as exc:
        print(f"release metadata failed: {exc}", file=sys.stderr)
        return 2
    print(_canonical(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
