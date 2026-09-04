from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from bitsign_motion.canonical import canonical_json_bytes
from tests.test_umi_release_e2e import (
    _development_wallet,
    _load_release_dependencies,
    _wire_response_sha256,
)

_INPUT_SCHEMA = "umi-s1-macos-capacity-input/1"
_RUN_SCHEMA = "umi-s1-macos-capacity-run/1"
_RUN_DOMAIN = b"umi-s1-macos-capacity-run-v1\0"
_CLIP_COUNT = 28
_VALIDATOR_COUNT = 4
_REQUEST_COUNT = _CLIP_COUNT * _VALIDATOR_COUNT
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _ExactVideoFetcher:
    videos: dict[str, bytes]

    async def fetch(self, descriptor: Any) -> bytes:
        video = self.videos.get(descriptor.sha256)
        if video is None or len(video) != descriptor.size_bytes:
            raise AssertionError("capacity video differs from its request authority")
        return video


def _required_new_output(name: str) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        pytest.fail(f"required capacity environment variable is missing: {name}")
    path = Path(raw)
    try:
        parent_status = path.parent.lstat()
        parent_resolved = path.parent.resolve(strict=True)
    except OSError:
        pytest.fail(f"{name} parent directory is unavailable")
    if (
        not path.is_absolute()
        or path.exists()
        or parent_resolved != path.parent
        or not stat.S_ISDIR(parent_status.st_mode)
        or parent_status.st_uid != os.geteuid()
        or parent_status.st_mode & 0o077
    ):
        pytest.fail(f"{name} must be a new absolute file in an existing directory")
    return path


def _direct_regular_file(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if (
        resolved != path
        or not stat.S_ISREG(metadata.st_mode)
        or not 1 <= metadata.st_size <= maximum_bytes
    ):
        raise ValueError(f"{label} must be a direct bounded regular file")
    payload = path.read_bytes()
    final = path.stat()
    if len(payload) != metadata.st_size or (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    ) != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
        raise ValueError(f"{label} changed while being read")
    return payload


def _canonical_object(
    path: Path, *, label: str, maximum_bytes: int
) -> tuple[dict[str, Any], bytes]:
    payload = _direct_regular_file(path, label=label, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} must be a canonical JSON object")
    return value, payload


def _load_capacity_input(path: Path) -> tuple[dict[str, Any], list[bytes], bytes]:
    value, payload = _canonical_object(
        path,
        label="capacity input",
        maximum_bytes=64 * 1024,
    )
    if set(value) != {
        "schema",
        "resolved_miner_release",
        "outer_inference_concurrency",
        "expected_actual_workers",
        "rights_cleared_for_private_testing",
        "clips",
    }:
        raise ValueError("capacity input field set differs")
    clips = value["clips"]
    if (
        value["schema"] != _INPUT_SCHEMA
        or value["rights_cleared_for_private_testing"] is not True
        or not isinstance(value["resolved_miner_release"], str)
        or not Path(value["resolved_miner_release"]).is_absolute()
        or type(value["outer_inference_concurrency"]) is not int
        or value["outer_inference_concurrency"] < _VALIDATOR_COUNT
        or value["outer_inference_concurrency"] % _VALIDATOR_COUNT != 0
        or type(value["expected_actual_workers"]) is not int
        or not 1 <= value["expected_actual_workers"] <= _CLIP_COUNT
        or not isinstance(clips, list)
        or len(clips) != _CLIP_COUNT
    ):
        raise ValueError("capacity input values differ")

    videos: list[bytes] = []
    digests: list[str] = []
    for index, item in enumerate(clips):
        if not isinstance(item, dict) or set(item) != {
            "path",
            "sha256",
            "size_bytes",
            "eligibility_attested",
        }:
            raise ValueError(f"capacity clip {index} field set differs")
        path_value = item["path"]
        digest = item["sha256"]
        size = item["size_bytes"]
        if (
            not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or type(size) is not int
            or not 1 <= size <= 16 * 1024 * 1024
            or item["eligibility_attested"] is not True
        ):
            raise ValueError(f"capacity clip {index} values differ")
        video = _direct_regular_file(
            Path(path_value),
            label=f"capacity clip {index}",
            maximum_bytes=16 * 1024 * 1024,
        )
        if len(video) != size or hashlib.sha256(video).hexdigest() != digest:
            raise ValueError(f"capacity clip {index} authority differs")
        videos.append(video)
        digests.append(digest)
    if len(set(digests)) != _CLIP_COUNT:
        raise ValueError("capacity clips must have 28 distinct video digests")
    return value, videos, payload


def _nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("latency sample is empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _response_set_sha256(outcomes: list[Any]) -> str:
    digests = sorted(bytes.fromhex(_wire_response_sha256(outcome)) for outcome in outcomes)
    digest = hashlib.sha256(b"umi-s1-macos-capacity-responses-v1\0")
    digest.update(len(digests).to_bytes(4, "big"))
    for value in digests:
        digest.update(value)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    payload = canonical_json_bytes(report)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:  # pragma: no cover
                raise OSError("capacity report write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise AssertionError("capacity report is not owner-only")


def _capacity_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    clips = []
    for index in range(_CLIP_COUNT):
        path = (tmp_path / f"{index:02d}.mp4").resolve()
        payload = f"fixture-{index}".encode()
        path.write_bytes(payload)
        clips.append(
            {
                "eligibility_attested": True,
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    value = {
        "clips": clips,
        "expected_actual_workers": 16,
        "outer_inference_concurrency": 64,
        "resolved_miner_release": str((tmp_path / "resolved.json").resolve()),
        "rights_cleared_for_private_testing": True,
        "schema": _INPUT_SCHEMA,
    }
    path = (tmp_path / "input.json").resolve()
    path.write_bytes(canonical_json_bytes(value))
    return path, value


def test_capacity_input_requires_28_distinct_bound_clips(tmp_path: Path) -> None:
    path, value = _capacity_fixture(tmp_path)

    loaded, videos, payload = _load_capacity_input(path)

    assert loaded == value
    assert len(videos) == _CLIP_COUNT
    assert hashlib.sha256(payload).hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest()


def test_capacity_input_rejects_duplicate_clip_digest(tmp_path: Path) -> None:
    path, value = _capacity_fixture(tmp_path)
    value["clips"][-1] = dict(value["clips"][0])
    path.write_bytes(canonical_json_bytes(value))

    with pytest.raises(ValueError, match="distinct video digests"):
        _load_capacity_input(path)


@pytest.mark.skipif(
    os.environ.get("BITSIGN_RUN_UMI_MACOS_CAPACITY") != "1",
    reason="set BITSIGN_RUN_UMI_MACOS_CAPACITY=1 for the 112-request Mac rehearsal",
)
def test_macos_four_validator_capacity(tmp_path: Path) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.fail("the Mac capacity rehearsal requires Apple Silicon macOS")
    input_raw = os.environ.get("BITSIGN_UMI_MACOS_CAPACITY_INPUT")
    if input_raw is None:
        pytest.fail("BITSIGN_UMI_MACOS_CAPACITY_INPUT is required")
    input_path = Path(input_raw)
    if not input_path.is_absolute():
        pytest.fail("BITSIGN_UMI_MACOS_CAPACITY_INPUT must be absolute")
    output = _required_new_output("BITSIGN_UMI_MACOS_CAPACITY_REPORT")
    capacity, videos, capacity_bytes = _load_capacity_input(input_path)

    dependencies = _load_release_dependencies()
    shadow_release = importlib.import_module("umi.shadow_release")
    policy_module = importlib.import_module("umi.policy")

    resolved_path = Path(capacity["resolved_miner_release"])
    resolved_bytes = _direct_regular_file(
        resolved_path,
        label="resolved miner release",
        maximum_bytes=256 * 1024,
    )
    resolved = shadow_release.ResolvedMinerRelease.model_validate_json(resolved_bytes)
    if dependencies.umi_canonical_json_bytes(resolved) != resolved_bytes:
        pytest.fail("resolved miner release is not canonical")
    policy_path = Path(resolved.policy_path)
    policy_bytes = _direct_regular_file(
        policy_path,
        label="resolved scoring policy",
        maximum_bytes=2 * 1024 * 1024,
    )
    policy = policy_module.ScoringPolicy.model_validate_json(policy_bytes)
    if dependencies.umi_canonical_json_bytes(policy) != policy_bytes:
        pytest.fail("resolved scoring policy is not canonical")
    policy_sha256 = policy_module.scoring_policy_hash(policy)
    if policy_sha256 != resolved.scoring_policy_sha256:
        pytest.fail("resolved scoring policy digest differs")
    if policy_module.umi_source_tree_sha256() != resolved.umi_source_tree_sha256:
        pytest.fail("installed UMI source tree differs from the resolved signed wheel")
    if len(policy.validator_registry) != _VALIDATOR_COUNT:
        pytest.fail("capacity rehearsal requires exactly four policy validators")

    if resolved.minimum_validator_transport_concurrency < _CLIP_COUNT:
        pytest.fail("every signed validator needs at least 28 transport slots for this burst")
    transport_timeout = resolved.minimum_validator_transport_timeout_seconds
    if transport_timeout > policy.clock.response_window_seconds:
        transport_timeout = float(policy.clock.response_window_seconds)

    outer_concurrency = capacity["outer_inference_concurrency"]
    expected_workers = capacity["expected_actual_workers"]
    if os.environ.get("UMI_S1_EXTRACTOR_PLATFORM") != "linux/amd64":
        pytest.fail("Mac capacity rehearsal requires the Linux/AMD64 extractor")
    if os.environ.get("UMI_S1_DEVICE") not in {"mps", "cpu"}:
        pytest.fail("Mac capacity rehearsal requires native MPS or CPU inference")

    validator_wallets = tuple(
        _development_wallet(dependencies, seed)
        for seed in ("//Alice", "//Charlie", "//Dave", "//Eve")
    )
    validator_hotkeys = frozenset(wallet.hotkey.ss58_address for wallet in validator_wallets)
    miner_wallet = _development_wallet(dependencies, "//Bob")
    miner_hotkey, signature_scheme = dependencies.identity(miner_wallet)
    limits = dependencies.Limits.from_policy(
        policy,
        inference_timeout_seconds=180,
        backend_lifecycle_timeout_seconds=60,
        inference_admission_timeout_seconds=10,
        maximum_inference_concurrency=outer_concurrency,
    )
    backend = dependencies.load_translator(
        "bitsign_motion.umi_reference_backend:translator",
        maximum_concurrency=outer_concurrency,
        expected_model_revision=os.environ.get("UMI_S1_INFERENCE_REVISION"),
    )
    translator = dependencies.WindowCoalescingTranslator(
        backend,
        model_revision=os.environ["UMI_S1_INFERENCE_REVISION"],
        maximum_workers=expected_workers,
        maximum_window_keys=limits.maximum_unique_videos_per_validator_window,
        maximum_inference_seconds=limits.inference_timeout_seconds,
    )
    authenticator = dependencies.RequestAuthenticator.sqlite(
        miner_hotkey,
        tmp_path / "nonces.sqlite3",
        max_age_seconds=limits.btauth_max_age_seconds,
        allowed_skew_seconds=limits.btauth_allowed_skew_seconds,
        allowed_hotkeys=validator_hotkeys,
        maximum_nonces_per_hotkey=limits.maximum_nonce_rows_per_validator,
        maximum_total_nonces=limits.maximum_nonce_rows_total,
        maximum_database_bytes=limits.maximum_nonce_database_bytes,
    )
    resource_ledger = dependencies.SQLiteMinerResourceLedger(
        tmp_path / "assignments.sqlite3",
        miner_hotkey=miner_hotkey,
        scoring_policy_sha256=policy_sha256,
        limits=limits,
    )
    response_deadline_blocks = math.ceil(
        (policy.clock.issue_allowance_seconds + policy.clock.response_window_seconds)
        / policy.clock.target_block_interval_seconds
    )
    runtime = dependencies.MinerRuntime(
        wallet=miner_wallet,
        hotkey_ss58=miner_hotkey,
        signature_scheme=signature_scheme,
        translator=translator,
        video_fetcher=_ExactVideoFetcher(
            {hashlib.sha256(video).hexdigest(): video for video in videos}
        ),
        allowed_validator_hotkeys=validator_hotkeys,
        authenticator=authenticator,
        limits=limits,
        scoring_policy_sha256=policy_sha256,
        response_deadline_blocks=response_deadline_blocks,
        resource_ledger=resource_ledger,
        window_authority=dependencies.LocalComponentWindowAuthority(),
        model_revision=os.environ.get("UMI_S1_INFERENCE_REVISION"),
        inference_semaphore=asyncio.Semaphore(outer_concurrency),
        work_semaphore=asyncio.Semaphore(outer_concurrency),
    )

    async def run() -> tuple[list[Any], list[tuple[int, int]], dict[str, Any], int, int, int]:
        app = dependencies.create_app(runtime)
        async with app.router.lifespan_context(app):
            current_round = dependencies.bt.timelock.current_round()
            response_close_round = current_round + math.ceil(
                policy.clock.response_window_seconds / 3
            )
            reveal_round = response_close_round + math.ceil(policy.clock.reveal_margin_seconds / 3)
            window_id = hashlib.sha256(
                b"umi-s1-macos-capacity-window-v1\0"
                + bytes.fromhex(policy_sha256)
                + hashlib.sha256(capacity_bytes).digest()
                + response_close_round.to_bytes(8, "big")
            ).hexdigest()
            batch_ids = (
                dependencies.base64url_encode(b"A" * 16),
                dependencies.base64url_encode(b"B" * 16),
            )
            requests = []
            for index, video in enumerate(videos):
                requests.append(
                    dependencies.TranslationRequest.model_validate(
                        {
                            "protocol": "umi-asl/0.1",
                            "window_id": window_id,
                            "batch_id": batch_ids[index // 14],
                            "challenge_id": dependencies.base64url_encode(bytes([index + 1]) * 16),
                            "issued_block": 100,
                            "issued_block_hash": "0x" + "30" * 32,
                            "deadline_block": 100 + response_deadline_blocks,
                            "response_close_round": response_close_round,
                            "reveal_round": reveal_round,
                            "video": {
                                "url": f"https://capacity.invalid/{index:032x}",
                                "sha256": hashlib.sha256(video).hexdigest(),
                                "size_bytes": len(video),
                                "media_type": "video/mp4",
                            },
                            "task": {
                                "source_language": "ase",
                                "target_language": "en",
                                "stratum": "continuous",
                            },
                            "scoring_policy_hash": policy_sha256,
                        }
                    )
                )

            transport = dependencies.httpx.ASGITransport(app=app)

            async def query(
                validator_index: int,
                request: Any,
            ) -> tuple[Any, tuple[int, int]]:
                started = time.perf_counter_ns()
                outcome = await dependencies.query_miner(
                    request,
                    wallet=validator_wallets[validator_index],
                    miner_url="http://miner.test",
                    miner_hotkey=miner_hotkey,
                    limits=limits,
                    timeout_seconds=transport_timeout,
                    transport=transport,
                )
                elapsed_ms = (time.perf_counter_ns() - started + 999_999) // 1_000_000
                return outcome, (validator_index, elapsed_ms)

            wall_started = time.perf_counter_ns()
            pairs = await asyncio.gather(
                *(
                    query(validator_index, request)
                    for validator_index in range(_VALIDATOR_COUNT)
                    for request in requests
                )
            )
            wall_ms = (time.perf_counter_ns() - wall_started + 999_999) // 1_000_000
            outcomes = [pair[0] for pair in pairs]
            latencies = [pair[1] for pair in pairs]
            snapshot = translator.capacity_snapshot()
            return (
                outcomes,
                latencies,
                snapshot,
                wall_ms,
                response_close_round,
                reveal_round,
            )

    try:
        outcomes, latencies, snapshot, wall_ms, response_close_round, reveal_round = asyncio.run(
            run()
        )
    finally:
        resource_ledger.close()

    if len(outcomes) != _REQUEST_COUNT or any(
        outcome.failure_code is not None
        or outcome.envelope is None
        or outcome.sealed_response is None
        or outcome.plaintext is not None
        or outcome.plaintext_bytes is not None
        for outcome in outcomes
    ):
        pytest.fail("one or more capacity responses were not valid sealed envelopes")
    if (
        snapshot["translation_jobs_started"] != _CLIP_COUNT
        or snapshot["translation_jobs_succeeded"] != _CLIP_COUNT
        or snapshot["maximum_workers"] != expected_workers
        or snapshot["maximum_window_keys"] != limits.maximum_unique_videos_per_validator_window
        or snapshot["maximum_active_workers"] > expected_workers
    ):
        pytest.fail("capacity run did not execute exactly 28 bounded worker jobs")
    if wall_ms >= int(transport_timeout * 1_000):
        pytest.fail("capacity burst exhausted the shortest validator transport timeout")

    temporary_root = Path(os.environ["UMI_S1_TEMP_ROOT"])
    if list(temporary_root.iterdir()):
        pytest.fail("capacity run left a private job directory")
    docker = os.environ["UMI_S1_DOCKER_EXECUTABLE"]
    containers = subprocess.run(
        [docker, "ps", "--all", "--filter", "name=bitsign-holistic-", "--format", "{{.ID}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if containers.stdout.strip():
        pytest.fail("capacity run left an extractor container")

    all_latencies = [latency for _validator, latency in latencies]
    validator_latency = {
        str(index): {
            "p50_ms": _nearest_rank(
                [latency for validator, latency in latencies if validator == index],
                0.50,
            ),
            "p95_ms": _nearest_rank(
                [latency for validator, latency in latencies if validator == index],
                0.95,
            ),
            "maximum_ms": max(latency for validator, latency in latencies if validator == index),
        }
        for index in range(_VALIDATOR_COUNT)
    }
    report = {
        "schema": _RUN_SCHEMA,
        "status": "passed",
        "input_manifest_sha256": hashlib.sha256(capacity_bytes).hexdigest(),
        "resolved_miner_release_sha256": hashlib.sha256(resolved_bytes).hexdigest(),
        "scoring_policy_sha256": policy_sha256,
        "umi_git_revision": resolved.umi_git_revision,
        "umi_source_tree_sha256": resolved.umi_source_tree_sha256,
        "minimum_validator_transport_concurrency": (
            resolved.minimum_validator_transport_concurrency
        ),
        "runtime": {
            "host_operating_system": platform.system(),
            "host_architecture": platform.machine(),
            "model_device": os.environ["UMI_S1_DEVICE"],
            "extractor_platform": os.environ["UMI_S1_EXTRACTOR_PLATFORM"],
            "outer_inference_concurrency": outer_concurrency,
            "actual_worker_limit": expected_workers,
        },
        "timing": {
            "issue_allowance_seconds": policy.clock.issue_allowance_seconds,
            "response_window_seconds": policy.clock.response_window_seconds,
            "response_deadline_blocks": response_deadline_blocks,
            "response_close_round": response_close_round,
            "reveal_round": reveal_round,
            "shortest_validator_transport_timeout_ms": int(transport_timeout * 1_000),
        },
        "execution": {
            "validator_count": _VALIDATOR_COUNT,
            "unique_clip_count": _CLIP_COUNT,
            "request_count": _REQUEST_COUNT,
            "sealed_response_count": len(outcomes),
            "worker_counters": snapshot,
            "wall_time_ms": wall_ms,
            "latency": {
                "p50_ms": _nearest_rank(all_latencies, 0.50),
                "p95_ms": _nearest_rank(all_latencies, 0.95),
                "maximum_ms": max(all_latencies),
                "by_validator": validator_latency,
            },
            "temporary_job_cleanup_passed": True,
            "extractor_container_cleanup_passed": True,
            "response_set_sha256": _response_set_sha256(outcomes),
            "plaintext_decryption_performed": False,
        },
        "claim_boundary": (
            "Private in-process ASGI Apple Silicon burst evidence for 112 signed requests "
            "over 28 rights-cleared clips. It excludes public-network, TLS, chain-finality, "
            "and peak-memory measurement. It measures the bound host and configuration only; "
            "it is not UMI activation evidence, model-quality evidence, or another host's "
            "capacity."
        ),
    }
    report["content_sha256"] = hashlib.sha256(
        _RUN_DOMAIN + canonical_json_bytes(report)
    ).hexdigest()
    _write_report(output, report)
