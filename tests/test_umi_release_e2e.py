from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_BACKEND_HARD_DEADLINE_SECONDS = 150
_UMI_INFERENCE_TIMEOUT_SECONDS = 180
_UMI_ADMISSION_TIMEOUT_SECONDS = 10
_UMI_LIFECYCLE_TIMEOUT_SECONDS = 60
_RESPONSE_WINDOW_ROUNDS = 80
_REVEAL_WAIT_TIMEOUT_SECONDS = 300
_PACKAGED_NUMERIC_VERSION = re.compile(
    r"(?P<core>[0-9]{1,4}(?:\.[0-9]{1,4}){1,3})"
    r"(?:[+~_-][0-9A-Za-z.+~_-]+)?"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("BITSIGN_RUN_UMI_RELEASE_E2E") != "1",
    reason="set BITSIGN_RUN_UMI_RELEASE_E2E=1 to run the real model-to-UMI release test",
)


@dataclass(frozen=True)
class _ExactVideoFetcher:
    videos: dict[str, bytes]

    async def fetch(self, descriptor: Any) -> bytes:
        video = self.videos.get(descriptor.sha256)
        if video is None or len(video) != descriptor.size_bytes:
            raise AssertionError("release fixture does not match its request authority")
        return video


def _load_release_dependencies() -> SimpleNamespace:
    bitsign_motion = importlib.import_module("bitsign_motion")
    bt = importlib.import_module("bittensor")
    httpx = importlib.import_module("httpx")
    numpy = importlib.import_module("numpy")
    safetensors = importlib.import_module("safetensors")
    torch = importlib.import_module("torch")
    umi = importlib.import_module("umi")
    miner = importlib.import_module("umi.miner")
    protocol = importlib.import_module("umi.protocol")
    validator = importlib.import_module("umi.validator")
    release_evidence = importlib.import_module("bitsign_motion.s1_release_evidence")
    return SimpleNamespace(
        bitsign_motion=bitsign_motion,
        bt=bt,
        httpx=httpx,
        numpy=numpy,
        safetensors=safetensors,
        torch=torch,
        umi=umi,
        model_canonical_json_bytes=importlib.import_module(
            "bitsign_motion.canonical"
        ).canonical_json_bytes,
        umi_canonical_json_bytes=protocol.canonical_json_bytes,
        validate_local_extractor_record=importlib.import_module(
            "bitsign_motion.local_extractor_release"
        ).validate_local_extractor_record,
        release_evidence=release_evidence,
        seal_private_release_e2e=release_evidence.seal_private_release_e2e,
        RequestAuthenticator=importlib.import_module("umi.auth").RequestAuthenticator,
        load_translator=importlib.import_module("umi.backends").load_translator,
        WindowCoalescingTranslator=importlib.import_module(
            "umi.model_scheduler"
        ).WindowCoalescingTranslator,
        Limits=importlib.import_module("umi.config").Limits,
        decrypt_response=importlib.import_module("umi.crypto").decrypt_response,
        LocalComponentWindowAuthority=importlib.import_module(
            "umi.miner_admission"
        ).LocalComponentWindowAuthority,
        SQLiteMinerResourceLedger=importlib.import_module(
            "umi.miner_resources"
        ).SQLiteMinerResourceLedger,
        MinerRuntime=miner.MinerRuntime,
        identity=miner._identity,
        create_app=miner.create_app,
        TranslationRequest=protocol.TranslationRequest,
        base64url_encode=protocol.base64url_encode,
        query_miner=validator.query_miner,
        validate_response_plaintext=validator.validate_response_plaintext,
    )


def _development_wallet(dependencies: SimpleNamespace, uri: str) -> Any:
    hotkey = dependencies.bt.sp_core.Keypair.create_from_uri(
        uri, crypto_type=dependencies.bt.sp_core.CRYPTO_SR25519
    )
    coldkey = dependencies.bt.sp_core.Keypair.create_from_uri(
        f"{uri}//cold", crypto_type=dependencies.bt.sp_core.CRYPTO_SR25519
    )
    return SimpleNamespace(coldkey=coldkey, coldkeypub=coldkey, hotkey=hotkey)


def _request(
    dependencies: SimpleNamespace,
    video: bytes,
    *,
    index: int,
    reveal_round: int,
    response_close_round: int,
) -> Any:
    return dependencies.TranslationRequest.model_validate(
        {
            "protocol": "umi-asl/0.1",
            "window_id": "10" * 32,
            "batch_id": dependencies.base64url_encode(b"R" * 16),
            "challenge_id": dependencies.base64url_encode(bytes([index]) * 16),
            "issued_block": 100,
            "issued_block_hash": "0x" + "30" * 32,
            "deadline_block": 110,
            "response_close_round": response_close_round,
            "reveal_round": reveal_round,
            "video": {
                "url": f"https://release.invalid/{index:032x}",
                "sha256": hashlib.sha256(video).hexdigest(),
                "size_bytes": len(video),
                "media_type": "video/mp4",
            },
            "task": {
                "source_language": "ase",
                "target_language": "en",
                "stratum": "continuous",
            },
            "scoring_policy_hash": "20" * 32,
        }
    )


def _required_path(name: str) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        pytest.fail(f"required release-test environment variable is missing: {name}")
    path = Path(raw)
    if not path.is_absolute() or not path.is_file():
        pytest.fail(f"{name} must identify an existing absolute file")
    return path


def _required_output_path(name: str) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        pytest.fail(f"required release-test environment variable is missing: {name}")
    path = Path(raw)
    if not path.is_absolute() or path.exists() or path.parent.resolve(strict=True) != path.parent:
        pytest.fail(f"{name} must identify a new file in an existing absolute directory")
    return path


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_git_revision(repository: Path) -> str:
    revision_result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "-C",
            str(repository),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    revision = revision_result.stdout.strip()
    assert len(revision) == 40 and revision == revision.lower()
    status_result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "-C",
            str(repository),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert status_result.stdout == "", f"release E2E repository is dirty: {repository}"
    return revision


def _assert_module_from_repository(module: Any, repository: Path) -> None:
    raw_origin = getattr(module, "__file__", None)
    assert isinstance(raw_origin, str) and raw_origin
    origin = Path(raw_origin).resolve(strict=True)
    source_root = (repository / "src").resolve(strict=True)
    assert origin.is_relative_to(source_root), (
        f"{module.__name__} was imported from {origin}, not {source_root}"
    )


def _assert_bittensor_matches_umi_lock(umi_repository: Path) -> None:
    lock_path = umi_repository / "uv.lock"
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)
    versions = [
        package.get("version")
        for package in lock.get("package", [])
        if package.get("name") == "bittensor"
    ]
    installed = importlib.metadata.version("bittensor")
    assert versions == [installed], (
        f"installed bittensor {installed} does not match the UMI lock: {versions}"
    )


def _wire_response_sha256(outcome: Any) -> str:
    assert outcome.envelope_bytes is not None
    assert outcome.response_signature is not None
    body = outcome.envelope_bytes
    signature = outcome.response_signature.encode("ascii")
    digest = hashlib.sha256(b"umi-s1-release-wire-response-v1\0")
    digest.update(len(body).to_bytes(8, "big"))
    digest.update(body)
    digest.update(len(signature).to_bytes(8, "big"))
    digest.update(signature)
    return digest.hexdigest()


def _plaintext_set_sha256(plaintexts: list[bytes]) -> str:
    digest = hashlib.sha256(b"umi-s1-release-plaintext-set-v1\0")
    digest.update(len(plaintexts).to_bytes(4, "big"))
    for plaintext in plaintexts:
        digest.update(len(plaintext).to_bytes(8, "big"))
        digest.update(plaintext)
    return digest.hexdigest()


def _docker_version(docker: str) -> str:
    result = subprocess.run(
        [docker, "version", "--format", "{{.Client.Version}}|{{.Server.Version}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    client, separator, server = result.stdout.strip().partition("|")
    assert separator and client and server
    client_match = _PACKAGED_NUMERIC_VERSION.fullmatch(client)
    server_match = _PACKAGED_NUMERIC_VERSION.fullmatch(server)
    assert client_match is not None and server_match is not None
    return f"client={client_match.group('core')};server={server_match.group('core')}"


def _write_private_report(
    path: Path,
    report: dict[str, Any],
    *,
    canonical_json_bytes: Callable[[Any], bytes],
) -> None:
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
            assert written > 0
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def _run_real_reference_model_umi_flow(
    dependencies: SimpleNamespace,
    *,
    reference_repository: Path,
    reference_revision: str,
    umi_repository: Path,
    umi_revision: str,
    state_root: Path,
) -> dict[str, Any]:
    started_at_utc = _utc_now()
    _assert_module_from_repository(dependencies.bitsign_motion, reference_repository)
    deployment_profile = os.environ.get("BITSIGN_UMI_DEPLOYMENT_PROFILE")
    if deployment_profile is None:
        _assert_module_from_repository(dependencies.umi, umi_repository)
    else:
        resolved_path = _required_path("BITSIGN_UMI_RESOLVED_MINER_RELEASE")
        resolved_bytes = resolved_path.read_bytes()
        resolved_model = importlib.import_module("umi.shadow_release").ResolvedMinerRelease
        resolved = resolved_model.model_validate_json(resolved_bytes)
        assert dependencies.umi_canonical_json_bytes(resolved) == resolved_bytes
        assert resolved.umi_git_revision == umi_revision
        assert importlib.import_module("umi.policy").umi_source_tree_sha256() == (
            resolved.umi_source_tree_sha256
        )
        umi_origin = Path(dependencies.umi.__file__).resolve(strict=True)
        assert not umi_origin.is_relative_to((umi_repository / "src").resolve(strict=True))
    _assert_bittensor_matches_umi_lock(umi_repository)

    video_path = _required_path("BITSIGN_UMI_RELEASE_VIDEO")
    build_record_path = _required_path("BITSIGN_UMI_EXTRACTOR_BUILD_RECORD")
    valid_video = video_path.read_bytes()
    invalid_video = b"not-an-mp4-release-fixture"
    revision = os.environ.get("UMI_S1_INFERENCE_REVISION")
    assert revision is not None
    base_revision = os.environ.get("UMI_S1_BASE_INFERENCE_REVISION")
    assert base_revision is not None and base_revision != revision
    assert os.environ.get("BITSIGN_UMI_RELEASE_VIDEO_RIGHTS_CLEARED") == "1"

    if deployment_profile is None:
        assert platform.system() == "Linux"
        assert platform.machine() == "x86_64"
        assert os.environ.get("UMI_S1_DEVICE") == "cpu"
    else:
        assert deployment_profile == dependencies.release_evidence.MACOS_MINER_DEPLOYMENT_PROFILE
        assert platform.system() == "Darwin"
        assert platform.machine() == "arm64"
        assert os.environ.get("BITSIGN_UMI_RELEASE_PROFILE") == (
            dependencies.release_evidence.PUBLIC_S1_FINETUNE_RELEASE_PROFILE
        )
        assert os.environ.get("UMI_S1_DEVICE") in {"cpu", "mps"}
        if os.environ.get("UMI_S1_DEVICE") == "mps":
            assert dependencies.torch.backends.mps.is_built()
            assert dependencies.torch.backends.mps.is_available()
    assert os.environ.get("UMI_S1_EXTRACTOR_PLATFORM") == "linux/amd64"
    assert os.environ.get("UMI_S1_HARD_DEADLINE_SECONDS") == str(_BACKEND_HARD_DEADLINE_SECONDS)
    docker = os.environ["UMI_S1_DOCKER_EXECUTABLE"]
    extractor_record_raw = build_record_path.read_bytes()
    extractor_record_value = json.loads(extractor_record_raw)
    extractor_record = dependencies.validate_local_extractor_record(
        extractor_record_value,
        docker_executable=Path(docker),
    )
    assert extractor_record["image_id"] == os.environ["UMI_S1_EXTRACTOR_IMAGE"]
    task_model = _required_path("UMI_S1_EXTRACTOR_MODEL").read_bytes()
    task_model_sha256 = hashlib.sha256(task_model).hexdigest()
    assert task_model_sha256 == "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8"

    validator_wallets = tuple(
        _development_wallet(dependencies, seed)
        for seed in ("//Alice", "//Charlie", "//Dave", "//Eve")
    )
    validator_hotkeys = frozenset(wallet.hotkey.ss58_address for wallet in validator_wallets)
    miner_wallet = _development_wallet(dependencies, "//Bob")
    miner_hotkey, signature_scheme = dependencies.identity(miner_wallet)
    limits = dependencies.Limits(
        inference_timeout_seconds=_UMI_INFERENCE_TIMEOUT_SECONDS,
        backend_lifecycle_timeout_seconds=_UMI_LIFECYCLE_TIMEOUT_SECONDS,
        inference_admission_timeout_seconds=_UMI_ADMISSION_TIMEOUT_SECONDS,
        maximum_inference_concurrency=len(validator_wallets),
    )
    translator = dependencies.load_translator(
        "bitsign_motion.umi_reference_backend:translator",
        maximum_concurrency=limits.maximum_inference_concurrency,
        expected_model_revision=revision,
    )
    state_root.mkdir(mode=0o700)
    authenticator = dependencies.RequestAuthenticator.sqlite(
        miner_hotkey,
        state_root / "nonces.sqlite3",
        max_age_seconds=limits.btauth_max_age_seconds,
        allowed_skew_seconds=limits.btauth_allowed_skew_seconds,
        allowed_hotkeys=validator_hotkeys,
        maximum_nonces_per_hotkey=limits.maximum_nonce_rows_per_validator,
        maximum_total_nonces=limits.maximum_nonce_rows_total,
        maximum_database_bytes=limits.maximum_nonce_database_bytes,
    )
    resource_ledger = dependencies.SQLiteMinerResourceLedger(
        state_root / "assignments.sqlite3",
        miner_hotkey=miner_hotkey,
        scoring_policy_sha256="20" * 32,
        limits=limits,
    )
    try:
        runtime = dependencies.MinerRuntime(
            wallet=miner_wallet,
            hotkey_ss58=miner_hotkey,
            signature_scheme=signature_scheme,
            translator=translator,
            video_fetcher=_ExactVideoFetcher(
                {
                    hashlib.sha256(valid_video).hexdigest(): valid_video,
                    hashlib.sha256(invalid_video).hexdigest(): invalid_video,
                }
            ),
            allowed_validator_hotkeys=validator_hotkeys,
            authenticator=authenticator,
            limits=limits,
            scoring_policy_sha256="20" * 32,
            response_deadline_blocks=10,
            resource_ledger=resource_ledger,
            window_authority=dependencies.LocalComponentWindowAuthority(),
            model_revision=revision,
            inference_semaphore=asyncio.Semaphore(len(validator_wallets)),
            work_semaphore=asyncio.Semaphore(len(validator_wallets)),
        )
        app = dependencies.create_app(runtime)
        async with app.router.lifespan_context(app):
            async with dependencies.httpx.AsyncClient(
                base_url="http://miner.test",
                transport=dependencies.httpx.ASGITransport(app=app),
            ) as client:
                health = await client.get("/healthz")
            assert health.status_code == 200
            assert health.json() == {
                "ok": True,
                "netuid": 78,
                "translation_weights_active": False,
                "protocol_conformance": False,
                "runtime_mode": "inactive_shadow",
                "scoring_policy_sha256": "20" * 32,
                "model_revision": revision,
                "window_authority": "LocalComponentWindowAuthority",
                "finality_service": "component_authority",
            }

            current_round = dependencies.bt.timelock.current_round()
            response_close_round = current_round + _RESPONSE_WINDOW_ROUNDS
            reveal_round = response_close_round + 2
            requests = (
                _request(
                    dependencies,
                    valid_video,
                    index=1,
                    reveal_round=reveal_round,
                    response_close_round=response_close_round,
                ),
                _request(
                    dependencies,
                    invalid_video,
                    index=2,
                    reveal_round=reveal_round,
                    response_close_round=response_close_round,
                ),
            )
            outcomes = list(
                await asyncio.gather(
                    *(
                        dependencies.query_miner(
                            request,
                            wallet=validator_wallet,
                            miner_url="http://miner.test",
                            miner_hotkey=miner_hotkey,
                            limits=limits,
                            timeout_seconds=_UMI_INFERENCE_TIMEOUT_SECONDS,
                            transport=dependencies.httpx.ASGITransport(app=app),
                        )
                        for request, validator_wallet in zip(
                            requests,
                            validator_wallets[:2],
                            strict=True,
                        )
                    )
                )
            )
            for outcome in outcomes:
                assert outcome.failure_code is None
                assert outcome.envelope is not None
                assert outcome.sealed_response is not None
                assert outcome.plaintext is None
                assert outcome.plaintext_bytes is None

            plaintext_bytes = []
            plaintexts = []
            for outcome in outcomes:
                assert outcome.envelope is not None
                assert outcome.sealed_response is not None
                raw_plaintext = await asyncio.to_thread(
                    dependencies.decrypt_response,
                    outcome.sealed_response,
                    reveal_round=reveal_round,
                    sha256_hex=outcome.sealed_response.sha256_hex,
                    wait=True,
                    timeout=_REVEAL_WAIT_TIMEOUT_SECONDS,
                )
                plaintext_bytes.append(raw_plaintext)
                plaintexts.append(
                    dependencies.validate_response_plaintext(
                        raw_plaintext,
                        envelope=outcome.envelope,
                        request=outcome.request,
                    )
                )
    finally:
        resource_ledger.close()

    successful, invalid = plaintexts
    assert successful.status == "ok"
    assert successful.model_revision == revision
    assert successful.hypothesis is not None
    assert len(successful.hypothesis.split()) <= 8
    assert invalid.status == "error"
    assert invalid.error_code == "inference_failed"
    assert invalid.hypothesis is None
    assert invalid.model_revision is None

    temporary_root = Path(os.environ["UMI_S1_TEMP_ROOT"])
    assert list(temporary_root.iterdir()) == []
    result = subprocess.run(
        [docker, "ps", "--all", "--filter", "name=bitsign-holistic-", "--format", "{{.ID}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == ""

    execution = {
        "request_count": 2,
        "valid_video_status": "ok",
        "invalid_video_status": "error",
        "invalid_video_error_code": "inference_failed",
        "signed_envelopes_verified": True,
        "plaintext_unavailable_before_reveal": True,
        "decryption_after_reveal_verified": True,
        "response_bindings_verified": True,
        "maximum_output_words": 8,
        "output_limit_passed": True,
        "temporary_job_cleanup_passed": True,
        "extractor_container_cleanup_passed": True,
        "video_fixture_distributed": False,
        "translation_quality_measured": False,
    }
    valid_wire_sha256 = _wire_response_sha256(outcomes[0])
    invalid_wire_sha256 = _wire_response_sha256(outcomes[1])
    plaintext_set_sha256 = _plaintext_set_sha256(plaintext_bytes)
    run_log = {
        "schema": "umi-s1-release-e2e-log/1",
        "derived_inference_revision": revision,
        "valid_wire_response_sha256": valid_wire_sha256,
        "invalid_wire_response_sha256": invalid_wire_sha256,
        "post_reveal_plaintext_set_sha256": plaintext_set_sha256,
        "execution": execution,
    }
    assert _clean_git_revision(reference_repository) == reference_revision
    assert _clean_git_revision(umi_repository) == umi_revision
    release_profile = os.environ.get("BITSIGN_UMI_RELEASE_PROFILE")
    if release_profile is None:
        release_id = "umi-s1-baseline-v0"
    else:
        assert release_profile == dependencies.release_evidence.PUBLIC_S1_FINETUNE_RELEASE_PROFILE
        release_id = dependencies.release_evidence.PUBLIC_S1_FINETUNE_RELEASE_ID
    runtime_evidence = {
        "host_operating_system": platform.system(),
        "host_architecture": platform.machine(),
        "container_platform": os.environ["UMI_S1_EXTRACTOR_PLATFORM"],
        "model_device": os.environ["UMI_S1_DEVICE"],
        "python_version": platform.python_version(),
        "torch_version": dependencies.torch.__version__.split("+")[0],
        "numpy_version": dependencies.numpy.__version__,
        "safetensors_version": dependencies.safetensors.__version__,
        "bittensor_version": importlib.metadata.version("bittensor"),
        "docker_engine_version": _docker_version(docker),
        "extractor_image_id": extractor_record["image_id"],
        "mediapipe_task_model_sha256": task_model_sha256,
    }
    if deployment_profile is not None:
        runtime_evidence.update(
            {
                "model_execution": f"native-pytorch-{os.environ['UMI_S1_DEVICE']}",
                "mps_is_built": dependencies.torch.backends.mps.is_built(),
                "mps_is_available": dependencies.torch.backends.mps.is_available(),
            }
        )
    capture = {
        "release_id": release_id,
        "status": "passed",
        "base_inference_revision": base_revision,
        "derived_inference_revision": revision,
        "tested_reference_model_git_revision": reference_revision,
        "umi_git_revision": umi_revision,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now(),
        "runtime": runtime_evidence,
        "timeouts_seconds": {
            "inner_model_hard_deadline": _BACKEND_HARD_DEADLINE_SECONDS,
            "outer_inference_timeout": _UMI_INFERENCE_TIMEOUT_SECONDS,
            "outer_admission_timeout": _UMI_ADMISSION_TIMEOUT_SECONDS,
            "outer_lifecycle_timeout": _UMI_LIFECYCLE_TIMEOUT_SECONDS,
        },
        "fixture": {
            "fixture_class": "rights-cleared-private-video",
            "video_sha256": hashlib.sha256(valid_video).hexdigest(),
            "rights_cleared_for_private_testing": True,
            "distributed": False,
        },
        "execution": execution,
        "evidence": {
            "extractor_build_record_content_sha256": extractor_record["content_sha256"],
            "extractor_build_record_file_sha256": hashlib.sha256(extractor_record_raw).hexdigest(),
            "valid_wire_response_sha256": valid_wire_sha256,
            "invalid_wire_response_sha256": invalid_wire_sha256,
            "post_reveal_plaintext_set_sha256": plaintext_set_sha256,
            "run_log_sha256": hashlib.sha256(
                dependencies.model_canonical_json_bytes(run_log)
            ).hexdigest(),
        },
    }
    if release_profile is not None:
        capture["release_profile"] = release_profile
    if deployment_profile is not None:
        capture["deployment_profile"] = deployment_profile
    return dependencies.seal_private_release_e2e(capture)


def test_real_reference_model_returns_signed_timelocked_umi_responses(tmp_path: Path) -> None:
    output = _required_output_path("BITSIGN_UMI_RELEASE_E2E_REPORT")
    reference_repository = Path(__file__).resolve().parents[1]
    umi_raw = os.environ.get("BITSIGN_UMI_REPOSITORY")
    if umi_raw is None:
        pytest.fail("required release-test environment variable is missing: BITSIGN_UMI_REPOSITORY")
    umi_repository = Path(umi_raw).resolve(strict=True)
    assert reference_repository.is_dir() and umi_repository.is_dir()
    reference_revision = _clean_git_revision(reference_repository)
    umi_revision = _clean_git_revision(umi_repository)

    dependencies = _load_release_dependencies()
    report = asyncio.run(
        _run_real_reference_model_umi_flow(
            dependencies,
            reference_repository=reference_repository,
            reference_revision=reference_revision,
            umi_repository=umi_repository,
            umi_revision=umi_revision,
            state_root=tmp_path / "miner-state",
        )
    )
    _write_private_report(
        output,
        report,
        canonical_json_bytes=dependencies.model_canonical_json_bytes,
    )
