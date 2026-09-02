from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

if os.environ.get("BITSIGN_RUN_UMI_RELEASE_E2E") != "1":
    pytest.skip(
        "set BITSIGN_RUN_UMI_RELEASE_E2E=1 to run the real model-to-UMI release test",
        allow_module_level=True,
    )

try:
    import bittensor as bt
    import httpx
    from umi.auth import RequestAuthenticator
    from umi.backends import load_translator
    from umi.config import Limits
    from umi.crypto import decrypt_response
    from umi.miner import MinerRuntime, _identity, create_app
    from umi.protocol import TranslationRequest, base64url_encode
    from umi.validator import query_miner, validate_response_plaintext
    from umi.video import VideoFetcher
except ModuleNotFoundError as exc:  # pragma: no cover - opt-in environment guard
    pytest.skip(f"UMI release test dependencies are unavailable: {exc}", allow_module_level=True)


@dataclass(frozen=True)
class _ExactVideoFetcher(VideoFetcher):
    videos: dict[str, bytes]

    async def fetch(self, descriptor: Any) -> bytes:
        video = self.videos.get(descriptor.sha256)
        if video is None or len(video) != descriptor.size_bytes:
            raise AssertionError("release fixture does not match its request authority")
        return video


def _development_wallet(uri: str) -> Any:
    hotkey = bt.sp_core.Keypair.create_from_uri(uri, crypto_type=bt.sp_core.CRYPTO_SR25519)
    coldkey = bt.sp_core.Keypair.create_from_uri(
        f"{uri}//cold", crypto_type=bt.sp_core.CRYPTO_SR25519
    )
    return SimpleNamespace(coldkey=coldkey, coldkeypub=coldkey, hotkey=hotkey)


def _request(
    video: bytes,
    *,
    index: int,
    reveal_round: int,
    response_close_round: int,
) -> TranslationRequest:
    return TranslationRequest.model_validate(
        {
            "protocol": "umi-asl/0.1",
            "window_id": "10" * 32,
            "batch_id": base64url_encode(b"R" * 16),
            "challenge_id": base64url_encode(bytes([index]) * 16),
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


async def _run_real_reference_model_umi_flow() -> None:
    video_path = _required_path("BITSIGN_UMI_RELEASE_VIDEO")
    valid_video = video_path.read_bytes()
    invalid_video = b"not-an-mp4-release-fixture"
    revision = os.environ.get("UMI_S1_INFERENCE_REVISION")
    assert revision is not None

    validator_wallet = _development_wallet("//Alice")
    miner_wallet = _development_wallet("//Bob")
    miner_hotkey, signature_scheme = _identity(miner_wallet)
    translator = load_translator(
        "bitsign_motion.umi_reference_backend:translator",
        maximum_concurrency=1,
    )
    limits = Limits(
        inference_timeout_seconds=120,
        backend_lifecycle_timeout_seconds=120,
        inference_admission_timeout_seconds=120,
    )
    runtime = MinerRuntime(
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
        allowed_validator_hotkeys=frozenset({validator_wallet.hotkey.ss58_address}),
        authenticator=RequestAuthenticator.in_memory(miner_hotkey),
        limits=limits,
        model_revision=revision,
        inference_semaphore=asyncio.Semaphore(1),
    )
    app = create_app(runtime)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            base_url="http://miner.test",
            transport=httpx.ASGITransport(app=app),
        ) as client:
            health = await client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {
            "ok": True,
            "netuid": 78,
            "translation_weights_active": False,
            "protocol_conformance": False,
            "model_revision": revision,
        }

        current_round = bt.timelock.current_round()
        reveal_round = current_round + 20
        response_close_round = reveal_round - 2
        requests = (
            _request(
                valid_video,
                index=1,
                reveal_round=reveal_round,
                response_close_round=response_close_round,
            ),
            _request(
                invalid_video,
                index=2,
                reveal_round=reveal_round,
                response_close_round=response_close_round,
            ),
        )
        outcomes = []
        for request in requests:
            outcome = await query_miner(
                request,
                wallet=validator_wallet,
                miner_url="http://miner.test",
                miner_hotkey=miner_hotkey,
                limits=limits,
                timeout_seconds=120,
                transport=httpx.ASGITransport(app=app),
            )
            assert outcome.failure_code is None
            assert outcome.envelope is not None
            assert outcome.sealed_response is not None
            assert outcome.plaintext is None
            assert outcome.plaintext_bytes is None
            outcomes.append(outcome)

        plaintexts = []
        for outcome in outcomes:
            assert outcome.envelope is not None
            assert outcome.sealed_response is not None
            raw_plaintext = await asyncio.to_thread(
                decrypt_response,
                outcome.sealed_response,
                reveal_round=reveal_round,
                sha256_hex=outcome.sealed_response.sha256_hex,
                wait=True,
                timeout=180,
            )
            plaintexts.append(
                validate_response_plaintext(
                    raw_plaintext,
                    envelope=outcome.envelope,
                    request=outcome.request,
                )
            )

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
    docker = os.environ["UMI_S1_DOCKER_EXECUTABLE"]
    result = subprocess.run(
        [docker, "ps", "--all", "--filter", "name=bitsign-holistic-", "--format", "{{.ID}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == ""


def test_real_reference_model_returns_signed_timelocked_umi_responses() -> None:
    asyncio.run(_run_real_reference_model_umi_flow())
