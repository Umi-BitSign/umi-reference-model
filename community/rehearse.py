"""Run one local smoke case through UMI's real offline CPU evaluator.

Invoke with the UMI Python environment, not the model environment. The policy
below is unsigned, already expired and restricted to this local smoke record.
This command has no chain client, wallet, signing or promotion operation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from umi.competition_artifacts import preserve_bundle
from umi.competition_runner import OfflineCpuRuntime, execute_offline_case
from umi.open_competition import CompetitionPolicy, Evaluator, ModelBundle, digest
from umi.protocol import canonical_json_bytes


def smoke_policy(runtime: OfflineCpuRuntime) -> CompetitionPolicy:
    return CompetitionPolicy(
        schema="umi-open-competition-policy/1",
        network="finney",
        netuid=78,
        sequence=1,
        predecessor_sha256=None,
        valid_from_block=1,
        valid_through_block=2,
        endpoint_reward_bps=7000,
        model_reward_bps=3000,
        minimum_score_bps=1000,
        promotion_margin_bps=100,
        minimum_cases_per_stratum=1,
        maximum_inference_ms=600_000,
        maximum_output_bytes=4096,
        maximum_bundle_bytes=4 * 1024**3,
        maximum_bundle_files=512,
        minimum_submission_interval_blocks=1,
        maximum_submission_lifetime_blocks=1,
        maximum_snapshot_age_blocks=0,
        maximum_uids=256,
        evaluators=(
            Evaluator(
                # Public AccountId32 0x0101..., no corresponding secret is created.
                hotkey="5C62Ck4UrFPiBtoCmeSrgF7x9yv9mn38446dhCpsi2mLHiFT",
                control_group="local-smoke-only",
            ),
        ),
        required_evaluator_groups=1,
        contribution_terms_sha256=hashlib.sha256(b"SMOKE ONLY. NO RIGHTS APPROVAL.").hexdigest(),
        accepted_model_licenses=("MIT",),
        evaluation_runtime_sha256=digest(runtime),
    )


async def run(args) -> dict:
    staged = args.staged.resolve(strict=True)
    bundle = ModelBundle.model_validate_json((staged / "manifest.json").read_bytes())
    intake = json.loads((staged / "intake.json").read_bytes())
    if digest(bundle) != intake["model_bundle_sha256"]:
        raise ValueError("staged intake does not bind the UMI model bundle")
    runtime = OfflineCpuRuntime(
        schema="umi-offline-cpu-runtime/2",
        image=args.image,
        cpus=4,
        memory_bytes=12 * 1024**3,
        scratch_bytes=512 * 1024**2,
        pids_limit=128,
        maximum_video_bytes=16 * 1024**2,
    )
    policy = smoke_policy(runtime)
    archive = args.archive.resolve()
    preserve_bundle(bundle, staged / "model", archive, policy)
    with args.video.open("rb") as source:
        video = source.read(runtime.maximum_video_bytes + 1)
    if not video or len(video) > runtime.maximum_video_bytes:
        raise ValueError("smoke video exceeds its byte bound or is empty")
    result = await execute_offline_case(
        bundle=bundle,
        archive=archive,
        runtime=runtime,
        policy=policy,
        case_id=hashlib.sha256(b"local-community-model-smoke").hexdigest(),
        video_sha256=hashlib.sha256(video).hexdigest(),
        video=video,
    )
    return {
        "schema": "umi-community-baseline-smoke/1",
        "synthetic_policy_only": True,
        "quality_evaluated": False,
        "independent_evaluation": False,
        "rights_reviewed": False,
        "chain_submission_authorized": False,
        "runtime": runtime.model_dump(mode="json", by_alias=True),
        "policy_sha256": digest(policy),
        "execution": result.model_dump(mode="json", by_alias=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new receipt path")
    report = asyncio.run(run(args))
    with args.output.open("xb") as output:
        output.write(canonical_json_bytes(report))
    args.output.chmod(0o400)
    execution = report["execution"]
    print(
        json.dumps(
            {
                "status": execution["output"]["status"],
                "reason": execution["reason"],
                "elapsed_ms": execution["output"]["elapsed_ms"],
                "model_sha256": execution["model_sha256"],
                "quality_evaluated": False,
                "chain_submission_authorized": False,
            }
        )
    )
    if execution["output"]["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
