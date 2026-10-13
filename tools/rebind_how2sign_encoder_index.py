"""Reuse a complete aligned encoder index in a larger How2Sign selection.

Run after the prior cache worker exits. Source clips, resampled clips and frozen
encoder states stay in place; the new index records their prior provenance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from cache_how2sign_encoder import checked_receipts
from freeze_how2sign_training_index import canonical_rows
from rebind_how2sign_clip_receipts import SOURCE_FIELDS


def rebound_rows(
    prior_selection: dict,
    expanded_selection: dict,
    prior_rows: list[dict],
    expanded_receipts: dict[str, dict],
    prior_selection_hash: str,
    expanded_selection_hash: str,
    prior_index_hash: str,
    revision: str,
) -> list[dict]:
    if (
        prior_selection.get("schema") != "umi-studio-how2sign-training-aligned/1"
        or expanded_selection.get("schema") != "umi-studio-how2sign-training-aligned/1"
        or any(
            key not in prior_selection
            or key not in expanded_selection
            or prior_selection[key] != expanded_selection[key]
            for key in SOURCE_FIELDS
        )
    ):
        raise ValueError("expanded selection uses different training sources")
    old = {row["sentence_name"]: row for row in prior_selection["rows"]}
    expanded = {row["sentence_name"]: row for row in expanded_selection["rows"]}
    records = {row["sentence_name"]: row for row in prior_rows}
    if (
        len(old) != len(prior_selection["rows"])
        or len(expanded) != len(expanded_selection["rows"])
        or len(records) != len(prior_rows)
        or len(expanded) <= len(old)
        or set(records) != set(old)
        or any(expanded.get(name) != row for name, row in old.items())
    ):
        raise ValueError("complete prior encoder selection is not an exact expansion subset")
    rebound = []
    for row in expanded_selection["rows"]:
        name = row["sentence_name"]
        if name not in records:
            continue
        prior = records[name]
        receipt = expanded_receipts.get(name)
        if (
            prior["selection_sha256"] != prior_selection_hash
            or receipt is None
            or receipt["selection_sha256"] != expanded_selection_hash
            or prior["source_clip_sha256"] != receipt["clip_sha256"]
            or prior["reference"] != row["reference"]
            or prior["signer"] != row["signer"]
        ):
            raise ValueError(f"expanded receipt or encoder state differs: {name}")
        rebound.append(
            {
                **prior,
                "selection_sha256": expanded_selection_hash,
                "clip_receipts_sha256": None,
                "rebind_source": {
                    "prior_selection_sha256": prior_selection_hash,
                    "prior_index_sha256": prior_index_hash,
                    "prior_encoder_cache_sha256": prior["encoder_cache_sha256"],
                    "rebind_miner_model_revision": revision,
                },
            }
        )
    return rebound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-selection", type=Path, required=True)
    parser.add_argument("--prior-receipts", type=Path, required=True)
    parser.add_argument("--prior-index", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clip-receipts", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--resampled-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if baseline != workspace / "artifacts/baseline-6fe8df59":
        parser.error("requires the independent frozen baseline")
    if args.output.exists() or args.output.resolve() == args.prior_index.resolve():
        parser.error("new index must not replace an existing index")
    prior_selection = json.loads(args.prior_selection.read_text())
    expanded_selection = json.loads(args.selection.read_text())
    prior_selection_hash = sha256(args.prior_selection)
    expanded_selection_hash = sha256(args.selection)
    prior_receipts = checked_receipts(
        prior_selection,
        prior_selection_hash,
        args.prior_receipts.read_text().splitlines(),
    )
    expanded_receipts = checked_receipts(
        expanded_selection,
        expanded_selection_hash,
        args.clip_receipts.read_text().splitlines(),
    )
    prior_index_hash = sha256(args.prior_index)
    prior_rows = [json.loads(line) for line in args.prior_index.read_text().splitlines()]
    canonical = canonical_rows(
        prior_selection,
        prior_selection_hash,
        prior_receipts,
        prior_rows,
        args.source_dir,
        args.resampled_dir,
        args.cache_dir,
        sha256(baseline / "MODEL_IDENTITY.json"),
        sha256(baseline / "runtime.py"),
        args.miner_health_url,
    )
    revision = miner_revision(args.miner_health_url, None)
    rebound = rebound_rows(
        prior_selection,
        expanded_selection,
        canonical,
        expanded_receipts,
        prior_selection_hash,
        expanded_selection_hash,
        prior_index_hash,
        revision,
    )
    if sha256(args.prior_index) != prior_index_hash:
        raise RuntimeError("prior encoder index changed while rebinding")
    miner_revision(args.miner_health_url, revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.jsonl")
    if temporary.exists():
        parser.error("unfinished expanded index needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for row in rebound:
                output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"rebound {len(rebound)} verified encoder states")


if __name__ == "__main__":
    main()
