"""Freeze a document-spread training pilot from the public FLEURS split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select(selection: dict, existing: list[dict], target: int) -> dict:
    if selection.get("schema") != "umi-studio-fleurs-dev-split/1":
        raise ValueError("unexpected development selection schema")
    training = {row["example_id"]: row for row in selection["training"]}
    if len(training) != len(selection["training"]):
        raise ValueError("training selection has duplicate examples")
    existing_ids = [row["example_id"] for row in existing]
    if len(existing_ids) != len(set(existing_ids)) or not set(existing_ids) <= training.keys():
        raise ValueError("existing cache is not a unique subset of training")
    if any(row.get("split") != "training" or row.get("signer") != 3 for row in existing):
        raise ValueError("existing cache is not signer-3 training data")
    if target <= len(existing_ids):
        raise ValueError("target must exceed the existing cache size")
    used_docs = {training[example_id]["document_id"] for example_id in existing_ids}
    by_document = defaultdict(list)
    for row in training.values():
        if row["document_id"] not in used_docs:
            by_document[row["document_id"]].append(row)
    count = target - len(existing_ids)
    if count > len(by_document):
        raise ValueError("not enough unused source documents for one-per-document sampling")
    documents = sorted(by_document, key=lambda doc: _digest(f"umi-studio-pilot-v1:{doc}"))
    added = [
        min(
            by_document[document],
            key=lambda row: _digest(f"umi-studio-pilot-row-v1:{row['example_id']}"),
        )["example_id"]
        for document in documents[:count]
    ]
    all_ids = existing_ids + added
    if set(all_ids) & {row["example_id"] for row in selection["validation"]}:
        raise ValueError("pilot overlaps validation")
    return {
        "schema": "umi-studio-fleurs-training-pilot/1",
        "target_examples": target,
        "existing_example_ids": existing_ids,
        "requested_example_ids": added,
        "all_example_ids": all_ids,
        "source_documents": len({training[example_id]["document_id"] for example_id in all_ids}),
        "selection_rule": (
            "keep existing cache; add one hash-ranked example from each hash-ranked unused document"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--training-index", type=Path, required=True)
    parser.add_argument("--target-examples", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection_bytes = args.selection.read_bytes()
    index_bytes = args.training_index.read_bytes()
    selection = json.loads(selection_bytes)
    existing = [json.loads(line) for line in index_bytes.splitlines()]
    result = select(selection, existing, args.target_examples)
    result["selection_sha256"] = hashlib.sha256(selection_bytes).hexdigest()
    result["existing_index_sha256"] = hashlib.sha256(index_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    print(
        f"pilot={len(result['all_example_ids'])} examples "
        f"across {result['source_documents']} source documents"
    )


if __name__ == "__main__":
    main()
