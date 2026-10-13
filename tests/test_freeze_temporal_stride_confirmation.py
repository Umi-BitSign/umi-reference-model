"""The stride-2 confirmation plan keeps both held-out domains sealed and fixed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_temporal_stride_confirmation import build_plan


def test_builds_fixed_42_plus_46_confirmation():
    h_rows = [
        {"sentence_name": f"h-{i}", "signer": (1, 3, 5, 8)[i % 4]}
        for i in range(58)
    ]
    h_receipts = [
        {
            "schema": "umi-studio-how2sign-validation-clip/1",
            "selection_sha256": "hs",
            "split": "validation",
            "sentence_name": row["sentence_name"],
            "clip_sha256": f"clip-{i}",
        }
        for i, row in enumerate(h_rows)
    ]
    f_rows = [
        {
            "example_id": f"f-{i}",
            "document_id": f"doc-{i // 2}",
            "signer": 4,
            "source_video": f"root/doc-{i // 2}.mp4",
            "start_time_sec": float(i),
            "end_time_sec": float(i + 1),
        }
        for i in range(52)
    ]
    plan = build_plan(
        decision={
            "schema": "umi-studio-visual-lora-corrective-decision/1",
            "decision": "proceed_to_confirmation",
            "confirmation_evaluated": False,
            "candidate_temporal_stride": 2,
            "candidate_adapter_sha256": "adapter",
            "how2sign": {"checkpoint_sha256": "checkpoint"},
        },
        how2sign_selection={
            "schema": "umi-studio-how2sign-validation-selection/1",
            "rows": h_rows,
        },
        how2sign_partition={
            "schema": "umi-studio-how2sign-validation-partition/1",
            "selection_sha256": "hs",
            "confirmation_predictions_inspected": False,
            "confirmation_sentence_names": [row["sentence_name"] for row in h_rows[16:]],
        },
        how2sign_receipts=h_receipts,
        fleurs_selection={
            "schema": "umi-studio-fleurs-dev-split/1",
            "validation": f_rows,
        },
        fleurs_partition={
            "schema": "umi-studio-fleurs-dev-evaluation-partition/1",
            "selection_sha256": "fs",
            "tuning_documents": ["doc-0", "doc-1", "doc-2"],
        },
        fleurs_receipt={
            "schema": "umi-studio-fleurs-dev-source-files/1",
            "selection_sha256": "fs",
            "split": "validation",
            "files": [
                {"path": f"root/doc-{i}.mp4", "sha256": f"source-{i}"}
                for i in range(26)
            ],
        },
        hashes={
            "decision": "decision",
            "how2sign_selection": "hs",
            "how2sign_partition": "hp",
            "how2sign_receipts": "hr",
            "fleurs_selection": "fs",
            "fleurs_partition": "fp",
            "fleurs_receipt": "fr",
        },
    )
    assert plan["confirmation_examples"] == {"how2sign": 42, "fleurs": 46}
    assert len(plan["records"]) == 88
    assert plan["candidate_temporal_stride"] == 2
    assert "reference" not in plan["records"][0]
