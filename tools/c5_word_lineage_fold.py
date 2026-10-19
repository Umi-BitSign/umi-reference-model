"""Reconstruct accepted work across committed word-training continuations.

Callers must authenticate the ledger bytes, common frozen recipe, terminal
state, raw grades, tensors and authorization separately. This module neither
admits a checkpoint nor launches or authorizes a continuation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class Segment:
    ledger: bytes
    selected_rows: int


def checked_rows(raw, batches):
    require(
        isinstance(raw, bytes) and raw and raw.endswith(b"\n"),
        "lineage ledger must contain complete newline-terminated records",
    )
    lines = raw.splitlines(keepends=True)
    require(len(lines) <= len(batches), "lineage ledger extends past the frozen curriculum")
    rows = []
    examples = updates = 0
    for index, line in enumerate(lines):
        row = json.loads(line)
        batch = batches[index]
        require(
            isinstance(row, dict)
            and type(row.get("batch")) is int
            and row["batch"] == index
            and batch["training_record_indices"]
            and all(row.get(k) == v for k, v in batch.items()),
            "lineage ledger changed batch order or frozen sampler",
        )
        qualities = row["Terra_qualities"]
        require(
            isinstance(qualities, list)
            and len(qualities) == len(batch["training_record_indices"])
            and all(
                isinstance(q, list)
                and q
                and q[0] == 10
                and all(type(x) is int and 0 <= x <= 10 for x in q)
                for q in qualities
            ),
            "lineage ledger contains incomplete or invalid accepted qualities",
        )
        contrast = sum(len(set(q)) > 1 for q in qualities)
        examples += len(qualities)
        updates += bool(contrast)
        require(
            type(row.get("training_examples")) is int
            and type(row.get("optimizer_updates")) is int
            and row["training_examples"] == examples
            and row["optimizer_updates"] == updates
            and type(row.get("contrasting_clips")) is int
            and row["contrasting_clips"] == contrast,
            "lineage counters differ from accepted clip/update coverage",
        )
        norm = row["gradient_norm_before_clip"]
        require(
            (type(norm) in (int, float) and math.isfinite(norm) and norm > 0)
            if contrast
            else norm is None,
            "lineage applied-step gradient evidence is invalid",
        )
        rows.append(row)
    return lines, rows


def fold(segments, batches):
    require(
        isinstance(segments, list | tuple) and segments and batches,
        "lineage needs an original accepted ledger and a frozen curriculum",
    )
    selected_prefix = b""
    selected_rows = 0
    selected_examples = selected_updates = 0
    actual_examples = actual_updates = 0
    root_rollback_examples = root_rollback_updates = 0
    resets = set()
    details = []
    for index, segment in enumerate(segments):
        require(
            isinstance(segment, Segment) and type(segment.selected_rows) is int,
            "lineage segments need exact selected-row counts",
        )
        lines, rows = checked_rows(segment.ledger, batches)
        chosen = segment.selected_rows
        require(
            0 < chosen <= len(lines),
            "lineage selected prefix is empty or exceeds its actual ledger",
        )
        if index:
            require(
                chosen == len(lines), "later continuation cannot silently roll back accepted work"
            )
            require(
                len(lines) >= selected_rows and b"".join(lines[:selected_rows]) == selected_prefix,
                "continuation lost its inherited byte-exact accepted prefix",
            )
            # Repeated zero-progress attempts preserve the same durable prefix.
            if len(lines) > selected_rows:
                resets.add(selected_rows)
        final = rows[-1]
        selected = rows[chosen - 1]
        new_examples = final["training_examples"] - selected_examples
        new_updates = final["optimizer_updates"] - selected_updates
        require(new_examples >= 0 and new_updates >= 0, "lineage work counters moved backwards")
        actual_examples += new_examples
        actual_updates += new_updates
        if index == 0:
            root_rollback_examples = final["training_examples"] - selected["training_examples"]
            root_rollback_updates = final["optimizer_updates"] - selected["optimizer_updates"]
        details.append(
            {
                "inherited_rows": selected_rows,
                "actual_rows": len(lines),
                "selected_rows": chosen,
                "new_accepted_exposures": new_examples,
                "new_optimizer_updates": new_updates,
            }
        )
        selected_rows = chosen
        selected_prefix = b"".join(lines[:chosen])
        selected_examples = selected["training_examples"]
        selected_updates = selected["optimizer_updates"]
    require(
        actual_examples - selected_examples == root_rollback_examples
        and actual_updates - selected_updates == root_rollback_updates,
        "lineage lost original rollback accounting",
    )
    return {
        "selected_model_trajectory_exposures": selected_examples,
        "selected_model_optimizer_updates": selected_updates,
        "actual_total_lineage_exposures": actual_examples,
        "actual_total_lineage_optimizer_updates": actual_updates,
        "rolled_back_accepted_exposures": root_rollback_examples,
        "rolled_back_optimizer_updates": root_rollback_updates,
        "completed_batches": selected_rows,
        "resource_reset_batches_with_accepted_work": sorted(resets),
        "resource_reset_batches_if_resumed": sorted(
            resets | ({selected_rows} if selected_rows < len(batches) else set())
        ),
        "segments": details,
    }
