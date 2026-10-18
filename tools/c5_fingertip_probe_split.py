"""Group retained probe clips before fitting, without inspecting feature targets.

Known signer IDs are shared across all STEM tranches. OpenASL's retained signer
field is a source-video identifier, so its split supports source disjointness.
It cannot establish that people in separate OpenASL videos are different.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

STEM = {"asl_stem_word", "asl_stem_original", "asl_stem_supplemental"}
EXCLUDED = {"P10", "P13", "P14", "P21", "P22", "P31", "P4", "P6", "P0", "P19"}
ROLES = ("fit", "validation", "confirmation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(record):
    return record["source"], record["example_id"]


def units(record):
    source, signer = record["source"], str(record["signer"])
    group, example = record["source_group"], record["example_id"]
    require(isinstance(group, str) and group, "missing source group")
    require(isinstance(example, str) and example, "missing example identity")
    if source in STEM:
        family, basis = "asl_stem", "known_signer"
        require(signer.startswith("P") and signer[1:].isdigit(), "unknown STEM signer")
        require(signer not in EXCLUDED, "reserved or diagnostic signer")
        require(group.startswith(f"asl_stem:{signer}:"), "STEM signer/group mismatch")
        main = (family, "known_signer", signer)
    elif source == "how2sign_precut_expansion":
        family, basis = "how2sign", "known_signer"
        require(signer.isdigit(), "unknown How2Sign signer")
        require(group.startswith("how2sign:"), "unknown How2Sign source group")
        main = (family, "known_signer", signer)
    else:
        require(source == "openasl_scale", "undeclared source identity convention")
        family, basis = "openasl", "source_video"
        require(group.startswith("openasl:"), "unknown OpenASL source group")
        main = (family, "source_video", group)
    video_sha = record["canonical_video_sha256"]
    require(
        isinstance(video_sha, str)
        and len(video_sha) == 64
        and all(character in "0123456789abcdef" for character in video_sha),
        "missing canonical-video binding",
    )
    return (
        family,
        basis,
        (
            main,
            (family, "source_group", group),
            (family, "example_id", example),
            ("canonical_video_sha256", video_sha),
        ),
    )


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def split(records, seed=20261017):
    """Allocate connected identity groups with a fixed, outcome-independent seed.

    Within each source-family stratum, about 60/20/20 percent of components go
    to fit/validation/confirmation. Strata with fewer than three components stay
    fit-only. Clip and class balance are reported by callers, never optimized.
    """
    require(type(seed) is int, "seed must be an integer")
    keys = [identity(record) for record in records]
    require(len(keys) == len(set(keys)) > 0, "duplicate or empty input inventory")
    declared = [units(record) for record in records]
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, (_, _, group_units) in enumerate(declared):
        for unit in group_units:
            if unit in seen:
                parents[root(index)] = root(seen[unit])
            else:
                seen[unit] = index
    joined = defaultdict(list)
    for index in range(len(records)):
        joined[root(index)].append(index)
    components, strata = [], defaultdict(list)
    for indices in joined.values():
        members = sorted(keys[index] for index in indices)
        families = tuple(sorted({declared[index][0] for index in indices}))
        component = {
            "component_id": stable_hash(members),
            "families": list(families),
            "record_indices": sorted(indices),
            "identity_units": sorted({unit for index in indices for unit in declared[index][2]}),
        }
        strata[families].append(component)
    for families in sorted(strata):
        entries = sorted(
            strata[families], key=lambda item: stable_hash([seed, item["component_id"]])
        )
        holdout = max(1, len(entries) // 5) if len(entries) >= 3 else 0
        fit_end, validation_end = len(entries) - 2 * holdout, len(entries) - holdout
        for position, component in enumerate(entries):
            component["role"] = (
                "fit"
                if position < fit_end
                else "validation"
                if position < validation_end
                else "confirmation"
            )
            components.append(component)
    assignments = []
    for component in sorted(components, key=lambda item: item["component_id"]):
        for index in component["record_indices"]:
            assignments.append(
                {
                    "record_index": index,
                    "source": records[index]["source"],
                    "example_id": records[index]["example_id"],
                    "family": declared[index][0],
                    "disjointness_basis": declared[index][1],
                    "component_id": component["component_id"],
                    "role": component["role"],
                }
            )
    return {
        "schema": "umi-studio-fingertip-auxiliary-probe-split/1",
        "seed": seed,
        "components": components,
        "assignments": sorted(assignments, key=lambda item: item["record_index"]),
        "known_signer_disjointness_scope": ["asl_stem", "how2sign"],
        "openasl_person_disjointness_proven": False,
        "source_group_and_exact_canonical_bytes_disjoint": True,
        "target_or_outcome_fields_used": False,
        "original_curriculum_or_primary_holdouts_changed": False,
    }
