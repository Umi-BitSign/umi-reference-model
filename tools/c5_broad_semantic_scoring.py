"""Blinded scoring and family-stratified paired source statistics.

These helpers do not admit a trained candidate, decode media or send requests.
The confirmation driver must bind eligible inference and the consumed cohort.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from fractions import Fraction

from probe_jeff_terra_quality import terra_payload as original_payload
from semantic_quality_judge import calibration_controls, parse_terra_scores

FAMILIES = ("how2sign", "openasl")
RECIPE = {
    "passes": 2,
    "shuffle_seed": 20261019,
    "maximum_pairs_per_request": 16,
    "bootstrap_samples": 5000,
    "bootstrap_seed": 20261019,
    "bootstrap_unit": "paired source groups, sampled independently within each family",
    "bootstrap_percentiles": [0.025, 0.975],
    "percentile_interpolation": "linear at (sample_count-1)*quantile",
    "repeat_scores": "average both complete passes per clip before source aggregation",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def comparison_items(examples, baseline, candidate):
    require(examples and len({r["clip_id"] for r in examples}) == len(examples),
            "comparison requires unique complete clip identities")
    expected = {r["clip_id"] for r in examples}
    arms = []
    for records in (baseline, candidate):
        indexed = {r["clip_id"]: r for r in records}
        require(len(indexed) == len(records) and set(indexed) == expected,
                "comparison cannot add, omit or duplicate an arm identity")
        arms.append(indexed)
    items, unique, cases = [], {}, []
    for row in examples:
        require(row["family"] in FAMILIES and row["training_allowed"] is False
                and row["source_group"].startswith(row["family"] + ":")
                and isinstance(row["reference"], str) and row["reference"].strip(),
                "comparison changed reference, family, source or confirmation role")
        indices = []
        for arm in arms:
            prediction = arm[row["clip_id"]]
            require(all(prediction[k] == row[k] for k in
                        ("clip_id", "family", "source_group", "reference", "video"))
                    and prediction["training_allowed"] is False
                    and isinstance(prediction["hypothesis"], str),
                    "comparison arm lost its actual video/reference/source join")
            pair = row["reference"], prediction["hypothesis"]
            if pair not in unique:
                unique[pair] = len(items)
                items.append({"reference": pair[0], "candidate": pair[1]})
            indices.append(unique[pair])
        cases.append({k: row[k] for k in ("clip_id", "family", "source_group", "reference")}
                     | {"indices": indices})
    controls = []
    for control in calibration_controls():
        controls.append({"id": control["id"], "index": len(items), "range": control["range"]})
        items.append({k: control[k] for k in ("reference", "candidate")})
    result = {"items": items, "cases": cases, "controls": controls}
    validate_comparison(result)
    return result


def validate_comparison(comparison):
    items, cases, controls = (comparison[k] for k in ("items", "cases", "controls"))
    require(len(items) > 4 and cases and len(controls) == 4,
            "comparison requires clips and all four original controls")
    require(len({c["clip_id"] for c in cases}) == len(cases)
            and {c["family"] for c in cases} == set(FAMILIES),
            "comparison needs unique identities in both families")
    body = len(items) - 4
    require(all(isinstance(r["reference"], str) and r["reference"].strip()
                and isinstance(r["candidate"], str) and set(r) == {"reference", "candidate"}
                for r in items), "comparison item text changed")
    require(len({(r["reference"], r["candidate"]) for r in items[:body]}) == body,
            "identical transcript/reference pairs must share one opaque item")
    used = set()
    for case in cases:
        require(case["family"] in FAMILIES
                and case["source_group"].startswith(case["family"] + ":")
                and isinstance(case["indices"], list) and len(case["indices"]) == 2
                and all(type(i) is int and 0 <= i < body for i in case["indices"]),
                "comparison case changed its paired indices or source family")
        require(all(items[i]["reference"] == case["reference"] for i in case["indices"]),
                "comparison case reference differs from its opaque items")
        used.update(case["indices"])
    require(used == set(range(body)), "comparison contains unpaired scoring items")
    for offset, original in enumerate(calibration_controls()):
        expected = {"id": original["id"], "index": body + offset, "range": original["range"]}
        require(canonical(controls[offset]) == canonical(expected)
                and items[body + offset] == {k: original[k] for k in ("reference", "candidate")},
                "original calibration identity, text, score range or position changed")


def batches(comparison):
    validate_comparison(comparison)
    result = []
    for repeat in range(RECIPE["passes"]):
        order = list(range(len(comparison["items"])))
        random.Random(RECIPE["shuffle_seed"] + repeat).shuffle(order)
        size = RECIPE["maximum_pairs_per_request"]
        result.append([order[i:i + size] for i in range(0, len(order), size)])
    return result


def polarity_payload(rows, protocol):
    require(protocol["schema"] == "umi-studio-Terra-polarity-clarification/1"
            and protocol["model"] == "gpt-5.6-terra" and protocol["reasoning_effort"] == "medium"
            and canonical(protocol["original_controls"]) == canonical(calibration_controls()),
            "judge protocol changed its model, reasoning or original controls")
    result = original_payload(rows)
    result["instructions"] += protocol["clarification_append_only"]
    require(hashlib.sha256(result["instructions"].encode()).hexdigest()
            == protocol["full_instructions_sha256"], "complete polarity instructions changed")
    return result


def percentile(values, quantile):
    position = (len(values) - 1) * quantile
    left = math.floor(position)
    return values[left] + (values[math.ceil(position)] - values[left]) * (position - left)


def summarize(comparison, passes, wer_diagnostics=None):
    validate_comparison(comparison)
    require(isinstance(passes, list) and len(passes) == 2
            and all(len(p) == len(comparison["items"]) for p in passes),
            "semantic statistics require two complete original passes")
    require(all(type(v) is int and 0 <= v <= 10 for p in passes for v in p),
            "semantic scores must be unchanged integer0-10 grades")
    calibrated = all(c["range"][0] <= p[c["index"]] <= c["range"][1]
                     for p in passes for c in comparison["controls"])
    groups = {family: defaultdict(list) for family in FAMILIES}
    per_clip = []
    for case in comparison["cases"]:
        qualities = [sum(Fraction(p[i], 2) for p in passes) for i in case["indices"]]
        groups[case["family"]][case["source_group"]].append(qualities)
        per_clip.append({**case, "quality_original_then_candidate": list(map(float, qualities)),
                         "quality_gain": float(qualities[1] - qualities[0])})
    means, deltas = {}, {}
    for family in FAMILIES:
        sources = groups[family]
        group_means = [[sum(pair[arm] for pair in clips) / len(clips) for arm in (0, 1)]
                       for _, clips in sorted(sources.items())]
        means[family] = [sum(pair[arm] for pair in group_means) / len(group_means) for arm in (0, 1)]
        deltas[family] = [float(b - a) for a, b in group_means]
    original, candidate = [sum(means[f][arm] for f in FAMILIES) / 2 for arm in (0, 1)]
    gain = candidate - original
    minimum = max(Fraction(1, 10), original / 20)
    rng = random.Random(RECIPE["bootstrap_seed"])
    sampled = []
    for _ in range(RECIPE["bootstrap_samples"]):
        sampled.append(math.fsum(math.fsum(rng.choice(deltas[f]) for _ in deltas[f])
                                 / len(deltas[f]) for f in FAMILIES) / 2)
    sampled.sort()
    interval = [percentile(sampled, q) for q in RECIPE["bootstrap_percentiles"]]
    family_gains = {f: means[f][1] - means[f][0] for f in FAMILIES}
    family_safe = all(value >= 0 for value in family_gains.values())
    return {
        "complete_scoring": True, "clips": len(per_clip), "passes": 2,
        "recipe": RECIPE, "calibration_controls_passed_both_passes": calibrated,
        "interpretation_permitted": calibrated, "original_mean_quality": float(original),
        "candidate_mean_quality": float(candidate), "absolute_gain": float(gain),
        "relative_gain": float(gain / original) if original else None,
        "required_absolute_gain": float(minimum),
        "paired_source_group_bootstrap95_interval": interval,
        "by_family": {f: {"source_groups": len(groups[f]),
                           "clips": sum(map(len, groups[f].values())),
                           "original_mean_quality": float(means[f][0]),
                           "candidate_mean_quality": float(means[f][1]),
                           "absolute_gain": float(family_gains[f])} for f in FAMILIES},
        "no_family_mean_regression": family_safe,
        "passes_semantic_statistics": calibrated and gain >= minimum and interval[0] > 0 and family_safe,
        "per_clip": per_clip, "WER_is_diagnostic_only": True, "WER_diagnostics": wer_diagnostics,
        "raw_response_authentication_performed": False,
        "candidate_or_runtime_admission_verified": False, "promotion_authorized": False,
    }


def authenticated_passes(comparison, directory, protocol):
    """Rejoin every complete raw request/response to its predeclared opaque item."""
    def read(name):
        return json.loads((directory / name).read_text())

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    require(canonical(read("blinded-inputs.json")) == canonical(comparison), "scored blinded pairings changed")
    result = []
    for repeat, planned in enumerate(batches(comparison), start=1):
        expected = {f"pass-{repeat}-batch-{i:03d}" for i in range(len(planned))}
        require({p.name for p in directory.glob(f"pass-{repeat}-batch-*")} == expected,
                "raw scoring request coverage changed")
        joined = [None] * len(comparison["items"])
        for batch, chosen in enumerate(planned):
            path = directory / f"pass-{repeat}-batch-{batch:03d}"
            request, started, response, saved = [json.loads((path / name).read_text()) for name in
                                                ("request.json", "started.json", "response.json", "scores.json")]
            require(canonical(request) == canonical(polarity_payload([comparison["items"][i] for i in chosen], protocol))
                    and started["request_sha256"] == sha(path / "request.json")
                    and type(started["automatic_retries"]) is int and started["automatic_retries"] == 0
                    and type(started["pairs"]) is int and started["pairs"] == len(chosen)
                    and all(r.get("model") == "gpt-5.6-terra" for r in (started, response, saved))
                    and all(r.get("reasoning_effort") == "medium" for r in (started, saved))
                    and response.get("reasoning", {}).get("effort") == "medium"
                    and saved["response_sha256"] == sha(path / "response.json"),
                    "scores lost their exact prompt/model/request/response binding")
            parsed = parse_terra_scores(response, [f"pair_{i:02d}" for i in range(len(chosen))])
            scores = [parsed[f"pair_{i:02d}"] for i in range(len(chosen))]
            require(all(type(v) is int for v in saved["scores"]) and scores == saved["scores"],
                    "saved scores differ from unchanged raw grades")
            for index, score in zip(chosen, scores, strict=True):
                joined[index] = score
        require(canonical(joined) == canonical(read(f"pass-{repeat}-scores.json")), "complete pass item mapping changed")
        result.append(joined)
    return result
