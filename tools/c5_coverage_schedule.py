"""Interleave training domains and signers without replacing or dropping clips."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from fractions import Fraction

from c5_all_source_training import domain


def group(row: dict) -> tuple[str, int]:
    name = domain(row)
    return name, int(row["signer"]) if name == "how2sign" else 0


def coverage_schedule(rows: list[dict], domain_mass: dict[str, float]) -> list[dict]:
    if len({(row["source"], row["id"]) for row in rows}) != len(rows):
        raise ValueError("coverage schedule input repeats an example")
    buckets = defaultdict(deque)
    for row in rows:
        buckets[group(row)].append(row)
    names = Counter(name for name, _ in buckets)
    masses = {name: Fraction(str(value)) for name, value in domain_mass.items()}
    if set(names) != set(masses) or any(value <= 0 for value in masses.values()) or sum(masses.values()) != 1:
        raise ValueError("coverage schedule needs complete normalized domain masses")
    weights = {key: masses[key[0]] / names[key[0]] for key in buckets}
    used = Counter()
    ordered = []
    while buckets:
        key = min(buckets, key=lambda item: (used[item] / weights[item], item))
        ordered.append(buckets[key].popleft())
        used[key] += 1
        if not buckets[key]:
            del buckets[key]
    return ordered


def prefix_counts(rows: list[dict], limit: int) -> dict:
    selected = rows[:limit]
    return {
        "examples": len(selected),
        "domains": dict(sorted(Counter(domain(row) for row in selected).items())),
        "how2sign_signers": dict(sorted(Counter(
            row["signer"] for row in selected if domain(row) == "how2sign"
        ).items())),
    }
