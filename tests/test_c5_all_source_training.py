import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from c5_all_source_training import RECIPE, domain, learning_rate_factor, objective_weights


def test_dataset_and_signer_weights_are_normalized():
    rows = [
        {"source": "how2sign_new", "signer": signer}
        for signer in (1, 2, 3, 4, 5, 8, 9, 11)
        for _ in range(signer)
    ] + ([{"source": "fleurs", "signer": 3}] * 3) + ([{"source": "asl_stem_original", "signer": "p"}] * 7) + [{"source": "openasl_scale", "signer": "s"}]
    weights = objective_weights(rows)
    totals = defaultdict(float)
    signer_totals = defaultdict(float)
    for row, weight in zip(rows, weights):
        totals[domain(row)] += weight / len(rows)
        if domain(row) == "how2sign":
            signer_totals[row["signer"]] += weight / len(rows)
    assert dict(totals) == pytest.approx(RECIPE["domain_mass"])
    assert all(mass == pytest.approx(0.6 / 8) for mass in signer_totals.values())
    assert sum(weights) == pytest.approx(len(rows))


def test_cosine_schedule_warms_up_and_respects_floor():
    assert learning_rate_factor(0, 1933) == 0.01
    assert learning_rate_factor(99, 1933) == 1.0
    assert learning_rate_factor(100, 1933) == 1.0
    assert 0.1 < learning_rate_factor(1000, 1933) < 1.0
    assert learning_rate_factor(1933, 1933) == pytest.approx(0.1)
