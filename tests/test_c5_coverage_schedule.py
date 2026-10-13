import copy
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_coverage_schedule import coverage_schedule, group
from c5_retention import COVERAGE_RECIPE, RECIPE as CONTROL_RECIPE
from run_c5_stem_projection_frozen_tuning import tuning_config

MASS = {"how2sign": 0.6, "fleurs": 0.2, "asl_stem": 0.15, "openasl": 0.05}


def fixture_rows():
    rows = []
    for source, signer, count in [("how2sign_original", 8, 800), ("how2sign_raw", 1, 200), ("fleurs", 0, 400), ("asl_stem_original", 0, 400), ("openasl_scale", 0, 100)]:
        rows.extend({"source": source, "id": f"{signer}-{i}", "signer": signer, "loss_weight": i + 1} for i in range(count))
    return rows


def test_balances_early_coverage_preserving_every_row_and_group_order():
    original = fixture_rows()
    ordered = coverage_schedule(original, MASS)
    assert len(ordered) == len(original)
    assert {id(row) for row in ordered} == {id(row) for row in original}
    assert ordered == coverage_schedule(original, MASS)
    counts = Counter(group(row) for row in ordered[:200])
    assert counts == {("how2sign", 1): 60, ("how2sign", 8): 60, ("fleurs", 0): 40, ("asl_stem", 0): 30, ("openasl", 0): 10}
    for key in {group(row) for row in original}:
        assert [row for row in ordered if group(row) == key] == [row for row in original if group(row) == key]


def test_rejects_duplicates_and_incomplete_domains():
    original = fixture_rows()
    with pytest.raises(ValueError, match="repeats"):
        coverage_schedule(original + [original[0]], MASS)
    with pytest.raises(ValueError, match="normalized"):
        coverage_schedule(original, {**MASS, "fleurs": 0.3})
    with pytest.raises(ValueError, match="normalized"):
        coverage_schedule([row for row in original if row["source"] != "fleurs"], MASS)


def test_coverage_recipe_changes_only_schedule_policy():
    comparable = {key: value for key, value in COVERAGE_RECIPE.items() if key not in {"id", "schedule_policy"}}
    assert comparable == {key: value for key, value in CONTROL_RECIPE.items() if key != "id"}
    plan = {"schema": "umi-studio-c5-frozen-replay-plan/1", "recipe": copy.deepcopy(COVERAGE_RECIPE)}
    assert tuning_config(plan)[0] == COVERAGE_RECIPE
    plan["recipe"]["schedule_policy"]["loss_weights_unchanged"] = False
    with pytest.raises(ValueError, match="recipe changed"):
        tuning_config(plan)
