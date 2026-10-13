import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from generate_c5_text_corrector_pairs import load_frozen_schedule


def test_loads_training_only_schedule():
    plan, schedule = load_frozen_schedule(
        Path("runs/c5-anchored-projection-v1/plan.json")
    )
    assert len(schedule) == 1_024
    assert plan["recipe"]["train_examples"] == len(schedule)
    assert {row["source"] for row in schedule} == {
        "asl_stem_original",
        "asl_stem_supplemental",
        "fleurs",
        "how2sign_new",
        "how2sign_original",
        "how2sign_precut_expansion",
        "how2sign_raw",
        "openasl",
    }
    assert all(row["reference"].strip() for row in schedule)
    assert all(row.get("split", "training") == "training" for row in schedule)
