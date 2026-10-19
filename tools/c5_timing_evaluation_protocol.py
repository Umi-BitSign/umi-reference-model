"""Full reserved timing comparison data and unchanged semantic gate.

Authenticates original public controls as data. Rejected historical candidates
and their weights are never loaded or admitted by this module.
"""

from __future__ import annotations

from c5_timing_development_input import development_row, expected_header
from c5_timing_feature_data import ROOT, artifact, bound, read, require, sha, terminal

WORD_DATA = ROOT / "runs/terra-word-native112-data-v1/plan.json"
WORD_RESULT = ROOT / "checkpoints/terra-word-native112-data-v1/data.json"
WORD_PUBLIC = ROOT / "runs/c5-public112-word-diagnostic-v1/plan.json"
SENTENCE_INPUTS = ROOT / "runs/roi-short-translation-evaluation-v1/plan.json"
SENTENCE_NATIVE = ROOT / "runs/c5-native112-evaluation-v1/plan.json"


def inventory(model_plan):
    from c5_word_evaluation_io import development_records
    from evaluate_roi_short_translation import checked as checked_sentences
    from evaluate_roi_short_translation import prediction
    from terra_training_recovery import verify_snapshot
    from terra_word_verified_inputs import checked_result

    words = development_records(checked_result(WORD_DATA, WORD_RESULT))
    sentences = checked_sentences(SENTENCE_INPUTS)
    native = read(SENTENCE_NATIVE)
    verify_snapshot(SENTENCE_NATIVE, native)
    public = read(WORD_PUBLIC)
    verify_snapshot(WORD_PUBLIC, public)
    require(
        public["records"] == words
        and native["records"] == sentences["records"]
        and public["reference"]
        == sentences["reference"]
        == native["reference"]
        == model_plan["reference"]
        and public["initial_reference_hashes"]
        == sentences["initial_reference_hashes"]
        == native["initial_reference_hashes"]
        == model_plan["initial_reference_hashes"]
        and public["candidate_visual_state_sha256"] == model_plan["pretrained_state_sha256"]
        and public["trained_or_partial_visual_checkpoint_installed_for_inference"] is False,
        "timing comparison metadata lost exact original public C5 values or full reserved inputs",
    )
    baseline_path = bound(public["baseline_control"])
    original_plan_path = bound(public["original_evaluation_plan"])
    original = read(original_plan_path)
    require(
        original["records"] == words
        and original["reference"] == model_plan["reference"]
        and original["initial_reference_hashes"] == model_plan["initial_reference_hashes"],
        "timing original word control metadata changed its source inventory or reference",
    )
    baseline = read(baseline_path)
    word_authority = terminal(baseline_path.parent)
    require(
        baseline["complete"] is True
        and baseline["clips"] == 96
        and baseline["plan_sha256"] == sha(original_plan_path)
        and baseline["all_original_C5_values_unchanged"] is True
        and baseline["original_visual_state_sha256"] == model_plan["pretrained_state_sha256"]
        and baseline["optimizer_updates"] == baseline["new_Terra_requests"] == 0
        and len(baseline["baseline_files"]) == len(baseline["predictions"]) == 96,
        "timing original word controls require complete independent public224 inference",
    )
    assets = [
        artifact(p) for p in (WORD_DATA, WORD_RESULT, WORD_PUBLIC, SENTENCE_INPUTS, SENTENCE_NATIVE)
    ]
    assets += [artifact(baseline_path), artifact(original_plan_path), *word_authority.values()]
    cases, controls, predictions = [], [], []
    for index, (case, descriptor) in enumerate(zip(words, baseline["baseline_files"], strict=True)):
        path = bound(descriptor)
        saved = read(path)
        wrapped = {"kind": "word", "word": case}
        row = development_row(wrapped)
        expected_header(wrapped)
        expected = prediction(row, saved["hypothesis"])
        require(
            path == baseline_path.with_name(f"baseline-{index:04d}.json")
            and all(saved[k] == v for k, v in expected.items())
            and saved["input_receipt"] == case["receipt"]
            and saved["native_input_side"] == 224
            and saved["native_crop_parity"] is True
            and isinstance(saved["tokens"], list)
            and 1 <= len(saved["tokens"]) <= 2048
            and all(type(v) is int and 0 <= v < 384 for v in saved["tokens"]),
            "timing original word control lost full native token/input/role evidence",
        )
        cases.append(wrapped)
        predictions.append(expected)
        controls.append(
            {
                "row": row,
                "hypothesis": saved["hypothesis"],
                "tokens": saved["tokens"],
                "evidence": descriptor,
                "token_identity_required": True,
            }
        )
        assets.append(descriptor)
    require(
        predictions == baseline["predictions"], "timing word controls differ from original receipt"
    )
    sentence_control_path = bound(sentences["control"])
    sentence_control = read(sentence_control_path)
    sentence_authority = terminal(sentence_control_path.parent)
    curriculum_path = bound(sentences["curriculum"])
    curriculum = read(curriculum_path)
    require(
        sentence_control["complete"] is True
        and sentence_control["clips"] == 34
        and sentence_control["plan_sha256"] == sha(curriculum_path)
        and curriculum["reference"] == model_plan["reference"]
        and sentence_control["original_C5_generation"] == model_plan["reference"]["generation"]
        and sentence_control["all_weights_unchanged"] is True
        and sentence_control["optimizer_updates"] == sentence_control["new_Terra_requests"] == 0,
        "timing original sentence controls require complete unchanged C5 inference",
    )
    indexed = {(r["source"], r["example_id"]): r for r in sentence_control["predictions"]}
    require(
        len(indexed) == len(sentence_control["predictions"]) == 34,
        "timing sentence control identities repeated",
    )
    for case in sentences["records"]:
        wrapped = {"kind": "sentence", "sentence": case}
        row = development_row(wrapped)
        expected_header(wrapped)
        saved = indexed[(row["source"], row["example_id"])]
        require(
            all(saved[k] == v for k, v in prediction(row, saved["hypothesis"]).items()),
            "timing sentence control changed its exact source/role/reference",
        )
        cases.append(wrapped)
        controls.append(
            {
                "row": row,
                "hypothesis": saved["hypothesis"],
                "tokens": None,
                "evidence": artifact(sentence_control_path),
                "token_identity_required": False,
            }
        )
    assets += [
        artifact(sentence_control_path),
        artifact(curriculum_path),
        *sentence_authority.values(),
    ]
    identities = [(c["row"]["source"], c["row"]["example_id"]) for c in controls]
    require(
        len(cases) == len(controls) == len(set(identities)) == 130,
        "timing comparison dropped or repeated an input",
    )
    return {
        "cases": cases,
        "controls": controls,
        "assets": assets,
        "source_sha256": {**public["source_sha256"], **sentences["source_sha256"]},
        "counts": {"word": 96, "sentence": 34},
        "sampled_frames": {"word": 4828, "sentence": 3077},
        "historical_trained_candidate_admitted": False,
        "confirmation_consumed": False,
    }


def compare(kind, rows, baseline, predictions):
    from terra_short_curriculum_learning import comparison_items

    require(
        kind in ("word", "sentence"), "timing quality comparison requires a complete named family"
    )
    count = 96 if kind == "word" else 34
    require(
        len(rows) == len(baseline) == len(predictions) == count,
        "timing quality cannot score a partial family",
    )
    return comparison_items(rows, baseline, predictions)


def summarize(kind, comparison, scores, baseline, predictions):
    from terra_short_curriculum_learning import semantic_decision

    require(
        kind in ("word", "sentence") and len(comparison["cases"]) == (96 if kind == "word" else 34),
        "timing quality decision cannot use partial or substituted development coverage",
    )
    result = semantic_decision(comparison, scores, baseline, predictions)
    return {
        **result,
        "scope": f"complete_reserved_{kind}_timing_comparison",
        "candidate_kind": "four_parameter_native224_timing",
        "word_recognition_is_continuous_translation_evidence": False,
        "independent_confirmation_complete": False,
        "promotion_authorized": False,
    }


def both_families(word, sentence):
    require(
        word["clips"] == 96 and sentence["clips"] == 34,
        "timing overall gate requires both complete families",
    )
    return {
        "word": word,
        "sentence": sentence,
        "passes_both_semantic_research_gates": all(
            result["passes_semantic_research_gate"]
            and result.get("complete_scoring") is True
            and result.get("calibration_passed") is True
            for result in (word, sentence)
        ),
        "independent_confirmation_required": True,
        "promotion_authorized": False,
    }
