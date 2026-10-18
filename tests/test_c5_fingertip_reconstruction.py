from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_fingertip_probe_data import tip_continuity
from c5_fingertip_reconstruction import (
    METHODS,
    balanced_weights,
    features,
    fit,
    predict,
)


def tips_fixture(frames=19):
    return np.random.default_rng(23).normal(0.4, 0.03, (frames, 2, 5, 2))


@pytest.mark.parametrize("method", METHODS)
def test_translation_equivariance_and_finite_predictions(method):
    tips = tips_fixture()
    continuity = tip_continuity(tips)
    first = features(tips, continuity, method)
    shift = np.array([0.7, -0.6])
    moved = features(tips + shift, continuity, method)
    np.testing.assert_allclose(first["x"], moved["x"], atol=1e-12)
    np.testing.assert_allclose(predict(moved) - predict(first), np.broadcast_to(shift, tips.shape))


def test_missing_hand_and_reset_do_not_enter_temporal_context():
    tips = tips_fixture()
    tips[6, 0] = np.nan
    continuity = tip_continuity(tips)
    continuity[11, 0] = False
    first = features(tips, continuity, "temporal_normalized")
    changed = tips.copy()
    changed[:7, 0] = 100
    changed[11:, 0] = -100
    second = features(changed, continuity, "temporal_normalized")
    np.testing.assert_array_equal(first["x"][9 * 2], second["x"][9 * 2])
    assert np.isnan(predict(first)[6, 0]).all()
    assert np.isfinite(predict(first)[6, 1]).all()


def test_bimanual_missingness_is_explicit_and_never_removes_current_hand():
    tips = tips_fixture()
    tips[:, 1] = np.nan
    view = features(tips, tip_continuity(tips), "bimanual_normalized")
    assert (view["x"][:, 10:] == 0).all()
    assert np.isfinite(predict(view)[:, 0]).all()
    assert np.isnan(predict(view)[:, 1]).all()


@pytest.mark.parametrize("method", METHODS[2:])
def test_normalized_features_preserve_positive_scale_equivariance(method):
    tips = tips_fixture()
    continuity = tip_continuity(tips)
    first = features(tips, continuity, method)
    scaled = features(tips * 3, continuity, method)
    np.testing.assert_allclose(first["x"], scaled["x"], atol=1e-12)
    np.testing.assert_allclose(first["scale"] * 3, scaled["scale"])


def test_balanced_weights_assign_equal_mass_to_family_group_clip():
    rows = [
        {"family": "a", "group": "1"},
        {"family": "a", "group": "1"},
        {"family": "a", "group": "2"},
        {"family": "b", "group": "3"},
        {"family": "b", "group": "empty"},
    ]
    masks = [np.ones(count, bool) for count in (2, 20, 100, 1, 0)]
    masses = [weights.sum() for weights in balanced_weights(rows, masks)]
    np.testing.assert_allclose(masses, [0.125, 0.125, 0.25, 0.5, 0])


def test_ridge_recovers_unseen_affine_geometry_and_refuses_validation_fit():
    tips = tips_fixture(200)
    view = features(tips, tip_continuity(tips), "centered_linear")
    targets = tips.reshape(-1, 5, 2).copy()
    targets[:, 0] = targets[:, 1] * 0.75 + targets[:, 4] * 0.25
    row = {
        "features": view,
        "targets": targets,
        "target_valid": np.ones((400, 5), bool),
        "family": "a",
        "group": "one",
        "role": "fit",
    }
    model = fit([row], ridge=1e-8)
    test = tips_fixture(30) + 0.15
    expected = test.copy()
    expected[:, :, 0] = test[:, :, 1] * 0.75 + test[:, :, 4] * 0.25
    np.testing.assert_allclose(
        predict(features(test, tip_continuity(test), "centered_linear"), model),
        expected,
        atol=1e-7,
    )
    row["role"] = "validation"
    with pytest.raises(ValueError, match="fit-only"):
        fit([row])


def test_target_masks_only_affect_their_own_fitted_joint():
    tips = tips_fixture(100)
    row = {
        "features": features(tips, tip_continuity(tips), "normalized_linear"),
        "targets": tips.reshape(-1, 5, 2).copy(),
        "target_valid": np.ones((200, 5), bool),
        "family": "a",
        "group": "one",
        "role": "fit",
    }
    first = fit([row])
    row["target_valid"][:100, 0] = False
    row["targets"][:100, 0] = np.nan
    second = fit([row])
    assert second["count_0"] == 100
    for joint in range(1, 5):
        np.testing.assert_array_equal(first[f"coef_{joint}"], second[f"coef_{joint}"])


def test_degenerate_tip_cloud_has_finite_features():
    tips = np.full((5, 2, 5, 2), 0.25)
    for method in METHODS:
        view = features(tips, tip_continuity(tips), method)
        assert np.isfinite(view["x"]).all()
        np.testing.assert_allclose(predict(view), tips)
