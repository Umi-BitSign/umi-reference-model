from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bitsign_motion.mediapipe_mapping import (
    FACE_ANCHOR_MAPPING,
    FACE_INDICES,
    MEDIAPIPE_MAPPING_PROFILE,
    MEDIAPIPE_MAPPING_STATUS,
    MediaPipeMappingError,
    MediaPipeMappingPolicy,
    RawHolisticSequence,
    RawLandmarkTrack,
    map_holistic_to_canonical,
)
from bitsign_motion.motion_artifact import MOTION_POINT_COUNT, compose_motion


def _empty_track(frame_count: int, point_count: int) -> dict[str, np.ndarray]:
    return {
        "coordinates": np.zeros((frame_count, point_count, 3), dtype=np.float32),
        "coordinate_available": np.zeros((frame_count, point_count), dtype=np.bool_),
        "presence": np.zeros((frame_count, point_count), dtype=np.float32),
        "presence_available": np.zeros((frame_count, point_count), dtype=np.bool_),
        "visibility": np.zeros((frame_count, point_count), dtype=np.float32),
        "visibility_available": np.zeros((frame_count, point_count), dtype=np.bool_),
    }


def _track(values: dict[str, np.ndarray]) -> RawLandmarkTrack:
    return RawLandmarkTrack(**values)


def _set_coordinates(
    values: dict[str, np.ndarray],
    frame_index: int,
    points: np.ndarray,
) -> None:
    values["coordinates"][frame_index] = points
    values["coordinate_available"][frame_index] = True


def _canonical_hand(anatomical_side: str) -> np.ndarray:
    sign = np.float32(-1.0 if anatomical_side == "left" else 1.0)
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[:, 0] = sign * np.linspace(0.65, 1.05, 21, dtype=np.float32)
    hand[:, 1] = np.linspace(0.10, -0.35, 21, dtype=np.float32)
    hand[:, 2] = np.linspace(-0.08, 0.16, 21, dtype=np.float32)
    anchor_values = np.asarray(
        [
            [0.70 * sign, 0.20, 0.10],
            [0.92 * sign, 0.03, 0.31],
            [0.53 * sign, -0.16, 0.22],
            [1.02 * sign, -0.28, -0.19],
        ],
        dtype=np.float32,
    )
    hand[np.asarray([0, 4, 8, 20])] = anchor_values
    return hand


def _base_raw(
    *,
    frame_count: int = 2,
    hand_wrist_x: tuple[float, float] = (0.24, 0.76),
    missing_shoulders: bool = False,
    missing_face_anchor: bool = False,
    missing_left_hand_anchor: bool = False,
    distort_left_hand: bool = False,
    reflect_left_hand: bool = False,
    reflect_left_pose_anchor_depth: bool = False,
) -> RawHolisticSequence:
    pose_image = _empty_track(frame_count, 33)
    pose_world = _empty_track(frame_count, 33)
    hand_images = [_empty_track(frame_count, 21), _empty_track(frame_count, 21)]
    hand_worlds = [_empty_track(frame_count, 21), _empty_track(frame_count, 21)]
    face_image = _empty_track(frame_count, 478)

    left_hand = _canonical_hand("left")
    right_hand = _canonical_hand("right")
    left_pose_indices = np.asarray([15, 21, 19, 17])
    right_pose_indices = np.asarray([16, 22, 20, 18])
    hand_indices = np.asarray([0, 4, 8, 20])

    face_anchor_coordinates = {
        1: [0.50, 0.30, 0.16],
        263: [0.64, 0.26, 0.02],
        33: [0.36, 0.26, 0.02],
        291: [0.59, 0.48, 0.08],
        61: [0.41, 0.48, 0.08],
        454: [0.76, 0.37, -0.09],
        234: [0.24, 0.37, -0.09],
    }
    for frame_index in range(frame_count):
        pose_world_points = np.zeros((33, 3), dtype=np.float32)
        pose_world_points[11] = [-1.0, 0.0, 0.0]
        pose_world_points[12] = [1.0, 0.0, 0.0]
        left_pose_anchors = np.array(left_hand[hand_indices], copy=True)
        if reflect_left_pose_anchor_depth:
            center_z = np.mean(left_pose_anchors[:, 2], dtype=np.float32)
            left_pose_anchors[:, 2] = np.float32(2.0) * center_z - left_pose_anchors[:, 2]
        pose_world_points[left_pose_indices] = left_pose_anchors
        pose_world_points[right_pose_indices] = right_hand[hand_indices]
        for face_index, pose_index in FACE_ANCHOR_MAPPING:
            pose_world_points[pose_index] = face_anchor_coordinates[face_index]
        _set_coordinates(pose_world, frame_index, pose_world_points)

        pose_image_points = np.zeros((33, 3), dtype=np.float32)
        pose_image_points[15] = [0.75, 0.50, 0.0]
        pose_image_points[16] = [0.25, 0.50, 0.0]
        _set_coordinates(pose_image, frame_index, pose_image_points)

        # Observed slot 0 is the anatomical right hand; slot 1 is anatomical left.
        hand_image_0 = np.asarray(right_hand * np.float32(0.2), dtype=np.float32)
        hand_image_1 = np.asarray(left_hand * np.float32(0.2), dtype=np.float32)
        hand_image_0[:, 0] += np.float32(hand_wrist_x[0]) - hand_image_0[0, 0]
        hand_image_1[:, 0] += np.float32(hand_wrist_x[1]) - hand_image_1[0, 0]
        hand_image_0[:, 1] += np.float32(0.50) - hand_image_0[0, 1]
        hand_image_1[:, 1] += np.float32(0.50) - hand_image_1[0, 1]
        _set_coordinates(hand_images[0], frame_index, hand_image_0)
        _set_coordinates(hand_images[1], frame_index, hand_image_1)

        left_source = np.array(left_hand, copy=True)
        if distort_left_hand:
            left_source[20] += np.asarray([1.2, -0.7, 0.9], dtype=np.float32)
        if reflect_left_hand:
            center_x = np.mean(left_source[hand_indices, 0], dtype=np.float32)
            left_source[:, 0] = np.float32(2.0) * center_x - left_source[:, 0]
        _set_coordinates(hand_worlds[0], frame_index, right_hand)
        _set_coordinates(hand_worlds[1], frame_index, left_source)

        face_points = np.zeros((478, 3), dtype=np.float32)
        for point_index in FACE_INDICES:
            face_points[point_index] = np.asarray(
                [
                    np.float32(0.20 + point_index / 1000.0),
                    np.float32(0.18 + point_index / 2000.0),
                    np.float32(-0.05 + point_index / 4000.0),
                ],
                dtype=np.float32,
            )
        for face_index, coordinate in face_anchor_coordinates.items():
            face_points[face_index] = np.asarray(coordinate, dtype=np.float32)
        _set_coordinates(face_image, frame_index, face_points)

    pose_world["presence"][:] = np.float32(0.91)
    pose_world["presence_available"][:] = True
    pose_image["visibility"][:] = np.float32(0.82)
    pose_image["visibility_available"][:] = True
    for image, world in zip(hand_images, hand_worlds, strict=True):
        image["presence"][:] = np.float32(0.73)
        image["presence_available"][:] = True
        world["visibility"][:] = np.float32(0.64)
        world["visibility_available"][:] = True
    face_image["presence"][:] = np.float32(0.55)
    face_image["presence_available"][:] = True

    if missing_shoulders:
        pose_world["coordinates"][:, [11, 12]] = np.float32(0.0)
        pose_world["coordinate_available"][:, [11, 12]] = False
    if missing_face_anchor:
        face_index = FACE_ANCHOR_MAPPING[0][0]
        face_image["coordinates"][:, face_index] = np.float32(0.0)
        face_image["coordinate_available"][:, face_index] = False
    if missing_left_hand_anchor:
        hand_worlds[1]["coordinates"][:, 20] = np.float32(0.0)
        hand_worlds[1]["coordinate_available"][:, 20] = False

    return RawHolisticSequence(
        pose_image=_track(pose_image),
        pose_world=_track(pose_world),
        observed_hand_images=(_track(hand_images[0]), _track(hand_images[1])),
        observed_hand_worlds=(_track(hand_worlds[0]), _track(hand_worlds[1])),
        face_image=_track(face_image),
        timestamps_us=np.arange(frame_count, dtype=np.int64) * 125_000 + 125_000,
        clip_start_us=0,
        clip_end_us=frame_count * 125_000 + 1,
    )


def _normalized(points: np.ndarray) -> np.ndarray:
    return np.asarray(points / np.float32(2.0) * [1.0, -1.0, -1.0], dtype=np.float32)


def test_mapping_is_deterministic_assigns_swapped_hands_and_maps_face_order() -> None:
    raw = _base_raw()
    first = map_holistic_to_canonical(raw)
    second = map_holistic_to_canonical(raw)
    sequence = first.sequence

    assert first.diagnostics == second.diagnostics
    np.testing.assert_array_equal(sequence.coordinates, second.sequence.coordinates)
    np.testing.assert_array_equal(sequence.coordinate_present, second.sequence.coordinate_present)
    assert sequence.coordinates.shape == (2, MOTION_POINT_COUNT, 3)
    assert sequence.coordinates.dtype == np.float32
    assert not sequence.coordinates.flags.writeable
    assert first.diagnostics.profile == MEDIAPIPE_MAPPING_PROFILE
    assert first.diagnostics.status == MEDIAPIPE_MAPPING_STATUS
    assert first.diagnostics.policy["face_indices"] == list(FACE_INDICES)
    assert first.diagnostics.policy["face_anchor_mapping"] == [
        list(pair) for pair in FACE_ANCHOR_MAPPING
    ]
    assert first.diagnostics.frames[0].observed_hand_assignments == ("right", "left")
    assert first.diagnostics.frames[0].left_hand_fit_residual == pytest.approx(0.0, abs=1e-6)
    assert first.diagnostics.frames[0].right_hand_fit_residual == pytest.approx(0.0, abs=1e-6)
    assert first.diagnostics.frames[0].left_hand_image_world_orientation_determinant > 0.0
    assert first.diagnostics.frames[0].right_hand_image_world_orientation_determinant > 0.0

    left_source = raw.observed_hand_worlds[1].coordinates[0]
    right_source = raw.observed_hand_worlds[0].coordinates[0]
    np.testing.assert_allclose(sequence.coordinates[0, 33:54], _normalized(left_source), atol=1e-6)
    np.testing.assert_allclose(sequence.coordinates[0, 54:75], _normalized(right_source), atol=1e-6)
    selected_face = raw.face_image.coordinates[0, np.asarray(FACE_INDICES)]
    np.testing.assert_allclose(
        sequence.coordinates[0, 75:107], _normalized(selected_face), atol=1e-6
    )
    np.testing.assert_array_equal(sequence.track_present, np.ones((2, 4), dtype=np.bool_))
    assert not sequence.motion_boundary.any()

    # World scores take precedence and image scores fill only unavailable world scores.
    np.testing.assert_array_equal(sequence.presence[0, :33], np.full(33, 0.91, np.float32))
    np.testing.assert_array_equal(sequence.visibility[0, :33], np.full(33, 0.82, np.float32))
    np.testing.assert_array_equal(sequence.presence[0, 33:75], np.full(42, 0.73, np.float32))
    np.testing.assert_array_equal(sequence.visibility[0, 33:75], np.full(42, 0.64, np.float32))
    np.testing.assert_array_equal(sequence.presence[0, 75:107], np.full(32, 0.55, np.float32))
    assert sequence.presence_available[0].all()
    assert sequence.visibility_available[0, :75].all()
    assert not sequence.visibility_available[0, 75:107].any()

    # The result already satisfies the real fixed-tensor composer's strict contract.
    composed = compose_motion(sequence)
    assert composed.motion.shape == (120, 1184)
    assert composed.frame_mask.tolist() == [1, 1] + [0] * 118


def test_equal_hand_costs_are_ambiguous_and_fail_closed() -> None:
    result = map_holistic_to_canonical(_base_raw(hand_wrist_x=(0.50, 0.50)))

    assert result.diagnostics.ambiguous_hand_frame_count == 2
    assert result.diagnostics.frames[0].observed_hand_assignments == (
        "ambiguous",
        "ambiguous",
    )
    assert "ambiguous-hand-assignment" in result.diagnostics.frames[0].reasons
    assert not result.sequence.coordinate_present[:, 33:75].any()
    assert not result.sequence.track_present[:, 1:3].any()
    assert np.all(result.sequence.coordinates[:, 33:75] == 0.0)
    assert not np.signbit(result.sequence.coordinates[:, 33:75]).any()


def test_missing_torso_fails_every_coordinate_track_closed_but_preserves_pose_scores() -> None:
    result = map_holistic_to_canonical(_base_raw(missing_shoulders=True))

    assert result.diagnostics.invalid_torso_frame_count == 2
    assert not result.sequence.coordinate_present.any()
    assert not result.sequence.track_present.any()
    assert not result.sequence.motion_boundary.any()
    assert result.sequence.presence_available[:, :33].all()
    np.testing.assert_array_equal(
        result.sequence.presence[:, :33], np.full((2, 33), 0.91, dtype=np.float32)
    )
    assert np.all(result.sequence.coordinates == 0.0)
    assert not np.signbit(result.sequence.coordinates).any()


def test_bad_hand_fit_residual_rejects_coordinates_but_retains_assigned_scores() -> None:
    policy = replace(
        MediaPipeMappingPolicy(),
        maximum_hand_fit_residual=0.01,
        reflection_residual_epsilon=1.0,
    )
    result = map_holistic_to_canonical(
        _base_raw(distort_left_hand=True),
        policy,
    )

    assert result.diagnostics.rejected_hand_fit_count == 2
    assert "left-hand-fit-residual" in result.diagnostics.frames[0].reasons
    assert not result.sequence.coordinate_present[:, 33:54].any()
    assert result.sequence.presence_available[:, 33:54].all()
    np.testing.assert_array_equal(
        result.sequence.presence[:, 33:54], np.full((2, 21), 0.73, dtype=np.float32)
    )
    assert result.sequence.coordinate_present[:, 54:75].all()


def test_image_world_reflection_is_rejected_before_pose_fit() -> None:
    result = map_holistic_to_canonical(_base_raw(reflect_left_hand=True))

    assert result.diagnostics.rejected_hand_fit_count == 2
    assert "left-hand-image-world-reflection" in result.diagnostics.frames[0].reasons
    assert not result.sequence.coordinate_present[:, 33:54].any()
    assert result.sequence.coordinate_present[:, 54:75].all()


def test_noisy_pose_depth_chirality_uses_proper_fit_without_reflection() -> None:
    result = map_holistic_to_canonical(_base_raw(reflect_left_pose_anchor_depth=True))

    frame = result.diagnostics.frames[0]
    assert frame.left_hand_reflection_alternative_lower_residual is True
    assert frame.left_hand_image_world_orientation_determinant > 0.0
    assert frame.left_hand_fit_residual is not None
    assert frame.left_hand_fit_residual < MediaPipeMappingPolicy().maximum_hand_fit_residual
    assert result.sequence.coordinate_present[:, 33:54].all()
    assert result.diagnostics.rejected_hand_fit_count == 0


def test_missing_declared_face_anchor_rejects_face_only_and_preserves_scores() -> None:
    result = map_holistic_to_canonical(_base_raw(missing_face_anchor=True))

    assert "face-missing-fit-anchors" in result.diagnostics.frames[0].reasons
    assert not result.sequence.coordinate_present[:, 75:107].any()
    assert not result.sequence.track_present[:, 3].any()
    assert result.sequence.presence_available[:, 75:107].all()
    np.testing.assert_array_equal(
        result.sequence.presence[:, 75:107], np.full((2, 32), 0.55, dtype=np.float32)
    )
    assert result.sequence.coordinate_present[:, :75].all()


def test_missing_hand_fit_anchor_rejects_hand_only_and_preserves_scores() -> None:
    result = map_holistic_to_canonical(_base_raw(missing_left_hand_anchor=True))

    assert "left-hand-missing-world-fit-anchors" in result.diagnostics.frames[0].reasons
    assert not result.sequence.coordinate_present[:, 33:54].any()
    assert not result.sequence.track_present[:, 1].any()
    assert result.sequence.presence_available[:, 33:54].all()
    np.testing.assert_array_equal(
        result.sequence.presence[:, 33:54], np.full((2, 21), 0.73, dtype=np.float32)
    )
    assert result.sequence.coordinate_present[:, 54:107].all()


def test_raw_contract_rejects_nonfinite_bad_scores_and_negative_zero() -> None:
    values = _empty_track(1, 1)
    values["coordinates"][0, 0, 0] = np.float32(np.nan)
    with pytest.raises(MediaPipeMappingError, match="finite float32"):
        _track(values)

    values = _empty_track(1, 1)
    values["presence_available"][0, 0] = True
    values["presence"][0, 0] = np.float32(1.01)
    with pytest.raises(MediaPipeMappingError, match=r"presence must be in \[0,1\]"):
        _track(values)

    values = _empty_track(1, 1)
    values["coordinates"][0, 0, 0] = np.float32(-0.0)
    with pytest.raises(MediaPipeMappingError, match="positive zero"):
        _track(values)


def test_sequence_contract_rejects_duplicate_semantic_timestamps() -> None:
    raw = _base_raw()
    with pytest.raises(MediaPipeMappingError, match="strictly increasing"):
        RawHolisticSequence(
            pose_image=raw.pose_image,
            pose_world=raw.pose_world,
            observed_hand_images=raw.observed_hand_images,
            observed_hand_worlds=raw.observed_hand_worlds,
            face_image=raw.face_image,
            timestamps_us=np.asarray([125_000, 125_000], dtype=np.int64),
            clip_start_us=0,
            clip_end_us=250_001,
        )
