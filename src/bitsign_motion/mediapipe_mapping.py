from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .motion_artifact import (
    MOTION_FRAME_COUNT,
    MOTION_POINT_COUNT,
    CanonicalPointSequence,
)

MEDIAPIPE_MAPPING_PROFILE: Final = "normalized-monocular-3d/2-ex203-candidate"
MEDIAPIPE_MAPPING_STATUS: Final = "candidate-pending-ex203-real-and-cross-runtime-study"

POSE_POINT_COUNT: Final = 33
HAND_POINT_COUNT: Final = 21
FACE_POINT_COUNT: Final = 478

FACE_INDICES: Final = (
    0,
    1,
    4,
    10,
    13,
    14,
    17,
    33,
    61,
    63,
    66,
    70,
    78,
    105,
    107,
    133,
    145,
    152,
    159,
    234,
    263,
    291,
    293,
    296,
    300,
    308,
    334,
    336,
    362,
    374,
    386,
    454,
)

# MediaPipe face indices are paired with BlazePose anatomical landmarks. The two
# face-oval points stand in for the ears because the face task has no ear landmark.
FACE_ANCHOR_MAPPING: Final = (
    (1, 0),  # nose tip -> nose
    (263, 2),  # anatomical left eye -> left eye
    (33, 5),  # anatomical right eye -> right eye
    (291, 9),  # anatomical left mouth corner -> mouth left
    (61, 10),  # anatomical right mouth corner -> mouth right
    (454, 7),  # anatomical left face oval -> left ear
    (234, 8),  # anatomical right face oval -> right ear
)

HAND_ANCHOR_INDICES: Final = (0, 4, 8, 20)
LEFT_POSE_HAND_ANCHORS: Final = (15, 21, 19, 17)
RIGHT_POSE_HAND_ANCHORS: Final = (16, 22, 20, 18)

POSE_LEFT_SHOULDER: Final = 11
POSE_RIGHT_SHOULDER: Final = 12
POSE_LEFT_WRIST: Final = 15
POSE_RIGHT_WRIST: Final = 16

# MediaPipe world coordinates are camera relative. This candidate keeps image-right
# positive, changes image-down to up, and changes camera-away to camera-near.
CAMERA_RELATIVE_AXIS_SIGNS: Final = (1.0, -1.0, -1.0)

_POSITIVE_ZERO32 = np.float32(0.0)

Float32Array = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
Int64Array = NDArray[np.int64]


class MediaPipeMappingError(ValueError):
    """Raised when raw Holistic observations violate the candidate contract."""


def _positive_zero(array: np.ndarray) -> bool:
    return bool(np.all(array == 0) and not np.signbit(array).any())


@dataclass(frozen=True, slots=True, repr=False)
class RawLandmarkTrack:
    """One strict MediaPipe landmark track with explicit optional-score masks."""

    coordinates: Float32Array
    coordinate_available: BoolArray
    presence: Float32Array
    presence_available: BoolArray
    visibility: Float32Array
    visibility_available: BoolArray

    @property
    def frame_count(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.coordinates.shape[1])

    def __repr__(self) -> str:
        return f"RawLandmarkTrack(frame_count={self.frame_count}, point_count={self.point_count})"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.coordinates, np.ndarray)
            or self.coordinates.dtype != np.float32
            or self.coordinates.ndim != 3
            or self.coordinates.shape[2] != 3
            or self.coordinates.shape[0] < 1
            or self.coordinates.shape[1] < 1
            or not np.isfinite(self.coordinates).all()
        ):
            raise MediaPipeMappingError(
                "track coordinates must be finite float32 [frames,points,3]"
            )
        shape = self.coordinates.shape[:2]
        if (
            not isinstance(self.coordinate_available, np.ndarray)
            or self.coordinate_available.dtype != np.bool_
            or self.coordinate_available.shape != shape
        ):
            raise MediaPipeMappingError("track coordinate_available must be bool [frames,points]")
        if not _positive_zero(self.coordinates[~self.coordinate_available]):
            raise MediaPipeMappingError(
                "unavailable track coordinates must be explicit positive zero"
            )
        for name, values, available in (
            ("presence", self.presence, self.presence_available),
            ("visibility", self.visibility, self.visibility_available),
        ):
            if (
                not isinstance(values, np.ndarray)
                or values.dtype != np.float32
                or values.shape != shape
                or not np.isfinite(values).all()
            ):
                raise MediaPipeMappingError(f"track {name} must be finite float32 [frames,points]")
            if (
                not isinstance(available, np.ndarray)
                or available.dtype != np.bool_
                or available.shape != shape
            ):
                raise MediaPipeMappingError(f"track {name}_available must be bool [frames,points]")
            if not _positive_zero(values[~available]):
                raise MediaPipeMappingError(
                    f"unavailable track {name} must be explicit positive zero"
                )
            if np.any(values[available] < np.float32(0.0)) or np.any(
                values[available] > np.float32(1.0)
            ):
                raise MediaPipeMappingError(f"available track {name} must be in [0,1]")


@dataclass(frozen=True, slots=True, repr=False)
class RawHolisticSequence:
    """Selected synchronous Holistic frames before anatomical normalization."""

    pose_image: RawLandmarkTrack
    pose_world: RawLandmarkTrack
    observed_hand_images: tuple[RawLandmarkTrack, RawLandmarkTrack]
    observed_hand_worlds: tuple[RawLandmarkTrack, RawLandmarkTrack]
    face_image: RawLandmarkTrack
    timestamps_us: Int64Array
    clip_start_us: int
    clip_end_us: int

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_us.size)

    def __repr__(self) -> str:
        return (
            "RawHolisticSequence("
            f"frame_count={self.frame_count}, clip_start_us={self.clip_start_us}, "
            f"clip_end_us={self.clip_end_us})"
        )

    def __post_init__(self) -> None:
        if not isinstance(self.pose_image, RawLandmarkTrack) or not isinstance(
            self.pose_world, RawLandmarkTrack
        ):
            raise TypeError("pose tracks must be RawLandmarkTrack values")
        if not isinstance(self.face_image, RawLandmarkTrack):
            raise TypeError("face_image must be a RawLandmarkTrack")
        for name, tracks in (
            ("observed_hand_images", self.observed_hand_images),
            ("observed_hand_worlds", self.observed_hand_worlds),
        ):
            if (
                type(tracks) is not tuple
                or len(tracks) != 2
                or not all(isinstance(track, RawLandmarkTrack) for track in tracks)
            ):
                raise MediaPipeMappingError(
                    f"{name} must contain exactly two RawLandmarkTrack values"
                )
        if (
            not isinstance(self.timestamps_us, np.ndarray)
            or self.timestamps_us.dtype != np.int64
            or self.timestamps_us.ndim != 1
            or not 1 <= self.timestamps_us.size <= MOTION_FRAME_COUNT
        ):
            raise MediaPipeMappingError("timestamps_us must be int64 [1..120]")
        if self.timestamps_us.size > 1 and np.any(
            self.timestamps_us[1:] <= self.timestamps_us[:-1]
        ):
            raise MediaPipeMappingError("Holistic timestamps must be strictly increasing")
        if type(self.clip_start_us) is not int or type(self.clip_end_us) is not int:
            raise MediaPipeMappingError("clip bounds must be integer microseconds")
        if self.clip_end_us <= self.clip_start_us:
            raise MediaPipeMappingError("clip end must be later than clip start")
        if (
            int(self.timestamps_us[0]) < self.clip_start_us
            or int(self.timestamps_us[-1]) > self.clip_end_us
        ):
            raise MediaPipeMappingError("Holistic timestamps escape the displayed clip")

        expected = (
            ("pose_image", self.pose_image, POSE_POINT_COUNT),
            ("pose_world", self.pose_world, POSE_POINT_COUNT),
            ("observed_hand_image_0", self.observed_hand_images[0], HAND_POINT_COUNT),
            ("observed_hand_image_1", self.observed_hand_images[1], HAND_POINT_COUNT),
            ("observed_hand_world_0", self.observed_hand_worlds[0], HAND_POINT_COUNT),
            ("observed_hand_world_1", self.observed_hand_worlds[1], HAND_POINT_COUNT),
            ("face_image", self.face_image, FACE_POINT_COUNT),
        )
        for name, track, point_count in expected:
            if track.frame_count != self.frame_count or track.point_count != point_count:
                raise MediaPipeMappingError(
                    f"{name} must have shape [{self.frame_count},{point_count}]"
                )


@dataclass(frozen=True, slots=True)
class MediaPipeMappingPolicy:
    """Candidate thresholds; EX-203 has not frozen or release-qualified them."""

    minimum_shoulder_width: float = 1e-4
    hand_assignment_ambiguity_image_distance: float = 0.05
    maximum_hand_wrist_image_distance: float = 0.35
    maximum_hand_fit_residual: float = 0.30
    maximum_face_fit_residual: float = 0.40
    minimum_hand_image_world_orientation_determinant: float = 1e-5
    reflection_residual_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_shoulder_width", self.minimum_shoulder_width),
            (
                "hand_assignment_ambiguity_image_distance",
                self.hand_assignment_ambiguity_image_distance,
            ),
            ("maximum_hand_wrist_image_distance", self.maximum_hand_wrist_image_distance),
            ("maximum_hand_fit_residual", self.maximum_hand_fit_residual),
            ("maximum_face_fit_residual", self.maximum_face_fit_residual),
            (
                "minimum_hand_image_world_orientation_determinant",
                self.minimum_hand_image_world_orientation_determinant,
            ),
            ("reflection_residual_epsilon", self.reflection_residual_epsilon),
        ):
            if type(value) is not float or not np.isfinite(value) or value <= 0.0:
                raise MediaPipeMappingError(f"{name} must be a positive finite float")
        if self.hand_assignment_ambiguity_image_distance >= self.maximum_hand_wrist_image_distance:
            raise MediaPipeMappingError(
                "hand ambiguity threshold must be below the maximum wrist distance"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": MEDIAPIPE_MAPPING_PROFILE,
            "status": MEDIAPIPE_MAPPING_STATUS,
            "minimum_shoulder_width": self.minimum_shoulder_width,
            "hand_assignment_ambiguity_image_distance": (
                self.hand_assignment_ambiguity_image_distance
            ),
            "maximum_hand_wrist_image_distance": self.maximum_hand_wrist_image_distance,
            "maximum_hand_fit_residual": self.maximum_hand_fit_residual,
            "maximum_face_fit_residual": self.maximum_face_fit_residual,
            "minimum_hand_image_world_orientation_determinant": (
                self.minimum_hand_image_world_orientation_determinant
            ),
            "reflection_residual_epsilon": self.reflection_residual_epsilon,
            "camera_relative_axis_signs": list(CAMERA_RELATIVE_AXIS_SIGNS),
            "hand_anchor_indices": list(HAND_ANCHOR_INDICES),
            "left_pose_hand_anchors": list(LEFT_POSE_HAND_ANCHORS),
            "right_pose_hand_anchors": list(RIGHT_POSE_HAND_ANCHORS),
            "face_indices": list(FACE_INDICES),
            "face_anchor_mapping": [list(pair) for pair in FACE_ANCHOR_MAPPING],
            "motion_boundary": "positive-zero-candidate-until-ex203-freezes-detector",
        }


DEFAULT_MEDIAPIPE_MAPPING_POLICY: Final = MediaPipeMappingPolicy()


@dataclass(frozen=True, slots=True)
class FrameMappingDiagnostic:
    timestamp_us: int
    torso_valid: bool
    shoulder_width: float | None
    observed_hand_assignments: tuple[str, str]
    hand_assignment_cost_delta: float | None
    left_hand_fit_residual: float | None
    right_hand_fit_residual: float | None
    left_hand_image_world_orientation_determinant: float | None
    right_hand_image_world_orientation_determinant: float | None
    left_hand_reflection_alternative_lower_residual: bool | None
    right_hand_reflection_alternative_lower_residual: bool | None
    face_fit_residual: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MediaPipeMappingDiagnostics:
    profile: str
    status: str
    policy: dict[str, object]
    frames: tuple[FrameMappingDiagnostic, ...]
    invalid_torso_frame_count: int
    ambiguous_hand_frame_count: int
    rejected_hand_fit_count: int
    rejected_face_fit_count: int
    missing_point_count: int


@dataclass(frozen=True, slots=True)
class MediaPipeMappingResult:
    sequence: CanonicalPointSequence
    diagnostics: MediaPipeMappingDiagnostics


@dataclass(frozen=True, slots=True)
class _SimilarityFit:
    coordinates: Float32Array
    residual: float
    reflection_preferred: bool


def _normalized_orientation_determinant_2d(
    source: Float32Array,
    target: Float32Array,
) -> float | None:
    """Return a scale-independent 2D orientation determinant.

    MediaPipe rotates hand-world x/y out of the hand ROI and into image axes,
    while its Holistic graph only translates those coordinates to the pose
    wrist. Consequently, hand image and hand world x/y must retain orientation.
    This check is independent of noisy depth estimates from the separate hand
    and pose models.
    """

    source_centered = np.asarray(
        source[:, :2] - np.mean(source[:, :2], axis=0, dtype=np.float32),
        dtype=np.float32,
    )
    target_centered = np.asarray(
        target[:, :2] - np.mean(target[:, :2], axis=0, dtype=np.float32),
        dtype=np.float32,
    )
    source_energy = np.sum(source_centered * source_centered, dtype=np.float32)
    target_energy = np.sum(target_centered * target_centered, dtype=np.float32)
    if (
        not np.isfinite(source_energy)
        or not np.isfinite(target_energy)
        or source_energy <= np.float32(1e-12)
        or target_energy <= np.float32(1e-12)
    ):
        return None
    denominator = np.float32(
        np.sqrt(source_energy, dtype=np.float32) * np.sqrt(target_energy, dtype=np.float32)
    )
    covariance = np.asarray(source_centered.T @ target_centered, dtype=np.float32)
    determinant = np.float32(np.linalg.det(covariance) / (denominator * denominator))
    if not np.isfinite(determinant):
        return None
    return float(determinant)


def _copy_optional_scores(
    *,
    destination_values: Float32Array,
    destination_available: BoolArray,
    primary_values: Float32Array,
    primary_available: BoolArray,
    fallback_values: Float32Array | None = None,
    fallback_available: BoolArray | None = None,
) -> None:
    destination_values[:] = np.where(primary_available, primary_values, _POSITIVE_ZERO32).astype(
        np.float32
    )
    destination_available[:] = primary_available
    if fallback_values is not None and fallback_available is not None:
        use_fallback = ~primary_available & fallback_available
        destination_values[use_fallback] = fallback_values[use_fallback]
        destination_available[use_fallback] = True


def _image_distance(first: Float32Array, second: Float32Array) -> np.float32:
    delta = np.asarray(first[:2] - second[:2], dtype=np.float32)
    return np.float32(np.sqrt(np.sum(delta * delta, dtype=np.float32), dtype=np.float32))


def _similarity_fit(
    source_anchors: Float32Array,
    target_anchors: Float32Array,
    points: Float32Array,
    *,
    residual_scale: np.float32,
    reflection_epsilon: np.float32,
) -> _SimilarityFit | None:
    source_center = np.mean(source_anchors, axis=0, dtype=np.float32)
    target_center = np.mean(target_anchors, axis=0, dtype=np.float32)
    source_centered = np.asarray(source_anchors - source_center, dtype=np.float32)
    target_centered = np.asarray(target_anchors - target_center, dtype=np.float32)
    source_energy = np.sum(source_centered * source_centered, dtype=np.float32)
    if not np.isfinite(source_energy) or source_energy <= np.float32(1e-12):
        return None
    covariance = np.asarray(source_centered.T @ target_centered, dtype=np.float32)
    try:
        left, singular_values, right_transpose = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    left = np.asarray(left, dtype=np.float32)
    singular_values = np.asarray(singular_values, dtype=np.float32)
    right_transpose = np.asarray(right_transpose, dtype=np.float32)
    if not (
        np.isfinite(left).all()
        and np.isfinite(singular_values).all()
        and np.isfinite(right_transpose).all()
    ):
        return None
    unconstrained_rotation = np.asarray(right_transpose.T @ left.T, dtype=np.float32)
    determinant = np.float32(np.linalg.det(unconstrained_rotation))
    correction = np.ones(3, dtype=np.float32)
    if determinant < np.float32(0.0):
        correction[2] = np.float32(-1.0)
    rotation = np.asarray(
        right_transpose.T @ np.diag(correction).astype(np.float32) @ left.T,
        dtype=np.float32,
    )
    if np.float32(np.linalg.det(rotation)) <= np.float32(0.0):
        return None
    scale_numerator = np.sum(singular_values * correction, dtype=np.float32)
    scale = np.float32(scale_numerator / source_energy)
    if not np.isfinite(scale) or scale <= np.float32(0.0):
        return None

    transformed = np.asarray(
        (points - source_center) @ rotation.T * scale + target_center,
        dtype=np.float32,
    )
    fitted_anchors = np.asarray(
        source_centered @ rotation.T * scale + target_center,
        dtype=np.float32,
    )
    proper_error = np.asarray(fitted_anchors - target_anchors, dtype=np.float32)
    proper_residual = np.float32(
        np.sqrt(
            np.mean(np.sum(proper_error * proper_error, axis=1, dtype=np.float32)),
            dtype=np.float32,
        )
        / residual_scale
    )
    reflection_preferred = False
    if determinant < np.float32(0.0):
        reflected_scale = np.float32(np.sum(singular_values, dtype=np.float32) / source_energy)
        reflected = np.asarray(
            source_centered @ unconstrained_rotation.T * reflected_scale + target_center,
            dtype=np.float32,
        )
        reflected_error = np.asarray(reflected - target_anchors, dtype=np.float32)
        reflected_residual = np.float32(
            np.sqrt(
                np.mean(np.sum(reflected_error * reflected_error, axis=1, dtype=np.float32)),
                dtype=np.float32,
            )
            / residual_scale
        )
        reflection_preferred = bool(reflected_residual + reflection_epsilon < proper_residual)
    if not np.isfinite(transformed).all() or not np.isfinite(proper_residual):
        return None
    return _SimilarityFit(
        coordinates=transformed,
        residual=float(proper_residual),
        reflection_preferred=reflection_preferred,
    )


def _body_normalize(
    coordinates: Float32Array,
    *,
    shoulder_center: Float32Array,
    shoulder_width: np.float32,
) -> Float32Array:
    axis_signs = np.asarray(CAMERA_RELATIVE_AXIS_SIGNS, dtype=np.float32)
    return np.asarray(
        ((coordinates - shoulder_center) / shoulder_width) * axis_signs,
        dtype=np.float32,
    )


def _hand_assignment(
    raw: RawHolisticSequence,
    frame_index: int,
    policy: MediaPipeMappingPolicy,
) -> tuple[tuple[str, str], float | None, dict[str, int]]:
    left_pose_available = bool(raw.pose_image.coordinate_available[frame_index, POSE_LEFT_WRIST])
    right_pose_available = bool(raw.pose_image.coordinate_available[frame_index, POSE_RIGHT_WRIST])
    usable: list[int] = []
    states = ["missing", "missing"]
    for slot in range(2):
        image = raw.observed_hand_images[slot]
        if not image.coordinate_available[frame_index, 0]:
            continue
        usable.append(slot)
    if not usable:
        return (states[0], states[1]), None, {}
    if not left_pose_available or not right_pose_available:
        for slot in usable:
            states[slot] = "unassignable-pose-wrists"
        return (states[0], states[1]), None, {}

    left_wrist = raw.pose_image.coordinates[frame_index, POSE_LEFT_WRIST]
    right_wrist = raw.pose_image.coordinates[frame_index, POSE_RIGHT_WRIST]
    distances: dict[int, tuple[np.float32, np.float32]] = {}
    for slot in usable:
        hand_wrist = raw.observed_hand_images[slot].coordinates[frame_index, 0]
        distances[slot] = (
            _image_distance(hand_wrist, left_wrist),
            _image_distance(hand_wrist, right_wrist),
        )
    ambiguity = np.float32(policy.hand_assignment_ambiguity_image_distance)
    maximum = np.float32(policy.maximum_hand_wrist_image_distance)
    assignments: dict[str, int] = {}
    cost_delta: float | None = None
    if len(usable) == 1:
        slot = usable[0]
        left_distance, right_distance = distances[slot]
        cost_delta = float(np.abs(left_distance - right_distance))
        if np.abs(left_distance - right_distance) <= ambiguity:
            states[slot] = "ambiguous"
            return (states[0], states[1]), cost_delta, {}
        side = "left" if left_distance < right_distance else "right"
        selected_distance = left_distance if side == "left" else right_distance
        if selected_distance > maximum:
            states[slot] = "rejected-distance"
            return (states[0], states[1]), cost_delta, {}
        states[slot] = side
        assignments[side] = slot
        return (states[0], states[1]), cost_delta, assignments

    first, second = usable
    direct = np.float32(distances[first][0] + distances[second][1])
    swapped = np.float32(distances[first][1] + distances[second][0])
    cost_delta = float(np.abs(direct - swapped))
    if np.abs(direct - swapped) <= ambiguity:
        states[first] = "ambiguous"
        states[second] = "ambiguous"
        return (states[0], states[1]), cost_delta, {}
    if direct < swapped:
        proposed = {"left": first, "right": second}
    else:
        proposed = {"left": second, "right": first}
    if distances[proposed["left"]][0] > maximum or distances[proposed["right"]][1] > maximum:
        states[first] = "rejected-distance"
        states[second] = "rejected-distance"
        return (states[0], states[1]), cost_delta, {}
    states[proposed["left"]] = "left"
    states[proposed["right"]] = "right"
    return (states[0], states[1]), cost_delta, proposed


def map_holistic_to_canonical(
    raw: RawHolisticSequence,
    policy: MediaPipeMappingPolicy = DEFAULT_MEDIAPIPE_MAPPING_POLICY,
) -> MediaPipeMappingResult:
    """Map raw Holistic tracks into the EX-203 candidate point contract.

    The mapping is executable research scaffolding. Its profile and thresholds are not
    frozen, release-qualified, or evidence that desktop and iOS runtimes agree.
    """

    if not isinstance(raw, RawHolisticSequence):
        raise TypeError("raw must be a RawHolisticSequence")
    if not isinstance(policy, MediaPipeMappingPolicy):
        raise TypeError("policy must be a MediaPipeMappingPolicy")
    frame_count = raw.frame_count
    coordinates = np.zeros((frame_count, MOTION_POINT_COUNT, 3), dtype=np.float32)
    coordinate_present = np.zeros((frame_count, MOTION_POINT_COUNT), dtype=np.bool_)
    presence = np.zeros((frame_count, MOTION_POINT_COUNT), dtype=np.float32)
    presence_available = np.zeros((frame_count, MOTION_POINT_COUNT), dtype=np.bool_)
    visibility = np.zeros((frame_count, MOTION_POINT_COUNT), dtype=np.float32)
    visibility_available = np.zeros((frame_count, MOTION_POINT_COUNT), dtype=np.bool_)
    track_present = np.zeros((frame_count, 4), dtype=np.bool_)
    motion_boundary = np.zeros(frame_count, dtype=np.bool_)

    frame_diagnostics: list[FrameMappingDiagnostic] = []
    invalid_torso_count = 0
    ambiguous_hand_count = 0
    rejected_hand_fit_count = 0
    rejected_face_fit_count = 0
    face_indices_array = np.asarray(FACE_INDICES, dtype=np.int64)
    hand_anchors_array = np.asarray(HAND_ANCHOR_INDICES, dtype=np.int64)
    axis_minimum = np.float32(policy.minimum_shoulder_width)
    hand_orientation_minimum = policy.minimum_hand_image_world_orientation_determinant
    reflection_epsilon = np.float32(policy.reflection_residual_epsilon)

    for frame_index in range(frame_count):
        reasons: list[str] = []
        shoulder_width_value: float | None = None
        left_fit_residual: float | None = None
        right_fit_residual: float | None = None
        left_orientation_determinant: float | None = None
        right_orientation_determinant: float | None = None
        left_reflection_alternative: bool | None = None
        right_reflection_alternative: bool | None = None
        face_fit_residual: float | None = None

        _copy_optional_scores(
            destination_values=presence[frame_index, :33],
            destination_available=presence_available[frame_index, :33],
            primary_values=raw.pose_world.presence[frame_index],
            primary_available=raw.pose_world.presence_available[frame_index],
            fallback_values=raw.pose_image.presence[frame_index],
            fallback_available=raw.pose_image.presence_available[frame_index],
        )
        _copy_optional_scores(
            destination_values=visibility[frame_index, :33],
            destination_available=visibility_available[frame_index, :33],
            primary_values=raw.pose_world.visibility[frame_index],
            primary_available=raw.pose_world.visibility_available[frame_index],
            fallback_values=raw.pose_image.visibility[frame_index],
            fallback_available=raw.pose_image.visibility_available[frame_index],
        )
        _copy_optional_scores(
            destination_values=presence[frame_index, 75:107],
            destination_available=presence_available[frame_index, 75:107],
            primary_values=raw.face_image.presence[frame_index, face_indices_array],
            primary_available=raw.face_image.presence_available[frame_index, face_indices_array],
        )
        _copy_optional_scores(
            destination_values=visibility[frame_index, 75:107],
            destination_available=visibility_available[frame_index, 75:107],
            primary_values=raw.face_image.visibility[frame_index, face_indices_array],
            primary_available=raw.face_image.visibility_available[frame_index, face_indices_array],
        )

        shoulder_available = bool(
            raw.pose_world.coordinate_available[frame_index, POSE_LEFT_SHOULDER]
            and raw.pose_world.coordinate_available[frame_index, POSE_RIGHT_SHOULDER]
        )
        torso_valid = False
        shoulder_center = np.zeros(3, dtype=np.float32)
        shoulder_width = _POSITIVE_ZERO32
        if shoulder_available:
            left_shoulder = raw.pose_world.coordinates[frame_index, POSE_LEFT_SHOULDER]
            right_shoulder = raw.pose_world.coordinates[frame_index, POSE_RIGHT_SHOULDER]
            shoulder_delta = np.asarray(right_shoulder - left_shoulder, dtype=np.float32)
            shoulder_width = np.float32(
                np.sqrt(
                    np.sum(shoulder_delta * shoulder_delta, dtype=np.float32),
                    dtype=np.float32,
                )
            )
            if np.isfinite(shoulder_width) and shoulder_width >= axis_minimum:
                shoulder_center = np.asarray(
                    (left_shoulder + right_shoulder) * np.float32(0.5),
                    dtype=np.float32,
                )
                torso_valid = True
                shoulder_width_value = float(shoulder_width)
        if not torso_valid:
            invalid_torso_count += 1
            reasons.append("missing-or-degenerate-shoulders")
            frame_diagnostics.append(
                FrameMappingDiagnostic(
                    timestamp_us=int(raw.timestamps_us[frame_index]),
                    torso_valid=False,
                    shoulder_width=None,
                    observed_hand_assignments=("unmapped-torso", "unmapped-torso"),
                    hand_assignment_cost_delta=None,
                    left_hand_fit_residual=None,
                    right_hand_fit_residual=None,
                    left_hand_image_world_orientation_determinant=None,
                    right_hand_image_world_orientation_determinant=None,
                    left_hand_reflection_alternative_lower_residual=None,
                    right_hand_reflection_alternative_lower_residual=None,
                    face_fit_residual=None,
                    reasons=tuple(reasons),
                )
            )
            continue

        pose_present = raw.pose_world.coordinate_available[frame_index]
        pose_normalized = _body_normalize(
            raw.pose_world.coordinates[frame_index],
            shoulder_center=shoulder_center,
            shoulder_width=shoulder_width,
        )
        coordinates[frame_index, :33][pose_present] = pose_normalized[pose_present]
        coordinate_present[frame_index, :33] = pose_present

        assignments, assignment_delta, assigned_slots = _hand_assignment(raw, frame_index, policy)
        if "ambiguous" in assignments:
            ambiguous_hand_count += 1
            reasons.append("ambiguous-hand-assignment")
        for side, output_slice, target_anchor_indices in (
            ("left", slice(33, 54), LEFT_POSE_HAND_ANCHORS),
            ("right", slice(54, 75), RIGHT_POSE_HAND_ANCHORS),
        ):
            slot = assigned_slots.get(side)
            if slot is None:
                continue
            image_track = raw.observed_hand_images[slot]
            world_track = raw.observed_hand_worlds[slot]
            _copy_optional_scores(
                destination_values=presence[frame_index, output_slice],
                destination_available=presence_available[frame_index, output_slice],
                primary_values=world_track.presence[frame_index],
                primary_available=world_track.presence_available[frame_index],
                fallback_values=image_track.presence[frame_index],
                fallback_available=image_track.presence_available[frame_index],
            )
            _copy_optional_scores(
                destination_values=visibility[frame_index, output_slice],
                destination_available=visibility_available[frame_index, output_slice],
                primary_values=world_track.visibility[frame_index],
                primary_available=world_track.visibility_available[frame_index],
                fallback_values=image_track.visibility[frame_index],
                fallback_available=image_track.visibility_available[frame_index],
            )
            if not world_track.coordinate_available[frame_index, hand_anchors_array].all():
                reasons.append(f"{side}-hand-missing-world-fit-anchors")
                continue
            if not image_track.coordinate_available[frame_index, hand_anchors_array].all():
                reasons.append(f"{side}-hand-missing-image-orientation-anchors")
                continue
            orientation_determinant = _normalized_orientation_determinant_2d(
                world_track.coordinates[frame_index, hand_anchors_array],
                image_track.coordinates[frame_index, hand_anchors_array],
            )
            if side == "left":
                left_orientation_determinant = orientation_determinant
            else:
                right_orientation_determinant = orientation_determinant
            if orientation_determinant is None or (
                abs(orientation_determinant) < hand_orientation_minimum
            ):
                rejected_hand_fit_count += 1
                reasons.append(f"{side}-hand-degenerate-image-world-orientation")
                continue
            if orientation_determinant < 0.0:
                rejected_hand_fit_count += 1
                reasons.append(f"{side}-hand-image-world-reflection")
                continue
            target_indices = np.asarray(target_anchor_indices, dtype=np.int64)
            if not raw.pose_world.coordinate_available[frame_index, target_indices].all():
                reasons.append(f"{side}-hand-missing-pose-fit-anchors")
                continue
            fit = _similarity_fit(
                world_track.coordinates[frame_index, hand_anchors_array],
                raw.pose_world.coordinates[frame_index, target_indices],
                world_track.coordinates[frame_index],
                residual_scale=shoulder_width,
                reflection_epsilon=reflection_epsilon,
            )
            if fit is not None:
                if side == "left":
                    left_fit_residual = fit.residual
                    left_reflection_alternative = fit.reflection_preferred
                else:
                    right_fit_residual = fit.residual
                    right_reflection_alternative = fit.reflection_preferred
            if fit is None:
                rejected_hand_fit_count += 1
                reasons.append(f"{side}-hand-degenerate-fit")
                continue
            if fit.residual > policy.maximum_hand_fit_residual:
                rejected_hand_fit_count += 1
                reasons.append(f"{side}-hand-fit-residual")
                continue
            hand_present = world_track.coordinate_available[frame_index]
            hand_normalized = _body_normalize(
                fit.coordinates,
                shoulder_center=shoulder_center,
                shoulder_width=shoulder_width,
            )
            coordinates[frame_index, output_slice][hand_present] = hand_normalized[hand_present]
            coordinate_present[frame_index, output_slice] = hand_present

        face_anchor_indices = np.asarray(
            [face_index for face_index, _ in FACE_ANCHOR_MAPPING], dtype=np.int64
        )
        pose_face_anchor_indices = np.asarray(
            [pose_index for _, pose_index in FACE_ANCHOR_MAPPING], dtype=np.int64
        )
        face_anchors_available = bool(
            raw.face_image.coordinate_available[frame_index, face_anchor_indices].all()
            and raw.pose_world.coordinate_available[frame_index, pose_face_anchor_indices].all()
        )
        if face_anchors_available:
            face_fit = _similarity_fit(
                raw.face_image.coordinates[frame_index, face_anchor_indices],
                raw.pose_world.coordinates[frame_index, pose_face_anchor_indices],
                raw.face_image.coordinates[frame_index, face_indices_array],
                residual_scale=shoulder_width,
                reflection_epsilon=reflection_epsilon,
            )
            if face_fit is not None:
                face_fit_residual = face_fit.residual
            if face_fit is None:
                rejected_face_fit_count += 1
                reasons.append("face-degenerate-fit")
            elif face_fit.reflection_preferred:
                rejected_face_fit_count += 1
                reasons.append("face-reflection-required")
            elif face_fit.residual > policy.maximum_face_fit_residual:
                rejected_face_fit_count += 1
                reasons.append("face-fit-residual")
            else:
                face_present = raw.face_image.coordinate_available[frame_index, face_indices_array]
                face_normalized = _body_normalize(
                    face_fit.coordinates,
                    shoulder_center=shoulder_center,
                    shoulder_width=shoulder_width,
                )
                coordinates[frame_index, 75:107][face_present] = face_normalized[face_present]
                coordinate_present[frame_index, 75:107] = face_present
        else:
            reasons.append("face-missing-fit-anchors")

        for track_index, point_slice in enumerate(
            (slice(0, 33), slice(33, 54), slice(54, 75), slice(75, 107))
        ):
            track_present[frame_index, track_index] = bool(
                coordinate_present[frame_index, point_slice].any()
            )
        frame_diagnostics.append(
            FrameMappingDiagnostic(
                timestamp_us=int(raw.timestamps_us[frame_index]),
                torso_valid=True,
                shoulder_width=shoulder_width_value,
                observed_hand_assignments=assignments,
                hand_assignment_cost_delta=assignment_delta,
                left_hand_fit_residual=left_fit_residual,
                right_hand_fit_residual=right_fit_residual,
                left_hand_image_world_orientation_determinant=(left_orientation_determinant),
                right_hand_image_world_orientation_determinant=(right_orientation_determinant),
                left_hand_reflection_alternative_lower_residual=(left_reflection_alternative),
                right_hand_reflection_alternative_lower_residual=(right_reflection_alternative),
                face_fit_residual=face_fit_residual,
                reasons=tuple(reasons),
            )
        )

    for array in (coordinates, presence, visibility):
        if not np.isfinite(array).all():
            raise MediaPipeMappingError("canonical mapping produced a nonfinite value")
    if not _positive_zero(coordinates[~coordinate_present]):
        raise MediaPipeMappingError("canonical missing coordinates are not positive zero")
    if not _positive_zero(presence[~presence_available]) or not _positive_zero(
        visibility[~visibility_available]
    ):
        raise MediaPipeMappingError("canonical unavailable scores are not positive zero")
    if motion_boundary.any():
        raise AssertionError("EX-203 candidate boundary bits must remain zero")

    for array in (
        coordinates,
        coordinate_present,
        presence,
        presence_available,
        visibility,
        visibility_available,
        track_present,
        motion_boundary,
    ):
        array.setflags(write=False)
    timestamps = np.array(raw.timestamps_us, dtype=np.int64, copy=True)
    timestamps.setflags(write=False)
    sequence = CanonicalPointSequence(
        coordinates=coordinates,
        coordinate_present=coordinate_present,
        presence=presence,
        presence_available=presence_available,
        visibility=visibility,
        visibility_available=visibility_available,
        timestamps_us=timestamps,
        clip_start_us=raw.clip_start_us,
        clip_end_us=raw.clip_end_us,
        track_present=track_present,
        motion_boundary=motion_boundary,
    )
    diagnostics = MediaPipeMappingDiagnostics(
        profile=MEDIAPIPE_MAPPING_PROFILE,
        status=MEDIAPIPE_MAPPING_STATUS,
        policy=policy.as_dict(),
        frames=tuple(frame_diagnostics),
        invalid_torso_frame_count=invalid_torso_count,
        ambiguous_hand_frame_count=ambiguous_hand_count,
        rejected_hand_fit_count=rejected_hand_fit_count,
        rejected_face_fit_count=rejected_face_fit_count,
        missing_point_count=int((~coordinate_present).sum()),
    )
    return MediaPipeMappingResult(sequence=sequence, diagnostics=diagnostics)
