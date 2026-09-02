from __future__ import annotations

SCHEMA_VERSION = "bitsign-skeleton/1"
EXTRACTOR_VISION_2D = "apple-vision-2d/1"
EXTRACTOR_OPENPOSE_2D = "openpose-bfh-2d/1"
EXTRACTOR_SYNTHETIC = "synthetic-motion-code/1"

BODY_JOINTS = (
    "body.root",
    "body.neck",
    "body.left_shoulder",
    "body.right_shoulder",
    "body.left_elbow",
    "body.right_elbow",
    "body.left_wrist",
    "body.right_wrist",
    "body.left_hip",
    "body.right_hip",
)

HAND_LANDMARKS = (
    "wrist",
    "thumb_cmc",
    "thumb_mp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
)

LEFT_HAND_JOINTS = tuple(f"left_hand.{name}" for name in HAND_LANDMARKS)
RIGHT_HAND_JOINTS = tuple(f"right_hand.{name}" for name in HAND_LANDMARKS)

FACE_JOINTS = (
    "face.left_eye",
    "face.right_eye",
    "face.left_brow",
    "face.right_brow",
    "face.nose_tip",
    "face.nose_bridge",
    "face.mouth_left",
    "face.mouth_right",
    "face.upper_lip",
    "face.lower_lip",
    "face.jaw_left",
    "face.jaw_right",
)

JOINT_NAMES = BODY_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS + FACE_JOINTS
JOINT_COUNT = len(JOINT_NAMES)
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}

BODY_SLICE = slice(0, len(BODY_JOINTS))
LEFT_HAND_SLICE = slice(BODY_SLICE.stop, BODY_SLICE.stop + len(LEFT_HAND_JOINTS))
RIGHT_HAND_SLICE = slice(LEFT_HAND_SLICE.stop, LEFT_HAND_SLICE.stop + len(RIGHT_HAND_JOINTS))
FACE_SLICE = slice(RIGHT_HAND_SLICE.stop, JOINT_COUNT)
STREAM_SLICES = (BODY_SLICE, LEFT_HAND_SLICE, RIGHT_HAND_SLICE, FACE_SLICE)

RAW_CHANNEL_NAMES = ("x", "y", "z", "confidence", "present", "z_valid")
FEATURE_CHANNEL_NAMES = (
    "x",
    "y",
    "z",
    "delta_x",
    "delta_y",
    "delta_z",
    "confidence",
    "present",
    "z_valid",
)
RAW_CHANNELS = len(RAW_CHANNEL_NAMES)
FEATURE_CHANNELS = len(FEATURE_CHANNEL_NAMES)

DEFAULT_MAX_FRAMES = 96
DEFAULT_MAX_OUTPUT_TOKENS = 64
MIN_SEQUENCE_FRAMES = 2
MAX_SEQUENCE_SECONDS = 6.0
MAX_SEQUENCE_FRAMES = 720

assert JOINT_COUNT == 64
