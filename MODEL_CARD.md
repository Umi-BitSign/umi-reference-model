# Model card: UMI S1 baseline

## Model identity

S1 is a compact skeletal-motion-to-English model built as a reference implementation
for UMI miners. Each sealed release publishes its inference revision in the portable
bundle's `inference-identity.json`. The release manifest binds that revision to the
model archive, code revision, and source-rights decision. Each operator derives a new
inference revision that binds the immutable ID of the extractor image built locally.

Release status: `component_test_no_weight`

## Architecture and input

S1 consumes up to 120 frames sampled at 8 Hz. Each frame contains 1,184 normalized
values derived from face, pose, and hand landmarks, plus a frame mask. The tokenizer
contains 4,096 pieces. The decoder contract permits up to 128 tokens, while this
release applies a validation-selected cap of eight whitespace-delimited output words.

The Linux reference miner uses a locally built Linux/AMD64 MediaPipe Holistic
extractor image. The derived bundle binds its exact local image ID. No claim is made
that independently built images have identical bytes. Linux/AMD64 and Linux/ARM64
landmark tensors are separately identified and are not bit-equivalent.

## Training lineage

The selected arm initializes the S1 encoder from FSboard fingerspelling pretraining,
then trains the ASL-to-English model on FLEURS-ASL. The frozen run used seed 17,
completed 40 epochs, selected epoch 20 from validation, and opened the held-out test
partition once after model and postprocessing selection.

The public release contains weights and inference metadata only. It does not contain
training videos, source annotations, derived source records, or signer identifiers.

## Evaluation

| Evaluation | Mean normalized score |
|---|---:|
| Frozen validation, raw decoder | 0.015885 |
| Held-out test, raw decoder | 0.014881 |
| Frozen validation, selected eight-word cap | 0.044418 |

Scores use exact single-reference WER through the pinned UMI normalization adapter.
The output cap was selected using validation predictions before test opening. No
test-set choice changed the model or cap.

The model is below the provisional UMI utility floor of 0.10. This evaluation is a
bootstrap measurement. It is outside the independent WER/CER validity study, public
calibration, ten-tempo agreement trial, and 30-day activation soak required by the UMI
specification.

## Intended use

The release supports:

- integration testing of a real asynchronous UMI miner backend;
- a common baseline for miners developing better models;
- profiling of raw-video extraction, model latency, and resource use;
- research on compact and mobile ASL translation models.

## Limitations and excluded use

Outputs are often wrong, repetitive, or incomplete. The eight-word cap can truncate
meaning. Landmark extraction can fail under occlusion, unusual framing, motion blur,
or unsupported media. Skeletal features discard appearance cues that may be relevant
to translation.

Do not use this model as an ASL interpreter, accessibility certification, or the sole
input to medical, legal, financial, emergency, employment, or safety decisions. Users
must see that output is machine-generated and may be incorrect.

## Licenses

The model weights and portable bundle are licensed under CC BY-SA 4.0. Training
lineage includes FLEURS-ASL under CC BY-SA 4.0 and FSboard under CC BY 4.0. Required
attributions and modification notices are in `NOTICE`. Runtime code is Apache-2.0.

The MediaPipe Holistic task binary is not redistributed with the model. Operators
download it from the upstream Google storage URL and verify its pinned SHA-256 digest.
