# Preserved native baseline adapters

These are the exact adapter bytes used by the September 16 Apple Silicon
rehearsal. `manifest.json` identifies the complete preserved bundle as
`ce459641c180c680aae32009052985165fa8bbffd165b0dca05d2dade0e619ed`.
No tensors or private evaluation data are stored here.

To reconstruct that bundle from the original supplied ZIP:

```sh
mkdir -m 700 -p /ABSOLUTE/PRIVATE/native-models
python3 tools/stage_native_baseline.py \
  --archive /ABSOLUTE/PATH/umi-community-baseline-v0.2.zip \
  --destination /ABSOLUTE/PRIVATE/native-models/v0.2-mps
```

The command verifies the original ZIP using the existing importer, applies the
reviewed adapters, then verifies every output file against the preserved bundle.
It refuses an existing destination. It performs no inference, dependency install,
wallet operation, endpoint registration, baseline promotion or weight submission.
The separate runtime installation, sandbox and 120-second limit still apply.
Public model-asset distribution remains pending.

`runtime.py` comes from `candidates/community-baseline-v0.2/runtime.py`, including
the RGB crop correction. The native entrypoint selects MPS with five beams,
2048-token decoding, DINO batches of 128 and one execution at a time. It sends
diagnostics to stderr and only the bounded translation to stdout.

`umi_video_reader.py` is the narrow OpenCV-backed RGB reader used on the Studio.
`umi_landmarks.py` derives from the supplied SHuBERT demo code and runs face and
hand landmark graphs concurrently while using CPU MediaPipe delegates. Their
exact bytes are retained here, including unused legacy helpers. They are not a
new general-purpose preprocessing API. Preserve the original package's licences
and provenance; the outer manifest records the derived files, while its inherited
`FILE_LIST.json` and `SHA256SUMS` describe the original archive.

`LOCAL_RUNTIME_DERIVATION.json` is the historical RGB-correction record retained
in the measured bundle. The outer manifest also binds the later MPS entrypoint
and native compatibility files. Reformatting an adapter changes the bundle
identity and requires a new qualification record.

See [qualification results](../QUALIFICATION.md) for the six-clip evidence and
its limits. This imported baseline has no model-contributor reward attribution.
