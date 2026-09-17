# Private Apple Silicon MediaPipe overlay

This build removes the headless OpenGL dependency from landmark preprocessing.
The supplied translation model still uses MPS. It is an internal serving
integration, not a published wheel or a qualified competition runtime.

`inputs.json` records the source revisions, retained compatibility patch, the
complete non-cache inventory of the installed MediaPipe package, and hashes for
the completed binding and all eight OpenCV libraries. The inventory digest is
the SHA-256 of the canonical JSON list of `{path,sha256,size_bytes}` records, as
declared by its `sha256-canonical-json-v1` algorithm. `stage_macos_overlay.py`
runs on the ARM64 build host. It accepts only those reviewed inputs, copies and
hashes each source through one open descriptor into a private temporary tree,
then replaces the copied extension. After exact output verification it publishes
the tree with no-replace semantics. The installed environment is left unchanged.

```sh
python community/native-build/stage_macos_overlay.py \
  --build-bin /ABSOLUTE/BUILD/bazel-bin \
  --installed-package /ABSOLUTE/ENV/lib/python3.10/site-packages/mediapipe \
  --destination /ABSOLUTE/NEW/mediapipe-cpu-overlay
```

The package requires eight private OpenCV dylibs. The stager validates their
dependency names, copies them next to the extension, removes build search paths,
relocates dependencies to `@loader_path` and ad-hoc signs the changed files.
It then inventories all copied files, requires the exact retained live manifest
digest and makes the directory read-only. These checks preserve the existing
overlay manifest schema and identity; no provenance fields are added to the live
artifact. Keep the manifest digest outside that directory as the reviewed
identity. Do not use a digest supplied by an untrusted candidate as approval to
load code.

Before importing MediaPipe, the launcher must call
`verify_macos_overlay.verify_overlay` with that reviewed digest. Keep the
overlay read-only to the worker for its whole lifetime. File modes alone do
not enforce this against their owner; the Studio diagnostic also denies writes
to the exact overlay path in the macOS sandbox. The verifier does not create a
sandbox or certify the rest of the Python environment or model bundle.

Real-input results and limitations are in
[the headless Studio record](../HEADLESS_STUDIO_CHECK.md). This packaging step
does not change model weights, approve contribution rights, or establish ASL
accuracy. The Linux evaluator bundle retains its separate runtime and tests.
