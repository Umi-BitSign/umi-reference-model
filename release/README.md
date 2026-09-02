# Release metadata

This directory keeps small, reviewable metadata in Git:

- `runtime-files.txt` is the explicit public source staging allowlist.
- `release-manifest.json` binds a sealed release to code, rights, and the base model
  identity. It records that the extractor is built and rebound locally.
- `SHA256SUMS` binds the downloadable portable model archive.

Only the model ZIP is uploaded as a release asset. Model weights remain outside Git.
No Docker image archive is published. Operators build the checked-in source and bind
their local immutable image ID into a derived bundle.

Generate and verify the metadata with `tools/release_artifacts.py` after completing
`docs/RELEASE.md`. The manifest records the source-only parent commit; the release tag
points to its metadata-only child. Do not hand-edit a generated manifest or checksum
file.
