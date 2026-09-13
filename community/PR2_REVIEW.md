# Community candidate source review

Reviewed [PR #2](https://github.com/Umi-BitSign/umi-reference-model/pull/2) at
`ea72ce5f66d1764992348169b552e716f7ae3cf5` against the separately supplied ZIP
already used by [the community evaluator integration](README.md).

The comparison found 439 byte-identical files, one changed README and three
new files: `.gitignore`, `MODEL_ARTIFACT.json` and `verify_model_assets.py`.
The five model/task binaries and the ZIP's two inventory files are omitted from
the Git candidate. All eight declared model/configuration asset sizes and
SHA-256 values were checked against the ZIP's actual decompressed bytes. The
ZIP identity remains
`f78979599486456e06e7886b126169f8d45e5e1e7d16d7a2b617648615eef3d4`.

The candidate exposes the same inference assembly for inspection. It introduces
no newly trained weights and does not change the S1 release, CPU adapter,
evaluation policy, contributor attribution or live validators.

Integration changes:

- The candidate guide links to the existing Linux CPU evaluator instructions
  and explains which checksum inventories exist only in the ZIP.
- The asset checker rejects empty, duplicate and malformed inventories, unsafe
  paths, oversized declarations and files that grow during hashing. It remains
  a standard-library checker and does not import model code.
- Tests cover those failures, matching synthetic assets, the committed model
  metadata and the absence of model binaries from the Git candidate.

The supplied `MODEL_IDENTITY.json` has an `inference_sha256` field without a
derivation specification. It is retained as supplied metadata and is not used
for UMI identity verification. UMI uses the pinned whole ZIP and the canonical
bundle manifest produced by the importer.

Verification: 175 tests passed, four skipped; repository guard passed; 428
candidate Python files parsed under Python 3.10 syntax without importing them.
Ruff checks cover the maintained code and checker. Vendored inference source
keeps its supplied bytes and upstream formatting.

This review adds no ASL accuracy result, independent evaluator attestation or
rights approval. The [qualification record](QUALIFICATION.md) remains the source
for completed execution checks and outstanding launch requirements.
