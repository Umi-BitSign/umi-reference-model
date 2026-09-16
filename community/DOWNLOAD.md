# Download the community baseline

The original supplied ZIP is available in the
[community baseline v0.2 prerelease](https://github.com/Umi-BitSign/umi-reference-model/releases/tag/community-baseline-v0.2).
It contains inference assets and their notices, without private evaluation data.
Publishing these assets does not activate competition rewards or award a model
contribution share.

The ZIP is split into two parts. Run these commands in a new directory with at
least 6 GB free for the parts and reconstructed archive, plus space for extraction:

```sh
base=https://github.com/Umi-BitSign/umi-reference-model/releases/download/community-baseline-v0.2
curl -fL -O "$base/SHA256SUMS.parts"
curl -fL -O "$base/umi-community-baseline-v0.2.zip.part-aa"
curl -fL -O "$base/umi-community-baseline-v0.2.zip.part-ab"
shasum -a 256 -c SHA256SUMS.parts
cat umi-community-baseline-v0.2.zip.part-aa umi-community-baseline-v0.2.zip.part-ab > umi-community-baseline-v0.2.zip
shasum -a 256 umi-community-baseline-v0.2.zip
```

On Linux, `sha256sum -c SHA256SUMS.parts` and `sha256sum FILE` can replace
the two `shasum` commands. Do not continue if either part fails verification.

The complete ZIP must be 2,934,700,086 bytes with SHA-256
`f78979599486456e06e7886b126169f8d45e5e1e7d16d7a2b617648615eef3d4`.
The importers below independently enforce this complete archive identity.

For the corrected Apple Silicon bundle used in the September 16 operational
rehearsal, follow [native reconstruction](native/README.md). For the distinct
original CPU bundle, follow [the CPU importer](README.md#verify-and-stage).
Neither path changes the model tensors. Read the included provenance and licence
notices before redistribution. These artifacts provide a research baseline,
not a claim of general translation accuracy or clinical suitability.
