# Community baseline integration

This directory integrates the supplied `umi-community-baseline-v0.2.zip` as a
candidate for UMI's `offline_bundle/1` evaluator. It does not replace the released
S1 baseline or activate open-competition rewards.

The supplied package contains SHuBERT/ByT5 inference weights and vendored code.
Its provenance says no model tensor was trained, pruned or quantized in this
assembly. Its supplied qualification is an Apple Silicon functional smoke check,
not a measured accuracy result or improvement claim.

## Verify and stage

Keep the ZIP and extracted weights outside Git. The importer uses only Python's
standard library and never imports the supplied model code:

```sh
mkdir -m 700 -p /ABSOLUTE/PRIVATE/community-models
python3 tools/import_community_baseline.py \
  --archive /ABSOLUTE/PATH/umi-community-baseline-v0.2.zip \
  --destination /ABSOLUTE/PRIVATE/community-models/v0.2-cpu
```

The expected ZIP is 2,934,700,086 bytes, SHA-256
`f78979599486456e06e7886b126169f8d45e5e1e7d16d7a2b617648615eef3d4`.
The importer rejects different bytes, traversal, links, duplicate paths, inventory
disagreements and content above its bounds. It retains all 447 supplied files
unchanged and adds the UMI entrypoint and container recipe. No model installation
or execution happens during intake.

The new directory contains `model/`, a canonical `manifest.json` and `intake.json`.
`model_bundle_sha256` uses UMI's domain-separated competition digest;
`manifest_file_sha256` is the plain file checksum. They are different identities.
Changing the adapter or container recipe changes the bundle identity. An existing
destination is never reused as a new intake.

## Linux CPU image and execution

Build on a separate, wallet-free Linux host with rootless Podman and cgroup v2:

```sh
podman build --tag localhost/umi-community-cpu:rehearsal \
  --file /ABSOLUTE/PRIVATE/community-models/v0.2-cpu/model/runtime.Dockerfile \
  /ABSOLUTE/PRIVATE/community-models/v0.2-cpu
podman image inspect localhost/umi-community-cpu:rehearsal --format '{{.Digest}}'
```

The image installs the supplied Python 3.10 dependency pins separately from UMI.
Model weights remain outside the image. Linux ARM64 dependency installation and
one real-model synthetic-clip run have passed. That cold-start run took 231,638 ms
under the limits below. It does not establish ASL accuracy or adequate throughput.
Other architectures require their own run.

Use the UMI Python environment to invoke the rehearsal harness. It calls UMI's
actual `preserve_bundle` and `execute_offline_case` functions. Replace the image
digest with the value inspected above:

```sh
python community/rehearse.py \
  --staged /ABSOLUTE/PRIVATE/community-models/v0.2-cpu \
  --archive /ABSOLUTE/PRIVATE/community-models/preserved \
  --video /ABSOLUTE/PATH/smoke.mp4 \
  --image localhost/umi-community-cpu@sha256:YOUR_IMAGE_DIGEST \
  --output /ABSOLUTE/PRIVATE/community-models/smoke.json
```

This uses an unsigned, expired smoke-only policy, four CPUs, a 12 GiB memory
ceiling, 512 MiB scratch, 128 PIDs and a ten-minute cold-start deadline. These
are rehearsal bounds, not approved competition limits. The model gets a
read-only model tree and video, no network, no wallet and no reference answer.
Podman enforces the outer deadline and cleanup, including native code that
cannot be cancelled by an asyncio task. The CPU entrypoint emits English text
directly, with diagnostics on stderr. It preserves five-beam, 2048-token decoding.

Use a UMI revision supporting `umi-offline-cpu-runtime/2`. That version divides
the scratch allowance between `/tmp` (448 MiB here) and private `/dev/shm`
(64 MiB). Fairseq creates POSIX semaphore locks during import. A real-model run
exposed a read-only `/dev/shm` failure under runtime v1; v1 policies retain that
original mount behavior. The new runtime has a different policy-bound digest.

The receipt retains raw bounded output and the model/runtime identities. A
synthetic clip tests execution only. Accuracy, signer diversity, latency across
supported clip lengths and concurrent capacity need separate evaluation. This
serial cold-start evaluator is not a warm public miner sidecar.

## Rights, preservation and promotion

Keep the package's `PROVENANCE.md`, `LICENSE`, dependency licenses and file
inventories with every retained copy. The supplied SHuBERT tensor declaration is
MIT; wrapper/DINO/ByT5 components have separate notices. The manifest's `MIT`
identifier records the supplied model declaration and does not override those
notices or establish a rights review. This package does not inherit the separate
S1 model's CC BY-SA label.

The supplied provenance identifies public upstream revisions but expressly does
not provide a detailed training-data rights audit. Model integration does not
resolve that review. See UMI's
[contributor preparation checklist](https://github.com/Umi-BitSign/umi/blob/main/docs/MODEL_CONTRIBUTION_REVIEW.md).

An imported baseline has no contributor reward recipient. The 30% track requires
a qualifying promotion under the approved policy, including independent
evaluation, preservation and rights review. The importer and smoke command never
create that attribution or submit chain transactions.

An immutable object-store copy and restore verification are still required before
publishing a baseline download. No public artifact URL is assigned by this tool.
