# Live native-worker bundle contract

The launch worker verifies bundle identity
`2fd4c30f5455e29630ffa06ad2c3cbd674b5bd4fa47aaaf1ea9b78fc40d95c58`.
It is the preserved historical native bundle with only the manifest entry and
bytes for `umi_inference.py` replaced by the file in this directory. The exact
derivation and both identities are pinned in `contract.json`.

The worker imports `runtime.py` directly through `community/native_worker.py`;
it does not execute the bundle's offline CPU entrypoint. The entrypoint remains
part of the verified bundle identity and is published here so every unique live
source byte is available. The historical `community/native` bundle remains
unchanged at identity `ce459641c180c680aae32009052985165fa8bbffd165b0dca05d2dade0e619ed`.
