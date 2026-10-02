# Studio competitive-model workspace

This checkout is the isolated starting point for improving UMI's public reference model.

## Bound starting point

- Source repository: https://github.com/Umi-BitSign/umi-reference-model
- Source commit: `abc1a32bf59b234b3f4910bfce3e008ae67fe3fd`
- Local branch: `studio-competitive-training`
- Frozen baseline model identity: `6fe8df59ec11ba89f4dfe0474a673fe757e378cd9184449965861c5a7c59b641`
- Independent local baseline: `artifacts/baseline-6fe8df59/`
- Baseline `MODEL_IDENTITY.json` SHA-256: `50b948db61f8d621af082d6f9bd79c78eb01de7861b1c0807735b2b2e290f4a3`

The artifact directory is an APFS copy-on-write clone. It begins byte-identical to the preserved baseline but is independent of the live miner. The live serving release and its state remain under `/Users/sam/.local/share/umi/`; do not edit or repoint them from this workspace.

## First objective

Produce a reproducible model that beats the frozen baseline on signer-disjoint development data and remains compatible with the published UMI bundle runtime. Keep a clean experiment ledger so a result can be reconstructed, licensed, packaged and submitted to the public-model track.

Start by reproducing baseline inference and measuring its task-level errors. Then make one measurable change at a time. Favor data quality, split integrity and a strong fingerspelling path over blind hyperparameter sweeps.

## Hard boundaries

- Keep all training data, checkpoints, runs and local artifacts in the ignored `data/`, `checkpoints/`, `runs/` and `artifacts/` trees.
- Never train on service or evaluation material from any cohort, private holdout labels, miner assignments, or cohort result packages. Operational corpora under `/Users/sam/.local/share/umi/` are excluded from training.
- Keep source recording, session and signer groups in one split. Detect exact duplicates, re-encodes and overlapping intervals before splitting.
- Do not use test or devtest for model selection. Record every evaluation call.
- Preserve each source's original bytes, license, attribution and transform history. Public availability does not imply commercial training rights.
- Do not download multi-terabyte sources until storage and rights are settled. The Studio had about 309 GiB free when this workspace was created; FSboard alone is roughly 1.3 TiB.
- Do not stop, reconfigure or load-test the live miner. Schedule sustained training only after checking memory, GPU and miner request load.

Read `DATA_SOURCES.md` before selecting data. It records all sources currently known to the project, including noncommercial and permission-gated collections.
