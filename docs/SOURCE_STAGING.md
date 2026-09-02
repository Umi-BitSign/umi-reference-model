# Runtime source staging

The portable bundle hashes its inference runtime modules. Copying those files before
the final rights-bound runtime is frozen would make the public checkout disagree with
the release identity.

After the private release branch passes its rights and export tests, review
`release/runtime-files.txt` against the runtime closure recorded by the final bundle.
Add every imported runtime and backend module before staging. Training-only code must
not enter the public closure. Then run:

```bash
uv run --frozen --extra dev python tools/stage_runtime.py \
  --source ../bitsign-motion-research-private \
  --manifest release/runtime-files.txt \
  --check-only

uv run --frozen --extra dev python tools/stage_runtime.py \
  --source ../bitsign-motion-research-private \
  --manifest release/runtime-files.txt
```

The command accepts regular files from the explicit allowlist, rejects symlinks and
path traversal, and refuses to overwrite an existing destination. Review the staged
diff before using `--replace` on a later refresh.

After staging, run:

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev pytest
uv run --frozen --extra dev python tools/repo_guard.py
```

Compare the SHA-256 of every module named in `inference-identity.json` with the staged
checkout. Also compare the preprocessing source hashes for the Docker workers,
requirements locks, mapping, and motion conversion. A mismatch blocks publication.

Two integration test files are maintained as public adaptations rather than copied by
the allowlist. The backend test stubs task-file verification only while testing that
bundle rejection happens before Docker access. The AMD64 smoke test receives its video
and task paths through explicit environment variables rather than local research paths.
Production task hash verification remains unchanged in the staged backend.
