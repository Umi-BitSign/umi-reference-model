"""Synthetic file transactions only; no trained weights or optimizer are used."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_word_working_checkpoints as working


def append(output, examples, updates):
    path = output / "steps.jsonl"
    count = len(path.read_bytes().splitlines()) if path.exists() else 0
    with path.open("a") as stream:
        stream.write(
            json.dumps(
                {"batch": count, "training_examples": examples, "optimizer_updates": updates}
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint(workspace, directory, examples, updates):
    visual = directory / f"visual-after-{examples:06d}.safetensors"
    optimizer = directory / f"visual-after-{examples:06d}-optimizer.safetensors"
    visual.write_bytes(f"synthetic visual {examples}".encode())
    optimizer.write_bytes(f"synthetic optimizer {updates}".encode())

    def descriptor(path):
        return {"path": str(path.relative_to(workspace)), "sha256": working.file_sha(path)}

    commit = visual.with_suffix(".json")
    commit.write_text(
        json.dumps(
            {
                "complete": True,
                "committed_batch_RNG": True,
                "training_examples": examples,
                "optimizer_updates": updates,
                "visual": descriptor(visual),
                "optimizer": descriptor(optimizer),
            }
        )
    )
    return commit


@pytest.fixture
def seeded(tmp_path):
    output = tmp_path / "checkpoints/recovery"
    output.mkdir(parents=True)
    reservations = []
    store = working.WorkingCheckpoints(
        output,
        workspace=tmp_path,
        plan_sha="a" * 64,
        storage_check=reservations.append,
        pair_reserve_bytes=4096,
    )
    append(output, 4, 1)
    seed = checkpoint(tmp_path, output, 4, 1)
    store.seed(seed)
    return store, tmp_path, output, reservations


def save(fixture, examples, updates):
    store, workspace, output, _ = fixture
    append(output, examples, updates)
    return store.commit(
        examples=examples,
        updates=updates,
        writer=lambda directory: checkpoint(workspace, directory, examples, updates),
    )


def test_next_batch_has_a_complete_bound_state_and_reservations(seeded):
    store, workspace, _, reservations = seeded
    state = save(seeded, 8, 2)
    assert store.latest() == state
    assert state["training_examples"] == 8 and state["optimizer_updates"] == 2
    assert state["ledger_rows"] == 2 and state["inherited_seed"] is False
    for key in ("visual", "optimizer", "commit"):
        assert working.file_sha(workspace / state[key]["path"]) == state[key]["sha256"]
    assert reservations == [16384, 4096 + 16384, 16384]


def test_tied_batch_can_advance_exposures_without_an_optimizer_step(seeded):
    assert save(seeded, 8, 1)["optimizer_updates"] == 1


def test_failed_writer_preserves_previous_state_and_is_not_retried(seeded):
    store, _, output, _ = seeded
    previous = store.latest()
    append(output, 8, 2)
    calls = []

    def fail(directory):
        calls.append(directory)
        (directory / "partial.safetensors").write_bytes(b"partial")
        raise OSError("synthetic disk failure")

    with pytest.raises(OSError, match="disk failure"):
        store.commit(examples=8, updates=2, writer=fail)
    assert store.latest() == previous and len(calls) == 1
    assert (calls[0] / "partial.safetensors").exists()
    with pytest.raises(ValueError, match="never retried"):
        store.commit(examples=8, updates=2, writer=fail)
    assert len(calls) == 1


def test_storage_failure_occurs_before_writer_or_directory_creation(seeded):
    store, _, output, _ = seeded
    append(output, 8, 2)

    def fail(_):
        raise OSError("reserve lost")

    store.storage_check = fail
    with pytest.raises(OSError, match="reserve lost"):
        store.commit(examples=8, updates=2, writer=lambda _: pytest.fail("writer called"))
    assert store.latest()["training_examples"] == 4
    assert not store.directory.exists()


def test_failure_before_pointer_replacement_keeps_previous_pair(seeded, monkeypatch):
    store, _, _, _ = seeded
    previous = store.latest()

    def fail(_):
        raise OSError("pointer publication failed")

    monkeypatch.setattr(store, "_publish", fail)
    with pytest.raises(OSError, match="publication"):
        save(seeded, 8, 2)
    assert store.latest() == previous
    assert len(list(store.directory.rglob("*.safetensors"))) == 2
    assert any(
        event["kind"] == "prepared" and event["state"]["training_examples"] == 8
        for event in store._events()
    )


def test_published_pointer_remains_valid_if_completion_journal_append_fails(seeded, monkeypatch):
    store, _, _, _ = seeded
    original = store._append

    def fail(event):
        if event["kind"] == "committed" and event["state"]["training_examples"] == 8:
            raise OSError("completion append failed")
        original(event)

    monkeypatch.setattr(store, "_append", fail)
    with pytest.raises(OSError, match="completion append"):
        save(seeded, 8, 2)
    assert store.latest()["training_examples"] == 8


def test_retirement_preserves_two_pairs_all_commit_receipts_and_seed(seeded):
    store, workspace, output, _ = seeded
    seed = store.latest()
    states = [save(seeded, n, n // 4) for n in (8, 12, 16, 20)]
    for state in states[:2]:
        assert not (workspace / state["visual"]["path"]).exists()
        assert not (workspace / state["optimizer"]["path"]).exists()
        assert (workspace / state["commit"]["path"]).is_file()
    for state in [seed, *states[-2:]]:
        assert all(
            (workspace / state[key]["path"]).is_file() for key in ("visual", "optimizer", "commit")
        )
    assert len((output / "steps.jsonl").read_bytes().splitlines()) == 5
    assert len(list(store.directory.rglob("*.safetensors"))) == 4
    store.retire()
    assert store.latest() == states[-1]


def test_changed_tensor_or_prefix_is_rejected(seeded):
    store, workspace, output, _ = seeded
    state = save(seeded, 8, 2)
    path = workspace / state["optimizer"]["path"]
    original = path.read_bytes()
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes changed"):
        store.latest()
    path.write_bytes(original)
    ledger = output / "steps.jsonl"
    ledger.write_bytes(
        ledger.read_bytes().replace(b'"training_examples": 4', b'"training_examples": 5')
    )
    with pytest.raises(ValueError, match="prefix changed"):
        store.latest()


def test_latest_accepts_its_intact_prefix_even_with_a_partial_later_append(seeded):
    store, _, output, _ = seeded
    previous = store.latest()
    with (output / "steps.jsonl").open("ab") as stream:
        stream.write(b'{"uncommitted":')
    assert store.latest() == previous
    with pytest.raises(ValueError, match="complete nonempty ledger"):
        store.commit(examples=8, updates=2, writer=lambda _: pytest.fail("writer called"))


@pytest.mark.parametrize("examples,updates", [(9, 2), (8, 3), (8, True)])
def test_counter_jumps_or_boolean_counters_are_rejected(seeded, examples, updates):
    store, _, output, _ = seeded
    append(output, examples, updates)
    with pytest.raises(ValueError, match="exactly one complete"):
        store.commit(
            examples=examples, updates=updates, writer=lambda _: pytest.fail("writer called")
        )


def test_wrong_writer_counters_are_rejected_before_publication(seeded):
    store, workspace, output, _ = seeded
    append(output, 8, 2)
    with pytest.raises(ValueError, match="counters differ"):
        store.commit(
            examples=8, updates=2, writer=lambda directory: checkpoint(workspace, directory, 12, 3)
        )
    assert store.latest()["training_examples"] == 4


def test_writer_cannot_point_to_a_root_epoch(seeded):
    store, workspace, output, _ = seeded
    seed = workspace / store.latest()["commit"]["path"]
    append(output, 8, 2)
    with pytest.raises(ValueError, match="escaped"):
        store.commit(examples=8, updates=2, writer=lambda _: seed)
    assert seed.is_file() and store.latest()["training_examples"] == 4


def test_storage_accounting_includes_files_outside_working_directory(seeded, monkeypatch):
    store, _, output, _ = seeded
    before = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    monkeypatch.setattr(working, "BYTE_BUDGET", before + 4096 + 16384 + 1000)
    (output / "other-existing-state.safetensors").write_bytes(b"x" * 4096)
    append(output, 8, 2)
    with pytest.raises(ValueError, match="whole-run byte budget"):
        store.commit(examples=8, updates=2, writer=lambda _: pytest.fail("writer called"))


def test_symlink_control_paths_are_rejected_without_touching_the_target(seeded):
    store, workspace, _, _ = seeded
    outside = workspace / "outside.jsonl"
    outside.write_bytes(b"untouched")
    store.journal.unlink()
    store.journal.symlink_to(outside)
    with pytest.raises(ValueError, match="aliases"):
        store._append({"kind": "unsafe"})
    assert outside.read_bytes() == b"untouched"


def test_retirement_failure_keeps_latest_committed_pair_recoverable(seeded, monkeypatch):
    store, workspace, _, _ = seeded
    oldest = save(seeded, 8, 2)
    save(seeded, 12, 3)
    target = workspace / oldest["visual"]["path"]
    unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path == target:
            raise OSError("retirement interrupted")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    with pytest.raises(OSError, match="retirement interrupted"):
        save(seeded, 16, 4)
    assert store.latest()["training_examples"] == 16
    monkeypatch.setattr(Path, "unlink", unlink)
    store.retire()
    assert not target.exists() and store.latest()["training_examples"] == 16


def test_retirement_intent_handles_a_crash_after_unlink_before_acknowledgment(seeded, monkeypatch):
    store, workspace, _, _ = seeded
    oldest = save(seeded, 8, 2)
    save(seeded, 12, 3)
    original = store._append

    def fail(event):
        if event["kind"] == "retired" and event["artifact"] == oldest["visual"]:
            raise OSError("retirement acknowledgment failed")
        original(event)

    monkeypatch.setattr(store, "_append", fail)
    with pytest.raises(OSError, match="acknowledgment"):
        save(seeded, 16, 4)
    assert not (workspace / oldest["visual"]["path"]).exists()
    assert store.latest()["training_examples"] == 16
    monkeypatch.setattr(store, "_append", original)
    store.retire()
    assert not (workspace / oldest["optimizer"]["path"]).exists()


def test_forged_retirement_reference_cannot_delete_a_retained_pair(seeded):
    store, workspace, _, _ = seeded
    save(seeded, 8, 2)
    save(seeded, 12, 3)
    latest = save(seeded, 16, 4)
    events = store._events()
    for event in events:
        if event["kind"] == "committed" and event["state"]["training_examples"] == 8:
            event["state"]["visual"] = latest["visual"]
    store.journal.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValueError, match="retirement cannot reach"):
        store.retire()
    assert (workspace / latest["visual"]["path"]).exists()


def test_ledger_change_during_writer_prevents_publication(seeded):
    store, workspace, output, _ = seeded
    append(output, 8, 2)

    def writer(directory):
        path = checkpoint(workspace, directory, 8, 2)
        append(output, 12, 3)
        return path

    with pytest.raises(ValueError, match="ledger changed during"):
        store.commit(examples=8, updates=2, writer=writer)
    assert store.latest()["training_examples"] == 4


def test_missing_optimizer_cannot_be_published(seeded):
    store, workspace, output, _ = seeded
    append(output, 8, 2)

    def writer(directory):
        path = checkpoint(workspace, directory, 8, 2)
        report = json.loads(path.read_text())
        (workspace / report["optimizer"]["path"]).unlink()
        return path

    with pytest.raises(ValueError, match="bytes changed"):
        store.commit(examples=8, updates=2, writer=writer)
    assert store.latest()["training_examples"] == 4


def test_artifact_sync_failure_preserves_the_previous_pointer(seeded, monkeypatch):
    store, _, _, _ = seeded
    original = store.latest()

    def fail(_):
        raise OSError("tensor flush failed")

    monkeypatch.setattr(store, "_sync_artifacts", fail)
    with pytest.raises(OSError, match="tensor flush"):
        save(seeded, 8, 2)
    assert store.latest() == original
    assert not any(
        event["kind"] == "prepared" and event["state"]["training_examples"] == 8
        for event in store._events()
    )


def test_an_incorrect_batch_index_is_rejected_before_writer(seeded):
    store, _, output, _ = seeded
    append(output, 8, 2)
    path = output / "steps.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["batch"] = 19
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="contiguous prefix"):
        store.commit(examples=8, updates=2, writer=lambda _: pytest.fail("writer called"))
    assert store.latest()["training_examples"] == 4


def test_tensor_pair_cannot_be_two_hardlinks_to_one_file(seeded):
    store, workspace, output, _ = seeded
    append(output, 8, 2)

    def writer(directory):
        path = checkpoint(workspace, directory, 8, 2)
        report = json.loads(path.read_text())
        first = workspace / report["visual"]["path"]
        second = workspace / report["optimizer"]["path"]
        second.unlink()
        os.link(first, second)
        report["optimizer"]["sha256"] = working.file_sha(second)
        path.write_text(json.dumps(report))
        return path

    with pytest.raises(ValueError, match="aliased tensor pair"):
        store.commit(examples=8, updates=2, writer=writer)
    assert store.latest()["training_examples"] == 4
