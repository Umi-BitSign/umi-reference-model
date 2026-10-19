"""Resume saved512 under standing approval to train a substantially better model."""

from __future__ import annotations

import json
from pathlib import Path

import train_c5_word_storage_recovery as template
from benchmark_fleurs_dev import sha256
from c5_word_http502_recovery import (
    checked_authorization,
    checked_binding,
    claim,
    inherit_prefix,
    make_binding,
    read,
    restore_committed,
)
from launch_terra_visual_job import write
from terra_polarity_judge import bind
from terra_training_recovery import artifact, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "umi-studio-C5-polarity-HTTP502-recovery/1"
SOURCES = (
    "train_c5_word_http502_recovery.py",
    "c5_word_http502_recovery.py",
    "c5_word_http502_final.py",
    "audit_c5_word_http502_lineage.py",
    "evaluate_c5_word_http502_final.py",
    "evaluate_c5_word_http502_words.py",
    "evaluate_c5_word_http502_sentences.py",
)
derived_fit_ast = template.derived_fit_ast
lineage_counts = template.lineage_counts
store_for = template.store_for


def bound_fit():
    # Preserve the complete already reviewed per-batch loop and rolling saves.
    return bind(template.bound_fit, checked=checked, restore_committed=restore_committed)()


def specification(path, source_path, checkpoint, audit_path, authorization_path):
    binding = make_binding(source_path, checkpoint, audit_path, authorization_path)
    source = read(source_path)
    sources = dict(source["source_sha256"])
    for name in SOURCES:
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        **source,
        "schema": SCHEMA,
        "source_sha256": sources,
        "resume": binding,
        "initialization": "explicitly approved saved512 state from the original public-C5 lineage",
        "manual_continuations_authorized": None,
        "explicit_second_HTTP502_recovery_exception": binding["authorization"],
        "automatic_further_continuations": True,
        "continuation_claim_limit_per_source": 1,
        "stopping_condition": "training a substantially better model",
        "lineage_audit_source": {
            "path": str(path.with_name("terminal-audit.py").relative_to(ROOT)),
            "sha256": sha256(ROOT / "tools/audit_c5_word_http502_lineage.py"),
        },
    }


def checked(path):
    path = path.resolve()
    plan = read(path)
    binding = plan["resume"]
    expected = specification(
        path,
        ROOT / binding["source_plan"]["path"],
        ROOT / binding["visual"]["path"],
        ROOT / binding["terminal_audit"]["path"],
        ROOT / binding["authorization"]["path"],
    )
    if (
        plan != expected
        or artifact(ROOT / plan["lineage_audit_source"]["path"]) != plan["lineage_audit_source"]
    ):
        raise ValueError("HTTP502 recovery changed its approved source, recipe or final auditor")
    verify_snapshot(path, plan)
    return plan


def freeze(path, source_path, checkpoint, audit_path, authorization_path):
    from c5_experiment_sources import archive_sources

    checked_authorization(authorization_path)
    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("HTTP502 recovery requires a fresh owned plan")
    template.word.storage(template.word.STORAGE["checkpoint_pair_reserve_bytes"])
    plan = specification(
        path,
        source_path.resolve(),
        checkpoint.resolve(),
        audit_path.resolve(),
        authorization_path.resolve(),
    )
    path.parent.mkdir(parents=True)
    with path.with_name("terminal-audit.py").open("xb") as stream:
        stream.write((ROOT / "tools/audit_c5_word_http502_lineage.py").read_bytes())
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan": artifact(path),
                "inherited_exposures": 512,
                "remaining_exposures": 344,
                "newly_rolled_back_exposures": 0,
                "actual_lineage_exposures_if_complete": 960,
            }
        ),
        flush=True,
    )


def fit(path, output, plan):
    checked_binding(plan["resume"])
    write(output / "resume-spacing.json", template.resumed.source_cooldown(plan["resume"]))
    claim(path, output, plan)
    inheritance = inherit_prefix(output, plan)
    store_for(path, output).seed(ROOT / inheritance["working_seed"]["commit"]["path"])
    return bound_fit()(path, output, plan)


def run(path, output):
    return bind(template.original.run, checked=checked, fit=fit)(path, output)


def main(argv=None):
    return bind(
        template.main,
        checked_authorization=checked_authorization,
        freeze=freeze,
        checked=checked,
        run=run,
    )(argv)


if __name__ == "__main__":
    main()
