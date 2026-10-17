"""A fresh complete word curriculum from public C5, after actual polarity risk qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import probe_c5_polarity_semantic as proof
import train_c5_word_semantic as original
from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import now, write
from terra_polarity_judge import Pacer, bind, protocol, score_training_batch
from terra_training_recovery import artifact, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
PROOF_PLAN = ROOT / "runs/c5-polarity-semantic-gradient-v1/plan.json"
PROOF = ROOT / "checkpoints/c5-polarity-semantic-gradient-v1/qualify/qualification.json"
SCHEMA = "umi-studio-C5-polarity-word-training/1"
SOURCES = (
    "train_c5_polarity_words.py",
    "c5_polarity_word_final.py",
    "audit_c5_polarity_word.py",
    "c5_word_final.py",
    "evaluate_c5_polarity_words.py",
    "evaluate_c5_polarity_sentences.py",
    "evaluate_c5_polarity_final.py",
    "evaluate_c5_words.py",
    "evaluate_c5_word_sentences.py",
    "evaluate_c5_word_final.py",
    "c5_word_evaluation_io.py",
    "c5_mixed_risk_evidence.py",
    "c5_mixed_semantic_model.py",
    "probe_c5_mixed_semantic.py",
    "terra_semantic_selection.py",
)


def qualified():
    return bind(
        original.qualified,
        PROOF_PLAN=PROOF_PLAN,
        PROOF=PROOF,
        checked_proof_plan=proof.checked,
        checked_qualification=proof.checked_qualification,
    )()


def specification(path):
    base = bind(original.specification, qualified=qualified, PROOF_PLAN=PROOF_PLAN, PROOF=PROOF)()
    if base["model_plan"].get("judge_protocol") != protocol():
        raise ValueError("fresh word training requires the actually qualified polarity protocol")
    sources = dict(base["source_sha256"])
    for name in SOURCES:
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        **base,
        "schema": SCHEMA,
        "judge_protocol": protocol(),
        "source_sha256": sources,
        "initialization": "original public C5; no inherited exposures, Adam updates or RNG state",
        "single_optimizer_execution_per_proof": True,
        "manual_continuations_authorized": 0,
        "failed_old_word_lineage_remains_exhausted": True,
        "terminal_audit_source": {
            "path": str(path.with_name("terminal-audit.py").relative_to(ROOT)),
            "sha256": sha256(ROOT / "tools/audit_c5_polarity_word.py"),
        },
    }


def checked(path):
    path = path.resolve()
    plan = json.loads(path.read_text())
    if "resume" in plan or plan != specification(path):
        raise ValueError(
            "fresh polarity training changed public initialization, complete recipe or proof"
        )
    if artifact(ROOT / plan["terminal_audit_source"]["path"]) != plan["terminal_audit_source"]:
        raise ValueError("fresh polarity training changed its archived terminal auditor")
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("fresh polarity curriculum requires a fresh owned plan")
    plan = specification(path)
    original.storage(original.STORAGE["checkpoint_pair_reserve_bytes"])
    path.parent.mkdir(parents=True)
    write(path, plan)
    with path.with_name("terminal-audit.py").open("xb") as stream:
        stream.write((ROOT / "tools/audit_c5_polarity_word.py").read_bytes())
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan": artifact(path),
                "training_clips": len(plan["records"]),
                "exposures": plan["planned_training_examples"],
                "inherited_exposures": 0,
                "optimizer_executions_authorized": 1,
            }
        ),
        flush=True,
    )


def claim(path, output, plan):
    record = {
        "plan": artifact(path),
        "worker": artifact(output / "worker.json"),
        "output": str(output.relative_to(ROOT)),
        "qualified_plan": plan["qualified_plan"],
        "claimed_at": now(),
        "single_execution": True,
        "inherited_exposures": 0,
        "inherited_optimizer_updates": 0,
        "manual_continuations_authorized": 0,
    }
    source = (ROOT / plan["qualified_plan"]["path"]).with_name("optimizer-child-claim.json")
    write(source, record)
    write(path.with_name("execution-claim.json"), record)


def bound_fit():
    return bind(
        original.fit, checked=checked, RecoveryPacer=Pacer, score_batch=score_training_batch
    )


def fit(path, output, plan):
    # original.run holds the existing exclusive C5 lease before calling fit.
    claim(path, output, plan)
    return bound_fit()(path, output, plan)


def run(path, output):
    return bind(original.run, checked=checked, fit=fit)(path, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "plan", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    original.torch.set_num_interop_threads(1)
    if args.freeze:
        freeze(args.freeze)
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output:
        run(args.plan, args.output)
    else:
        parser.error("freeze after completed qualification, or run one fresh frozen curriculum")


if __name__ == "__main__":
    main()
