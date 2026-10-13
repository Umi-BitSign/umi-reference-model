"""Distill cached C5 encoder states into the compact fresh stream encoder."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from benchmark_fleurs_dev import sha256
from train_fresh_lowres import language_artifact_receipt


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush(); os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--language-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--mps-idle-fraction", type=float, default=0.5)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (
        args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to((root / "checkpoints").resolve())
        or args.epochs < 1 or args.learning_rate <= 0 or args.gradient_accumulation < 1
        or args.checkpoint_every < 1 or not 0 <= args.mps_idle_fraction < 1
        or not args.miner_maintenance_mode
    ):
        parser.error("invalid distillation bounds or miner policy")
    for key, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(root / "artifacts/fresh-stream-v1/huggingface"),
        "TORCH_HOME": str(root / "artifacts/fresh-stream-v1/torch"),
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    }.items(): os.environ[key] = value
    import torch
    import torch.nn.functional as F
    from fresh_stream_data import load_inventory, load_stream_tensor, load_teacher_tensor
    from fresh_stream_model import FreshStreamTranslator, configure_visual_pretraining
    from safetensors.torch import save_file
    from transformers import T5ForConditionalGeneration

    torch.set_num_threads(2); torch.manual_seed(args.seed); random.seed(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available(): parser.error("MPS unavailable")
    receipt = language_artifact_receipt(root, args.language_model)
    rows = load_inventory(args.inventory, root)
    language_model = T5ForConditionalGeneration.from_pretrained(args.language_model, local_files_only=True)
    model = FreshStreamTranslator(language_model)
    visual = configure_visual_pretraining(model)
    projection = torch.nn.Linear(256, 1536, bias=False)
    model.to(args.device); projection.to(args.device)
    parameters = {**visual, "distill_projection.weight": projection.weight}
    optimizer = torch.optim.AdamW(parameters.values(), lr=args.learning_rate, weight_decay=0.01)
    schedule = []
    for epoch in range(1, args.epochs + 1):
        ordered = list(rows); random.Random(args.seed + epoch).shuffle(ordered)
        schedule.extend((epoch, row) for row in ordered)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True); args.output_dir.mkdir()
    run = {
        "schema": "umi-studio-fresh-stream-distillation-run/1",
        "inventory_sha256": sha256(args.inventory), "language_model": receipt,
        "examples": len(rows), "epochs": args.epochs, "updates": len(schedule),
        "learning_rate": args.learning_rate, "gradient_accumulation": args.gradient_accumulation,
        "seed": args.seed, "device": args.device, "mps_idle_fraction": args.mps_idle_fraction,
        "teacher": "cached-c5-encoder-hidden-state-1536", "student_width": 256,
        "model_source_sha256": sha256(Path(__file__).with_name("fresh_stream_model.py")),
        "data_source_sha256": sha256(Path(__file__).with_name("fresh_stream_data.py")),
        "trainer_source_sha256": sha256(Path(__file__)),
        "miner_health_policy": "user_authorized_maintenance",
    }
    run_path=args.output_dir/"run.json"; run_path.write_text(json.dumps(run,sort_keys=True,indent=2)+"\n")
    ledger=args.output_dir/"steps.jsonl"; verified=set(); optimizer.zero_grad(set_to_none=True)
    model.train()

    def save(path: Path, update: int) -> None:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in visual.items()}, str(path),
            metadata={"schema":"umi-studio-fresh-stream-distillation/1","run_sha256":sha256(run_path),"update":str(update)},
        )

    for update,(epoch,row) in enumerate(schedule,start=1):
        started=time.perf_counter(); identity=(row["source"],row["example_id"])
        streams=load_stream_tensor(row,verify_hash=identity not in verified)
        teacher=load_teacher_tensor(row,verify_hash=False); verified.add(identity)
        x=torch.from_numpy(streams).unsqueeze(0).to(args.device)
        target=torch.from_numpy(teacher).unsqueeze(0).to(args.device)
        mask=torch.ones((1,len(streams)),dtype=torch.bool,device=args.device)
        hidden,_=model.encode_visual(x,mask); predicted=projection(hidden)
        loss=(1-F.cosine_similarity(predicted,target,dim=-1)).mean()
        if not torch.isfinite(loss): raise RuntimeError("non-finite distillation loss")
        start=((update-1)//args.gradient_accumulation)*args.gradient_accumulation+1
        end=min(start+args.gradient_accumulation-1,len(schedule)); size=end-start+1
        (loss/size).backward(); norm=None
        if update==end:
            norm=torch.nn.utils.clip_grad_norm_(parameters.values(),1.0)
            if not torch.isfinite(norm): raise RuntimeError("non-finite distillation gradient")
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if args.device=="mps": torch.mps.synchronize()
        active=time.perf_counter()-started; idle=0.0
        if args.device=="mps" and args.mps_idle_fraction:
            idle=active*args.mps_idle_fraction/(1-args.mps_idle_fraction); time.sleep(idle)
        append_jsonl(ledger,{"update":update,"epoch":epoch,"source":row["source"],"example_id":row["example_id"],"frames":len(streams),"cosine_loss":float(loss.detach().cpu()),"gradient_norm":float(norm.detach().cpu()) if norm is not None else None,"active_seconds":active,"idle_seconds":idle})
        print(f"update={update} source={row['source']} frames={len(streams)} cosine_loss={float(loss.detach().cpu()):.4f} seconds={active+idle:.2f}",flush=True)
        if update%args.checkpoint_every==0 and update!=len(schedule): save(args.output_dir/f"visual-after-{update:06d}.safetensors",update)
    checkpoint=args.output_dir/"visual.safetensors"; save(checkpoint,len(schedule))
    print(json.dumps({"checkpoint":str(checkpoint),"sha256":sha256(checkpoint)},sort_keys=True))


if __name__ == "__main__": main()
