"""Isolate raw-RGB visual backward errors with matched CPU/Metal activation gradients."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


def main() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    import torch
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_fit import ROOT, RgbDutyCycle, metal_working_set_bytes
    from tiny_rgb_model import RmsNorm
    from tiny_rgb_mps_safe import PortableGelu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("operation probe requires a new private runs/ receipt")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(6 * 1024**3 / metal_working_set_bytes())
    duty = RgbDutyCycle(torch, ROOT / "runs/c5-resource-policy-v1/gpu-budget.json")
    records = []
    for name in ("native_gelu", "portable_gelu", "channel_rms"):
        for strided in (False, True):
            torch.manual_seed(317)
            source = torch.randn(16, 96, 12, 22)
            if strided:
                upstream = torch.randn(16, 12 * 22, 96).transpose(1, 2).reshape(source.shape)
                assert not upstream.is_contiguous()
            else:
                upstream = torch.randn_like(source)
            module = {
                "native_gelu": torch.nn.GELU(),
                "portable_gelu": PortableGelu(),
                "channel_rms": RmsNorm(96, channel_first=True),
            }[name]
            cpu_input = source.clone().requires_grad_()
            cpu_result = module(cpu_input)
            cpu_result.backward(upstream)
            cpu_gradient = cpu_input.grad.clone()
            metal_module = copy.deepcopy(module).to("mps")
            metal_input = source.to("mps").requires_grad_()

            def probe(
                metal_module=metal_module,
                metal_input=metal_input,
                upstream=upstream,
                cpu_result=cpu_result,
                cpu_gradient=cpu_gradient,
            ):
                metal_result = metal_module(metal_input)
                metal_result.backward(upstream.to("mps"))
                gradient = metal_input.grad.detach().cpu()
                return {
                    "forward_max_abs_error": float(
                        (metal_result.detach().cpu() - cpu_result.detach()).abs().max()
                    ),
                    "cpu_input_gradient_norm": float(cpu_gradient.norm()),
                    "mps_input_gradient_norm": float(gradient.norm()),
                    "gradient_relative_l2_error": float(
                        (gradient - cpu_gradient).norm() / cpu_gradient.norm()
                    ),
                }

            record = {"operation": name, "strided_upstream_gradient": strided, **duty.run(probe)}
            records.append(record)
            print(json.dumps(record), flush=True)
            del probe, metal_module, metal_input
            torch.mps.empty_cache()
    receipt = {
        "schema": "umi-studio-tiny-rgb-operation-numerics/1",
        "source_sha256": sha256(Path(__file__)),
        "cases": records,
        "optimizer_updates": 0,
        "weights_saved": False,
        **duty.receipt(),
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
