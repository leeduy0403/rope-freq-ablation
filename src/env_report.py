"""Phase 0: print and save environment info. Gate G0 = a GPU is visible."""
import argparse
import sys

import torch

from src.config import lib_versions, write_json


def report():
    info = lib_versions()
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        free, total = torch.cuda.mem_get_info(0)
        info.update(
            gpu_name=torch.cuda.get_device_name(0),
            gpu_capability=".".join(map(str, torch.cuda.get_device_capability(0))),
            bf16_supported=torch.cuda.is_bf16_supported(),
            vram_total_gb=round(total / 2**30, 2),
            vram_free_gb=round(free / 2**30, 2),
        )
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/env.json")
    args = ap.parse_args()
    info = report()
    for k, v in info.items():
        print(f"{k:16s} {v}")
    write_json(args.out, info)
    print(f"\nsaved -> {args.out}")
    if not info["cuda_available"]:
        print("GATE G0: FAIL — no GPU visible")
        sys.exit(1)
    print("GATE G0: PASS")


if __name__ == "__main__":
    main()
