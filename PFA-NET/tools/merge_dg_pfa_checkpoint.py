"""Merge learned DG-only parameters with an untouched PFA checkpoint."""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    args = parse_args()
    baseline = load_checkpoint(args.baseline)
    adapted = load_checkpoint(args.adapted)
    baseline_state = baseline["model_state"]
    adapted_state = OrderedDict(adapted["model_state"])

    restored_keys = []
    for key, value in baseline_state.items():
        if key in adapted_state and adapted_state[key].shape == value.shape:
            adapted_state[key] = value.clone()
            restored_keys.append(key)

    dg_only_keys = [key for key in adapted_state if key not in baseline_state]
    merged = dict(adapted)
    merged["model_state"] = adapted_state
    merged["optimizer_state"] = None
    merged["merge_metadata"] = {
        "baseline": str(args.baseline.resolve()),
        "adapted": str(args.adapted.resolve()),
        "restored_baseline_keys": len(restored_keys),
        "retained_dg_only_keys": len(dg_only_keys),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(
        f"wrote {args.output}: restored {len(restored_keys)} baseline keys, "
        f"retained {len(dg_only_keys)} DG-only keys"
    )


if __name__ == "__main__":
    main()
