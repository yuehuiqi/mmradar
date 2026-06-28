#!/usr/bin/env python3
"""Run all 14 MMRadar models (original 7 + extended 7) for one dataset/class
configuration with a single command.

This is a thin orchestrator over the two existing suite runners:
  * run_mmradar_full_suite_wsl.py     -> original 7 models
  * run_mmradar_extended_suite_wsl.py -> extended 7 models

The three intended top-level invocations (``--config`` picks one):
  * aiqii_single     : single-class aiQii (all UAVs folded to ``Drone``)
  * aiqii_multiclass : 4-class aiQii (Air3S / Mini4pro / Mavic3Pro / jingling4)
  * mmaud            : MMAUD, single-class ``Drone``

The original separate commands keep working unchanged; this only adds a wrapper.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FULL_SUITE = HERE / "run_mmradar_full_suite_wsl.py"
EXTENDED_SUITE = HERE / "run_mmradar_extended_suite_wsl.py"

CONFIGS = {
    # config name        -> (dataset, aiqii_classes)
    "aiqii_single": ("aiqii", "single"),
    "aiqii_multiclass": ("aiqii", "multiclass"),
    "mmaud": ("mmaud", "single"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all 14 MMRadar models (original 7 + extended 7) for one "
        "dataset/class configuration.",
    )
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS),
                        help="Which dataset/class configuration to train all 14 models on.")
    parser.add_argument("--run-tag", default="all14_v1",
                        help="Unique tag for this batch; re-use it to resume.")
    parser.add_argument("--mode", default="full", choices=("full", "smoke"),
                        help="full = 80-epoch training; smoke = 1-epoch pipeline check "
                             "(only affects the extended suite; the original suite is always full).")
    parser.add_argument("--suites", default="both", choices=("both", "original", "extended"),
                        help="Run both suites (default), or only one of them.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override every model's train batch size.")
    parser.add_argument("--pcdet-batch-size", type=int, default=None,
                        help="Override OpenPCDet/RadarPillar-style batch size only.")
    parser.add_argument("--det3d-batch-size", type=int, default=None,
                        help="Override CenterPoint/PillarNet-LTS batch size (original suite only).")
    parser.add_argument("--radarnext-batch-size", type=int, default=None,
                        help="Override RadarNeXt batch size (extended suite only).")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--best-metric", default="center_ap_2m",
                        help="Metric used by the child suites to select the best validation epoch.")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="Abort the whole batch as soon as one model fails.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the underlying suite commands without training.")
    return parser.parse_args()


def build_full_cmd(args, dataset, aiqii_classes) -> list[str]:
    cmd = [sys.executable, str(FULL_SUITE),
           "--dataset", dataset,
           "--aiqii-classes", aiqii_classes,
           "--run-tag", args.run_tag,
           "--workers", str(args.workers),
           "--retries", str(args.retries),
           "--gpu", args.gpu,
           "--best-metric", args.best_metric]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.pcdet_batch_size is not None:
        cmd += ["--pcdet-batch-size", str(args.pcdet_batch_size)]
    if args.det3d_batch_size is not None:
        cmd += ["--det3d-batch-size", str(args.det3d_batch_size)]
    if args.stop_on_failure:
        cmd += ["--stop-on-failure"]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def build_extended_cmd(args, dataset, aiqii_classes) -> list[str]:
    cmd = [sys.executable, str(EXTENDED_SUITE),
           "--dataset", dataset,
           "--aiqii-classes", aiqii_classes,
           "--run-tag", args.run_tag,
           "--mode", args.mode,
           "--workers", str(args.workers),
           "--retries", str(args.retries),
           "--gpu", args.gpu,
           "--best-metric", args.best_metric]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.pcdet_batch_size is not None:
        cmd += ["--pcdet-batch-size", str(args.pcdet_batch_size)]
    if args.radarnext_batch_size is not None:
        cmd += ["--radarnext-batch-size", str(args.radarnext_batch_size)]
    if args.stop_on_failure:
        cmd += ["--stop-on-failure"]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'#'*72}\n# {label}\n# {' '.join(cmd)}\n{'#'*72}", flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    args = parse_args()
    dataset, aiqii_classes = CONFIGS[args.config]

    plan = []
    if args.suites in ("both", "original"):
        plan.append(("ORIGINAL 7", build_full_cmd(args, dataset, aiqii_classes)))
    if args.suites in ("both", "extended"):
        plan.append(("EXTENDED 7", build_extended_cmd(args, dataset, aiqii_classes)))

    print(f"[all14] config={args.config} dataset={dataset} classes={aiqii_classes} "
          f"run_tag={args.run_tag} mode={args.mode} suites={args.suites}", flush=True)

    failures = []
    for label, cmd in plan:
        rc = run(cmd, label)
        if rc != 0:
            failures.append((label, rc))
            print(f"[all14] {label} returned non-zero ({rc}).", flush=True)
            if args.stop_on_failure:
                break

    if failures:
        print(f"\n[all14][DONE-WITH-ISSUES] suites with failures: "
              f"{', '.join(f'{l}({rc})' for l, rc in failures)}", flush=True)
        return 1
    print("\n[all14][DONE] all requested suites finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
