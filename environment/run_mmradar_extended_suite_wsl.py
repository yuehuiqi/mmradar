#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/mnt/e/Scholar/mmradarDetect")
ENV_ROOT = Path("/home/yuehui/miniforge3/envs")
RUN_ROOT = ROOT / "environment" / "extended_runs"
LINUX_BASE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
TOTAL_EPOCHS = {"full": 80, "smoke": 1}


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    project: str
    env: str
    stem: str
    aiqii_batch: int = 2
    mmaud_batch: int = 2

    @property
    def project_root(self) -> Path:
        return ROOT / self.project

    @property
    def python(self) -> Path:
        return ENV_ROOT / self.env / "bin" / "python"

    def batch_size(
        self,
        dataset: str,
        batch_size: int | None = None,
        pcdet_batch_size: int | None = None,
        radarnext_batch_size: int | None = None,
    ) -> int:
        override = radarnext_batch_size if self.kind == "radarnext" else pcdet_batch_size
        if override is None:
            override = batch_size
        if override is not None:
            return override
        return self.aiqii_batch if dataset == "aiqii" else self.mmaud_batch

    def dataset_key(self, dataset: str, aiqii_classes: str) -> str:
        return "aiqii_multiclass" if dataset == "aiqii" and aiqii_classes == "multiclass" else dataset

    def cfg(self, dataset: str, mode: str, aiqii_classes: str = "single") -> str:
        cfg_dataset = self.dataset_key(dataset, aiqii_classes)
        if self.kind == "radarnext":
            return f"configs/mmradar/radarnext_centerpoint_{cfg_dataset}_{mode}.py"
        if self.kind == "radarpillar":
            return f"tools/cfgs/mmradar_models/radarpillar_{cfg_dataset}_{mode}.yaml"
        return f"cfgs/mmradar_models/{self.stem}_{cfg_dataset}_{mode}.yaml"

    def output_dir(self, dataset: str, run_tag: str, mode: str, aiqii_classes: str = "single") -> Path:
        cfg_dataset = self.dataset_key(dataset, aiqii_classes)
        stem = f"{self.stem}_{cfg_dataset}_{mode}"
        if self.kind == "radarpillar":
            return self.project_root / "output" / "cfgs" / "mmradar_models" / stem / run_tag
        if self.kind == "radarnext":
            return self.project_root / "work_dirs" / f"{self.name.lower()}_{cfg_dataset}_{mode}_{run_tag}"
        return self.project_root / "output" / "mmradar_models" / stem / run_tag

    def metrics_history_path(self, dataset: str, run_tag: str, mode: str, aiqii_classes: str = "single") -> Path:
        return self.output_dir(dataset, run_tag, mode, aiqii_classes) / "periodic_metrics" / "metrics_history.json"

    def cwd(self) -> Path:
        if self.kind == "pcdet":
            return self.project_root / "tools"
        return self.project_root

    def command(
        self,
        dataset: str,
        run_tag: str,
        mode: str,
        workers: int,
        resume: bool,
        aiqii_classes: str = "single",
        batch_size_override: int | None = None,
        pcdet_batch_size: int | None = None,
        radarnext_batch_size: int | None = None,
    ) -> list[str]:
        cfg = self.cfg(dataset, mode, aiqii_classes)
        train_batch_size = self.batch_size(
            dataset,
            batch_size=batch_size_override,
            pcdet_batch_size=pcdet_batch_size,
            radarnext_batch_size=radarnext_batch_size,
        )
        batch_size = str(train_batch_size)
        if self.kind == "radarnext":
            cmd = [
                str(self.python),
                "tools/train.py",
                cfg,
                "--work-dir",
                str(self.output_dir(dataset, run_tag, mode, aiqii_classes)),
            ]
            if resume and (self.output_dir(dataset, run_tag, mode, aiqii_classes) / "last_checkpoint").exists():
                cmd.extend(["--resume", "auto"])
            cfg_options = [
                f"train_dataloader.batch_size={train_batch_size}",
                f"train_dataloader.num_workers={workers}",
                f"val_dataloader.num_workers={workers}",
            ]
            if mode == "smoke":
                cfg_options.extend([
                    "train_cfg.max_epochs=1",
                    "train_cfg.val_interval=1",
                    "default_hooks.checkpoint.max_keep_ckpts=1",
                ])
            if cfg_options:
                cmd.extend([
                    "--cfg-options",
                    *cfg_options,
                ])
            return cmd

        if self.kind == "radarpillar":
            cmd = [
                str(self.python),
                "tools/train.py",
                "--cfg_file",
                cfg,
                "--workers",
                str(workers),
                "--batch_size",
                batch_size,
                "--extra_tag",
                run_tag,
                "--ckpt_save_interval",
                "1" if mode == "smoke" else "10",
                "--max_ckpt_save_num",
                "1" if mode == "smoke" else "9",
                "--max_waiting_mins",
                "0",
            ]
            if mode == "smoke":
                cmd.extend(["--epochs", "1"])
            return cmd

        cmd = [
            str(self.python),
            "train.py",
            "--cfg_file",
            cfg,
            "--workers",
            str(workers),
            "--batch_size",
            batch_size,
            "--extra_tag",
            run_tag,
            "--ckpt_save_interval",
            "1" if mode == "smoke" else "10",
            "--max_ckpt_save_num",
            "1" if mode == "smoke" else "9",
            "--eval_interval",
            "1" if mode == "smoke" else "5",
            "--max_waiting_mins",
            "0",
        ]
        if mode == "smoke":
            cmd.extend(["--epochs", "1"])
        return cmd


EXPERIMENTS = [
    Experiment("PointRCNN", "pcdet", "OpenPCDet", "PointPillar", "pointrcnn"),
    Experiment("PartA2", "pcdet", "OpenPCDet", "PointPillar", "parta2"),
    Experiment("PVRCNN", "pcdet", "OpenPCDet", "PointPillar", "pvrcnn"),
    Experiment("VoxelRCNN", "pcdet", "OpenPCDet", "PointPillar", "voxelrcnn"),
    Experiment("PVRCNNPlusPlus", "pcdet", "OpenPCDet", "PointPillar", "pvrcnnplusplus"),
    Experiment("RadarPillar", "radarpillar", "RadarPillar", "RadarPillar", "radarpillar"),
    Experiment("RadarNeXt", "radarnext", "RadarNeXt", "RadarNeXt", "radarnext_centerpoint", 1, 1),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the extended MMRadar model suite.")
    parser.add_argument("--dataset", required=True, choices=("aiqii", "mmaud"))
    parser.add_argument(
        "--aiqii-classes",
        default="single",
        choices=("single", "multiclass"),
        help="aiQii class mode: single folds all UAVs into Drone; multiclass keeps Air3S/Mini4pro/Mavic3Pro/jingling4.",
    )
    parser.add_argument("--run-tag", default="default")
    parser.add_argument("--mode", default="full", choices=("full", "smoke"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None, help="Override every extended model's train batch size.")
    parser.add_argument("--pcdet-batch-size", type=int, default=None, help="Override OpenPCDet/RadarPillar-style batch size.")
    parser.add_argument("--radarnext-batch-size", type=int, default=None, help="Override RadarNeXt train dataloader batch size.")
    parser.add_argument("--retries", type=int, default=2, help="Retries after the first attempt.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def normalize_metric_keys(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        key = str(key)
        if key.startswith("mmradar/"):
            key = key[len("mmradar/"):]
        out[key] = value
    return out


def read_pcdet_history(exp: Experiment, dataset: str, run_tag: str, mode: str, aiqii_classes: str) -> list[dict]:
    history = read_json(exp.metrics_history_path(dataset, run_tag, mode, aiqii_classes), [])
    if not isinstance(history, list):
        return []
    normalized = []
    for item in history:
        if not isinstance(item, dict):
            continue
        metrics = normalize_metric_keys(item.get("metrics", {}))
        normalized.append({**item, "metrics": metrics})
    return normalized


def read_radarnext_history(exp: Experiment, dataset: str, run_tag: str, mode: str, aiqii_classes: str) -> list[dict]:
    output_dir = exp.output_dir(dataset, run_tag, mode, aiqii_classes)
    scalar_files = sorted(output_dir.glob("*/vis_data/scalars.json"), key=lambda p: p.stat().st_mtime)
    history = []
    for scalar_file in scalar_files:
        for line in scalar_file.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = normalize_metric_keys(item)
            if not metrics:
                continue
            if not any(key.startswith("center_ap_") or key == "frames" for key in metrics):
                continue
            epoch = int(item.get("epoch", item.get("step", len(history) + 1)))
            history.append({"epoch": epoch, "metrics": metrics, "source": str(scalar_file)})
    return history


def metrics_history(exp: Experiment, dataset: str, run_tag: str, mode: str, aiqii_classes: str) -> list[dict]:
    if exp.kind == "radarnext":
        return read_radarnext_history(exp, dataset, run_tag, mode, aiqii_classes)
    return read_pcdet_history(exp, dataset, run_tag, mode, aiqii_classes)


def final_metrics(exp: Experiment, dataset: str, run_tag: str, mode: str, aiqii_classes: str) -> dict | None:
    history = metrics_history(exp, dataset, run_tag, mode, aiqii_classes)
    if not history:
        return None
    return max(history, key=lambda item: int(item.get("epoch", -1)))


def is_complete(exp: Experiment, dataset: str, run_tag: str, mode: str, aiqii_classes: str) -> bool:
    final = final_metrics(exp, dataset, run_tag, mode, aiqii_classes)
    return bool(final and int(final.get("epoch", -1)) >= TOTAL_EPOCHS[mode])


def prune_checkpoints(exp: Experiment, dataset: str, run_tag: str, mode: str, aiqii_classes: str) -> None:
    output_dir = exp.output_dir(dataset, run_tag, mode, aiqii_classes)
    if exp.kind in ("pcdet", "radarpillar"):
        checkpoint_dir = output_dir / "ckpt"
        pattern = "checkpoint_epoch_*.pth"
        regex = r"checkpoint_epoch_(\d+)\.pth"
        limit = 1 if mode == "smoke" else 9
    else:
        checkpoint_dir = output_dir
        pattern = "epoch_*.pth"
        regex = r"epoch_(\d+)\.pth"
        limit = 1 if mode == "smoke" else 10

    checkpoints = []
    for path in checkpoint_dir.glob(pattern):
        match = re.fullmatch(regex, path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    for _, path in checkpoints[:-limit]:
        path.unlink()


def metric(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if not isinstance(value, (int, float)):
        return "-"
    if "distance" in key:
        return f"{value:.3f}"
    return f"{100 * value:.2f}%"


def write_summary(run_dir: Path, dataset: str, run_tag: str, mode: str, statuses: dict, aiqii_classes: str) -> None:
    rows = []
    full_results = {}
    for exp in EXPERIMENTS:
        status = statuses.get(exp.name, {"state": "pending"})
        history = metrics_history(exp, dataset, run_tag, mode, aiqii_classes)
        final = final_metrics(exp, dataset, run_tag, mode, aiqii_classes)
        metrics = final.get("metrics", {}) if final else {}
        full_results[exp.name] = {
            "status": status,
            "final": final,
            "history": history,
            "output_dir": str(exp.output_dir(dataset, run_tag, mode, aiqii_classes)),
        }
        rows.append(
            "| {name} | {state} | {epoch} | {center2} | {center4} | {bev25} | {bev50} | {iou25} | {iou50} | {dist} |".format(
                name=exp.name,
                state=status.get("state", "pending"),
                epoch=final.get("epoch", "-") if final else "-",
                center2=metric(metrics, "center_ap_2m"),
                center4=metric(metrics, "center_ap_4m"),
                bev25=metric(metrics, "bev_iou_ap_0.25"),
                bev50=metric(metrics, "bev_iou_ap_0.5"),
                iou25=metric(metrics, "3d_iou_ap_0.25"),
                iou50=metric(metrics, "3d_iou_ap_0.5"),
                dist=metric(metrics, "mean_best_center_distance"),
            )
        )

    write_json(run_dir / "EXTENDED_MODELS_METRICS.json", full_results)
    md = [
        f"# MMRadar extended suite metrics: {dataset} / {aiqii_classes} / {run_tag} / {mode}",
        "",
        "| Model | State | Final epoch | Center AP@2m | Center AP@4m | BEV AP@0.25 | BEV AP@0.5 | 3D AP@0.25 | 3D AP@0.5 | Mean center dist |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "完整历史指标见 `EXTENDED_MODELS_METRICS.json`；各模型自己的周期指标仍保存在对应 output/work_dirs 目录。",
    ]
    (run_dir / "EXTENDED_MODELS_METRICS.md").write_text("\n".join(md), encoding="utf-8")


def env_for(exp: Experiment, gpu: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PATH"] = str(exp.python.parent) + os.pathsep + LINUX_BASE_PATH
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(exp.project_root) + os.pathsep + env.get("PYTHONPATH", "")
    env_root = ENV_ROOT / exp.env
    ld_parts = [
        env_root / "lib" / "python3.10" / "site-packages" / "torch" / "lib",
        env_root / "targets" / "x86_64-linux" / "lib",
        env_root / "lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in ld_parts) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def selected_experiments(args: argparse.Namespace) -> list[Experiment]:
    only = set(args.only or [])
    skip = set(args.skip or [])
    selected = []
    for exp in EXPERIMENTS:
        if only and exp.name not in only:
            continue
        if exp.name in skip:
            continue
        selected.append(exp)
    return selected


def main() -> int:
    args = parse_args()
    if args.dataset != "aiqii" and args.aiqii_classes != "single":
        raise ValueError("--aiqii-classes multiclass is only valid with --dataset aiqii")
    dataset_run_name = args.dataset if args.aiqii_classes == "single" else "aiqii_multiclass"
    run_dir = RUN_ROOT / dataset_run_name / args.mode / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    statuses = read_json(run_dir / "suite_status.json", {})
    if not isinstance(statuses, dict):
        statuses = {}

    selected = selected_experiments(args)
    if args.dry_run:
        for exp in selected:
            print(f"[DRY] {exp.name}: cwd={exp.cwd()}")
            print("      " + " ".join(exp.command(
                args.dataset,
                args.run_tag,
                args.mode,
                args.workers,
                resume=False,
                aiqii_classes=args.aiqii_classes,
                batch_size_override=args.batch_size,
                pcdet_batch_size=args.pcdet_batch_size,
                radarnext_batch_size=args.radarnext_batch_size,
            )))
        return 0

    for exp in selected:
        if is_complete(exp, args.dataset, args.run_tag, args.mode, args.aiqii_classes):
            statuses[exp.name] = {"state": "skipped_complete", "time": time.strftime("%F %T")}
            write_json(run_dir / "suite_status.json", statuses)
            continue

        success = False
        for attempt in range(1, args.retries + 2):
            log_path = run_dir / f"{exp.name}_attempt_{attempt}.log"
            cmd = exp.command(
                args.dataset,
                args.run_tag,
                args.mode,
                args.workers,
                resume=(attempt > 1),
                aiqii_classes=args.aiqii_classes,
                batch_size_override=args.batch_size,
                pcdet_batch_size=args.pcdet_batch_size,
                radarnext_batch_size=args.radarnext_batch_size,
            )
            statuses[exp.name] = {
                "state": "running",
                "attempt": attempt,
                "command": cmd,
                "log": str(log_path),
                "time": time.strftime("%F %T"),
            }
            write_json(run_dir / "suite_status.json", statuses)
            print(f"[RUN] {exp.name} attempt={attempt} log={log_path}", flush=True)
            started = time.time()
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                log.write("$ " + " ".join(cmd) + "\n")
                log.write(f"# cwd={exp.cwd()}\n\n")
                log.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(exp.cwd()),
                    env=env_for(exp, args.gpu),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            prune_checkpoints(exp, args.dataset, args.run_tag, args.mode, args.aiqii_classes)
            elapsed = round(time.time() - started, 2)
            if proc.returncode == 0:
                success = True
                statuses[exp.name] = {
                    "state": "ok",
                    "attempt": attempt,
                    "elapsed_sec": elapsed,
                    "log": str(log_path),
                    "time": time.strftime("%F %T"),
                }
                print(f"[OK] {exp.name} elapsed={elapsed}s", flush=True)
                break

            statuses[exp.name] = {
                "state": "failed_attempt",
                "attempt": attempt,
                "returncode": proc.returncode,
                "elapsed_sec": elapsed,
                "log": str(log_path),
                "time": time.strftime("%F %T"),
            }
            print(f"[FAIL] {exp.name} attempt={attempt} code={proc.returncode}", flush=True)
            write_json(run_dir / "suite_status.json", statuses)

        if not success:
            statuses[exp.name]["state"] = "failed"
            if args.stop_on_failure:
                write_summary(run_dir, args.dataset, args.run_tag, args.mode, statuses, args.aiqii_classes)
                write_json(run_dir / "suite_status.json", statuses)
                return 1

        write_json(run_dir / "suite_status.json", statuses)
        write_summary(run_dir, args.dataset, args.run_tag, args.mode, statuses, args.aiqii_classes)

    write_json(run_dir / "suite_status.json", statuses)
    write_summary(run_dir, args.dataset, args.run_tag, args.mode, statuses, args.aiqii_classes)
    failed = any(status.get("state") == "failed" for status in statuses.values())
    print(f"[DONE] summary={run_dir / 'EXTENDED_MODELS_METRICS.md'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
