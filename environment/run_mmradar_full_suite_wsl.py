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
RUN_ROOT = ROOT / "environment" / "full_runs"
TOTAL_EPOCHS = 80


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    project: str
    env: str
    config_stem: str
    aiqii_cfg: str
    mmaud_cfg: str
    aiqii_batch: int
    mmaud_batch: int

    @property
    def project_root(self):
        return ROOT / self.project

    @property
    def python(self):
        return ENV_ROOT / self.env / "bin" / "python"

    def config(self, dataset):
        return self.aiqii_cfg if dataset == "aiqii" else self.mmaud_cfg

    def batch_size(self, dataset):
        return self.aiqii_batch if dataset == "aiqii" else self.mmaud_batch

    def output_dir(self, dataset, run_tag):
        if self.kind == "pcdet":
            return (
                self.project_root
                / "output"
                / "mmradar_models"
                / self.config_stem.format(dataset=dataset)
                / run_tag
            )
        return self.project_root / "work_dirs" / f"{self.name.lower()}_{dataset}_{run_tag}"

    def metrics_history(self, dataset, run_tag):
        return self.output_dir(dataset, run_tag) / "periodic_metrics" / "metrics_history.json"

    def latest_det3d_checkpoint(self, dataset, run_tag):
        candidates = []
        for path in self.output_dir(dataset, run_tag).glob("epoch_*.pth"):
            match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        return max(candidates, default=(None, None))[1]

    def command(self, dataset, run_tag, workers, resume=False):
        config = self.config(dataset)
        if self.kind == "pcdet":
            return [
                str(self.python),
                "train.py",
                "--cfg_file",
                config,
                "--workers",
                str(workers),
                "--batch_size",
                str(self.batch_size(dataset)),
                "--extra_tag",
                run_tag,
                "--ckpt_save_interval",
                "10",
                "--max_ckpt_save_num",
                "9",
                "--eval_interval",
                "5",
                "--max_waiting_mins",
                "0",
            ]

        command = [
            str(self.python),
            "tools/train.py",
            config,
            "--work_dir",
            str(self.output_dir(dataset, run_tag)),
            "--gpus",
            "1",
        ]
        checkpoint = self.latest_det3d_checkpoint(dataset, run_tag) if resume else None
        if checkpoint is not None:
            match = re.fullmatch(r"epoch_(\d+)\.pth", checkpoint.name)
            if match and int(match.group(1)) >= TOTAL_EPOCHS:
                return [
                    str(self.python),
                    "tools/dist_test.py",
                    config,
                    "--work_dir",
                    str(self.output_dir(dataset, run_tag)),
                    "--checkpoint",
                    str(checkpoint),
                    "--gpus",
                    "1",
                ]
            command.extend(["--resume_from", str(checkpoint)])
        return command

    def cwd(self):
        return self.project_root / "tools" if self.kind == "pcdet" else self.project_root


EXPERIMENTS = [
    Experiment(
        "OpenPCDet", "pcdet", "OpenPCDet", "PointPillar",
        "pointpillar_{dataset}_full",
        "cfgs/mmradar_models/pointpillar_aiqii_full.yaml",
        "cfgs/mmradar_models/pointpillar_mmaud_full.yaml", 2, 2,
    ),
    Experiment(
        "InterFusion", "pcdet", "InterFusion", "InterFusion",
        "pointpillar_{dataset}_full",
        "cfgs/mmradar_models/pointpillar_aiqii_full.yaml",
        "cfgs/mmradar_models/pointpillar_mmaud_full.yaml", 2, 2,
    ),
    Experiment(
        "PFANet", "pcdet", "PFA-NET", "PFANet",
        "pointpillar_{dataset}_full",
        "cfgs/mmradar_models/pointpillar_aiqii_full.yaml",
        "cfgs/mmradar_models/pointpillar_mmaud_full.yaml", 2, 2,
    ),
    Experiment(
        "DSVT", "pcdet", "DSVT", "DSVT",
        "dsvt_{dataset}_full",
        "cfgs/mmradar_models/dsvt_aiqii_full.yaml",
        "cfgs/mmradar_models/dsvt_mmaud_full.yaml", 1, 2,
    ),
    Experiment(
        "VoxelNeXt", "pcdet", "VoxelNeXt", "VoxelNeXt",
        "voxelnext_{dataset}_full",
        "cfgs/mmradar_models/voxelnext_aiqii_full.yaml",
        "cfgs/mmradar_models/voxelnext_mmaud_full.yaml", 1, 2,
    ),
    Experiment(
        "CenterPoint", "det3d", "CenterPoint", "CenterPoint", "",
        "configs/mmradar/centerpoint_aiqii_full.py",
        "configs/mmradar/centerpoint_mmaud_full.py", 1, 1,
    ),
    Experiment(
        "PillarNetLTS", "det3d", "PillarNet-LTS", "PillarNetLTS", "",
        "configs/mmradar/pillarnet_aiqii_full.py",
        "configs/mmradar/pillarnet_mmaud_full.py", 1, 1,
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequentially train all seven MMRadar models on one dataset."
    )
    parser.add_argument("--dataset", required=True, choices=("aiqii", "mmaud"))
    parser.add_argument("--run-tag", default="default")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2, help="Retries after the first attempt.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting training.")
    return parser.parse_args()


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def final_metrics(experiment, dataset, run_tag):
    history = read_json(experiment.metrics_history(dataset, run_tag), [])
    if not history:
        return None
    final = max(history, key=lambda item: int(item.get("epoch", -1)))
    if int(final.get("epoch", -1)) < TOTAL_EPOCHS:
        return None
    return final


def prune_checkpoints(experiment, dataset, run_tag):
    output_dir = experiment.output_dir(dataset, run_tag)
    if experiment.kind == "pcdet":
        checkpoint_dir = output_dir / "ckpt"
        latest_model = checkpoint_dir / "latest_model.pth"
        if latest_model.exists():
            latest_model.unlink()
        limit = 9
        pattern = "checkpoint_epoch_*.pth"
        regex = r"checkpoint_epoch_(\d+)\.pth"
    else:
        checkpoint_dir = output_dir
        limit = 10
        pattern = "epoch_*.pth"
        regex = r"epoch_(\d+)\.pth"
    checkpoints = []
    for path in checkpoint_dir.glob(pattern):
        match = re.fullmatch(regex, path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    for _, path in checkpoints[:-limit]:
        path.unlink()


def metric(metrics, key):
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return f"{100 * value:.2f}%" if "distance" not in key else f"{value:.3f}"
    return "-"


def write_summary(run_dir, dataset, run_tag, statuses):
    rows = []
    full_results = {}
    for experiment in EXPERIMENTS:
        status = statuses.get(experiment.name, {})
        history = read_json(experiment.metrics_history(dataset, run_tag), [])
        final = final_metrics(experiment, dataset, run_tag)
        metrics = final.get("metrics", {}) if final else {}
        full_results[experiment.name] = {
            "status": status,
            "final": final,
            "history": history,
            "history_file": str(experiment.metrics_history(dataset, run_tag)),
            "output_dir": str(experiment.output_dir(dataset, run_tag)),
        }
        rows.append(
            "| {name} | {status} | {epoch} | {center2} | {center4} | {bev50} | {iou25} | {iou50} | {distance} |".format(
                name=experiment.name,
                status=status.get("state", "pending"),
                epoch=final.get("epoch", "-") if final else "-",
                center2=metric(metrics, "center_ap_2m"),
                center4=metric(metrics, "center_ap_4m"),
                bev50=metric(metrics, "bev_iou_ap_0.5"),
                iou25=metric(metrics, "3d_iou_ap_0.25"),
                iou50=metric(metrics, "3d_iou_ap_0.5"),
                distance=(
                    f"{metrics.get('mean_best_center_distance', 0):.3f}"
                    if isinstance(metrics.get("mean_best_center_distance"), (int, float))
                    else "-"
                ),
            )
        )
    write_json(run_dir / "ALL_MODELS_METRICS.json", full_results)
    markdown = [
        f"# MMRadar 七模型训练汇总：{dataset}",
        "",
        f"- run tag：`{run_tag}`",
        f"- 每 5 epoch 验证，完整指标见 JSON 和各模型 `periodic_metrics/metrics_history.json`。",
        "- AP 基于 score threshold 与 NMS 后的预测；比较不同论文前须统一评价协议。",
        "",
        "| 模型 | 状态 | 最终 epoch | Center AP@2m | Center AP@4m | BEV AP@0.5 | 3D AP@0.25 | 3D AP@0.5 | 平均中心误差(m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
    ]
    (run_dir / "ALL_MODELS_METRICS.md").write_text("\n".join(markdown), encoding="utf-8")


def main():
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_tag):
        raise ValueError("--run-tag may contain only letters, numbers, dot, underscore and dash")
    selected_names = set(args.only or [experiment.name for experiment in EXPERIMENTS])
    unknown = selected_names - {experiment.name for experiment in EXPERIMENTS}
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    if args.dry_run:
        for experiment in EXPERIMENTS:
            if experiment.name not in selected_names:
                continue
            command = experiment.command(args.dataset, args.run_tag, args.workers)
            print(f"[{experiment.name}] cwd={experiment.cwd()}")
            print("  " + " ".join(command))
            print(f"  output={experiment.output_dir(args.dataset, args.run_tag)}")
        return 0

    run_dir = RUN_ROOT / args.dataset / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "suite_status.json"
    statuses = read_json(status_path, {})
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")

    for experiment in EXPERIMENTS:
        if experiment.name not in selected_names:
            continue
        existing_final = final_metrics(experiment, args.dataset, args.run_tag)
        if existing_final is not None:
            statuses[experiment.name] = {
                **statuses.get(experiment.name, {}),
                "state": "completed",
                "epoch": existing_final["epoch"],
            }
            write_json(status_path, statuses)
            print(f"[SKIP] {experiment.name}: epoch {existing_final['epoch']} already complete", flush=True)
            continue

        completed = False
        for attempt in range(1, args.retries + 2):
            command = experiment.command(
                args.dataset, args.run_tag, args.workers, resume=attempt > 1
            )
            log_path = run_dir / f"{experiment.name}_attempt_{attempt}.log"
            statuses[experiment.name] = {
                "state": "running",
                "attempt": attempt,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "command": command,
                "log": str(log_path),
                "output_dir": str(experiment.output_dir(args.dataset, args.run_tag)),
            }
            write_json(status_path, statuses)
            print(f"[RUN] {experiment.name} attempt {attempt} -> {log_path}", flush=True)
            started = time.time()
            try:
                with log_path.open("w", encoding="utf-8", errors="replace") as log:
                    log.write("$ " + " ".join(command) + "\n\n")
                    log.flush()
                    process = subprocess.run(
                        command,
                        cwd=str(experiment.cwd()),
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                return_code = process.returncode
            except KeyboardInterrupt:
                statuses[experiment.name].update(
                    state="interrupted", elapsed_sec=round(time.time() - started, 2)
                )
                write_json(status_path, statuses)
                write_summary(run_dir, args.dataset, args.run_tag, statuses)
                print("Interrupted. Re-run the same command to resume.", flush=True)
                return 130

            final = final_metrics(experiment, args.dataset, args.run_tag)
            if return_code == 0 and final is not None:
                prune_checkpoints(experiment, args.dataset, args.run_tag)
                statuses[experiment.name].update(
                    state="completed",
                    epoch=final["epoch"],
                    elapsed_sec=round(time.time() - started, 2),
                    returncode=return_code,
                )
                completed = True
                print(f"[OK] {experiment.name}", flush=True)
                break

            statuses[experiment.name].update(
                state="retrying" if attempt <= args.retries else "failed",
                elapsed_sec=round(time.time() - started, 2),
                returncode=return_code,
                error="training failed or final epoch metrics are missing",
            )
            write_json(status_path, statuses)
            print(f"[FAIL] {experiment.name} returncode={return_code}", flush=True)

        write_json(status_path, statuses)
        write_summary(run_dir, args.dataset, args.run_tag, statuses)
        if not completed and args.stop_on_failure:
            return 1

    write_summary(run_dir, args.dataset, args.run_tag, statuses)
    failed = [
        name for name, status in statuses.items()
        if name in selected_names and status.get("state") == "failed"
    ]
    print(f"[DONE] summary={run_dir / 'ALL_MODELS_METRICS.md'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
