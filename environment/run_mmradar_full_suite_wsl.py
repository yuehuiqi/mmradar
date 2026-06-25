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
LINUX_BASE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


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
    aiqii_multiclass_cfg: str | None = None
    aiqii_multiclass_stem: str | None = None

    @property
    def project_root(self):
        return ROOT / self.project

    @property
    def python(self):
        return ENV_ROOT / self.env / "bin" / "python"

    def config(self, dataset, aiqii_classes="single"):
        if dataset == "aiqii" and aiqii_classes == "multiclass" and self.aiqii_multiclass_cfg:
            return self.aiqii_multiclass_cfg
        return self.aiqii_cfg if dataset == "aiqii" else self.mmaud_cfg

    def stem(self, dataset, aiqii_classes="single"):
        if dataset == "aiqii" and aiqii_classes == "multiclass":
            if self.aiqii_multiclass_stem:
                return self.aiqii_multiclass_stem
            return self.config_stem.format(dataset="aiqii_multiclass")
        return self.config_stem.format(dataset=dataset)

    def batch_size(self, dataset, batch_size=None, pcdet_batch_size=None, det3d_batch_size=None):
        override = pcdet_batch_size if self.kind == "pcdet" else det3d_batch_size
        if override is None:
            override = batch_size
        if override is not None:
            return override
        return self.aiqii_batch if dataset == "aiqii" else self.mmaud_batch

    def output_dir(self, dataset, run_tag, aiqii_classes="single"):
        dataset_part = "aiqii_multiclass" if dataset == "aiqii" and aiqii_classes == "multiclass" else dataset
        if self.kind == "pcdet":
            return (
                self.project_root
                / "output"
                / "mmradar_models"
                / self.stem(dataset, aiqii_classes)
                / run_tag
            )
        return self.project_root / "work_dirs" / f"{self.name.lower()}_{dataset_part}_{run_tag}"

    def metrics_history(self, dataset, run_tag, aiqii_classes="single"):
        return self.output_dir(dataset, run_tag, aiqii_classes) / "periodic_metrics" / "metrics_history.json"

    def latest_det3d_checkpoint(self, dataset, run_tag, aiqii_classes="single"):
        candidates = []
        for path in self.output_dir(dataset, run_tag, aiqii_classes).glob("epoch_*.pth"):
            match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        return max(candidates, default=(None, None))[1]

    def command(
        self,
        dataset,
        run_tag,
        workers,
        resume=False,
        aiqii_classes="single",
        run_dir=None,
        batch_size=None,
        pcdet_batch_size=None,
        det3d_batch_size=None,
    ):
        config = self.config(dataset, aiqii_classes)
        if self.kind == "pcdet":
            return [
                str(self.python),
                "train.py",
                "--cfg_file",
                config,
                "--workers",
                str(workers),
                "--batch_size",
                str(self.batch_size(dataset, batch_size, pcdet_batch_size, det3d_batch_size)),
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

        config = maybe_make_det3d_batch_config(
            self,
            config,
            run_dir,
            dataset,
            run_tag,
            aiqii_classes,
            self.batch_size(dataset, batch_size, pcdet_batch_size, det3d_batch_size),
            workers,
            batch_size is not None or det3d_batch_size is not None,
        )
        command = [
            str(self.python),
            "tools/train.py",
            config,
            "--work_dir",
            str(self.output_dir(dataset, run_tag, aiqii_classes)),
            "--gpus",
            "1",
        ]
        checkpoint = self.latest_det3d_checkpoint(dataset, run_tag, aiqii_classes) if resume else None
        if checkpoint is not None:
            match = re.fullmatch(r"epoch_(\d+)\.pth", checkpoint.name)
            if match and int(match.group(1)) >= TOTAL_EPOCHS:
                return [
                    str(self.python),
                    "tools/dist_test.py",
                    config,
                    "--work_dir",
                    str(self.output_dir(dataset, run_tag, aiqii_classes)),
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
        "cfgs/mmradar_models/pointpillar_aiqii_multiclass_full.yaml",
        "pointpillar_aiqii_multiclass_full",
    ),
    Experiment(
        "InterFusion", "pcdet", "InterFusion", "InterFusion",
        "pointpillar_{dataset}_full",
        "cfgs/mmradar_models/pointpillar_aiqii_full.yaml",
        "cfgs/mmradar_models/pointpillar_mmaud_full.yaml", 2, 2,
        "cfgs/mmradar_models/pointpillar_aiqii_multiclass_full.yaml",
        "pointpillar_aiqii_multiclass_full",
    ),
    Experiment(
        "PFANet", "pcdet", "PFA-NET", "PFANet",
        "pointpillar_{dataset}_full",
        "cfgs/mmradar_models/pointpillar_aiqii_full.yaml",
        "cfgs/mmradar_models/pointpillar_mmaud_full.yaml", 2, 2,
        "cfgs/mmradar_models/pointpillar_aiqii_multiclass_full.yaml",
        "pointpillar_aiqii_multiclass_full",
    ),
    Experiment(
        "DSVT", "pcdet", "DSVT", "DSVT",
        "dsvt_{dataset}_full",
        "cfgs/mmradar_models/dsvt_aiqii_full.yaml",
        "cfgs/mmradar_models/dsvt_mmaud_full.yaml", 2, 2,
        "cfgs/mmradar_models/dsvt_aiqii_multiclass_full.yaml",
        "dsvt_aiqii_multiclass_full",
    ),
    Experiment(
        "VoxelNeXt", "pcdet", "VoxelNeXt", "VoxelNeXt",
        "voxelnext_{dataset}_full",
        "cfgs/mmradar_models/voxelnext_aiqii_full.yaml",
        "cfgs/mmradar_models/voxelnext_mmaud_full.yaml", 2, 2,
        "cfgs/mmradar_models/voxelnext_aiqii_multiclass_full.yaml",
        "voxelnext_aiqii_multiclass_full",
    ),
    Experiment(
        "CenterPoint", "det3d", "CenterPoint", "CenterPoint", "",
        "configs/mmradar/centerpoint_aiqii_full.py",
        "configs/mmradar/centerpoint_mmaud_full.py", 1, 1,
        "configs/mmradar/centerpoint_aiqii_multiclass_full.py",
        "centerpoint_aiqii_multiclass_full",
    ),
    Experiment(
        "PillarNetLTS", "det3d", "PillarNet-LTS", "PillarNetLTS", "",
        "configs/mmradar/pillarnet_aiqii_full.py",
        "configs/mmradar/pillarnet_mmaud_full.py", 1, 1,
        "configs/mmradar/pillarnet_aiqii_multiclass_full.py",
        "pillarnet_aiqii_multiclass_full",
    ),
]


def maybe_make_det3d_batch_config(
    experiment,
    config,
    run_dir,
    dataset,
    run_tag,
    aiqii_classes,
    batch_size,
    workers,
    force_override,
):
    if experiment.kind == "pcdet" or not force_override:
        return config
    if run_dir is None:
        return config
    config_path = Path(config)
    module_dir = experiment.project_root / config_path.parent
    module_name = config_path.stem
    generated_dir = run_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated = generated_dir / f"{experiment.name}_{dataset}_{aiqii_classes}_{run_tag}.py"
    generated.write_text(
        "\n".join(
            [
                "import sys",
                f"sys.path.insert(0, {str(module_dir)!r})",
                f"from {module_name} import *  # noqa: F401,F403",
                "",
                f"data['samples_per_gpu'] = {int(batch_size)}",
                f"data['workers_per_gpu'] = {int(workers)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(generated)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequentially train all seven MMRadar models on one dataset."
    )
    parser.add_argument("--dataset", required=True, choices=("aiqii", "mmaud"))
    parser.add_argument(
        "--aiqii-classes",
        default="single",
        choices=("single", "multiclass"),
        help="aiQii class mode: single folds all UAVs into Drone; multiclass keeps Air3S/Mini4pro/Mavic3Pro/jingling4.",
    )
    parser.add_argument("--run-tag", default="default")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None, help="Override every model's train batch size.")
    parser.add_argument("--pcdet-batch-size", type=int, default=None, help="Override OpenPCDet-style model batch size.")
    parser.add_argument("--det3d-batch-size", type=int, default=None, help="Override CenterPoint/PillarNet-LTS batch size.")
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


def final_metrics(experiment, dataset, run_tag, aiqii_classes="single"):
    history = read_json(experiment.metrics_history(dataset, run_tag, aiqii_classes), [])
    if not history:
        return None
    final = max(history, key=lambda item: int(item.get("epoch", -1)))
    if int(final.get("epoch", -1)) < TOTAL_EPOCHS:
        return None
    return final


def prune_checkpoints(experiment, dataset, run_tag, aiqii_classes="single"):
    output_dir = experiment.output_dir(dataset, run_tag, aiqii_classes)
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


def write_summary(run_dir, dataset, run_tag, statuses, aiqii_classes="single"):
    rows = []
    full_results = {}
    for experiment in EXPERIMENTS:
        status = statuses.get(experiment.name, {})
        history = read_json(experiment.metrics_history(dataset, run_tag, aiqii_classes), [])
        final = final_metrics(experiment, dataset, run_tag, aiqii_classes)
        metrics = final.get("metrics", {}) if final else {}
        full_results[experiment.name] = {
            "status": status,
            "final": final,
            "history": history,
            "history_file": str(experiment.metrics_history(dataset, run_tag, aiqii_classes)),
            "output_dir": str(experiment.output_dir(dataset, run_tag, aiqii_classes)),
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
        f"# MMRadar full suite metrics: {dataset} / {aiqii_classes}",
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
    if args.dataset != "aiqii" and args.aiqii_classes != "single":
        raise ValueError("--aiqii-classes multiclass is only valid with --dataset aiqii")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_tag):
        raise ValueError("--run-tag may contain only letters, numbers, dot, underscore and dash")
    selected_names = set(args.only or [experiment.name for experiment in EXPERIMENTS])
    unknown = selected_names - {experiment.name for experiment in EXPERIMENTS}
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    if args.dry_run:
        dry_run_dataset = args.dataset if args.aiqii_classes == "single" else "aiqii_multiclass"
        dry_run_dir = RUN_ROOT / dry_run_dataset / args.run_tag
        for experiment in EXPERIMENTS:
            if experiment.name not in selected_names:
                continue
            command = experiment.command(
                args.dataset,
                args.run_tag,
                args.workers,
                aiqii_classes=args.aiqii_classes,
                run_dir=dry_run_dir,
                batch_size=args.batch_size,
                pcdet_batch_size=args.pcdet_batch_size,
                det3d_batch_size=args.det3d_batch_size,
            )
            print(f"[{experiment.name}] cwd={experiment.cwd()}")
            print("  " + " ".join(command))
            print(f"  output={experiment.output_dir(args.dataset, args.run_tag, args.aiqii_classes)}")
        return 0

    dataset_run_name = args.dataset if args.aiqii_classes == "single" else "aiqii_multiclass"
    run_dir = RUN_ROOT / dataset_run_name / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "suite_status.json"
    statuses = read_json(status_path, {})
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")

    for experiment in EXPERIMENTS:
        if experiment.name not in selected_names:
            continue
        existing_final = final_metrics(experiment, args.dataset, args.run_tag, args.aiqii_classes)
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
                args.dataset,
                args.run_tag,
                args.workers,
                resume=attempt > 1,
                aiqii_classes=args.aiqii_classes,
                run_dir=run_dir,
                batch_size=args.batch_size,
                pcdet_batch_size=args.pcdet_batch_size,
                det3d_batch_size=args.det3d_batch_size,
            )
            log_path = run_dir / f"{experiment.name}_attempt_{attempt}.log"
            statuses[experiment.name] = {
                "state": "running",
                "attempt": attempt,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "command": command,
                "log": str(log_path),
                "output_dir": str(experiment.output_dir(args.dataset, args.run_tag, args.aiqii_classes)),
            }
            write_json(status_path, statuses)
            print(f"[RUN] {experiment.name} attempt {attempt} -> {log_path}", flush=True)
            started = time.time()
            experiment_environment = environment.copy()
            experiment_environment["PATH"] = str(experiment.python.parent) + os.pathsep + LINUX_BASE_PATH
            try:
                with log_path.open("w", encoding="utf-8", errors="replace") as log:
                    log.write("$ " + " ".join(command) + "\n\n")
                    log.flush()
                    process = subprocess.run(
                        command,
                        cwd=str(experiment.cwd()),
                        env=experiment_environment,
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
                write_summary(run_dir, args.dataset, args.run_tag, statuses, args.aiqii_classes)
                print("Interrupted. Re-run the same command to resume.", flush=True)
                return 130

            final = final_metrics(experiment, args.dataset, args.run_tag, args.aiqii_classes)
            if return_code == 0 and final is not None:
                prune_checkpoints(experiment, args.dataset, args.run_tag, args.aiqii_classes)
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
        write_summary(run_dir, args.dataset, args.run_tag, statuses, args.aiqii_classes)
        if not completed and args.stop_on_failure:
            return 1

    write_summary(run_dir, args.dataset, args.run_tag, statuses, args.aiqii_classes)
    failed = [
        name for name, status in statuses.items()
        if name in selected_names and status.get("state") == "failed"
    ]
    print(f"[DONE] summary={run_dir / 'ALL_MODELS_METRICS.md'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
