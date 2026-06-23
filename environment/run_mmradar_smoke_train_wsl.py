#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/mnt/e/Scholar/mmradarDetect")
ENV_ROOT = Path("/home/yuehui/miniforge3/envs")
LOG_ROOT = ROOT / "environment" / "smoke_logs"
LINUX_BASE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    project: str
    env: str
    cfg: str
    dataset: str
    extra_tag: str
    batch_size: str = "1"

    @property
    def project_root(self) -> Path:
        return ROOT / self.project

    @property
    def python(self) -> Path:
        return ENV_ROOT / self.env / "bin" / "python"

    def command(self) -> list[str]:
        if self.kind == "pcdet":
            return [
                str(self.python),
                "train.py",
                "--cfg_file",
                self.cfg,
                "--epochs",
                "1",
                "--workers",
                "0",
                "--batch_size",
                self.batch_size,
                "--extra_tag",
                self.extra_tag,
                "--max_waiting_mins",
                "0",
            ]
        if self.kind == "det3d":
            return [
                str(self.python),
                "tools/train.py",
                self.cfg,
                "--work_dir",
                f"work_dirs/{self.extra_tag}",
                "--gpus",
                "1",
            ]
        raise ValueError(self.kind)

    def cwd(self) -> Path:
        return self.project_root / "tools" if self.kind == "pcdet" else self.project_root


EXPERIMENTS = [
    Experiment("OpenPCDet_aiqii", "pcdet", "OpenPCDet", "PointPillar", "cfgs/mmradar_models/pointpillar_aiqii_smoke.yaml", "aiQiiDataset", "smoke_aiqii_v3"),
    Experiment("OpenPCDet_mmaud", "pcdet", "OpenPCDet", "PointPillar", "cfgs/mmradar_models/pointpillar_mmaud_smoke.yaml", "mmaud", "smoke_mmaud_v1"),
    Experiment("InterFusion_aiqii", "pcdet", "InterFusion", "InterFusion", "cfgs/mmradar_models/pointpillar_aiqii_smoke.yaml", "aiQiiDataset", "smoke_aiqii_v1"),
    Experiment("InterFusion_mmaud", "pcdet", "InterFusion", "InterFusion", "cfgs/mmradar_models/pointpillar_mmaud_smoke.yaml", "mmaud", "smoke_mmaud_v1"),
    Experiment("PFANet_aiqii", "pcdet", "PFA-NET", "PFANet", "cfgs/mmradar_models/pointpillar_aiqii_smoke.yaml", "aiQiiDataset", "smoke_aiqii_v1"),
    Experiment("PFANet_mmaud", "pcdet", "PFA-NET", "PFANet", "cfgs/mmradar_models/pointpillar_mmaud_smoke.yaml", "mmaud", "smoke_mmaud_v1"),
    Experiment("DSVT_aiqii", "pcdet", "DSVT", "DSVT", "cfgs/mmradar_models/dsvt_aiqii_smoke.yaml", "aiQiiDataset", "smoke_aiqii_v1", "2"),
    Experiment("DSVT_mmaud", "pcdet", "DSVT", "DSVT", "cfgs/mmradar_models/dsvt_mmaud_smoke.yaml", "mmaud", "smoke_mmaud_v1", "2"),
    Experiment("VoxelNeXt_aiqii", "pcdet", "VoxelNeXt", "VoxelNeXt", "cfgs/mmradar_models/voxelnext_aiqii_smoke.yaml", "aiQiiDataset", "smoke_aiqii_v1", "2"),
    Experiment("VoxelNeXt_mmaud", "pcdet", "VoxelNeXt", "VoxelNeXt", "cfgs/mmradar_models/voxelnext_mmaud_smoke.yaml", "mmaud", "smoke_mmaud_v1", "2"),
    Experiment("CenterPoint_aiqii", "det3d", "CenterPoint", "CenterPoint", "configs/mmradar/centerpoint_aiqii_smoke.py", "aiQiiDataset", "mmradar_centerpoint_aiqii_smoke"),
    Experiment("CenterPoint_mmaud", "det3d", "CenterPoint", "CenterPoint", "configs/mmradar/centerpoint_mmaud_smoke.py", "mmaud", "mmradar_centerpoint_mmaud_smoke"),
    Experiment("PillarNetLTS_aiqii", "det3d", "PillarNet-LTS", "PillarNetLTS", "configs/mmradar/pillarnet_aiqii_smoke.py", "aiQiiDataset", "mmradar_pillarnet_aiqii_smoke"),
    Experiment("PillarNetLTS_mmaud", "det3d", "PillarNet-LTS", "PillarNetLTS", "configs/mmradar/pillarnet_mmaud_smoke.py", "mmaud", "mmradar_pillarnet_mmaud_smoke"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Run only these experiment names.")
    parser.add_argument("--skip", nargs="*", default=[], help="Skip these experiment names.")
    return parser.parse_args()


def tail_text(path: Path, lines: int = 24) -> str:
    if not path.exists():
        return ""
    data = path.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])


def main() -> int:
    args = parse_args()
    selected = []
    only = set(args.only or [])
    skip = set(args.skip or [])
    for exp in EXPERIMENTS:
        if only and exp.name not in only:
            continue
        if exp.name in skip:
            continue
        selected.append(exp)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = LOG_ROOT / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    results = []

    for exp in selected:
        log_path = run_dir / f"{exp.name}.log"
        started = time.time()
        print(f"[RUN] {exp.name} -> {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write("$ " + " ".join(exp.command()) + "\n")
            log.write(f"# cwd={exp.cwd()}\n\n")
            log.flush()
            exp_env = env.copy()
            exp_env["PYTHONPATH"] = str(exp.project_root) + os.pathsep + exp_env.get("PYTHONPATH", "")
            exp_env["PATH"] = str(exp.python.parent) + os.pathsep + LINUX_BASE_PATH
            proc = subprocess.run(
                exp.command(),
                cwd=str(exp.cwd()),
                env=exp_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elapsed = round(time.time() - started, 2)
        item = {
            "name": exp.name,
            "project": exp.project,
            "dataset": exp.dataset,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "log": str(log_path),
        }
        results.append(item)
        status = "OK" if proc.returncode == 0 else "FAIL"
        print(f"[{status}] {exp.name} elapsed={elapsed}s", flush=True)
        if proc.returncode != 0:
            print(tail_text(log_path), flush=True)
            break

    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (LOG_ROOT / "latest_status.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] status={status_path}", flush=True)
    return 0 if all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
