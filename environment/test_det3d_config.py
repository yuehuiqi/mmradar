#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("cfg_file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    from det3d.datasets import build_dataset
    from det3d.models import build_detector
    from det3d.torchie import Config

    cfg = Config.fromfile(str(args.cfg_file))
    dataset = build_dataset(cfg.data.train)
    sample = dataset[0]
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    print(
        "ok",
        project_root.name,
        args.cfg_file.name,
        "samples",
        len(dataset),
        "sample_keys",
        sorted(sample.keys()),
        "model",
        model.__class__.__name__,
    )


if __name__ == "__main__":
    main()
