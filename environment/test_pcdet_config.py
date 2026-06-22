#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
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
    tools_dir = project_root / "tools"
    sys.path.insert(0, str(tools_dir))
    sys.path.insert(0, str(project_root))
    try:
        import _init_path  # noqa: F401
    except ModuleNotFoundError:
        pass
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network

    cfg.clear()
    cfg_from_yaml_file(str(args.cfg_file), cfg)
    logger = logging.getLogger("test_pcdet_config")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=True,
    )
    batch = next(iter(loader))
    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    print(
        "ok",
        project_root.name,
        args.cfg_file.name,
        "samples",
        len(dataset),
        "batch_keys",
        sorted(batch.keys()),
        "model",
        model.__class__.__name__,
    )


if __name__ == "__main__":
    main()
