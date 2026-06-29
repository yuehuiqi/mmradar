"""Evaluate one DG-PFA checkpoint at several bypass fusion strengths."""

import argparse
import json
from pathlib import Path

import numpy as np

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    args = parse_args()
    cfg_from_yaml_file(str(args.cfg_file), cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(args.output_dir / "fusion_sweep.log", rank=0)

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    model.load_params_from_file(filename=str(args.ckpt), logger=logger, to_cpu=True)
    model.cuda()

    records = []
    for scale in args.scales:
        if not 0.0 <= scale <= 1.0:
            raise ValueError(f"fusion scale must be in [0, 1], got {scale}")
        model.vfe.inference_fusion_scale = float(scale)
        result_dir = args.output_dir / f"scale_{scale:g}"
        metrics = eval_utils.eval_one_epoch(
            cfg,
            model,
            dataloader,
            f"fusion_{scale:g}",
            logger,
            dist_test=False,
            save_to_file=False,
            result_dir=result_dir,
        )
        records.append({"scale": scale, "metrics": json_value(metrics)})

    output_path = args.output_dir / "fusion_sweep_metrics.json"
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
