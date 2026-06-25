#!/usr/bin/env python3
"""Build normalized metadata for all mmradarDetect projects.

The default ``mmradar_infos_*.pkl`` / ``mmradar_det3d_infos_*.pkl`` files are
single-class metadata where every UAV is normalized to ``Drone``.  For aiQii we
also emit ``*_multiclass*.pkl`` files that keep the original four UAV labels
while using the same sparse-frame filtering rules as the single-class training
set.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


DATASETS = {
    "aiqii": Path("/mnt/e/Scholar/dataset/aiQiiDataset/radar_openpcdet"),
    "mmaud": Path("/mnt/e/Scholar/dataset/mmaud/mmaud_radar_camera_openpcdet"),
}


class NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy 2.x pickles in the NumPy 1.x training environments."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return NumpyCompatUnpickler(handle).load()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aiqii-root", type=Path, default=DATASETS["aiqii"])
    parser.add_argument("--mmaud-root", type=Path, default=DATASETS["mmaud"])
    parser.add_argument("--smoke-train", type=int, default=64)
    parser.add_argument("--smoke-val", type=int, default=32)
    parser.add_argument(
        "--aiqii-min-train-points",
        type=int,
        default=16,
        help="Drop aiQii train frames with fewer in-range radar points.",
    )
    parser.add_argument(
        "--aiqii-min-train-bev-cells",
        type=int,
        default=16,
        help="Drop aiQii train frames with fewer occupied 0.5m BEV cells.",
    )
    parser.add_argument(
        "--aiqii-min-train-voxels",
        type=int,
        default=16,
        help="Drop aiQii train frames with fewer occupied 0.5m 3D voxels.",
    )
    return parser.parse_args()


def read_ids(root: Path, split: str) -> list[str]:
    path = root / "ImageSets" / f"{split}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 8:
            raise ValueError(f"Expected 8 label fields in {path}, got {len(fields)}")
        boxes.append([float(value) for value in fields[:7]])
    box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    names = np.full((len(box_array),), "Drone", dtype="<U5")
    return box_array, names


def spread_subset(items: list[dict], count: int) -> list[dict]:
    if count >= len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=count, dtype=np.int64)
    return [items[int(index)] for index in indices]


def write_pickle(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def count_npy_points(path: Path) -> int:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] < 4:
        raise ValueError(f"Expected point cloud with shape [N, >=4] in {path}, got {array.shape}")
    return int(array.shape[0])


def load_points(path: Path, point_format: str) -> np.ndarray:
    if point_format == "bin":
        points = np.fromfile(str(path), dtype=np.float32)
        if points.size % 4:
            raise ValueError(f"Expected float32 xyzi binary point cloud in {path}")
        return points.reshape(-1, 4)
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Expected point cloud with shape [N, >=4] in {path}, got {points.shape}")
    return points[:, :4].astype(np.float32, copy=False)


def occupied_cells(points: np.ndarray, voxel_size: tuple[float, float, float]) -> int:
    if not len(points):
        return 0
    point_cloud_range = np.asarray([-16, -20, -8, 64, 20, 8], dtype=np.float32)
    lower, upper = point_cloud_range[:3], point_cloud_range[3:]
    mask = np.logical_and(points[:, :3] >= lower, points[:, :3] < upper).all(axis=1)
    in_range = points[mask, :3]
    if not len(in_range):
        return 0
    voxel = np.asarray(voxel_size, dtype=np.float32)
    coords = np.floor((in_range - lower) / voxel).astype(np.int64)
    return int(len(np.unique(coords, axis=0)))


def in_range_point_count(points: np.ndarray) -> int:
    if not len(points):
        return 0
    point_cloud_range = np.asarray([-16, -20, -8, 64, 20, 8], dtype=np.float32)
    lower, upper = point_cloud_range[:3], point_cloud_range[3:]
    return int(np.logical_and(points[:, :3] >= lower, points[:, :3] < upper).all(axis=1).sum())


def canonical_single_drone_box(boxes: np.ndarray, names: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Normalize aiQii duplicate multi-box frames into one single-drone box.

    aiQii is a single-UAV capture set. A few converted training frames contain
    duplicate or near-duplicate Drone boxes from timestamp fusion. Keeping all
    of them can trigger legacy OpenPCDet anchor target shape bugs, so we keep a
    robust median box for those frames.
    """

    if len(boxes) <= 1:
        return boxes, names, 0
    box = np.median(boxes.astype(np.float32), axis=0, keepdims=True).astype(np.float32)
    box[:, 6] = 0.0
    return box, np.asarray(["Drone"], dtype="<U5"), int(len(boxes) - 1)


def canonical_single_class_box(boxes: np.ndarray, names: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge rare duplicate aiQii boxes while preserving the UAV model label."""

    if len(boxes) <= 1:
        return boxes, names, 0
    box = np.median(boxes.astype(np.float32), axis=0, keepdims=True).astype(np.float32)
    box[:, 6] = 0.0
    unique_names, counts = np.unique(names.astype(str), return_counts=True)
    kept_name = unique_names[int(np.argmax(counts))] if len(unique_names) else "Drone"
    return box, np.asarray([kept_name], dtype=f"<U{max(len(kept_name), 1)}"), int(len(boxes) - 1)


def prepare_dataset(
    name: str,
    root: Path,
    smoke_train: int,
    smoke_val: int,
    aiqii_min_train_points: int,
    aiqii_min_train_bev_cells: int,
    aiqii_min_train_voxels: int,
) -> dict[str, object]:
    root = root.resolve()
    points_dir = root / "points"
    labels_dir = root / "labels"
    points_bin_dir = root / "points_bin"

    summary: dict[str, object] = {"name": name, "root": str(root), "splits": {}}

    for split in ("train", "val"):
        ids = read_ids(root, split)
        source_infos_by_id: dict[str, dict] = {}
        if name == "aiqii":
            source_path = root / "infos" / f"aiqii_infos_{split}.pkl"
            source_infos = load_pickle(source_path)
            source_infos_by_id = {
                info["point_cloud"]["lidar_idx"]: info for info in source_infos
            }
        pcdet_infos: list[dict] = []
        det3d_infos: list[dict] = []
        pcdet_multiclass_infos: list[dict] = []
        det3d_multiclass_infos: list[dict] = []
        all_boxes: list[np.ndarray] = []
        point_counts: list[int] = []
        skipped_sparse = 0
        duplicate_boxes_removed = 0
        skipped_sparse_examples: list[dict[str, object]] = []

        for sample_id in ids:
            points_path = points_dir / f"{sample_id}.npy"
            label_path = labels_dir / f"{sample_id}.txt"
            if not points_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Missing sample files for {sample_id}")

            if sample_id in source_infos_by_id:
                source_info = source_infos_by_id[sample_id]
                boxes = np.asarray(source_info["annos"]["gt_boxes_lidar"], dtype=np.float32)
                source_names = np.asarray(source_info["annos"]["name"])
                names = np.full((len(boxes),), "Drone", dtype="<U5")
                multiclass_names = source_names.astype(str)
                point_count = int(source_info["point_cloud"].get("num_points", -1))
                lidar_path = points_bin_dir / f"{sample_id}.bin"
                if not lidar_path.is_file():
                    raise FileNotFoundError(lidar_path)
                point_format = "bin"
                coordinate_transform = None
            else:
                boxes, names = read_labels(label_path)
                multiclass_names = names.copy()
                point_count = count_npy_points(points_path)
                # MMAUD publishes camera-style coordinates: x-right, y-down, z-forward.
                # Convert both points and boxes to x-forward, y-left, z-up at load time.
                centers = boxes[:, :3].copy()
                sizes = boxes[:, 3:6].copy()
                boxes[:, 0] = centers[:, 2]
                boxes[:, 1] = -centers[:, 0]
                boxes[:, 2] = -centers[:, 1]
                boxes[:, 3] = sizes[:, 2]
                boxes[:, 4] = sizes[:, 0]
                boxes[:, 5] = sizes[:, 1]
                boxes[:, 6] = 0.0
                lidar_path = points_path
                point_format = "npy"
                coordinate_transform = "camera_to_lidar"

            if not len(boxes):
                raise ValueError(f"No boxes for {sample_id}")
            multiclass_boxes = boxes.copy()
            if name == "aiqii":
                boxes, names, removed = canonical_single_drone_box(boxes, names)
                duplicate_boxes_removed += removed
                multiclass_boxes, multiclass_names, _ = canonical_single_class_box(
                    multiclass_boxes,
                    multiclass_names,
                )
                if split == "train":
                    points = load_points(lidar_path, point_format)
                    in_range_points = in_range_point_count(points)
                    bev_cells = occupied_cells(points, (0.5, 0.5, 16.0))
                    voxels = occupied_cells(points, (0.5, 0.5, 0.5))
                    if (
                        in_range_points < aiqii_min_train_points
                        or bev_cells < aiqii_min_train_bev_cells
                        or voxels < aiqii_min_train_voxels
                    ):
                        skipped_sparse += 1
                        if len(skipped_sparse_examples) < 20:
                            skipped_sparse_examples.append(
                                {
                                    "sample_id": sample_id,
                                    "in_range_points": in_range_points,
                                    "bev_cells": bev_cells,
                                    "voxels": voxels,
                                }
                            )
                        continue

            annos = {
                "name": names,
                "gt_boxes_lidar": boxes,
                "num_points_in_gt": np.full((len(boxes),), -1, dtype=np.int32),
            }
            pcdet_infos.append(
                {
                    "point_cloud": {
                        "num_features": 4,
                        "lidar_idx": sample_id,
                        "num_points": point_count,
                    },
                    "annos": annos,
                }
            )
            det3d_infos.append(
                {
                    "token": sample_id,
                    "lidar_path": str(lidar_path),
                    "point_format": point_format,
                    "coordinate_transform": coordinate_transform,
                    "sweeps": [],
                    "gt_boxes": boxes,
                    "gt_names": names,
                }
            )
            if name == "aiqii":
                multiclass_annos = {
                    "name": multiclass_names,
                    "gt_boxes_lidar": multiclass_boxes,
                    "num_points_in_gt": np.full((len(multiclass_boxes),), -1, dtype=np.int32),
                }
                pcdet_multiclass_infos.append(
                    {
                        "point_cloud": {
                            "num_features": 4,
                            "lidar_idx": sample_id,
                            "num_points": point_count,
                        },
                        "annos": multiclass_annos,
                    }
                )
                det3d_multiclass_infos.append(
                    {
                        "token": sample_id,
                        "lidar_path": str(lidar_path),
                        "point_format": point_format,
                        "coordinate_transform": coordinate_transform,
                        "sweeps": [],
                        "gt_boxes": multiclass_boxes,
                        "gt_names": multiclass_names,
                    }
                )
            all_boxes.append(boxes)
            point_counts.append(point_count)

        smoke_count = smoke_train if split == "train" else smoke_val
        pcdet_smoke = spread_subset(pcdet_infos, smoke_count)
        det3d_smoke = spread_subset(det3d_infos, smoke_count)

        write_pickle(root / f"mmradar_infos_{split}.pkl", pcdet_infos)
        write_pickle(root / f"mmradar_infos_{split}_smoke.pkl", pcdet_smoke)
        write_pickle(root / f"mmradar_det3d_infos_{split}.pkl", det3d_infos)
        write_pickle(root / f"mmradar_det3d_infos_{split}_smoke.pkl", det3d_smoke)
        if name == "aiqii":
            pcdet_multiclass_smoke = spread_subset(pcdet_multiclass_infos, smoke_count)
            det3d_multiclass_smoke = spread_subset(det3d_multiclass_infos, smoke_count)
            write_pickle(root / f"mmradar_infos_{split}_multiclass.pkl", pcdet_multiclass_infos)
            write_pickle(root / f"mmradar_infos_{split}_multiclass_smoke.pkl", pcdet_multiclass_smoke)
            write_pickle(root / f"mmradar_det3d_infos_{split}_multiclass.pkl", det3d_multiclass_infos)
            write_pickle(root / f"mmradar_det3d_infos_{split}_multiclass_smoke.pkl", det3d_multiclass_smoke)

        boxes_array = np.concatenate(all_boxes, axis=0)
        summary["splits"][split] = {
            "source_samples": len(ids),
            "samples": len(pcdet_infos),
            "smoke_samples": len(pcdet_smoke),
            "objects": int(len(boxes_array)),
            "skipped_sparse_train_samples": skipped_sparse,
            "duplicate_boxes_removed": duplicate_boxes_removed,
            "skipped_sparse_examples": skipped_sparse_examples,
            "point_count_min": min(point_counts),
            "point_count_max": max(point_counts),
            "point_count_mean": float(np.mean(point_counts)),
            "box_center_min": boxes_array[:, :3].min(axis=0).tolist(),
            "box_center_max": boxes_array[:, :3].max(axis=0).tolist(),
            "box_size_mean": boxes_array[:, 3:6].mean(axis=0).tolist(),
        }
        if name == "aiqii":
            multiclass_counts: dict[str, int] = {}
            for info in pcdet_multiclass_infos:
                for label in info["annos"]["name"]:
                    multiclass_counts[str(label)] = multiclass_counts.get(str(label), 0) + 1
            summary["splits"][split]["multiclass_samples"] = len(pcdet_multiclass_infos)
            summary["splits"][split]["multiclass_objects_by_class"] = multiclass_counts

    summary_path = root / "mmradar_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summaries = [
        prepare_dataset(
            "aiqii",
            args.aiqii_root,
            args.smoke_train,
            args.smoke_val,
            args.aiqii_min_train_points,
            args.aiqii_min_train_bev_cells,
            args.aiqii_min_train_voxels,
        ),
        prepare_dataset("mmaud", args.mmaud_root, args.smoke_train, args.smoke_val, 0, 0, 0),
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
