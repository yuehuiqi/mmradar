from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import center_distance_metrics, format_metrics


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class MMRadarDet3DDatasetMixin:
    NumPointFeatures = 4

    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        nsweeps=1,
        load_interval=1,
        **kwargs,
    ):
        self.load_interval = load_interval
        self.nsweeps = nsweeps
        if self.nsweeps != 1:
            raise ValueError("MMRadarDataset only supports single-frame radar point clouds.")
        print("Using 1 radar frame")
        super().__init__(
            root_path,
            info_path,
            pipeline,
            test_mode=test_mode,
            class_names=class_names,
        )
        self._info_path = info_path
        self._class_names = class_names or ["Drone"]
        self._num_point_features = self.NumPointFeatures

    def load_infos(self, info_path):
        with open(info_path, "rb") as handle:
            infos = pickle.load(handle)
        self._mmradar_infos = infos[:: self.load_interval]
        print("Using {} MMRadar frames".format(len(self._mmradar_infos)))

    def __len__(self):
        if not hasattr(self, "_mmradar_infos"):
            self.load_infos(self._info_path)
        return len(self._mmradar_infos)

    def get_sensor_data(self, idx):
        info = self._mmradar_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "MMRadarDataset",
        }
        data, _ = self.pipeline(res, info)
        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    @property
    def ground_truth_annotations(self):
        if not hasattr(self, "_mmradar_infos"):
            self.load_infos(self._info_path)
        annotations = []
        for info in self._mmradar_infos:
            boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32).reshape(-1, 7)
            names = np.asarray(info.get("gt_names", []))
            annotations.append(
                {
                    "gt_boxes": boxes,
                    "gt_boxes_lidar": boxes,
                    "name": names,
                }
            )
        return annotations

    def evaluation(self, detections, output_dir=None, testset=False):
        if not hasattr(self, "_mmradar_infos"):
            self.load_infos(self._info_path)

        if isinstance(detections, dict):
            ordered_detections = [
                detections.get(info["token"], {}) for info in self._mmradar_infos
            ]
        else:
            ordered_detections = list(detections)

        det_annos = []
        for det in ordered_detections:
            boxes = (
                det.get("boxes_lidar")
                if isinstance(det, dict)
                else None
            )
            if boxes is None and isinstance(det, dict):
                boxes = det.get("box3d_lidar", det.get("boxes", det.get("box3d_lidar_preds")))
            scores = det.get("score", det.get("scores", det.get("scores_lidar"))) if isinstance(det, dict) else None
            labels = det.get("label_preds", det.get("labels", det.get("labels_3d"))) if isinstance(det, dict) else None
            label_array = _to_numpy(labels).reshape(-1).astype(np.int64)
            class_names = np.asarray(self._class_names)
            names = np.asarray([], dtype="<U1")
            if len(label_array):
                names = np.asarray(
                    [
                        str(class_names[label]) if 0 <= label < len(class_names) else str(label)
                        for label in label_array
                    ]
                )
            det_annos.append(
                {
                    "boxes_lidar": _to_numpy(boxes).reshape(-1, 7),
                    "score": _to_numpy(scores).reshape(-1),
                    "name": names,
                }
            )
        metrics = center_distance_metrics(det_annos, self.ground_truth_annotations)
        formatted = format_metrics(metrics)
        result_dict = {
            "results": {"mmradar": formatted},
            "detail": {"mmradar": metrics},
        }
        return result_dict, metrics
