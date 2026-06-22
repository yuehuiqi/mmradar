from __future__ import annotations

import copy
import pickle

import numpy as np

from .metrics import center_distance_metrics, format_metrics


class MMRadarDatasetMixin:
    """Dataset implementation compatible with old and new OpenPCDet forks."""

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.mmradar_infos = []
        self.include_mmradar_data(self.mode)

    def include_mmradar_data(self, mode):
        infos = []
        for relative_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / relative_path
            if not info_path.is_file():
                raise FileNotFoundError(info_path)
            with info_path.open("rb") as handle:
                infos.extend(pickle.load(handle))
        self.mmradar_infos = infos
        if self.logger is not None:
            self.logger.info("Total samples for MMRadar dataset: %d", len(infos))

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.mmradar_infos) * self.total_epochs
        return len(self.mmradar_infos)

    def get_lidar(self, sample_id):
        path = self.root_path / self.dataset_cfg.get("POINTS_DIR", "points") / f"{sample_id}.npy"
        points = np.load(path)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(f"Expected Nx4 points in {path}, got {points.shape}")
        points = points.astype(np.float32, copy=False)
        if self.dataset_cfg.get("COORDINATE_TRANSFORM", None) == "camera_to_lidar":
            transformed = points.copy()
            transformed[:, 0] = points[:, 2]
            transformed[:, 1] = -points[:, 0]
            transformed[:, 2] = -points[:, 1]
            points = transformed
        return points

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index %= len(self.mmradar_infos)
        info = copy.deepcopy(self.mmradar_infos[index])
        sample_id = info["point_cloud"]["lidar_idx"]
        data_dict = {
            "frame_id": sample_id,
            "points": self.get_lidar(sample_id),
            # Newer OpenPCDet forks expect these augmentation bookkeeping fields
            # to exist even when no augmentation has touched the sample yet.
            "flip_x": False,
            "flip_y": False,
            "noise_rot": 0.0,
            "noise_scale": 1.0,
        }
        if "annos" in info:
            data_dict.update(
                gt_names=np.asarray(info["annos"]["name"]),
                gt_boxes=np.asarray(info["annos"]["gt_boxes_lidar"], dtype=np.float32),
            )
        return self.prepare_data(data_dict=data_dict)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            scores = box_dict["pred_scores"].detach().cpu().numpy()
            boxes = box_dict["pred_boxes"].detach().cpu().numpy()
            labels = box_dict["pred_labels"].detach().cpu().numpy()
            names = np.asarray(class_names)[labels - 1] if len(labels) else np.asarray([], dtype="<U5")
            anno = {
                "name": names,
                "score": scores,
                "boxes_lidar": boxes,
                "pred_labels": labels,
                "frame_id": batch_dict["frame_id"][index],
            }
            annos.append(anno)
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if not self.mmradar_infos or "annos" not in self.mmradar_infos[0]:
            return "No ground-truth boxes for evaluation", {}
        gt_annos = [info["annos"] for info in self.mmradar_infos]
        metrics = center_distance_metrics(det_annos, gt_annos)
        return format_metrics(metrics), metrics
