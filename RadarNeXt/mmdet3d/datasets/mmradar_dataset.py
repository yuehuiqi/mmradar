from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import mmengine
import numpy as np

_MMRADAR_ROOT = Path(__file__).resolve().parents[3]
if str(_MMRADAR_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMRADAR_ROOT))

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes

from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class MMRadarDataset(Det3DDataset):
    """Single-class millimeter-wave radar dataset for aiQiiDataset/MMAUD.

    The converter in ``environment/prepare_mmradar_datasets.py`` writes a
    compact list of samples with absolute point-cloud paths and OpenPCDet-style
    boxes.  RadarNeXt is based on MMDetection3D 1.x, so this adapter exposes
    that list as standard ``Det3DDataset`` data records.
    """

    METAINFO = {
        'classes': ('Drone', ),
        'palette': [(255, 77, 77)],
    }

    def load_data_list(self) -> List[dict]:
        raw_infos = mmengine.load(self.ann_file, backend_args=self.backend_args)
        data_list = []
        classes = self.metainfo['classes']

        for sample_idx, info in enumerate(raw_infos):
            boxes = np.asarray(info.get('gt_boxes', []), dtype=np.float32).reshape(-1, 7)
            names = np.asarray(info.get('gt_names', []))

            instances = []
            for box, name in zip(boxes, names):
                label = classes.index(str(name)) if str(name) in classes else -1
                instances.append({
                    'bbox_3d': box.tolist(),
                    'bbox_label_3d': label,
                })

            lidar_path = str(info['lidar_path'])
            data_info = {
                'sample_idx': sample_idx,
                'token': info.get('token', str(sample_idx)),
                'lidar_points': {
                    'lidar_path': lidar_path,
                    'num_pts_feats': 4,
                },
                'instances': instances,
            }
            data_list.append(self.parse_data_info(data_info))

        return data_list

    def parse_ann_info(self, info: dict) -> dict:
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = {
                'gt_bboxes_3d': np.zeros((0, 7), dtype=np.float32),
                'gt_labels_3d': np.zeros((0, ), dtype=np.int64),
            }

        ann_info = self._remove_dontcare(ann_info)
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=7,
            origin=(0.5, 0.5, 0.5),
        )
        return ann_info
