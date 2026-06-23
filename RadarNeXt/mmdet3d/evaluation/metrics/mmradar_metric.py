from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import mmengine
import numpy as np
from mmengine.evaluator import BaseMetric

_MMRADAR_ROOT = Path(__file__).resolve().parents[4]
if str(_MMRADAR_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMRADAR_ROOT))

from mmradar_common.metrics import center_distance_metrics
from mmdet3d.registry import METRICS


def _to_numpy(value) -> np.ndarray:
    if value is None:
        return np.empty((0, ), dtype=np.float32)
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@METRICS.register_module()
class MMRadarMetric(BaseMetric):
    """MMRadar center-distance/BEV/3D metrics for RadarNeXt."""

    default_prefix = 'mmradar'

    def __init__(
        self,
        ann_file: str,
        prefix: Optional[str] = None,
        collect_device: str = 'cpu',
        backend_args: Optional[dict] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.backend_args = backend_args

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            boxes = _to_numpy(pred_3d.get('bboxes_3d', None)).reshape(-1, 7)
            scores = _to_numpy(pred_3d.get('scores_3d', None)).reshape(-1)
            labels = _to_numpy(pred_3d.get('labels_3d', None)).reshape(-1)
            self.results.append({
                'sample_idx': int(data_sample['sample_idx']),
                'boxes_lidar': boxes,
                'score': scores,
                'label_preds': labels,
            })

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        infos = mmengine.load(self.ann_file, backend_args=self.backend_args)
        gt_annos = []
        for info in infos:
            boxes = np.asarray(info.get('gt_boxes', []), dtype=np.float32).reshape(-1, 7)
            gt_annos.append({
                'gt_boxes_lidar': boxes,
                'gt_boxes': boxes,
                'name': np.asarray(info.get('gt_names', [])),
            })

        det_annos = []
        for result in sorted(results, key=lambda item: item['sample_idx']):
            det_annos.append({
                'boxes_lidar': np.asarray(result['boxes_lidar'], dtype=np.float32).reshape(-1, 7),
                'score': np.asarray(result['score'], dtype=np.float32).reshape(-1),
            })

        metrics = center_distance_metrics(det_annos, gt_annos)
        metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metric_dict[key] = float(value)
        return metric_dict
