from __future__ import annotations

import sys
from pathlib import Path

_MMRADAR_ROOT = Path(__file__).resolve().parents[3]
if str(_MMRADAR_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMRADAR_ROOT))

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS
from mmradar_common.det3d_dataset import MMRadarDet3DDatasetMixin


@DATASETS.register_module
class MMRadarDataset(MMRadarDet3DDatasetMixin, PointCloudDataset):
    pass
