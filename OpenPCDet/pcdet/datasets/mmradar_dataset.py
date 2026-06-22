import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mmradar_common.pcdet_dataset import MMRadarDatasetMixin
from .dataset import DatasetTemplate


class MMRadarDataset(MMRadarDatasetMixin, DatasetTemplate):
    pass

