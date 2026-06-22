from .mmradar import MMRadarDataset

try:
    from .nuscenes import NuScenesDataset
except ImportError:
    NuScenesDataset = None
try:
    from .waymo import WaymoDataset
except ImportError:
    WaymoDataset = None

dataset_factory = {"MMRADAR": MMRadarDataset}
if NuScenesDataset is not None:
    dataset_factory["NUSC"] = NuScenesDataset
if WaymoDataset is not None:
    dataset_factory["WAYMO"] = WaymoDataset


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
