from pillarnet_aiqii_multiclass_smoke import *  # noqa: F401,F403

train_anno = data_root + "/mmradar_det3d_infos_train_multiclass.pkl"
val_anno = data_root + "/mmradar_det3d_infos_val_multiclass.pkl"

data["train"]["info_path"] = train_anno
data["train"]["ann_file"] = train_anno
data["val"]["info_path"] = val_anno
data["val"]["ann_file"] = val_anno
data["test"]["info_path"] = val_anno
data["test"]["ann_file"] = val_anno

total_epochs = 80
checkpoint_config = dict(interval=5, max_keep_ckpts=17)
workflow = [("train", 5), ("val", 1)]
work_dir = "./work_dirs/mmradar_pillarnet_aiqii_multiclass_full"
