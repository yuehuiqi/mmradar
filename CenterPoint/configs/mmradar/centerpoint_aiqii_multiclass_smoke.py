from centerpoint_aiqii_smoke import *  # noqa: F401,F403

tasks = [dict(num_class=4, class_names=["Air3S", "Mini4pro", "Mavic3Pro", "jingling4"])]
class_names = ["Air3S", "Mini4pro", "Mavic3Pro", "jingling4"]

target_assigner["tasks"] = tasks
model["bbox_head"]["tasks"] = tasks
train_cfg["assigner"]["target_assigner"]["tasks"] = tasks
train_preprocessor["class_names"] = class_names

train_anno = data_root + "/mmradar_det3d_infos_train_multiclass_smoke.pkl"
val_anno = data_root + "/mmradar_det3d_infos_val_multiclass_smoke.pkl"
data["train"]["info_path"] = train_anno
data["train"]["ann_file"] = train_anno
data["train"]["class_names"] = class_names
data["val"]["info_path"] = val_anno
data["val"]["ann_file"] = val_anno
data["val"]["class_names"] = class_names
data["test"]["info_path"] = val_anno
data["test"]["ann_file"] = val_anno
data["test"]["class_names"] = class_names

work_dir = "./work_dirs/mmradar_centerpoint_aiqii_multiclass_smoke"

