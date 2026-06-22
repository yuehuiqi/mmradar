import itertools
import logging
from det3d.utils.config_tool import get_downsample_factor

tasks = [dict(num_class=1, class_names=["Drone"])]
class_names = list(itertools.chain(*[task["class_names"] for task in tasks]))

point_cloud_range = [-16, -20, -8, 64, 20, 8]
voxel_size = [0.5, 0.5, 18]
post_center_limit_range = point_cloud_range

target_assigner = dict(tasks=tasks)

model = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64, 64],
        num_input_features=4,
        with_distance=False,
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[1, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[1, 2, 4],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=128 * 3,
        tasks=tasks,
        dataset="waymo",
        weight=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={"reg": (2, 2), "height": (1, 2), "dim": (3, 2), "rot": (2, 2)},
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=1,
)

train_cfg = dict(assigner=assigner)
test_cfg = dict(
    post_center_limit_range=post_center_limit_range,
    nms=dict(nms_pre_max_size=4096, nms_post_max_size=500, nms_iou_threshold=0.1),
    score_threshold=0.05,
    pc_range=point_cloud_range[:2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=voxel_size[:2],
)

dataset_type = "MMRadarDataset"
nsweeps = 1
data_root = "/mnt/e/Scholar/dataset/mmaud/mmaud_radar_camera_openpcdet"
train_anno = data_root + "/mmradar_det3d_infos_train_smoke.pkl"
val_anno = data_root + "/mmradar_det3d_infos_val_smoke.pkl"

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    db_sampler=None,
    class_names=class_names,
    no_augmentation=True,
)
val_preprocessor = dict(mode="val", shuffle_points=False)

voxel_generator = dict(
    range=point_cloud_range,
    voxel_size=voxel_size,
    max_points_in_voxel=32,
    max_voxel_num=[12000, 20000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False)
lr_config = dict(type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4)
checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type="TextLoggerHook")])

total_epochs = 1
device_ids = range(1)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "./work_dirs/mmradar_centerpoint_mmaud_smoke"
load_from = None
resume_from = None
workflow = [("train", 1)]
