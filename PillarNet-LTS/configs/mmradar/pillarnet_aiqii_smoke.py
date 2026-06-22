import itertools
import logging

tasks = [dict(stride=8, class_names=["Drone"])]
class_names = list(itertools.chain(*[task["class_names"] for task in tasks]))

pillar_size = 0.5
point_cloud_range = [-16, -20, -8, 64, 20, 8]
post_center_limit_range = point_cloud_range

model = dict(
    type="PillarNet",
    reader=dict(
        type="DynamicPFE",
        in_channels=4,
        num_filters=(32,),
        pillar_size=pillar_size,
        pc_range=point_cloud_range,
    ),
    backbone=dict(type="PillarResNet18", in_channels=32),
    neck=dict(
        type="RPNV1",
        layer_nums=[3, 3],
        num_filters=128,
        in_channels=[256, 256],
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        tasks=tasks,
        in_channels=[128],
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={"reg": (2, 2), "height": (1, 2), "dim": (3, 2), "rot": (2, 2)},
        reg_iou="GIoU",
        pillar_size=pillar_size,
        point_cloud_range=point_cloud_range,
    ),
)

train_cfg = dict(
    assigner=dict(
        target_assigner=dict(tasks=tasks),
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=1,
        pc_range=point_cloud_range,
        pillar_size=pillar_size,
    ),
    hm_weight=1,
    bbox_weight=2,
    iou_weight=1,
    reg_iou_weight=2,
)

test_cfg = dict(
    nms=dict(
        use_multi_class_nms=True,
        nms_pre_max_size=[2048],
        nms_post_max_size=[200],
        nms_iou_threshold=[0.1],
    ),
    rectifier=[0],
    score_threshold=0.05,
    post_center_limit_range=post_center_limit_range,
)

dataset_type = "MMRadarDataset"
nsweeps = 1
data_root = "/mnt/e/Scholar/dataset/aiQiiDataset/radar_openpcdet"
train_anno = data_root + "/mmradar_det3d_infos_train_smoke.pkl"
val_anno = data_root + "/mmradar_det3d_infos_val_smoke.pkl"

db_sampler = None

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    db_sampler=db_sampler,
    class_names=class_names,
)
val_preprocessor = dict(mode="val", shuffle_points=False)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
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
        use_cbgs=False,
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
work_dir = "./work_dirs/mmradar_pillarnet_aiqii_smoke"
load_from = None
resume_from = None
workflow = [("train", 1)]
