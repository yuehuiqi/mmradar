_base_ = ['../_base_/default_runtime.py']

backend_args = None
class_names = ['Drone']
metainfo = dict(classes=class_names)
dataset_type = 'MMRadarDataset'
input_modality = dict(use_lidar=True, use_camera=False)

point_cloud_range = [-16, -20, -8, 64, 20, 8]
voxel_size = [0.5, 0.5, 16]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points']),
]

data_root = ''
train_ann = 'mmradar_det3d_infos_train_smoke.pkl'
val_ann = 'mmradar_det3d_infos_val_smoke.pkl'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(pts=''),
            ann_file=train_ann,
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=''),
        ann_file=val_ann,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='MMRadarMetric',
    ann_file=val_ann,
    backend_args=backend_args)
test_evaluator = val_evaluator

model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            max_voxels=(12000, 20000),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)),
    pts_voxel_encoder=dict(
        type='Radar7PillarFeatureNet',
        feat_channels=[64],
        in_channels=4,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        with_distance=False,
        use_xyz=True,
        use_rcs=True,
        use_vr=False,
        use_vr_comp=False,
        use_time=False,
        use_elevation=True,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=(80, 160)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=384,
        tasks=[dict(num_class=1, class_names=['Drone'])],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range[:2],
            max_num=200,
            score_threshold=0.05,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[160, 80, 1],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=128,
            min_radius=1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=point_cloud_range,
            max_per_img=200,
            max_pool_nms=False,
            min_radius=[1],
            score_threshold=0.05,
            pc_range=point_cloud_range[:2],
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=100,
            nms_thr=0.2)))

lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=32,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=32,
        eta_min=lr * 10),
    dict(
        type='CosineAnnealingLR',
        T_max=48,
        begin=32,
        by_epoch=True,
        convert_to_iter_based=True,
        end=80,
        eta_min=lr * 1e-4),
    dict(
        type='CosineAnnealingMomentum',
        T_max=32,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=32,
        eta_min=0.85 / 0.95),
    dict(
        type='CosineAnnealingMomentum',
        T_max=48,
        begin=32,
        by_epoch=True,
        convert_to_iter_based=True,
        end=80,
        eta_min=1),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

auto_scale_lr = dict(enable=False, base_batch_size=16)
