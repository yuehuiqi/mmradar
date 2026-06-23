_base_ = './radarnext_centerpoint_mmradar_base.py'

data_root = '/mnt/e/Scholar/dataset/mmaud/mmaud_radar_camera_openpcdet'
train_ann = data_root + '/mmradar_det3d_infos_train_smoke.pkl'
val_ann = data_root + '/mmradar_det3d_infos_val_smoke.pkl'

train_dataloader = dict(
    dataset=dict(dataset=dict(data_root=data_root, ann_file=train_ann)))
val_dataloader = dict(dataset=dict(data_root=data_root, ann_file=val_ann))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=val_ann)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=2, val_interval=1)
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3))
work_dir = './work_dirs/radarnext_centerpoint_mmaud_smoke'
