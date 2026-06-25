_base_ = './radarnext_centerpoint_aiqii_multiclass_smoke.py'

data_root = '/mnt/e/Scholar/dataset/aiQiiDataset/radar_openpcdet'
train_ann = data_root + '/mmradar_det3d_infos_train_multiclass.pkl'
val_ann = data_root + '/mmradar_det3d_infos_val_multiclass.pkl'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(dataset=dict(ann_file=train_ann)))
val_dataloader = dict(num_workers=4, dataset=dict(ann_file=val_ann))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=val_ann)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=80, val_interval=5)
default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=10))
work_dir = './work_dirs/radarnext_centerpoint_aiqii_multiclass_full'
