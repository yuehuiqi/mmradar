_base_ = './radarnext_centerpoint_aiqii_smoke.py'

class_names = ['Air3S', 'Mini4pro', 'Mavic3Pro', 'jingling4']
metainfo = dict(classes=class_names)

data_root = '/mnt/e/Scholar/dataset/aiQiiDataset/radar_openpcdet'
train_ann = data_root + '/mmradar_det3d_infos_train_multiclass_smoke.pkl'
val_ann = data_root + '/mmradar_det3d_infos_val_multiclass_smoke.pkl'

train_dataloader = dict(
    dataset=dict(dataset=dict(ann_file=train_ann, metainfo=metainfo)))
val_dataloader = dict(dataset=dict(ann_file=val_ann, metainfo=metainfo))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=val_ann)
test_evaluator = val_evaluator

model = dict(
    pts_bbox_head=dict(
        tasks=[dict(num_class=4, class_names=class_names)]))

work_dir = './work_dirs/radarnext_centerpoint_aiqii_multiclass_smoke'
