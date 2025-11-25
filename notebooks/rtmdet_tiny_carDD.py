
_base_ = 'mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

model = dict(
    bbox_head=dict(num_classes=1)
)

train_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root='./',
        ann_file='train.json',
        data_prefix=dict(img='images/'),
    )
)

val_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root='./',
        ann_file='val.json',
        data_prefix=dict(img='images/'),
    )
)

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3
    )
)

work_dir = './work_dirs/rtmdet_tiny'
