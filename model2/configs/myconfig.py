_base_ = "./cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py"
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    backbone=dict(
        type="ResNeSt",
        stem_channels=64,
        depth=50,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnest50"),
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                type="Shared4Conv1FCBBoxHead",
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                norm_cfg=norm_cfg,
                roi_feat_size=7,
                num_classes=28,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared4Conv1FCBBoxHead",
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                norm_cfg=norm_cfg,
                roi_feat_size=7,
                num_classes=28,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared4Conv1FCBBoxHead",
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                norm_cfg=norm_cfg,
                roi_feat_size=7,
                num_classes=28,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
        ],
    ),
)
# # use ResNeSt img_norm
img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True
)
img_scale = (1280, 1280)
train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
    ),
    dict(
        type="MixUp",
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        flip_ratio=0.2,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data_root = "data/"
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        ann_file=data_root + "annotations/instances_train2017.json",
        img_prefix=data_root + "train2017/",
    ),
    val=dict(
        pipeline=test_pipeline,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
    ),
)
# optimizer
optimizer = dict(type="Adam", lr=0.0003, weight_decay=0.0001)
lr_config = dict(  # 余弦退火
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr=0.000005,
)

# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=100)
log_config = dict(
    interval=1, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
evaluation = dict(interval=1, metric="bbox", save_best="mAP")
checkpoint_config = dict(interval=1)
workflow = [("train", 1), ("val", 1)]
