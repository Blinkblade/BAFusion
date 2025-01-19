_base_ = [
    # '../_base_/schedules/cyclic-40e.py', 
    '../_base_/schedules/SGD.py',     
    '../_base_/default_runtime.py',
    # '../_base_/models/parta2.py',
    '../_base_/datasets/kitti-3d-3class_LC_aug.py',
]


# log setting
default_hooks = dict(
    # - `LoggerHook`：该钩子用来从`执行器（Runner）`的不同组件收集日志并将其写入终端，json 文件，tensorboard 和 wandb 等。
    # 每1轮写一次日志
    logger=dict(type='LoggerHook', interval=50),
    # 每10轮保存一次pth
    checkpoint=dict(type='CheckpointHook', interval=1),
    visualization=dict(type='Det3DVisualizationHook'),
)


# model settings
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# 多头注意力每个头的向量长度
head_vector_len = 64
# fpn second 256 + 256 = 512
feature_channel = 512


model = dict(
    type='BAPartA2',
    # 数据预处理
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    # point 特征提取
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseUNet',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    
    # 图像特征提取
    img_backbone=dict(
        type='mmdet.CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.375,
        # 只更新最后一层
        frozen_stages = 2,
        init_cfg=dict(type='Pretrained', checkpoint='./checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'),
    ),
    # 注意力后面再看
    img_channel_attention = None,    
    # img neck层，参数将随着梯度调整,在这里传入backbone提取的四层特征，考虑引入通道注意力
    img_neck=dict(
        type='YOLOXPAFPN1',
        in_channels=[96, 192, 384],
        # 输出512通道
        out_channels=feature_channel,
        num_csp_blocks=1),

    # 不使用位置编码效果会更好
    # 位置编码层,用于为图像输入交叉注意力提供位置编码
    # position_embedding = dict(
    #     # 默认使用正余弦编码,无需梯度
    #     type = "PositionEmbeddingSine",
    #     # num_pos_feats必须是img_neck的out channel和 hiddensize的1/2
    #     num_pos_feats= int(feature_channel/2),
    # ), 

    # 融合层次
    # image-point cross attention 交叉注意力融合
    IP_cross_attention = dict(
        type = "CrossFocusedLinearAttention",
        # dim必须和img_neck的输出以及pts_backbone的输入通道完全相同
        dim = feature_channel,
        num_patches = None,
        num_heads = int(feature_channel/head_vector_len),
        # 偏置与尺度,选择None为自动生成
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0.1, 
        proj_drop=0.1, 
        # 不进行下采样
        sr_ratio=1,
        focusing_factor=3, 
        kernel_size=3,
        # 输出值和q加和进行融合
        add = True,
    ),


    
    # point-image cross attention 交叉注意力融合
    PI_cross_attention = dict(
        type = "CrossFocusedLinearAttention",
        # dim必须和img_neck的输出以及pts_backbone的输入通道完全相同
        dim = feature_channel,
        num_patches=None,
        num_heads = int(feature_channel/head_vector_len),
        # 偏置与尺度,选择None为自动生成
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0.1, 
        proj_drop=0.1, 
        # 不进行下采样
        sr_ratio=1,
        focusing_factor=3, 
        kernel_size=3,
        # 输出值和q加和进行融合
        add = True,
    ),  
    
    # ip-feature聚合层,用于压缩通道与融合
    IP_compress_layer = dict(
        type = "MyConv2D",
        # 因为concat因此通道数×2
         Cin = feature_channel*2,
        #  压缩后的输出通道数
         Cout = feature_channel,
        #  采用1×1卷积
         kernel_size=3, 
         stride=1,
        #  自适应计算padding填充来保证大小不变
         padding=None,
        #  group和扩大
         g=1, 
         d=1, 
        #  默认采用nn.SiLU激活
         act=True,
        #  不激活也不norm
         norm = True,
    ), 

    # pi-feature聚合层,用于压缩通道与融合,特征最终输入主干
    PI_compress_layer = dict(
        type = "MyConv2D",
        # 因为concat因此通道数×2
         Cin = feature_channel*2,
        #  压缩后的输出通道数
         Cout = feature_channel,
        #  采用1×1卷积
         kernel_size=3, 
         stride=1,
        #  自适应计算padding填充来保证大小不变
         padding=None,
        #  group和扩大
         g=1, 
         d=1, 
        #  默认采用nn.SiLU激活
         act=True,
        #  不激活也不norm
         norm = True,
    ),  





    rpn_head=dict(
        type='PartA2RPNHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        assigner_per_size=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    roi_head=dict(
        type='PartAggregationROIHead',
        num_classes=3,
        semantic_head=dict(
            type='PointwiseSemanticHead',
            in_channels=16,
            extra_width=0.2,
            seg_score_thr=0.3,
            num_classes=3,
            loss_seg=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_part=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)),
        seg_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='max')),
        bbox_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='avg')),
        bbox_head=dict(
            type='PartA2BboxHead',
            num_classes=3,
            seg_in_channels=16,
            part_in_channels=4,
            seg_conv_channels=[64, 64],
            part_conv_channels=[64, 64],
            merge_conv_channels=[128, 128],
            down_conv_channels=[128, 256],
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            shared_fc_channels=[256, 512, 512, 512],
            cls_channels=[256, 256],
            reg_channels=[256, 256],
            dropout_ratio=0.1,
            roi_feat_size=14,
            with_corner_loss=True,
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1)
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=9000,
            nms_post=512,
            max_num=512,
            nms_thr=0.8,
            score_thr=0,
            use_rotate_nms=False),
        rcnn=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1)
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.55,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.75,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1024,
            nms_post=100,
            max_num=100,
            nms_thr=0.7,
            score_thr=0,
            use_rotate_nms=True),
        rcnn=dict(
            use_rotate_nms=True,
            use_raw_score=True,
            nms_thr=0.01,
            score_thr=0.1)))





# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
# 多模态
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(classes=class_names)
backend_args = None


# db_sampler = dict(
#     type='MMDataBaseSamplerV2',
#     data_root=data_root,
#     info_path=data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
#     blending_type=None,
#     depth_consistent=True,
#     check_2D_collision=True,
#     collision_thr=[0, 0.3, 0.5, 0.7],
#     # collision_in_classes=True,
#     mixup=0.7,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
#     classes=class_names,
#     sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     backend_args=backend_args)




train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    # 加载图片
    dict(type='LoadImageFromFile', backend_args=None),
    # 加载2d 3d annotation
    dict(type='LoadAnnotations3D', with_bbox_3d=True,
                                    with_label_3d=True,
                                    with_bbox=True,
                                    with_label=True),
    # 一定要先采样,不然经过别的变换就乱了
    # dict(type='ObjectSampleV2', db_sampler=db_sampler, sample_2d=True),
    # noise放着,挺好
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    # resize一下不然采样完了总是不匹配
    dict(type='Resize3D',scale=(1280,384)),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    #新打包,这些数据都徐需要
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 
            'gt_labels_3d', 'gt_bboxes','gt_labels'
        ]),
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    # 加载图像数据
    dict(type='LoadImageFromFile', backend_args=None),
    # resize一下不然采样完了总是不匹配
    dict(type='Resize3D',scale=(1280,384)),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img']),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = test_pipeline



train_dataloader = dict(
    batch_size=6,
    num_workers=6,
#     repeat=2
    dataset=dict(        
        type='RepeatDataset',
        times=2,
        dataset=dict(pipeline=train_pipeline, metainfo=metainfo)
    )
)

test_dataloader = dict(    
    batch_size=1,
    num_workers=1,
    dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(pipeline=test_pipeline, metainfo=metainfo))

# train_dataloader = dict(
#     batch_size=8,
#     num_workers=10,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type='RepeatDataset',
#         times=2,
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             ann_file='kitti_infos_train.pkl',
#             data_prefix=dict(pts='training/velodyne_reduced'),
#             pipeline=train_pipeline,
#             modality=input_modality,
#             metainfo=dict(classes=class_names),
#             box_type_3d='LiDAR',
#             test_mode=False,
#             backend_args=backend_args)))
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='kitti_infos_test.pkl',
#         data_prefix=dict(pts='testing/velodyne_reduced'),
#         pipeline=test_pipeline,
#         modality=input_modality,
#         metainfo=dict(classes=class_names),
#         box_type_3d='LiDAR',
#         test_mode=True,
#         backend_args=backend_args))
# val_dataloader = dict(
#     batch_size=24,
#     num_workers=10,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='kitti_infos_val.pkl',
#         data_prefix=dict(pts='training/velodyne_reduced'),
#         pipeline=eval_pipeline,
#         modality=input_modality,
#         metainfo=dict(classes=class_names),
#         box_type_3d='LiDAR',
#         test_mode=True,
#         backend_args=backend_args))


val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# Part-A2 uses a different learning rate from what SECOND uses.
lr = 0.001
epoch_num = 10
optim_wrapper = dict(
    # norm相当于罚,当grad的l2范数超过10的时候会被限制
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=5, norm_type=2))

# param_scheduler = [
#     dict(
#         type='CosineAnnealingLR',
#         T_max=epoch_num * 0.4,
#         eta_min=lr * 10,
#         begin=0,
#         end=epoch_num * 0.4,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=epoch_num * 0.6,
#         eta_min=lr * 1e-4,
#         begin=epoch_num * 0.4,
#         end=epoch_num * 1,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingMomentum',
#         T_max=epoch_num * 0.4,
#         eta_min=0.85 / 0.95,
#         begin=0,
#         end=epoch_num * 0.4,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingMomentum',
#         T_max=epoch_num * 0.6,
#         eta_min=1,
#         begin=epoch_num * 0.4,
#         end=epoch_num * 1,
#         convert_to_iter_based=True)
# ]

find_unused_parameters = True

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)


train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# pretrained = None
# mmdetection3d/work_dirs/BAPartA2-cyc50e/epoch_50.pth
load_from = "./work_dirs/BAPartA2-cyc50e/epoch_50.pth"
resume = None
