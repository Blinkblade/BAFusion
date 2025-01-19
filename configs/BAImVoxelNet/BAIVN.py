_base_ = [
    '../_base_/schedules/mmdet-schedule-1x.py', '../_base_/default_runtime.py'
]


dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
# class_names = ['Pedestrian', 'Cyclist', 'Car']
class_names = ['Car']
input_modality = dict(use_lidar=True, use_camera=True)
# point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
metainfo = dict(classes=class_names)
backend_args = None
# 点云设定
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range=[0, -40, -3, 70.4, 40, 1]
# 多头注意力每个头的向量长度
head_vector_len = 64

# 按照imvoxenet 的 neck3d0来
feature_channel = 256


default_hooks = dict(
    # - `LoggerHook`：该钩子用来从`执行器（Runner）`的不同组件收集日志并将其写入终端，json 文件，tensorboard 和 wandb 等。
    # 每1轮写一次日志
    logger=dict(type='LoggerHook', interval=5),
    # 最多保存十个
    checkpoint=dict(type='CheckpointHook', max_keep_ckpts=10),
    visualization=dict(type='Det3DVisualizationHook'),
)


model = dict(
    type='BAImVoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        # img setting
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        
        
        # pts setting
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=[0, -40, -3, 70.4, 40, 1],
            voxel_size=voxel_size,
            max_voxels=(16000, 40000)),
        
        ),
    
    
    # img branch
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_3d=dict(type='OutdoorImVoxelNeck', in_channels=64, out_channels=256),
    
    # pts branch
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    
    # 点云主干        
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        # 最后一层输出之和为256与neck3d0一致(256)
        out_channels=[128, 128]),
    # 不需要这两个
    # img_channel_attention = None,   
    # 不使用位置编码效果会更好
    # 位置编码层,用于为图像输入交叉注意力提供位置编码
    # position_embedding = None
    
    # 融合模块
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
         norm = True,
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



    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-0.16, -39.68, -1.78, 68.96, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
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
    n_voxels=[216, 248, 12],
    coord_type='LIDAR',
    prior_generator=dict(
        type='AlignedAnchor3DRangeGenerator',
        ranges=[[-0.16, -39.68, -3.08, 68.96, 39.68, 0.76]],
        rotations=[.0]),
    train_cfg=dict(
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))



train_pipeline = [
    # 加载点云
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args), 
    # 加载图片
    # dict(type='LoadImageFromFile', backend_args=backend_args),
    # Mono3D和普通的相比还额外加载了一些相机参数
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    
    # 其实没必要加载这么多label
    dict(type='LoadAnnotations3D', with_bbox_3d=True,
                                    with_label_3d=True,
                                    with_bbox=True,
                                    with_label=True,
                                    backend_args=backend_args),
    # dict(type='LoadAnnotations3D', backend_args=backend_args),
    # resize一下不然采样完了总是不匹配
    dict(type='Resize3D',scale=(1280,384)),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # 不再随机resize
    # dict(
    #     type='RandomResize', scale=[(1173, 352), (1387, 416)],
    #     keep_ratio=True),

    # 经典过滤
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),

    #新打包,这些数据都需要
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 
            'gt_labels_3d', 'gt_bboxes','gt_labels'
        ]),

    # dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),    
    
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='Resize3D',scale=(1280,384)),
    # dict(type='Resize', scale=(1280, 384), keep_ratio=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),

    # 打包图像和点云
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

train_dataloader = dict(
    batch_size=3,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            # data_prefix=dict(img='training/image_2'),
            data_prefix=dict(pts='training/velodyne_reduced', 
                             img='training/image_2'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_val.pkl',
        # data_prefix=dict(img='training/image_2'),
        data_prefix=dict(pts='training/velodyne_reduced', 
                         img='training/image_2'),

        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]


# runtime
find_unused_parameters = True  # only 1 of 4 FPN outputs is used

# 设置tensorboard
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
