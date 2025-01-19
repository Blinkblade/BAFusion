_base_ = [
    # '../_base_/datasets/nus-3d.py',
    '../_base_/datasets/nus-3d-LC.py',
    '../_base_/models/centerpoint_voxel01_second_secfpn_nus.py',
    '../_base_/schedules/cyclic-20e.py', '../_base_/default_runtime.py'
]

# 数据环境设置
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-51.2, -52, -5.0, 51.2, 50.4, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
backend_args = None
# 模态
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)


voxel_size = [0.1, 0.1, 0.2]

# 模型结构相关参数设置
# 多头注意力每个头的向量长度
head_vector_len = 64
# fpn second 256 + 256 = 512
feature_channel = 512

# 日志设置
default_hooks = dict(
    # - `LoggerHook`：该钩子用来从`执行器（Runner）`的不同组件收集日志并将其写入终端，json 文件，tensorboard 和 wandb 等。
    # 每50个iter写一次日志
    logger=dict(type='LoggerHook', interval=500),
    # 每10轮保存一次pth
    checkpoint=dict(type='CheckpointHook', interval=1),
    visualization=dict(type='Det3DVisualizationHook'),
)

model = dict(
    type='BACenterPoint',
    # 数据预处理
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            max_num_points=10,
            voxel_size=voxel_size,
            max_voxels=(90000, 120000))),

    # 点云特征提取流程 
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    # voxel encoder
    pts_middle_encoder=dict(
        type='SparseEncoder',
        # 点云维度
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    # 图像特征提取流程
    # 图像主干
    img_backbone=dict(
        type='mmdet.CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.375,
        # 只更新最后一层
        frozen_stages = 2,
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='./checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
            checkpoint='./checkpoints/yolox_nus_800_new.pth'
            ),
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

    # 融合层

    # 不使用位置编码效果会更好
    # 位置编码层,用于为图像输入交叉注意力提供位置编码
    # position_embedding = dict(
    #     # 默认使用正余弦编码,无需梯度
    #     type = "PositionEmbeddingSine",
    #     # num_pos_feats必须是img_neck的out channel和 hiddensize的1/2
    #     num_pos_feats= int(feature_channel/2),
    # ), 

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
         norm = True,
    ),  

    # pi-feature聚合层,用于压缩通道与融合,特征最终输入主干
    MV_compress_layer = dict(
        type = "MyConv2D",
        # 因为concat因此通道数×2
         Cin = feature_channel*6,
        #  压缩后的输出通道数
         Cout = feature_channel,
        #  采用1×1卷积
         kernel_size=1, 
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


    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',

            pc_range=[-51.2,-51.2,],

            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,

            voxel_size=voxel_size[:2],

            code_size=9),

        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range = point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),

    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            pc_range=[-51.2,-51.2,],
            )
        )
    )



# nusences采样和kitti又不一样,因此用独立的采样
db_sampler = dict(
    type='NusDataBaseSampler',
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    blending_type=None,
    depth_consistent=True,
    check_2D_collision=True,
    collision_thr=[0, 0.3, 0.5, 0.7],
    # collision_in_classes=True,
    mixup=0.7,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
        backend_args=backend_args
        )

train_pipeline = [
    # 读取点云
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    # 读取多视角图像 
    dict(type='LoadMultiViewImageFromFiles', 
            backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True,
                                    with_label_3d=True,
                                    with_bbox=True,
                                    with_label=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    # 在所有数据增强之前先进行采样增强
    dict(type='ObjectSampleV2', db_sampler=db_sampler, sample_2d=True),

    # 输入图像规整
    # 多视角图像,原来的resize3d不可用需要重新实现
    dict(type='ResizeNus',scale=(1440, 800)),
    # dict(type='Resize3D',scale=(1280,384)),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='NusRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        # 其实不需要2D标注,只需要Img
        keys=[
            'points', 'img', 'gt_bboxes_3d', 
            'gt_labels_3d', 
            'gt_bboxes','gt_labels'
        ]),
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    # 读取多视角图像 
    dict(type='LoadMultiViewImageFromFiles', 
            backend_args=backend_args),
    # 输入图像规整
    # 多视角图像,原来的resize3d不可用需要重新实现
    dict(type='ResizeNus',scale=(1440, 800)),
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1440, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='GlobalRotScaleTrans',
    #             rot_range=[0, 0],
    #             scale_ratio_range=[1., 1.],
    #             translation_std=[0, 0, 0]),
    #         dict(type='NusRandomFlip3D'),
    #         dict(
    #             type='PointsRangeFilter', point_cloud_range=point_cloud_range)
    #     ]),
    # 打包图像和点云
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]
eval_pipeline = test_pipeline

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=dict(classes=class_names),
            test_mode=False,
            data_prefix=data_prefix,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))
test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))







max_epoch = 20
auto_scale_lr = dict(base_batch_size=32, enable=True)
lr = 1e-4
# This schedule is mainly used by models on nuScenes dataset
# max_norm=10 is better for SECOND
optim_wrapper = dict(
    optimizer=dict(lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=5, norm_type=2))
# learning rate
optim_wrapper = dict(
    # norm相当于罚,当grad的l2范数超过10的时候会被限制
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=5, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=max_epoch * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch * 0.6,
        eta_min=lr * 1e-4,
        begin=max_epoch * 0.4,
        end=max_epoch * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=max_epoch * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=max_epoch * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=max_epoch * 0.6,
        eta_min=1,
        begin=max_epoch * 0.4,
        end=max_epoch * 1,
        convert_to_iter_based=True)
]
# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=max_epoch, val_interval=1)
val_cfg = dict()
test_cfg = dict()


# pretrained = None
load_from = None
resume = None

