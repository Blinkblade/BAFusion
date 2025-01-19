_base_ = [
    # '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    # '../_base_/datasets/kitti-3d-3class_LC.py',
    '../_base_/datasets/kitti-3d-3class_LC_aug.py',
    # '../_base_/datasets/kitti-3d-tiny-3class_LC.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

# 用concat&batchnorm代替add&layernorm
default_hooks = dict(
    # - `LoggerHook`：该钩子用来从`执行器（Runner）`的不同组件收集日志并将其写入终端，json 文件，tensorboard 和 wandb 等。
    # 每1轮写一次日志
    logger=dict(type='LoggerHook', interval=50),
    # 每10轮保存一次pth
    checkpoint=dict(type='CheckpointHook', interval=5),
    visualization=dict(type='Det3DVisualizationHook'),
)



# 确定传递特征图的通道数
# 这个通道数是由fpn second的输出确定的
# fpn second输出为128,128,128,多尺度特征concat生成384通道的最终输出
feature_channel = 384
# 这个是voxel encoder输出的特征维度,默认为64,为了方便修改统一表示成一个参数
voxel_channel = 64
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
# 多头注意力每个头的向量长度
head_vector_len = 64
# dataset settings
# 测试小数据集
# data_root = 'data/kitti_tiny_3D/'
# 正经数据集
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
backend_args = None

voxel_size = [0.16, 0.16, 4]

model = dict(
    # 在空间中切格子都用VoxelNet
    type='BAVoxelNet',
    # 数据预处理
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        # 其中,体素化的步骤是在voxel_layer这里进行的
        voxel_layer=dict(
            # 不够就补0
            max_num_points=32,  # max_points_per_voxel
            # 取点云的范围
            # 分别为:Xmin Ymin Zmin Xmax Ymax Zmax
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            # 每个voxel的大小,这里设置了高为4,是因为pointpillars只分柱体
            # 而不细分为每个格子.因此,voxel的高设置为和取值相同
            voxel_size=voxel_size,
            # 分别代表train和test
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        # 输入维度为4,因为点云原始维度为四维
        in_channels=4,
        # 最终映射为64维的特征
        # feat_channels=[64],
        feat_channels=[voxel_channel],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    middle_encoder=dict(
        # 形成伪图像,特征维度为64,HW相当于496和432,496和432是由range除以0.16计算而来的
        # 到这一步,3D点云数据就变成了一个BEV视角的2D特征图
        type='PointPillarsScatter', in_channels=voxel_channel, output_shape=[496, 432]),
    # 添加图像分支
    # img_backbone=dict(
    #     type='mmdet.ResNet',
    #     depth=50,
    #     # 总共四个阶段,取后三个
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     # out_indices=(3,),
    #     # -1代表冻结所有参数,1代表只冻结第一层
    #     # 3代表冻结前三层只有最后一层更新
    #     frozen_stages=3,
    #     # frozen_stages=-1,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     norm_eval=True,
    #     style='caffe'),
    
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
        out_channels=feature_channel,
        num_csp_blocks=1),
    
    # 不带img的backbone默认为Pts Backbone等等
    backbone=dict(
        type='SECOND',
        # 至此相当于有一个64通道的特征图,因此yinchannel=64
        in_channels=voxel_channel,
        # 3个layer,每个layer为33卷积55卷积55卷积
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        # 每个layer stage的输出通道定义为64 128 256
        out_channels=[64, 128, 256]),
    
    neck=dict(
        type='SECONDFPN',
        # 在上述过程中得到的3个尺度的特征图输入到NECK
        in_channels=[64, 128, 256],
        # 上采样到(扩充HW)相同大小,
        # backbone中的下采样的第二特征图和第三特征图分别为2和2×2
        # 因此上采样为1,2,4
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),



    # # 位置编码层,用于为图像输入交叉注意力提供位置编码
    # position_embedding = dict(
    #     # 默认使用正余弦编码,无需梯度
    #     type = "PositionEmbeddingSine",
    #     # num_pos_feats必须是img_neck的out channel和 hiddensize的1/2
    #     num_pos_feats= int(feature_channel/2),
    #     # 默认值
    #     temperature=10000, 
    #     # sine情况下,normalize为True,可学习编码为False
    #     normalize=True,
    #     # scale为None时会自动计算
    #     scale=None,
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
        # 不再用q和v加和,
        # 而是将add&norm替换成输出特征与原始query(2D)concat来代替
        add = False,
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
        add = False,
    ),  
    
    # ip-feature聚合层,用于压缩通道与融合
    IP_compress_layer = dict(
        type = "MyConv2D",
        # 因为concat因此通道数×2
         Cin = feature_channel*2,
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
         kernel_size=1, 
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
        # PointPillar的head是基于Anchor的
        type='Anchor3DHead',
        num_classes=3,
        # input为neck的三个128concat,因此为384
        in_channels=384,
        feat_channels=384,
        # 方向分类器.PointPIllar和Second有,而早期的PointNet和Voxel没有
        use_direction_classifier=True,
        # label分配
        assign_per_class=True,
        # 生成anchor
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -1.78, 69.12, 39.68, -1.78],
            ],
            # 三个大小对应了人车和自行车的经典大小
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            # 两种转角0与90
            rotations=[0, 1.57],
            reshape_out=False),
        # 是否使用角度差的正弦值作为角度损失
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        # 分类loss
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
    
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
    #    角度分类Loss
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        # model中的assign_per_class对应到这里
        # 三种类别分别匹配
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        # nms thr小的原因是,3D物体几乎不会重叠
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        # 只取置信度最高的前100个框进行NMS
        nms_pre=100,
        # 最多保留50个
        max_num=50))


# 别忘了修改加载图像
# PointPillars adopted a different sampling strategies among classes
# 采样需要同时采样点云和图像
# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
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
#     # 采样图像,检查db_sampler后发现没有image loader
#     # image_loader = dict(type='LoadImageFromFile', backend_args=None),
#     backend_args=backend_args)
# AAV2采样
db_sampler = dict(
    type='MMDataBaseSamplerV2',
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    blending_type=None,
    depth_consistent=True,
    check_2D_collision=True,
    collision_thr=[0, 0.3, 0.5, 0.7],
    # collision_in_classes=True,
    mixup=0.7,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    # 加载图片
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True,
                                    with_label_3d=True,
                                    with_bbox=True,
                                    with_label=True),
    # 一定要先采样,不然经过别的变换就乱了
    dict(type='ObjectSampleV2', db_sampler=db_sampler, sample_2d=True),
    # sample 2d 参数 改为True
    # dict(type='ObjectSample', db_sampler=db_sampler,sample_2d = True ,use_ground_plane=False),
    # 3D随机翻转,同时增强图像和对应点云,保留对应关系
    # 只需要设置flip_ratio_bev_horizontal这一参数,因为图像会默认与点云同步翻转
    # 图像的点云翻转对应点云的水平翻转
    # resize一下不然采样完了总是不匹配
    dict(type='Resize3D',scale=(1280,384)),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # 采用mvx的方法设置
    dict(
        type='GlobalRotScaleTrans',
        # -π/4~π/4
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
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
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1280, 384),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='GlobalRotScaleTrans',
    #             rot_range=[0, 0],
    #             scale_ratio_range=[1., 1.],
    #             translation_std=[0, 0, 0]),
    #         dict(type='RandomFlip3D'),
    #         dict(
    #             type='PointsRangeFilter', point_cloud_range=point_cloud_range)
    #     ]),
    # 打包图像和点云
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

train_dataloader = dict(
    batch_size=12,
    num_workers=10,
    dataset=dict(dataset=dict(pipeline=train_pipeline, metainfo=metainfo)))
test_dataloader = dict(    
    batch_size=1,
    num_workers=1,
    dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_dataloader = dict(
    batch_size=48,
    num_workers=10,
    dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
epoch_num = 100
auto_scale_lr = dict(enable=False, base_batch_size=1) # 自动缩放学习率的设置，用来根据基础批量大小来自动调整学习率。
optim_wrapper = dict(
    # norm相当于罚,当grad的l2范数超过10的时候会被限制
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=5, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True)
]
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()



# pretrained = None
load_from = None
resume = False