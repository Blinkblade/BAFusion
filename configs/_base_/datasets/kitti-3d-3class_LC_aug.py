# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(classes=class_names)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/kitti/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

# db_sampler = dict(
#     data_root=data_root,
#     # 生成的pkl标注
#     info_path=data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
#     classes=class_names,
#     sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     backend_args=backend_args)

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
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # x, y, z, intensity
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
    # dict(
    #     type='ObjectNoise',
    #     num_try=100,
    #     translation_std=[1.0, 1.0, 0.5],
    #     global_rot_range=[0.0, 0.0],
    #     rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
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
    #     img_scale=(1333, 800),
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
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced', img='training/image_2'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
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
        data_prefix=dict(pts='training/velodyne_reduced', img='training/image_2'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='testing/velodyne_reduced', img='testing/image_2'),
        ann_file='kitti_infos_test.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_test.pkl',
    metric='bbox',
    backend_args=backend_args)


# 设置tensorboard
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
