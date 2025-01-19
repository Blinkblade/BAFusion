# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy
from os import path as osp
from pathlib import Path
from typing import Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from mmdet3d.structures.det3d_data_sample import SampleList


# 解析kitti calib txt文件
def parse_calibration_file(calib_file_path):
    """
    解析KITTI风格的标定文件，返回包含变换矩阵的字典，安全地处理空行和意外格式。
    
    参数:
    - calib_file_path (str): 标定文件的路径。

    返回值:
    - dict: 包含变换矩阵的numpy数组的字典。
    """
    calib_matrices = {}
    with open(calib_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            parts = line.split(':')
            if len(parts) < 2:
                continue  # 跳过不包含键值对的行
            key = parts[0].strip()  # 提取键名
            matrix_values = np.array([float(x) for x in parts[1].strip().split()])
            if 'P' in key:
                # 投影矩阵应该重新整形为3x4
                if len(matrix_values) == 12:  # 确保矩阵正好有12个元素
                    calib_matrices[key] = matrix_values.reshape(3, 4)
            elif 'R0_rect' in key or 'Tr_velo_to_cam' in key or 'Tr_imu_to_velo' in key:
                if len(matrix_values) == 9:
                    calib_matrices[key] = matrix_values.reshape(3, 3)
                elif len(matrix_values) == 12:
                    calib_matrices[key] = matrix_values.reshape(3, 4)

    return calib_matrices


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               palette: str = 'none',
               cfg_options: Optional[dict] = None):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    convert_SyncBN(config.model)
    config.model.train_cfg = None
    init_default_scope(config.get('default_scope', 'mmdet3d'))
    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmdet3d 1.x
            model.dataset_meta = checkpoint['meta']['dataset_meta']
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmdet3d 1.x
            classes = checkpoint['meta']['CLASSES']
            model.dataset_meta = {'classes': classes}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
        else:
            # < mmdet3d 1.x
            model.dataset_meta = {'classes': config.class_names}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']

        test_dataset_cfg = deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
    if device != 'cpu':
        torch.cuda.set_device(device)
    else:
        warnings.warn('Don\'t suggest using CPU device. '
                      'Some functions are not supported for now.')

    model.to(device)
    model.eval()
    return model


PointsType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]
ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def inference_detector(model: nn.Module,
                       pcds: PointsType) -> Union[Det3DDataSample, SampleList]:
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcds (str, ndarray, Sequence[str/ndarray]):
            Either point cloud files or loaded point cloud.

    Returns:
        :obj:`Det3DDataSample` or list[:obj:`Det3DDataSample`]:
        If pcds is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(pcds, (list, tuple)):
        is_batch = True
    else:
        pcds = [pcds]
        is_batch = False

    cfg = model.cfg

    if not isinstance(pcds[0], str):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.test_dataloader.dataset.pipeline[0].type = 'LoadPointsFromDict'

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    data = []
    for pcd in pcds:
        # prepare data
        if isinstance(pcd, str):
            # load from point cloud file
            data_ = dict(
                lidar_points=dict(lidar_path=pcd),
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        else:
            # directly use loaded point cloud
            data_ = dict(
                points=pcd,
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        data_ = test_pipeline(data_)
        data.append(data_)

    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

    if not is_batch:
        return results[0], data[0]
    else:
        return results, data


def inference_multi_modality_detector(model: nn.Module,
                                      pcds: Union[str, Sequence[str]],
                                      imgs: Union[str, Sequence[str]],
                                      ann_file: Union[str, Sequence[str]],
                                      cam_type: str = 'CAM2'):
    """Inference point cloud with the multi-modality detector. Now we only
    support multi-modality detector for KITTI and SUNRGBD datasets since the
    multi-view image loading is not supported yet in this inference function.

    Args:
        model (nn.Module): The loaded detector.
        pcds (str, Sequence[str]):
            Either point cloud files or loaded point cloud.
        imgs (str, Sequence[str]):
           Either image files or loaded images.
        ann_file (str, Sequence[str]): Annotation files.
        cam_type (str): Image of Camera chose to infer. When detector only uses
            single-view image, we need to specify a camera view. For kitti
            dataset, it should be 'CAM2'. For sunrgbd, it should be 'CAM0'.
            When detector uses multi-view images, we should set it to 'all'.

    Returns:
        :obj:`Det3DDataSample` or list[:obj:`Det3DDataSample`]:
        If pcds is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(pcds, (list, tuple)):
        is_batch = True
        assert isinstance(imgs, (list, tuple))
        assert len(pcds) == len(imgs)
    else:
        pcds = [pcds]
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    data_list = mmengine.load(ann_file)['data_list']

    data = []
    for index, pcd in enumerate(pcds):
        # get data info containing calib
        data_info = data_list[index]
        img = imgs[index]

        if cam_type != 'all':
            assert osp.isfile(img), f'{img} must be a file.'
            img_path = data_info['images'][cam_type]['img_path']
            if osp.basename(img_path) != osp.basename(img):
                raise ValueError(
                    f'the info file of {img_path} is not provided.')
            data_ = dict(
                lidar_points=dict(lidar_path=pcd),
                img_path=img,
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
            data_info['images'][cam_type]['img_path'] = img
            if 'cam2img' in data_info['images'][cam_type]:
                # The data annotation in SRUNRGBD dataset does not contain
                # `cam2img`
                data_['cam2img'] = np.array(
                    data_info['images'][cam_type]['cam2img'])

            # LiDAR to image conversion for KITTI dataset
            if box_mode_3d == Box3DMode.LIDAR:
                if 'lidar2img' in data_info['images'][cam_type]:
                    data_['lidar2img'] = np.array(
                        data_info['images'][cam_type]['lidar2img'])
            # Depth to image conversion for SUNRGBD dataset
            elif box_mode_3d == Box3DMode.DEPTH:
                data_['depth2img'] = np.array(
                    data_info['images'][cam_type]['depth2img'])
        else:
            assert osp.isdir(img), f'{img} must be a file directory'
            for _, img_info in data_info['images'].items():
                img_info['img_path'] = osp.join(img, img_info['img_path'])
                assert osp.isfile(img_info['img_path']
                                  ), f'{img_info["img_path"]} does not exist.'
            data_ = dict(
                lidar_points=dict(lidar_path=pcd),
                images=data_info['images'],
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)

        if 'timestamp' in data_info:
            # Using multi-sweeps need `timestamp`
            data_['timestamp'] = data_info['timestamp']

        data_ = test_pipeline(data_)
        data.append(data_)

    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

    if not is_batch:
        return results[0], data[0]
    else:
        return results, data

def inference_multi_modality_detector2(model: nn.Module,
                                      pcds: Union[str, Sequence[str]],
                                      imgs: Union[str, Sequence[str]],
                                      ann_file: Union[str, Sequence[str]],
                                      cam_type: str = 'CAM2'):
    # 定义一个用于多模态检测的推理函数

    # 参数说明：
    # model: 加载的检测器模型
    # pcds: 点云文件或数据
    # imgs: 图像文件或数据
    # ann_file: 标注文件
    # cam_type: 指定相机视图，默认为'CAM2'

    # 判断pcds是否为列表或元组，决定是否批量处理
    if isinstance(pcds, (list, tuple)):
        is_batch = True  # 批量处理标志
        assert isinstance(imgs, (list, tuple))  # 确保imgs也为列表或元组
        assert len(pcds) == len(imgs)  # 确保pcds和imgs长度相同
    else:
        pcds = [pcds]  # 将单个pcd转换为列表
        imgs = [imgs]  # 将单个img转换为列表
        is_batch = False  # 非批量处理

    cfg = model.cfg  # 从模型中获取配置

    # 构建数据处理管道
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)  # 深拷贝测试数据加载管道
    test_pipeline = Compose(test_pipeline)  # 组合管道中的处理步骤

    # 获取3D框的类型和模式
    box_type_3d, box_mode_3d = get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    # 加载标注文件并获取数据列表
    # dict_keys(['sample_id', 'images', 'lidar_points', 'instances'])
    if ann_file.split(".")[-1] == "pkl":
        print("== load pkl info from {} ==".format(ann_file))
        data_list = mmengine.load(ann_file)['data_list']

    elif ann_file.split(".")[-1] == "txt":
        # 仅支持单batch推理
        assert not is_batch, "Calib txt file as ann only support one batch!"
        print("== load txt info from {} ==".format(ann_file))
        # 读取calib txt文件
        calibration_data = parse_calibration_file(ann_file)
        # print(calibration_data)
        P2 = calibration_data["P2"]
        R0_rect = calibration_data["R0_rect"]
        Tr_velo_to_cam = calibration_data["Tr_velo_to_cam"]
        # === 从calib txt文件中的内容计算出相关的投影矩阵
        # 构建 'cam2img'
        cam2img = P2

        # 扩展 Tr_velo_to_cam 为 4x4 矩阵
        Tr_velo_to_cam_extended = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))

        # 构建 'lidar2cam'
        # 在R0_rect和Tr_velo_to_cam之间插入单位矩阵来适应维度
        R0_rect_extended = np.eye(4)
        R0_rect_extended[:3, :3] = R0_rect
        lidar2cam = R0_rect_extended @ Tr_velo_to_cam_extended

        # 构建 'lidar2img'
        lidar2img = P2 @ lidar2cam

        # 构造返回数据结构
        calibration_data_norm = {
            'CAM2': {
                'img_path': None,
                'cam2img': cam2img.tolist(),
                'lidar2cam': lidar2cam.tolist(),
                'lidar2img': lidar2img.tolist()
            },
            'R0_rect': R0_rect_extended.tolist()
        }

        data_list = [calibration_data_norm]


    data = []  # 初始化数据列表
    # 单帧推理就一个
    for index, pcd in enumerate(pcds):
        data_info = data_list[index]  # 获取索引对应的数据信息
        img = imgs[index]  # 获取索引对应的图像

        # 处理单个或多视图的图像路径验证和数据组装
        if cam_type != 'all':
            # 使用pkl文件时走原来的路径
            if ann_file.split(".")[-1] == "pkl":
                assert osp.isfile(img), f'{img} must be a file.'  # 确认图像文件存在
                img_path = data_info['images'][cam_type]['img_path']  # 获取标注的图像路径
                if osp.basename(img_path) != osp.basename(img):  # 确保文件名匹配
                    raise ValueError(f'the info file of {img_path} is not provided.')
                data_ = dict(
                    lidar_points=dict(lidar_path=pcd),  # 点云路径
                    img_path=img,  # 图像路径
                    box_type_3d=box_type_3d,  # 3D框类型
                    box_mode_3d=box_mode_3d)  # 3D框模式
                data_info['images'][cam_type]['img_path'] = img  # 更新数据信息中的图像路径
                if 'cam2img' in data_info['images'][cam_type]:  # 如果存在相机到图像的变换
                    data_['cam2img'] = np.array(data_info['images'][cam_type]['cam2img'])
            
            elif ann_file.split(".")[-1] == "txt":
                # 使用calib txt
                # print("== load txt info from {} ==".format(ann_file))
                # 构建data_字典
                data_ = dict(
                    lidar_points=dict(lidar_path=pcd),  # 点云路径
                    img_path=img,  # 图像路径
                    box_type_3d=box_type_3d,  # 3D框类型
                    box_mode_3d=box_mode_3d)  # 3D框模式
                
                data_info[cam_type]['img_path'] = img
                if 'cam2img' in data_info[cam_type]:  # 如果存在相机到图像的变换
                    data_['cam2img'] = np.array(data_info[cam_type]['cam2img'])
                pass
            
            else:
                # 抛出异常
                raise TypeError(f'the info file of Must be TXT or PKL but got type {ann_file.split(".")[-1]}')
        # 环视路径, 我们不会进入这个路径不用管
        else:
            assert osp.isdir(img), f'{img} must be a file directory'  # 确认图像目录存在
            for _, img_info in data_info['images'].items():
                img_info['img_path'] = osp.join(img, img_info['img_path'])  # 更新图像路径
                assert osp.isfile(img_info['img_path']), f'{img_info["img_path"]} does not exist.'  # 确认图像文件存在
            data_ = dict(
                lidar_points=dict(lidar_path=pcd),  # 点云路径
                images=data_info['images'],  # 图像信息
                box_type_3d=box_type_3d,  # 3D框类型
                box_mode_3d=box_mode_3d)  # 3D框模式

        if 'timestamp' in data_info:  # 如果数据中包含时间戳
            data_['timestamp'] = data_info['timestamp']  # 添加时间戳

        data_ = test_pipeline(data_)  # 对数据进行预处理
        data.append(data_)  # 添加到数据列表

    collate_data = pseudo_collate(data)  # 整合数据
    print("=============== collate_data KEYS: {} (mmdet3d/api/inference.py) ===============".format(collate_data.keys()))
    # 模型推理
    with torch.no_grad():  # 不计算梯度
        results = model.test_step(collate_data)  # 进行推理

    # 根据是否批量处理返回不同的结果
    if not is_batch:
        return results[0], data[0]  # 返回单个结果
    else:
        return results, data  # 返回批量结果







def inference_mono_3d_detector(model: nn.Module,
                               imgs: ImagesType,
                               ann_file: Union[str, Sequence[str]],
                               cam_type: str = 'CAM_FRONT'):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, Sequence[str]):
           Either image files or loaded images.
        ann_files (str, Sequence[str]): Annotation files.
        cam_type (str): Image of Camera chose to infer.
            For kitti dataset, it should be 'CAM_2',
            and for nuscenes dataset, it should be
            'CAM_FRONT'. Defaults to 'CAM_FRONT'.

    Returns:
        :obj:`Det3DDataSample` or list[:obj:`Det3DDataSample`]:
        If pcds is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    data_list = mmengine.load(ann_file)['data_list']
    assert len(imgs) == len(data_list)

    data = []
    for index, img in enumerate(imgs):
        # get data info containing calib
        data_info = data_list[index]
        img_path = data_info['images'][cam_type]['img_path']
        if osp.basename(img_path) != osp.basename(img):
            raise ValueError(f'the info file of {img_path} is not provided.')

        # replace the img_path in data_info with img
        data_info['images'][cam_type]['img_path'] = img
        # avoid data_info['images'] has multiple keys anout camera views.
        mono_img_info = {f'{cam_type}': data_info['images'][cam_type]}
        data_ = dict(
            images=mono_img_info,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)

        data_ = test_pipeline(data_)
        data.append(data_)

    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

    if not is_batch:
        return results[0]
    else:
        return results


def inference_segmentor(model: nn.Module, pcds: PointsType):
    """Inference point cloud with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        pcds (str, Sequence[str]):
            Either point cloud files or loaded point cloud.

    Returns:
        :obj:`Det3DDataSample` or list[:obj:`Det3DDataSample`]:
        If pcds is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(pcds, (list, tuple)):
        is_batch = True
    else:
        pcds = [pcds]
        is_batch = False

    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)

    new_test_pipeline = []
    for pipeline in test_pipeline:
        if pipeline['type'] != 'LoadAnnotations3D' and pipeline[
                'type'] != 'PointSegClassMapping':
            new_test_pipeline.append(pipeline)
    test_pipeline = Compose(new_test_pipeline)

    data = []
    # TODO: support load points array
    for pcd in pcds:
        data_ = dict(lidar_points=dict(lidar_path=pcd))
        data_ = test_pipeline(data_)
        data.append(data_)

    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

    if not is_batch:
        return results[0], data[0]
    else:
        return results, data
