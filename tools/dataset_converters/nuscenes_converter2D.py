# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import mmengine
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.datasets.convert_utils import NuScenesNameMapping
from mmdet3d.structures import points_cam2img
import cv2
# 更改自aav2, 来获取2D BBOX

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

# bbox 调色盘
color_list = [[0, 153, 51], [0, 153, 255], [255, 255, 0], [204, 102, 255],\
              [255, 0, 0], [204, 0, 255], [102, 255, 255], [255, 102, 102]]
ratio = 1
# 绘制目标框用于测试
def draw_bbox(img, bbox_score_data, index=0, show_score=False):
    global color_list
    index %= 5
    test = list(map(lambda x: int(x), bbox_score_data[:4]))
    img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), color_list[index], 2)
    if show_score:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ""
        if show_score:
            text += str(bbox_score_data[4])[:4]
        rec_width = len(text) * 6
        img = cv2.rectangle(img, (test[0], test[1]), (test[0]+rec_width, test[1]+10), color_list[index], -1)
        cv2.putText(img, text, (test[0], test[1]+8), font, 0.35, (0, 0, 0), 1)
    return img




def draw_bbox(img, bbox_score_data, index=0, show_score=False):
    global color_list
    index %= 5
    test = list(map(lambda x: int(x), bbox_score_data[:4]))
    img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), color_list[index], 2)
    if show_score:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ""
        if show_score:
            text += str(bbox_score_data[4])[:4]
        rec_width = len(text) * 6
        img = cv2.rectangle(img, (test[0], test[1]), (test[0]+rec_width, test[1]+10), color_list[index], -1)
        cv2.putText(img, text, (test[0], test[1]+8), font, 0.35, (0, 0, 0), 1)
    return img





# 整合标签Info的函数
def create_nuscenes_infos2D(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset including 2d ann.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    print("===================执行create_nuscenes_infos2D in ./tools/dataset_converters/nuscenes_converter2D.py ===========================")

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
        
    # 填充train_val的infos 
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        # metadata就是version
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmengine.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmengine.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'num_features': 5,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix



        # 这里开始读取图像的相关info 
        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, box_list, cam_intrinsic = nusc.get_sample_data(cam_token)

            # 测试一下读取的图像是否正确
            # tmp_img = cv2.imread(cam_path)
            # cv2.imwrite('nus_code/fig/%s' % cam_path.split('/')[-1], tmp_img)
            # 测试结束

            # # 测试一下读取的图像是否正确-mini
            # tmp_img = cv2.imread(cam_path)
            # cv2.imwrite('nus_code/mini/fig/%s' % cam_path.split('/')[-1], tmp_img)
            # # 测试结束

            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesNameMapping:
                    names[i] = NuScenesNameMapping[names[i]]
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            
            # print("\n=================读入2d gt_boxes : {}\n".format(gt_boxes[0]))
            # 这里的gt_boxes实际上指的是七个维度的3d bbox
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

            if 'lidarseg' in nusc.table_names:
                info['pts_semantic_mask_path'] = osp.join(
                    nusc.dataroot,
                    nusc.get('lidarseg', lidar_token)['filename'])
                pass

            # 这里添加一对一生成2d bbox
            # NOTE generate projected one-to-one 2d box
            
            # 定义一个空列表，用来存储 2D 边界框的坐标和对应的相机索引
            gt_boxes_2d = []
            # 定义一个空列表，用来存储所有的 2D 边界框的坐标和对应的相机索引
            all_2d_boxes = []
            # 遍历 annotations，它是一个包含了 3D 边界框的标注信息的列表
            for anno in annotations:
                # 使用 nusc.get_box 函数，根据 anno 的 token 属性，得到一个 Box 对象，它是一个表示 3D 边界框的类
                box = nusc.get_box(anno['token'])
                # 定义一个布尔变量，用来标记是否已经生成了对应的 2D 边界框
                # proj表示投影, 即从3D向2D投影来生成2D框
                has_proj = False
                # 遍历 camera_types，它是一个包含了不同相机类型的列表，例如 'CAM_FRONT'，'CAM_BACK' 等
                # index代表相机ID
                for index, cam in enumerate(camera_types):
                    # 复制 box 对象，得到一个临时的 Box 对象
                    tmp_box = box.copy()
                    # 从 info 中获取相应的相机信息，它是一个字典，包含了相机的内参和外参等数据
                    cam_info = info['cams'][cam]
                    # 使用 translate 方法，把临时的 Box 对象沿着负的 ego2global_translation 方向平移，这是一个从 ego 坐标系到全局坐标系的平移向量
                    tmp_box.translate(-np.array(cam_info['ego2global_translation']))
                    # 使用 rotate 方法，把临时的 Box 对象旋转 Quaternion(cam_info['ego2global_rotation']).inverse 角度，这是一个从 ego 坐标系到全局坐标系的旋转四元数的逆
                    tmp_box.rotate(Quaternion(cam_info['ego2global_rotation']).inverse)
                    # 把临时的 Box 对象移动到校准后的传感器坐标系
                    # 使用 translate 方法，把临时的 Box 对象沿着负的 sensor2ego_translation 方向平移，这是一个从传感器坐标系到 ego 坐标系的平移向量
                    tmp_box.translate(-np.array(cam_info['sensor2ego_translation']))
                    # 使用 rotate 方法，把临时的 Box 对象旋转 Quaternion(cam_info['sensor2ego_rotation']).inverse 角度，这是一个从传感器坐标系到 ego 坐标系的旋转四元数的逆
                    tmp_box.rotate(Quaternion(cam_info['sensor2ego_rotation']).inverse)

                    # Filter out the corners that are not in front of the calibrated sensor.
                    # 过滤掉那些不在校准后的传感器前方的角点
                    # 使用 corners 方法，得到临时的 Box 对象的 3D 角点坐标，它是一个 3 x 8 的数组
                    corners_3d = tmp_box.corners()
                    # 在数组的第三行中，找到那些大于 0 的元素的索引，它们表示在传感器前方的角点
                    in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                    # 从数组中只保留这些索引对应的角点坐标
                    corners_3d = corners_3d[:, in_front]

                    # 把 3D 边界框投影到 2D
                    # 使用 view_points 函数，根据传感器的内参矩阵，把 3D 角点坐标转换成 2D 像素坐标，返回一个 3 x N 的数组，第三行是深度值
                    # 使用 T 属性，转置这个数组，得到一个 N x 3 的数组
                    # 使用切片操作，只保留前两列，得到一个 N x 2 的数组，每一行是一个 2D 角点坐标
                    # 使用 tolist 方法，把数组转换成一个列表
                    corner_coords = view_points(corners_3d, cam_info['cam_intrinsic'], True).T[:, :2].tolist()
                    # 只保留那些落在图像范围内的角点
                    # 使用 post_process_coords 函数，得到重投影的角点的凸包和图像画布的交集，
                    # 如果没有交集，返回 None
                    # 这是为了判断该物体的投影是否可以出现在对应相机上
                    final_coords = post_process_coords(corner_coords)
                    # 如果交集为空，表示重投影的边界框在图像上不可见，跳过这个相机
                    if final_coords is None:
                        continue
                    # 否则，得到交集的左上角和右下角的坐标，即 (min_x, min_y, max_x, max_y)
                    else:
                        min_x, min_y, max_x, max_y = final_coords
                        # 如果还没有生成对应的 2D 边界框，把这个坐标和相机索引加入到 gt_boxes_2d 列表中，表示这是一个有效的 2D 边界框
                        if has_proj == False:
                            # gt_boxes_2d的属性是xyxy index,index是相机id
                            gt_boxes_2d.append([min_x, min_y, max_x, max_y, index])
                            # 把 has_proj 设为 True，表示已经生成了对应的 2D 边界框
                            has_proj = True
                        # 下面的代码是用来可视化 2D 边界框的，被注释掉了
                        # 测试生成的2D框是否正确, 在图中绘制bbox
                        # img = cv2.imread('nus_code/fig/%s' % cam_info['data_path'].split('/')[-1])
                        # img = draw_bbox(img, final_coords)
                        # cv2.imwrite('nus_code/fig/%s' % cam_info['data_path'].split('/')[-1], img)
                        # 这一段在使用mini时启用
                        # img = cv2.imread('nus_code/mini/fig/%s' % cam_info['data_path'].split('/')[-1])
                        # img = draw_bbox(img, final_coords)
                        # cv2.imwrite('nus_code/mini/fig/%s' % cam_info['data_path'].split('/')[-1], img)
                        # 测试结束
                            
                        # 把这个坐标和相机索引加入到 all_2d_boxes 列表中，表示这是一个可能的 2D 边界框
                        all_2d_boxes.append([min_x, min_y, max_x, max_y, index])
                # 如果遍历完所有的相机，都没有生成对应的 2D 边界框，表示这个 3D 边界框在所有的图像上都不可见
                if has_proj == False:
                    # 把一个无效的 2D 边界框坐标和相机索引加入到 gt_boxes_2d 列表中，表示这是一个无效的 2D 边界框
                    gt_boxes_2d.append([0, 0, 1, 1, 5])
            # 把 gt_boxes_2d 列表转换成一个数组，赋值给 info 的 gt_boxes_2d 属性，
            # 表示每个 3D 边界框对应的有效的 2D 边界框
            # print("\n======= there is a gt_boxes_2d : {} ========".format(gt_boxes_2d.shape))
            info['gt_boxes_2d'] = np.array(gt_boxes_2d)
            # 把 all_2d_boxes 列表转换成一个数组，赋值给 info 的 all_2d_boxes 属性，
            # 表示每个 3D 边界框对应的所有的 2D 边界框
            info['all_2d_boxes'] = np.array(all_2d_boxes)
            # 使用 assert 语句，检查 info 的 gt_boxes_2d 属性的形状的第一个维度是否等于 gt_boxes 的长度，
            # 它们都应该是 3D 边界框的数量, 即, 为每一个3d object生成他们在环视相机中的对应2D BBOX
            assert info['gt_boxes_2d'].shape[0] == len(gt_boxes), (len(gt_boxes), info['gt_boxes_2d'])
            # 2d bbox 到此生成结束 



        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmengine.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmengine.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmengine.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesNameMapping:
        return None
    cat_name = NuScenesNameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
