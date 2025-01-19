# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import List, Optional

import mmengine
import numpy as np
from mmengine.fileio import get_local_path

from mmdet3d.datasets.transforms import data_augment_utils
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.ops import box_np_ops
import cv2


class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str, optional): The category of samples. Defaults to None.
        epoch (int, optional): Sampling epoch. Defaults to None.
        shuffle (bool): Whether to shuffle indices. Defaults to False.
        drop_reminder (bool): Drop reminder. Defaults to False.
    """

    def __init__(self,
                 sampled_list: List[dict],
                 name: Optional[str] = None,
                 epoch: Optional[int] = None,
                 shuffle: bool = True,
                 drop_reminder: bool = False) -> None:
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num: int) -> List[int]:
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self) -> None:
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num: int) -> List[dict]:
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@TRANSFORMS.register_module()
class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str], optional): List of classes. Defaults to None.
        points_loader (dict): Config of points loader. Defaults to
            dict(type='LoadPointsFromFile', load_dim=4, use_dim=[0, 1, 2, 3]).
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 info_path: str,
                 data_root: str,
                 rate: float,
                 prepare: dict,
                 sample_groups: dict,
                 classes: Optional[List[str]] = None,
                 points_loader: dict = dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3],
                     backend_args=None),
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = TRANSFORMS.build(points_loader)
        self.backend_args = backend_args

        # load data base infos
        with get_local_path(
                info_path, backend_args=self.backend_args) as local_path:
            # loading data from a file-like object needs file format
            db_infos = mmengine.load(open(local_path, 'rb'), file_format='pkl')

        # filter database infos
        from mmengine.logging import MMLogger
        logger: MMLogger = MMLogger.get_current_instance()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos in DataBaseSampler')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos in DataBaseSampler')

        self.db_infos = db_infos

        # =======sampler db_infos keys:dict_keys(['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Tram', 'Misc', 'Person_sitting'])========
        print("=======sampler db_infos keys:{}========".format(db_infos.keys()))
        # 对应装的是car的采样数据
        # print("=======sampler db_infos['Car']:{}========".format(db_infos['Car']))
        # =======sampler type of db_infos['Car']:<class 'list'>========
        # print("=======sampler type of db_infos['Car']:{}========".format(type(db_infos['Car'])))
        # =======sampler type of db_infos['Car'][0]:<class 'dict'>========  
        # print("=======sampler type of db_infos['Car'][0]:{}========".format(type(db_infos['Car'][0])))
        # dict_keys(['name', 'path', 'image_idx', 'gt_idx', 'box3d_lidar', 'num_points_in_gt', 'difficulty', 'group_id', 'score'])========
        # print("=======sampler db_infos['Car'][0] keys:{}========".format(db_infos['Car'][0].keys()))
        # print("=======sampler db_infos['car'][0] keys:{}========".format(db_infos['car'][0].keys()))
        # print("=======sampler db_infos['Car'][0] :{}========".format(db_infos['Car'][0]))

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos: dict, removed_difficulty: list) -> dict:
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos: dict, min_gt_points_dict: dict) -> dict:
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self,
                   gt_bboxes: np.ndarray,
                   gt_labels: np.ndarray,
                   img: Optional[np.ndarray] = None,
                   ground_plane: Optional[np.ndarray] = None) -> dict:
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.
            img (np.ndarray, optional): Image array. Defaults to None.
            ground_plane (np.ndarray, optional): Ground plane information.
                Defaults to None.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels
                  of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`):
                  sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list = []
            count = 0
            for info in sampled:
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                results = dict(lidar_points=dict(lidar_path=file_path))
                s_points = self.points_loader(results)['points']
                s_points.translate(info['box3d_lidar'][:3])

                count += 1

                s_points_list.append(s_points)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.int64)

            if ground_plane is not None:
                xyz = sampled_gt_bboxes[:, :3]
                dz = (ground_plane[:3][None, :] *
                      xyz).sum(-1) + ground_plane[3]
                sampled_gt_bboxes[:, 2] -= dz
                for i, s_points in enumerate(s_points_list):
                    s_points.tensor[:, 2].sub_(dz[i])

            ret = {
                'gt_labels_3d':
                gt_labels,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                s_points_list[0].cat(s_points_list),
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled))
            }

        return ret

    def sample_class_v2(self, name: str, num: int,
                        gt_bboxes: np.ndarray) -> List[dict]:
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples



@TRANSFORMS.register_module()
class MMDataBaseSampler(DataBaseSampler):

    def __init__(self,
                 info_path: str,
                 data_root: str,
                 rate: float,
                 prepare: dict,
                 sample_groups: dict,
                 classes: Optional[List[str]] = None,
                 check_2D_collision=False,
                 collision_thr=0,
                 collision_in_classes=False,
                 depth_consistent=True,
                 blending_type=None,
                 img_loader=dict(type='LoadImageFromFile',backend_args=None),
                 mask_loader=dict(
                     type='LoadImageFromFile', color_type='grayscale',backend_args=None),
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     load_dim=4,
                     coord_type='LIDAR',
                     use_dim=[0, 1, 2, 3],
                     backend_args=None),
                 backend_args: Optional[dict] = None
                 ):
        super(MMDataBaseSampler, self).__init__(
            info_path=info_path,
            data_root=data_root,
            rate=rate,
            prepare=prepare,
            sample_groups=sample_groups,
            classes=classes,
            points_loader=points_loader,
            backend_args = backend_args)
        
        self.blending_type = blending_type
        self.depth_consistent = depth_consistent
        self.check_2D_collision = check_2D_collision
        self.collision_thr = collision_thr
        self.collision_in_classes = collision_in_classes
        # self.img_loader = mmcv.build_from_cfg(img_loader, PIPELINES)
        # self.mask_loader = mmcv.build_from_cfg(mask_loader, PIPELINES)
        # build处会不同
        self.img_loader = TRANSFORMS.build(img_loader)
        self.mask_loader = TRANSFORMS.build(mask_loader)
        print("===========BUILD MMDataBaseSampler =============")


    def sample_all(self, gt_bboxes_3d, gt_names, gt_bboxes_2d=None, img=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        # class_name:种类;max_sample_num:对应种类的采样数目
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            # 生成采样数目录入对应种类的sampled_num_dict
            sampled_num_dict[class_name] = sampled_num
            # 每一类采样多少
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes_3d = []
        sampled_gt_bboxes_2d = []
        avoid_coll_boxes_3d = gt_bboxes_3d
        avoid_coll_boxes_2d = gt_bboxes_2d
        # class_name = 种类名称,如"car";sampled_num = 对应种类的采样数量,在上面计算了
        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            # 如果采样数量多大于0就进行sample_class_v2
            if sampled_num > 0:
                # 输入:采样类别名称,采样数量,2d 3d两个bbox
                # print("============sampled_num:{}===============".format(sampled_num))
                # print("============avoid_coll_boxes_2d:{}===============".format(avoid_coll_boxes_2d))
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes_3d,
                                                   avoid_coll_boxes_2d)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box_3d = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                        sampled_gt_box_2d = sampled_cls[0]['box2d_camera'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box_3d = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)
                        sampled_gt_box_2d = np.stack(
                            [s['box2d_camera'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes_3d += [sampled_gt_box_3d]
                    sampled_gt_bboxes_2d += [sampled_gt_box_2d]
                    if self.collision_in_classes:
                        # TODO: check whether check collision check among
                        # classes is necessary
                        avoid_coll_boxes_3d = np.concatenate(
                            [avoid_coll_boxes_3d, sampled_gt_box_3d], axis=0)
                        avoid_coll_boxes_2d = np.concatenate(
                            [avoid_coll_boxes_2d, sampled_gt_box_2d], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes_3d = np.concatenate(sampled_gt_bboxes_3d, axis=0)
            sampled_gt_bboxes_2d = np.concatenate(sampled_gt_bboxes_2d, axis=0)

            s_points_list = []
            count = 0
            # 按深度排序,因为要先贴后面的避免后面的遮挡前面
            if self.depth_consistent:
                # change the paster order based on distance
                center = sampled_gt_bboxes_3d[:, 0:3]
                # 取得粘贴顺序
                paste_order = np.argsort(
                    -np.power(np.sum(np.power(center, 2), axis=-1), 1 / 2),
                    axis=-1)

            for idx in range(len(sampled)):
                # 由于有深度排序,不能直接取得info,按照inds的顺序取
                if self.depth_consistent:
                    inds = np.where(paste_order == idx)[0][0]
                    info = sampled[inds]
                else:
                    info = sampled[idx]
                # 原始的file_path这里命名为pcd_file_path
                pcd_file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                # print("==============pcd_file_path : {}===============".format(pcd_file_path))
                # 其他file_path
                img_file_path = pcd_file_path + '.png'
                mask_file_path = pcd_file_path + '.mask.png'
                # 新版result字典key换了
                # results = dict(pts_filename=pcd_file_path)
                results = dict(lidar_points=dict(lidar_path=pcd_file_path))
                s_points = self.points_loader(results)['points']
                # 之后的translate在获取result之后
                # perform points sampling before pasting

                patch_results = dict(
                    img_prefix=None, img_info=dict(filename=img_file_path))
                mask_results = dict(
                    img_prefix=None, img_info=dict(filename=mask_file_path))
                s_patch = self.img_loader(patch_results)['img']
                s_mask = self.mask_loader(mask_results)['img']

                # the points of each sample already minus the object center
                # so this time it needs to add the offset back
                s_points.translate(info['box3d_lidar'][:3])
                img = self.paste_obj(
                    img,
                    s_patch,
                    s_mask,
                    bbox_2d=info['box2d_camera'].astype(np.int32))

                count += 1
                s_points_list.append(s_points)
            # dtype设置为long,默认无dtype
            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.int64)
            ret = dict(
                img=img,
                gt_labels=gt_labels,
                gt_labels_3d=copy.deepcopy(gt_labels),
                gt_bboxes_3d=sampled_gt_bboxes_3d,
                gt_bboxes_2d=sampled_gt_bboxes_2d,
                points=s_points_list[0].cat(s_points_list),
                group_ids=np.arange(gt_bboxes_3d.shape[0],
                                    gt_bboxes_3d.shape[0] + len(sampled)))

        return ret

    def paste_obj(self, img, obj_img, obj_mask, bbox_2d):
        # paste the image patch back
        x1, y1, x2, y2 = bbox_2d
        # the bbox might exceed the img size because the img is different
        img_h, img_w = img.shape[:2]
        w = np.maximum(min(x2, img_w - 1) - x1 + 1, 1)
        h = np.maximum(min(y2, img_h - 1) - y1 + 1, 1)
        obj_mask = obj_mask[:h, :w]
        obj_img = obj_img[:h, :w]

        # choose a blend option
        if not self.blending_type:
            blending_op = 'none'

        else:
            blending_choice = np.random.randint(len(self.blending_type))
            blending_op = self.blending_type[blending_choice]

        if blending_op.find('poisson') != -1:
            # options: cv2.NORMAL_CLONE=1, or cv2.MONOCHROME_TRANSFER=3
            # cv2.MIXED_CLONE mixed the texture, thus is not used.
            if blending_op == 'poisson':
                mode = np.random.choice([1, 3], 1)[0]
            elif blending_op == 'poisson_normal':
                mode = cv2.NORMAL_CLONE
            elif blending_op == 'poisson_transfer':
                mode = cv2.MONOCHROME_TRANSFER
            else:
                raise NotImplementedError
            center = (int(x1 + w / 2), int(y1 + h / 2))
            img = cv2.seamlessClone(obj_img, img, obj_mask * 255, center, mode)
        else:
            if blending_op == 'gaussian':
                obj_mask = cv2.GaussianBlur(
                    obj_mask.astype(np.float32), (5, 5), 2)
            elif blending_op == 'box':
                obj_mask = cv2.blur(obj_mask.astype(np.float32), (3, 3))
            paste_mask = 1 - obj_mask
            img[y1:y1 + h,
                x1:x1 + w] = (img[y1:y1 + h, x1:x1 + w].astype(np.float32) *
                              paste_mask[..., None]).astype(np.uint8)
            img[y1:y1 + h, x1:x1 + w] += (obj_img.astype(np.float32) *
                                          obj_mask[..., None]).astype(np.uint8)

        return img

    def sample_class_v2(self, name, num, gt_bboxes_3d, gt_bboxes_2d):

        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        print("==============num of sampled:{}==============".format(len(sampled)))
        print("==============sampled:{}==============".format(sampled))
        num_gt = gt_bboxes_3d.shape[0]
        num_sampled = len(sampled)
        # avoid collision in BEV first
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes_3d[:, 0:2], gt_bboxes_3d[:, 3:5], gt_bboxes_3d[:, 6])
        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes[:, 0:2], sp_boxes[:, 3:5], sp_boxes[:, 6])
        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)

        # Then avoid collision in 2D space
        if self.check_2D_collision:
            # 这里的sampled就是info_db['class'],sample class可能就是根据class读取数据进行采样
            # ======sampled[0].keys:dict_keys(['name', 'path', 'image_idx', 'gt_idx', 'box3d_lidar', 'num_points_in_gt', 'difficulty', 'group_id', 'score'])=======
            print("======sampled[0].keys:{}=======".format(sampled[0].keys()))
            sp_boxes_2d = np.stack([i['box2d_camera'] for i in sampled],
                                   axis=0)
            total_bbox_2d = np.concatenate([gt_bboxes_2d, sp_boxes_2d],
                                           axis=0)  # Nx4
            # random select a collision threshold
            if isinstance(self.collision_thr, float):
                collision_thr = self.collision_thr
            elif isinstance(self.collision_thr, list):
                collision_thr = np.random.choice(self.collision_thr)
            elif isinstance(self.collision_thr, dict):
                mode = self.collision_thr.get('mode', 'value')
                if mode == 'value':
                    collision_thr = np.random.choice(
                        self.collision_thr['thr_range'])
                elif mode == 'range':
                    collision_thr = np.random.uniform(
                        self.collision_thr['thr_range'][0],
                        self.collision_thr['thr_range'][1])

            if collision_thr == 0:
                # use similar collision test as BEV did
                # Nx4 (x1, y1, x2, y2) -> corners: Nx4x2
                # ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
                x1y1 = total_bbox_2d[:, :2]
                x2y2 = total_bbox_2d[:, 2:]
                x1y2 = np.stack([total_bbox_2d[:, 0], total_bbox_2d[:, 3]],
                                axis=-1)
                x2y1 = np.stack([total_bbox_2d[:, 2], total_bbox_2d[:, 1]],
                                axis=-1)
                total_2d = np.stack([x1y1, x2y1, x1y2, x2y2], axis=1)
                coll_mat_2d = data_augment_utils.box_collision_test(
                    total_2d, total_2d)
            else:
                # use iof rather than iou to protect the foreground
                overlaps = box_np_ops.iou_jit(total_bbox_2d, total_bbox_2d,
                                              'iof')
                coll_mat_2d = overlaps > collision_thr
            coll_mat = coll_mat + coll_mat_2d

        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])

        return valid_samples
    
    
    

    
    
@TRANSFORMS.register_module()
class MMDataBaseSamplerV2(DataBaseSampler):

    def __init__(self,
                 info_path: str,
                 data_root: str,
                 rate: float,
                 prepare: dict,
                 sample_groups: dict,
                 classes: Optional[List[str]] = None,
                 check_2D_collision=False,
                 collision_thr=0,
                 collision_in_classes=False,
                 depth_consistent=False,
                 blending_type=None,
                 mixup=1.0,
                 img_num=1,
                 img_loader=dict(type='LoadImageFromFile',backend_args=None),
                #  mask_loader=dict(
                #      type='LoadImageFromFile', color_type='grayscale'),
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     load_dim=4,
                     coord_type='LIDAR',
                     use_dim=[0, 1, 2, 3],
                     backend_args=None),
                 backend_args: Optional[dict] = None                 
                 ):
        super(MMDataBaseSamplerV2, self).__init__(
            info_path=info_path,
            data_root=data_root,
            rate=rate,
            prepare=prepare,
            sample_groups=sample_groups,
            classes=classes,
            points_loader=points_loader)
        
        self.blending_type = blending_type
        self.depth_consistent = depth_consistent
        self.check_2D_collision = check_2D_collision
        self.collision_thr = collision_thr
        self.collision_in_classes = collision_in_classes
        # 构建img-loader
        self.img_loader = TRANSFORMS.build(img_loader)
        self.mixup = mixup
        self.img_num = img_num
        # self.mask_loader = mmcv.build_from_cfg(mask_loader, PIPELINES)
        print("===========BUILD MMDataBaseSamplerV2 =============")
    def sample_all(self, gt_bboxes_3d, gt_names, gt_bboxes_2d=None, img=None, img_filename=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        # class_name:种类;max_sample_num:对应种类的采样数目
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            # 生成采样数目录入对应种类的sampled_num_dict
            sampled_num_dict[class_name] = sampled_num
            # 每一类采样多少
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes_3d = []
        sampled_gt_bboxes_2d = []
        avoid_coll_boxes_3d = gt_bboxes_3d
        avoid_coll_boxes_2d = gt_bboxes_2d
        # class_name = 种类名称,如"car";sampled_num = 对应种类的采样数量,在上面计算了
        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            # 如果采样数量多大于0就进行sample_class_v2
            if sampled_num > 0:
                # 输入:采样类别名称,采样数量,2d 3d两个bbox
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes_3d,
                                                   avoid_coll_boxes_2d)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box_3d = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                        sampled_gt_box_2d = sampled_cls[0]['box2d_camera'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box_3d = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)
                        sampled_gt_box_2d = np.stack(
                            [s['box2d_camera'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes_3d += [sampled_gt_box_3d]
                    sampled_gt_bboxes_2d += [sampled_gt_box_2d]
                    if self.collision_in_classes:
                        # TODO: check whether check collision check among
                        # classes is necessary
                        avoid_coll_boxes_3d = np.concatenate(
                            [avoid_coll_boxes_3d, sampled_gt_box_3d], axis=0)
                        avoid_coll_boxes_2d = np.concatenate(
                            [avoid_coll_boxes_2d, sampled_gt_box_2d], axis=0)

        ret = None
        # img shape : 375×1242×3,一张标准的三通道图,不存在什么camera id
        origin_img = img.copy()
        # print("==================shape of img : {}=================".format(img.shape))
        if len(sampled) > 0:
            sampled_gt_bboxes_3d = np.concatenate(sampled_gt_bboxes_3d, axis=0)
            sampled_gt_bboxes_2d = np.concatenate(sampled_gt_bboxes_2d, axis=0)

            # Get all objects out along with its label (if original or virtual)
            num_origin = gt_bboxes_2d.shape[0]
            origin_label = np.zeros((num_origin, 1))
            num_virtual = sampled_gt_bboxes_2d.shape[0]
            virtual_label = np.ones((num_virtual, 1))

            all_labels = np.concatenate([origin_label, virtual_label], axis=0)
            all_bboxes_3d = np.concatenate([gt_bboxes_3d, sampled_gt_bboxes_3d], axis=0)
            bboxes_2d = np.concatenate([gt_bboxes_2d, sampled_gt_bboxes_2d], axis=0)
            # kitti数据集不存在camera id这个东西,Bbox就是全部
            # print("======================bboxes_2d:{}========================".format(bboxes_2d))
            # all_camera_idx = bboxes_2d[:, -1]
            all_bboxes_2d = bboxes_2d[:, :4]

            num_obj = all_bboxes_3d.shape[0]
            # 调整2d粘贴顺序,先粘贴后面的再粘贴前面的
            imgs_paste_order = np.argsort(-all_bboxes_3d[:, 0])
            # first of all, we need to get all points
            point_list = []
            # 分开处理点云和图像,点云不用排序直接粘贴,按原方法来
            for idx in range(len(sampled)):
                info = sampled[idx]
                # 原版的file path 替换乘pcd_file_path
                pcd_file_path = os.path.join(self.data_root, info['path'])\
                    if self.data_root else info['path']
                # print("=============pcd_file_path : {}==============".format(pcd_file_path))
                # result字典的key更换了新名字
                # results = dict(pts_filename=pcd_file_path)
                results = dict(lidar_points=dict(lidar_path=pcd_file_path))
                s_points = self.points_loader(results)['points']
                s_points.translate(info['box3d_lidar'][:3])
                # s_points_list名称更换为point_list
                point_list.append(s_points)
            # 提前制作完concat点云,返回时直接调用而不再是返回的时候再拼接 
            sampled_points = point_list[0].cat(point_list)
            # 接下来粘贴图像,和点云不同,图像需要按照物体的深度粘贴
            # 因此调换顺序
            # paste images in a far to near manner
            for idx in range(num_obj):
                inds = np.where(imgs_paste_order == idx)[0][0]
                bbox_2d = all_bboxes_2d[inds]; bbox_3d = all_bboxes_3d[inds]
                                                 
                # # use only one image
                # if camera_idx >= self.img_num:
                #     continue
                # 按照粘贴顺序排序
                label = all_labels[inds]
                if label == 0:
                    x1, y1, x2, y2 = [int(ii) for ii in bbox_2d]
                    # 没有camera id
                    # img[camera_idx][x1:x2, y1:y2] = self.mixup * origin_img[camera_idx][x1:x2, y1:y2] + \
                    #         (1 - self.mixup) * img[camera_idx][x1:x2, y1:y2]
                    img[x1:x2, y1:y2] = self.mixup * origin_img[x1:x2, y1:y2] + \
                            (1 - self.mixup) * img[x1:x2, y1:y2]
                else:
                    inds -= num_origin
                    info = sampled[inds]
                    pcd_file_path = os.path.join(
                        self.data_root,
                        info['path']) if self.data_root else info['path']
                    img_file_path = pcd_file_path + '.png'
                    # print("=============img_file_path : {}==============".format(img_file_path))
                    # filename已经不适应这个版本了
                    # patch_results = dict(
                    #     img_prefix=None, img_info=dict(filename=img_file_path))
                    patch_results = dict(
                        img_prefix=None, img_info=dict(filename=img_file_path),img_path = img_file_path)
                    # # result字典的key更换了新名字
                    # # results = dict(pts_filename=pcd_file_path)
                    # results = dict(lidar_points=dict(lidar_path=pcd_file_path))
                    # 读取对应的img patch
                    # dict_keys(['img_prefix', 'img_info', 'img_path', 'img', 'img_shape', 'ori_shape'])
                    # print("=============self.img_loader(patch_results) keys : {}==============".format(self.img_loader(patch_results).keys()))
                    
                    s_patch = self.img_loader(patch_results)['img']
                    # 经过测试s_patch可以获取图像
                    # cv2.imwrite("s_patch_{}.jpg".format(inds),s_patch)
                    # img[camera_idx] = self.paste_obj(
                    #     img[camera_idx],
                    #     s_patch,
                    #     # s_mask,
                    #     bbox_2d=bbox_2d.astype(np.int32))

                    img = self.paste_obj(
                        img,
                        s_patch,
                        # s_mask,
                        bbox_2d=bbox_2d.astype(np.int32))

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],dtype=np.int64)
            # print("==================gt_labels : {}================".format(gt_labels))
            # 写入一下粘贴后的图片
            # 经测试可以正常粘贴,就是有点拥挤,可以考虑采样率调低一些
            # cv2.imwrite('kitti_code/figs/{}.png'.format(img_file_path.split("\\")[-1].split(".")[0]), img)
            ret = dict(
                img=img,
                gt_labels=gt_labels,
                gt_labels_3d=copy.deepcopy(gt_labels),
                gt_bboxes_3d=sampled_gt_bboxes_3d,
                gt_bboxes_2d=sampled_gt_bboxes_2d,
                points=sampled_points,
                group_ids=np.arange(gt_bboxes_3d.shape[0],
                                    gt_bboxes_3d.shape[0] + len(sampled)))
        # sample all函数正常执行结束
        return ret

    def paste_obj(self, img, obj_img, bbox_2d):
        # paste the image patch back
        x1, y1, x2, y2 = bbox_2d
        # the bbox might exceed the img size because the img is different
        img_h, img_w = img.shape[:2]
        w = np.maximum(min(x2, img_w - 1) - x1 + 1, 1)
        h = np.maximum(min(y2, img_h - 1) - y1 + 1, 1)
        # obj_mask = obj_mask[:h, :w]
        obj_img = obj_img[:h, :w]
        obj_mask = np.zeros((h, w))
        margin_h = int(0.05 * h)
        margin_w = int(0.05 * w)
        obj_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 1.

        # choose a blend option
        if not self.blending_type:
            blending_op = 'none'

        else:
            blending_choice = np.random.randint(len(self.blending_type))
            blending_op = self.blending_type[blending_choice]

        if blending_op.find('poisson') != -1:
            # options: cv2.NORMAL_CLONE=1, or cv2.MONOCHROME_TRANSFER=3
            # cv2.MIXED_CLONE mixed the texture, thus is not used.
            if blending_op == 'poisson':
                mode = np.random.choice([1, 3], 1)[0]
            elif blending_op == 'poisson_normal':
                mode = cv2.NORMAL_CLONE
            elif blending_op == 'poisson_transfer':
                mode = cv2.MONOCHROME_TRANSFER
            else:
                raise NotImplementedError
            center = (int(x1 + w / 2), int(y1 + h / 2))
            img = cv2.seamlessClone(obj_img, img, obj_mask * 255, center, mode)
        else:
            if blending_op == 'gaussian':
                obj_mask = cv2.GaussianBlur(
                    obj_mask.astype(np.float32), (5, 5), 2)
            elif blending_op == 'box':
                obj_mask = cv2.blur(obj_mask.astype(np.float32), (3, 3))
            paste_mask = 1 - obj_mask * self.mixup
            img[y1:y1 + h,
                x1:x1 + w] = (img[y1:y1 + h, x1:x1 + w].astype(np.float32) *
                              paste_mask[..., None]).astype(np.uint8)
            img[y1:y1 + h, x1:x1 + w] += (self.mixup * obj_img.astype(np.float32) *
                                          obj_mask[..., None]).astype(np.uint8)

        return img

    def sample_class_v2(self, name, num, gt_bboxes_3d, gt_bboxes_2d):
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes_3d.shape[0]
        num_sampled = len(sampled)
        # avoid collision in BEV first
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes_3d[:, 0:2], gt_bboxes_3d[:, 3:5], gt_bboxes_3d[:, 6])
        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes[:, 0:2], sp_boxes[:, 3:5], sp_boxes[:, 6])
        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        # Then avoid collision in 2D space
        if self.check_2D_collision:
            sp_boxes_2d = np.stack([i['box2d_camera'] for i in sampled],
                                   axis=0)
            total_bbox_2d = np.concatenate([gt_bboxes_2d, sp_boxes_2d],
                                           axis=0)  # Nx4
            # random select a collision threshold
            if isinstance(self.collision_thr, float):
                collision_thr = self.collision_thr
            elif isinstance(self.collision_thr, list):
                collision_thr = np.random.choice(self.collision_thr)
            elif isinstance(self.collision_thr, dict):
                mode = self.collision_thr.get('mode', 'value')
                if mode == 'value':
                    collision_thr = np.random.choice(
                        self.collision_thr['thr_range'])
                elif mode == 'range':
                    collision_thr = np.random.uniform(
                        self.collision_thr['thr_range'][0],
                        self.collision_thr['thr_range'][1])

            if collision_thr == 0:
                # use similar collision test as BEV did
                # Nx4 (x1, y1, x2, y2) -> corners: Nx4x2
                # ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
                x1y1 = total_bbox_2d[:, :2]
                x2y2 = total_bbox_2d[:, 2:4]
                x1y2 = np.stack([total_bbox_2d[:, 0], total_bbox_2d[:, 3]],
                                axis=-1)
                x2y1 = np.stack([total_bbox_2d[:, 2], total_bbox_2d[:, 1]],
                                axis=-1)
                total_2d = np.stack([x1y1, x2y1, x1y2, x2y2], axis=1)
                coll_mat_2d = data_augment_utils.box_collision_test(
                    total_2d, total_2d)
            else:
                # use iof rather than iou to protect the foreground
                overlaps = box_np_ops.iou_jit(total_bbox_2d, total_bbox_2d,
                                              'iof')
                coll_mat_2d = overlaps > collision_thr
            coll_mat = coll_mat + coll_mat_2d

        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])

        return valid_samples
    







@TRANSFORMS.register_module()
class NusDataBaseSampler(DataBaseSampler):

    def __init__(self,
                 info_path: str,
                 data_root: str,
                 rate: float,
                 prepare: dict,
                 sample_groups: dict,
                 classes: Optional[List[str]] = None,
                 check_2D_collision=False,
                 collision_thr=0,
                 collision_in_classes=False,
                 depth_consistent=False,
                 blending_type=None,
                 mixup=1.0,
                 img_num=1,
                #  Imge_loader是加载patch的因此之加载一张图片
                 img_loader=dict(type='LoadImageFromFile',backend_args=None),
                #  img_loader=dict(type='LoadMultiViewImageFromFiles', backend_args=None),
                #  mask_loader=dict(
                #      type='LoadImageFromFile', color_type='grayscale'),
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     load_dim=4,
                     coord_type='LIDAR',
                     use_dim=[0, 1, 2, 3],
                     backend_args=None),
                 backend_args: Optional[dict] = None                 
                 ):
        super(NusDataBaseSampler, self).__init__(
            info_path=info_path,
            data_root=data_root,
            rate=rate,
            prepare=prepare,
            sample_groups=sample_groups,
            classes=classes,
            points_loader=points_loader)
        
        self.blending_type = blending_type
        self.depth_consistent = depth_consistent
        self.check_2D_collision = check_2D_collision
        self.collision_thr = collision_thr
        self.collision_in_classes = collision_in_classes
        # 构建img-loader
        self.img_loader = TRANSFORMS.build(img_loader)
        self.mixup = mixup
        self.img_num = img_num
        # self.mask_loader = mmcv.build_from_cfg(mask_loader, PIPELINES)
        print("=========== BUILD NusDataBaseSampler =============")
        print("=========== The img_num of NusDataBaseSampler is {} =============".format(img_num))

    def sample_all(self, gt_bboxes_3d, gt_names, gt_bboxes_2d=None, img=None, img_filename=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        # class_name:种类;max_sample_num:对应种类的采样数目
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            # 生成采样数目录入对应种类的sampled_num_dict
            sampled_num_dict[class_name] = sampled_num
            # 每一类采样多少
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes_3d = []
        sampled_gt_bboxes_2d = []
        avoid_coll_boxes_3d = gt_bboxes_3d
        avoid_coll_boxes_2d = gt_bboxes_2d
        # class_name = 种类名称,如"car";sampled_num = 对应种类的采样数量,在上面计算了
        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            # 如果采样数量多大于0就进行sample_class_v2
            if sampled_num > 0:
                # 输入:采样类别名称,采样数量,2d 3d两个bbox
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes_3d,
                                                   avoid_coll_boxes_2d)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box_3d = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                        sampled_gt_box_2d = sampled_cls[0]['box2d_camera'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box_3d = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)
                        sampled_gt_box_2d = np.stack(
                            [s['box2d_camera'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes_3d += [sampled_gt_box_3d]
                    sampled_gt_bboxes_2d += [sampled_gt_box_2d]
                    if self.collision_in_classes:
                        # TODO: check whether check collision check among
                        # classes is necessary
                        avoid_coll_boxes_3d = np.concatenate(
                            [avoid_coll_boxes_3d, sampled_gt_box_3d], axis=0)
                        avoid_coll_boxes_2d = np.concatenate(
                            [avoid_coll_boxes_2d, sampled_gt_box_2d], axis=0)

        ret = None
        # kitti img shape : 375×1242×3,一张标准的三通道图,不存在什么camera id
        # nusences img shape 可能有四通道,有camera id
        origin_img = img.copy()
        # print("==================shape of nusences img : {}=================".format(np.array(img).shape))
        if len(sampled) > 0:
            sampled_gt_bboxes_3d = np.concatenate(sampled_gt_bboxes_3d, axis=0)
            sampled_gt_bboxes_2d = np.concatenate(sampled_gt_bboxes_2d, axis=0)

            # Get all objects out along with its label (if original or virtual)
            num_origin = gt_bboxes_2d.shape[0]
            origin_label = np.zeros((num_origin, 1))
            num_virtual = sampled_gt_bboxes_2d.shape[0]
            virtual_label = np.ones((num_virtual, 1))

            all_labels = np.concatenate([origin_label, virtual_label], axis=0)
            all_bboxes_3d = np.concatenate([gt_bboxes_3d, sampled_gt_bboxes_3d], axis=0)
            bboxes_2d = np.concatenate([gt_bboxes_2d, sampled_gt_bboxes_2d], axis=0)
            # kitti数据集不存在camera id这个东西,Bbox就是全部
            # 但是nusences数据集上有多个相机
            # print("======================bboxes_2d:{}========================".format(bboxes_2d))
            all_camera_idx = bboxes_2d[:, -1]
            all_bboxes_2d = bboxes_2d[:, :4]

            num_obj = all_bboxes_3d.shape[0]
            # 调整2d粘贴顺序,先粘贴后面的再粘贴前面的
            imgs_paste_order = np.argsort(-all_bboxes_3d[:, 0])
            # first of all, we need to get all points
            point_list = []
            # 分开处理点云和图像,点云不用排序直接粘贴,按原方法来
            for idx in range(len(sampled)):
                info = sampled[idx]
                # 原版的file path 替换成 pcd_file_path
                pcd_file_path = os.path.join(self.data_root, info['path'])\
                    if self.data_root else info['path']
                # print("=============pcd_file_path : {}==============".format(pcd_file_path))
                # result字典的key更换了新名字
                # results = dict(pts_filename=pcd_file_path)
                results = dict(lidar_points=dict(lidar_path=pcd_file_path))
                s_points = self.points_loader(results)['points']
                s_points.translate(info['box3d_lidar'][:3])
                # s_points_list名称更换为point_list
                point_list.append(s_points)
            # 提前制作完concat点云,返回时直接调用而不再是返回的时候再拼接 
            sampled_points = point_list[0].cat(point_list)


            # 打印最终的文件粘贴
            # img_filename = ["cam0.jpg","cam1.jpg","cam2.jpg","cam3.jpg","cam4.jpg","cam5.jpg"]
            # 接下来粘贴图像,和点云不同,图像需要按照物体的深度粘贴
            # 因此调换顺序
            # paste images in a far to near manner
            for idx in range(num_obj):
                inds = np.where(imgs_paste_order == idx)[0][0]
                bbox_2d = all_bboxes_2d[inds]; bbox_3d = all_bboxes_3d[inds]
                # nusences数据集需要记录camera idx来判断是哪个视角的相机
                camera_idx = int(all_camera_idx[inds])                                        
                # use only one image
                if camera_idx >= self.img_num:
                    continue
                # 按照粘贴顺序排序
                label = all_labels[inds]
                if label == 0:
                    x1, y1, x2, y2 = [int(ii) for ii in bbox_2d]
                    # 使用camera id(nusences多图时)
                    img[camera_idx][x1:x2, y1:y2] = self.mixup * origin_img[camera_idx][x1:x2, y1:y2] + \
                            (1 - self.mixup) * img[camera_idx][x1:x2, y1:y2]
                    # 不使用camera id(KITTI单图时)
                    # img[x1:x2, y1:y2] = self.mixup * origin_img[x1:x2, y1:y2] + \
                    #         (1 - self.mixup) * img[x1:x2, y1:y2]
                else:
                    inds -= num_origin
                    info = sampled[inds]
                    pcd_file_path = os.path.join(
                        self.data_root,
                        info['path']) if self.data_root else info['path']
                    img_file_path = pcd_file_path + '.png'
                    # print("=============img_file_path : {}==============".format(img_file_path))
                    # filename已经不适应这个版本了
                    # patch_results = dict(
                    #     img_prefix=None, img_info=dict(filename=img_file_path))
                    patch_results = dict(
                        img_prefix=None, img_info=dict(filename=img_file_path),img_path = img_file_path)
                    # # result字典的key更换了新名字
                    # # results = dict(pts_filename=pcd_file_path)
                    # results = dict(lidar_points=dict(lidar_path=pcd_file_path))
                    # 读取对应的img patch
                    # dict_keys(['img_prefix', 'img_info', 'img_path', 'img', 'img_shape', 'ori_shape'])
                    # print("=============self.img_loader(patch_results) keys : {}==============".format(self.img_loader(patch_results).keys()))
                    
                    # img_loader只是用于家在间切好的path img, 因此只设置加载单图完全正常
                    s_patch = self.img_loader(patch_results)['img']


                    # 经过测试s_patch可以获取图像
                    # 测试是否正确切出图像
                    # cv2.imwrite("s_patch_{}.jpg".format(inds),s_patch)

                    # 在不同视角中粘贴
                    img[camera_idx] = self.paste_obj(
                        img[camera_idx],
                        s_patch,
                        # s_mask,
                        bbox_2d=bbox_2d.astype(np.int32))
                    # 单图粘贴
                    # img = self.paste_obj(
                    #     img,
                    #     s_patch,
                    #     # s_mask,
                    #     bbox_2d=bbox_2d.astype(np.int32))

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],dtype=np.int64)
            # print("==================gt_labels : {}================".format(gt_labels))
            # 写入一下粘贴后的图片
            # 经测试可以正常粘贴,就是有点拥挤,可以考虑采样率调低一些
            # cv2.imwrite('kitti_code/figs/{}.png'.format(img_file_path.split("\\")[-1].split(".")[0]), img)
            # for camera_idx, each_img in enumerate(img):
            #     # cv2.imwrite('nus_code/gtdb_figs/%s' % img_filename[camera_idx].split('/')[-1], each_img)
            #     cv2.imwrite('nus_code/gtdb_figs/%s' % img_filename[camera_idx], each_img)
            # print("===========================测试 object paste for img 完毕 将退出程序===================================")
            # exit()
            ret = dict(
                img=img,
                gt_labels=gt_labels,
                gt_labels_3d=copy.deepcopy(gt_labels),
                gt_bboxes_3d=sampled_gt_bboxes_3d,
                gt_bboxes_2d=sampled_gt_bboxes_2d,
                points=sampled_points,
                group_ids=np.arange(gt_bboxes_3d.shape[0],
                                    gt_bboxes_3d.shape[0] + len(sampled)))
        # sample all函数正常执行结束
        return ret

    def paste_obj(self, img, obj_img, bbox_2d):
        # paste the image patch back
        x1, y1, x2, y2 = bbox_2d
        # the bbox might exceed the img size because the img is different
        img_h, img_w = img.shape[:2]
        w = np.maximum(min(x2, img_w - 1) - x1 + 1, 1)
        h = np.maximum(min(y2, img_h - 1) - y1 + 1, 1)
        # obj_mask = obj_mask[:h, :w]
        obj_img = obj_img[:h, :w]
        obj_mask = np.zeros((h, w))
        margin_h = int(0.05 * h)
        margin_w = int(0.05 * w)
        obj_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 1.

        # choose a blend option
        if not self.blending_type:
            blending_op = 'none'

        else:
            blending_choice = np.random.randint(len(self.blending_type))
            blending_op = self.blending_type[blending_choice]

        if blending_op.find('poisson') != -1:
            # options: cv2.NORMAL_CLONE=1, or cv2.MONOCHROME_TRANSFER=3
            # cv2.MIXED_CLONE mixed the texture, thus is not used.
            if blending_op == 'poisson':
                mode = np.random.choice([1, 3], 1)[0]
            elif blending_op == 'poisson_normal':
                mode = cv2.NORMAL_CLONE
            elif blending_op == 'poisson_transfer':
                mode = cv2.MONOCHROME_TRANSFER
            else:
                raise NotImplementedError
            center = (int(x1 + w / 2), int(y1 + h / 2))
            img = cv2.seamlessClone(obj_img, img, obj_mask * 255, center, mode)
        else:
            if blending_op == 'gaussian':
                obj_mask = cv2.GaussianBlur(
                    obj_mask.astype(np.float32), (5, 5), 2)
            elif blending_op == 'box':
                obj_mask = cv2.blur(obj_mask.astype(np.float32), (3, 3))
            paste_mask = 1 - obj_mask * self.mixup
            img[y1:y1 + h,
                x1:x1 + w] = (img[y1:y1 + h, x1:x1 + w].astype(np.float32) *
                              paste_mask[..., None]).astype(np.uint8)
            img[y1:y1 + h, x1:x1 + w] += (self.mixup * obj_img.astype(np.float32) *
                                          obj_mask[..., None]).astype(np.uint8)

        return img

    def sample_class_v2(self, name, num, gt_bboxes_3d, gt_bboxes_2d):
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes_3d.shape[0]
        num_sampled = len(sampled)
        # avoid collision in BEV first
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes_3d[:, 0:2], gt_bboxes_3d[:, 3:5], gt_bboxes_3d[:, 6])
        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes[:, 0:2], sp_boxes[:, 3:5], sp_boxes[:, 6])
        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        # Then avoid collision in 2D space
        if self.check_2D_collision:
            sp_boxes_2d = np.stack([i['box2d_camera'] for i in sampled],
                                   axis=0)
            total_bbox_2d = np.concatenate([gt_bboxes_2d, sp_boxes_2d],
                                           axis=0)  # Nx4
            # random select a collision threshold
            if isinstance(self.collision_thr, float):
                collision_thr = self.collision_thr
            elif isinstance(self.collision_thr, list):
                collision_thr = np.random.choice(self.collision_thr)
            elif isinstance(self.collision_thr, dict):
                mode = self.collision_thr.get('mode', 'value')
                if mode == 'value':
                    collision_thr = np.random.choice(
                        self.collision_thr['thr_range'])
                elif mode == 'range':
                    collision_thr = np.random.uniform(
                        self.collision_thr['thr_range'][0],
                        self.collision_thr['thr_range'][1])

            if collision_thr == 0:
                # use similar collision test as BEV did
                # Nx4 (x1, y1, x2, y2) -> corners: Nx4x2
                # ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
                x1y1 = total_bbox_2d[:, :2]
                x2y2 = total_bbox_2d[:, 2:4]
                x1y2 = np.stack([total_bbox_2d[:, 0], total_bbox_2d[:, 3]],
                                axis=-1)
                x2y1 = np.stack([total_bbox_2d[:, 2], total_bbox_2d[:, 1]],
                                axis=-1)
                total_2d = np.stack([x1y1, x2y1, x1y2, x2y2], axis=1)
                coll_mat_2d = data_augment_utils.box_collision_test(
                    total_2d, total_2d)
            else:
                # use iof rather than iou to protect the foreground
                overlaps = box_np_ops.iou_jit(total_bbox_2d, total_bbox_2d,
                                              'iof')
                coll_mat_2d = overlaps > collision_thr
            coll_mat = coll_mat + coll_mat_2d

        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])

        return valid_samples