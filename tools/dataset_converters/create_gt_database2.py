# Copyright (c) OpenMMLab. All rights reserved.
# 用于生成带有2dbbox的gt db
import pickle
from os import path as osp

import mmcv
import mmengine
import numpy as np
from mmcv.ops import roi_align
from mmdet.evaluation import bbox_overlaps
from mmengine import track_iter_progress
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops as box_np_ops




def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def _parse_coco_ann_info(ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        if ann['area'] <= 0:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get('iscrowd', False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_masks_ann.append(ann['segmentation'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(
        bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair
    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = (
        torch.arange(num_pos,
                     device=device).to(dtype=pos_proposals.dtype)[:, None])
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks).to(device).index_select(
            0, pos_assigned_gt_inds).to(dtype=rois.dtype))
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    targets = (
        roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1))
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1:y1 + h, x1:x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1:y1 + h, x1:x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks

def crop_img_patch_with_box(pos_proposals, org_img):
    num_pos = pos_proposals.shape[0]
    img_patches = []
    for i in range(num_pos):
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        img_patch = org_img[y1:y1 + h, x1:x1 + w]
        img_patches.append(img_patch)
    return img_patches


# 多视角相机应该用这个
# org_img_list应该是六个不同视角相机的图像，里面按顺序装了六个图像
def crop_img_patch_with_box_v2(pos_proposals, org_img_list):
    # 有多少个bbox
    num_pos = pos_proposals.shape[0]
    img_patches = []
    for i in range(num_pos):
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2, num_img = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        img_patch = org_img_list[num_img][y1:y1 + h, x1:x1 + w]
        img_patches.append(img_patch)
    return img_patches



def create_groundtruth_database(dataset_class_name,
                                data_path,
                                info_prefix,
                                info_path=None,
                                mask_anno_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None,
                                with_mask=False):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
    """
    print(f'Create GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path)
    if dataset_class_name == 'KittiDataset':
        backend_args = None
        dataset_cfg.update(
            modality=dict(
                use_lidar=True,
                use_camera=with_mask,
            ),
            data_prefix=dict(
                pts='training/velodyne_reduced', img='training/image_2'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    backend_args=backend_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    backend_args=backend_args)
            ])

    elif dataset_class_name == 'NuScenesDataset':
        dataset_cfg.update(
            use_valid_flag=True,
            data_prefix=dict(
                pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])

    elif dataset_class_name == 'WaymoDataset':
        backend_args = None
        dataset_cfg.update(
            test_mode=False,
            data_prefix=dict(
                pts='training/velodyne', img='', sweeps='training/velodyne'),
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=6,
                    use_dim=6,
                    backend_args=backend_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    backend_args=backend_args)
            ])

    dataset = DATASETS.build(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f'{info_prefix}_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path,
                                     f'{info_prefix}_dbinfos_train.pkl')
    mmengine.mkdir_or_exist(database_save_path)
    all_db_infos = dict()
    if with_mask:
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0]
            file2id.update({info['file_name']: i})

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        data_info = dataset.get_data_info(j)
        example = dataset.pipeline(data_info)
        # print("=============example:{}================".format(example))
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        names = [dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        if with_mask:
            # prepare masks
            gt_boxes = annos['gt_bboxes']
            img_path = osp.split(example['img_info']['filename'])[-1]
            if img_path not in file2id.keys():
                print(f'skip image {img_path} for empty mask')
                continue
            img_id = file2id[img_path]
            kins_annIds = coco.getAnnIds(imgIds=img_id)
            kins_raw_info = coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos['img_shape'][:2]
            gt_masks = [
                _poly2mask(mask, h, w) for mask in kins_ann_info['masks']
            ]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info['bboxes'], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = (bbox_iou.max(axis=0) > 0.5)

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos['img'])

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f'{info_prefix}_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            if with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + '.png'
                mask_patch_path = abs_filepath + '.mask.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if with_mask:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


class GTDatabaseCreater:
    """Given the raw data, generate the ground truth database. This is the
    parallel version. For serialized version, please refer to
    `create_groundtruth_database`

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
        num_worker (int, optional): the number of parallel workers to use.
            Default: 8.
    """

    def __init__(self,
                 dataset_class_name,
                 data_path,
                 info_prefix,
                 info_path=None,
                 mask_anno_path=None,
                 used_classes=None,
                 database_save_path=None,
                 db_info_save_path=None,
                 relative_path=True,
                 add_rgb=False,
                 lidar_only=False,
                 bev_only=False,
                 coors_range=None,
                 with_mask=False,
                 num_worker=8) -> None:
        self.dataset_class_name = dataset_class_name
        self.data_path = data_path
        self.info_prefix = info_prefix
        self.info_path = info_path
        self.mask_anno_path = mask_anno_path
        self.used_classes = used_classes
        self.database_save_path = database_save_path
        self.db_info_save_path = db_info_save_path
        self.relative_path = relative_path
        self.add_rgb = add_rgb
        self.lidar_only = lidar_only
        self.bev_only = bev_only
        self.coors_range = coors_range
        self.with_mask = with_mask
        self.num_worker = num_worker
        self.pipeline = None

    def create_single(self, input_dict):
        group_counter = 0
        single_db_infos = dict()
        example = self.pipeline(input_dict)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        names = [
            self.dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']
        ]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        if self.with_mask:
            # prepare masks
            gt_boxes = annos['gt_bboxes']
            img_path = osp.split(example['img_info']['filename'])[-1]
            if img_path not in self.file2id.keys():
                print(f'skip image {img_path} for empty mask')
                return single_db_infos
            img_id = self.file2id[img_path]
            kins_annIds = self.coco.getAnnIds(imgIds=img_id)
            kins_raw_info = self.coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos['img_shape'][:2]
            gt_masks = [
                _poly2mask(mask, h, w) for mask in kins_ann_info['masks']
            ]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info['bboxes'], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = (bbox_iou.max(axis=0) > 0.5)

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos['img'])

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(self.database_save_path, filename)
            rel_filepath = osp.join(f'{self.info_prefix}_gt_database',
                                    filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            if self.with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + '.png'
                mask_patch_path = abs_filepath + '.mask.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (self.used_classes is None) or names[i] in self.used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if self.with_mask:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in single_db_infos:
                    single_db_infos[names[i]].append(db_info)
                else:
                    single_db_infos[names[i]] = [db_info]

        return single_db_infos

    def create(self):
        print(f'Create GT Database of {self.dataset_class_name}')
        dataset_cfg = dict(
            type=self.dataset_class_name,
            data_root=self.data_path,
            ann_file=self.info_path)
        if self.dataset_class_name == 'KittiDataset':
            backend_args = None
            dataset_cfg.update(
                test_mode=False,
                data_prefix=dict(
                    pts='training/velodyne_reduced', img='training/image_2'),
                modality=dict(
                    use_lidar=True,
                    use_depth=False,
                    use_lidar_intensity=True,
                    use_camera=self.with_mask,
                ),
                pipeline=[
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=4,
                        use_dim=4,
                        backend_args=backend_args),
                    dict(
                        type='LoadAnnotations3D',
                        with_bbox_3d=True,
                        with_label_3d=True,
                        backend_args=backend_args)
                ])

        elif self.dataset_class_name == 'NuScenesDataset':
            dataset_cfg.update(
                use_valid_flag=True,
                data_prefix=dict(
                    pts='samples/LIDAR_TOP', img='',
                    sweeps='sweeps/LIDAR_TOP'),
                pipeline=[
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=5,
                        use_dim=5),
                    dict(
                        type='LoadPointsFromMultiSweeps',
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True),
                    dict(
                        type='LoadAnnotations3D',
                        with_bbox_3d=True,
                        with_label_3d=True)
                ])

        elif self.dataset_class_name == 'WaymoDataset':
            backend_args = None
            dataset_cfg.update(
                test_mode=False,
                data_prefix=dict(
                    pts='training/velodyne',
                    img='',
                    sweeps='training/velodyne'),
                modality=dict(
                    use_lidar=True,
                    use_depth=False,
                    use_lidar_intensity=True,
                    use_camera=False,
                ),
                pipeline=[
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=6,
                        use_dim=6,
                        backend_args=backend_args),
                    dict(
                        type='LoadAnnotations3D',
                        with_bbox_3d=True,
                        with_label_3d=True,
                        backend_args=backend_args)
                ])

        self.dataset = DATASETS.build(dataset_cfg)
        self.pipeline = self.dataset.pipeline
        if self.database_save_path is None:
            self.database_save_path = osp.join(
                self.data_path, f'{self.info_prefix}_gt_database')
        if self.db_info_save_path is None:
            self.db_info_save_path = osp.join(
                self.data_path, f'{self.info_prefix}_dbinfos_train.pkl')
        mmengine.mkdir_or_exist(self.database_save_path)
        if self.with_mask:
            self.coco = COCO(osp.join(self.data_path, self.mask_anno_path))
            imgIds = self.coco.getImgIds()
            self.file2id = dict()
            for i in imgIds:
                info = self.coco.loadImgs([i])[0]
                self.file2id.update({info['file_name']: i})

        def loop_dataset(i):
            input_dict = self.dataset.get_data_info(i)
            input_dict['box_type_3d'] = self.dataset.box_type_3d
            input_dict['box_mode_3d'] = self.dataset.box_mode_3d
            return input_dict

        multi_db_infos = mmengine.track_parallel_progress(
            self.create_single,
            ((loop_dataset(i)
              for i in range(len(self.dataset))), len(self.dataset)),
            self.num_worker)
        print('Make global unique group id')
        group_counter_offset = 0
        all_db_infos = dict()
        for single_db_infos in track_iter_progress(multi_db_infos):
            group_id = -1
            for name, name_db_infos in single_db_infos.items():
                for db_info in name_db_infos:
                    group_id = max(group_id, db_info['group_id'])
                    db_info['group_id'] += group_counter_offset
                if name not in all_db_infos:
                    all_db_infos[name] = []
                all_db_infos[name].extend(name_db_infos)
            group_counter_offset += (group_id + 1)

        for k, v in all_db_infos.items():
            print(f'load {len(v)} {k} database infos')

        with open(self.db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
            
            
        
# 没有mask但是自带2d bbox的gt_dbinfo创建
def create_groundtruth_database2D(dataset_class_name,
                                data_path,
                                info_prefix,
                                info_path=None,
                                mask_anno_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None,
                                with_2D = True):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
        with_2D (bool, optional): Whether to use 2D image.
            Default: True.
    """
    print(f'Create GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path)
    if dataset_class_name == 'KittiDataset':
        backend_args = None
        dataset_cfg.update(
            modality=dict(
                use_lidar=True,
                use_camera= with_2D ,
            ),
            data_prefix=dict(
                pts='training/velodyne_reduced', img='training/image_2'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    backend_args=backend_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_bbox=True,
                    with_label=True,
                    backend_args=backend_args)
            ])

    elif dataset_class_name == 'NuScenesDataset':
        dataset_cfg.update(
            use_valid_flag=True,
            data_prefix=dict(
                pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])

    elif dataset_class_name == 'WaymoDataset':
        backend_args = None
        dataset_cfg.update(
            test_mode=False,
            data_prefix=dict(
                pts='training/velodyne', img='', sweeps='training/velodyne'),
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=6,
                    use_dim=6,
                    backend_args=backend_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    backend_args=backend_args)
            ])

    dataset = DATASETS.build(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f'{info_prefix}_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path,
                                     f'{info_prefix}_dbinfos_train.pkl')
    mmengine.mkdir_or_exist(database_save_path)
    all_db_infos = dict()

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        data_info = dataset.get_data_info(j)
        example = dataset.pipeline(data_info)
        # print("=============example:{}================".format(example))
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        # # 2D图像和标签获取改在下面了,改自AAV2
        # if with_2D:
        #     gt_boxes = annos["gt_bboxes"]
            
        names = [dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)


        # 从AAV2移植过来的
        # prepare bboxed img
        if with_2D:
            # example从pipeline中来,由于pipeline设置了with 2dbbox所以有bbox
            # 3d bbox和mask都是从anno里读取的,这一步有待商榷
            # 不过AAV2中的nuscenes数据集里也是这么写的,所以可能还行
            # 悲惨的是,with_mask的代码根本没有更新,需要重写这里的读取路径
            gt_boxes = example['gt_bboxes']
            # 对照了新版的with_mask,这里应该没问题
            # img_path = osp.split(example['img_info']['filename'])[-1]
            # 明显比with mask的更合适
            if gt_boxes.shape[0] == 0:              
                print(f'skip image {img_path} for no bboxes')
                continue
            # h w 也没用,而且新版key也变了,直接删掉
            # h, w = example['img_shape'][:2]
            # object_img_patches = crop_img_patch_with_box(gt_boxes, example['img'])
            # 新版的images还是一个字典,还需要再取数
            # 从example["images"]["CAM2"]["img_path"]提取路径和从example['img_path']中提取是一样的
            # 我们直接从example['img_path']提取
            img_path_cam2 = example["images"]["CAM2"]["img_path"]
            image_data = mmcv.imread(img_path_cam2)
            object_img_patches = crop_img_patch_with_box(gt_boxes, image_data)
            # 输入到crop_img_patch_with_box需要是已经读取的图像,用mmcv的imread读取
            # 由于nus有六个相机,因此需要用一个列表装入六张图像,然后使用crop patch bbox v2
            # image_data_list = []
            # images  = example["images"]
            # print("==========key for images=========")
            # for key,value in images:
            #     print(key)
            #     img_path = value['img_path']
            #     image_data = mmcv.imread(img_path)
            #     # 逐一读取装入列表,按key的顺序
            #     image_data_list.append(image_data)
                # object_img_patches = crop_img_patch_with_box(gt_boxes, image_data)
            # object_img_patches = crop_img_patch_with_box_v2(gt_boxes, image_data_list)
        # 跳过mask因为不需要

        # 截出所有的object对应的点云和图像块
        for i in range(num_obj):
            # filename是最终存储的object点云的位置,名字结构是: "对应图片id_object类别名_object序号.bin"
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f'{info_prefix}_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            # 把对象的图像块写入相应的文件夹下
            # 同样和with mask对照过了
            if with_2D:
                # 写在同一文件夹下同一位置
                img_patch_path = abs_filepath + '.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)


            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                # name指的是类别名称
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if with_2D:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)




# 没有mask但是自带2d bbox的gt_dbinfo创建
# 这份代码应用与nusences数据集
def create_groundtruth_database2D_nus(dataset_class_name,
                                    data_path,
                                    info_prefix,
                                    info_path=None,
                                    mask_anno_path=None,
                                    used_classes=None,
                                    database_save_path=None,
                                    db_info_save_path=None,
                                    relative_path=True,
                                    add_rgb=False,
                                    lidar_only=False,
                                    bev_only=False,
                                    coors_range=None,
                                    with_2D = True):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
        with_2D (bool, optional): Whether to use 2D image.
            Default: True.
    """
    print(f'Create GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path)
    if dataset_class_name == 'KittiDataset':
        backend_args = None
        dataset_cfg.update(
            modality=dict(
                use_lidar=True,
                use_camera= with_2D ,
            ),
            data_prefix=dict(
                pts='training/velodyne_reduced', img='training/image_2'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    backend_args=backend_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_bbox=True,
                    with_label=True,
                    backend_args=backend_args)
            ])

    elif dataset_class_name == 'NuScenesDataset':
        # dataset_cfg.update(
        #     use_valid_flag=True,
        #     data_prefix=dict(
        #         pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
        #     pipeline=[
        #         dict(
        #             type='LoadPointsFromFile',
        #             coord_type='LIDAR',
        #             load_dim=5,
        #             use_dim=5),
        #         dict(
        #             type='LoadPointsFromMultiSweeps',
        #             sweeps_num=10,
        #             use_dim=[0, 1, 2, 3, 4],
        #             pad_empty_sweeps=True,
        #             remove_close=True),
        #         dict(
        #             type='LoadAnnotations3D',
        #             with_bbox_3d=True,
        #             with_label_3d=True)
        #     ])
        backend_args = None
        dataset_cfg.update(
            use_valid_flag=True,
            # 添加前缀
            data_prefix = dict(
                pts='samples/LIDAR_TOP',

                CAM_FRONT='samples/CAM_FRONT',
                CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                CAM_BACK='samples/CAM_BACK',
                CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
                CAM_BACK_LEFT='samples/CAM_BACK_LEFT',

                sweeps='sweeps/LIDAR_TOP'),
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                # 有2D就是true
                use_camera=True,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    backend_args=backend_args,),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(type='LoadMultiViewImageFromFiles'),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_bbox=True,
                    with_label=True,
                    backend_args=backend_args)
            ])


    elif dataset_class_name == 'WaymoDataset':
        backend_args = None
        dataset_cfg.update(
            test_mode=False,
            data_prefix=dict(
                pts='training/velodyne', img='', sweeps='training/velodyne'),
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=6,
                    use_dim=6,
                    backend_args=backend_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    backend_args=backend_args)
            ])

    dataset = DATASETS.build(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f'{info_prefix}_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path,
                                     f'{info_prefix}_dbinfos_train.pkl')
    mmengine.mkdir_or_exist(database_save_path)
    all_db_infos = dict()

    # with mask相关代码已经移除

    group_counter = 0
    # 这里的j每次会取出一个点云以及对应的六张图像，相当于一次采集的sample
    for j in track_iter_progress(list(range(len(dataset)))):
        # get_Data_info和v1.0时不同
        # data_info包含一次场景采集, 一个点云文件和六张图像
        data_info = dataset.get_data_info(j)
        with open('nus_data_info.pkl', 'wb') as f:
            pickle.dump(data_info, f)

        # print("\n=============data_info:{}================".format(data_info.keys()))
        # ann的keys: dict_keys(['gt_bboxes_labels', 'gt_bboxes_3d', 'bbox_3d_isvalid', 'gt_labels_3d', 
        # 'num_lidar_pts', 'num_radar_pts', 'velocities', 'instances'])
        # 其中 instance keys:dict_keys(['bbox_label', 'bbox_3d', 'bbox_3d_isvalid', 
        # 'bbox_label_3d', 'num_lidar_pts', 'num_radar_pts', 'velocity'])
        # 每次采样包含n个Instance,代表n个被检测物体
        # print("\n=============ann_info:{}================".format(data_info["ann_info"].keys()))
        example = dataset.pipeline(data_info)
        # print("=============example:{}================".format(example.keys()))
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
            
        names = [dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)


        # 从AAV2移植过来的
        # prepare bboxed img
        if with_2D:
            # example从pipeline中来,由于pipeline设置了with 2dbbox所以有bbox
            # 3d bbox和mask都是从anno里读取的,这一步有待商榷
            # 不过AAV2中的nuscenes数据集里也是这么写的,所以可能还行
            # 悲惨的是,with_mask的代码根本没有更新,需要重写这里的读取路径
            # 需要注意的是, gt_boxes和gt_boxes_3d的数量一致, 有多个图像的结果(1个Object只对应一张图像)
            # 经过修改，这个值有可能是None
            gt_boxes = example['gt_bboxes']
            # 对照了新版的with_mask,这里应该没问题
            # print("\n===create_gt_database2.py >>> create_groundtruth_database2D() >>> example.keys() : {}".format(example.keys()))
            # print("\n===create_gt_database2.py >>> create_groundtruth_database2D() >>> image_idx : {}".format(image_idx))
            # print("\n===create_gt_database2.py >>> create_groundtruth_database2D() >>> example[lidar_path] : {}".format(example["lidar_path"]))
            # print("\n===create_gt_database2.py >>> create_groundtruth_database2D() >>> example[img_path] : {}".format(example["img_path"]))
            # 这个img_path没有用处只是打印一下,不过还是更新一下新的代码
            # 提取出nusences的cam type列表
            # images包含六个相机,每个相机对应一张图像
            # img_path = osp.split(example['img_info']['filename'])[-1]
            # 明显比with mask的更合适
            # 如果没有gt_boxes或长度为0，都认为没有gt_boxes
            if gt_boxes is None:
                print(f'skip image {img_path} for bboxes is None')
                continue
            if gt_boxes.shape[0] == 0:              
                print(f'skip image {img_path} for no bboxes')
                continue
            # h w 也没用,而且新版key也变了,直接删掉
            # h, w = example['img_shape'][:2]
            # object_img_patches = crop_img_patch_with_box(gt_boxes, example['img'])
            # 新版的images还是一个字典,还需要再取数
            # 从example["images"]["CAM2"]["img_path"]提取路径和从example['img_path']中提取是一样的
            # 我们直接从example['img_path']提取
            # img_path_cam2 = example["images"]["CAM2"]["img_path"]
            # 输入到crop_img_patch_with_box需要是已经读取的图像,用mmcv的imread读取
            image_data_list = []
            images  = example["images"]
            # print("==========key for images=========")
            for key,value in images.items():
                # print(key)
                img_path = value['img_path']
                image_data = mmcv.imread(img_path)
                # 逐一读取装入列表,按key的顺序
                image_data_list.append(image_data)
                # object_img_patches = crop_img_patch_with_box(gt_boxes, image_data)
            object_img_patches = crop_img_patch_with_box_v2(gt_boxes, image_data_list)
        # 跳过mask因为不需要

        # 截出所有的object对应的点云和图像块
        for i in range(num_obj):
            # filename是最终存储的object点云的位置,名字结构是: "对应图片id_object类别名_object序号.bin"
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f'{info_prefix}_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            # 把对象的图像块写入相应的文件夹下
            # 同样和with mask对照过了
            if with_2D:
                # 写在同一文件夹下同一位置
                img_patch_path = abs_filepath + '.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)


            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                # name指的是类别名称
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if with_2D:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

