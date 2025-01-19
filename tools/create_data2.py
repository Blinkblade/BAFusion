# Copyright (c) OpenMMLab. All rights reserved.
# 改版,用于读取2d标签bbox
import argparse
from os import path as osp

from tools.dataset_converters import indoor_converter as indoor
from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters import lyft_converter as lyft_converter

from tools.dataset_converters import nuscenes_converter as nuscenes_converter
from tools.dataset_converters import nuscenes_converter2D as nuscenes_converter2D

from tools.dataset_converters import semantickitti_converter
from tools.dataset_converters.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)
from tools.dataset_converters.create_gt_database2 import create_groundtruth_database2D
from tools.dataset_converters.create_gt_database2 import create_groundtruth_database2D_nus
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos
from tools.dataset_converters.update_infos_to_v2_nus2d import update_pkl_infos_nus2d


def kitti_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    with_plane=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_trainval_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_test_path)
    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


# 写一个新的函数用于处理nuscenes数据集
# 他可以生成带有2d_gt_bboxes的框并可以选择是否生成db用于测试
def nuscenes_data_prep2D(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10,
                       gt_database = True,
                       data_root = 'nuscenes',
                       ):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
        gt_database: whether create_groundtruth_database or not
    """
    nuscenes_converter2D.create_nuscenes_infos2D(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    # nuscenes_converter.create_nuscenes_infos(
    #     root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos_nus2d(data_root, out_dir=out_dir, pkl_path=info_test_path)
        return

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    # 用新写的生成带有2DBBOX的nuscenes
    update_pkl_infos_nus2d(data_root, out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos_nus2d(data_root, out_dir=out_dir, pkl_path=info_val_path)
    # 默认处理流程李,执行普通的create gt database
    if gt_database:
        create_groundtruth_database2D_nus(dataset_name, root_path, info_prefix,
                                          f'{info_prefix}_infos_train.pkl')






def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path)
        return

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path)
    # 默认处理流程李,执行普通的create gt database
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{info_prefix}_infos_train.pkl')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    if version == 'v1.01-test':
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_test_path)
    elif version == 'v1.01-train':
        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_train_path)
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_val_path)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_test_path)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    splits = [f'Area_{i}' for i in [1, 2, 3, 4, 5, 6]]
    for split in splits:
        filename = osp.join(out_dir, f'{info_prefix}_infos_{split}.pkl')
        update_pkl_infos('s3dis', out_dir=out_dir, pkl_path=filename)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('sunrgbd', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('sunrgbd', out_dir=out_dir, pkl_path=info_val_path)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 5. Here we store pose information of these frames
            for later use.
    """
    from tools.dataset_converters import waymo_converter as waymo

    splits = [
        'training', 'validation', 'testing', 'testing_3d_camera_only_detection'
    ]
    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split
                       in ['testing', 'testing_3d_camera_only_detection']))
        converter.convert()

    from tools.dataset_converters.waymo_converter import \
        create_ImageSets_img_ids
    create_ImageSets_img_ids(osp.join(out_dir, 'kitti_format'), splits)
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(
        out_dir, info_prefix, max_sweeps=max_sweeps, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_trainval_path)
    update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_test_path)
    GTDatabaseCreater(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        num_worker=workers).create()


def semantickitti_data_prep(info_prefix, out_dir):
    """Prepare the info file for SemanticKITTI dataset.

    Args:
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
    """
    semantickitti_converter.create_semantickitti_info_file(
        info_prefix, out_dir)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--only-gt-database',
    action='store_true',
    help='Whether to only generate ground truth database.')
# with_2d参数,自动读取anno中的2dbbox
parser.add_argument(
    '--with-2D',
    action='store_true',
    help='读取生成的2Danno标签')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    if args.dataset == 'kitti':
        if args.only_gt_database:
            if args.with_2D:
                print("==========================执行create_groundtruth_database2D========================")
                create_groundtruth_database2D(
                    'KittiDataset',
                    args.root_path,
                    args.extra_tag,
                    f'{args.extra_tag}_infos_train.pkl',
                    relative_path=False,
                    mask_anno_path='instances_train.json',
                    # 这里修改with mask
                    with_2D = True)
            else:
                create_groundtruth_database(
                    'KittiDataset',
                    args.root_path,
                    args.extra_tag,
                    f'{args.extra_tag}_infos_train.pkl',
                    relative_path=False,
                    mask_anno_path='instances_train.json',
                    # 这里修改with mask
                    with_mask=(args.version == 'mask'))
        else:
            kitti_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=args.version,
                out_dir=args.out_dir,
                with_plane=args.with_plane)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        # nuscenes完整版修改2d
        if args.only_gt_database:
            if args.with_2D:
                print("==========================执行create_groundtruth_database2D_nus========================")
                create_groundtruth_database2D_nus('NuScenesDataset', 
                                                args.root_path,
                                                args.extra_tag,
                                                f'{args.extra_tag}_infos_train.pkl',
                                                with_2D=True)
                pass
            else:
                # 没有with_2D按正常标准来
                create_groundtruth_database('NuScenesDataset', args.root_path,
                                            args.extra_tag,
                                            f'{args.extra_tag}_infos_train.pkl')
        else:
            if args.with_2D:
                # 额外生成2d bboxes
                print("==========================nuscenes_data_prep2D========================")
                train_version = f'{args.version}-trainval'
                nuscenes_data_prep2D(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=train_version,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps,
                    # 先不生成gt db用于快速测试
                    # gt db可以之后单独生成
                    gt_database=False
                )
                test_version = f'{args.version}-test'
                nuscenes_data_prep2D(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=test_version,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps,
                    # test本来就步生成gt db
                    gt_database=False
                    )
            else:
                train_version = f'{args.version}-trainval'
                nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=train_version,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps)
                test_version = f'{args.version}-test'
                nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=test_version,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps)
    # nuscenes mini 单独处理            
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        if args.only_gt_database:
            if args.with_2D:
                print("==========================执行create_groundtruth_database2D_nus========================")
                create_groundtruth_database2D_nus('NuScenesDataset', 
                                                args.root_path,
                                                args.extra_tag,
                                                f'{args.extra_tag}_infos_train.pkl',
                                                with_2D=True)
                pass
            else:            
                create_groundtruth_database('NuScenesDataset', 
                                            args.root_path,
                                            args.extra_tag,
                                            f'{args.extra_tag}_infos_train.pkl')
        else:
            if args.with_2D:
                # 额外生成2d bboxes
                print("==========================nuscenes_data_prep2D========================")
                train_version = f'{args.version}'
                nuscenes_data_prep2D(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=train_version,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps,
                    # 先不生成gt db用于快速测试
                    # gt db可以之后单独生成
                    gt_database=False,
                    data_root='nuscenes_mini'
                )

            else:
                train_version = f'{args.version}'
                nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=train_version,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'semantickitti':
        semantickitti_data_prep(
            info_prefix=args.extra_tag, out_dir=args.out_dir)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
