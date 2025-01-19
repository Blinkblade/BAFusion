# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.necks.fpn import FPN
# from mmdet.models.necks.yolox_pafpn import YOLOXPAFPN

from .dla_neck import DLANeck
from .imvoxel_neck import IndoorImVoxelNeck, OutdoorImVoxelNeck
from .pointnet2_fp_neck import PointNetFPNeck
from .second_fpn import SECONDFPN
from .yolox_pafpn import YOLOXPAFPN1
from .simple_neck import SimpleNeck


__all__ = [
    'FPN', 'SECONDFPN', 'OutdoorImVoxelNeck', 'PointNetFPNeck', 'DLANeck',
    'IndoorImVoxelNeck',"YOLOXPAFPN1",
    'SimpleNeck',    
]   
