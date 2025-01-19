# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DDetector
from .centerpoint import CenterPoint
# from .centerpoint import CenterPoint,FusionCenterPoint,DeepFusionCenterPoint
from .dfm import DfM
from .dynamic_voxelnet import DynamicVoxelNet
from .fcos_mono3d import FCOSMono3D
from .groupfree3dnet import GroupFree3DNet
from .h3dnet import H3DNet
from .imvotenet import ImVoteNet
from .imvoxelnet import ImVoxelNet
from .mink_single_stage import MinkSingleStage3DDetector
from .multiview_dfm import MultiViewDfM
from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
from .mvx_two_stage import MVXTwoStageDetector
from .parta2 import PartA2
from .point_rcnn import PointRCNN
from .pv_rcnn import PointVoxelRCNN
from .sassd import SASSD
from .single_stage_mono3d import SingleStageMono3DDetector
from .smoke_mono3d import SMOKEMono3D
from .ssd3dnet import SSD3DNet
from .votenet import VoteNet
from .voxelnet import VoxelNet
# from .voxelnet import VoxelNet,FusionVoxelNet,BiFusionVoxelNet
from .BA_voxelnet import BAVoxelNet,DynamicBAVoxelNet,BEAVoxelNet
from .BA_centerpoint import BACenterPoint
# from .BA_voxelnetV2 import BAVoxelNetV2
from .BA_imvoxelnet import BAImVoxelNet,BAImVoxelNetS
from .BA_parta2 import BAPartA2,BAPartA2S
from .Fusion_VoxelNet import FusionVoxelNet, BiFusionVoxelNet
# PRUNE
from .BA_voxelnet_Prune import BAVoxelNetPrune

# parallel
from .BA_Voxelnet_Parallel import BAVoxelNetParallel
__all__ = [
    'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
    'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
    'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
    'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet', 'PointRCNN', 'SMOKEMono3D',
    'SASSD', 'MinkSingleStage3DDetector', 'MultiViewDfM', 'DfM',
    'PointVoxelRCNN',
    'BAVoxelNet','DynamicBAVoxelNet',
    'FusionVoxelNet','BiFusionVoxelNet',
    'BACenterPoint',
    'BAImVoxelNet', 'BAImVoxelNetS',
    'BAPartA2','BAPartA2S',
    'BAVoxelNetParallel',
    'BEAVoxelNet',
    'BAVoxelNetPrune',
]
