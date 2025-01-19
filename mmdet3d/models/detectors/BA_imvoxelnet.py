# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
from mmengine.structures import InstanceData

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models.layers.fusion_layers.point_fusion import point_sample
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptInstanceList


@MODELS.register_module()
class BAImVoxelNet(Base3DDetector):
    r"""`ImVoxelNet <https://arxiv.org/abs/2106.01178>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        neck_3d (:obj:`ConfigDict` or dict): The 3D neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        prior_generator (:obj:`ConfigDict` or dict): The prior points
            generator config.
        n_voxels (list): Number of voxels along x, y, z axis.
        coord_type (str): The type of coordinates of points cloud:
            'DEPTH', 'LIDAR', or 'CAMERA'.
        train_cfg (:obj:`ConfigDict` or dict, optional): Config dict of
            training hyper-parameters. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Config dict of test
            hyper-parameters. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (:obj:`ConfigDict` or dict, optional): The initialization
            config. Defaults to None.
    """

    def __init__(self,
                #  img 特征提取流程
                 backbone: ConfigType,
                 neck: ConfigType,
                 neck_3d: ConfigType,
                 
                #  head
                 bbox_head: ConfigType,
                 prior_generator: ConfigType,
                 n_voxels: List,
                 coord_type: str,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 
                 #  数据预处理,用于处理点云和图像
                 data_preprocessor: OptConfigType = None,
                 
                 init_cfg: OptConfigType = None,
                 #  在最后新加层次
                
                 #  points 特征提取流程
                 voxel_encoder: ConfigType = None,
                 middle_encoder: ConfigType = None,
                 pts_backbone: ConfigType = None,
                 pts_neck: ConfigType = None,
                 #  通道注意力
                 img_channel_attention: ConfigType = None,
                 #  交叉注意力融合模块
                 # image-point融合模块
                 IP_cross_attention: ConfigType = None, 
                 # Point-image融合模块  
                 PI_cross_attention: ConfigType = None, 
                 # 位置编码
                 position_embedding: ConfigType = None,
                 # 输入主干网络前的压缩层,压缩通道、聚合特征
                 # image-point聚合
                 IP_compress_layer:ConfigType = None,   
                 # point-image聚合
                 PI_compress_layer:ConfigType = None,                   
                 
                 
                 ):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # 构建图像特征提取流程
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.neck_3d = MODELS.build(neck_3d)
        # 构建点云特征提取流程
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        # 构建BAFusion组件
        # build layer
        if pts_backbone:
            self.pts_backbone = MODELS.build(pts_backbone)
            print("====================get type of {} pts_backbone layer===================".format(pts_backbone["type"]))
            pass
        
        if pts_neck:
            self.pts_neck = MODELS.build(pts_neck)
            print("====================get type of {} pts_neck layer===================".format(pts_neck["type"]))
            pass
        
        if img_channel_attention:
            self.img_channel_attention = MODELS.build(img_channel_attention)
            print("====================get type of {} img_channel_attention layer===================".format(img_channel_attention["type"]))
            pass
        
        if IP_cross_attention:
            self.IP_cross_attention = MODELS.build(IP_cross_attention)
            self.attention_type = IP_cross_attention["type"]
            print("====================get type of {} IP_cross_attention layer===================".format(IP_cross_attention["type"]))
            pass
        
        if PI_cross_attention:
            self.PI_cross_attention = MODELS.build(PI_cross_attention)
            # 二者attention type保持一致
            self.attention_type = PI_cross_attention["type"]
            print("====================get type of {} PI_cross_attention layer===================".format(PI_cross_attention["type"]))
            pass
        
        if position_embedding:
            self.position_embedding = MODELS.build(position_embedding)
            print("====================get type of {} position_embedding layer===================".format(position_embedding["type"]))
            pass
        
        if IP_compress_layer:
            self.IP_compress_layer = MODELS.build(IP_compress_layer)
            print("====================get type of {} IP_compress_layer layer===================".format(IP_compress_layer["type"]))
            pass
        
        if PI_compress_layer:
            self.PI_compress_layer = MODELS.build(PI_compress_layer)
            print("====================get type of {} PI_compress_layer layer===================".format(PI_compress_layer["type"]))
            pass
    
        
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.n_voxels = n_voxels
        self.coord_type = coord_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        print("=======================BAImVoxelNet for detectors.BA_imvoxelnet is used====================")
        pass        
 
    # property是属性不是函数
    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D pts backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None
    
    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in pts branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None
    
    @property
    def with_img_channel_attention(self):
        """bool: Whether the detector has a img_channel_attention."""
        return hasattr(self,'img_channel_attention') and self.img_channel_attention is not None
    
    @property
    def with_IP_cross_attention(self):
        """bool: Whether the detector has a IP_cross_attention."""
        return hasattr(self,'IP_cross_attention') and self.IP_cross_attention is not None

    @property
    def with_PI_cross_attention(self):
        """bool: Whether the detector has a PI_cross_attention."""
        return hasattr(self,'PI_cross_attention') and self.PI_cross_attention is not None

    @property
    def with_position_embedding(self):
        """bool: Whether the detector has a position_embedding."""
        return hasattr(self,'position_embedding') and self.position_embedding is not None
    
    @property
    def with_IP_compress_layer(self):
        """bool: Whether the detector has a IP_compress_layer."""
        return hasattr(self,'IP_compress_layer') and self.IP_compress_layer is not None

    @property
    def with_PI_compress_layer(self):
        """bool: Whether the detector has a PI_compress_layer."""
        return hasattr(self,'PI_compress_layer') and self.PI_compress_layer is not None
            


    def encode_feature(self,featrue):
        """encode feature to transformer format

        Args:
            feature (Tensor): 2d feature map
        Returns:
            Tensor: encoded_feature:encoded 1d feature
        """
        # 然后，我们需要将 pts_BEV_feature 的二维特征图展平成一维的特征向量，并且按照 muti_head x hidden_size_head 的顺序重新排列
        # 我们可以使用 permute 和 flatten 方法来实现这一步
        # 假设 muti_head 是 16 ， hidden_size_head 是 dim // muti_head ，也就是 16
        encoded_feature = featrue.flatten(2).permute(0, 2, 1)
        # 最后，我们返回 encoded_pts_feature ，它的大小应该是 [1, 64, 256] ，其中 64 是 4 x 16 的结果，也就是每个头部的特征向量的长度
        return encoded_feature

    def remap_feature(self,fusion_feature, channels, height, width):
        """remap transformer feature 1D to 2D feature map
        Args:
            fusion_feature (Tensor): feature map fuse by transformer,shape should be (batch_size,seq_len,dim)
            which seq_len = height × width;dim = channel

        """
        # 先从encode中还原位置,第0维不动,12维交换回来
        # 这样一来,第1维变回通道数,而第二维则是height×width的特征图
        fusion_feature = fusion_feature.permute(0, 2, 1)
        # 将tansformer提取的特征从1D还原到2D
        fusion_feature = fusion_feature.reshape(-1, channels, height, width) 
        return fusion_feature


    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        -> 3d neck -> bbox_head.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Tuple:
            - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).
            - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).
        """
        img = batch_inputs_dict['imgs']
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        # print("==============ImVoxelNet input img shape:{}============".format(img.shape))
        x = self.backbone(img)
        
        # i = 0
        # for xb in x:
        #     print("==============ImVoxelNet backbone{} x shape:{}============".format(i,xb.shape))
        #     i = i + 1
            
        x = self.neck(x)[0]
        # print("==============ImVoxelNet neck0 x shape:{}============".format(x.shape))
        points = self.prior_generator.grid_anchors([self.n_voxels[::-1]],
                                                   device=img.device)[0][:, :3]
        volumes, valid_preds = [], []
        for feature, img_meta in zip(x, batch_img_metas):
            img_scale_factor = (
                points.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                points.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = points.new_tensor(
                get_proj_mat_by_coord_type(img_meta, self.coord_type))
            volume = point_sample(
                img_meta,
                img_features=feature[None, ...],
                points=points,
                proj_mat=points.new_tensor(proj_mat),
                coord_type=self.coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img.shape[-2:],
                img_shape=img_meta['img_shape'][:2],
                aligned=False)
            volumes.append(
                volume.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0))
            valid_preds.append(
                ~torch.all(volumes[-1] == 0, dim=0, keepdim=True))
        x = torch.stack(volumes)
        x = self.neck_3d(x)
        
        # i = 0
        # for xb in x:
        #     print("==============ImVoxelNet neck_3d0 x shape:{}============".format(xb.shape))
        #     i = i + 1
        # 到这里x是提取出的neck特征,通道数为256,x为一个list,里面只装有一个特征
        # 取出列表里的特征
        x = x[0]
        # print("==============ImVoxelNet neck_3d0 x shape:{}============".format(x.shape))
        # 取得点云体素
        voxel_dict = batch_inputs_dict['voxels']
        # 体素化:voxel_num × voxel_dim
        # 在每个体素内提取特征
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        # 进一步编码点云特征 feature
        pts_feature = self.middle_encoder(voxel_features, 
                                          voxel_dict['coors'],batch_size)
        # 送入pts主干提取点云特征
        pts_feature = self.pts_backbone(pts_feature)
        # 进一步提取点云特征
        if self.with_pts_neck:
            pts_feature = self.pts_neck(pts_feature)
        # seconde fpn提取结果是一个列表
        # 里面只装了一个最后汇总输入head的向量,因此可以直接提出 
        pts_feature = pts_feature[0]
        # print("==============pts_feature neck shape:{}============".format(pts_feature.shape))
        # img特征的shape
        img_height = x.shape[2]
        img_width = x.shape[3]
        # channels大家都一样就不区分了
        channels = x.shape[1]
        batch_size = x.shape[0]
        
        # pts特征的shape
        pts_height = pts_feature.shape[2]
        pts_width = pts_feature.shape[3]
        # 交叉注意力融合,需要二者都有才能BiAttention
        if self.with_IP_cross_attention and self.with_PI_cross_attention:
            # 将图像特征编码到1D
            encoded_img_feature = self.encode_feature(x)
            # 将点云特征编码到1D            
            encoded_pts_feature = self.encode_feature(pts_feature)
            
            # print("=================encoded_img_feats shape {} =================".format(encoded_img_feature.shape))
            # print("=================encoded_pts_feat shape {} =================".format(encoded_pts_feature.shape))
            if self.attention_type == "SoftmaxCrossAttention" or "CrossFocusedLinearAttention":
                # FLA系列，输入顺序为query, key, value, H, W
                # H和W指的是key-value对的height和width,不过实际上,这个值只是保证attention层正常得到尺度
                # IP特征,即用point增强图像,key图像作为query,点云作为k-v对
                # PI特征,即用img增强point,points作为query,img作为k-v对
                # FLA系列，输入顺序为query, key, value, H, W
                PI_feature = self.PI_cross_attention(encoded_pts_feature,
                                                     encoded_img_feature,
                                                     encoded_img_feature,
                                                     H = img_height,
                                                     W = img_width)
                # print("=================PI_feature shape {} =================".format(PI_feature.shape))
                # remap到2D与原始pts特征concat并经过卷积层聚合
                PI_feature2D = self.remap_feature(PI_feature,
                                                  channels=channels,
                                                  height=pts_height,
                                                  width=pts_width)
                # print("=================PI_feature2D shape {} =================".format(PI_feature2D.shape))
                PI_feature2D = torch.concat((PI_feature2D,pts_feature),dim=1)
                # print("=================concated PI_feature2D shape {} =================".format(PI_feature2D.shape))
                # PI特征聚合
                compressed_PI_feature2D = self.PI_compress_layer(PI_feature2D)
                # print("=================compressed_PI_feature2D shape {} =================".format(compressed_PI_feature2D.shape))
                # 重新把PI特征编码为一维
                encoded_PI_feature = self.encode_feature(compressed_PI_feature2D)
                # print("=================encoded_PI_feature shape {} =================".format(encoded_PI_feature.shape))
                # img增强的pts特征同img进一步融合
                fusion_feature = self.IP_cross_attention(encoded_img_feature,
                                                         encoded_PI_feature,
                                                         encoded_PI_feature,
                                                         H=pts_height,
                                                         W=pts_width)
            else:
                raise ValueError("self.attention_type must be SoftmaxCrossAttention or CrossFocusedLinearAttention!")
            # print("=================transformer fusion_feature shape {} =================".format(fusion_feature.shape))
            # # 将tansformer提取的特征从1D还原到2D
            fusion_feature = self.remap_feature(fusion_feature,
                                                channels=channels,
                                                height=img_height,
                                                width=img_width)
            # print("=================fusion_feature shape {} =================".format(fusion_feature.shape))
            # 最终的x输出
            x = torch.concat((x,fusion_feature),dim=1)
            # print("=================deep_fusion_feature shape {} =================".format(x.shape))
            # 融合后是否压缩通道
            # 压缩通道
            if self.with_IP_compress_layer:
                x = self.IP_compress_layer(x)
                pass
            # print("==================input pts feat shape : {} =========================".format(x.shape))

        # 把列表里的特征装回列表,还原格式
        x = [x]    
        return x, torch.stack(valid_preds).float()

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH':
            x += (valid_preds, )
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input images. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH':
            x += (valid_preds, )
        results_list = \
            self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH':
            x += (valid_preds, )
        results = self.bbox_head.forward(x)
        return results

    def convert_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: OptInstanceList = None,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples
    
    
    
@MODELS.register_module()
class BAImVoxelNetS(Base3DDetector):
    r"""`ImVoxelNet <https://arxiv.org/abs/2106.01178>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        neck_3d (:obj:`ConfigDict` or dict): The 3D neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        prior_generator (:obj:`ConfigDict` or dict): The prior points
            generator config.
        n_voxels (list): Number of voxels along x, y, z axis.
        coord_type (str): The type of coordinates of points cloud:
            'DEPTH', 'LIDAR', or 'CAMERA'.
        train_cfg (:obj:`ConfigDict` or dict, optional): Config dict of
            training hyper-parameters. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Config dict of test
            hyper-parameters. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (:obj:`ConfigDict` or dict, optional): The initialization
            config. Defaults to None.
    """

    def __init__(self,
                #  img 特征提取流程
                 backbone: ConfigType,
                 neck: ConfigType,
                 neck_3d: ConfigType,
                 
                #  head
                 bbox_head: ConfigType,
                 prior_generator: ConfigType,
                 n_voxels: List,
                 coord_type: str,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 
                 #  数据预处理,用于处理点云和图像
                 data_preprocessor: OptConfigType = None,
                 
                 init_cfg: OptConfigType = None,
                 #  在最后新加层次
                
                 #  points 特征提取流程
                 voxel_encoder: ConfigType = None,
                 middle_encoder: ConfigType = None,
                 pts_backbone: ConfigType = None,
                 pts_neck: ConfigType = None,
                 #  通道注意力
                 img_channel_attention: ConfigType = None,
                 #  交叉注意力融合模块
                 cross_attention: ConfigType = None,   
                 # 位置编码
                 position_embedding: ConfigType = None,
                 # 输入centerpoint前的压缩层,压缩通道、聚合特征
                 compress_layer:ConfigType = None,  
                 
                 
                 ):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # 构建图像特征提取流程
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.neck_3d = MODELS.build(neck_3d)
        # 构建点云特征提取流程
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        # 构建BAFusion组件
        # build layer
        if pts_backbone:
            self.pts_backbone = MODELS.build(pts_backbone)
            print("====================get type of {} pts_backbone layer===================".format(pts_backbone["type"]))
            pass
        
        if pts_neck:
            self.pts_neck = MODELS.build(pts_neck)
            print("====================get type of {} pts_neck layer===================".format(pts_neck["type"]))
            pass
        
        if img_channel_attention:
            self.img_channel_attention = MODELS.build(img_channel_attention)
            print("====================get type of {} img_channel_attention layer===================".format(img_channel_attention["type"]))
            pass
        
        if cross_attention:
            self.cross_attention = MODELS.build(cross_attention)
            self.attention_type = cross_attention["type"]
            print("====================get type of {} cross_attention layer===================".format(cross_attention["type"]))
            pass
        
        
        if position_embedding:
            self.position_embedding = MODELS.build(position_embedding)
            print("====================get type of {} position_embedding layer===================".format(position_embedding["type"]))
            pass
        
        if compress_layer:
            self.compress_layer = MODELS.build(compress_layer)
            print("====================get type of {} compress_layer layer===================".format(compress_layer["type"]))
            pass
    
        
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.n_voxels = n_voxels
        self.coord_type = coord_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        print("=======================BAImVoxelNetS for detectors.BA_imvoxelnet is used====================")
        pass        
 
    # property是属性不是函数
    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D pts backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None
    
    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in pts branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None
    
    @property
    def with_img_channel_attention(self):
        """bool: Whether the detector has a img_channel_attention."""
        return hasattr(self,'img_channel_attention') and self.img_channel_attention is not None
    
    @property
    def with_cross_attention(self):
        """bool: Whether the detector has a IP_cross_attention."""
        return hasattr(self,'cross_attention') and self.cross_attention is not None


    @property
    def with_position_embedding(self):
        """bool: Whether the detector has a position_embedding."""
        return hasattr(self,'position_embedding') and self.position_embedding is not None
    
    @property
    def with_compress_layer(self):
        """bool: Whether the detector has a IP_compress_layer."""
        return hasattr(self,'compress_layer') and self.compress_layer is not None

            


    def encode_feature(self,featrue):
        """encode feature to transformer format

        Args:
            feature (Tensor): 2d feature map
        Returns:
            Tensor: encoded_feature:encoded 1d feature
        """
        # 然后，我们需要将 pts_BEV_feature 的二维特征图展平成一维的特征向量，并且按照 muti_head x hidden_size_head 的顺序重新排列
        # 我们可以使用 permute 和 flatten 方法来实现这一步
        # 假设 muti_head 是 16 ， hidden_size_head 是 dim // muti_head ，也就是 16
        encoded_feature = featrue.flatten(2).permute(0, 2, 1)
        # 最后，我们返回 encoded_pts_feature ，它的大小应该是 [1, 64, 256] ，其中 64 是 4 x 16 的结果，也就是每个头部的特征向量的长度
        return encoded_feature

    def remap_feature(self,fusion_feature, channels, height, width):
        """remap transformer feature 1D to 2D feature map
        Args:
            fusion_feature (Tensor): feature map fuse by transformer,shape should be (batch_size,seq_len,dim)
            which seq_len = height × width;dim = channel

        """
        # 先从encode中还原位置,第0维不动,12维交换回来
        # 这样一来,第1维变回通道数,而第二维则是height×width的特征图
        fusion_feature = fusion_feature.permute(0, 2, 1)
        # 将tansformer提取的特征从1D还原到2D
        fusion_feature = fusion_feature.reshape(-1, channels, height, width) 
        return fusion_feature


    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        -> 3d neck -> bbox_head.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Tuple:
            - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).
            - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).
        """
        img = batch_inputs_dict['imgs']
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        # print("==============ImVoxelNet input img shape:{}============".format(img.shape))
        x = self.backbone(img)
        
        # i = 0
        # for xb in x:
        #     print("==============ImVoxelNet backbone{} x shape:{}============".format(i,xb.shape))
        #     i = i + 1
            
        x = self.neck(x)[0]
        # print("==============ImVoxelNet neck0 x shape:{}============".format(x.shape))
        points = self.prior_generator.grid_anchors([self.n_voxels[::-1]],
                                                   device=img.device)[0][:, :3]
        volumes, valid_preds = [], []
        for feature, img_meta in zip(x, batch_img_metas):
            img_scale_factor = (
                points.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                points.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = points.new_tensor(
                get_proj_mat_by_coord_type(img_meta, self.coord_type))
            volume = point_sample(
                img_meta,
                img_features=feature[None, ...],
                points=points,
                proj_mat=points.new_tensor(proj_mat),
                coord_type=self.coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img.shape[-2:],
                img_shape=img_meta['img_shape'][:2],
                aligned=False)
            volumes.append(
                volume.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0))
            valid_preds.append(
                ~torch.all(volumes[-1] == 0, dim=0, keepdim=True))
        x = torch.stack(volumes)
        x = self.neck_3d(x)
        
        # i = 0
        # for xb in x:
        #     print("==============ImVoxelNet neck_3d0 x shape:{}============".format(xb.shape))
        #     i = i + 1
        # 到这里x是提取出的neck特征,通道数为256,x为一个list,里面只装有一个特征
        # 取出列表里的特征
        x = x[0]
        # print("==============ImVoxelNet neck_3d0 x shape:{}============".format(x.shape))
        # 取得点云体素
        voxel_dict = batch_inputs_dict['voxels']
        # 体素化:voxel_num × voxel_dim
        # 在每个体素内提取特征
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        # 进一步编码点云特征 feature
        pts_feature = self.middle_encoder(voxel_features, 
                                          voxel_dict['coors'],batch_size)
        # 送入pts主干提取点云特征
        pts_feature = self.pts_backbone(pts_feature)
        # 进一步提取点云特征
        if self.with_pts_neck:
            pts_feature = self.pts_neck(pts_feature)
        # seconde fpn提取结果是一个列表
        # 里面只装了一个最后汇总输入head的向量,因此可以直接提出 
        pts_feature = pts_feature[0]
        # print("==============pts_feature neck shape:{}============".format(pts_feature.shape))
        # img特征的shape
        img_height = x.shape[2]
        img_width = x.shape[3]
        # channels大家都一样就不区分了
        channels = x.shape[1]
        batch_size = x.shape[0]
        
        # pts特征的shape
        pts_height = pts_feature.shape[2]
        pts_width = pts_feature.shape[3]
        # 交叉注意力融合,需要交叉注意力层才能BiAttention
        if self.with_cross_attention:
            # 将图像特征编码到1D
            encoded_img_feature = self.encode_feature(x)
            # 将点云特征编码到1D            
            encoded_pts_feature = self.encode_feature(pts_feature)
            
            # print("=================encoded_img_feats shape {} =================".format(encoded_img_feature.shape))
            # print("=================encoded_pts_feat shape {} =================".format(encoded_pts_feature.shape))
            if self.attention_type == "SoftmaxCrossAttention" or "CrossFocusedLinearAttention":
                # FLA系列，输入顺序为query, key, value, H, W
                # img增强的pts特征同img进一步融合
                fusion_feature = self.cross_attention(encoded_img_feature,
                                                         encoded_pts_feature,
                                                         encoded_pts_feature,
                                                         H=pts_height,
                                                         W=pts_width)
            else:
                raise ValueError("self.attention_type must be SoftmaxCrossAttention or CrossFocusedLinearAttention!")
            # print("=================transformer fusion_feature shape {} =================".format(fusion_feature.shape))
            # # 将tansformer提取的特征从1D还原到2D
            fusion_feature = self.remap_feature(fusion_feature,
                                                channels=channels,
                                                height=img_height,
                                                width=img_width)
            # print("=================fusion_feature shape {} =================".format(fusion_feature.shape))
            # 最终的x输出
            x = torch.concat((x,fusion_feature),dim=1)
            # print("=================deep_fusion_feature shape {} =================".format(x.shape))
            # 融合后是否压缩通道
            # 压缩通道
            if self.with_compress_layer:
                x = self.compress_layer(x)
                pass
            # print("==================input pts feat shape : {} =========================".format(x.shape))

        # 把列表里的特征装回列表,还原格式
        x = [x]    
        return x, torch.stack(valid_preds).float()

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH':
            x += (valid_preds, )
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input images. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH':
            x += (valid_preds, )
        results_list = \
            self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH':
            x += (valid_preds, )
        results = self.bbox_head.forward(x)
        return results

    def convert_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: OptInstanceList = None,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples    
    
    
    
    

    
