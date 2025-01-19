# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple,List
import torch
from torch import Tensor
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStage3DDetector
from ..layers.fusion_layers.position_embedding import NestedTensor
"""
之前的voxel net是在voxel encoder后融合,这次改成late fusion,在pts fpn之后融合
"""
# Bi-Attention VoxelNet
# 双向线性交叉注意、late fusion特征融合
# 剪枝特供版，去除所有选择分支，简化所有操作方便计算图识别
@MODELS.register_module("BAVoxelNetPrune")
class BAVoxelNetPrune(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                voxel_encoder: ConfigType,
                middle_encoder: ConfigType,
                backbone: ConfigType,
                neck: OptConfigType = None,
                bbox_head: OptConfigType = None,
                train_cfg: OptConfigType = None,
                test_cfg: OptConfigType = None,
                data_preprocessor: OptConfigType = None,
                init_cfg: OptMultiConfig = None,
                img_backbone: ConfigType = None,
                img_neck: ConfigType = None,
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
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        # build layer
        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
            print("====================get type of {} img_backbone layer===================".format(img_backbone["type"]))
            pass
        
        if img_neck:
            self.img_neck = MODELS.build(img_neck)
            print("====================get type of {} img_neck layer===================".format(img_neck["type"]))
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
        
        print("=======================BAFusionVoxelNet for detectors.BA_voxelnet is used====================")
        pass
    
    # property是属性不是函数
    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None
    
    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None
    
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
    
    # 重写img feat的提取过程,将backbone neck流程拆解,方便在backbone和neck之间加入通道注意力
    # 首先从backbone中提取
    def extract_img_feat_backbone(self, img: Tensor, input_metas: List[dict]) -> dict:
        # print("=======extract_img_feat_backbone input : {}=======".format(img.shape))
        if self.with_img_backbone and img is not None:
            # 取得Img的高和宽,依次为height和width
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            # for img_meta in input_metas:
            #     img_meta.update(input_shape=input_shape)
            # batch number channel height width
            # dim == 5是因为可能有多图像的输入,kitti中只有一张
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            # 处理多图像的输入,方法是将Batch和number统一为batch = B×N
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            # input dim = 4
            img_feats = self.img_backbone(img)
        else:
            return None
        # 注意,backbone中得到的是一个tuple
        # 这是因为通常会输出多尺度的特征到FPN中进行融合
        # 每个元组中,装着一个B C H W
        # 每经过一个block,通道数加倍,特征图大小减半
        # print("=======extract_img_feat_backbone type: {}=======".format(type(img_feats)))
        # for i in range(len(img_feats)):
        #     print("=======extract_img_feat_backbone output{} : {}=======".format(i,img_feats[i].shape))
        return img_feats        
    
    # 从neck中进一步提取特征
    def extract_img_feat_neck(self,img_feats: Tuple[Tensor]) -> dict:
        # 从backbone和channel attention中接收的多尺度特征
        # 如果没有img_neck,就什么也不做,这样返回值一定是backbone的返回值
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            # B * N, C, H, W; dim == 4
        # 这里的img_feats依然是一个元组 tuple
        # print("=======extract_img_feat_neck type: {}=======".format(type(img_feats)))
        # if img_feats is not None:
            # for i in range(len(img_feats)):
            #     print("=======extract_img_feat_neck output{} : {}=======".format(i,img_feats[i].shape))
        return img_feats
    

    def encode_image(self,image_feature_tuple:tuple):
        """encode image feature to transformer format

        Args:
            image_feature_tuple (tuple): 
        Returns:
            Tensor: concat encoded_images
        """
        encoded_images_list = []
        for image_f in image_feature_tuple:
            image_f = image_f.flatten(2).permute(0, 2, 1) 
            encoded_images_list.append(image_f)
        encoded_images = torch.concat(encoded_images_list,dim=1)
        return encoded_images

    def encode_pts(self,pts_BEV_feature):
        """encode points bev feature to transformer format

        Args:
            pts_BEV_feature (Tensor): 2d feature map of bev
        Returns:
            Tensor: encoded_pts_feature:encoded 1d feature
        """
        # 然后，我们需要将 pts_BEV_feature 的二维特征图展平成一维的特征向量，并且按照 muti_head x hidden_size_head 的顺序重新排列
        # 我们可以使用 permute 和 flatten 方法来实现这一步
        # 假设 muti_head 是 16 ， hidden_size_head 是 dim // muti_head ，也就是 16
        encoded_pts_feature = pts_BEV_feature.flatten(2).permute(0, 2, 1)
        # 最后，我们返回 encoded_pts_feature ，它的大小应该是 [1, 64, 256] ，其中 64 是 4 x 16 的结果，也就是每个头部的特征向量的长度
        return encoded_pts_feature
    
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
    
    # 对图像特征进行位置编码 
    def position_encode(self,img_feats,batch_size):
        # 没有位置编码直接返回特征
        if not self.with_position_embedding:
            # print("==========================No image Position Encode=======================")
            return img_feats
        
        pe_img_feats = []
        # 为每一个img_feat计算掩码:
        for img_feat in img_feats:
            # 为img_feat生成掩码来计算pos emb
            # mask全0代表所有位置都被使用
            mask = torch.zeros((batch_size, *img_feat.shape[-2:]), dtype=torch.bool)
            tensor_list = NestedTensor(img_feat, mask) # 将图像特征和掩码封装成一个NestedTensor对象
            pos = self.position_embedding(tensor_list)
            # print("=================pos shape {} =================".format(pos.shape))
            img_feat = img_feat + pos
            # print("=================pos__img_feats shape {} =================".format(img_feat.shape))
            pe_img_feats.append(img_feat)
            pass
        return tuple(pe_img_feats)
    
    # 对点云特征进行位置编码 
    # 点云特征是单尺度的无元组,因此重写一个函数
    def pts_position_encode(self,pts_feat,batch_size):
        # 没有就位置编码直接返回特征
        if not self.with_position_embedding:
            # print("==========================No points Position Encode=======================")
            return pts_feat

        # 为img_feat生成掩码来计算pos emb
        # mask全0代表所有位置都被使用
        # shape得到最后两个值是w h
        mask = torch.zeros((batch_size, *pts_feat.shape[-2:]), dtype=torch.bool)
        tensor_list = NestedTensor(pts_feat, mask) # 将pts特征和掩码封装成一个NestedTensor对象
        pos = self.position_embedding(tensor_list)
        # print("=================pts pos shape {} =================".format(pos.shape))
        # 原特征与位置编码按位置相加
        pe_pts_feat = pts_feat + pos
        # print("=================pos_pts_feat shape {} =================".format(pe_pts_feat.shape))
        return pe_pts_feat

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features from points."""
        # print("======================= Begin BA voxelnet extract feat! ==========================")

        voxel_dict = batch_inputs_dict['voxels']
        # 体素化:voxel_num × voxel_dim
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        # 提取voxel feature
        x = self.middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        # 送入pts主干提取点云特征
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        # seconde fpn提取结果是一个列表
        # 里面只装了一个最后汇总输入head的向量,因此可以直接提出 
        x = x[0]
        # print("=====================second fpn x shape : {}========================".format(x.shape))
        # 如果有图像分支就进入图像分支

        # 从inputs中获取输入图像
        imgs = batch_inputs_dict.get('imgs', None)
        # voxel特征的shape
        height = x.shape[2]
        width = x.shape[3]
        channels = x.shape[1]
        batch_size = x.shape[0]
        # 提取图像特征
        img_feats_backbone = self.extract_img_feat_backbone(imgs,batch_inputs_dict)
        # # 图像特征的HW,注意,这里只有第一层次的图像
        # H = img_feats_backbone[0].shape[2]
        # W = img_feats_backbone[0].shape[3]
        # 通道注意力暂时舍弃
        if self.with_img_channel_attention:
            img_feats_backbone = self.img_channel_attention(x,img_feats_backbone)
            pass
        img_feats = self.extract_img_feat_neck(img_feats_backbone)

        # print("=====================img fpn x shape : {}========================".format(img_feats[0].shape))

        # 图像特征的HW,注意,这里只有一种尺度
        H = img_feats[0].shape[2]
        W = img_feats[0].shape[3]

        qkv_img_feats = self.encode_image(img_feats)
        # 同样区分pts feat的key 和 value
        # key同时作为query进行交互

        qkv_pts_feat = self.encode_pts(x)
        
        # 编码后的图像特征实际上也丢弃了元组而作为单尺度特征
        # 这是因为多尺度特征会被concat到单尺度作为综合的token传入自注意力层
        # print("=================encoded_img_feats shape {} =================".format(qkv_img_feats.shape))
        # print("=================kv_pts_feat shape {} =================".format(qkv_pts_feat.shape))

        assert self.attention_type in ["SoftmaxCrossAttention","CrossFocusedLinearAttention","AgentCrossAttention","CrossFocusedLinearAttentionPrune"]
        # FA系列，输入顺序为query, key, value, H, W
        # H和W指的是key-value对的height和width,不过实际上,这个值只是保证attention层正常得到尺度
        # IP特征,即用point增强图像,key图像作为query,点云作为k-v对

        # IP_feature = self.IP_cross_attention(qkv_img_feats,qkv_pts_feat,qkv_pts_feat,H=height,W=width)
        IP_feature = self.IP_cross_attention(qkv_img_feats,qkv_pts_feat,qkv_pts_feat,H=H,W=W)


        # print("=================IP_feature shape {} =================".format(IP_feature.shape))
        # remap到2D与原始图像特征concat并经过卷积层聚合
        # H和W是图像的宽高,而height和width是点云的宽高
        IP_feature2D = self.remap_feature(IP_feature,channels=channels,height=H,width=W)
        # print("=================IP_feature2D shape {} =================".format(IP_feature2D.shape))

        # 因为暂时只有一层,先这么写着
        # IP_feature2D = torch.concat((IP_feature2D,img_feats[0]),dim=1)
        # 试试add
        IP_feature2D = IP_feature2D + img_feats[0]


        # print("=================concated IP_feature2D shape {} =================".format(IP_feature2D.shape))
        # IP特征聚合
        compressed_IP_feature2D = self.IP_compress_layer(IP_feature2D)
        # print("=================compressed_IP_feature2D shape {} =================".format(compressed_IP_feature2D.shape))

        qkv_IP_featrue = self.encode_pts(compressed_IP_feature2D)

        # print("=================key_IP_feature shape {} =================".format(qkv_IP_featrue.shape))
        # 点云增强的图像特征同点云进一步融合
        # key作query,因为没有添加位置编码
        # fusion_feature = self.PI_cross_attention(qkv_pts_feat,qkv_IP_featrue,qkv_IP_featrue,H=H,W=W)
        fusion_feature = self.PI_cross_attention(qkv_pts_feat,qkv_IP_featrue,qkv_IP_featrue,H=height,W=width)
        # print("=================transformer fusion_feature shape {} =================".format(fusion_feature.shape))
        # # 将tansformer提取的特征从1D还原到2D
        # fusion_feature = fusion_feature.reshape(-1, height, width, channels) # 或者 output = output.view(1, 32, 32, 256)
        # # batch channel height width
        # fusion_feature = fusion_feature.permute(0, 3, 1, 2)
        fusion_feature = self.remap_feature(fusion_feature,channels=channels,height=height,width=width)
        # print("=================fusion_feature shape {} =================".format(fusion_feature.shape))
        # 最终的x输出
        # x = torch.concat((x,fusion_feature),dim=1)
        # 试试add
        x = x+fusion_feature

        # 压缩通道
        x = self.PI_compress_layer(x)
        # print("==================input pts feat shape : {} =========================".format(x.shape))
        # 图像特征融合到这里结束
        # SECOND FPN的输出是一个列表,事实上只有一个向量,
        # 因此,返回值输出的时候别忘了装回列表里    
        x = [x]     
        # print("======================= finish BA voxelnet extract feat! ==========================")
        return x