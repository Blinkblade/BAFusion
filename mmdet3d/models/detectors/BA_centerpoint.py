# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import torch
from torch import Tensor
from ..layers.fusion_layers.position_embedding import NestedTensor
from mmdet3d.registry import MODELS
from .mvx_two_stage import MVXTwoStageDetector
from typing import Tuple,List

@MODELS.register_module("BACenterPoint")
class BACenterPoint(MVXTwoStageDetector):
    """Base class of Multi-modality BACenterPoint.

    Args:
        pts_voxel_encoder (dict, optional): Point voxelization
            encoder layer. Defaults to None.
        pts_middle_encoder (dict, optional): Middle encoder layer
            of points cloud modality. Defaults to None.
        pts_fusion_layer (dict, optional): Fusion layer.
            Defaults to None.
        img_backbone (dict, optional): Backbone of extracting
            images feature. Defaults to None.
        pts_backbone (dict, optional): Backbone of extracting
            points features. Defaults to None.
        img_neck (dict, optional): Neck of extracting
            image features. Defaults to None.
        pts_neck (dict, optional): Neck of extracting
            points features. Defaults to None.
        pts_bbox_head (dict, optional): Bboxes head of
            point cloud modality. Defaults to None.
        img_roi_head (dict, optional): RoI head of image
            modality. Defaults to None.
        img_rpn_head (dict, optional): RPN head of image
            modality. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
    """


    def __init__(self,
                pts_voxel_encoder: Optional[dict] = None,
                pts_middle_encoder: Optional[dict] = None,
                pts_fusion_layer: Optional[dict] = None,
                img_backbone: Optional[dict] = None,
                pts_backbone: Optional[dict] = None,
                img_neck: Optional[dict] = None,
                pts_neck: Optional[dict] = None,
                pts_bbox_head: Optional[dict] = None,
                img_roi_head: Optional[dict] = None,
                img_rpn_head: Optional[dict] = None,
                train_cfg: Optional[dict] = None,
                test_cfg: Optional[dict] = None,
                init_cfg: Optional[dict] = None,
                data_preprocessor: Optional[dict] = None,
                #  通道注意力
                img_channel_attention: Optional[dict] = None,
                #  交叉注意力融合模块
                # image-point融合模块
                IP_cross_attention: Optional[dict] = None, 
                # Point-image融合模块  
                PI_cross_attention: Optional[dict] = None, 
                # 位置编码
                position_embedding: Optional[dict] = None,
                # 输入主干网络前的压缩层,压缩通道、聚合特征
                # image-point聚合
                IP_compress_layer:Optional[dict] = None,   
                # point-image聚合
                PI_compress_layer:Optional[dict] = None,  

                # 多视角通道融合
                MV_compress_layer:Optional[dict] = None,  
                 **kwargs):

        # super(BACenterPoint,
        #       self).__init__(pts_voxel_encoder, pts_middle_encoder,
        #                      pts_fusion_layer, img_backbone, pts_backbone,
        #                      img_neck, pts_neck, pts_bbox_head, img_roi_head,
        #                      img_rpn_head, train_cfg, test_cfg, init_cfg,
        #                      data_preprocessor, **kwargs)
        super().__init__(pts_voxel_encoder, pts_middle_encoder,
                             pts_fusion_layer, img_backbone, pts_backbone,
                             img_neck, pts_neck, pts_bbox_head, img_roi_head,
                             img_rpn_head, train_cfg, test_cfg, init_cfg,
                             data_preprocessor, **kwargs)
        # 构建网络模块
        if pts_voxel_encoder:
            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
            print("====================get type of {} pts_voxel_encoder layer===================".format(pts_voxel_encoder["type"]))
        if pts_middle_encoder:
            self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
            print("====================get type of {} pts_middle_encoder layer===================".format(pts_middle_encoder["type"]))
        if pts_backbone:
            self.pts_backbone = MODELS.build(pts_backbone)
            print("====================get type of {} pts_backbone layer===================".format(pts_backbone["type"]))
        if pts_fusion_layer:
            self.pts_fusion_layer = MODELS.build(pts_fusion_layer)
            print("====================get type of {} pts_fusion_layer layer===================".format(pts_fusion_layer["type"]))
        if pts_neck is not None:
            self.pts_neck = MODELS.build(pts_neck)
            print("====================get type of {} pts_neck layer===================".format(pts_neck["type"]))
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = MODELS.build(pts_bbox_head)
            print("====================get type of {} pts_bbox_head layer===================".format(pts_bbox_head["type"]))

        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
            print("====================get type of {} img_backbone layer===================".format(img_backbone["type"]))
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
            print("====================get type of {} img_neck layer===================".format(img_neck["type"]))
        if img_rpn_head is not None:
            self.img_rpn_head = MODELS.build(img_rpn_head)
            print("====================get type of {} img_rpn_head layer===================".format(img_rpn_head["type"]))
        if img_roi_head is not None:
            self.img_roi_head = MODELS.build(img_roi_head)
            print("====================get type of {} img_roi_head layer===================".format(img_roi_head["type"]))
        
            
        # 构建融合模块

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

        if MV_compress_layer:
            self.MV_compress_layer = MODELS.build(MV_compress_layer)
            print("====================get type of {} MV_compress_layer layer===================".format(MV_compress_layer["type"]))
            pass

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        print("=======================BAFusionCenterPoint for detectors.BA_CenterPoint is used====================")
        pass

    # 标识是否拥有这些结构
    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self,
                       'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self,
                       'middle_encoder') and self.middle_encoder is not None

    # property是属性不是函数
    # @property
    # def with_img_backbone(self):
    #     """bool: Whether the detector has a 2D image backbone."""
    #     return hasattr(self, 'img_backbone') and self.img_backbone is not None
    
    # @property
    # def with_img_neck(self):
    #     """bool: Whether the detector has a neck in image branch."""
    #     return hasattr(self, 'img_neck') and self.img_neck is not None
    
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
    
    @property
    def with_MV_compress_layer(self):
        """bool: Whether the detector has a MV_compress_layer."""
        return hasattr(self,'MV_compress_layer') and self.MV_compress_layer is not None

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

    # nus多视角图像分别提取，因此需要合并保持通道一致
    # 这个方法通过reshape 到(batch_size,n * channel,w,h)使用卷积降维实现
    def MV_CONV_Compress(self,img_feats, batch_size, H, W):
        img_feature = img_feats[0]
        # 将batchsize还原,n到channel
        img_feature = img_feature.reshape(batch_size,-1,H,W)
        # 降维
        img_feature = self.MV_compress_layer(img_feature)
        # 放到元组里
        return tuple([img_feature])

    # 这个方法通过reshape 到(batch_size,n * channel,w,h)使用相加平均实现降维
    def MV_AVG_Compress(self,img_feats, batch_size, H, W):
        # 假设你的图像特征是一个(batch_size * n, channel, H, W)大小的张量，装在一个元组里

        # 从元组中取出这个张量
        img_feats = img_feats[0]
        # 将batchsize还原,n到channel
        img_feature = img_feature.reshape(batch_size,-1,H,W)

        # 沿着第二个维度求和，得到一个(batch_size, 1, H, W)大小的特征图
        img_feats = torch.sum(img_feats, dim=1, keepdim=True)
        img_feats = img_feats / 6
        # 返回特征图
        return img_feats



    # def extract_feat(self, batch_inputs_dict: dict,
    #                  batch_input_metas: List[dict]) -> tuple:


    def extract_feat(self,
                    batch_inputs_dict: dict,
                    batch_input_metas: List[dict]) -> Tuple[Tensor]:
        """Extract features from points."""
        # print("============extract feat input keys {}===============".format(batch_inputs_dict.keys()))
        # voxel_dict = batch_inputs_dict['voxels']
        voxel_dict = batch_inputs_dict.get('voxels', None)
        # 体素化:voxel_num × voxel_dim
        # voxel_features = self.voxel_encoder(voxel_dict['voxels'],
        #                                     voxel_dict['num_points'],
        #                                     voxel_dict['coors'])
        # 体素化
        # voxel_features.shape is voxel_num × point_dims
        img_feats = None
        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'], img_feats,
                                                batch_input_metas)
        batch_size = voxel_dict['coors'][-1, 0].item() + 1


        # voxel_bev_feature = self.extract_voxel_BEV_feat(voxel_features,
        #                                                 voxel_dict['coors'])
        # # batch_size = voxel_coors[-1, 0] + 1
        # voxel_bev_features = self.pts_middle_encoder(voxel_features, voxel_coors,batch_size)    

        # 提取voxel feature
        x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        # 送入pts主干提取点云特征
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        # seconde fpn提取结果是一个列表
        # 里面只装了一个最后汇总输入head的向量,因此可以直接提出 
        x = x[0]
        # print("=====================second fpn x shape : {}========================".format(x.shape))
        # 如果有图像分支就进入图像分支
        if self.with_img_backbone:
            # 从inputs中获取输入图像
            imgs = batch_inputs_dict.get('imgs', None)
            # print("=====================imgs from inputs dict shape : {}========================".format(imgs.shape))
            # voxel特征的shape, 小写的形状代表 点云的shape
            height = x.shape[2]
            width = x.shape[3]
            channels = x.shape[1]
            batch_size = x.shape[0]
            # 提取图像特征
            # img_feats_backbone = self.extract_img_feat_backbone(imgs,batch_inputs_dict)
            img_feats_backbone = self.extract_img_feat_backbone(imgs,batch_input_metas)


            # 通道注意力暂时舍弃
            if self.with_img_channel_attention:
                img_feats_backbone = self.img_channel_attention(x,img_feats_backbone)
                pass
        
            img_feats = self.extract_img_feat_neck(img_feats_backbone)
            # print("=====================img_feats neck shape : {}========================".format(img_feats[0].shape))
            # 图像特征的HW,注意,这里只有一种尺度
            H = img_feats[0].shape[2]
            W = img_feats[0].shape[3]            
            img_feats = self.MV_CONV_Compress(img_feats=img_feats,batch_size=batch_size,H=H,W=W)

            # 交叉注意力融合,需要二者都有才能BiAttention
            if self.with_IP_cross_attention and self.with_PI_cross_attention:
                # 如果使用位置编码则添加位置编码
                # if self.with_position_embedding:
                # 为每一个img_feat计算掩码:
                # 添加掩码后的img_feat是key
                # 未添加掩码的img_feat是value
                pe_img_feats = self.position_encode(img_feats=img_feats,batch_size=batch_size)
                # point也要融合,因此给points也添加位置编码
                pe_pts_feat = self.pts_position_encode(pts_feat=x,batch_size=batch_size)
                # 先编码2D特征为1D
                # 添加位置编码后的img_feat是key,用于进行交互计算注意力
                # 未添加位置编码的img_feat是value,与注意力相乘并进一步相加
                # value_img_feats = self.encode_image(img_feats)
                key_img_feats = self.encode_image(pe_img_feats)
                # 同样区分pts feat的key 和 value
                # key同时作为query进行交互
                # encoded_pts_feat = self.encode_pts(x)
                # value_pts_feat = self.encode_pts(pe_pts_feat)
                # key_pts_feat = self.encode_pts(x)
                value_pts_feat = self.encode_pts(x)
                key_pts_feat = self.encode_pts(pe_pts_feat)
                
                
                # 编码后的图像特征实际上也丢弃了元组而作为单尺度特征
                # 这是因为多尺度特征会被concat到单尺度作为综合的token传入自注意力层
                # print("=================encoded_img_feats shape {} =================".format(key_img_feats.shape))
                # print("=================encoded_pts_feat shape {} =================".format(key_pts_feat.shape))
                # if self.attention_type == "SoftmaxCrossAttention" or "CrossFocusedLinearAttention":
                if self.attention_type in ["SoftmaxCrossAttention","CrossFocusedLinearAttention","AgentCrossAttention"]:
                    # FA系列，输入顺序为query, key, value, H, W
                    # H和W指的是key-value对的height和width,不过实际上,这个值只是保证attention层正常得到尺度
                    # IP特征,即用point增强图像,key图像作为query,点云作为k-v对
                    
                    IP_feature = self.IP_cross_attention(key_img_feats,key_pts_feat,value_pts_feat,H=height,W=width)
                    # print("=================IP_feature shape {} =================".format(IP_feature.shape))
                    # remap到2D与原始图像特征concat并经过卷积层聚合
                    # H和W是图像的宽高,而height和width是点云的宽高
                    IP_feature2D = self.remap_feature(IP_feature,channels=channels,height=H,width=W)
                    # print("=================IP_feature2D shape {} =================".format(IP_feature2D.shape))
                    # 因为暂时只有一层,先这么写着
                    IP_feature2D = torch.concat((IP_feature2D,img_feats[0]),dim=1)
                    # print("=================concated IP_feature2D shape {} =================".format(IP_feature2D.shape))
                    # IP特征聚合
                    compressed_IP_feature2D = self.IP_compress_layer(IP_feature2D)
                    # print("=================compressed_IP_feature2D shape {} =================".format(compressed_IP_feature2D.shape))
                    # 重新编码聚合IP特征,首先添加位置编码然后再次编码为一维
                    # IP特征形式与pts特征类似,都采用pts的函数
                    pe_compressed_IP_feature2D = self.pts_position_encode(pts_feat=compressed_IP_feature2D,
                                                                          batch_size=batch_size)
                    # 同样,key是带位置编码的,value是不带位置编码的
                    key_IP_feature = self.encode_pts(pe_compressed_IP_feature2D)
                    value_IP_featrue = self.encode_pts(compressed_IP_feature2D)


                    # print("=================key_IP_feature shape {} =================".format(key_IP_feature.shape))
                    # 点云增强的图像特征同点云进一步融合
                    # key作query,因为没有添加位置编码
                    fusion_feature = self.PI_cross_attention(key_pts_feat,key_IP_feature,value_IP_featrue,H=H,W=W)
                    pass
                else:
                    # fusion_feature = self.cross_attention(value_img_feats,key_img_feats,encoded_pts_feat)
                    raise ValueError("self.attention_type must be SoftmaxCrossAttention or CrossFocusedLinearAttention or AgentCrossAttention!")

                # print("=================transformer fusion_feature shape {} =================".format(fusion_feature.shape))
                # # 将tansformer提取的特征从1D还原到2D
                # fusion_feature = fusion_feature.reshape(-1, height, width, channels) # 或者 output = output.view(1, 32, 32, 256)
                # # batch channel height width
                # fusion_feature = fusion_feature.permute(0, 3, 1, 2)
                fusion_feature = self.remap_feature(fusion_feature,channels=channels,height=height,width=width)
                # print("=================fusion_feature shape {} =================".format(fusion_feature.shape))
                # 最终的x输出
                x = torch.concat((x,fusion_feature),dim=1)
                # print("=================deep_fusion_feature shape {} =================".format(x.shape))
                # 融合后是否压缩通道
                # 压缩通道
                if self.with_PI_compress_layer:
                    x = self.PI_compress_layer(x)
                    pass
                # print("==================input pts feat shape : {} =========================".format(x.shape))
                pass#交叉注意里融合到这里结束
            # 图像特征融合到这里结束
            pass
        
        else:
            print("***=======================There is no cross-attention fusion=======================***")
        
        # SECOND FPN的输出是一个列表,事实上只有一个向量,
        # 因此,返回值输出的时候别忘了装回列表里    
        x = [x]    
        # 不需要img feat,融合在特征提取中已经完成了 
        return (None, x)