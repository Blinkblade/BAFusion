# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
# from ..layers import CSPLayer
from mmdet.models.layers.csp_layer import CSPLayer

# class DarknetBottleneck(BaseModule):
#     """The basic bottleneck block used in Darknet.
#     Each ResBlock consists of two ConvModules and the input is added to the
#     final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
#     The first convLayer has filter size of 1x1 and the second one has the
#     filter size of 3x3.
#     Args:
#         in_channels (int): The input channels of this Module.
#         out_channels (int): The output channels of this Module.
#         expansion (int): The kernel size of the convolution. Default: 0.5
#         add_identity (bool): Whether to add identity to the out.
#             Default: True
#         use_depthwise (bool): Whether to use depthwise separable convolution.
#             Default: False
#         conv_cfg (dict): Config dict for convolution layer. Default: None,
#             which means using conv2d.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: dict(type='BN').
#         act_cfg (dict): Config dict for activation layer.
#             Default: dict(type='Swish').
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  expansion=0.5,
#                  add_identity=True,
#                  use_depthwise=False,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
#                  act_cfg=dict(type='Swish'),
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         hidden_channels = int(out_channels * expansion)
#         conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
#         self.conv1 = ConvModule(
#             in_channels,
#             hidden_channels,
#             1,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg)
#         self.conv2 = conv(
#             hidden_channels,
#             out_channels,
#             3,
#             stride=1,
#             padding=1,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg)
#         self.add_identity = \
#             add_identity and in_channels == out_channels

#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.conv2(out)

#         if self.add_identity:
#             return out + identity
#         else:
#             return out


# class CSPLayer(BaseModule):
#     """Cross Stage Partial Layer.
#     Args:
#         in_channels (int): The input channels of the CSP layer.
#         out_channels (int): The output channels of the CSP layer.
#         expand_ratio (float): Ratio to adjust the number of channels of the
#             hidden layer. Default: 0.5
#         num_blocks (int): Number of blocks. Default: 1
#         add_identity (bool): Whether to add identity in blocks.
#             Default: True
#         use_depthwise (bool): Whether to depthwise separable convolution in
#             blocks. Default: False
#         conv_cfg (dict, optional): Config dict for convolution layer.
#             Default: None, which means using conv2d.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: dict(type='BN')
#         act_cfg (dict): Config dict for activation layer.
#             Default: dict(type='Swish')
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  expand_ratio=0.5,
#                  num_blocks=1,
#                  add_identity=True,
#                  use_depthwise=False,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
#                  act_cfg=dict(type='Swish'),
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         mid_channels = int(out_channels * expand_ratio)
#         self.main_conv = ConvModule(
#             in_channels,
#             mid_channels,
#             1,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg)
#         self.short_conv = ConvModule(
#             in_channels,
#             mid_channels,
#             1,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg)
#         self.final_conv = ConvModule(
#             2 * mid_channels,
#             out_channels,
#             1,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg)

#         self.blocks = nn.Sequential(*[
#             DarknetBottleneck(
#                 mid_channels,
#                 mid_channels,
#                 1.0,
#                 add_identity,
#                 use_depthwise,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg) for _ in range(num_blocks)
#         ])

#     def forward(self, x):
#         x_short = self.short_conv(x)

#         x_main = self.main_conv(x)
#         x_main = self.blocks(x_main)

#         x_final = torch.cat((x_main, x_short), dim=1)
#         return self.final_conv(x_final)



# @MODELS.register_module()
class YOLOXPAFPN(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(YOLOXPAFPN, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)
    
    
    


# 只输出index = 1的out feature,这一层是既经过top-down又经过bottom-up的中间特征
# 同时具有较大特征图,高分辨率保证注意力选择的丰富性
@MODELS.register_module()
class YOLOXPAFPN1(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(YOLOXPAFPN1, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        # downsamples和Bottom-up都只需要第一层
        # self.downsamples = nn.ModuleList()
        # self.bottom_up_blocks = nn.ModuleList()
        # for idx in range(len(in_channels) - 1):
        #     self.downsamples.append(
        #         conv(
        #             in_channels[idx],
        #             in_channels[idx],
        #             3,
        #             stride=2,
        #             padding=1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg))
        #     self.bottom_up_blocks.append(
        #         CSPLayer(
        #             in_channels[idx] * 2,
        #             in_channels[idx + 1],
        #             num_blocks=num_csp_blocks,
        #             add_identity=False,
        #             use_depthwise=use_depthwise,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg))
        # 只建立index1的特征输出conv,节省计算量 
        self.downsample1 = conv(in_channels[0],in_channels[0],3,stride=2,padding=1,
                                conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.bottom_up_block1 = CSPLayer(in_channels[0] * 2,in_channels[1],num_blocks=num_csp_blocks,
                                         add_identity=False,use_depthwise=use_depthwise,
                                         conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg)
        # self.out_convs = nn.ModuleList()
        # for i in range(len(in_channels)):
        #     self.out_convs.append(
        #         ConvModule(
        #             in_channels[i],
        #             out_channels,
        #             1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg))
        # 只建立index1的特征输出conv,节省计算量 
        self.out_conv1 = ConvModule(in_channels[1],out_channels,1,
                                    conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg)

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        # top-down ==> 自上而下融合
        # top指最高级别的语义特征,特征图最小,语义最丰富
        inner_outs = [inputs[-1]]
        # 倒着采样,先从倒数第二小的特征图读取
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # heigh和low是和当前特征图的level比较
            # heigh ==> map更小,语义更高
            # 因为上一层特征图聚合完后会被装入Innerout,所以总是从inneouts0取数
            feat_heigh = inner_outs[0]
            # low就是自己的前一层
            feat_low = inputs[idx - 1]
            # reduce layer是1×1卷积层,目的是压缩通道数
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh
            # height feature 二倍上采样到和当前feature一样
            upsample_feat = self.upsample(feat_heigh)
            # 不同级别的特征concat到一起然后过一层卷积
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            # 当前特征插入到inner outs index 0
            # 这样一来inner outs中的顺序依然是bottom-top
            inner_outs.insert(0, inner_out)

        # bottom-up ==> 自底向上融合
        # bottom指的是最低级别的语义特征,特征图最大,粒度最粗
        # bottom-up path
        # 首先录入bottom特征,即map最大的特征
        # 需要保留列表形式来进行元组化
        outs = []
        # for idx in range(len(self.in_channels) - 1):
        #     # low和heigh依然是相对于当前特征来看的
        #     # fetas low是最新加入到outs中的特征,语义比当前特征低一级,map更大
        #     feat_low = outs[-1]
        #     # feats height是当前特征
        #     # 如idx = 0,feat_heigh就是inner outs[1],
        #     # inner outs[1] 和 inner outs[0]融合
        #     feat_height = inner_outs[idx + 1]
        #     # 对low降采样
        #     downsample_feat = self.downsamples[idx](feat_low)
        #     out = self.bottom_up_blocks[idx](
        #         torch.cat([downsample_feat, feat_height], 1))
        #     outs.append(out)
        # 我们只需要index为1的特征输出,因此bottom-up这一层只需要做一次
        # 即0-->1的bottom up
        # 最bottom的特征就是index0
        feat_low = inner_outs[0]
        feat_height = inner_outs[1]
        # 下采样和bottom up 下标比inner outs小1
        downsample_feat = self.downsample1(feat_low)
        out = self.bottom_up_block1(
            torch.cat([downsample_feat, feat_height], 1))    
        
        # out convs
        # for idx, conv in enumerate(self.out_convs):
        #     outs[idx] = conv(outs[idx])
        # 过一层卷积
        out = self.out_conv1(out)
        # 先装入列表再转换为元组
        outs.append(out)
        return tuple(outs)
