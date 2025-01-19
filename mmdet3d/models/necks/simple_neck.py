# 只使用backbone提取的分辨率最高的特征图
import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
import os
from mmengine.model import BaseModule
import math
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 自动计算出一个使特征图大小保持不变的padding值
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# 只用backbone的第一层分辨率最高的特征
# 将通道统一
@MODELS.register_module('SimpleNeck')
class SimpleNeck(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
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
                 in_channel=256,
                 out_channel=512,
                 kernel_size=1,
                 use_depthwise=False,
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
        super(SimpleNeck, self).__init__(init_cfg)
        self.in_channel = in_channel
        self.out_channel = out_channel

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # 只使用一个卷积层来改变通道维度 
        self.conv = conv(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=autopad(k=kernel_size),
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
            )


    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        # 输入的是一个元组,我们只取第一层
        input_featrue = inputs[0]
        out_feature = self.conv(input_featrue)
        # print("============simple_neck out_feature shape {} =================".format(out_feature.shape))
        out_feature = [out_feature]
        return tuple(out_feature)
