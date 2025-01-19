import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
# 一些上采样策略


# 自动计算出一个使特征图大小保持不变的padding值
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# 渐进式上采样,先采样到中间尺寸,然后通过卷积细化特征,然后再采样到目标尺寸
@MODELS.register_module(name="ProgressiveUpsampling")
class ProgressiveUpsampling(nn.Module):
    def __init__(self, channels = 384, kernel_size = 3, padding = None):
        super(ProgressiveUpsampling, self).__init__()
        # 计算Pading
        padding = autopad(k=kernel_size, p=padding)
        # 初始化卷积层，用于处理每次上采样后的特征图
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)


    def forward(self, x, tarH, tarW):
        # 第一次上采样到中间尺寸
        x = F.interpolate(x, size=(tarH//2, tarW//2), mode='bilinear', align_corners=True)
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 第二次上采样到目标尺寸
        x = F.interpolate(x, size=(tarH, tarW), mode='bilinear', align_corners=True)
        x = self.relu(self.bn2(self.conv2(x)))
        
        return x

# 假设 IP_feature2D 是你的初始小尺寸特征图
# up_model = ProgressiveUpsampling()
# IP_feature2D_resized = up_model(IP_feature2D)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        # coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
        #     B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

# 先dysample2倍,再直接提升到特征图大小
@MODELS.register_module(name="DyUpsample")
class DyUpsample(nn.Module):
    def __init__(self, channels = 384, scale = 2):
        super().__init__()
        self.dysample = DySample(in_channels=channels, scale=scale, style='lp', groups=4)  # 确保参数与输入匹配

    def forward(self, x, tarH, tarW):
        # 使用 DySample 进行 2 倍上采样
        x_upsampled = self.dysample(x)
        
        # 使用双线性插值进一步上采样到最终目标尺寸
        x_final = F.interpolate(x_upsampled, size=(tarH, tarW), mode='bilinear', align_corners=True)
        return x_final

 


# 渐进动态上采样,先用dysample上采样到2倍,然后再渐进上采样,拥有更多细节
@MODELS.register_module(name="GradualDyUpsampling")
class GradualDyUpsampling(nn.Module):
    def __init__(self, channels = 384, kernel_size = 3, padding = None, scale = 2):
        super().__init__()
        # 动态计算padding
        padding = autopad(k=kernel_size, p = padding)
        self.dysample = DySample(in_channels=channels, scale=scale, style='lp', groups=4)
        self.refine_conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.refine_conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x , intermediate_size:tuple, target_size:tuple):
        # 使用 DySample 进行 2 倍上采样
        x = self.dysample(x)
        # 中间步骤上采样
        x = F.interpolate(x, size=intermediate_size, mode='bilinear', align_corners=True)
        x = F.relu(self.refine_conv1(x))
        # 最终上采样到目标尺寸
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        x = F.relu(self.refine_conv2(x))
        return x




# if __name__ == '__main__':
#     x = torch.rand(2, 64, 4, 7)
#     dys = DySample(64)
#     print(dys(x).shape)

