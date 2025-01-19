import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 自动计算出一个使特征图大小保持不变的padding值
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# 定义个接口简单的conv2d,mm写的太恶心了
@MODELS.register_module(name="MyConv2D")
class MyConv2D(nn.Module):
    # 设置默认激活函数为SiLU
    default_act = nn.SiLU()  # default activation
    
    # 定义一个卷积模块，输入通道数是Cin，输出通道数是Cout
    def __init__(self, Cin, Cout, kernel_size=1, stride=1,padding=None,g=1, d=1, act=True,norm=True):
        super(MyConv2D, self).__init__()
        # 定义一个卷积层，使用1×1卷积，无偏置
        # self.conv = nn.Conv2d(Cin, Cout, kernel_size=1, stride=1, padding=0, bias=False)
        padding = autopad(kernel_size, padding, d)
        self.conv = nn.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=stride, 
                              padding=padding, groups=g, dilation=d, bias=False)
        # self.conv = nn.Conv2d(Cout*2, Cout, kernel_size=kernel_size, stride=stride, 
        #                       padding=padding, groups=g, dilation=d, bias=False)
        # # 使用kaiming初始化
        # nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        # 使用xavier均匀分布初始化
        # 不用激活函数就可以使用默认值不设置其它参数
        nn.init.xavier_uniform_(self.conv.weight)
        # 定义一个Norm层，使用批归一化
        if norm:
            self.norm = nn.BatchNorm2d(Cout)
        else:
            # 否则不进行norm
            self.norm = nn.Identity()
        # 定义一个激活函数，使用SiLU
        # 如果输入的act是一个nn.Module,那么就是这个nn.Module
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        print("=================Build MyConv2D with kernel_size = {},padding={} ==================".format(kernel_size,padding))

    def forward(self, x):
        # x是输入的特征图，维度是(B, Cin, H, W)
        # 将x通过卷积层
        x = self.conv(x)
        # 将x通过Norm层
        x = self.norm(x)
        # 将x通过激活函数
        x = self.act(x)
        # 返回输出的特征图，维度是(B, Cout, H, W)
        return x
    
    



# 定义个接口简单的conv2d,mm写的太恶心了
@MODELS.register_module(name="MyConv2DRelu")
class MyConv2DRelu(nn.Module):
    # 设置默认激活函数为SiLU
    default_act = nn.ReLU()  # default activation
    
    # 定义一个卷积模块，输入通道数是Cin，输出通道数是Cout
    def __init__(self, Cin, Cout, kernel_size=1, stride=1,padding=None,g=1, d=1, act=True,norm=True):
        super(MyConv2DRelu, self).__init__()
        # 定义一个卷积层，使用1×1卷积，无偏置
        # self.conv = nn.Conv2d(Cin, Cout, kernel_size=1, stride=1, padding=0, bias=False)
        padding = autopad(kernel_size, padding, d)
        self.conv = nn.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=stride, 
                              padding=padding, groups=g, dilation=d, bias=False)
        # # 使用kaiming初始化
        # 参数抄的mm的3d backbone,a = 0为默认值不填
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        # 使用xavier均匀分布初始化
        # 不用激活函数就可以使用默认值不设置其它参数
        # nn.init.xavier_uniform_(self.conv.weight)
        # 定义一个Norm层，使用批归一化
        if norm:
            self.norm = nn.BatchNorm2d(Cout)
        else:
            # 否则不进行norm
            self.norm = nn.Identity()
        # 定义一个激活函数，使用SiLU
        # 如果输入的act是一个nn.Module,那么就是这个nn.Module
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        print("=================Build MyConv2D with kernel_size = {},padding={} ==================".format(kernel_size,padding))

    def forward(self, x):
        # x是输入的特征图，维度是(B, Cin, H, W)
        # 将x通过卷积层
        x = self.conv(x)
        # 将x通过Norm层
        x = self.norm(x)
        # 将x通过激活函数
        x = self.act(x)
        # 返回输出的特征图，维度是(B, Cout, H, W)
        return x
    
    
# 定义个接口简单的neck,目前单尺度用不到FPN
@MODELS.register_module(name="MyNeck")
class MyNeck(nn.Module):
    # 设置默认激活函数为SiLU
    default_act = nn.SiLU()  # default activation
    # 这个模块用来将backbone输出的通道降维到指定通道
    def __init__(self, Cin=2048, Cout=256, kernel_size=3, stride=1,padding=None,g=1, d=1, act=True):
        super(MyNeck, self).__init__()
        # 自动填充
        padding = autopad(kernel_size, padding, d)
        self.conv = nn.Conv2d(Cin, Cout, kernel_size=kernel_size, stride=stride, 
                              padding=padding, groups=g, dilation=d, bias=False)
        # 使用kaiming初始化
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        # 定义一个Norm层，使用批归一化
        self.norm = nn.BatchNorm2d(Cout)
        # 定义一个激活函数，使用ReLU
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        print("=================Build MyNeck with kernel_size = {},padding={} ==================".format(kernel_size,padding))

    def forward(self, x_tuple):
        # x是输入的特征图tuple，维度是(n ,B, Cin, H, W)
        # print(type(x_tuple))
        x_list = []
        for x in x_tuple:
            # 将x通过卷积层
            y = self.conv(x)
            # 将x通过Norm层
            y = self.norm(y)
            # 将x通过激活函数
            y = self.act(y)
            # 装入tuple中,这是为了保持FPN的输出格式
            x_list.append(y)
        return tuple(x_list)


# 构建多个卷积堆叠的模块
@MODELS.register_module(name="ConvStack")  
class ConvStack(nn.Module):
    # 定义一个卷积堆叠模块，输入通道数是Cin，输出通道数是Cout
    def __init__(self, Cin=768, Cout=384, n=1, kernel_sizes=(3, 1,), stride=1,
                 padding=None, g=1, d=1, act=True, norm=True, res=False):
        super(ConvStack, self).__init__()
        # 定义一个列表，用来装多个MyConv2DRelu模块
        self.model = nn.Sequential()
        self.res = res  # 添加res参数
        for i, kernel_size in enumerate(kernel_sizes):
            # 循环n次，将MyConv2DRelu模块加入列表
            self.model.add_module(
                name="stack_conv{}".format(str(i)),
                module=MyConv2D(Cin, Cout, kernel_size=kernel_size,
                                stride=stride, padding=padding,
                                g=g, d=d, act=act, norm=norm)
            )
            # 更新输入通道数
            Cin = Cout
            pass
        print("================= Build ConvStack with kernel_sizes = {} ================".format(kernel_sizes))
        pass

    # 前向推理
    def forward(self, x):
        identity = x
        # 将x传入列表，依次进行卷积
        for layer in self.model:
            x = layer(x)
        if self.res:
            x += identity
        return x

