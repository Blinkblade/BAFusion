import torch
import torch.nn as nn
from einops import rearrange
from mmdet3d.registry import MODELS
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 自定义一个MLP层
class Mlp(nn.Module): # 定义一个类，继承自 PyTorch 的 nn.Module 基类，这样可以方便地使用 PyTorch 的各种功能
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): # 定义初始化方法，接收一些参数
        """MLP
        Args:
            in_features (int): 输入特征维度 
            hidden_features (int, optional): 隐藏层. Defaults to None. 
            out_features (int, optional): 输出维度. Defaults to None. 
            act_layer (激活函数, optional): Defaults to nn.GELU. 
            drop (float, optional): . Defaults to 0.. 
        """
        super().__init__() # 调用父类的初始化方法，完成一些基本的设置
        out_features = out_features or in_features # 如果没有指定输出维度，就将其设为输入维度
        hidden_features = hidden_features or in_features # 如果没有指定隐藏维度，就将其设为输入维度
        self.fc1 = nn.Linear(in_features, hidden_features) # 定义第一个全连接层，接收输入特征，并输出隐藏特征
        self.act = act_layer() # 定义激活函数层，接收隐藏特征，并输出激活后的特征
        self.fc2 = nn.Linear(hidden_features, out_features) # 定义第二个全连接层，接收激活后的特征，并输出最终特征
        self.drop = nn.Dropout(drop) # 定义随机失活层，接收最终特征，并输出部分置零后的特征

    def forward(self, x): # 定义前向传播方法，接收一个输入张量 x，并返回一个输出张量
        x = self.fc1(x) # 将 x 传入第一个全连接层，并得到隐藏特征
        x = self.act(x) # 将隐藏特征传入激活函数层，并得到激活后的特征
        x = self.drop(x) # 将激活后的特征传入随机失活层，并得到部分置零后的特征
        x = self.fc2(x) # 将部分置零后的特征传入第二个全连接层，并得到最终特征
        x = self.drop(x) # 将最终特征再次传入随机失活层，并得到最后的输出
        return x # 返回输出张量

# # dim相当于每个token的维度

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         else:
#             kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

class Attention(nn.Module): # 定义一个类，继承自nn.Module，这是PyTorch中构建神经网络模块的基类
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1): # 定义类的初始化函数，接收以下参数：
        # dim: 输入序列的特征维度，也是输出序列的特征维度
        # num_heads: 注意力头的数量，用于将输入序列分割成多个子空间
        # qkv_bias: 是否为查询、键和值矩阵添加偏置项，默认为False
        # qk_scale: 缩放点积注意力的因子，如果为None，则使用特征维度的平方根作为缩放因子
        # attn_drop: 注意力权重的dropout概率，默认为0
        # proj_drop: 输出投影层的dropout概率，默认为0
        # sr_ratio: 线性投影的下采样比例，用于减少计算复杂度，默认为1，表示不下采样
        super().__init__() # 调用父类的初始化函数
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}." 
        # 断言特征维度可以被注意力头的数量整除，否则抛出异常

        self.dim = dim # 将特征维度赋值给类属性
        self.num_heads = num_heads # 将注意力头的数量赋值给类属性
        head_dim = dim // num_heads # 计算每个注意力头的子空间维度
        # 计算缩放因子，如果qk_scale不为None，则使用qk_scale，否则使用子空间维度的倒数平方根
        self.scale = qk_scale or head_dim ** -0.5 

        # 定义一个线性层，用于将输入序列映射为查询矩阵，输入和输出维度都是dim，是否添加偏置由qkv_bias决定
        self.q = nn.Linear(dim, dim, bias=qkv_bias) 
        # 定义一个线性层，用于将输入序列映射为键和值矩阵，输入维度是dim，输出维度是dim * 2，
        # 即键和值矩阵拼接在一起，是否添加偏置由qkv_bias决定
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias) 
        # 定义一个dropout层，用于对注意力权重进行随机失活，失活概率由attn_drop决定
        self.attn_drop = nn.Dropout(attn_drop) 
        # 定义一个线性层，用于对注意力输出进行投影，输入和输出维度都是dim
        self.proj = nn.Linear(dim, dim) 
        # 定义一个dropout层，用于对投影输出进行随机失活，失活概率由proj_drop决定
        self.proj_drop = nn.Dropout(proj_drop) 


        self.sr_ratio = sr_ratio # 将下采样比例赋值给类属性
        if sr_ratio > 1: # 如果下采样比例大于1，则需要进行线性投影
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) # 定义一个卷积层，用于对输入序列进行下采样，输入和输出通道数都是dim，卷积核大小和步长都是sr_ratio
            self.norm = nn.LayerNorm(dim) # 定义一个层归一化层，用于对下采样后的序列进行归一化，归一化维度是dim


    def forward(self, x, H, W): # 定义类的前向传播函数，接收以下参数：
        # x: 输入序列，维度是(B, N, C)，其中B是批量大小，N是序列长度，C是特征维度
        # H: 输入序列对应的高度，用于将序列重塑为二维图像
        # W: 输入序列对应的宽度，用于将序列重塑为二维图像
        # B是批量大小，N是序列长度，C是特征维度(原通道数)
        B, N, C = x.shape # 获取输入序列的形状
        # 将输入序列映射为查询矩阵，然后重塑为(B, N, num_heads, head_dim)，
        # 再交换第二和第三维，得到(B, num_heads, N, head_dim)        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 


        if self.sr_ratio > 1: # 如果下采样比例大于1，则需要进行线性投影
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) # 将输入序列交换第二和第三维，然后重塑为(B, C, H, W)，即二维图像的形式
            # 对二维图像进行下采样，然后重塑为(B, C, N')，其中N'是下采样后的序列长度，再交换第二和第三维，得到(B, N', C) 
            # sr就相当于下采样的卷积层,由于下采样后H W变化,所以N也变化   
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) 
            x_ = self.norm(x_) # 对下采样后的序列进行层归一化
            # 将下采样后的序列映射为键和值矩阵，然后重塑为(B, N', 2, num_heads, head_dim)，
            # 再交换第一和第三维，得到(2, B, num_heads, N, head_dim)          
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 

        else: # 如果下采样比例等于1，则不需要进行线性投影
            # 将输入序列映射为键和值矩阵，然后重塑为(B, N, 2, num_heads, head_dim)，
            # 再交换第一和第三维，得到(2, B, num_heads, N, head_dim)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
            
        k, v = kv[0], kv[1] # 将键和值矩阵分开，维度都是(B, num_heads, N', head_dim)
        # 计算查询矩阵和键矩阵的点积，然后乘以缩放因子，得到注意力分数矩阵，维度是(B, num_heads, N, N')
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1) # 对最后一维进行softmax函数，得到注意力权重矩阵，维度不变
        attn = self.attn_drop(attn) # 对注意力权重矩阵进行dropout操作，维度不变
        # 计算注意力权重矩阵和值矩阵的乘积，然后交换第二和第三维，再重塑为(B, N, C)，得到注意力输出序列
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x) # 对注意力输出序列进行线性投影，维度不变
        x = self.proj_drop(x) # 对投影输出序列进行dropout操作，维度不变
        return x # 返回最终的输出序列，维度是(B, N, C)





class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1)
        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
    
    
@MODELS.register_module(name="SoftmaxCrossAttention")
class SoftmaxCrossAttention(nn.Module): # 定义一个类，继承自nn.Module，这是PyTorch中构建神经网络模块的基类
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1): # 定义类的初始化函数，接收以下参数：
        # dim: 输入序列的特征维度，也是输出序列的特征维度
        # num_heads: 注意力头的数量，用于将输入序列分割成多个子空间
        # qkv_bias: 是否为查询、键和值矩阵添加偏置项，默认为False
        # qk_scale: 缩放点积注意力的因子，如果为None，则使用特征维度的平方根作为缩放因子
        # attn_drop: 注意力权重的dropout概率，默认为0
        # proj_drop: 输出投影层的dropout概率，默认为0
        # sr_ratio: 线性投影的下采样比例，用于减少计算复杂度，默认为1，表示不下采样
        super().__init__() # 调用父类的初始化函数
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}." 
        # 断言特征维度可以被注意力头的数量整除，否则抛出异常

        self.dim = dim # 将特征维度赋值给类属性
        self.num_heads = num_heads # 将注意力头的数量赋值给类属性
        head_dim = dim // num_heads # 计算每个注意力头的子空间维度
        # 计算缩放因子，如果qk_scale不为None，则使用qk_scale，否则使用子空间维度的倒数平方根
        self.scale = qk_scale or head_dim ** -0.5 

        # 定义一个线性层，用于将输入序列映射为查询矩阵，输入和输出维度都是dim，是否添加偏置由qkv_bias决定
        self.q = nn.Linear(dim, dim, bias=qkv_bias) 
        self.k = nn.Linear(dim, dim, bias=qkv_bias) 
        self.v = nn.Linear(dim, dim, bias=qkv_bias) 
        # 定义一个dropout层，用于对注意力权重进行随机失活，失活概率由attn_drop决定
        self.attn_drop = nn.Dropout(attn_drop) 
        # 定义一个线性层，用于对注意力输出进行投影，输入和输出维度都是dim
        self.proj = nn.Linear(dim, dim) 
        # 定义一个dropout层，用于对投影输出进行随机失活，失活概率由proj_drop决定
        self.proj_drop = nn.Dropout(proj_drop) 

        self.sr_ratio = sr_ratio # 将下采样比例赋值给类属性
        if sr_ratio > 1: # 如果下采样比例大于1，则需要进行线性投影
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) # 定义一个卷积层，用于对输入序列进行下采样，输入和输出通道数都是dim，卷积核大小和步长都是sr_ratio
            self.norm = nn.LayerNorm(dim) # 定义一个层归一化层，用于对下采样后的序列进行归一化，归一化维度是dim

    def forward(self, query, key, value, H, W): # 定义类的前向传播函数，接收以下参数：
        # x: 输入序列，维度是(B, N, C)，其中B是批量大小，N是序列长度，C是特征维度
        # H: 输入序列对应的高度，用于将序列重塑为二维图像
        # W: 输入序列对应的宽度，用于将序列重塑为二维图像
        # B是批量大小，N是序列长度，C是特征维度(原通道数)
        B, N, C = query.shape # 获取输入序列的形状
        # kv shape和query区别开,实际上只有N不同
        Bkv, Nkv, Ckv = key.shape
        # 将输入序列映射为查询矩阵，然后重塑为(B, N, num_heads, head_dim)，
        # 再交换第二和第三维，得到(B, num_heads, N, head_dim)        
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        # 接下来生成key 和 value
        if self.sr_ratio > 1: # 如果下采样比例大于1，则需要进行线性投影
            # 先将k v reshape回原始特征图
            # 共享同一个卷积核，这样卷积操作是相同的
            k_ = key.permute(0, 2, 1).reshape(B, C, H, W) # 将输入序列交换第二和第三维，然后重塑为(B, C, H, W)，即二维图像的形式
            v_ = value.permute(0, 2, 1).reshape(B, C, H, W) # 将输入序列交换第二和第三维，然后重塑为(B, C, H, W)，即二维图像的形式
            # 对二维图像进行下采样，然后重塑为(B, C, N')，其中N'是下采样后的序列长度，再交换第二和第三维，得到(B, N', C) 
            # sr就相当于下采样的卷积层,由于下采样后H W变化,所以N也变化   
            k_ = self.sr(k_).reshape(B, C, -1).permute(0, 2, 1) 
            k_ = self.norm(k_) # 对下采样后的序列进行层归一化
            v_ = self.sr(v_).reshape(B, C, -1).permute(0, 2, 1) 
            v_ = self.norm(v_) # 对下采样后的序列进行层归一化
            # 将下采样后的序列映射为键和值矩阵，然后重塑为(B, N', num_heads, head_dim)，
            # 再交换第一和第三维，得到(B, num_heads, N, head_dim)            
            k = self.k(k_).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
            v = self.v(v_).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        else: # 如果下采样比例等于1，则不需要进行线性投影
            # 将输入序列映射为键和值矩阵，然后重塑为(B, N, 2, num_heads, head_dim)，
            # 再交换第一和第三维，得到(2, B, num_heads, N, head_dim)
            k = self.k(key).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
            v = self.v(value).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
            
        # k, v = kv[0], kv[1] # 将键和值矩阵分开，维度都是(B, num_heads, N', head_dim)
        # 计算查询矩阵和键矩阵的点积，然后乘以缩放因子，得到注意力分数矩阵，维度是(B, num_heads, N, N')
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1) # 对最后一维进行softmax函数，得到注意力权重矩阵，维度不变
        attn = self.attn_drop(attn) # 对注意力权重矩阵进行dropout操作，维度不变
        # 计算注意力权重矩阵和值矩阵的乘积，然后交换第二和第三维，再重塑为(B, N, C)，得到注意力输出序列
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x) # 对注意力输出序列进行线性投影，维度不变
        x = self.proj_drop(x) # 对投影输出序列进行dropout操作，维度不变
        return x # 返回最终的输出序列，维度是(B, N, C)


@MODELS.register_module(name="CrossFocusedLinearAttention")
class CrossFocusedLinearAttention(nn.Module): # 定义一个类，继承自nn.Module，这是PyTorch中构建神经网络模块的基类
    def __init__(self, dim, num_patches=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5,add = False, layernorm = False): # 定义类的初始化函数，接收以下参数：
        # dim: 输入序列的特征维度，也是输出序列的特征维度
        # num_patches: 输入序列的分块数量，用于添加位置编码
        # num_heads: 注意力头的数量，用于将输入序列分割成多个子空间
        # qkv_bias: 是否为查询、键和值矩阵添加偏置项，默认为False
        # qk_scale: 缩放点积注意力的因子，如果为None，则使用特征维度的平方根作为缩放因子
        # attn_drop: 注意力权重的dropout概率，默认为0
        # proj_drop: 输出投影层的dropout概率，默认为0
        # sr_ratio: 线性投影的下采样比例，用于减少计算复杂度，默认为1，表示不下采样，实际实施中这里恒定设置为1
        # focusing_factor: 焦点化因子，用于调整注意力分布的尖锐程度，默认为3
        # kernel_size: 深度可分离卷积的核大小，用于增强局部信息，默认为5
        # add & layernorm 是否对相加后的特征使用add&layernorm
        super().__init__() # 调用父类的初始化函数
        # 断言特征维度可以被注意力头的数量整除，否则抛出异常        
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}." 
        self.dim = dim # 将特征维度赋值给类属性
        self.num_heads = num_heads # 将注意力头的数量赋值给类属性
        head_dim = dim // num_heads # 计算每个注意力头的子空间维度
        # 定义一个线性层，用于将输入序列映射为查询矩阵，输入和输出维度都是dim，是否添加偏置由qkv_bias决定
        self.q = nn.Linear(dim, dim, bias=qkv_bias) 
        # 定义一个线性层，用于映射key
        self.k = nn.Linear(dim, dim, bias=qkv_bias) 
        # 定义一个线性层，用于映射value
        self.v = nn.Linear(dim, dim, bias=qkv_bias) 
        # 定义一个dropout层，用于对注意力权重进行随机失活，失活概率由attn_drop决定        
        self.attn_drop = nn.Dropout(attn_drop) 
        # 定义一个线性层，用于对注意力输出进行投影，输入和输出维度都是dim
        self.proj = nn.Linear(dim, dim)
        # 定义一个dropout层，用于对投影输出进行随机失活，失活概率由proj_drop决定
        self.proj_drop = nn.Dropout(proj_drop) 

        # add & layernorm
        self.add = add
        self.use_LN = layernorm
        if layernorm:
            self.layernorm = nn.LayerNorm(dim)

        self.sr_ratio = sr_ratio # 将下采样比例赋值给类属性
        if sr_ratio > 1: # 如果下采样比例大于1，则需要进行线性投影
            # 定义一个卷积层，用于对输入序列进行下采样，输入和输出通道数都是dim，卷积核大小和步长都是sr_ratio
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) 
            # 定义一个层归一化层，用于对下采样后的序列进行归一化，归一化维度是dim
            self.norm = nn.LayerNorm(dim) 
        # 将焦点化因子赋值给类属性
        self.focusing_factor = focusing_factor 
        # 定义一个深度可分离卷积层，用于增强局部信息，输入和输出通道数都是head_dim，卷积核大小由kernel_size决定，
        # 分组数也是head_dim，表示每个通道单独卷积，填充大小为kernel_size // 2，保持输出尺寸不变       
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2) 
        # 定义一个可学习的参数，用于缩放注意力分数，初始值为零向量，维度是(1, 1, dim)        
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim))) 
        # 定义一个可学习的参数，用于添加位置编码，初始值为零矩阵，维度是(1, num_patches // (sr_ratio * sr_ratio), dim)，
        # 其中num_patches // (sr_ratio * sr_ratio)表示下采样后的序列长度
        # 如果不输入num_patches,就不生成位置编码
        if num_patches is not None:
            self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim))) 
            pass
        else:
            self.positional_encoding = None
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size)) # 打印一些参数信息

    def forward(self, query, key, value, H, W): # 定义类的前向传播函数，接收以下参数：
        # x: 输入序列，维度是(B, N, C)，其中B是批量大小，N是序列长度，C是特征维度
        # H: 输入序列对应的高度，用于将序列重塑为二维图像
        # W: 输入序列对应的宽度，用于将序列重塑为二维图像
        B, N, C = query.shape # 获取输入序列的形状
        q = self.q(query) # 将输入序列映射为查询矩阵，维度不变
        # kv shape和query区别开,实际上只有N不同
        Bkv, Nkv, Ckv = key.shape
        # 生成KV,IF基本用不到
        if self.sr_ratio > 1: # 如果下采样比例大于1，则需要进行线性投影
            k_ = key.permute(0, 2, 1).reshape(B, C, H, W) # 将输入序列交换第二和第三维，然后重塑为(B, C, H, W)，即二维图像的形式
            k_ = self.sr(k_).reshape(B, C, -1).permute(0, 2, 1) # 对二维图像进行下采样，然后重塑为(B, C, N')，其中N'是下采样后的序列长度，再交换第二和第三维，得到(B, N', C)
            k_ = self.norm(k_) # 对下采样后的序列进行层归一化
            k = self.k(k_).reshape(B, -1, 2, C).permute(2, 0, 1, 3) # 将下采样后的序列映射为键和值矩阵，然后重塑为(B, N', 2,
            
            v_ = value.permute(0, 2, 1).reshape(B, C, H, W) # 将输入序列交换第二和第三维，然后重塑为(B, C, H, W)，即二维图像的形式
            v_ = self.sr(v_).reshape(B, C, -1).permute(0, 2, 1) # 对二维图像进行下采样，然后重塑为(B, C, N')，其中N'是下采样后的序列长度，再交换第二和第三维，得到(B, N', C)
            v_ = self.norm(v_) # 对下采样后的序列进行层归一化
            v = self.v(v_).reshape(B, -1, 2, C).permute(2, 0, 1, 3) # 将下采样后的序列映射为键和值矩阵，然后重塑为(B, N', 2,
            
        else: # 如果下采样比例等于1，则不需要进行线性投影
            # shape为 b n dim
            k = self.k(key)
            v = self.v(value)
            
        if self.positional_encoding is not None:
            k = k + self.positional_encoding # 将位置编码添加到键矩阵中，维度不变
        focusing_factor = self.focusing_factor # 获取焦点化因子
        kernel_function = nn.ReLU() # 定义一个激活函数，用于对查询和键矩阵进行非线性变换
        scale = nn.Softplus()(self.scale) # 定义一个缩放函数，用于对缩放参数进行正值化
        # 只对q k 进行操作,因为只有q k 参与注意力权重的计算
        q = kernel_function(q) + 1e-6 # 对查询矩阵进行激活函数变换，并加上一个小的常数，避免出现零值，维度不变
        k = kernel_function(k) + 1e-6 # 对键矩阵进行激活函数变换，并加上一个小的常数，避免出现零值，维度不变
        q = q / scale # 对查询矩阵进行缩放操作，维度不变
        k = k / scale # 对键矩阵进行缩放操作，维度不变
        q_norm = q.norm(dim=-1, keepdim=True) # 计算查询矩阵的范数，维度是(B, N, 1)
        k_norm = k.norm(dim=-1, keepdim=True) # 计算键矩阵的范数，维度是(B, N', 1)
        q = q ** focusing_factor # 对查询矩阵进行幂运算，增加注意力分布的尖锐程度，维度不变
        k = k ** focusing_factor # 对键矩阵进行幂运算，增加注意力分布的尖锐程度，维度不变
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm # 对查询矩阵进行归一化，并乘以原始范数，保持范数不变，维度不变
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm # 对键矩阵进行归一化，并乘以原始范数，保持范数不变，维度不变
        # 将查询、键和值矩阵按注意力头的数量分割成多个子空间，并重塑为(bh, n', c)，其中bh表示批量大小乘以注意力头的数量，
        # n'表示序列长度或下采样后的序列长度，c表示子空间维度        
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v]) 

        # i是query的序列长度(n),j是key的序列长度(value的序列长度与之相同)
        # c和d是key和value每个token的维度,即Dim,理论上应该相同,即channel
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1] # 获取各个维度的大小
        # 计算归一化因子，即query矩阵和key矩阵按最后一维求和的点积的倒数，
        # 加上一个小的常数，避免出现零值或除零错误，维度是(bh, n)
        # 这个计算过程是这样的先对key在dim = 1(n,即序列长度)上进行求和运算(即每个token加和),从b n c得到b c
        # 然后query(b n c)矩阵和求和后的key(b i)进行矩阵乘法,相当于batch单独隔离,然后i×c * c×1 ==> i
        # 计算完得到b×i之后取倒数,作为归一化因子
        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6) 

        # 线性注意力就是先进行key和value的相乘即c和d做乘法 c和d是key和value每个token的维度
        if i * j * (c + d) > c * d * (i + j): # 如果按照传统的方法计算注意力分数和输出会导致更大的计算量，则使用一种更高效的方法
            # 计算key矩阵和value矩阵按第二维求和的乘积，维度是(bh, c, d)
            kv = torch.einsum("b j c, b j d -> b c d", k, v) 
            # 计算查询矩阵和键值矩阵乘积的乘积，再乘以归一化因子，得到注意力输出，维度是(bh, n, d)    
            # 这个if else语句中的x就是最终的注意力分数乘以value
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z) 
        # 这个else里就是传统注意力，只不过不再使用softmax而是采用线性计算的归一化因子Z
        else: # 如果按照传统的方法计算注意力分数和输出不会导致更大的计算量，则使用传统的方法
            qk = torch.einsum("b i c, b j c -> b i j", q, k) # 计算查询矩阵和键矩阵的点积，得到注意力分数，维度是(bh, n, n')
            # 计算注意力分数和值矩阵的乘积，再乘以归一化因子，得到注意力输出，维度是(bh, n, d)            
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z) 

        if self.sr_ratio > 1: # 如果下采样比例大于1，则需要对值矩阵进行上采样，恢复原始序列长度
            v = nn.functional.interpolate(v.permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1) # 对值矩阵按第二维进行线性插值，然后交换第二和第三维，得到(bh, n, d)
        # 这里原始设置的是W = H,因此不适用,改成按比例算
        # num = int(v.shape[1] ** 0.5) # 计算值矩阵对应的高度或宽度，即序列长度的平方根
        numH = int((v.shape[1] / (H*W) )*H)
        numW = int((v.shape[1] / (H*W) )*W)     
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=numW, h=numH) # 将值矩阵重塑为二维图像的形式，维度是(bh, c, num, num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c") # 对二维图像进行深度可分离卷积，增强局部信息，然后重塑为(bh, n, c)
        
        # 这里可以考虑使用x + query来实现多模态的信息融合
        if self.add:
            # 既可以考虑使用映射后的q也可以考虑使用初始的query
            # x = x + query
            x = x + q
     
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads) # 将输出按注意力头的数量合并成一个空间，并重塑为(B, N, C)，其中B是批量大小，N是序列长度，C是特征维度
       
        # 实现add & layernorm 
        if self.use_LN:
            x = self.layernorm(x)       

        x = self.proj(x) # 对输出进行线性投影，维度不变
        x = self.proj_drop(x) # 对投影输出进行dropout操作，维度不变
        return x # 返回最终的输出序列，维度是(B, N, C)



# @MODELS.register_module(name="CrossFocusedLinearAttentionPrune")
# class CrossFocusedLinearAttentionPrune(nn.Module):
#     def __init__(self, dim, num_patches=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
#                  sr_ratio=1, focusing_factor=3, kernel_size=5, add=False, layernorm=False):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.add = add
#         self.use_LN = layernorm
#         if layernorm:
#             self.layernorm = nn.LayerNorm(dim)

#         self.sr_ratio = sr_ratio
#         self.focusing_factor = focusing_factor
#         # self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
#         #                      groups=head_dim, padding=kernel_size // 2)
#         self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
#         if num_patches is not None:
#             self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
#         else:
#             self.positional_encoding = None
#         print('Linear Attention sr_ratio{} f{} kernel{}'.format(sr_ratio, focusing_factor, kernel_size))

#     def forward(self, query, key, value, H, W):
#         B, N, C = query.shape
#         q = self.q(query)
#         Bkv, Nkv, Ckv = key.shape
#         k = self.k(key)
#         v = self.v(value)

#         if self.positional_encoding is not None:
#             k = k + self.positional_encoding

#         focusing_factor = self.focusing_factor
#         kernel_function = nn.ReLU()
#         scale = nn.Softplus()(self.scale)

#         q = kernel_function(q) + 1e-6
#         k = kernel_function(k) + 1e-6
#         q = q / scale
#         k = k / scale

#         q_norm = q.norm(dim=-1, keepdim=True)
#         k_norm = k.norm(dim=-1, keepdim=True)
#         q = (q ** focusing_factor) / q.norm(dim=-1, keepdim=True) * q_norm
#         k = (k ** focusing_factor) / k.norm(dim=-1, keepdim=True) * k_norm

#         # 拆分多头
#         head_dim = C // self.num_heads
#         q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2).reshape(B * self.num_heads, N, head_dim)
#         k = k.view(B, Nkv, self.num_heads, head_dim).transpose(1, 2).reshape(B * self.num_heads, Nkv, head_dim)
#         v = v.view(B, Nkv, self.num_heads, head_dim).transpose(1, 2).reshape(B * self.num_heads, Nkv, head_dim)

#         # 更新维度信息
#         b = B * self.num_heads
#         # i, j, c, d = q.shape[1], k.shape[1], k.shape[2], v.shape[2]

#         # 计算 z
#         k_sum = k.sum(dim=1)  # 形状 (b, c)
#         z_num = (q * k_sum.unsqueeze(1)).sum(dim=2)  # 形状 (b, i)
#         z = 1 / (z_num + 1e-6)  # 形状 (b, i)

#         # 计算 kv：形状 (b, c, d)
#         kv = torch.bmm(k.transpose(1, 2), v)

#         # 计算 x：形状 (b, i, d)
#         x = torch.bmm(q, kv)

#         # 乘以 z
#         x = x * z.unsqueeze(2)

#         # if i * j * (c + d) > c * d * (i + j):
#         #     # 计算 kv：形状 (b, c, d)
#         #     kv = torch.bmm(k.transpose(1, 2), v)

#         #     # 计算 x：形状 (b, i, d)
#         #     x = torch.bmm(q, kv)

#         #     # 乘以 z
#         #     x = x * z.unsqueeze(2)
#         # else:
#         #     # 计算 qk：形状 (b, i, j)
#         #     qk = torch.bmm(q, k.transpose(1, 2))

#         #     # 计算 x：形状 (b, i, d)
#         #     x = torch.bmm(qk, v)

#         #     # 乘以 z
#         #     x = x * z.unsqueeze(2)

#         # 将 x 重塑为二维特征图
#         # numH, numW = H, W
#         # x_feature_map = x.transpose(1, 2).reshape(b, head_dim, numH, numW)
#         # x_feature_map = self.dwc(x_feature_map)
#         # x = x_feature_map.reshape(b, head_dim, N).transpose(1, 2)

#         if self.add:
#             x = x + q

#         # 重新组合多头
#         x = x.reshape(B, self.num_heads, N, head_dim).transpose(1, 2).reshape(B, N, C)

#         if self.use_LN:
#             x = self.layernorm(x)

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x



@MODELS.register_module(name="CrossFocusedLinearAttentionPrune")
class CrossFocusedLinearAttentionPrune(nn.Module):
    def __init__(self, dim, num_patches=None, num_heads=1, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1, focusing_factor=3,
                 kernel_size=5, add=False, layernorm=False):
        super().__init__()
        self.dim = dim
        self.num_heads = 1  # 固定为1
        head_dim = dim  # 因为 num_heads=1，所以 head_dim=dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.add = add
        self.use_LN = layernorm
        # if layernorm:
        #     self.layernorm = nn.LayerNorm(dim)



        self.sr_ratio = sr_ratio
        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim,
                             kernel_size=kernel_size, groups=head_dim,
                             padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))

        self.kernel_function = nn.ReLU()
        self.Softplus = nn.Softplus()


        if num_patches is not None:
            self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        else:
            self.positional_encoding = None
        print('Linear Attention sr_ratio{} f{} kernel{}'.format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, query, key, value, H, W):
        B, N, C = query.shape
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)

        # if self.positional_encoding is not None:
        #     k = k + self.positional_encoding

        focusing_factor = self.focusing_factor
        kernel_function = self.kernel_function
        scale = self.Softplus(self.scale)


        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale

        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = (q ** focusing_factor) / q.norm(dim=-1, keepdim=True) * q_norm
        k = (k ** focusing_factor) / k.norm(dim=-1, keepdim=True) * k_norm

        # 计算 z
        k_sum = k.sum(dim=1)  # 形状 (B, C)
        z_num = (q * k_sum.unsqueeze(1)).sum(dim=2)  # 形状 (B, N)
        z = 1 / (z_num + 1e-6)  # 形状 (B, N)

        # 计算注意力输出 x
        # 计算 kv：形状 (b, c, d)
        kv = torch.bmm(k.transpose(1, 2), v)

        # 计算 x：形状 (b, i, d)
        x = torch.bmm(q, kv)

        # 乘以 z
        x = x * z.unsqueeze(2)
        # print("============ THERE IS NO Z =============")

        # 将 x 转换为二维特征图，进行深度可分离卷积
        x_feature_map = x.transpose(1, 2).view(B, C, H, W)
        x_feature_map = self.dwc(x_feature_map)
        x = x_feature_map.view(B, C, N).transpose(1, 2)  # 形状 (B, N, C)
        # 选择分支和用不到的组件全部移除
        # if self.add:
        x = x + q

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


        